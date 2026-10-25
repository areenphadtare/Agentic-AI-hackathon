"""ArgusGraph adaptive video -> YOLO/CLIP/BLIP -> Gemini -> FalkorDB.
Embeddings are saved to NPZ and never written into FalkorDB.
"""
import argparse, json, os, re, time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO
from transformers import (CLIPModel, CLIPProcessor, BlipProcessor,
    BlipForConditionalGeneration)
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = ""
CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_CAPTION_NAME = "Salesforce/blip-image-captioning-base"

class Scene(BaseModel):
    scene_description: str
    scene_type: str = "unknown"
    activities: list[str] = Field(default_factory=list)
    safety_concerns: list[str] = Field(default_factory=list)
    visible_evidence: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


def video_id_for(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_") or "video"


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def unit_vector(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    return x / n if n else x


def adaptive_sample(video_path, frames_dir, base_interval=1.0, scan_interval=0.2,
                    min_gap=0.5, threshold=0.13, max_per_second=3):
    """Keep baseline frames plus frames where low-res image appearance changes."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release(); raise RuntimeError("Video FPS/frame count unavailable")
    duration = total / fps
    scan_step, base_step = max(1, round(fps*scan_interval)), max(1, round(fps*base_interval))
    chosen, prev, last_time, per_sec = [], None, -1e9, {}
    idx = 0
    next_baseline_time = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % scan_step == 0:
            ts = idx / fps
            thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY), (5,5), 0)
            diff = float(np.mean(cv2.absdiff(gray, prev))/255.0) if prev is not None else 0.0
            prev = gray
            baseline = ts + (scan_interval / 2) >= next_baseline_time
            if baseline:
                while next_baseline_time <= ts + (scan_interval / 2):
                    next_baseline_time += base_interval
            bucket = int(ts)
            adaptive = diff >= threshold and ts-last_time >= min_gap and per_sec.get(bucket, 0) < max_per_second
            if baseline or adaptive:
                chosen.append({"frame_index": idx, "timestamp_seconds": round(ts,3),
                    "change_score": round(diff,5), "sampling_reason": "baseline" if baseline else "visual_change"})
                last_time = ts; per_sec[bucket] = per_sec.get(bucket,0)+1
        idx += 1
    cap.release()
    cap = cv2.VideoCapture(str(video_path)); frames=[]
    for n, item in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"]); ok, frame = cap.read()
        if not ok: continue
        fid = f"{video_id_for(video_path)}_frame_{n:06d}"
        image_path = frames_dir / f"{fid}.jpg"
        if not cv2.imwrite(str(image_path), frame): continue
        frames.append({**item,"frame_id":fid,"image_path":str(image_path),"width":int(frame.shape[1]),"height":int(frame.shape[0])})
    cap.release()
    return {"video_id":video_id_for(video_path),"video_name":Path(video_path).name,
        "video_path":str(video_path),"duration_seconds":round(duration,3),"fps":fps,
        "original_frame_count":total,"frames":frames}


def load_models(device):
    print("Loading YOLO, CLIP and BLIP models; first run downloads model weights.")
    return {
      "yolo":YOLO("yolov8n.pt"), "device":device,
      "clip_processor":CLIPProcessor.from_pretrained(CLIP_NAME),
      "clip_model":CLIPModel.from_pretrained(CLIP_NAME).to(device).eval(),
      "cap_processor":BlipProcessor.from_pretrained(BLIP_CAPTION_NAME),
      "cap_model":BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_NAME).to(device).eval(),
    }

@torch.inference_mode()
def enrich_frame(record, models):
    image = Image.open(record["image_path"]).convert("RGB"); device=models["device"]
    pred=models["yolo"].predict(source=np.array(image), conf=0.25, verbose=False)[0]
    det=[]
    if pred.boxes is not None:
        for b in pred.boxes:
            cls=int(b.cls.item())
            det.append({"label":models["yolo"].names.get(cls,str(cls)),"confidence":round(float(b.conf.item()),4),
                "bbox_xyxy":[round(float(v),2) for v in b.xyxy[0].tolist()]})
    ci=models["clip_processor"](images=image,return_tensors="pt").to(device)
    cv=models["clip_model"].get_image_features(**ci)[0].detach().cpu().numpy()
    # BLIP captioning checkpoint includes a compatible vision encoder.
    # Reuse that encoder for a pooled visual feature vector instead of loading
    # the incompatible BLIP ITM checkpoint as a generic BlipModel.
    ti=models["cap_processor"](images=image,return_tensors="pt").to(device)
    ids=models["cap_model"].generate(**ti,max_new_tokens=60)
    caption=models["cap_processor"].decode(ids[0],skip_special_tokens=True)
    vision_out=models["cap_model"].vision_model(
        pixel_values=ti["pixel_values"], return_dict=True
    )
    bv=vision_out.pooler_output[0].detach().cpu().numpy()
    return {**record,"object_count":len(det),"detections":det,"blip_caption":caption,
        "gemini_eligible":len(det)>1,"clip_embedding":unit_vector(cv),"blip_embedding":unit_vector(bv)}


def gemini_scene(client, record, retries=2):
    prompt=f'''Analyze this property-security video frame. Return structured scene interpretation.
Frame ID: {record['frame_id']}; timestamp seconds: {record['timestamp_seconds']}.
YOLO detections (may be wrong): {json.dumps(record['detections'])}
BLIP caption (may be wrong): {record['blip_caption']}
Only describe visually supported facts. Do not infer criminal intent from appearance.
Do not claim an event across time from one frame. Separate evidence from uncertainty.'''
    last_error=None
    for attempt in range(retries+1):
        try:
            response=client.models.generate_content(model=GEMINI_MODEL,
                contents=[prompt,types.Part.from_bytes(data=Path(record['image_path']).read_bytes(),mime_type="image/jpeg")],
                config=types.GenerateContentConfig(response_mime_type="application/json",response_schema=Scene,temperature=0.1))
            if not response.text: raise RuntimeError("Empty Gemini response")
            return {"status":"success","scene":Scene.model_validate_json(response.text).model_dump(),"error":None}
        except Exception as e:
            last_error=str(e)
            if attempt<retries: time.sleep(1.5*(attempt+1))
    return {"status":"fallback_blip_yolo","scene":{
        "scene_description":record["blip_caption"],"scene_type":"unknown","activities":[],"safety_concerns":[],
        "visible_evidence":[f"YOLO detected {d['label']} (confidence {d['confidence']})" for d in record["detections"]],
        "uncertainty":["Gemini failed; fallback uses BLIP caption and YOLO detections."]},"error":last_error}


def ingest_falkordb(graph_name, meta, records):
    """Store the video as Video -> Frame -> FrameAttribute nodes.

    Every field present on a frame record becomes its own FrameAttribute node.
    The attribute node's `name` and `key` are the JSON field name; `value`
    stores the original scalar or a JSON-serialized representation of a list/dict.
    """
    from falkordb import FalkorDB

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD") or None
    graph = FalkorDB(host=host, port=port, password=password).select_graph(graph_name)

    # One video node per video.
    graph.query(
        """
        MERGE (v:Video {video_id: $video_id})
        SET v.name = $video_name,
            v.source_path = $video_path,
            v.duration_seconds = $duration,
            v.fps = $fps
        """,
        {
            "video_id": meta["video_id"],
            "video_name": meta["video_name"],
            "video_path": meta["video_path"],
            "duration": meta["duration_seconds"],
            "fps": meta["fps"],
        },
    )

    for frame_number, record in enumerate(records, start=1):
        frame_id = record.get("frame_id", f"{meta['video_id']}_frame_{frame_number:06d}")
        frame_name = f"Frame no. {frame_number:03d}"

        # MERGE makes the frame stable across reruns. Remove its old attribute
        # nodes first so rerunning ingestion doesn't duplicate attributes.
        graph.query(
            """
            MATCH (v:Video {video_id: $video_id})
            MERGE (f:Frame {frame_id: $frame_id})
            SET f.name = $frame_name
            MERGE (v)-[:HAS_FRAME]->(f)
            WITH f
            OPTIONAL MATCH (f)-[:HAS_ATTRIBUTE]->(old:FrameAttribute)
            DETACH DELETE old
            """,
            {
                "video_id": meta["video_id"],
                "frame_id": frame_id,
                "frame_name": frame_name,
            },
        )

        # Each JSON field becomes a separate attribute node. Do not use the
        # BLIP caption as a replacement for scene_description; store fields as-is.
        for key, original_value in record.items():
            if original_value is None:
                # FalkorDB properties cannot store null; omit absent values.
                continue

            if isinstance(original_value, (dict, list)):
                value = json.dumps(original_value, ensure_ascii=False)
            else:
                value = original_value

            graph.query(
                """
                MATCH (f:Frame {frame_id: $frame_id})
                CREATE (a:FrameAttribute {
                    name: $attribute_name,
                    key: $attribute_name,
                    value: $attribute_value
                })
                CREATE (f)-[:HAS_ATTRIBUTE]->(a)
                """,
                {
                    "frame_id": frame_id,
                    "attribute_name": key,
                    "attribute_value": value,
                },
            )

    # Verify the counts after ingestion.
    result = graph.query(
        """
        MATCH (v:Video {video_id: $video_id})-[:HAS_FRAME]->(f:Frame)
        RETURN v.name AS video, count(f) AS frames
        """,
        {"video_id": meta["video_id"]},
    )
    attribute_result = graph.query(
        """
        MATCH (v:Video {video_id: $video_id})-[:HAS_FRAME]->(f:Frame)
              -[:HAS_ATTRIBUTE]->(a:FrameAttribute)
        RETURN count(a) AS attributes
        """,
        {"video_id": meta["video_id"]},
    )
    print("FalkorDB frame verification:", result.result_set)
    print("FalkorDB attribute verification:", attribute_result.result_set)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True); p.add_argument("--output-dir",default="argusgraph_output")
    p.add_argument("--graph-name",default="video_graph"); p.add_argument("--no-graph",action="store_true")
    p.add_argument("--no-gemini",action="store_true"); p.add_argument("--base-interval",type=float,default=1.0)
    p.add_argument("--scan-interval",type=float,default=0.2); p.add_argument("--change-threshold",type=float,default=0.13)
    args=p.parse_args(); video=Path(args.video)
    if not video.exists(): raise FileNotFoundError(f"Video not found: {video}")
    out=Path(args.output_dir); frames_dir=out/"frames"; emb_dir=out/"embeddings"; emb_dir.mkdir(parents=True,exist_ok=True)
    meta=adaptive_sample(video,frames_dir,args.base_interval,args.scan_interval,threshold=args.change_threshold)
    if not meta["frames"]: raise RuntimeError("No frames were sampled")
    print(f"Selected {len(meta['frames'])} frames from {meta['duration_seconds']}s video")
    device="cuda" if torch.cuda.is_available() else "cpu"; models=load_models(device)
    key=os.getenv("GEMINI_API_KEY") or (GEMINI_API_KEY if GEMINI_API_KEY != "PASTE_YOUR_GEMINI_API_KEY_HERE" else None); client=None
    if not args.no_gemini and key: client=genai.Client(api_key=key)
    elif not args.no_gemini: print("GEMINI_API_KEY missing; Gemini-eligible frames will use fallback")
    records=[]; clip_vecs=[]; blip_vecs=[]
    for i,sampled in enumerate(meta["frames"],1):
        print(f"[{i}/{len(meta['frames'])}] {sampled['frame_id']}")
        r=enrich_frame(sampled,models)
        if r["gemini_eligible"] and client is not None and not args.no_gemini:
            g=gemini_scene(client,r)
        elif r["gemini_eligible"]:
            g={"status":"fallback_blip_yolo","scene":{"scene_description":r["blip_caption"],"scene_type":"unknown","activities":[],"safety_concerns":[],
                "visible_evidence":[f"YOLO detected {d['label']} (confidence {d['confidence']})" for d in r["detections"]],
                "uncertainty":["Gemini skipped/unavailable; BLIP/YOLO fallback used."]},"error":"Gemini disabled or API key missing"}
        else: g={"status":"skipped_object_count_le_1","scene":None,"error":None}
        r["gemini_status"]=g["status"]; r["gemini_error"]=g["error"]; r["gemini_scene"]=g["scene"]
        clip_vecs.append(r.pop("clip_embedding")); blip_vecs.append(r.pop("blip_embedding")); records.append(r)
    emb_path=emb_dir/f"{meta['video_id']}_embeddings.npz"
    np.savez_compressed(emb_path,frame_ids=np.array([r["frame_id"] for r in records]),
        timestamps=np.array([r["timestamp_seconds"] for r in records],dtype=np.float32),
        clip_embeddings=np.vstack(clip_vecs).astype(np.float32),blip_embeddings=np.vstack(blip_vecs).astype(np.float32))
    # Graph fields are flattened from Gemini schema for convenient Cypher queries.
    for r in records:
        s=r.get("gemini_scene") or {}
        r["scene_description"]=s.get("scene_description",""); r["scene_type"]=s.get("scene_type","unknown")
        r["activities"]=s.get("activities",[]); r["safety_concerns"]=s.get("safety_concerns",[])
        r["visible_evidence"]=s.get("visible_evidence",[]); r["uncertainty"]=s.get("uncertainty",[])
    write_json(out/"argusgraph_index.json",{**meta,"embedding_file":str(emb_path),"frames":records})
    write_json(out/"gemini_scenes.json",{"video_id":meta["video_id"],"frames":[{"frame_id":r["frame_id"],"timestamp_seconds":r["timestamp_seconds"],"object_count":r["object_count"],"gemini_status":r["gemini_status"],"scene":r["gemini_scene"],"error":r["gemini_error"]} for r in records if r["gemini_eligible"]]})
    if not args.no_graph:
        try: ingest_falkordb(args.graph_name,meta,records)
        except Exception as e: print(f"WARNING: FalkorDB ingestion failed; local outputs are saved. Error: {e}")
    print("Done. JSON index:",out/"argusgraph_index.json"); print("Gemini scenes:",out/"gemini_scenes.json"); print("Embeddings (outside graph):",emb_path)

if __name__=="__main__": 
    main()