from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


GOOGLE_API_KEY = ""
AGENT_MODEL_NAME = os.getenv("ARGUSGRAPH_AGENT_MODEL", "gemini-2.5-flash")

MCP_URL = os.getenv("ARGUSGRAPH_MCP_URL", "http://127.0.0.1:8001/mcp")

MAX_HISTORY_TURNS = max(1, int(os.getenv("ARGUSGRAPH_MAX_HISTORY_TURNS", "6")))

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
UPLOAD_DIR = Path(os.getenv("ARGUSGRAPH_UPLOAD_DIR", str(PROJECT_DIR / "uploads"))).resolve()
OUTPUT_DIR = Path(os.getenv("ARGUSGRAPH_OUTPUT_DIR", str(PROJECT_DIR / "argusgraph_output"))).resolve()

PIPELINE_SCRIPT_RAW = Path(os.getenv("ARGUSGRAPH_PIPELINE_SCRIPT", str(PROJECT_DIR / "video_pipeline.py")))

PIPELINE_SCRIPT = (PROJECT_DIR / PIPELINE_SCRIPT_RAW).resolve() if not PIPELINE_SCRIPT_RAW.is_absolute() else PIPELINE_SCRIPT_RAW.resolve()

GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "video_graph")

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")

FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD") or None

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ArgusGraph API", version="0.1.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3001,http://127.0.0.1:3001").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# In-memory job state is suitable for the first local integration milestone.
# Use Redis/a durable queue if this becomes a multi-worker deployment.
jobs: dict[str, dict[str, Any]] = {}

# Agent runtime and lightweight in-memory per-session conversation state.
agent_instance: Any = None
mcp_client: Any = None
agent_init_error: str | None = None
agent_sessions: dict[str, dict[str, Any]] = {}
agent_session_locks: dict[str, asyncio.Lock] = {}

SYSTEM_PROMPT = """
# IDENTITY
You are ArgusGraph, an evidence-grounded CCTV investigation agent. Investigate videos represented as a spatiotemporal knowledge graph in FalkorDB using only the registered MCP tools.

# CORE RULES
- The graph is the source of truth for stored records and relationships.
- Never invent video IDs, frame IDs, timestamps, detections, counts, attributes, relationships, or tool results.
- Never claim you inspected a frame unless a tool returned evidence for it.
- Treat conversation history as context, not graph evidence. Re-query when current factual verification matters.
- Keep retrieval targeted; do not dump the entire graph into context.
- Distinguish observed records from inference. Do not infer real-world identity, intent, ownership, or wrongdoing from appearance or detector labels.
- Missing evidence is not proof that an event did not occur.
- Keep `blip_caption`, `gemini_scene`, and `scene_description` distinct.
- Use only registered MCP tools and their actual input schemas. Never attempt graph writes or mutations.

# GRAPH MODEL
The graph may resemble (Video)-[:HAS_FRAME]->(Frame)-[:HAS_ATTRIBUTE]->(FrameAttribute), but verify actual labels, properties, and relationships through tools. Possible fields include video_id, frame_id, frame_index, timestamp_seconds, object_count, detections, blip_caption, gemini_scene, scene_description, scene_type, activities, safety_concerns, visible_evidence, and uncertainty. These fields are examples, not guarantees.

Never confuse frame_index with timestamp_seconds, frame_id with video_id, a file path with visual evidence, a generated caption with verified ground truth, or a graph relationship with proof of identity or causation.

# TOOL SELECTION
- For graph overview/counts, use `get_graph_overview` or available aggregate tools such as `graph_stats` and `list_videos`. Prefer aggregate summaries over loading all frames.
- To find an object/activity, use `search_frames`, then inspect relevant candidates with `get_frame_details` when available.
- For event progression, verify the video ID and timestamp first, then use bounded `get_temporal_context` or relevant `get_frame_neighbors` tools. Nearby sampled frames do not prove uninterrupted activity.
- Use `run_readonly_cypher` only for targeted read-only queries when dedicated tools are insufficient. Follow its actual schema, limit returned records, and never bypass read-only restrictions.
- Inspect actual tool descriptions and schemas. Tool names above describe intended roles and may not all be registered.
- Respect pagination contracts; do not invent cursors or totals.

# INVESTIGATION METHOD
1. Establish the relevant video, time range, and question.
2. Retrieve a small, relevant set of records.
3. Verify identifiers and inspect source attributes.
4. Compare detections, captions, and scene descriptions without treating generated text as independently verified fact.
5. Order observations chronologically when useful and distinguish observed transitions from inferred continuity.
6. Report meaningful conflicts accurately. Never invent calibrated probability percentages.

# SESSION CONTEXT
Use the provided scope and recent conversation to understand follow-up questions. Session memory is not evidence. Revalidate identifiers and current counts when they matter. Do not store or reveal credentials.

# RESPONSE STYLE — FINAL OBSERVATIONS ONLY
Write a concise, neatly formatted answer focused on the final observations and the evidence that supports them. Do not add a standalone “Limitations” section, generic caveats, long methodology narration, or repetitive summaries.
For a simple question, answer directly. For an investigation, use only the sections that add value, usually:
## Final observations
A short direct conclusion.
## Evidence
A compact bullet list of actual video IDs, frame IDs, timestamps, detections, scene descriptions, or graph attributes returned by tools.
## Timeline
Include only when event order matters.

Use inline evidence references when identifiers are available: [Video: actual_id | Frame: actual_id | Time: actual_seconds s]. Never fabricate identifiers or evidence. Mention uncertainty briefly only when it materially changes what the evidence supports; do not create a separate limitations section. Use Markdown headings and short bullets; avoid dense paragraphs and excessive nested lists.

# FAILURE HANDLING
If no relevant records are found, state that the search returned no supporting records; do not claim the event never occurred. If an identifier cannot be verified, ask a concise clarification. If a tool fails, do not fabricate its result; try another registered tool if appropriate, otherwise state the issue briefly.

Evidence first. Conclusions should be traceable to returned graph records.

"""

class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    session_id: str = Field(..., min_length=1, max_length=128)
    scope: str | None = Field(default=None, max_length=2000)


class AgentResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


def extract_agent_answer(result: Any) -> str:
    """Extract the final assistant message from a Deep Agents result."""
    if not isinstance(result, dict):
        return str(result)
    for message in reversed(result.get("messages", [])):
        message_type = getattr(message, "type", None)
        if message_type is None and isinstance(message, dict):
            message_type = message.get("type") or message.get("role")
        if message_type not in {"ai", "assistant"}:
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                    if block.get("text"):
                        parts.append(block["text"])
            if parts:
                return "\n".join(parts).strip()
        if content:
            return str(content)
    return "No assistant response was returned."


def get_agent_session(session_id: str) -> dict[str, Any]:
    return agent_sessions.setdefault(session_id, {"scope": "", "history": []})


async def initialize_agent() -> None:
    """Discover MCP tools and build the graph-investigation agent once."""
    global agent_instance, mcp_client, agent_init_error
    if not GOOGLE_API_KEY:
        agent_init_error = "Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in backend .env."
        return
    try:
        from deepagents import create_deep_agent
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_google_genai import ChatGoogleGenerativeAI

        mcp_client = MultiServerMCPClient({
            "argusgraph": {"transport": "http", "url": MCP_URL}
        })
        tools = await mcp_client.get_tools()
        if not tools:
            raise RuntimeError(f"No MCP tools discovered at {MCP_URL}.")
        llm = ChatGoogleGenerativeAI(
            model=AGENT_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
        )
        agent_instance = create_deep_agent(
            model=llm, tools=tools, system_prompt=SYSTEM_PROMPT
        )
        agent_init_error = None
        print(f"ArgusGraph agent initialized with {len(tools)} MCP tools from {MCP_URL}.")
    except Exception as exc:
        agent_instance = None
        agent_init_error = f"Agent initialization failed ({type(exc).__name__}): {exc}"
        print(agent_init_error)


@app.on_event("startup")
async def startup_agent() -> None:
    await initialize_agent()



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(video_path: Path, output_dir: Path, job_id: str) -> None:
    jobs[job_id].update(status="processing", stage="Running existing ArgusGraph pipeline", started_at=utc_now())
    try:
        if not PIPELINE_SCRIPT.exists():
            raise FileNotFoundError(
                f"Video pipeline script not found at {PIPELINE_SCRIPT}. Set ARGUSGRAPH_PIPELINE_SCRIPT to the script containing adaptive_sample(), enrich_frame(), gemini_scene(), and ingest_falkordb(). Do not point this at agent.py."
            )
        configured_python = os.getenv("ARGUSGRAPH_PYTHON")
        if configured_python:
            python_path = Path(configured_python)
            pipeline_python = str((PROJECT_DIR / python_path).resolve() if not python_path.is_absolute() else python_path.resolve())
        else:
            windows_venv = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
            posix_venv = PROJECT_DIR / "venv" / "bin" / "python"
            pipeline_python = str(windows_venv if windows_venv.exists() else posix_venv if posix_venv.exists() else Path(os.sys.executable))
        cmd = [
            pipeline_python, str(PIPELINE_SCRIPT),
            "--video", str(video_path), "--output-dir", str(output_dir), "--graph-name", GRAPH_NAME,
        ]
        jobs[job_id]["command"] = cmd
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(PIPELINE_SCRIPT.parent))
        jobs[job_id]["stdout"] = result.stdout[-12000:]
        jobs[job_id]["stderr"] = result.stderr[-12000:]
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline exited with code {result.returncode}: {result.stderr[-3000:]}")
        index_path = output_dir / "argusgraph_index.json"
        if not index_path.exists():
            raise RuntimeError(f"Pipeline completed but index file was not created: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        video_id = str(index.get("video_id") or video_path.stem)
        # Confirm that the same video exists in FalkorDB and return the actual graph records.
        graph = get_graph_for_video(video_id)
        # Return the existing pipeline's frame analysis without recomputing it.
        # Gemini scene_description and BLIP blip_caption remain distinct fields.
        frames = []
        for record in index.get("frames", []):
            scene = record.get("gemini_scene") or {}
            frames.append({
                "frame_id": record.get("frame_id"),
                "timestamp_seconds": record.get("timestamp_seconds"),
                "frame_index": record.get("frame_index"),
                "image_path": record.get("image_path"),
                "object_count": record.get("object_count"),
                "detections": record.get("detections", []),
                "blip_caption": record.get("blip_caption"),
                "gemini_status": record.get("gemini_status"),
                "gemini_error": record.get("gemini_error"),
                "gemini_scene": scene,
                "scene_description": record.get("scene_description", scene.get("scene_description", "")),
                "scene_type": record.get("scene_type", scene.get("scene_type", "unknown")),
                "activities": record.get("activities", scene.get("activities", [])),
                "safety_concerns": record.get("safety_concerns", scene.get("safety_concerns", [])),
                "visible_evidence": record.get("visible_evidence", scene.get("visible_evidence", [])),
                "uncertainty": record.get("uncertainty", scene.get("uncertainty", [])),
            })
        jobs[job_id].update(
            status="completed", stage="Graph loaded from FalkorDB", completed_at=utc_now(),
            video_id=video_id, graph=graph, frames=frames,
            summary={"frames_processed": len(index.get("frames", [])), "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                     "video_name": index.get("video_name", video_path.name), "duration_seconds": index.get("duration_seconds")},
        )
    except Exception as exc:
        jobs[job_id].update(status="failed", stage="Failed", error=str(exc), completed_at=utc_now())


def get_graph_for_video(video_id: str) -> dict[str, Any]:
    """Read the real Video -> Frame -> FrameAttribute graph from FalkorDB."""
    try:
        from falkordb import FalkorDB
        client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT, password=FALKORDB_PASSWORD)
        graph = client.select_graph(GRAPH_NAME)
        result = graph.query(
            """
            MATCH (v:Video {video_id: $video_id})-[r:HAS_FRAME]->(f:Frame)
            OPTIONAL MATCH (f)-[ar:HAS_ATTRIBUTE]->(a:FrameAttribute)
            RETURN v.video_id, v.name, id(v), f.frame_id, f.name, id(f), id(r),
                   a.name, a.key, a.value, id(a), id(ar)
            ORDER BY f.frame_id
            """,
            {"video_id": video_id},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not read graph '{GRAPH_NAME}' from FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}: {exc}") from exc

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for row in result.result_set:
        values = list(row)
        v_id, v_name, v_dbid, f_id, f_name, f_dbid, r_dbid, a_name, a_key, a_value, a_dbid, ar_dbid = (values + [None] * 12)[:12]
        if not f_id:
            continue
        video_node_id = f"video:{v_id}"
        frame_node_id = f"frame:{f_id}"
        nodes[video_node_id] = {"data": {"id": video_node_id, "label": str(v_name or v_id), "type": "video", "meta": "Video source", "video_id": v_id, "db_id": v_dbid}}
        nodes[frame_node_id] = {"data": {"id": frame_node_id, "label": str(f_name or f_id), "type": "frame", "meta": "Frame", "frame_id": f_id, "video_id": v_id, "db_id": f_dbid}}
        edge_id = f"rel:has-frame:{r_dbid if r_dbid is not None else f_id}"
        edges[edge_id] = {"data": {"id": edge_id, "source": video_node_id, "target": frame_node_id, "label": "HAS_FRAME", "type": "HAS_FRAME"}}
        if a_dbid is not None and a_key is not None:
            attr_id = f"attribute:{a_dbid}"
            value_text = str(a_value) if a_value is not None else ""
            nodes[attr_id] = {"data": {"id": attr_id, "label": str(a_key), "type": "attribute", "meta": "Frame attribute", "key": a_key, "value": value_text, "frame_id": f_id, "db_id": a_dbid}}
            rel_id = f"rel:has-attribute:{ar_dbid if ar_dbid is not None else f'{f_id}:{a_dbid}'}"
            edges[rel_id] = {"data": {"id": rel_id, "source": frame_node_id, "target": attr_id, "label": "HAS_ATTRIBUTE", "type": "HAS_ATTRIBUTE"}}

    if not nodes:
        raise RuntimeError(f"No graph records found for video_id={video_id!r} in FalkorDB graph {GRAPH_NAME!r}.")
    # Return data ready for Cytoscape.js; positions are left to the frontend layout engine.
    return {"video_id": video_id, "graph_name": GRAPH_NAME, "nodes": list(nodes.values()), "edges": list(edges.values())}


@app.get("/api/agent/status")
def get_agent_status() -> dict[str, Any]:
    return {
        "ready": agent_instance is not None,
        "model": AGENT_MODEL_NAME,
        "mcp_url": MCP_URL,
        "error": agent_init_error,
    }


@app.post("/api/agent/query")
async def query_agent(request: AgentQueryRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if agent_instance is None:
        raise HTTPException(
            status_code=503,
            detail=agent_init_error or "Agent is not initialized. Check the MCP server and backend logs.",
        )

    session_id = request.session_id
    lock = agent_session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = get_agent_session(session_id)
        if request.scope is not None:
            session["scope"] = request.scope.strip()

        scope = session.get("scope") or "No user-defined investigation scope."
        messages = [{
            "role": "system",
            "content": (
                "SESSION CONTEXT\nUse this only to understand conversation and scope; it is not verified graph evidence. "
                "Re-query MCP tools when factual verification is needed.\n\n"
                f"User-defined investigation scope:\n{scope}"
            ),
        }]
        messages.extend(session["history"][-(MAX_HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": question})
        try:
            result = await agent_instance.ainvoke({"messages": messages})
            answer = extract_agent_answer(result)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Agent query failed ({type(exc).__name__}): {exc}",
            ) from exc

        session["history"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        session["history"] = session["history"][-(MAX_HISTORY_TURNS * 2):]
        return {"session_id": session_id, "question": question, "answer": answer, "scope": session.get("scope", "")}


@app.post("/api/agent/reset")
def reset_agent_session(request: AgentResetRequest) -> dict[str, Any]:
    agent_sessions.pop(request.session_id, None)
    return {"session_id": request.session_id, "reset": True}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "argusgraph-api"}


@app.post("/api/videos/upload", status_code=202)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "video.mp4").name
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
        raise HTTPException(status_code=400, detail="Unsupported video extension. Use MP4, MOV, AVI, MKV, WEBM, or M4V.")
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / filename
    try:
        with video_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = {"job_id": job_id, "status": "queued", "stage": "Upload saved; waiting for processor", "filename": filename, "created_at": utc_now()}
    background_tasks.add_task(asyncio.to_thread, run_pipeline, video_path, output_dir, job_id)
    return {"job_id": job_id, "status": "queued", "filename": filename, "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found. Jobs are currently stored in memory; restart clears them.")
    return job


@app.get("/api/videos/{video_id}/graph")
def get_video_graph(video_id: str) -> dict[str, Any]:
    try:
        return get_graph_for_video(video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/frames")
def get_job_frames(job_id: str) -> dict[str, Any]:
    """Return frame analysis emitted by the existing pipeline, unchanged in meaning."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Pipeline job is {job.get('status')}.")
    return {"job_id": job_id, "video_id": job.get("video_id"), "frames": job.get("frames", [])}