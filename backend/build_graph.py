import os
import json
from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

JSON_PATH = os.getenv(
    "ARGUSGRAPH_JSON",
    "argusgraph_output/argusgraph_index.json"
)

GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "video_graph")

db = FalkorDB(
    host=os.getenv("FALKORDB_HOST", "localhost"),
    port=int(os.getenv("FALKORDB_PORT", "6379")),
    password=os.getenv("FALKORDB_PASSWORD") or None,
)

graph = db.select_graph(GRAPH_NAME)


# --------------------------------------------------
# 2. Read JSON
# --------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

video_id = data.get("video_id", "Sample")
video_name = data.get("video_name", "Sample.mp4")
frames = data.get("frames", [])

if not frames:
    raise ValueError("No frames found in the JSON file.")

print(f"Video: {video_name}")
print(f"Frames found: {len(frames)}")
print(f"Target graph: {GRAPH_NAME}")


# --------------------------------------------------
# 3. Helper: prepare frame attributes
# --------------------------------------------------

def prepare_attributes(frame):
    """
    Each returned pair is:
        (attribute_name, attribute_value)

    Nested values are serialized as JSON strings.
    Detections are represented by their labels.
    """

    attributes = []

    # Preserve the frame fields except raw detections,
    # which are converted into detection_labels below.
    for key, value in frame.items():
        if key == "detections":
            continue

        if value is None:
            continue

        # Avoid duplicate video-level fields on every frame.
        # All other available frame fields are retained.
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)

        attributes.append((key, value))

    detections = frame.get("detections", [])

    if isinstance(detections, list):
        labels = [
            item.get("label")
            for item in detections
            if isinstance(item, dict) and item.get("label") is not None
        ]

        if labels:
            attributes.append(("detection_labels", labels))

    return attributes


# --------------------------------------------------
# 4. Build one Cypher CREATE query
# --------------------------------------------------
# A single query creates the Video node, every Frame,
# every FrameAttribute, and their relationships.
#
# No MATCH, MERGE, extra node types, or NEXT_FRAME edges.

clauses = [
    """
    CREATE (v:Video {
        video_id: $video_id,
        name: $video_name
    })
    """
]

params = {
    "video_id": video_id,
    "video_name": video_name,
}

attribute_count = 0

for frame_number, frame in enumerate(frames, start=1):
    frame_id = frame.get("frame_id", f"{video_id}_frame_{frame_number:06d}")

    # Frame node's display name
    frame_name = f"Frame no. {frame_number:03d}"

    frame_var = f"f{frame_number}"
    frame_id_param = f"frame_id_{frame_number}"
    frame_name_param = f"frame_name_{frame_number}"

    params[frame_id_param] = frame_id
    params[frame_name_param] = frame_name

    clauses.append(
        f"""
        CREATE ({frame_var}:Frame {{
            frame_id: ${frame_id_param},
            name: ${frame_name_param}
        }})
        """
    )

    clauses.append(f"CREATE (v)-[:HAS_FRAME]->({frame_var})")

    attributes = prepare_attributes(frame)

    for attribute_number, (key, value) in enumerate(attributes, start=1):
        attribute_var = f"a{frame_number}_{attribute_number}"
        key_param = f"key_{frame_number}_{attribute_number}"
        value_param = f"value_{frame_number}_{attribute_number}"

        params[key_param] = key
        params[value_param] = value

        clauses.append(
            f"""
            CREATE ({attribute_var}:FrameAttribute {{
                name: ${key_param},
                key: ${key_param},
                value: ${value_param}
            }})
            """
        )

        clauses.append(
            f"CREATE ({frame_var})-[:HAS_ATTRIBUTE]->({attribute_var})"
        )

        attribute_count += 1


# --------------------------------------------------
# 5. Execute graph creation
# --------------------------------------------------

cypher = "\n".join(clauses)

graph.query(cypher, params)

print("\nGraph creation completed.")
print(f"Video nodes requested: 1")
print(f"Frame nodes created: {len(frames)}")
print(f"FrameAttribute nodes created: {attribute_count}")
print(f"Graph name: {GRAPH_NAME}")