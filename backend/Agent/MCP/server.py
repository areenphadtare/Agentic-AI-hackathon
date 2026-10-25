"""ArgusGraph read-only MCP server for FalkorDB.

Exposes bounded, parameterized tools for an LLM agent to investigate video
frames without loading the whole graph into its context.
"""
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from falkordb import FalkorDB

load_dotenv()

mcp = FastMCP(
    name="ArgusGraph MCP",
    instructions=(
        "Investigate CCTV/video evidence using read-only FalkorDB tools. "
        "Start with list_videos or search_frames, then retrieve frame details "
        "and temporal context. Treat stored descriptions as evidence, not certainty. "
        "Cite video_id, frame_id, and timestamps in answers. Never claim an event "
        "is proven if the returned observations do not support it."
    ),
)


def _get_graph():
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    password = os.getenv("FALKORDB_PASSWORD") or None
    graph_name = os.getenv("FALKORDB_GRAPH", "video_graph")
    client = FalkorDB(host=host, port=port, password=password)
    return client.select_graph(graph_name), graph_name


def _rows(query: str, params: dict[str, Any] | None = None) -> list:
    graph, _ = _get_graph()
    result = graph.query(query, params or {})
    return result.result_set


def _json_safe(value: Any) -> Any:
    """Convert driver values and JSON-encoded attribute values to tool-friendly data."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{\"" or stripped in ("true", "false", "null"):
                try:
                    return _json_safe(json.loads(stripped))
                except (json.JSONDecodeError, TypeError):
                    pass
        return value
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _attribute_map(raw_attributes: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if not isinstance(raw_attributes, list):
        return attributes
    for item in raw_attributes:
        if not isinstance(item, dict):
            continue
        key = item.get("name") or item.get("key")
        if key:
            attributes[str(key)] = _json_safe(item.get("value"))
    return attributes


FRAME_SELECT = """
MATCH (v:Video)-[:HAS_FRAME]->(f:Frame)
OPTIONAL MATCH (f)-[:HAS_ATTRIBUTE]->(a:FrameAttribute)
WITH v, f, collect(CASE WHEN a IS NULL THEN null ELSE {
    name: coalesce(a.name, a.key), key: coalesce(a.key, a.name), value: a.value
} END) AS raw_attrs
WITH v, f, [x IN raw_attrs WHERE x IS NOT NULL] AS attrs
"""


@mcp.tool
async def list_videos() -> dict[str, Any]:
    """List videos indexed in ArgusGraph, including their names and frame counts."""
    rows = _rows("""
        MATCH (v:Video)
        OPTIONAL MATCH (v)-[:HAS_FRAME]->(f:Frame)
        RETURN v.video_id AS video_id, v.name AS name,
               count(DISTINCT f) AS frame_count
        ORDER BY name
    """)
    return {
        "videos": [
            {"video_id": r[0], "name": r[1], "frame_count": r[2]}
            for r in rows
        ]
    }


@mcp.tool
async def graph_stats() -> dict[str, Any]:
    """Return compact counts of videos, frames, attributes, and graph relationships."""
    graph, graph_name = _get_graph()
    node_rows = graph.query(
        "MATCH (n) RETURN labels(n) AS labels, count(n) AS total"
    ).result_set
    rel_rows = graph.query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS total"
    ).result_set
    return {
        "graph_name": graph_name,
        "nodes_by_label": {
            str(r[0][0] if r[0] else "unknown"): r[1] for r in node_rows
        },
        "relationships": {str(r[0]): r[1] for r in rel_rows},
    }


@mcp.tool
async def get_graph_overview(
    video_id: str | None = None,
    frame_limit: int = 5,
    frame_offset: int = 0,
    sample_attribute_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Return graph totals and a paginated inventory of frames.

    Call this first to discover the graph, available video IDs, total frame counts,
    and frame IDs/names. Pagination is over frames, not videos. Set video_id to
    inspect one video; leave it null to page through frames across all videos.
    Use next_frame_offset when pagination.has_more is true.
    """
    frame_limit = max(1, min(int(frame_limit), 200))
    frame_offset = max(0, int(frame_offset))
    default_keys = [
        "timestamp_seconds",
        "scene_type",
        "scene_description",
        "object_count",
        "activities",
        "safety_concerns",
    ]
    selected_keys = sample_attribute_keys or default_keys

    graph, graph_name = _get_graph()

    node_rows = graph.query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS total ORDER BY label"
    ).result_set
    relationship_rows = graph.query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS total ORDER BY type"
    ).result_set

    video_rows = graph.query("""
        MATCH (v:Video)
        OPTIONAL MATCH (v)-[:HAS_FRAME]->(f:Frame)
        RETURN v.video_id AS video_id, v.name AS video_name,
               count(DISTINCT f) AS total_frames
        ORDER BY video_name, video_id
    """).result_set
    videos = [
        {"video_id": row[0], "video_name": row[1], "total_frames": int(row[2] or 0)}
        for row in video_rows
    ]
    total_videos = len(videos)
    total_frames_all = sum(video["total_frames"] for video in videos)

    if video_id is not None:
        selected_video = next((v for v in videos if v["video_id"] == video_id), None)
        if selected_video is None:
            return {
                "graph_name": graph_name,
                "error": f"Unknown video_id: {video_id}",
                "available_video_ids": [v["video_id"] for v in videos if v["video_id"] is not None],
            }
        total_frames = selected_video["total_frames"]
    else:
        total_frames = total_frames_all

    # Page frames and their attributes. Avoid type-sensitive operations on attribute values.
    frame_query = """
        MATCH (v:Video)-[:HAS_FRAME]->(f:Frame)
        WHERE ($video_id IS NULL OR v.video_id = $video_id)
        OPTIONAL MATCH (f)-[:HAS_ATTRIBUTE]->(a:FrameAttribute)
        WITH v, f, collect(CASE WHEN a IS NULL THEN null ELSE {
            name: coalesce(a.name, a.key),
            key: coalesce(a.key, a.name),
            value: a.value
        } END) AS raw_attrs
        WITH v, f, [x IN raw_attrs WHERE x IS NOT NULL] AS attrs
        RETURN v.video_id AS video_id, v.name AS video_name,
               f.frame_id AS frame_id, f.name AS frame_name, attrs
        ORDER BY video_id, f.name, frame_id
        SKIP $offset
        LIMIT $limit
    """
    frame_rows = graph.query(frame_query, {
        "video_id": video_id,
        "offset": frame_offset,
        "limit": frame_limit,
    }).result_set

    frames: list[dict[str, Any]] = []
    for row in frame_rows:
        attributes = _attribute_map(row[4])
        compact_attributes = {
            key: _json_safe(attributes[key])
            for key in selected_keys
            if key in attributes
        }
        timestamp = attributes.get("timestamp_seconds")
        if isinstance(timestamp, list) and len(timestamp) == 1:
            timestamp = timestamp[0]
        frames.append({
            "video_id": row[0],
            "video_name": row[1],
            "frame_id": row[2],
            "frame_name": row[3],
            "timestamp_seconds": _json_safe(timestamp),
            "attributes": compact_attributes,
        })

    next_offset = frame_offset + len(frames)
    has_more = next_offset < total_frames
    return {
        "graph_name": graph_name,
        "overview": {
            "total_videos": total_videos,
            "total_frames": total_frames_all,
            "frames_in_scope": total_frames,
            "scope_video_id": video_id,
            "node_counts_by_label": {str(row[0]): row[1] for row in node_rows},
            "relationship_counts_by_type": {str(row[0]): row[1] for row in relationship_rows},
        },
        "videos": videos,
        "pagination": {
            "scope_video_id": video_id,
            "frame_limit": frame_limit,
            "frame_offset": frame_offset,
            "frames_returned": len(frames),
            "total_frames_in_scope": total_frames,
            "has_more": has_more,
            "next_frame_offset": next_offset if has_more else None,
        },
        "frames": frames,
        "how_to_use": {
            "next_step": "Use a video_id from videos to scope searches. Use frame_id with get_frame_details or timestamp_seconds with get_temporal_context. If pagination.has_more is true, call get_graph_overview again with the same video_id and pagination.next_frame_offset.",
            "available_video_ids": [v["video_id"] for v in videos if v.get("video_id") is not None],
        },
    }


@mcp.tool
async def search_frames(
    query: str = "",
    video_id: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    scene_type: str | None = None,
    object_label: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search frame attributes safely, including list/dict-valued attributes.

    Cypher only narrows by video_id and retrieves a bounded candidate set. All
    type-sensitive matching is done in Python so list-valued properties never
    reach Cypher toString()/CONTAINS expressions.
    """
    limit = max(1, min(int(limit), 100))
    candidate_limit = 5000
    params = {"video_id": video_id, "candidate_limit": candidate_limit}

    # IMPORTANT: no toString(), CONTAINS, or numeric conversion of a.value in
    # this query. Attribute values may be lists/maps in the actual graph schema.
    cypher = FRAME_SELECT + """
        WHERE ($video_id IS NULL OR v.video_id = $video_id)
        WITH v, f, attrs
        RETURN v.video_id AS video_id, v.name AS video_name,
               f.frame_id AS frame_id, f.name AS frame_name,
               attrs AS attributes
        ORDER BY f.name
        LIMIT $candidate_limit
    """
    rows = _rows(cypher, params)

    def flatten_text(value: Any) -> str:
        """Convert nested JSON-like values to searchable text in Python."""
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(
                f"{flatten_text(k)} {flatten_text(v)}" for k, v in value.items()
            ).lower()
        if isinstance(value, (list, tuple, set)):
            return " ".join(flatten_text(item) for item in value).lower()
        return str(value).lower()

    def as_float(value: Any) -> float | None:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    query_text = query.strip().lower()
    scene_text = scene_type.strip().lower() if scene_type else ""
    object_text = object_label.strip().lower() if object_label else ""
    results: list[dict[str, Any]] = []

    for r in rows:
        attributes = _attribute_map(r[4])
        timestamp = as_float(attributes.get("timestamp_seconds"))

        if start_time is not None and (timestamp is None or timestamp < start_time):
            continue
        if end_time is not None and (timestamp is None or timestamp > end_time):
            continue

        if scene_text and scene_text not in flatten_text(attributes.get("scene_type")):
            continue

        if object_text:
            object_values = " ".join(
                flatten_text(attributes.get(key))
                for key in ("detection_labels", "detections", "objects")
            )
            if object_text not in object_values:
                continue

        if query_text:
            searchable_text = " ".join(
                f"{key} {flatten_text(value)}"
                for key, value in attributes.items()
            ).lower()
            if query_text not in searchable_text:
                continue

        results.append({
            "video_id": r[0],
            "video_name": r[1],
            "frame_id": r[2],
            "frame_name": r[3],
            "timestamp_seconds": timestamp,
            "attributes": attributes,
        })

    results.sort(key=lambda item: (
        item["timestamp_seconds"] is None,
        item["timestamp_seconds"] if item["timestamp_seconds"] is not None else 0.0,
    ))
    matched_count = len(results)
    results = results[:limit]
    return {
        "count": len(results),
        "matched_count_in_scanned_candidates": matched_count,
        "candidates_scanned": len(rows),
        "candidate_scan_truncated": len(rows) >= candidate_limit,
        "results": results,
        "note": (
            "Lexical search is performed in Python over stored attribute values, "
            "including lists/maps. It is not vector/semantic similarity. "
            "If candidate_scan_truncated is true, narrow the search with video_id."
        ),
    }


@mcp.tool
async def get_frame_details(frame_id: str) -> dict[str, Any]:
    """Get one frame and all its stored attributes by exact frame_id."""
    rows = _rows("""
        MATCH (v:Video)-[:HAS_FRAME]->(f:Frame {frame_id: $frame_id})
        OPTIONAL MATCH (f)-[:HAS_ATTRIBUTE]->(a:FrameAttribute)
        RETURN v.video_id AS video_id, v.name AS video_name,
               f.frame_id AS frame_id, f.name AS frame_name,
               collect(CASE WHEN a IS NULL THEN null ELSE {
                   name: coalesce(a.name, a.key),
                   key: coalesce(a.key, a.name),
                   value: a.value
               } END) AS attributes
    """, {"frame_id": frame_id})
    if not rows:
        return {"found": False, "frame_id": frame_id}
    r = rows[0]
    return {
        "found": True,
        "video_id": r[0],
        "video_name": r[1],
        "frame_id": r[2],
        "frame_name": r[3],
        "attributes": _attribute_map(r[4]),
    }


@mcp.tool
async def get_temporal_context(
    video_id: str,
    timestamp_seconds: float,
    before_seconds: float = 5.0,
    after_seconds: float = 5.0,
    limit: int = 30,
) -> dict[str, Any]:
    """Retrieve frames in a bounded time window around an observation.

    Window is clamped to at most 10 minutes total and limit is capped to keep
    the LLM context small.
    """
    before_seconds = max(0.0, min(float(before_seconds), 300.0))
    after_seconds = max(0.0, min(float(after_seconds), 300.0))
    limit = max(1, min(int(limit), 100))
    start = max(0.0, timestamp_seconds - before_seconds)
    end = timestamp_seconds + after_seconds
    cypher = FRAME_SELECT + """
        WHERE v.video_id = $video_id
          AND any(x IN attrs WHERE x.name = 'timestamp_seconds'
              AND toFloat(toString(x.value)) >= $start)
          AND any(x IN attrs WHERE x.name = 'timestamp_seconds'
              AND toFloat(toString(x.value)) <= $end)
        WITH v, f, attrs,
             [x IN attrs WHERE x.name = 'timestamp_seconds' |
                toFloat(toString(x.value))][0] AS ts
        RETURN v.video_id AS video_id, v.name AS video_name,
               f.frame_id AS frame_id, f.name AS frame_name,
               ts AS timestamp_seconds, attrs AS attributes
        ORDER BY ts
        LIMIT $limit
    """
    rows = _rows(cypher, {
        "video_id": video_id,
        "start": start,
        "end": end,
        "limit": limit,
    })
    return {
        "video_id": video_id,
        "window": {"start_seconds": start, "end_seconds": end},
        "count": len(rows),
        "frames": [{
            "frame_id": r[2],
            "frame_name": r[3],
            "timestamp_seconds": r[4],
            "attributes": _attribute_map(r[5]),
        } for r in rows],
    }


@mcp.tool
async def get_frame_neighbors(
    frame_id: str,
    before_count: int = 3,
    after_count: int = 3,
) -> dict[str, Any]:
    """Get nearby sampled frames ordered by timestamp around a frame ID.

    This uses timestamp attributes, not an assumed NEXT_FRAME relationship.
    """
    before_count = max(0, min(int(before_count), 20))
    after_count = max(0, min(int(after_count), 20))
    base = await get_frame_details(frame_id)
    if not base.get("found"):
        return {"found": False, "frame_id": frame_id, "neighbors": []}
    ts = base["attributes"].get("timestamp_seconds")
    video_id = base["video_id"]
    if ts is None:
        return {
            "found": True,
            "frame": base,
            "neighbors": [],
            "note": "Frame has no timestamp_seconds attribute.",
        }
    context = await get_temporal_context(
        video_id=video_id,
        timestamp_seconds=float(ts),
        before_seconds=300,
        after_seconds=300,
        limit=100,
    )
    frames = [x for x in context["frames"] if x["frame_id"] != frame_id]
    before = [
        x for x in frames
        if x.get("timestamp_seconds") is not None
        and float(x["timestamp_seconds"]) < float(ts)
    ]
    after = [
        x for x in frames
        if x.get("timestamp_seconds") is not None
        and float(x["timestamp_seconds"]) > float(ts)
    ]
    before = before[-before_count:] if before_count else []
    after = after[:after_count] if after_count else []
    return {"found": True, "frame": base, "neighbors": before + after}


@mcp.tool
async def run_readonly_cypher(
    query: str,
    parameters_json: str = "{}",
) -> dict[str, Any]:
    """Run a restricted read-only Cypher query for advanced investigations.

    Only one query beginning with MATCH or OPTIONAL MATCH and containing RETURN
    is accepted. Writes, multiple statements, schema changes, and admin commands
    are rejected. Prefer purpose-built tools for normal questions.
    """
    normalized = query.strip()
    upper = normalized.upper()
    forbidden = [
        "CREATE ", "MERGE ", "DELETE ", "DETACH ", " SET ", " REMOVE ",
        " DROP ", " CALL ", " FOREACH ", ";", "LOAD CSV", "GRANT ", "REVOKE ",
    ]
    if not (upper.startswith("MATCH ") or upper.startswith("OPTIONAL MATCH ")):
        return {"ok": False, "error": "Query must start with MATCH or OPTIONAL MATCH."}
    if "RETURN" not in upper or any(token in upper for token in forbidden):
        return {"ok": False, "error": "Only a single read-only MATCH/RETURN query is allowed."}
    try:
        params = json.loads(parameters_json)
        if not isinstance(params, dict):
            raise ValueError("parameters_json must decode to an object")
    except Exception as exc:
        return {"ok": False, "error": f"Invalid parameters_json: {exc}"}
    try:
        rows = _rows(normalized, params)
        return {
            "ok": True,
            "row_count": len(rows),
            "rows": [_json_safe(r) for r in rows[:100]],
            "truncated": len(rows) > 100,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    mcp.run(transport="http", host=host, port=port, path="/mcp")