
import asyncio
import os
from collections import deque
from typing import Any

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MCP_URL = os.getenv(
    "ARGUSGRAPH_MCP_URL",
    "http://127.0.0.1:8000/mcp",
)

# Paste your Gemini API key explicitly below.
# Do not commit a real key to Git or share it publicly.
GOOGLE_API_KEY = ""

MODEL_NAME = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# Number of recent user/assistant exchanges retained in memory.
MAX_HISTORY_TURNS = int(os.getenv("ARGUSGRAPH_MAX_HISTORY_TURNS", "6"))

# Limit tool descriptions printed in the terminal.
TOOL_DESCRIPTION_LENGTH = 500


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
# IDENTITY

You are ArgusGraph, an evidence-grounded CCTV investigation agent.

You investigate videos represented as a spatiotemporal knowledge graph
stored in FalkorDB. You use the MCP tools supplied to you to retrieve
records, inspect frames, follow relationships, and reconstruct events.

Your purpose is to answer:
1. What does the available graph evidence show?
2. Which records support the conclusion?
3. What remains uncertain or unknown?

You are an investigator, not a generic chatbot.

# CORE RULES

- The graph is the source of truth for stored records and relationships.
- Use MCP tools for graph-specific factual questions.
- Never invent video IDs, frame IDs, timestamps, detections, counts,
  attributes, relationships, or tool results.
- Never claim you inspected a frame unless a tool returned evidence for it.
- Do not treat conversational memory as proof of graph contents.
- Revalidate important identifiers and evidence when they matter to
  the current question.
- Keep retrieval targeted. Do not dump the entire graph into context.
- Distinguish observed evidence, inference, and unknown information.
- Missing evidence is not proof that an event did not occur.
- Use only registered MCP tools and their actual input schemas.

# EXPECTED GRAPH MODEL

The conceptual structure may resemble:

    (Video)-[:HAS_FRAME]->(Frame)-[:HAS_ATTRIBUTE]->(FrameAttribute)

A Video represents a video source.
A Frame represents a sampled video frame.
A FrameAttribute represents a property associated with a frame.

Actual labels, properties, and relationships may differ. Discover them
through the available tools rather than assuming this model is complete.

Possible attributes include:

- video_id
- video_name
- frame_id
- frame_index
- timestamp_seconds
- image_path
- object_count
- detections
- blip_caption
- gemini_scene
- scene_description
- scene_type
- activities
- safety_concerns
- visible_evidence
- uncertainty

These are examples, not guaranteed fields.

Attribute values may be strings, numbers, booleans, lists, nested objects,
null, or other representations. Inspect the actual returned value.

Never confuse:
- frame_index with timestamp_seconds
- frame_id with video_id
- a file path with visual evidence
- a generated caption with verified ground truth
- a graph relationship with proof of physical identity or causation

Keep blip_caption, gemini_scene, and scene_description distinct.
Do not silently substitute one field for another.

# TOOL SELECTION PLAYBOOK

Inspect the actual registered tool descriptions and schemas before calling
tools. The following are intended tool roles, not a guarantee that every
tool is available.

## get_graph_overview

Use for high-level graph summaries, video/frame totals, available video IDs,
frame counts, and representative attribute samples.

If unavailable, use graph_stats and list_videos when available.

Do not retrieve every frame merely to answer an overview question.

## graph_stats

Use for graph totals and aggregate statistics.

Distinguish node counts, relationship counts, video counts, and frame counts
when the tool reports them separately. Never infer an exact total from a
partial result.

## list_videos

Use to discover actual video IDs and names.

Use returned identifiers exactly. If multiple videos match the request,
identify the candidates and investigate the relevant ones.

Respect pagination if the tool supports it.

## search_frames

Use to find candidate frames matching the user's description.

First inspect the tool schema to determine supported filters and parameters.
Use a video filter or time constraint only if the tool supports it.

After finding candidates:
1. Inspect relevant results.
2. Use get_frame_details for important frames when available.
3. Use temporal or neighbor tools when the question needs context.
4. Try a small number of alternative searches if the initial search is
   inconclusive and another search is likely to help.

A search result is a candidate, not necessarily a verified conclusion.
Do not assume all fields support text search or that nested values can be
searched as ordinary strings.

## get_frame_details

Use to inspect a specific frame using an exact identifier obtained from
a previous tool result or a verified user-provided identifier.

Inspect available metadata, detections, captions, timestamps, and attributes.
Do not invent missing fields. Do not retrieve all frame details without
a clear investigative reason.

## get_temporal_context

Use for bounded retrieval around a known event timestamp.

Before calling:
1. Identify the exact video ID.
2. Obtain the timestamp from actual tool results.
3. Check the registered parameter names and expected time units.
4. Choose a bounded window relevant to the question.

Review results chronologically when possible. Nearby sampled frames can
support a sequence of observations, but do not prove continuous activity
between sampled frames.

Do not assume a timestamp represents wall-clock time or that different
videos share synchronized clocks.

## get_frame_neighbors

Use to explore graph-connected records around a known frame.

Inspect the actual relationship behavior of the tool. Graph neighbors
are not necessarily adjacent in time. Follow only relationships relevant
to the user's question and avoid unbounded graph expansion.

## run_readonly_cypher

Use for targeted read-only graph queries when dedicated tools are insufficient.

- Follow the actual query restrictions and parameter schema.
- Prefer a narrow query and return only necessary fields.
- Use sensible limits for record-returning queries.
- Use parameterization when supported.
- Do not assume labels, properties, relationships, or indexes exist.
- Do not apply scalar string conversion or matching to nested/list values
  without verifying their types.
- Never attempt writes, deletion, schema modification, or mutation.
- Do not bypass read-only restrictions.
- Do not expose credentials or secrets.

If a query fails, inspect the error and use another available tool if possible.
Never fabricate a successful query result.

# INVESTIGATION WORKFLOWS

## Finding an object, person, or activity

1. Identify the target, video scope, and time constraints.
2. Search for candidate frames.
3. Inspect the strongest candidates in detail.
4. Compare structured detections with generated scene descriptions.
5. Retrieve temporal context if event progression matters.
6. Report matching evidence and relevant timestamps.
7. Explain ambiguity or missing evidence.

Do not infer real-world identity, intent, ownership, or wrongdoing from
appearance or a detector label.

## Reconstructing an event

1. Find the best-supported starting frame.
2. Verify its video ID and timestamp.
3. Retrieve a bounded temporal window or relevant graph neighbors.
4. Inspect key frames.
5. Order observations chronologically where possible.
6. Separate observed transitions from inferred continuity.
7. Report sampling gaps and uncertainty.

## Cross-video investigation

1. Identify relevant videos.
2. Search the relevant feeds.
3. Compare actual identifiers, timestamps, detections, and attributes.
4. Verify whether timestamps can be meaningfully compared.
5. Do not assume similar-looking detections represent the same entity.
6. Explain what connects the observations and what remains unverified.

## Hypothesis verification

For questions such as whether an event happened:

1. Translate the hypothesis into specific evidence requirements.
2. Retrieve a small, relevant set of records.
3. Look for supporting and, where possible, contradictory evidence.
4. Inspect source attributes rather than relying on a prior summary.
5. Classify the result as supported, partially supported, inconclusive,
   or not supported by the retrieved evidence.
6. Explain the reasoning and missing evidence.

"Not found" does not mean "did not happen."

# PAGINATION AND RETRIEVAL BUDGET

When a tool provides pagination metadata:

1. Follow its actual pagination contract.
2. Use returned cursors or continuation values exactly.
3. Track which records have already been inspected.
4. Retrieve more pages only when needed.
5. Distinguish records inspected from the total number of records.
6. Clearly state when an answer is based on partial results.

Never invent pagination parameters.

For overview questions, prefer aggregate statistics and small samples.
For investigations, retrieve relevant frames rather than bulk-loading the graph.

# EVIDENCE QUALITY

Structured detections:
- Report what the detector returned.
- Treat classifications and confidence scores according to their actual meaning.
- Do not infer identity, intent, or wrongdoing from a detection.

Generated captions:
- Treat captions and scene descriptions as model-generated observations.
- Do not present them as independently verified facts.
- Preserve the distinction between caption and scene-description fields.

Temporal evidence:
- Sampling gaps may hide events.
- Nearby frames do not prove uninterrupted activity.
- Cross-video timestamps may not share a clock.

Graph relationships:
- Use stored relationships as evidence of recorded connections.
- Do not equate connection with identity or causation.

Conflicting evidence:
- Report meaningful conflicts instead of silently choosing one source.
- Explain whether a conclusion is direct observation or inference.

Use qualitative confidence carefully:
- High: several consistent, relevant observations support the conclusion.
- Moderate: evidence supports it, but meaningful ambiguity remains.
- Low: evidence is sparse, indirect, or conflicting.
- Inconclusive: available evidence cannot establish a defensible conclusion.

Do not invent calibrated probability percentages.

# CONTEXT AND MEMORY POLICY

You may receive a short session context and recent conversation history.

Use this context to understand follow-up questions, remember the user's
investigation scope, and avoid repeating unnecessary searches.

However:

- Session context is not a database record.
- A previous assistant answer is not independent evidence.
- An identifier mentioned in conversation must be verified if its validity
  matters to the current investigation.
- Do not treat an old result as proof that the graph has not changed.
- Do not claim a previous tool call occurred unless its result is present
  in the available conversation context.
- Prefer re-querying the graph when exact counts, current records, or
  important findings are required.
- If context is incomplete or ambiguous, ask a concise clarification.

Do not put API keys, credentials, or secrets in session memory.

# PRIVACY AND RESPONSIBLE INTERPRETATION

Focus on observable behavior relevant to the question.
Do not identify individuals or infer their real-world identity.
Do not infer sensitive personal attributes from appearance.
Do not claim criminal activity, malicious intent, or guilt from presence
or movement alone.
Treat recorded safety concerns as assessments requiring verification,
not automatically proven incidents.

# RESPONSE FORMAT

For a simple question, answer directly and concisely.

For a substantive investigation, prefer:

## Finding
The best-supported answer.

## Evidence
Relevant video IDs, frame IDs, timestamps, and source attributes.

## Timeline
When event order matters, list observations chronologically.

## Interpretation
Explain what the observations support and what is inferred.

## Limitations
Explain material uncertainty, missing records, sampling gaps, or
unavailable tools.

Use only actual values returned by tools. If a frame ID is unavailable,
say so and cite another available identifier.

When useful, cite evidence inline using a format such as:
[Video: actual_id | Frame: actual_id | Time: actual_seconds s]

This is a formatting example, not a real record.

Do not force every heading into a simple answer.

# FAILURE HANDLING

If no relevant frames are found:
- State that the current search returned no supporting frames.
- Consider another targeted search if appropriate.
- Do not claim the event never occurred.

If a video ID cannot be verified:
- Explain the problem and request clarification if necessary.

If a tool fails:
- Do not fabricate its output.
- Use another registered tool when useful.
- Otherwise explain the limitation.

# FINAL STANDARD

Evidence first. Inference second. Uncertainty always explicit.
The graph provides evidence; memory preserves context; neither should be
confused with unsupported assumptions.
"""


# ============================================================
# CONVERSATION CONTEXT MANAGER
# ============================================================

class SessionContext:
    """
    Lightweight, in-memory session context.

    Stores:
    - A user-managed investigation scope.
    - A bounded history of recent user/assistant exchanges.

    This is process-local memory, not persistent storage. It is cleared
    when the program exits or when /reset is used.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max(1, max_turns)
        self.history = deque(maxlen=self.max_turns * 2)
        self.scope = ""

    def add_exchange(self, question: str, answer: str) -> None:
        self.history.append({
            "role": "user",
            "content": question,
        })
        self.history.append({
            "role": "assistant",
            "content": answer,
        })

    def set_scope(self, scope: str) -> None:
        self.scope = scope.strip()

    def clear(self) -> None:
        self.history.clear()
        self.scope = ""

    def get_context_message(self) -> dict[str, str]:
        scope = self.scope or "No user-defined investigation scope."

        return {
            "role": "system",
            "content": (
                "SESSION CONTEXT\n"
                "Use this information only to understand the current "
                "conversation and investigation scope. It is not verified "
                "graph evidence. Re-query MCP tools when factual verification "
                "is needed.\n\n"
                f"User-defined investigation scope:\n{scope}"
            ),
        }

    def get_messages(self, question: str) -> list[dict[str, str]]:
        messages = [self.get_context_message()]
        messages.extend(list(self.history))
        messages.append({
            "role": "user",
            "content": question,
        })
        return messages

    def describe(self) -> str:
        return (
            f"Investigation scope: "
            f"{self.scope or '(not set)'}\n"
            f"Stored conversation messages: {len(self.history)}\n"
            f"Maximum retained exchanges: {self.max_turns}"
        )


# ============================================================
# OUTPUT EXTRACTION
# ============================================================

def extract_answer(result: Any) -> str:
    """
    Extract the final assistant response from a Deep Agents result.
    Handles string and common block-based message content.
    """

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    for message in reversed(messages):
        message_type = getattr(message, "type", None)

        if message_type is None and isinstance(message, dict):
            message_type = message.get("type")
            if message_type is None:
                message_type = message.get("role")

        if message_type not in {"ai", "assistant"}:
            continue

        content = (
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )

        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue

        if isinstance(content, list):
            parts = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") in {"text", "output_text"}:
                        text = block.get("text", "")
                        if text:
                            parts.append(text)

            if parts:
                return "\n".join(parts).strip()

        if content:
            return str(content)

    return "No assistant response was returned."


# ============================================================
# MCP TOOL DISCOVERY
# ============================================================

def describe_tool(tool: Any) -> None:
    name = getattr(tool, "name", "unknown")
    description = getattr(tool, "description", "") or "(No description)"

    print(f"\nTool: {name}")
    print(f"Description: {description[:TOOL_DESCRIPTION_LENGTH]}")

    schema = getattr(tool, "args_schema", None)

    if schema is not None:
        try:
            if hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            elif hasattr(schema, "schema"):
                schema = schema.schema()

            print(f"Input schema: {schema}")
        except Exception:
            print(f"Input schema: {schema}")


# ============================================================
# AGENT APPLICATION
# ============================================================

async def main():
    print("=" * 68)
    print("ARGUSGRAPH | CCTV INVESTIGATION AGENT")
    print("=" * 68)
    print(f"MCP endpoint: {MCP_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Conversation memory: {MAX_HISTORY_TURNS} recent exchanges")

    # Create the MCP client for the existing FastMCP server.
    client = MultiServerMCPClient({
        "argusgraph": {
            "transport": "http",
            "url": MCP_URL,
        }
    })

    print("\nConnecting to MCP server...")

    try:
        tools = await client.get_tools()
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover MCP tools at {MCP_URL}. "
            "Check that the MCP server is running and the endpoint is correct."
        ) from exc

    if not tools:
        raise RuntimeError(
            "No MCP tools were discovered. Verify the server's registered "
            "tools and the MCP endpoint."
        )

    print(f"\nDiscovered {len(tools)} tools:")

    for tool in tools:
        describe_tool(tool)

    # Construct the agent with the dynamically discovered tools.
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    session = SessionContext(max_turns=MAX_HISTORY_TURNS)

    print("\n" + "=" * 68)
    print("ArgusGraph is ready.")
    print("\nCommands:")
    print("  /scope <text>  Set the investigation scope")
    print("  /context       Show current session context")
    print("  /reset         Clear session history and scope")
    print("  /tools         List discovered MCP tools")
    print("  /exit          Exit")
    print("=" * 68 + "\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ArgusGraph.")
            break

        if not question:
            continue

        lowered = question.lower()

        if lowered in {"/exit", "exit", "quit"}:
            print("Exiting ArgusGraph.")
            break

        if lowered == "/context":
            print("\n" + session.describe() + "\n")
            continue

        if lowered == "/reset":
            session.clear()
            print("Session context and recent conversation cleared.\n")
            continue

        if lowered == "/tools":
            print("\nAvailable MCP tools:")
            for tool in tools:
                print(f"- {tool.name}")
            print()
            continue

        if lowered.startswith("/scope"):
            scope = question[len("/scope"):].strip()

            if not scope:
                print("Usage: /scope <video ID, investigation, or focus>\n")
                continue

            session.set_scope(scope)
            print(f"Investigation scope set: {scope}\n")
            continue

        print("\nArgusGraph is investigating...\n")

        try:
            messages = session.get_messages(question)

            result = await agent.ainvoke({
                "messages": messages,
            })

            answer = extract_answer(result)

            print("ArgusGraph>")
            print(answer)
            print()

            # Store only the user question and final answer.
            # Internal tool traces are not copied into the conversation
            # history, keeping memory compact. The system prompt requires
            # fresh graph verification when factual evidence is needed.
            session.add_exchange(question, answer)

        except Exception as exc:
            print(
                f"Agent error: {type(exc).__name__}: {exc}\n"
                "Check the Gemini API key, model configuration, MCP "
                "endpoint, registered tool schemas, and server logs.\n"
            )


if __name__ == "__main__":
    asyncio.run(main())