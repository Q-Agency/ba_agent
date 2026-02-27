"""
Session endpoints — the core of the BA Agent API.

POST /session/start    → create session, run first agent turn, return initial messages
POST /session/message  → continue conversation, return new agent messages
POST /session/get      → fetch session + full message history from DB
POST /session/review   → approve or request changes on a generated SPEC
GET  /models           → list available LLM models
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from psycopg.rows import dict_row

from app import database as db
from app.config import settings
from app.routers.project_settings import _blob_to_raw_url
from app.graph.graph import build_graph, extract_finalize_args
from app.graph.nodes import MODEL_REGISTRY, DEFAULT_MODEL
from app.graph.state import DEFAULT_COMPLETENESS, AgentState
from app.schemas.api import (
    CompletenessMap,
    GetSessionRequest,
    Message,
    MessagesResponse,
    ModelInfo,
    ModelsResponse,
    ResetSessionRequest,
    ReviewRequest,
    ReviewResponse,
    SendMessageRequest,
    Session,
    SessionWithMessages,
    StartSessionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph(request: Request):
    """Retrieve the compiled graph from app state."""
    return request.app.state.graph


# Limit concurrent agent runs to prevent resource exhaustion
_agent_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(settings.max_concurrent_agent_runs)
    return _agent_semaphore


def _make_config(session_id: str, model_id: str = DEFAULT_MODEL) -> dict:
    """LangGraph config for a session — thread_id drives checkpointing."""
    return {
        "configurable": {"thread_id": session_id, "model_id": model_id},
        "recursion_limit": 25,
    }


async def _fetch_constitution_content(blob_url: str) -> str | None:
    """Download the raw markdown content of a constitution file from GitHub.
    Retries up to 3 times with exponential backoff on transient failures."""
    raw_url = _blob_to_raw_url(blob_url)
    if not raw_url:
        return None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(raw_url, follow_redirects=True)
                if r.status_code == 200:
                    return r.text
                if r.status_code == 429 and attempt < 2:
                    await asyncio.sleep(2.0 * (2 ** attempt))
                    continue
                logger.warning("Constitution fetch %s → %s", raw_url, r.status_code)
                return None
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
        except Exception:
            logger.warning("Failed to fetch constitution from %s", raw_url, exc_info=True)
            return None
    logger.warning("Failed to fetch constitution from %s after 3 retries: %s", raw_url, last_exc)
    return None


def _db_row_to_session(row: dict) -> Session:
    comp = row.get("completeness") or {}
    if isinstance(comp, str):
        comp = json.loads(comp)
    return Session(
        id=row["id"],
        task_id=row.get("task_id", ""),
        teamwork_task_id=row.get("teamwork_task_id", ""),
        teamwork_task_title=row.get("teamwork_task_title", ""),
        project_name=row.get("project_name", ""),
        status=row.get("status", "in_progress"),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
        created_by=row.get("created_by", "agent"),
        spec_md=row.get("spec_md"),
        completeness=CompletenessMap(**comp),
        model=row.get("model", "claude-sonnet-4-6"),
    )


def _db_row_to_message(row: dict) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content_type=row["content_type"],
        content=row["content"],
        created_at=str(row["created_at"]),
        sources=row.get("sources"),
    )


async def _run_agent_turn(
    graph,
    session_id: str,
    input_messages: list,
    initial_state: dict | None = None,
    state_overrides: dict | None = None,
    model_id: str = DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    """
    Run one agent turn. Returns (frontend_messages, updates_dict).

    - initial_state: provided only on session start (sets session metadata)
    - state_overrides: DB-sourced values (completeness, spec_md, decisions) to
      inject into the graph state so build_system_prompt sees current data.
    - subsequent turns just pass new messages; LangGraph resumes from checkpoint
    """
    config = _make_config(session_id, model_id)

    graph_input: dict = {"messages": input_messages}
    if initial_state:
        graph_input.update(initial_state)
    if state_overrides:
        graph_input.update(state_overrides)

    async with _get_semaphore():
        try:
            async with asyncio.timeout(settings.agent_timeout_seconds):
                final_state: AgentState = await graph.ainvoke(graph_input, config)
        except TimeoutError:
            logger.error(
                "Agent timed out after %ds for session %s",
                settings.agent_timeout_seconds, session_id,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Agent timed out after {settings.agent_timeout_seconds}s. Please try again.",
            )

    # Extract finalize_turn args from last AI message
    finalize_args = extract_finalize_args(final_state)
    if finalize_args is None:
        # Fallback: agent returned plain text without calling finalize_turn
        last_msg = final_state["messages"][-1]
        text = getattr(last_msg, "content", str(last_msg))
        logger.info("Model did not call finalize_turn — using plain text fallback")
        finalize_args = {
            "messages": [{"content_type": "text", "content": text, "citations": []}],
            "completeness": None,
            "decisions": [],
            "spec_md": None,
        }
    else:
        logger.info("finalize_turn args: %s", json.dumps(finalize_args, default=str)[:2000])

    raw_messages = finalize_args.get("messages", [])
    new_completeness = finalize_args.get("completeness")
    decisions = finalize_args.get("decisions") or []
    spec_md = finalize_args.get("spec_md")

    # Some models (e.g. Qwen via vLLM) return tool args as JSON strings — parse them
    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except (json.JSONDecodeError, TypeError):
            raw_messages = [{"content_type": "text", "content": raw_messages, "citations": []}]
    if isinstance(decisions, str):
        try:
            decisions = json.loads(decisions)
        except (json.JSONDecodeError, TypeError):
            decisions = []
    if isinstance(spec_md, str):
        # Some models return spec_md as a JSON-wrapped string — unwrap it
        if spec_md.startswith(("{", '"')):
            try:
                parsed = json.loads(spec_md)
                if isinstance(parsed, str):
                    spec_md = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        # LLMs often return literal \n instead of real newlines in tool args
        if isinstance(spec_md, str) and "\\n" in spec_md:
            spec_md = spec_md.replace("\\n", "\n")
    if isinstance(new_completeness, str):
        try:
            new_completeness = json.loads(new_completeness)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse completeness string: %r", new_completeness)
            new_completeness = None

    # Use agent's full completeness map if provided, else keep current state
    current_completeness = dict(final_state.get("completeness") or DEFAULT_COMPLETENESS)
    if isinstance(new_completeness, dict):
        for key in current_completeness:
            if key in new_completeness:
                val = new_completeness[key]
                try:
                    current_completeness[key] = max(0, min(100, int(val)))
                except (ValueError, TypeError):
                    logger.warning("Could not parse completeness value %r for %s", val, key)

    # Fall back to previous spec_md if agent omitted it
    if spec_md is None:
        spec_md = final_state.get("spec_md")

    return raw_messages, {
        "completeness": current_completeness,
        "decisions": decisions,
        "spec_md": spec_md,
        "turn_number": final_state.get("turn_number", 0),
    }


async def _persist_agent_messages(
    session_id: str,
    raw_messages: list[dict],
) -> list[Message]:
    """Write agent messages to ba_messages and return typed Message objects."""
    result: list[Message] = []
    for i, m in enumerate(raw_messages):
        content_type = m.get("content_type", "text")
        content = m.get("content", "")
        citations = m.get("citations") or []
        sources = citations if citations else None

        row = await db.insert_message(
            session_id=session_id,
            role="agent",
            content_type=content_type,
            content=content,
            sources=sources,
        )
        result.append(_db_row_to_message(row))
    return result


# ---------------------------------------------------------------------------
# n8n webhook notification
# ---------------------------------------------------------------------------


async def _notify_n8n(event_path: str, payload: dict, webhook_mode: str = "test"):
    """POST to n8n webhook. Raises on non-2xx so callers can propagate the failure."""
    if not settings.n8n_base_url:
        logger.info("n8n webhook skipped — N8N_BASE_URL not set")
        return
    prefix = "webhook-test" if webhook_mode == "test" else "webhook"
    url = f"{settings.n8n_base_url}/{prefix}/{event_path}"
    logger.info("n8n webhook POST %s  payload keys=%s", url, list(payload.keys()))
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            logger.info("n8n webhook %s → %s  body=%s", url, r.status_code, r.text[:500])
            if r.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"n8n webhook returned {r.status_code}",
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("n8n webhook failed (%s): %s", url, exc)
        raise HTTPException(status_code=502, detail=f"n8n webhook unreachable: {exc}")


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """Return all available LLM models."""
    models = [
        ModelInfo(id=mid, label=entry["label"], provider=entry["provider"])
        for mid, entry in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(models=models)


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------


@router.post("/session/start", response_model=SessionWithMessages)
async def start_session(body: StartSessionRequest, request: Request):
    # ── Constitution gate ─────────────────────────────────────────────
    # A valid constitution file is mandatory for every SPEC session.
    project_id = body.projectId
    if not project_id:
        raise HTTPException(
            status_code=422,
            detail="projectId is required to look up the project constitution.",
        )

    settings_row = await db.get_project_settings(project_id)
    constitution_url = (settings_row or {}).get("constitution_url")
    constitution_status = (settings_row or {}).get("constitution_status")

    if not constitution_url or constitution_status != "valid":
        raise HTTPException(
            status_code=422,
            detail="Cannot start a SPEC session without a valid CONSTITUTION.md file. "
                   "Please link a valid constitution file in the project settings first.",
        )

    # Fetch the actual markdown content — fail if unreachable
    constitution_content = await _fetch_constitution_content(constitution_url)
    if not constitution_content:
        raise HTTPException(
            status_code=422,
            detail="The CONSTITUTION.md file could not be downloaded. "
                   "Please verify the file is accessible and try again.",
        )

    session_id = f"session-{uuid.uuid4().hex}"

    # 1. Insert into ba_sessions
    session_row = await db.create_session(
        session_id=session_id,
        task_id=body.taskId,
        teamwork_task_id=body.teamworkTaskId,
        task_title=body.taskTitle,
        project_name=body.projectName,
        project_id=body.projectId,
        model=body.model,
    )

    # 2. Build initial LangGraph state
    initial_state = {
        "session_id": session_id,
        "task_id": body.taskId,
        "teamwork_task_id": body.teamworkTaskId,
        "task_title": body.taskTitle,
        "task_description": body.taskDescription,
        "project_name": body.projectName,
        "constitution_md": constitution_content,
        "phase": "research",
        "completeness": dict(DEFAULT_COMPLETENESS),
        "decisions": [],
        "spec_md": None,
        "turn_number": 0,
        "turn_output": None,
    }

    trigger_message = HumanMessage(
        content=(
            f"New intake session started.\n"
            f"Project: {body.projectName}\n"
            f"Task: {body.taskTitle}\n"
            f"Description: {body.taskDescription or 'No description provided'}\n\n"
            "Please research this task and ask the first questions to the BA."
        )
    )

    # 3. Run agent
    graph = _graph(request)
    raw_messages, updates = await _run_agent_turn(
        graph, session_id, [trigger_message], initial_state, model_id=body.model
    )

    # 4. Persist agent messages + update session
    messages = await _persist_agent_messages(session_id, raw_messages)
    await db.update_session(
        session_id,
        completeness=updates["completeness"],
        spec_md=updates.get("spec_md"),
    )

    # 5. Return — re-fetch to get fully updated row
    session_row["completeness"] = updates["completeness"]
    session_row["spec_md"] = updates.get("spec_md")
    return SessionWithMessages(
        session=_db_row_to_session(session_row),
        messages=messages,
    )


# ---------------------------------------------------------------------------
# POST /session/message
# ---------------------------------------------------------------------------


@router.post("/session/message", response_model=SessionWithMessages)
async def send_message(body: SendMessageRequest, request: Request):
    session_row = await db.get_session(body.sessionId)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_row.get("status") == "approved":
        raise HTTPException(
            status_code=409,
            detail="Session is approved. Use /session/review to request changes.",
        )

    # 1. Save BA message to DB
    await db.insert_message(
        session_id=body.sessionId,
        role="ba",
        content_type="text",
        content=body.content,
    )

    # 2. Run agent — inject DB-sourced state so build_system_prompt sees
    #    current completeness/spec_md (LangGraph checkpoint doesn't store these).
    graph = _graph(request)
    user_msg = HumanMessage(content=body.content)
    db_completeness = session_row.get("completeness") or DEFAULT_COMPLETENESS
    if isinstance(db_completeness, str):
        db_completeness = json.loads(db_completeness)
    raw_messages, updates = await _run_agent_turn(
        graph,
        body.sessionId,
        [user_msg],
        state_overrides={
            "completeness": dict(db_completeness),
            "spec_md": session_row.get("spec_md"),
        },
        model_id=body.model,
    )

    # 3. Persist agent messages + update session
    messages = await _persist_agent_messages(body.sessionId, raw_messages)

    # Manage status transitions based on completeness scores
    current_status = session_row.get("status", "in_progress")
    new_status: str | None = None
    if all(v >= 80 for v in updates["completeness"].values()):
        # Only promote to spec_ready from in_progress (don't overwrite approved)
        if current_status == "in_progress":
            new_status = "spec_ready"
    else:
        # If scores dropped below threshold, revert spec_ready back to in_progress
        if current_status == "spec_ready":
            new_status = "in_progress"

    await db.update_session(
        body.sessionId,
        completeness=updates["completeness"],
        spec_md=updates.get("spec_md"),
        status=new_status,
    )

    # 4. Re-fetch the updated session row for the response
    updated_row = await db.get_session(body.sessionId)
    return SessionWithMessages(
        session=_db_row_to_session(updated_row),
        messages=messages,
    )


# ---------------------------------------------------------------------------
# POST /session/message/stream  (SSE)
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event block."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _extract_ft_streaming_text(partial_json: str, offset: int) -> tuple[str, int]:
    """Incrementally extract the first text `content` value from a partial finalize_turn JSON.

    The finalize_turn args look like:
      {"messages": [{"content_type": "text", "content": "THE RESPONSE...", ...}], ...}

    We scan for the literal sequence ``"content": "`` and decode the JSON string
    that follows it, returning only the newly available characters (after *offset*).

    Returns:
        (new_text, new_offset)  — new_text is the freshly available decoded text;
        new_offset is the updated position within the content value string.
    """
    marker = '"content": "'
    pos = partial_json.find(marker)
    if pos == -1:
        return "", offset
    content_start = pos + len(marker)
    available = partial_json[content_start + offset:]
    chars: list[str] = []
    i = 0
    while i < len(available):
        c = available[i]
        if c == "\\":
            if i + 1 >= len(available):
                break  # incomplete escape — safe to stop and retry later
            n = available[i + 1]
            if n == "n":
                chars.append("\n"); i += 2
            elif n == "t":
                chars.append("\t"); i += 2
            elif n == "r":
                chars.append("\r"); i += 2
            elif n == '"':
                chars.append('"'); i += 2
            elif n == "\\":
                chars.append("\\"); i += 2
            elif n == "u":
                if i + 5 < len(available):
                    try:
                        chars.append(chr(int(available[i + 2:i + 6], 16)))
                        i += 6
                    except ValueError:
                        break  # bad unicode — stop
                else:
                    break  # incomplete \uXXXX — stop and retry later
            else:
                break  # unknown escape — stop
        elif c == '"':
            break  # end of JSON string — content fully received
        else:
            chars.append(c)
            i += 1
    new_text = "".join(chars)
    new_offset = offset + i
    return new_text, new_offset


@router.post("/session/message/stream")
async def stream_message(body: SendMessageRequest, request: Request):
    """
    SSE streaming variant of POST /session/message.

    Emits real-time step events while the LangGraph agent runs, then a final
    'done' event containing the complete SessionWithMessages payload.

    Event types:
      step  → {"type": "thinking"|"tool_call"|"tool_done", "label": str, ...}
      token → {"text": str}  (LLM text delta, emitted per token)
      done  → {"session": {...}, "messages": [...]}
      error → {"message": str}
    """
    session_row = await db.get_session(body.sessionId)
    if session_row is None:
        async def _not_found():
            yield _sse_event("error", {"message": "Session not found"})
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    if session_row.get("status") == "approved":
        async def _approved():
            yield _sse_event("error", {"message": "Session is approved. Use /session/review to request changes."})
        return StreamingResponse(_approved(), media_type="text/event-stream")

    graph = _graph(request)

    async def event_generator():
        # 1. Persist BA message immediately (same order as send_message)
        await db.insert_message(
            session_id=body.sessionId,
            role="ba",
            content_type="text",
            content=body.content,
        )

        # 2. Build graph input (mirrors send_message)
        user_msg = HumanMessage(content=body.content)
        db_completeness = session_row.get("completeness") or DEFAULT_COMPLETENESS
        if isinstance(db_completeness, str):
            db_completeness = json.loads(db_completeness)

        graph_input: dict = {
            "messages": [user_msg],
            "completeness": dict(db_completeness),
            "spec_md": session_row.get("spec_md"),
        }
        config = _make_config(body.sessionId, body.model or DEFAULT_MODEL)

        # 3. Stream events from the graph — also capture the last AIMessage
        #    for finalize_turn extraction (more reliable than aget_state).
        iteration_counter = 0
        last_ai_message = None  # captured from on_chat_model_end events
        ft_partial_json = ""   # accumulated partial args JSON for finalize_turn
        ft_content_emitted = 0  # chars of content field already emitted as tokens
        try:
            async with _get_semaphore():
                async with asyncio.timeout(settings.agent_timeout_seconds):
                    async for event in graph.astream_events(graph_input, config, version="v2"):
                        kind = event.get("event", "")
                        name = event.get("name", "")
                        data = event.get("data", {})

                        if kind == "on_chat_model_start":
                            iteration_counter += 1
                            label = (
                                "Analyzing your request..."
                                if iteration_counter == 1
                                else "Formulating response..."
                            )
                            yield _sse_event("step", {
                                "type": "thinking",
                                "label": label,
                                "iteration": iteration_counter,
                            })

                        elif kind == "on_chat_model_stream":
                            chunk = data.get("chunk")
                            if chunk is not None:
                                # --- Plain text content (rare for tool-heavy agents) ---
                                chunk_content = getattr(chunk, "content", None)
                                if isinstance(chunk_content, str) and chunk_content:
                                    yield _sse_event("token", {"text": chunk_content})
                                elif isinstance(chunk_content, list):
                                    for block in chunk_content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text_delta = block.get("text", "")
                                            if text_delta:
                                                yield _sse_event("token", {"text": text_delta})
                                        elif isinstance(block, str) and block:
                                            yield _sse_event("token", {"text": block})

                                # --- finalize_turn tool input streaming ---
                                # Claude delivers responses through tool_call_chunks whose
                                # `args` field accumulates the partial JSON for finalize_turn.
                                # We extract the first text message's `content` value as it
                                # streams so the frontend can show a live response preview.
                                for tc in getattr(chunk, "tool_call_chunks", None) or []:
                                    args_delta = tc.get("args") or ""
                                    if not args_delta:
                                        continue
                                    if tc.get("name") == "finalize_turn" or ft_partial_json:
                                        ft_partial_json += args_delta
                                        new_text, ft_content_emitted = _extract_ft_streaming_text(
                                            ft_partial_json, ft_content_emitted
                                        )
                                        if new_text:
                                            yield _sse_event("token", {"text": new_text})

                        elif kind == "on_chat_model_end":
                            # Capture each LLM output — the last one contains
                            # finalize_turn (overwritten each iteration).
                            output = data.get("output")
                            if output is not None:
                                last_ai_message = output

                        elif kind == "on_tool_start" and name == "search_teamwork":
                            args = data.get("input") or {}
                            task_id = args.get("task_id", "") if isinstance(args, dict) else ""
                            label = (
                                f"Searching Teamwork for task {task_id}..."
                                if task_id
                                else "Searching Teamwork for task details..."
                            )
                            yield _sse_event("step", {
                                "type": "tool_call",
                                "tool": "search_teamwork",
                                "label": label,
                            })

                        elif kind == "on_tool_end" and name == "search_teamwork":
                            yield _sse_event("step", {
                                "type": "tool_done",
                                "tool": "search_teamwork",
                                "label": "Found task information",
                            })

        except TimeoutError:
            logger.error(
                "Stream agent timed out after %ds for session %s",
                settings.agent_timeout_seconds, body.sessionId,
            )
            yield _sse_event("error", {
                "message": f"Agent timed out after {settings.agent_timeout_seconds}s. Please try again."
            })
            return
        except Exception as exc:
            logger.exception("Stream error for session %s: %s", body.sessionId, exc)
            yield _sse_event("error", {"message": f"Agent error: {exc}"})
            return

        # 4. Extract finalize_turn from captured AIMessage (primary) or checkpoint (fallback)
        try:
            final_state: dict = {}  # populated from checkpoint if available
            finalize_args = None
            if last_ai_message is not None:
                for tc in (getattr(last_ai_message, "tool_calls", None) or []):
                    if tc.get("name") == "finalize_turn":
                        finalize_args = tc.get("args")
                        break

            # Fallback: try checkpoint state if captured AIMessage didn't have it
            if finalize_args is None:
                try:
                    checkpoint_state = await graph.aget_state(config)
                    final_state = checkpoint_state.values
                    finalize_args = extract_finalize_args(final_state)
                except Exception as chk_err:
                    logger.warning("Could not read checkpoint state: %s", chk_err)
                    final_state = {}

            if finalize_args is None:
                # Last resort: plain text from captured AIMessage or checkpoint
                text = ""
                if last_ai_message is not None:
                    raw_content = getattr(last_ai_message, "content", "")
                    # Anthropic models may return content as a list of blocks
                    if isinstance(raw_content, list):
                        text = " ".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw_content
                        )
                    else:
                        text = str(raw_content) if raw_content else ""
                logger.info("Stream: model did not call finalize_turn — plain text fallback (text length=%d)", len(text))
                finalize_args = {
                    "messages": [{"content_type": "text", "content": text, "citations": []}],
                    "completeness": None,
                    "decisions": [],
                    "spec_md": None,
                }
            else:
                logger.info("Stream finalize_turn args: %s", json.dumps(finalize_args, default=str)[:2000])

            raw_messages = finalize_args.get("messages", [])
            new_completeness = finalize_args.get("completeness")
            spec_md = finalize_args.get("spec_md")

            # Normalise types (same as _run_agent_turn)
            if isinstance(raw_messages, str):
                try:
                    raw_messages = json.loads(raw_messages)
                except (json.JSONDecodeError, TypeError):
                    raw_messages = [{"content_type": "text", "content": raw_messages, "citations": []}]
            if isinstance(spec_md, str):
                if spec_md.startswith(("{", '"')):
                    try:
                        parsed = json.loads(spec_md)
                        if isinstance(parsed, str):
                            spec_md = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                if isinstance(spec_md, str) and "\\n" in spec_md:
                    spec_md = spec_md.replace("\\n", "\n")
            if isinstance(new_completeness, str):
                try:
                    new_completeness = json.loads(new_completeness)
                except (json.JSONDecodeError, TypeError):
                    new_completeness = None

            # final_state may be empty if checkpoint read failed; fall back to DB values
            checkpoint_completeness = (
                final_state.get("completeness") if final_state else None
            ) or db_completeness
            current_completeness = dict(checkpoint_completeness or DEFAULT_COMPLETENESS)
            if isinstance(new_completeness, dict):
                for key in current_completeness:
                    if key in new_completeness:
                        val = new_completeness[key]
                        try:
                            current_completeness[key] = max(0, min(100, int(val)))
                        except (ValueError, TypeError):
                            pass

            if spec_md is None:
                spec_md = (final_state.get("spec_md") if final_state else None) or session_row.get("spec_md")

            # 5. Persist agent messages
            messages = await _persist_agent_messages(body.sessionId, raw_messages)

            # 6. Status transition (same logic as send_message)
            current_status = session_row.get("status", "in_progress")
            new_status: str | None = None
            if all(v >= 80 for v in current_completeness.values()):
                if current_status == "in_progress":
                    new_status = "spec_ready"
            else:
                if current_status == "spec_ready":
                    new_status = "in_progress"

            await db.update_session(
                body.sessionId,
                completeness=current_completeness,
                spec_md=spec_md,
                status=new_status,
            )

            # 7. Emit done event with full payload
            updated_row = await db.get_session(body.sessionId)
            session_obj = _db_row_to_session(updated_row)
            yield _sse_event("done", {
                "session": session_obj.model_dump(),
                "messages": [m.model_dump() for m in messages],
            })

        except Exception as exc:
            logger.exception("Post-stream persistence error for session %s: %s", body.sessionId, exc)
            yield _sse_event("error", {"message": f"Failed to save agent response: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /session/get
# ---------------------------------------------------------------------------


@router.post("/session/get", response_model=SessionWithMessages)
async def get_session(body: GetSessionRequest):
    session_row = await db.get_session(body.sessionId)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    message_rows = await db.get_messages(body.sessionId)
    messages = [_db_row_to_message(r) for r in message_rows]

    return SessionWithMessages(
        session=_db_row_to_session(session_row),
        messages=messages,
    )


# ---------------------------------------------------------------------------
# POST /session/review
# ---------------------------------------------------------------------------


@router.post("/session/review", response_model=ReviewResponse)
async def review_session(body: ReviewRequest, request: Request):
    session_row = await db.get_session(body.sessionId)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if body.action == "approve":
        spec_md = session_row.get("spec_md", "")
        if not spec_md:
            raise HTTPException(status_code=422, detail="Cannot approve: SPEC is empty.")

        # Prepare webhook payload before changing status
        project_id = session_row.get("project_id", "")
        constitution_url = ""
        if project_id:
            ps = await db.get_project_settings(project_id)
            constitution_url = (ps or {}).get("constitution_url", "")
        completeness = session_row.get("completeness", {})
        if isinstance(completeness, str):
            completeness = json.loads(completeness)

        tw_id = session_row.get("teamwork_task_id", "")
        spec_file_name = f"SPEC-{tw_id}.md" if tw_id else "SPEC.md"
        spec_file_size = len(spec_md.encode("utf-8"))

        # Notify n8n BEFORE marking approved — if webhook fails the session
        # stays in its previous state so the user can retry.
        previous_status = session_row.get("status", "spec_ready")
        await db.update_session(body.sessionId, status="approved")
        try:
            await _notify_n8n(settings.n8n_spec_approved_path, {
                "session_id": body.sessionId,
                "teamwork_task_id": tw_id,
                "task_title": session_row.get("teamwork_task_title", ""),
                "spec_md": spec_md,
                "spec_file_name": spec_file_name,
                "spec_file_size": spec_file_size,
                "project_id": project_id,
                "project_name": session_row.get("project_name", ""),
                "constitution_url": constitution_url,
                "approved_by": session_row.get("created_by", ""),
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "completeness": completeness,
            }, webhook_mode=body.webhookMode)
        except HTTPException:
            # Revert approval so the user can retry
            await db.update_session(body.sessionId, status=previous_status)
            raise

        updated_row = await db.get_session(body.sessionId)
        return ReviewResponse(
            status="approved",
            session=_db_row_to_session(updated_row),
        )

    # request_changes — feed feedback back into agent
    if body.feedback:
        graph = _graph(request)
        feedback_msg = HumanMessage(
            content=f"[REVISION REQUESTED] {body.feedback}\nPlease update the SPEC accordingly."
        )
        db_completeness = session_row.get("completeness") or DEFAULT_COMPLETENESS
        if isinstance(db_completeness, str):
            db_completeness = json.loads(db_completeness)
        raw_messages, updates = await _run_agent_turn(
            graph,
            body.sessionId,
            [feedback_msg],
            state_overrides={
                "completeness": dict(db_completeness),
                "spec_md": session_row.get("spec_md"),
            },
            model_id=body.model,
        )
        messages = await _persist_agent_messages(body.sessionId, raw_messages)
        await db.update_session(
            body.sessionId,
            completeness=updates["completeness"],
            spec_md=updates.get("spec_md"),
            status="in_progress",
        )
        updated_row = await db.get_session(body.sessionId)
        return ReviewResponse(
            status="revision_requested",
            session=_db_row_to_session(updated_row),
            messages=messages,
        )

    return ReviewResponse(status="revision_requested")


# ---------------------------------------------------------------------------
# POST /session/reset
# ---------------------------------------------------------------------------


@router.post("/session/reset")
async def reset_session(body: ResetSessionRequest):
    """Delete ALL sessions for a teamwork task so it can be started fresh."""
    session_row = await db.get_session(body.sessionId)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    tw_id = session_row.get("teamwork_task_id", "")
    task_title = session_row.get("teamwork_task_title", "")
    project_id = session_row.get("project_id", "")

    # Gather constitution_url before deleting (needed for n8n webhook)
    constitution_url = ""
    if project_id:
        ps = await db.get_project_settings(project_id)
        constitution_url = (ps or {}).get("constitution_url", "")

    # Find all session IDs for this teamwork task (there may be stale ones)
    all_session_ids = [body.sessionId]
    if tw_id:
        pool = await db.get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT id FROM ba_sessions WHERE teamwork_task_id = %s",
                    (tw_id,),
                )
                all_session_ids = [r["id"] for r in await cur.fetchall()]

    # Notify n8n orchestrator BEFORE deleting — if it fails the session
    # stays intact so the user can retry.
    await _notify_n8n(settings.n8n_spec_reset_path, {
        "teamwork_task_id": tw_id,
        "task_title": task_title,
        "constitution_url": constitution_url,
    }, webhook_mode=body.webhookMode)

    # Best-effort cleanup of LangGraph checkpoint tables for ALL sessions
    pool = await db.get_pool()
    for sid in all_session_ids:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            try:
                async with pool.connection() as conn:
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s",
                        (sid,),
                    )
            except Exception as exc:
                logger.warning("Could not clean checkpoint table %s for %s: %s", table, sid, exc)

    # Delete ALL sessions + messages for this teamwork task
    if tw_id:
        count = await db.delete_sessions_by_teamwork_id(tw_id)
        logger.info("Reset teamwork task %s: deleted %d session(s)", tw_id, count)
    else:
        await db.delete_session(body.sessionId)
        logger.info("Reset session %s (no teamwork_task_id)", body.sessionId)

    return {"status": "reset"}
