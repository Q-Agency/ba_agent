"""
Session endpoints — the core of the BA Agent API.

POST /session/start    → create session, run first agent turn, return initial messages
POST /session/message  → continue conversation, return new agent messages
POST /session/get      → fetch session + full message history from DB
POST /session/review   → approve or request changes on a generated SPEC
GET  /models           → list available LLM models
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from psycopg.rows import dict_row

from app import database as db
from app.routers.project_settings import _blob_to_raw_url
from app.graph.graph import build_graph, extract_finalize_args
from app.graph.nodes import MODEL_REGISTRY, DEFAULT_MODEL
from app.graph.state import DEFAULT_COMPLETENESS, AgentState
from app.routers.teamwork import upload_spec_to_task, move_task_to_stage
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


def _make_config(session_id: str, model_id: str = DEFAULT_MODEL) -> dict:
    """LangGraph config for a session — thread_id drives checkpointing."""
    return {"configurable": {"thread_id": session_id, "model_id": model_id}}


async def _fetch_constitution_content(blob_url: str) -> str | None:
    """Download the raw markdown content of a constitution file from GitHub."""
    raw_url = _blob_to_raw_url(blob_url)
    if not raw_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(raw_url, follow_redirects=True)
            if r.status_code == 200:
                return r.text
            logger.warning("Constitution fetch %s → %s", raw_url, r.status_code)
            return None
    except Exception:
        logger.warning("Failed to fetch constitution from %s", raw_url, exc_info=True)
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
    model_id: str = DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    """
    Run one agent turn. Returns (frontend_messages, updates_dict).

    - initial_state: provided only on session start (sets session metadata)
    - subsequent turns just pass new messages; LangGraph resumes from checkpoint
    """
    config = _make_config(session_id, model_id)

    graph_input: dict = {"messages": input_messages}
    if initial_state:
        graph_input.update(initial_state)

    final_state: AgentState = await graph.ainvoke(graph_input, config)

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
    if isinstance(spec_md, str) and spec_md.startswith("{"):
        # spec_md should be plain markdown, not JSON — leave it as-is unless it's a JSON wrapper
        try:
            parsed = json.loads(spec_md)
            if isinstance(parsed, str):
                spec_md = parsed
        except (json.JSONDecodeError, TypeError):
            pass
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

    # 2. Run agent (LangGraph resumes from checkpoint automatically)
    graph = _graph(request)
    user_msg = HumanMessage(content=body.content)
    raw_messages, updates = await _run_agent_turn(
        graph, body.sessionId, [user_msg], model_id=body.model
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
        await db.update_session(body.sessionId, status="approved")

        # Push approved SPEC to Teamwork (upload file + move column)
        tw_task_id = session_row.get("teamwork_task_id", "")
        spec_md = session_row.get("spec_md", "")
        task_title = session_row.get("teamwork_task_title", "spec")

        if tw_task_id and spec_md:
            try:
                await upload_spec_to_task(tw_task_id, spec_md, task_title)
                logger.info("Uploaded SPEC to Teamwork task %s", tw_task_id)
            except Exception as exc:
                logger.error("Failed to upload SPEC to Teamwork task %s: %s", tw_task_id, exc)

            try:
                await move_task_to_stage(tw_task_id)
                logger.info("Moved Teamwork task %s to Ready for Design", tw_task_id)
            except Exception as exc:
                logger.error("Failed to move Teamwork task %s to Ready for Design: %s", tw_task_id, exc)

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
        raw_messages, updates = await _run_agent_turn(
            graph, body.sessionId, [feedback_msg], model_id=body.model
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
