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

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from app import database as db
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
    session_id = f"session-{uuid.uuid4().hex}"

    # 1. Insert into ba_sessions
    session_row = await db.create_session(
        session_id=session_id,
        task_id=body.taskId,
        teamwork_task_id=body.teamworkTaskId,
        task_title=body.taskTitle,
        project_name=body.projectName,
    )

    # 2. Build initial LangGraph state
    initial_state = {
        "session_id": session_id,
        "task_id": body.taskId,
        "teamwork_task_id": body.teamworkTaskId,
        "task_title": body.taskTitle,
        "task_description": body.taskDescription,
        "project_name": body.projectName,
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

    # Only mark spec_ready when all dimensions score >= 80
    new_status: str | None = None
    if all(v >= 80 for v in updates["completeness"].values()):
        new_status = "spec_ready"

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
        # TODO: trigger git push workflow here
        return ReviewResponse(status="approved")

    # request_changes — feed feedback back into agent
    if body.feedback:
        graph = _graph(request)
        feedback_msg = HumanMessage(
            content=f"[REVISION REQUESTED] {body.feedback}\nPlease update the SPEC accordingly."
        )
        raw_messages, updates = await _run_agent_turn(
            graph, body.sessionId, [feedback_msg]
        )
        await _persist_agent_messages(body.sessionId, raw_messages)
        await db.update_session(
            body.sessionId,
            completeness=updates["completeness"],
            spec_md=updates.get("spec_md"),
            status="in_progress",
        )

    return ReviewResponse(status="revision_requested")
