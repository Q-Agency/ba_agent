"""
Pydantic models matching the exact API contract the React frontend expects.
Mirrors the TypeScript types in src/types/intake.ts.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared domain types
# ---------------------------------------------------------------------------


class CompletenessMap(BaseModel):
    user_roles: bool = False
    business_rules: bool = False
    acceptance_criteria: bool = False
    scope_boundaries: bool = False
    error_handling: bool = False
    data_model: bool = False


class Session(BaseModel):
    id: str
    task_id: str
    teamwork_task_id: str
    teamwork_task_title: str
    project_name: str
    status: str
    created_at: str
    updated_at: str
    created_by: str
    spec_md: str | None
    completeness: CompletenessMap


class Message(BaseModel):
    id: str
    session_id: str
    role: Literal["agent", "ba", "system"]
    content_type: Literal["text", "question_block", "decision", "spec_preview"]
    content: Any  # str for text/decision, dict for question_block/spec_preview
    created_at: str
    sources: list[dict] | None = None


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    taskId: str
    teamworkTaskId: str = ""
    taskTitle: str
    taskDescription: str = ""
    projectName: str


class SendMessageRequest(BaseModel):
    sessionId: str
    content: str


class GetSessionRequest(BaseModel):
    sessionId: str


class ReviewRequest(BaseModel):
    sessionId: str
    action: Literal["approve", "request_changes"]
    feedback: str | None = None


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------


class SessionWithMessages(BaseModel):
    session: Session
    messages: list[Message]


class MessagesResponse(BaseModel):
    messages: list[Message]


class ReviewResponse(BaseModel):
    status: Literal["approved", "revision_requested"]
