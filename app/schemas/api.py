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
    user_roles: int = 0
    business_rules: int = 0
    acceptance_criteria: int = 0
    scope_boundaries: int = 0
    error_handling: int = 0
    data_model: int = 0


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
    model: str = "claude-sonnet-4-6"


class SendMessageRequest(BaseModel):
    sessionId: str
    content: str
    model: str = "claude-sonnet-4-6"


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
    session: Session | None = None
    messages: list[Message] | None = None


class ModelInfo(BaseModel):
    id: str
    label: str
    provider: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
