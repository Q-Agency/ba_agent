"""
Pydantic models matching the exact API contract the React frontend expects.
Mirrors the TypeScript types in src/types/intake.ts.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


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
    model: str = "claude-sonnet-4-6"


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
    taskId: str = Field(..., max_length=200)
    teamworkTaskId: str = Field(default="", max_length=200)
    taskTitle: str = Field(..., max_length=500)
    taskDescription: str = Field(default="", max_length=10000)
    projectName: str = Field(..., max_length=200)
    projectId: str = Field(default="", max_length=200)
    model: str = Field(default="claude-sonnet-4-6", max_length=100)


class SendMessageRequest(BaseModel):
    sessionId: str = Field(..., max_length=200)
    content: str = Field(..., max_length=50000)
    model: str = Field(default="claude-sonnet-4-6", max_length=100)


class GetSessionRequest(BaseModel):
    sessionId: str = Field(..., max_length=200)


class ReviewRequest(BaseModel):
    sessionId: str = Field(..., max_length=200)
    action: Literal["approve", "request_changes"]
    feedback: str | None = Field(default=None, max_length=50000)
    model: str = Field(default="claude-sonnet-4-6", max_length=100)
    webhookMode: Literal["test", "prod"] = "test"


class ResetSessionRequest(BaseModel):
    sessionId: str = Field(..., max_length=200)
    webhookMode: Literal["test", "prod"] = "test"


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


# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------


class ProjectSettings(BaseModel):
    teamwork_project_id: str
    constitution_url: str | None = None
    constitution_status: str | None = None  # "valid" | "invalid" | None
    created_at: str | None = None
    updated_at: str | None = None


class GetProjectSettingsRequest(BaseModel):
    projectId: str


class SaveProjectSettingsRequest(BaseModel):
    projectId: str
    constitutionUrl: str | None = None


class ProjectSettingsResponse(BaseModel):
    settings: ProjectSettings
