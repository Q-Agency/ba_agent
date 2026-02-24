"""
LangGraph agent state.

`messages` is the internal LLM conversation history managed by LangGraph.
Everything else is session-level metadata persisted alongside it.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CompletenessMap(TypedDict):
    user_roles: int       # 0-100
    business_rules: int
    acceptance_criteria: int
    scope_boundaries: int
    error_handling: int
    data_model: int


class AgentState(TypedDict):
    # LangGraph manages this — new messages are appended each turn
    messages: Annotated[list, add_messages]

    # Set once at session creation, never mutated
    session_id: str
    task_id: str
    teamwork_task_id: str
    task_title: str
    task_description: str
    project_name: str

    # Mutated each turn
    phase: str  # "research" | "questioning" | "review"
    completeness: CompletenessMap
    decisions: list[dict]
    spec_md: str | None
    turn_number: int

    # Set by finalize_turn tool call; read by the router after graph.ainvoke
    # Contains the frontend-facing messages for this turn
    turn_output: list[dict] | None


DEFAULT_COMPLETENESS: CompletenessMap = {
    "user_roles": 0,
    "business_rules": 0,
    "acceptance_criteria": 0,
    "scope_boundaries": 0,
    "error_handling": 0,
    "data_model": 0,
}
