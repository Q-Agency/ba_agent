"""
Agent tools.

search_teamwork fetches task context from Teamwork.
finalize_turn is a special "output" tool — when the agent calls it, the graph
ends and the router extracts the structured response from the tool call args.
It is never executed; the graph routing detects it and terminates.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Literal

import httpx
from langchain_core.tools import tool

from app.config import settings


def _teamwork_headers() -> dict:
    """Teamwork API keys use Basic auth: base64(api_key + ':x')."""
    encoded = base64.b64encode(f"{settings.teamwork_api_key}:x".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


# ---------------------------------------------------------------------------
# External search tools
# ---------------------------------------------------------------------------


@tool
async def search_teamwork(task_id: str = "", query: str = "") -> str:
    """
    Fetch task details, description, comments and attachments from Teamwork.
    Use this at the start of a session to understand the task context.

    Args:
        task_id: Teamwork task ID (e.g. "12345")
        query: Optional keyword to filter comments/attachments
    """
    if not settings.teamwork_api_key or not settings.teamwork_domain:
        return json.dumps({
            "error": "Teamwork not configured",
            "hint": "Set TEAMWORK_API_KEY and TEAMWORK_DOMAIN in .env",
        })

    base = f"https://{settings.teamwork_domain}/projects/api/v3"

    async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=15) as client:
        try:
            # Fetch task
            r = await client.get(f"{base}/tasks/{task_id}.json")
            r.raise_for_status()
            task = r.json().get("task", {})

            # Fetch comments
            cr = await client.get(f"{base}/tasks/{task_id}/comments.json")
            comments = cr.json().get("comments", []) if cr.is_success else []

            return json.dumps({
                "id": task.get("id"),
                "title": task.get("name"),
                "description": task.get("description"),
                "status": task.get("status"),
                "comments": [
                    {"author": c.get("author", {}).get("firstName"), "body": c.get("body")}
                    for c in comments[:10]
                ],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# finalize_turn — output signal tool (never executed, just detected by router)
# ---------------------------------------------------------------------------


@tool
def finalize_turn(
    messages: list[dict],
    completeness: dict[str, int] | None = None,
    decisions: list[dict] | None = None,
    spec_md: str | None = None,
) -> str:
    """
    ALWAYS call this tool last to deliver your response to the BA.
    This ends your turn and sends the messages to the frontend.

    Args:
        messages: List of message objects to send. Each has:
            - content_type: "text" | "question_block" | "decision" | "spec_preview"
            - content: string (for text/decision) or structured dict (for question_block)
            - citations: optional list of {source, document, snippet, url}

            For question_block, content must be:
            {
              "category": "User Roles",
              "questions": [
                {
                  "id": "q1",
                  "text": "Who are the primary users?",
                  "type": "choice",
                  "options": ["Option A", "Option B"],
                  "answer": null
                }
              ]
            }

            Question types:
            - "choice": single-select (pick exactly one option)
            - "multi_choice": multi-select (pick one or more options). Use this when the BA should select multiple items from a list (e.g. "Which fields do you need?").
            - "freetext": open-ended text input
            - "data_model_table": table editor for entity fields. Requires a "dataModelTable" property on the question with:
              {
                "entityName": "House",
                "typeOptions": ["String", "Text", "Integer", "Decimal", "Boolean", "Date", "DateTime", "UUID", "JSON", "Enum"],
                "fields": [{"name": "Address", "type": "", "required": false, "description": ""}],
                "allowAddFields": true
              }
              Use this instead of individual per-field choice questions for data model fields.

            For decision, content must be:
            {
              "decision": "The system will use email-based login",
              "source": "BA response, Round 1",
              "round": 1
            }

        completeness: Full completeness map with integer scores 0-100 for ALL 6 dimensions.
            Score reflects the quality of that section in the current spec_md.
            ALWAYS provide all 6 keys. Scores CAN go down if spec content is weakened.
            e.g. {"user_roles": 85, "business_rules": 60, "acceptance_criteria": 30,
                  "scope_boundaries": 70, "error_handling": 0, "data_model": 45}

        decisions: List of decisions confirmed this turn:
            [{"decision": "...", "source": "...", "round": 1}]

        spec_md: Current SPEC.md draft. ALWAYS provide this, even as a partial skeleton
            with [Pending] placeholders on early turns. This is the single source of truth.
    """
    # This function body is never called — the graph detects finalize_turn
    # in the tool calls and exits before execution.
    return "finalize_turn detected"


# All tools the agent can use
AGENT_TOOLS = [
    search_teamwork,
    finalize_turn,
]
