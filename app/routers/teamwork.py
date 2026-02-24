"""
Teamwork project/task endpoints.

These mirror what the n8n teamwork workflows do.
For now they proxy directly to the Teamwork API.
If TEAMWORK_API_KEY is not set, they return mock data for local dev.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

MOCK_PROJECTS = [
    {
        "id": "mock-project-1",
        "name": "Mock Project Alpha",
        "description": "Local dev mock project",
        "task_lists": [
            {
                "id": "tl-1",
                "name": "Backlog",
                "tasks": [
                    {
                        "id": "task-1",
                        "teamwork_task_id": "TW-001",
                        "title": "Mock Task: User Authentication",
                        "description": "Implement email/password login with JWT",
                        "status": "new",
                        "session_id": None,
                        "spec_md": None,
                        "completeness": {
                            "user_roles": False,
                            "business_rules": False,
                            "acceptance_criteria": False,
                            "scope_boundaries": False,
                            "error_handling": False,
                            "data_model": False,
                        },
                    }
                ],
            }
        ],
    }
]


class ProjectsRequest(BaseModel):
    pass


class ProjectRequest(BaseModel):
    projectId: str


def _teamwork_headers() -> dict:
    # Teamwork API keys use Basic auth: base64(api_key + ":x")
    encoded = base64.b64encode(f"{settings.teamwork_api_key}:x".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _base_url() -> str:
    return f"https://{settings.teamwork_domain}/projects/api/v3"


ALLOWED_PROJECT_IDS = ["820301", "822891"]


async def _fetch_teamwork_projects() -> list[dict]:
    async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=20) as client:
        projects_raw = []
        for pid in ALLOWED_PROJECT_IDS:
            r = await client.get(f"{_base_url()}/projects/{pid}.json")
            if r.is_success:
                p = r.json().get("project", {})
                if p:
                    projects_raw.append(p)

        projects = []
        for p in projects_raw:
            pid = p.get("id")
            if str(pid) not in ALLOWED_PROJECT_IDS:
                continue
            # Fetch task lists for each project
            tlr = await client.get(f"{_base_url()}/projects/{pid}/tasklists.json")
            task_lists = []
            for tl in (tlr.json().get("tasklists", []) if tlr.is_success else []):
                tl_id = tl.get("id")
                tr = await client.get(
                    f"{_base_url()}/tasklists/{tl_id}/tasks.json",
                    params={"pageSize": 100},
                )
                logger.info("Tasks response for tasklist %s: status=%s keys=%s", tl_id, tr.status_code, list(tr.json().keys()) if tr.is_success else "FAILED")
                if tr.is_success:
                    logger.info("Task count: %s", len(tr.json().get("tasks", tr.json().get("todo-items", []))))
                tasks = [
                    {
                        "id": str(t.get("id")),
                        "teamwork_task_id": str(t.get("id")),
                        "title": t.get("name"),
                        "description": t.get("description"),
                        "status": "new",  # TODO: map from Teamwork status
                        "session_id": None,
                        "spec_md": None,
                        "completeness": {
                            "user_roles": False,
                            "business_rules": False,
                            "acceptance_criteria": False,
                            "scope_boundaries": False,
                            "error_handling": False,
                            "data_model": False,
                        },
                    }
                    for t in (tr.json().get("tasks", []) if tr.is_success else [])
                ]
                task_lists.append({"id": str(tl_id), "name": tl.get("name"), "tasks": tasks})

            projects.append(
                {"id": str(pid), "name": p.get("name"), "description": p.get("description"), "task_lists": task_lists}
            )
        return projects


@router.post("/teamwork/projects")
async def get_projects(_body: ProjectsRequest = ProjectsRequest()) -> dict:
    if not settings.teamwork_api_key:
        logger.debug("Teamwork not configured — returning mock projects")
        return {"projects": MOCK_PROJECTS}

    try:
        projects = await _fetch_teamwork_projects()
        return {"projects": projects}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("Teamwork fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Teamwork API error: {exc}")


@router.post("/teamwork/project")
async def get_project(body: ProjectRequest) -> dict:
    if not settings.teamwork_api_key:
        for p in MOCK_PROJECTS:
            if p["id"] == body.projectId:
                return {"project": p}
        return {"project": MOCK_PROJECTS[0]}

    try:
        async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=20) as client:
            r = await client.get(f"{_base_url()}/projects/{body.projectId}.json")
            r.raise_for_status()
            return {"project": r.json().get("project", {})}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        logger.exception("Teamwork project fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Teamwork API error: {exc}")
