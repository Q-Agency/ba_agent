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
from app import database as db

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
                            "user_roles": 0,
                            "business_rules": 0,
                            "acceptance_criteria": 0,
                            "scope_boundaries": 0,
                            "error_handling": 0,
                            "data_model": 0,
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


async def _fetch_teamwork_projects() -> list[dict]:
    connected_ids = await db.get_connected_project_ids()
    if not connected_ids:
        return []

    async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=20) as client:
        projects_raw = []
        for pid in connected_ids:
            r = await client.get(f"{_base_url()}/projects/{pid}.json")
            if r.is_success:
                p = r.json().get("project", {})
                if p:
                    projects_raw.append(p)

        projects = []
        for p in projects_raw:
            pid = p.get("id")
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
                            "user_roles": 0,
                            "business_rules": 0,
                            "acceptance_criteria": 0,
                            "scope_boundaries": 0,
                            "error_handling": 0,
                            "data_model": 0,
                        },
                    }
                    for t in (tr.json().get("tasks", []) if tr.is_success else [])
                ]
                task_lists.append({"id": str(tl_id), "name": tl.get("name"), "tasks": tasks})

            projects.append(
                {"id": str(pid), "teamwork_project_id": str(pid), "name": p.get("name"), "description": p.get("description"), "task_lists": task_lists}
            )

        # Enrich tasks with persisted session data from ba_sessions
        all_tw_ids = [
            t["teamwork_task_id"]
            for p in projects
            for tl in p["task_lists"]
            for t in tl["tasks"]
        ]
        if all_tw_ids:
            sessions = await db.get_sessions_by_teamwork_ids(all_tw_ids)
            status_map = {"in_progress": "has_session", "spec_ready": "spec_ready", "approved": "approved"}
            for p in projects:
                for tl in p["task_lists"]:
                    for t in tl["tasks"]:
                        sess = sessions.get(t["teamwork_task_id"])
                        if sess:
                            t["session_id"] = sess["id"]
                            t["status"] = status_map.get(sess["status"], "has_session")
                            t["spec_md"] = sess.get("spec_md")
                            t["completeness"] = sess.get("completeness", t["completeness"])
                            if sess.get("updated_at"):
                                t["updated_at"] = sess["updated_at"]

        # Enrich projects with constitution settings
        all_settings = await db.get_all_project_settings()
        for p in projects:
            ps = all_settings.get(p["id"])
            p["constitution_url"] = ps.get("constitution_url") if ps else None
            p["constitution_status"] = ps.get("constitution_status") if ps else None

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


@router.post("/teamwork/connect")
async def connect_project(body: ProjectRequest) -> dict:
    """Connect a Teamwork project: persist to DB and return its full tree."""
    if not settings.teamwork_api_key:
        project = MOCK_PROJECTS[0]
        await db.connect_project(project["id"], name=project["name"])
        return {"project": project}

    try:
        async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=20) as client:
            # Fetch project metadata from Teamwork
            r = await client.get(f"{_base_url()}/projects/{body.projectId}.json")
            r.raise_for_status()
            p = r.json().get("project", {})
            if not p:
                raise HTTPException(status_code=404, detail="Project not found in Teamwork")

            pid = str(p.get("id"))
            project_name = p.get("name", "")

            # Persist as connected project
            await db.connect_project(pid, name=project_name)

            # Fetch full task tree (same logic as _fetch_teamwork_projects for a single project)
            tlr = await client.get(f"{_base_url()}/projects/{pid}/tasklists.json")
            task_lists = []
            for tl in (tlr.json().get("tasklists", []) if tlr.is_success else []):
                tl_id = tl.get("id")
                tr = await client.get(
                    f"{_base_url()}/tasklists/{tl_id}/tasks.json",
                    params={"pageSize": 100},
                )
                tasks = [
                    {
                        "id": str(t.get("id")),
                        "teamwork_task_id": str(t.get("id")),
                        "title": t.get("name"),
                        "description": t.get("description"),
                        "status": "new",
                        "session_id": None,
                        "spec_md": None,
                        "completeness": {
                            "user_roles": 0,
                            "business_rules": 0,
                            "acceptance_criteria": 0,
                            "scope_boundaries": 0,
                            "error_handling": 0,
                            "data_model": 0,
                        },
                    }
                    for t in (tr.json().get("tasks", []) if tr.is_success else [])
                ]
                task_lists.append({"id": str(tl_id), "name": tl.get("name"), "tasks": tasks})

            project = {
                "id": pid,
                "teamwork_project_id": pid,
                "name": project_name,
                "description": p.get("description", ""),
                "task_lists": task_lists,
            }

            # Enrich with session data
            all_tw_ids = [
                t["teamwork_task_id"]
                for tl in project["task_lists"]
                for t in tl["tasks"]
            ]
            if all_tw_ids:
                sessions = await db.get_sessions_by_teamwork_ids(all_tw_ids)
                status_map = {"in_progress": "has_session", "spec_ready": "spec_ready", "approved": "approved"}
                for tl in project["task_lists"]:
                    for t in tl["tasks"]:
                        sess = sessions.get(t["teamwork_task_id"])
                        if sess:
                            t["session_id"] = sess["id"]
                            t["status"] = status_map.get(sess["status"], "has_session")
                            t["spec_md"] = sess.get("spec_md")
                            t["completeness"] = sess.get("completeness", t["completeness"])
                            if sess.get("updated_at"):
                                t["updated_at"] = sess["updated_at"]

            # Enrich with constitution settings
            ps = await db.get_project_settings(pid)
            project["constitution_url"] = ps.get("constitution_url") if ps else None
            project["constitution_status"] = ps.get("constitution_status") if ps else None

            return {"project": project}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Teamwork connect failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Teamwork API error: {exc}")


@router.post("/teamwork/disconnect")
async def disconnect_project(body: ProjectRequest) -> dict:
    """Disconnect a Teamwork project: remove from DB."""
    deleted = await db.disconnect_project(body.projectId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not connected")
    return {"ok": True}
