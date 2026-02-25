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


ALLOWED_PROJECT_IDS = ["820301", "822891"]

# Workflow & stage IDs for the shared "test" workflow
WORKFLOW_ID = 12487
STAGE_READY_FOR_DESIGN = 64704


def _base_url_v1() -> str:
    return f"https://{settings.teamwork_domain}"


async def upload_spec_to_task(task_id: str, spec_md: str, task_title: str) -> str:
    """Upload SPEC.md as a file attachment to a Teamwork task.

    Uses the presigned-URL flow:
      1. GET presigned URL + ref from Teamwork
      2. PUT file bytes to S3
      3. PUT task update with pendingFileAttachments ref

    Returns the file ref on success, or raises on failure.
    """
    file_bytes = spec_md.encode("utf-8")
    file_name = f"SPEC-{task_title[:40].replace(' ', '_')}.md"
    file_size = len(file_bytes)

    tw_headers = _teamwork_headers()

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Get presigned URL (needs Teamwork auth)
        r = await client.get(
            f"{_base_url_v1()}/projects/api/v1/pendingfiles/presignedurl.json",
            params={"fileName": file_name, "fileSize": file_size},
            headers=tw_headers,
        )
        r.raise_for_status()
        data = r.json()
        ref = data["ref"]
        upload_url = data["url"]
        logger.info("Got presigned URL for %s (ref=%s)", file_name, ref)

        # 2. Upload to S3 (NO Teamwork auth — presigned URL has its own)
        s3_resp = await client.put(
            upload_url,
            content=file_bytes,
            headers={
                "Content-Length": str(file_size),
                "Content-Type": "text/markdown",
                "x-amz-acl": "public-read",
            },
        )
        s3_resp.raise_for_status()
        logger.info("Uploaded %s to S3 (%d bytes)", file_name, file_size)

        # 3. Attach to task via V1 endpoint (needs Teamwork auth)
        attach_resp = await client.put(
            f"{_base_url_v1()}/tasks/{task_id}.json",
            json={"todo-item": {"pendingFileAttachments": ref}},
            headers=tw_headers,
        )
        attach_resp.raise_for_status()
        logger.info("Attached %s to task %s", ref, task_id)

    return ref


async def move_task_to_stage(task_id: str, stage_id: int = STAGE_READY_FOR_DESIGN) -> None:
    """Move a task to a workflow stage (board column) in Teamwork.

    Uses V3 Workflows API:
      POST /workflows/{workflowId}/stages/{stageId}/tasks.json
    """
    async with httpx.AsyncClient(headers=_teamwork_headers(), timeout=20) as client:
        r = await client.post(
            f"{_base_url()}/workflows/{WORKFLOW_ID}/stages/{stage_id}/tasks.json",
            json={"taskIds": [int(task_id)]},
        )
        r.raise_for_status()
        logger.info("Moved task %s to stage %s (workflow %s)", task_id, stage_id, WORKFLOW_ID)


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
