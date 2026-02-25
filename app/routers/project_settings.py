"""
Project-level settings endpoints.

POST /project-settings/get    -> fetch settings for a project
POST /project-settings/save   -> upsert constitution URL for a project
"""
from __future__ import annotations

import logging
import re

import httpx
from fastapi import APIRouter, HTTPException

from app import database as db
from app.schemas.api import (
    GetProjectSettingsRequest,
    SaveProjectSettingsRequest,
    ProjectSettings,
    ProjectSettingsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Matches https://github.com/{owner}/{repo}/blob/{branch}/{path}
_GITHUB_BLOB_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
)


def _clean_url(url: str) -> str:
    """Strip whitespace and trailing slashes."""
    return url.strip().rstrip("/")


def _validate_github_url(url: str) -> None:
    if not _GITHUB_BLOB_RE.match(url):
        logger.warning("GitHub URL validation failed for: %r", url)
        raise HTTPException(
            status_code=422,
            detail=f"URL must be a GitHub file URL (https://github.com/owner/repo/blob/branch/path). Got: {url}",
        )


def _blob_to_raw_url(blob_url: str) -> str | None:
    """Convert a GitHub blob URL to a raw.githubusercontent.com URL."""
    m = _GITHUB_BLOB_RE.match(blob_url)
    if not m:
        return None
    return f"https://raw.githubusercontent.com/{m['owner']}/{m['repo']}/{m['branch']}/{m['path']}"


async def _check_github_file(blob_url: str) -> str:
    """HEAD-check the raw file on GitHub. Returns 'valid' or 'invalid'."""
    raw_url = _blob_to_raw_url(blob_url)
    if not raw_url:
        return "invalid"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.head(raw_url, follow_redirects=True)
            logger.info("GitHub file check %s → %s", raw_url, r.status_code)
            return "valid" if r.status_code == 200 else "invalid"
    except Exception:
        logger.warning("Failed to reach %s", raw_url, exc_info=True)
        return "invalid"


def _row_to_settings(row: dict) -> ProjectSettings:
    return ProjectSettings(
        teamwork_project_id=row["teamwork_project_id"],
        constitution_url=row.get("constitution_url"),
        constitution_status=row.get("constitution_status"),
        created_at=str(row["created_at"]) if row.get("created_at") else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
    )


@router.post("/project-settings/get", response_model=ProjectSettingsResponse)
async def get_project_settings(body: GetProjectSettingsRequest):
    row = await db.get_project_settings(body.projectId)
    if row is None:
        return ProjectSettingsResponse(
            settings=ProjectSettings(teamwork_project_id=body.projectId)
        )
    return ProjectSettingsResponse(settings=_row_to_settings(row))


@router.post("/project-settings/save", response_model=ProjectSettingsResponse)
async def save_project_settings(body: SaveProjectSettingsRequest):
    url = _clean_url(body.constitutionUrl) if body.constitutionUrl else None
    logger.info("save_project_settings project=%s url=%r", body.projectId, url)

    if url:
        _validate_github_url(url)
        status = await _check_github_file(url)
    else:
        status = None

    row = await db.upsert_project_settings(
        body.projectId,
        constitution_url=url,
        constitution_status=status,
    )
    return ProjectSettingsResponse(settings=_row_to_settings(row))
