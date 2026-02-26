"""
GitHub service — create PRs with SPEC.md files via the GitHub REST API.

Uses the Contents API + Git References API to:
1. Get the SHA of the base branch
2. Create a feature branch
3. Create (or update) a file on the feature branch
4. Open a Pull Request

Handles re-approval gracefully: if the branch/file/PR already exist,
updates the file in-place and returns the existing PR URL.
"""
from __future__ import annotations

import base64
import logging
import re
import unicodedata

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_GITHUB_BLOB_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
)


def _github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def parse_github_url(blob_url: str) -> dict[str, str] | None:
    """Extract owner, repo, branch from a GitHub blob URL.

    Returns {"owner": ..., "repo": ..., "branch": ...} or None.
    """
    m = _GITHUB_BLOB_RE.match(blob_url)
    if not m:
        return None
    return {
        "owner": m.group("owner"),
        "repo": m.group("repo"),
        "branch": m.group("branch"),
    }


def slugify(text: str, max_length: int = 60) -> str:
    """Convert a task title to a URL/branch-safe slug.

    Examples:
        "User Export Feature" -> "user-export-feature"
        "Add SSO (SAML 2.0) Support!" -> "add-sso-saml-2-0-support"
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "spec"


def build_paths(teamwork_task_id: str, task_title: str) -> dict[str, str]:
    """Derive branch name + file path from task metadata.

    Returns:
        {
            "slug": "4521-user-export-feature",
            "branch": "spec/4521-user-export-feature",
            "file_path": "app/4521-user-export-feature/SPEC.md",
        }
    """
    title_slug = slugify(task_title)
    slug = f"{teamwork_task_id}-{title_slug}"
    return {
        "slug": slug,
        "branch": f"spec/{slug}",
        "file_path": f"app/{slug}/SPEC.md",
    }


async def create_spec_pr(
    *,
    owner: str,
    repo: str,
    base_branch: str,
    teamwork_task_id: str,
    task_title: str,
    spec_md: str,
    teamwork_domain: str = "",
) -> dict[str, str]:
    """Create a GitHub PR containing SPEC.md.

    Returns {"pr_url": "https://github.com/...", "branch": "spec/..."}.
    Raises on any GitHub API error.
    """
    paths = build_paths(teamwork_task_id, task_title)
    branch_name = paths["branch"]
    file_path = paths["file_path"]
    headers = _github_headers()
    api_base = f"https://api.github.com/repos/{owner}/{repo}"

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        # 1. Get SHA of base branch
        r = await client.get(f"{api_base}/git/ref/heads/{base_branch}")
        r.raise_for_status()
        base_sha = r.json()["object"]["sha"]
        logger.info("Base branch %s SHA: %s", base_branch, base_sha[:8])

        # 2. Create feature branch (or reuse existing)
        existing_branch = await client.get(f"{api_base}/git/ref/heads/{branch_name}")
        if existing_branch.status_code == 200:
            logger.info("Branch %s already exists — will update file", branch_name)
        else:
            r = await client.post(
                f"{api_base}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            )
            if r.status_code != 201:
                logger.error("GitHub git/refs POST failed (%s): %s", r.status_code, r.text)
            r.raise_for_status()
            logger.info("Created branch %s from %s", branch_name, base_sha[:8])

        # 3. Create or update SPEC.md file
        file_sha: str | None = None
        existing_file = await client.get(
            f"{api_base}/contents/{file_path}",
            params={"ref": branch_name},
        )
        if existing_file.status_code == 200:
            file_sha = existing_file.json().get("sha")
            logger.info("File %s exists (sha=%s) — updating", file_path, (file_sha or "?")[:8])

        content_b64 = base64.b64encode(spec_md.encode("utf-8")).decode("ascii")
        put_body: dict = {
            "message": f"feat(spec): add SPEC.md for {task_title}",
            "content": content_b64,
            "branch": branch_name,
        }
        if file_sha:
            put_body["sha"] = file_sha

        r = await client.put(f"{api_base}/contents/{file_path}", json=put_body)
        r.raise_for_status()
        logger.info("Committed SPEC.md to %s on branch %s", file_path, branch_name)

        # 4. Create PR (or find existing open PR)
        existing_prs = await client.get(
            f"{api_base}/pulls",
            params={"head": f"{owner}:{branch_name}", "base": base_branch, "state": "open"},
        )
        if existing_prs.status_code == 200 and existing_prs.json():
            pr = existing_prs.json()[0]
            pr_url = pr["html_url"]
            logger.info("PR already exists: %s — file was updated", pr_url)
            return {"pr_url": pr_url, "branch": branch_name}

        tw_link = ""
        if teamwork_domain and teamwork_task_id:
            tw_link = f"\n\nTeamwork task: https://{teamwork_domain}/app/tasks/{teamwork_task_id}"

        pr_body = (
            f"Auto-generated SPEC for Teamwork task **{task_title}** "
            f"(#{teamwork_task_id}).{tw_link}\n\n"
            f"_Created by BA Agent_"
        )

        r = await client.post(
            f"{api_base}/pulls",
            json={
                "title": f"SPEC: {task_title}",
                "body": pr_body,
                "head": branch_name,
                "base": base_branch,
            },
        )
        r.raise_for_status()
        pr_url = r.json()["html_url"]
        logger.info("Created PR: %s", pr_url)

        return {"pr_url": pr_url, "branch": branch_name}
