"""
GitHub URL parsing utility.

Used to extract owner/repo/branch from GitHub blob URLs
(e.g., the constitution_url in project settings).
"""
from __future__ import annotations

import re

_GITHUB_BLOB_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
)


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
