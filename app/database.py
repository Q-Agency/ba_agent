"""
Direct PostgreSQL access for ba_sessions and ba_messages tables.
LangGraph uses its own checkpoint tables (managed by AsyncPostgresSaver).
This module handles the application-level tables the frontend reads.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        await _pool.open()
        # Auto-migrate: add model column if missing
        try:
            async with _pool.connection() as conn:
                await conn.execute(
                    "ALTER TABLE ba_sessions ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'claude-sonnet-4-6'"
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ba_project_settings (
                        teamwork_project_id TEXT PRIMARY KEY,
                        constitution_url    TEXT,
                        constitution_status TEXT DEFAULT NULL,
                        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                await conn.execute(
                    "ALTER TABLE ba_project_settings ADD COLUMN IF NOT EXISTS constitution_status TEXT DEFAULT NULL"
                )
        except Exception:
            pass  # column/table already exists or table not yet created
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# ba_sessions CRUD
# ---------------------------------------------------------------------------

DEFAULT_COMPLETENESS = {
    "user_roles": 0,
    "business_rules": 0,
    "acceptance_criteria": 0,
    "scope_boundaries": 0,
    "error_handling": 0,
    "data_model": 0,
}


def normalize_completeness(raw: dict) -> dict:
    """Convert legacy boolean completeness to integer scores (0-100)."""
    result = {}
    for key in DEFAULT_COMPLETENESS:
        val = raw.get(key, 0)
        if isinstance(val, bool):
            result[key] = 100 if val else 0
        elif isinstance(val, (int, float)):
            result[key] = max(0, min(100, int(val)))
        else:
            result[key] = 0
    return result


async def create_session(
    *,
    session_id: str,
    task_id: str,
    teamwork_task_id: str,
    task_title: str,
    project_name: str,
    model: str = "claude-sonnet-4-6",
) -> dict:
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO ba_sessions
                (id, task_id, teamwork_task_id, teamwork_task_title,
                 project_name, status, created_at, updated_at,
                 created_by, spec_md, completeness, model)
            VALUES (%s, %s, %s, %s, %s, 'in_progress', %s, %s, 'agent', NULL, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                session_id,
                task_id,
                teamwork_task_id,
                task_title,
                project_name,
                now,
                now,
                json.dumps(DEFAULT_COMPLETENESS),
                model,
            ),
        )
    return {
        "id": session_id,
        "task_id": task_id,
        "teamwork_task_id": teamwork_task_id,
        "teamwork_task_title": task_title,
        "project_name": project_name,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "created_by": "agent",
        "spec_md": None,
        "completeness": DEFAULT_COMPLETENESS,
        "model": model,
    }


async def get_session(session_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ba_sessions WHERE id = %s LIMIT 1",
                (session_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    row = dict(row)
    if isinstance(row.get("completeness"), str):
        row["completeness"] = json.loads(row["completeness"])
    if row.get("completeness") is None:
        row["completeness"] = dict(DEFAULT_COMPLETENESS)
    else:
        row["completeness"] = normalize_completeness(row["completeness"])
    return row


async def update_session(
    session_id: str,
    *,
    completeness: dict | None = None,
    spec_md: str | None = None,
    status: str | None = None,
) -> None:
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    sets = ["updated_at = %s"]
    params: list = [now]

    if completeness is not None:
        sets.append("completeness = %s")
        params.append(json.dumps(completeness))
    if spec_md is not None:
        sets.append("spec_md = %s")
        params.append(spec_md)
    if status is not None:
        sets.append("status = %s")
        params.append(status)

    params.append(session_id)
    async with pool.connection() as conn:
        await conn.execute(
            f"UPDATE ba_sessions SET {', '.join(sets)} WHERE id = %s",
            params,
        )


async def delete_session(session_id: str) -> bool:
    """Delete a session and its messages. Returns True if the session row was deleted."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM ba_messages WHERE session_id = %s", (session_id,))
        cur = await conn.execute("DELETE FROM ba_sessions WHERE id = %s", (session_id,))
        deleted = cur.rowcount > 0
    return deleted


async def delete_sessions_by_teamwork_id(teamwork_task_id: str) -> int:
    """Delete ALL sessions (and their messages) for a given teamwork task. Returns count deleted."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM ba_messages WHERE session_id IN (SELECT id FROM ba_sessions WHERE teamwork_task_id = %s)",
            (teamwork_task_id,),
        )
        cur = await conn.execute(
            "DELETE FROM ba_sessions WHERE teamwork_task_id = %s",
            (teamwork_task_id,),
        )
        return cur.rowcount


async def get_sessions_by_teamwork_ids(
    teamwork_ids: list[str],
) -> dict[str, dict]:
    """Batch-fetch sessions keyed by teamwork_task_id for task enrichment."""
    if not teamwork_ids:
        return {}
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT DISTINCT ON (teamwork_task_id)
                       id, teamwork_task_id, status, spec_md,
                       completeness, updated_at
                FROM ba_sessions
                WHERE teamwork_task_id = ANY(%s)
                ORDER BY teamwork_task_id, updated_at DESC
                """,
                (teamwork_ids,),
            )
            rows = await cur.fetchall()
    result: dict[str, dict] = {}
    for row in rows:
        row = dict(row)
        if isinstance(row.get("completeness"), str):
            row["completeness"] = json.loads(row["completeness"])
        if row.get("completeness") is None:
            row["completeness"] = dict(DEFAULT_COMPLETENESS)
        else:
            row["completeness"] = normalize_completeness(row["completeness"])
        result[row["teamwork_task_id"]] = row
    return result


# ---------------------------------------------------------------------------
# ba_messages CRUD
# ---------------------------------------------------------------------------


async def insert_message(
    *,
    session_id: str,
    role: str,
    content_type: str,
    content: str | dict | list,
    sources: list | None = None,
    msg_id: str | None = None,
) -> dict:
    pool = await get_pool()
    mid = msg_id or f"msg-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    content_str = json.dumps(content) if not isinstance(content, str) else content
    sources_str = json.dumps(sources) if sources else None

    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO ba_messages
                (id, session_id, role, content_type, content, created_at, sources)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (mid, session_id, role, content_type, content_str, now, sources_str),
        )
    return {
        "id": mid,
        "session_id": session_id,
        "role": role,
        "content_type": content_type,
        "content": content,
        "created_at": now,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# ba_project_settings CRUD
# ---------------------------------------------------------------------------


async def get_project_settings(teamwork_project_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ba_project_settings WHERE teamwork_project_id = %s LIMIT 1",
                (teamwork_project_id,),
            )
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_all_project_settings() -> dict[str, dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM ba_project_settings")
            rows = await cur.fetchall()
    return {row["teamwork_project_id"]: dict(row) for row in rows}


async def upsert_project_settings(
    teamwork_project_id: str,
    *,
    constitution_url: str | None = None,
    constitution_status: str | None = None,
) -> dict:
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                INSERT INTO ba_project_settings
                    (teamwork_project_id, constitution_url, constitution_status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (teamwork_project_id) DO UPDATE SET
                    constitution_url = EXCLUDED.constitution_url,
                    constitution_status = EXCLUDED.constitution_status,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                (teamwork_project_id, constitution_url, constitution_status, now, now),
            )
            row = await cur.fetchone()
    return dict(row)


async def get_messages(session_id: str) -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, session_id, role, content_type, content, created_at, sources
                FROM ba_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
    result = []
    for row in rows:
        row = dict(row)
        # Parse content JSON
        if isinstance(row.get("content"), str):
            try:
                row["content"] = json.loads(row["content"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Parse sources JSON
        if isinstance(row.get("sources"), str):
            try:
                row["sources"] = json.loads(row["sources"])
            except (json.JSONDecodeError, TypeError):
                row["sources"] = None
        result.append(row)
    return result
