"""
FastAPI application entrypoint.

Lifecycle:
  startup  → open DB pool, open checkpointer pool, setup LangGraph checkpointer, compile graph
  shutdown → close checkpointer pool, close DB pool
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app import database as db
from app.config import settings
from app.graph.graph import build_graph
from app.routers import project_settings as project_settings_router
from app.routers import session as session_router
from app.routers import teamwork as teamwork_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Opening DB pool…")
    await db.get_pool()

    logger.info("Setting up LangGraph checkpointer…")
    # Use a connection pool so each checkpoint operation gets a fresh connection.
    # A single long-lived connection can be closed by Supabase/pooler (idle timeout
    # or transaction-mode), causing "the connection is closed" on later requests.
    conninfo = settings.database_url
    if "connect_timeout" not in conninfo:
        sep = "&" if "?" in conninfo else "?"
        conninfo = f"{conninfo}{sep}connect_timeout=10"
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        kwargs=dict(autocommit=True, prepare_threshold=0, row_factory=dict_row),
        min_size=2,
        max_size=20,
        open=False,
    )
    await pool.open(wait=True, timeout=30)
    try:
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()
        logger.info("Compiling LangGraph graph…")
        app.state.graph = build_graph(checkpointer)
        logger.info("BA Agent ready (checkpoint pool max_size=20).")
        yield
    finally:
        await pool.close()

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Closing DB pool…")
    await db.close_pool()


app = FastAPI(
    title="BA Intake Agent",
    version="0.1.0",
    description="LangGraph + FastAPI backend for the BA Dashboard intake workflow.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication (when BA_API_KEY is set in env)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: str | None = Depends(_api_key_header)):
    if settings.ba_api_key and api_key != settings.ba_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


_auth = [Depends(_verify_api_key)]

# Mount routers — paths match the n8n webhook paths the frontend already calls
app.include_router(project_settings_router.router, dependencies=_auth)
app.include_router(session_router.router, dependencies=_auth)
app.include_router(teamwork_router.router, dependencies=_auth)


@app.get("/health")
async def health():
    return {"status": "ok"}
