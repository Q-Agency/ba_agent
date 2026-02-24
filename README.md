# BA Intake Agent

LangGraph + FastAPI backend for the BA Dashboard.

Replaces the n8n session/message workflow with a fully code-driven LangGraph agent while keeping the same API contract — no frontend changes needed.

## Architecture

```
React Frontend  →  FastAPI (this service)  →  Claude (via LangGraph)
                        │                          │
                        │                     ┌────┴────┐
                        ▼                     │  Tools  │
                   PostgreSQL               search_teamwork
                  (shared DB)               search_slack
                 ba_sessions               search_email
                 ba_messages               search_gdrive
              langgraph_checkpoints        finalize_turn (signal)
```

The LangGraph graph:
```
START → agent → routing:
  "tools" → tools_node → agent (loop)
  "end"   → END  (agent called finalize_turn)
```

## Setup

### 1. Python 3.12+

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn[standard] langgraph langgraph-checkpoint-postgres \
    langchain-anthropic langchain-core "psycopg[binary,pool]" \
    pydantic pydantic-settings httpx python-dotenv
```

### 2. Environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and DATABASE_URL at minimum
```

### 3. Database

The service uses the **same PostgreSQL database** as n8n. Tables `ba_sessions` and `ba_messages` must already exist (created by the n8n session-start/message workflows or manually).

LangGraph checkpoint tables are created automatically on first startup.

### 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Point the frontend at this service

In `BA_Dashboard/.env`:
```
VITE_N8N_BASE_URL=http://localhost:8000
```

The frontend calls `POST /session/start`, `/session/message`, `/session/get`, `/teamwork/projects` — all matched exactly.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/session/start` | Create session, run first agent turn |
| POST | `/session/message` | Continue conversation |
| POST | `/session/get` | Fetch session + message history |
| POST | `/session/review` | Approve or request changes |
| POST | `/teamwork/projects` | Fetch all projects (proxies Teamwork API) |
| POST | `/teamwork/project` | Fetch single project |
| GET | `/health` | Health check |

## Adding Tools

To wire up a real integration, edit `app/graph/tools.py`:

- **Teamwork**: Fill in `search_teamwork` — credentials from `TEAMWORK_API_KEY` / `TEAMWORK_DOMAIN`
- **Slack**: Fill in `search_slack` — credentials from `SLACK_BOT_TOKEN`
- **Email**: Fill in `search_email` — wire up Gmail API or Exchange
- **Google Drive**: Fill in `search_gdrive` — wire up Drive API

## Agent Behaviour

Controlled by `app/graph/prompts.py`. The agent:
1. **Research phase** — auto-gathers context from tools before asking questions
2. **Questioning phase** — asks one category at a time, prefers multiple-choice
3. **Generating phase** — produces SPEC.md when all 6 dimensions complete
4. **Review phase** — handles BA feedback on the generated SPEC

The agent must call `finalize_turn` to end each turn with structured output — this is how completeness updates, decisions, and citations are extracted.
