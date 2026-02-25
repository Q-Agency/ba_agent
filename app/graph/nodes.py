"""
LangGraph node functions.

Graph topology:
  START → agent → routing:
    "tools"  → tools_node → agent (loop)
    "end"    → END          (when finalize_turn is called OR no tool calls)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.prompts import build_system_prompt
from app.graph.state import AgentState
from app.graph.tools import AGENT_TOOLS, search_teamwork

logger = logging.getLogger(__name__)

# Tool map for execution (all tools except finalize_turn)
_EXECUTABLE_TOOLS = {
    "search_teamwork": search_teamwork,
}

# ---------------------------------------------------------------------------
# Model registry & factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5",
    },
    "qwen3-coder-next": {
        "provider": "openai_compatible",
        "model": "qwen3-coder-next-fp8",
        "label": "Qwen 3 Coder (Local)",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
    },
}

DEFAULT_MODEL = "claude-sonnet-4-6"

# Cache LLM instances so we don't re-create them on every call
_llm_cache: dict[str, Any] = {}


def get_llm(model_id: str):
    """Return a cached LLM instance (with tools bound) for the given model_id."""
    if model_id in _llm_cache:
        return _llm_cache[model_id]

    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        logger.warning("Unknown model_id %r, falling back to %s", model_id, DEFAULT_MODEL)
        entry = MODEL_REGISTRY[DEFAULT_MODEL]
        model_id = DEFAULT_MODEL

    if entry["provider"] == "anthropic":
        llm = ChatAnthropic(
            model=entry["model"],
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=8192,
        ).bind_tools(AGENT_TOOLS)
    elif entry["provider"] == "openai_compatible":
        llm = ChatOpenAI(
            model=entry["model"],
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_tokens=8192,
        ).bind_tools(AGENT_TOOLS)
    elif entry["provider"] == "google":
        llm = ChatGoogleGenerativeAI(
            model=entry["model"],
            google_api_key=settings.google_api_key,
            max_output_tokens=8192,
        ).bind_tools(AGENT_TOOLS)
    else:
        raise ValueError(f"Unknown provider: {entry['provider']}")

    _llm_cache[model_id] = llm
    return llm


def _patch_dangling_tool_calls(messages: list) -> list:
    """
    Insert synthetic ToolMessage responses for any tool_use blocks that were
    never followed by a tool_result (e.g. finalize_turn, which exits the graph
    before tools_node runs). Without these, Claude returns a 400 on the next turn.
    """
    result = []
    for i, msg in enumerate(messages):
        result.append(msg)
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        # Collect tool_call_ids that are already covered by immediately following ToolMessages
        covered: set[str] = set()
        j = i + 1
        while j < len(messages) and isinstance(messages[j], ToolMessage):
            covered.add(messages[j].tool_call_id)
            j += 1
        for tc in msg.tool_calls:
            if tc["id"] not in covered:
                result.append(
                    ToolMessage(content="acknowledged", tool_call_id=tc["id"], name=tc["name"])
                )
    return result


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Call the LLM with the current conversation + dynamic system prompt."""
    model_id = config.get("configurable", {}).get("model_id", DEFAULT_MODEL)
    llm = get_llm(model_id)

    system_prompt = build_system_prompt(state)
    messages = _patch_dangling_tool_calls(list(state["messages"]))
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    response: AIMessage = await llm.ainvoke(messages_with_system)
    return {"messages": [response]}


async def tools_node(state: AgentState) -> dict:
    """
    Execute tool calls from the last AI message.
    finalize_turn is excluded — it's never executed (detected by routing).
    """
    last_msg: AIMessage = state["messages"][-1]
    results: list[ToolMessage] = []

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_fn = _EXECUTABLE_TOOLS.get(tool_name)

        if tool_fn is None:
            # Unknown or non-executable tool — return an error ToolMessage
            results.append(
                ToolMessage(
                    content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    tool_call_id=tc["id"],
                    name=tool_name,
                )
            )
            continue

        try:
            result = await tool_fn.ainvoke(tc["args"])
            content = result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            logger.exception("Tool %s raised: %s", tool_name, exc)
            content = json.dumps({"error": str(exc)})

        results.append(
            ToolMessage(content=content, tool_call_id=tc["id"], name=tool_name)
        )

    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Route after the agent node.

    - "tools"  → agent made tool calls (excluding finalize_turn)
    - "end"    → agent called finalize_turn, or returned plain text
    """
    last_msg = state["messages"][-1]

    if not isinstance(last_msg, AIMessage):
        return "end"
    if not last_msg.tool_calls:
        return "end"

    # Only end without running tools when the sole tool call is finalize_turn.
    # If the agent called finalize_turn together with search_teamwork etc., we must
    # run those tools first so the next turn doesn't see dangling tool_use blocks.
    tool_names = {tc["name"] for tc in last_msg.tool_calls}
    if tool_names == {"finalize_turn"}:
        return "end"
    return "tools"
