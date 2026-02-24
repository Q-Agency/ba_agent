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

from app.config import settings
from app.graph.prompts import build_system_prompt
from app.graph.state import AgentState
from app.graph.tools import AGENT_TOOLS, search_teamwork, search_slack, search_email, search_gdrive

logger = logging.getLogger(__name__)

# Tool map for execution (all tools except finalize_turn)
_EXECUTABLE_TOOLS = {
    "search_teamwork": search_teamwork,
    "search_slack": search_slack,
    "search_email": search_email,
    "search_gdrive": search_gdrive,
}

# Initialise the model with all tools bound
_llm = ChatAnthropic(
    model=settings.anthropic_model,
    anthropic_api_key=settings.anthropic_api_key,
    max_tokens=8192,
).bind_tools(AGENT_TOOLS)


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


async def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current conversation + dynamic system prompt."""
    system_prompt = build_system_prompt(state)
    messages = _patch_dangling_tool_calls(list(state["messages"]))
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    response: AIMessage = await _llm.ainvoke(messages_with_system)
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
