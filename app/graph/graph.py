"""
Compiled LangGraph agent graph.

build_graph(checkpointer) → compiled graph.
Each session maps to a LangGraph thread_id = session_id.
The checkpointer (AsyncPostgresSaver) persists state between turns automatically.
"""
from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import agent_node, should_continue, tools_node
from app.graph.state import AgentState


def build_graph(checkpointer: AsyncPostgresSaver):
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


def extract_finalize_args(state: AgentState) -> dict | None:
    """
    After graph.ainvoke, find the finalize_turn tool call in the CURRENT turn's
    AI messages (after the last HumanMessage). Returns None if the agent didn't
    call it this turn (fallback: treat last message text as plain response).
    """
    from langchain_core.messages import AIMessage, HumanMessage

    for msg in reversed(state["messages"]):
        # Stop at the turn boundary — don't pick up stale finalize_turn
        # from a previous turn's AIMessage.
        if isinstance(msg, HumanMessage):
            break
        if not isinstance(msg, AIMessage):
            continue
        for tc in (msg.tool_calls or []):
            if tc["name"] == "finalize_turn":
                return tc["args"]
    return None
