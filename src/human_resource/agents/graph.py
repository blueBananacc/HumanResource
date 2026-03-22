"""LangGraph 图定义。

定义 StateGraph 的 Nodes、Edges 和编译逻辑。
Orchestrator 即此图的 compiled graph。

图结构：
  START → load_context → classify_intent → route_agents
    → dispatch（条件路由）→ [rag_node / tool_node / memory_node / chitchat_response]
    → check_next_agent（检查是否还有后续 Agent）→ dispatch / generate_response
    → generate_response → post_process → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from human_resource.agents.orchestrator import (
    classify_intent_node,
    generate_response_node,
    load_context_node,
    memory_node,
    post_process_node,
    rag_node,
    register_default_tools,
    route_agents_node,
    tool_node,
)
from human_resource.schemas.state import AgentState

logger = logging.getLogger(__name__)


# ── 路由函数（conditional edges） ────────────────────────────
def _dispatch_router(state: AgentState) -> str:
    """根据 target_agents 列表和当前索引，路由到对应 Agent Node。"""
    target_agents = state.get("target_agents", [])
    idx = state.get("current_agent_index", 0)

    if idx >= len(target_agents):
        return "generate_response"

    current_target = target_agents[idx]

    # orchestrator 直接响应（chitchat）
    if current_target == "orchestrator":
        return "generate_response"

    agent_map = {
        "rag_agent": "rag_node",
        "tool_agent": "tool_node",
        "memory_agent": "memory_node",
    }
    return agent_map.get(current_target, "generate_response")


def _check_next_agent(state: AgentState) -> str:
    """Agent 执行完毕后，检查是否还有下一个 Agent 要执行。"""
    target_agents = state.get("target_agents", [])
    idx = state.get("current_agent_index", 0) + 1

    if idx < len(target_agents):
        return "dispatch"
    return "generate_response"


def _advance_agent_index(state: AgentState) -> dict[str, Any]:
    """推进 agent 索引。"""
    return {"current_agent_index": state.get("current_agent_index", 0) + 1}


# ── 图构建 ───────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """构建 LangGraph StateGraph。

    Returns:
        未编译的 StateGraph 实例。
    """
    graph = StateGraph(AgentState)

    # ── 添加 Nodes ──
    graph.add_node("load_context", load_context_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("route_agents", route_agents_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("advance_index", _advance_agent_index)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("post_process", post_process_node)

    # ── 添加 Edges ──
    # 入口 → 加载上下文 → 意图识别 → 路由决策
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_intent")
    graph.add_edge("classify_intent", "route_agents")

    # 路由决策 → 条件分发到具体 Agent
    graph.add_conditional_edges(
        "route_agents",
        _dispatch_router,
        {
            "rag_node": "rag_node",
            "tool_node": "tool_node",
            "memory_node": "memory_node",
            "generate_response": "generate_response",
        },
    )

    # 各 Agent 执行后 → 推进索引
    graph.add_edge("rag_node", "advance_index")
    graph.add_edge("tool_node", "advance_index")
    graph.add_edge("memory_node", "advance_index")

    # 推进索引后 → 检查是否还有后续 Agent
    graph.add_conditional_edges(
        "advance_index",
        _check_next_agent,
        {
            "dispatch": "route_agents",  # 回到路由分发下一个 Agent
            "generate_response": "generate_response",
        },
    )

    # 生成回复 → 后处理 → 结束
    graph.add_edge("generate_response", "post_process")
    graph.add_edge("post_process", END)

    return graph


def compile_graph():
    """构建并编译图，返回可执行的 CompiledGraph。"""
    # 注册默认工具
    register_default_tools()

    graph = build_graph()
    return graph.compile()
