"""LangGraph 图定义。

定义 StateGraph 的 Nodes、Edges 和编译逻辑。
Orchestrator 驱动决策循环架构。

图结构：
  START → load_context → memory_retrieval → intent_hints
    → [_intent_router] → orchestrator_decision ⟷ [rag_node / tool_node / memory_node] 循环
    → generate_response → post_process → END

  _intent_router:
    - skill_propose → generate_response（跳过 Orchestrator 循环）
    - 其他 → orchestrator_decision（正常流程）
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from human_resource.agents.orchestrator import (
    generate_response_node,
    intent_hints_node,
    load_context_node,
    memory_node,
    memory_retrieval_node,
    orchestrator_decision_node,
    post_process_node,
    rag_node,
    register_default_tools,
    tool_node,
)
from human_resource.schemas.state import AgentState

logger = logging.getLogger(__name__)


# ── 路由函数（conditional edges） ────────────────────────────
def _intent_router(state: AgentState) -> str:
    """intent_hints_node 后的条件路由。

    - skill_propose → 直接跳到 generate_response（跳过 Orchestrator 循环）
    - 其他 → 正常进入 orchestrator_decision
    """
    action = state.get("orchestrator_action")
    if action == "skill_propose":
        return "generate_response"
    return "orchestrator_decision"


def _decision_router(state: AgentState) -> str:
    """根据 Orchestrator 决策中心的 action 路由到对应节点。"""
    action = state.get("orchestrator_action", "answer")

    route_map = {
        "rag": "rag_node",
        "tool": "tool_node",
        "memory": "memory_node",
        "answer": "generate_response",
        "clarify": "generate_response",
    }
    return route_map.get(action, "generate_response")


# ── 图构建 ───────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """构建 LangGraph StateGraph（Orchestrator 决策循环）。

    Returns:
        未编译的 StateGraph 实例。
    """
    graph = StateGraph(AgentState)

    # ── 添加 Nodes ──
    graph.add_node("load_context", load_context_node)
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("intent_hints", intent_hints_node)
    graph.add_node("orchestrator_decision", orchestrator_decision_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("post_process", post_process_node)

    # ── 添加 Edges ──
    # 入口 → 加载上下文 → 长期记忆检索 → 意图提示 → 决策中心
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "memory_retrieval")
    graph.add_edge("memory_retrieval", "intent_hints")

    # 意图提示 → 条件路由（skill_propose 跳过 Orchestrator）
    graph.add_conditional_edges(
        "intent_hints",
        _intent_router,
        {
            "orchestrator_decision": "orchestrator_decision",
            "generate_response": "generate_response",
        },
    )

    # 决策中心 → 条件路由
    graph.add_conditional_edges(
        "orchestrator_decision",
        _decision_router,
        {
            "rag_node": "rag_node",
            "tool_node": "tool_node",
            "memory_node": "memory_node",
            "generate_response": "generate_response",
        },
    )

    # 各 Specialist Agent 执行后 → 回到决策中心（循环）
    graph.add_edge("rag_node", "orchestrator_decision")
    graph.add_edge("tool_node", "orchestrator_decision")
    graph.add_edge("memory_node", "orchestrator_decision")

    # 生成回复 → 后处理 → 结束
    graph.add_edge("generate_response", "post_process")
    graph.add_edge("post_process", END)

    return graph


def compile_graph():
    """构建并编译图，返回可执行的 CompiledGraph。"""
    # 注册默认内置工具
    register_default_tools()

    # 扫描 Skill 元数据
    try:
        from human_resource.config import SKILLS_DIR
        from human_resource.skills.loader import SkillLoader

        loader = SkillLoader(SKILLS_DIR)
        skills = loader.scan()
        if skills:
            logger.info("Skill 扫描完成: %d 个", len(skills))
    except Exception:
        logger.exception("Skill 扫描失败，技能功能不可用")

    # 注册 MCP 工具（启动本地 MCP Server 并发现工具）
    try:
        from human_resource.mcp.client import register_mcp_tools_sync

        mcp_count = register_mcp_tools_sync()
        if mcp_count > 0:
            logger.info("MCP 工具注册完成: %d 个", mcp_count)
    except Exception:
        logger.exception("MCP 工具注册失败，仅使用内置工具")

    graph = build_graph()
    return graph.compile()
