"""LangGraph State 定义。

Orchestrator StateGraph 在各 Node 之间传递和更新的全局状态。
Orchestrator 驱动决策循环架构：决策中心自主推理决定下一步动作。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from human_resource.schemas.models import RetrievalResult, ToolResult


class AgentState(TypedDict, total=False):
    """LangGraph StateGraph 全局状态。

    使用 TypedDict 以兼容 LangGraph StateGraph(AgentState)。
    total=False 使所有字段可选，Node 只需返回要更新的字段。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent_hints: str | None  # 意图提示（轻量分析，供 Orchestrator 参考）
    orchestrator_action: str | None  # 当前决策动作: rag/tool/memory/answer/clarify
    orchestrator_reasoning: str | None  # 决策推理过程
    orchestrator_action_input: dict[str, Any] | None  # 动作参数
    rag_results: RetrievalResult | None
    tool_results: list[ToolResult]
    session_context: list[str]
    memory_context: list[str]
    user_profile: dict[str, Any] | None
    final_response: str | None
    loop_count: int  # 当前决策循环次数
    max_loops: int  # 最大循环次数（默认 5）
    session_id: str
    user_id: str
