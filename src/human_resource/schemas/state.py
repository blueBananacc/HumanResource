"""LangGraph State 定义。

Orchestrator StateGraph 在各 Node 之间传递和更新的全局状态。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from human_resource.schemas.models import IntentResult, RetrievalResult, ToolResult


class AgentState:
    """LangGraph 使用的 TypedDict 风格状态。

    使用 class + __annotations__ 以兼容 LangGraph StateGraph(AgentState)。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: IntentResult | None
    target_agents: list[str]
    rag_results: RetrievalResult | None
    tool_results: list[ToolResult]
    memory_context: list[str]
    user_profile: dict[str, Any] | None
    final_response: str | None
    reflection_count: int
