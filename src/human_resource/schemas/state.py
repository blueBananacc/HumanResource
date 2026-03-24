"""LangGraph State 定义。

Orchestrator StateGraph 在各 Node 之间传递和更新的全局状态。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from human_resource.schemas.models import IntentResult, RetrievalResult, ToolResult


class AgentState(TypedDict, total=False):
    """LangGraph StateGraph 全局状态。

    使用 TypedDict 以兼容 LangGraph StateGraph(AgentState)。
    total=False 使所有字段可选，Node 只需返回要更新的字段。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: IntentResult | None
    target_agents: list[str]
    rag_results: RetrievalResult | None
    tool_results: list[ToolResult]
    memory_context: list[str]
    user_profile: dict[str, Any] | None
    needs_clarification: bool
    final_response: str | None
    reflection_count: int
    session_id: str
    user_id: str
    current_agent_index: int
    agent_intent_map: list[dict[str, Any]]
