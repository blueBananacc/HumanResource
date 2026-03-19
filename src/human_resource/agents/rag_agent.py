"""RAG Agent (ReAct)。

基于 HR 文档库执行检索增强生成。
只返回结构化中间结果（RetrievalResult），不生成用户可见的最终回答。
"""

from __future__ import annotations

from typing import Any

from human_resource.agents.base import BaseAgent
from human_resource.schemas.state import AgentState


class RAGAgent(BaseAgent):
    """RAG 检索增强生成 Agent。"""

    @property
    def name(self) -> str:
        return "rag_agent"

    def run(self, state: AgentState) -> dict[str, Any]:
        # TODO: 实现 RAG Agent 逻辑
        # 1. 从 state 获取查询
        # 2. 调用 RAG Pipeline（retrieve → rerank → context_filter）
        # 3. 将结果写入 state.rag_results
        raise NotImplementedError
