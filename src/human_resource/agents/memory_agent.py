"""Memory Agent。

管理会话记忆读写、长期记忆提取和检索。
协调 Session Memory（本地）和 Long-term Memory（mem0 Cloud）。
"""

from __future__ import annotations

from typing import Any

from human_resource.agents.base import BaseAgent
from human_resource.schemas.state import AgentState


class MemoryAgent(BaseAgent):
    """记忆管理 Agent。"""

    @property
    def name(self) -> str:
        return "memory_agent"

    def run(self, state: AgentState) -> dict[str, Any]:
        # TODO: 实现 Memory Agent 逻辑
        # 1. 从 state 获取查询
        # 2. 检索 session memory + long-term memory
        # 3. 将结果写入 state.memory_context
        raise NotImplementedError
