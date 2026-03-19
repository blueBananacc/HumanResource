"""Tool Agent (ReAct)。

执行内部工具调用和 MCP 工具调用。
只返回结构化中间结果（ToolResult），不生成用户可见的最终回答。
"""

from __future__ import annotations

from typing import Any

from human_resource.agents.base import BaseAgent
from human_resource.schemas.state import AgentState


class ToolAgent(BaseAgent):
    """工具调用 Agent。"""

    @property
    def name(self) -> str:
        return "tool_agent"

    def run(self, state: AgentState) -> dict[str, Any]:
        # TODO: 实现 Tool Agent 逻辑
        # 1. 从 state 获取工具名称和参数
        # 2. 通过 Tool Registry 查找并执行工具
        # 3. 将结果写入 state.tool_results
        raise NotImplementedError
