"""Orchestrator Agent。

系统唯一入口，负责全局流程控制：
1. 接收用户输入
2. 加载上下文（Session Memory + Long-term Memory）
3. 意图识别
4. 路由分发到 Specialist Agent
5. 聚合 Agent 结果
6. 生成最终回复
7. 更新记忆
"""

from __future__ import annotations

from typing import Any

from human_resource.agents.base import BaseAgent
from human_resource.schemas.state import AgentState


class OrchestratorAgent(BaseAgent):
    """中心协调者 Agent。"""

    @property
    def name(self) -> str:
        return "orchestrator"

    def run(self, state: AgentState) -> dict[str, Any]:
        # TODO: 实现 Orchestrator 逻辑
        raise NotImplementedError
