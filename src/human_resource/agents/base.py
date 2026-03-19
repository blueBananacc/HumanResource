"""BaseAgent 抽象基类。

定义所有 Specialist Agent 的统一接口。
每个 Agent 作为 LangGraph StateGraph 中的一个 Node 函数。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from human_resource.schemas.state import AgentState


class BaseAgent(ABC):
    """所有 Specialist Agent 的基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 标识名称。"""

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """执行 Agent 逻辑。

        Args:
            state: LangGraph 全局状态。

        Returns:
            需要更新的 state 字段字典，LangGraph 会自动合并到全局 state。
        """
