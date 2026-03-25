"""核心数据模型定义。

包含 AgentMessage、IntentResult、ToolResult、RetrievalResult 等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Intent 相关 ──────────────────────────────────────────────


class IntentLabel(str, Enum):
    """意图分类体系（MVP）。"""

    POLICY_QA = "policy_qa"
    PROCESS_INQUIRY = "process_inquiry"
    EMPLOYEE_LOOKUP = "employee_lookup"
    DOCUMENT_SEARCH = "document_search"
    TOOL_ACTION = "tool_action"
    MEMORY_RECALL = "memory_recall"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


@dataclass
class IntentItem:
    """单个意图识别结果。"""

    label: IntentLabel
    confidence: float = 0.0
    entities: dict[str, Any] = field(default_factory=dict)
    requires_tools: list[str] = field(default_factory=list)


@dataclass
class IntentResult:
    """意图识别完整结果（支持多意图）。

    intents 列表按逻辑执行顺序排列（由 LLM 决定），而非按置信度排序。
    每个 IntentItem 携带自己的 requires_tools，IntentResult.requires_tools
    为所有意图所需工具的有序并集。
    """

    intents: list[IntentItem] = field(default_factory=list)

    @property
    def requires_tools(self) -> list[str]:
        """所有意图所需工具的有序并集。"""
        tools: list[str] = []
        seen: set[str] = set()
        for intent in self.intents:
            for tool in intent.requires_tools:
                if tool not in seen:
                    tools.append(tool)
                    seen.add(tool)
        return tools

    @property
    def primary_intent(self) -> IntentItem | None:
        """返回第一个意图（逻辑执行顺序中的首个）。"""
        if not self.intents:
            return None
        return self.intents[0]


# ── Tool 相关 ────────────────────────────────────────────────


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    tool_name: str = ""
    data: Any = None
    error: str | None = None
    formatted: str = ""


# ── RAG 相关 ─────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """检索到的单个文档片段。"""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """RAG 检索完整结果。"""

    chunks: list[RetrievedChunk] = field(default_factory=list)


# ── Agent 通信 ───────────────────────────────────────────────


@dataclass
class AgentMessage:
    """Agent 间通信消息格式。"""

    sender: str  # "orchestrator" | "rag_agent" | "tool_agent" | "memory_agent"
    receiver: str
    content: Any
    message_type: str = "result"  # "request" | "result" | "error"
    metadata: dict[str, Any] = field(default_factory=dict)
