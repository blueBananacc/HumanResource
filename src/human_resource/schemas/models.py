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


@dataclass
class IntentResult:
    """意图识别完整结果（支持多意图）。"""

    intents: list[IntentItem] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)

    @property
    def primary_intent(self) -> IntentItem | None:
        """返回置信度最高的主意图。"""
        if not self.intents:
            return None
        return max(self.intents, key=lambda x: x.confidence)


# ── Tool 相关 ────────────────────────────────────────────────


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
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
