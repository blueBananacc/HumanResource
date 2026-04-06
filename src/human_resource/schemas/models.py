"""核心数据模型定义。

包含 ToolResult、RetrievedChunk、RetrievalResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
