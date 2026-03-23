"""Session Memory。

按 session_id 存储对话消息列表。
运行时使用 Python dict，持久化到 JSON 文件。

子模块职责：
- Conversation Store: 按 session_id 存储对话消息列表
- Memory Trimmer: 按策略裁剪超长对话历史
- Session Summarizer: 对旧对话进行增量摘要
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from human_resource.config import (
    SESSION_COMPRESS_THRESHOLD,
    SESSION_MEMORY_BUDGET_TOKENS,
    SESSIONS_DIR,
)


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMessage:
    """单条会话消息。"""

    role: str  # "user" | "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _now_iso()


@dataclass
class Session:
    """单个会话。"""

    session_id: str
    messages: list[SessionMessage] = field(default_factory=list)
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""
    intent_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def turn_count(self) -> int:
        """当前对话轮次数（1 轮 = 1 user + 1 assistant）。"""
        return len(self.messages) // 2


class SessionMemory:
    """Session Memory 管理。

    进程内 dict 存储 + JSON 文件持久化。
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._persist_dir = Path(persist_dir or SESSIONS_DIR)
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        """获取或创建会话。"""
        if session_id not in self._sessions:
            # 尝试从文件恢复
            session = self._load_from_file(session_id)
            if session is None:
                session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return self._sessions[session_id]

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加一条消息到会话。

        Args:
            session_id: 会话 ID。
            role: 消息角色 ("user" | "assistant")。
            content: 消息内容。
            metadata: 可选元数据（如 intent_label, tools_used）。
        """
        session = self.get_or_create(session_id)
        msg = SessionMessage(role=role, content=content, metadata=metadata or {})
        session.messages.append(msg)
        session.updated_at = _now_iso()

        # 记录 intent 历史
        if metadata and "intent_label" in metadata:
            session.intent_history.append(metadata["intent_label"])

    def get_history(self, session_id: str) -> list[SessionMessage]:
        """获取会话历史消息列表。"""
        session = self.get_or_create(session_id)
        return session.messages

    def get_summary(self, session_id: str) -> str:
        """获取会话的增量摘要。"""
        session = self.get_or_create(session_id)
        return session.summary

    def get_turn_count(self, session_id: str) -> int:
        """获取当前对话轮次数。"""
        session = self.get_or_create(session_id)
        return session.turn_count

    # ── Memory Trimmer ───────────────────────────────────────

    def trim_and_summarize(
        self,
        session_id: str,
        summarize_fn: Callable[[str], str],
    ) -> bool:
        """裁剪超长对话历史并生成增量摘要。

        当对话轮数超过阈值时：
        1. 将旧消息摘要化（增量合并到 session.summary）
        2. 仅保留最近 N 轮原文

        Args:
            session_id: 会话 ID。
            summarize_fn: 摘要生成函数，接收文本返回摘要。

        Returns:
            是否执行了裁剪。
        """
        session = self.get_or_create(session_id)
        turn_count = session.turn_count

        if turn_count <= SESSION_COMPRESS_THRESHOLD:
            return False

        keep_count = SESSION_COMPRESS_THRESHOLD * 2  # 保留的消息数
        split_idx = len(session.messages) - keep_count

        old_messages = session.messages[:split_idx]
        recent_messages = session.messages[split_idx:]

        # 构建待摘要的文本
        old_text = "\n".join(
            f"{m.role}: {m.content}" for m in old_messages
        )

        # 增量摘要：合并已有摘要 + 新旧消息
        if session.summary:
            merge_text = (
                f"已有摘要:\n{session.summary}\n\n"
                f"新增对话:\n{old_text}"
            )
            session.summary = summarize_fn(merge_text)
        else:
            session.summary = summarize_fn(old_text)

        # 裁剪：仅保留最近消息
        session.messages = recent_messages
        session.updated_at = _now_iso()

        return True

    # ── Persistence ──────────────────────────────────────────

    def save(self, session_id: str) -> None:
        """将会话持久化到 JSON 文件。"""
        session = self.get_or_create(session_id)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._persist_dir / f"{session_id}.json"

        data = {
            "session_id": session.session_id,
            "summary": session.summary,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "intent_history": session.intent_history,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "metadata": m.metadata,
                    "timestamp": m.timestamp,
                }
                for m in session.messages
            ],
        }
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_from_file(self, session_id: str) -> Session | None:
        """从 JSON 文件恢复会话。"""
        file_path = self._persist_dir / f"{session_id}.json"
        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text(encoding="utf-8"))
        session = Session(
            session_id=data["session_id"],
            summary=data.get("summary", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            intent_history=data.get("intent_history", []),
        )
        for msg in data.get("messages", []):
            session.messages.append(
                SessionMessage(
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata", {}),
                    timestamp=msg.get("timestamp", ""),
                )
            )
        return session
