"""Session Memory。

按 session_id 存储对话消息列表。
运行时使用 Python dict，持久化到 JSON 文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from human_resource.config import SESSIONS_DIR


@dataclass
class SessionMessage:
    """单条会话消息。"""

    role: str  # "user" | "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """单个会话。"""

    session_id: str
    messages: list[SessionMessage] = field(default_factory=list)
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""


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

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息到会话。"""
        session = self.get_or_create(session_id)
        session.messages.append(SessionMessage(role=role, content=content))

    def get_history(self, session_id: str) -> list[SessionMessage]:
        """获取会话历史消息列表。"""
        session = self.get_or_create(session_id)
        return session.messages

    def save(self, session_id: str) -> None:
        """将会话持久化到 JSON 文件。"""
        session = self.get_or_create(session_id)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._persist_dir / f"{session_id}.json"

        data = {
            "session_id": session.session_id,
            "summary": session.summary,
            "messages": [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
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
        )
        for msg in data.get("messages", []):
            session.messages.append(
                SessionMessage(
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata", {}),
                )
            )
        return session
