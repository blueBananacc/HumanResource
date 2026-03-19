"""Long-term Memory（mem0 Cloud 封装）。

通过 mem0ai SDK 管理跨会话的长期记忆：
- User Profile（用户画像）
- Episodic Memory（会话摘要）
- Factual Memory（用户明确告知的事实）
"""

from __future__ import annotations

from typing import Any

from human_resource.config import MEM0_API_KEY


class LongTermMemory:
    """长期记忆管理（mem0 Cloud）。"""

    def __init__(self) -> None:
        from mem0 import MemoryClient

        self._client = MemoryClient(api_key=MEM0_API_KEY)

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        memory_type: str = "factual",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """写入长期记忆。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]。
            user_id: 用户 ID。
            memory_type: 记忆类型（profile / episodic / factual）。
            metadata: 额外元数据。

        Returns:
            mem0 返回结果。
        """
        meta = {"type": memory_type}
        if metadata:
            meta.update(metadata)
        return self._client.add(messages, user_id=user_id, metadata=meta)

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """检索相关长期记忆。

        Args:
            query: 查询文本。
            user_id: 用户 ID。
            top_k: 返回结果数量。

        Returns:
            相关记忆列表。
        """
        results = self._client.search(query, user_id=user_id, limit=top_k)
        return results

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        """获取某用户的所有记忆。"""
        return self._client.get_all(user_id=user_id)
