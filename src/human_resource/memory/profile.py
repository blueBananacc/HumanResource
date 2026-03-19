"""User Profile 管理。

通过 mem0 metadata 管理用户画像信息（角色、部门、偏好等）。
"""

from __future__ import annotations

from typing import Any

from human_resource.memory.longterm import LongTermMemory


class UserProfileStore:
    """用户画像存储（基于 mem0 Cloud）。"""

    def __init__(self, longterm: LongTermMemory) -> None:
        self._longterm = longterm

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """获取用户画像。

        Args:
            user_id: 用户 ID。

        Returns:
            用户画像字典。
        """
        # 从 mem0 检索 type=profile 的记忆
        all_memories = self._longterm.get_all(user_id)
        profile: dict[str, Any] = {}
        for memory in all_memories:
            if memory.get("metadata", {}).get("type") == "profile":
                profile.update(memory.get("data", {}))
        return profile

    def update_profile(
        self,
        user_id: str,
        profile_data: dict[str, str],
    ) -> None:
        """更新用户画像。

        Args:
            user_id: 用户 ID。
            profile_data: 要更新的画像数据。
        """
        messages = [
            {
                "role": "user",
                "content": f"用户信息更新: {profile_data}",
            }
        ]
        self._longterm.add(
            messages,
            user_id=user_id,
            memory_type="profile",
        )
