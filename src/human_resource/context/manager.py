"""Token 预算管理 + 上下文组装。

在 LLM context window 限制内，最优地分配和管理各部分 token 预算。
"""

from __future__ import annotations

import tiktoken

from human_resource.config import TOKEN_BUDGET


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本的 token 数量。

    Args:
        text: 输入文本。
        encoding_name: tiktoken 编码名称。

    Returns:
        token 数量。
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


class ContextManager:
    """Token 预算管理器。"""

    def __init__(
        self,
        budget: dict[str, int] | None = None,
    ) -> None:
        self._budget = budget or TOKEN_BUDGET.copy()

    def truncate_to_budget(self, section: str, text: str) -> str:
        """将文本截断到指定段落的 token 预算内。"""
        limit = self._budget.get(section, 0)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= limit:
            return text
        return enc.decode(tokens[:limit])
