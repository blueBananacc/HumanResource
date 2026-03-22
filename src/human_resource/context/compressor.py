"""上下文压缩/摘要。

当会话历史或上下文片段超过 token 预算时，使用 DeepSeek-chat 进行压缩摘要。
策略：滑动窗口摘要 — 保留最近 K 轮原文 + 更早对话的增量摘要。
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from human_resource.config import SESSION_KEEP_RECENT_TURNS
from human_resource.context.manager import count_tokens
from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
请将以下对话历史压缩为一段简洁的摘要。
保留：关键问题、关键回答、用户意图变化、重要决策。
省略：寒暄、重复内容、无实质信息的对话。
用中文输出，不超过 200 字。"""


class ContextCompressor:
    """上下文压缩器。"""

    def __init__(self) -> None:
        self._llm = get_llm("context_compression")

    def compress_history(
        self,
        messages: list[dict[str, str]],
        keep_recent: int = SESSION_KEEP_RECENT_TURNS,
        token_budget: int = 2000,
    ) -> tuple[str, list[dict[str, str]]]:
        """压缩对话历史。

        将超出预算的早期对话摘要化，保留最近 keep_recent 轮原文。

        Args:
            messages: 对话消息列表 [{"role": "...", "content": "..."}]。
            keep_recent: 保留最近 N 轮原文（1 轮 = 1 user + 1 assistant）。
            token_budget: 对话历史的 token 预算。

        Returns:
            (summary, recent_messages) 元组：
            - summary: 早期对话的摘要文本（可为空字符串）。
            - recent_messages: 保留的最近消息列表。
        """
        turn_count = len(messages) // 2
        if turn_count <= keep_recent:
            return "", messages

        split_idx = len(messages) - keep_recent * 2
        early_messages = messages[:split_idx]
        recent_messages = messages[split_idx:]

        early_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in early_messages
        )
        if count_tokens(early_text) < 100:
            return "", messages

        summary = self._summarize(early_text)
        return summary, recent_messages

    def _summarize(self, text: str) -> str:
        """使用 LLM 生成摘要。"""
        try:
            response = self._llm.invoke([
                SystemMessage(content=_SUMMARIZE_PROMPT),
                HumanMessage(content=text),
            ])
            return response.content.strip()
        except Exception:
            logger.exception("对话摘要生成失败")
            return text[:200] + "..."

    def compress_context(self, text: str, max_tokens: int) -> str:
        """压缩单段上下文到 token 预算内。"""
        if count_tokens(text) <= max_tokens:
            return text

        try:
            prompt = (
                f"请将以下内容压缩到约 {max_tokens} tokens 以内，"
                "保留关键信息，用中文输出：\n\n" + text
            )
            response = self._llm.invoke([
                SystemMessage(content="你是一个文本压缩助手。保留关键信息，去除冗余。"),
                HumanMessage(content=prompt),
            ])
            return response.content.strip()
        except Exception:
            logger.exception("上下文压缩失败，使用截断")
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            return enc.decode(tokens[:max_tokens])
# 3. 滑动窗口摘要：保留最近 K 轮原文 + 更早对话的摘要
