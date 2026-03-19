"""上下文压缩 / 摘要。

使用 DeepSeek-chat 对超长上下文片段进行压缩和摘要。
"""

from __future__ import annotations

# TODO: 实现上下文压缩逻辑
# 1. 会话历史压缩：对较早对话轮次进行 LLM 摘要
# 2. RAG 上下文压缩：相邻 chunk 合并
# 3. 滑动窗口摘要：保留最近 K 轮原文 + 更早对话的摘要
