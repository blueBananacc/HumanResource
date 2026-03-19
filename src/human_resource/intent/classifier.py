"""LLM-based 意图分类器。

使用 DeepSeek-reasoner 进行意图分类，输出结构化 IntentResult。
支持多意图识别和低置信度回退。
"""

from __future__ import annotations

from human_resource.schemas.models import IntentResult


class IntentClassifier:
    """基于 LLM 的意图分类器。"""

    def __init__(self) -> None:
        # TODO: 初始化 LLM client (deepseek-reasoner)
        pass

    def classify(self, message: str, session_summary: str = "") -> IntentResult:
        """对用户消息进行意图分类。

        Args:
            message: 用户输入消息。
            session_summary: 当前会话摘要（可选）。

        Returns:
            IntentResult，包含一个或多个意图分类结果。
        """
        # TODO: 实现 LLM structured output 分类
        # 1. 构建分类 prompt（包含意图定义与 few-shot 示例）
        # 2. 调用 DeepSeek-reasoner
        # 3. 解析 JSON 输出为 IntentResult
        raise NotImplementedError
