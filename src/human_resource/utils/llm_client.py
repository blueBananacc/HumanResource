"""LLM 统一客户端。

提供 get_llm(purpose) 工厂方法，根据用途返回对应模型的 ChatOpenAI 实例。
通过 langchain-openai 的 ChatOpenAI 接入 DeepSeek API（OpenAI 兼容格式）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from human_resource.config import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_CONFIG,
)


def get_llm(purpose: str) -> ChatOpenAI:
    """根据用途获取对应模型的 ChatOpenAI 实例。

    Args:
        purpose: 使用场景，对应 config.MODEL_CONFIG 的 key。
            可选值: intent_classification, tool_selection, rag_retrieval,
            context_compression, response_simple, response_complex,
            reflexion, memory_extraction

    Returns:
        配置好的 ChatOpenAI 实例。

    Raises:
        ValueError: purpose 不在 MODEL_CONFIG 中时。
    """
    if purpose not in MODEL_CONFIG:
        raise ValueError(
            f"未知的 LLM 用途: {purpose}，"
            f"可选值: {list(MODEL_CONFIG.keys())}"
        )

    model_name = MODEL_CONFIG[purpose]
    return _build_llm(model_name)


@lru_cache(maxsize=8)
def _build_llm(model_name: str) -> ChatOpenAI:
    """构建并缓存 ChatOpenAI 实例（同一 model_name 复用）。"""
    return ChatOpenAI(
        model=model_name,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
