"""重排序器。

通过 HuggingFace Inference API 调用 BAAI/bge-reranker-v2-m3 (cross-encoder)。
对检索结果进行重排序，返回 top-N 最相关的 chunks。
"""

from __future__ import annotations

import logging

import requests
from langchain_core.documents import Document

from human_resource.config import HF_API_TOKEN, RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)

_API_URL = f"https://api-inference.huggingface.co/models/{RERANKER_MODEL}"
_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def rerank(
    query: str,
    documents: list[tuple[Document, float]],
    top_n: int = RERANK_TOP_N,
) -> list[tuple[Document, float]]:
    """使用 cross-encoder 重排序检索结果。

    Args:
        query: 原始查询文本。
        documents: (Document, score) 列表。
        top_n: 返回结果数量。

    Returns:
        重排序后的 (Document, relevance_score) 列表。

    Raises:
        requests.HTTPError: API 调用失败。
    """
    if not documents:
        return []

    # 构建 HuggingFace Inference API 请求
    pairs = [[query, doc.page_content] for doc, _ in documents]

    payload = {"inputs": pairs}

    response = requests.post(
        _API_URL, headers=_HEADERS, json=payload, timeout=30,
    )
    response.raise_for_status()
    raw_scores = response.json()

    # HuggingFace reranker 返回格式可能是:
    # - 直接 [score1, score2, ...] (float 列表)
    # - [{"score": ..., "label": ...}, ...] (dict 列表)
    # - [[{"score": ..., "label": ...}], ...] (嵌套列表)
    scores: list[float] = []
    for item in raw_scores:
        if isinstance(item, (int, float)):
            scores.append(float(item))
        elif isinstance(item, dict):
            scores.append(float(item.get("score", 0.0)))
        elif isinstance(item, list) and item:
            # 嵌套列表，取第一个元素
            inner = item[0]
            if isinstance(inner, dict):
                scores.append(float(inner.get("score", 0.0)))
            else:
                scores.append(float(inner))
        else:
            scores.append(0.0)

    if len(scores) != len(documents):
        logger.warning(
            "Reranker 返回分数数量 (%d) 与文档数量 (%d) 不匹配",
            len(scores), len(documents),
        )
        # 截取或补齐
        scores = scores[:len(documents)]
        while len(scores) < len(documents):
            scores.append(0.0)

    # 将分数与文档关联并排序
    scored_docs = list(zip([doc for doc, _ in documents], scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_n]
