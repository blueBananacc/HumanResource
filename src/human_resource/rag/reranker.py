"""重排序器。

通过 HuggingFace Inference API 调用 BAAI/bge-reranker-v2-m3 (cross-encoder)。
对检索结果进行重排序，返回 top-N 最相关的 chunks。
"""

from __future__ import annotations

import requests

from langchain_core.documents import Document

from human_resource.config import HF_API_TOKEN, RERANKER_MODEL, RERANK_TOP_N


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
    """
    if not documents:
        return []

    # 构建 HuggingFace Inference API 请求
    pairs = [[query, doc.page_content] for doc, _ in documents]

    api_url = f"https://api-inference.huggingface.co/models/{RERANKER_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": pairs}

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    scores = response.json()

    # 将分数与文档关联并排序
    scored_docs = list(zip([doc for doc, _ in documents], scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_n]
