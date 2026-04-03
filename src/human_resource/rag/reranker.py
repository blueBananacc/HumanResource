"""重排序器。

通过 HuggingFace Inference API 调用 BAAI/bge-reranker-v2-m3 (cross-encoder)。
对检索结果进行重排序，返回 top-N 最相关的 chunks。
"""

from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.documents import Document

from human_resource.config import HF_API_TOKEN, RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)

_API_URL = f"https://router.huggingface.co/hf-inference/models/{RERANKER_MODEL}"
_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# 构建带重试 & SSL 容错的 Session
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.verify = False  # 跳过 SSL 验证（解决国内网络 / 代理导致的证书问题）

# 抑制 InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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

    # 构建 HuggingFace Inference API 请求（text / text_pair dict 格式）
    pairs = [
        {"text": query, "text_pair": doc.page_content}
        for doc, _ in documents
    ]

    payload = {
        "inputs": pairs,
        "parameters": {"function_to_apply": "sigmoid"},
    }

    response = _session.post(
        _API_URL, headers=_HEADERS, json=payload, timeout=60,
    )
    response.raise_for_status()
    raw = response.json()

    # HuggingFace 响应格式：[[{"label": "LABEL_0", "score": 0.69}, ...]]
    # raw[0] 是一个 list，每个元素对应一个输入 pair 的分类结果
    scores: list[float] = []
    items = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    for item in items:
        if isinstance(item, (int, float)):
            scores.append(float(item))
        elif isinstance(item, dict):
            scores.append(float(item.get("score", 0.0)))
        elif isinstance(item, list) and item:
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
        scores = scores[:len(documents)]
        while len(scores) < len(documents):
            scores.append(0.0)

    # 将分数与文档关联并排序
    scored_docs = list(zip([doc for doc, _ in documents], scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_n]
