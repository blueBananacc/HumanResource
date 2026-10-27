"""租户 LLM Provider 的 OpenAI 兼容 embedding 服务。

- provider 需有 api_key + embedding_model 才会真正请求;base_url 缺省按
  OpenAI 公网(与 chat 侧 litellm 的 provider 默认一致),非 OpenAI 系厂商
  必须显式配 base_url
- 任何网络/协议错误 → 返回 None(调用方降级为纯文本检索,不阻塞业务;
  失败会写 warning 日志,不会无声)
- 向量存 ``knowledge_articles.embedding``(JSONB),Python 侧 cosine ——
  知识库量级(几百篇)不需要 pgvector
"""
from __future__ import annotations

import logging
import math

import httpx
from sqlalchemy.orm import Session

from app.integrations.llm.registry import resolve_provider

logger = logging.getLogger(__name__)

_EMBED_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_EMBED_DEFAULT_BASE = "https://api.openai.com/v1"


def embed_texts(db: Session, tenant_id: str | None, texts: list[str]) -> list[list[float]] | None:
    """对一组文本算 embedding;不可用返回 None。"""
    if not texts:
        return None
    p = resolve_provider(db, tenant_id)
    api_key = p.embedding_api_key or p.api_key
    if p.is_mock or not api_key or not p.embedding_model:
        return None
    base = (p.embedding_base_url or p.base_url or _EMBED_DEFAULT_BASE).rstrip("/")
    try:
        resp = httpx.post(
            f"{base}/embeddings",
            json={"model": p.embedding_model, "input": texts[:50]},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_EMBED_TIMEOUT,
        )
    except httpx.HTTPError as e:
        logger.warning("[embedding] provider 请求失败 tenant=%s base=%s err=%s", tenant_id, base, e)
        return None
    if resp.status_code != 200:
        logger.warning(
            "[embedding] provider 返回 %s tenant=%s model=%s — 检查 embedding_model/base_url",
            resp.status_code, tenant_id, p.embedding_model,
        )
        return None
    try:
        data = resp.json().get("data") or []
        out = [item.get("embedding") for item in sorted(data, key=lambda x: x.get("index", 0))]
        if not out or any(not isinstance(v, list) for v in out):
            return None
        return out  # type: ignore[return-value]
    except (ValueError, AttributeError, TypeError):
        return None


def embed_text(db: Session, tenant_id: str | None, text: str) -> list[float] | None:
    out = embed_texts(db, tenant_id, [text])
    return out[0] if out else None


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    """cosine 相似度;任一向量缺失/零向量返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
