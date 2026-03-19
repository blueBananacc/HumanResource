"""Embedding 生成器。

通过 HuggingFace Inference API 调用 BAAI/bge-m3，生成 1024 维向量。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from human_resource.config import EMBEDDING_MODEL, HF_API_TOKEN


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """获取 Embedding 模型实例（单例）。"""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_API_TOKEN,
    )
