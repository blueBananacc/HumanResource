"""Hybrid Search 检索器。

结合 Vector Search（ChromaDB 语义相似度）和 BM25（关键词匹配），
使用 Reciprocal Rank Fusion (RRF) 融合两路结果。
"""

from __future__ import annotations

from langchain_core.documents import Document

from human_resource.config import BM25_TOP_K, VECTOR_SEARCH_TOP_K
from human_resource.rag.vectorstore import get_vectorstore


def vector_search(
    query: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    collection_name: str = "hr_documents",
    metadata_filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """向量语义检索。

    Args:
        query: 查询文本。
        top_k: 返回结果数量。
        collection_name: 目标 collection。
        metadata_filter: 元数据过滤条件。

    Returns:
        (Document, score) 元组列表。
    """
    store = get_vectorstore(collection_name)
    kwargs = {"k": top_k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return store.similarity_search_with_relevance_scores(query, **kwargs)


def bm25_search(
    query: str,
    corpus_docs: list[Document],
    top_k: int = BM25_TOP_K,
) -> list[tuple[Document, float]]:
    """BM25 关键词检索。

    Args:
        query: 查询文本。
        corpus_docs: 语料库文档列表。
        top_k: 返回结果数量。

    Returns:
        (Document, score) 元组列表。
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.page_content.split() for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    # 按分数排序取 top_k
    scored_docs = list(zip(corpus_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


def reciprocal_rank_fusion(
    *result_lists: list[tuple[Document, float]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion 融合多路检索结果。

    Args:
        *result_lists: 多路检索结果列表。
        k: RRF 常数。

    Returns:
        融合并去重后的 (Document, rrf_score) 列表，按分数降序排列。
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, (doc, _score) in enumerate(results):
            doc_id = doc.metadata.get("source", "") + str(
                doc.metadata.get("chunk_index", "")
            )
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    fused = [(doc_map[doc_id], score) for doc_id, score in doc_scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
