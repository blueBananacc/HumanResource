"""Hybrid Search 检索器。

结合 Vector Search（ChromaDB 语义相似度）和 BM25（关键词匹配），
两路并行检索后使用 Reciprocal Rank Fusion (RRF) 融合结果。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

from human_resource.config import (
    BM25_TOP_K,
    DEFAULT_COLLECTION,
    RERANK_TOP_N,
    RELEVANCE_SCORE_THRESHOLD,
    VECTOR_SEARCH_TOP_K,
)
from human_resource.rag.vectorstore import get_all_documents, get_vectorstore
from human_resource.schemas.models import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
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
    if not corpus_docs:
        return []

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


def hybrid_search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    metadata_filter: dict | None = None,
    top_n: int = RERANK_TOP_N,
) -> RetrievalResult:
    """完整 Hybrid Search + Rerank 管线。

    流程：
    1. Vector Search 与 BM25 Search 并行执行
    2. RRF 融合两路结果
    3. Reranker 重排序
    4. 按 relevance threshold 过滤

    Args:
        query: 用户查询文本。
        collection_name: 向量库 collection 名称。
        metadata_filter: 元数据过滤条件（可选）。
        top_n: 最终返回的 chunk 数量。

    Returns:
        RetrievalResult 包含排序后的 chunks。
    """
    if not query.strip():
        return RetrievalResult(chunks=[])

    # ── 1. 并行执行 Vector Search 和 BM25 Search ──
    vector_results: list[tuple[Document, float]] = []
    bm25_results: list[tuple[Document, float]] = []

    def _do_vector() -> list[tuple[Document, float]]:
        return vector_search(
            query, top_k=VECTOR_SEARCH_TOP_K,
            collection_name=collection_name,
            metadata_filter=metadata_filter,
        )

    def _do_bm25() -> list[tuple[Document, float]]:
        corpus = get_all_documents(collection_name)
        return bm25_search(query, corpus, top_k=BM25_TOP_K)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_vector = pool.submit(_do_vector)
        future_bm25 = pool.submit(_do_bm25)

        try:
            vector_results = future_vector.result()
            logger.info("向量检索完成: %d 结果", len(vector_results))
        except Exception:
            logger.exception("向量检索失败")

        try:
            bm25_results = future_bm25.result()
            logger.info("BM25 检索完成: %d 结果", len(bm25_results))
        except Exception:
            logger.exception("BM25 检索失败")

    if not vector_results and not bm25_results:
        return RetrievalResult(chunks=[])

    # ── 2. RRF 融合 ──
    if vector_results and bm25_results:
        fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
    else:
        fused_results = vector_results or bm25_results
    logger.info("RRF 融合完成: %d 结果", len(fused_results))

    # ── 4. Reranker ──
    try:
        from human_resource.rag.reranker import rerank

        reranked_results = rerank(query, fused_results, top_n=top_n)
        logger.info("Rerank 完成: %d 结果", len(reranked_results))
    except Exception:
        logger.exception("Rerank 失败，使用 RRF 结果")
        reranked_results = fused_results[:top_n]

    # ── 5. 转换为 RetrievalResult ──
    chunks: list[RetrievedChunk] = []
    for doc, score in reranked_results:
        # 过滤低 relevance 结果
        if score < RELEVANCE_SCORE_THRESHOLD:
            continue
        chunks.append(RetrievedChunk(
            text=doc.page_content,
            score=score,
            metadata=doc.metadata,
        ))

    logger.info(
        "RAG 检索完成: query='%s', 最终 %d chunks (阈值 %.2f)",
        query[:30], len(chunks), RELEVANCE_SCORE_THRESHOLD,
    )
    return RetrievalResult(chunks=chunks)
