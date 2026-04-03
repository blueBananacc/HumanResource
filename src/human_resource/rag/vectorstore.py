"""ChromaDB 向量存储封装。

管理 ChromaDB collection 的创建、文档写入和检索。
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from human_resource.config import CHROMA_DB_DIR, DEFAULT_COLLECTION
from human_resource.rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

_LOCK_SUFFIXES = ("-wal", "-shm", "-journal")


def _cleanup_sqlite_locks() -> None:
    """清理 SQLite 残留的 WAL / SHM / journal 锁文件。

    debug 中断时这些文件可能处于不一致状态，导致 ChromaDB 无法连接。
    删除它们后 SQLite 会从主 .sqlite3 文件自动恢复，数据不会丢失。
    """
    if not CHROMA_DB_DIR.exists():
        return
    for f in CHROMA_DB_DIR.rglob("*.sqlite3*"):
        if any(f.name.endswith(s) for s in _LOCK_SUFFIXES):
            logger.info("清理残留锁文件: %s", f)
            f.unlink(missing_ok=True)


def get_vectorstore(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    """获取 ChromaDB 向量存储实例。

    当 debug 中断导致 SQLite 锁文件损坏时，自动清理锁文件并重试，
    主数据库文件不会被删除，已索引的数据得以保留。

    Args:
        collection_name: Collection 名称。

    Returns:
        Chroma 实例。
    """
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DB_DIR),
        )
    except (ValueError, AttributeError) as exc:
        logger.warning(
            "ChromaDB 连接失败，尝试清理 SQLite 锁文件后重试: %s", exc,
        )
        _cleanup_sqlite_locks()
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DB_DIR),
        )


def add_documents(
    documents: list[Document],
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """将文档添加到向量存储。

    Args:
        documents: 切分后的文档 chunk 列表。
        collection_name: 目标 collection 名称。
    """
    store = get_vectorstore(collection_name)
    store.add_documents(documents)


def get_all_documents(collection_name: str = DEFAULT_COLLECTION) -> list[Document]:
    """从向量存储中获取所有文档（供 BM25 检索使用）。

    Args:
        collection_name: Collection 名称。

    Returns:
        Document 列表。
    """
    store = get_vectorstore(collection_name)
    result = store.get(include=["documents", "metadatas"])
    docs: list[Document] = []
    if result and result.get("documents"):
        for text, meta in zip(result["documents"], result.get("metadatas", [{}])):
            if text:
                docs.append(Document(page_content=text, metadata=meta or {}))
    return docs
