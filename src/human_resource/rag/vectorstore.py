"""ChromaDB 向量存储封装。

管理 ChromaDB collection 的创建、文档写入和检索。
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from human_resource.config import CHROMA_DB_DIR
from human_resource.rag.embedder import get_embeddings


def get_vectorstore(collection_name: str = "hr_documents") -> Chroma:
    """获取 ChromaDB 向量存储实例。

    Args:
        collection_name: Collection 名称。

    Returns:
        Chroma 实例。
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DB_DIR),
    )


def add_documents(
    documents: list[Document],
    collection_name: str = "hr_documents",
) -> None:
    """将文档添加到向量存储。

    Args:
        documents: 切分后的文档 chunk 列表。
        collection_name: 目标 collection 名称。
    """
    store = get_vectorstore(collection_name)
    store.add_documents(documents)
