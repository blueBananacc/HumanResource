"""文档切片器。

使用 RecursiveCharacterTextSplitter 按段落 → 句子 → 字符递归切分。
保留 chunk 级元数据。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from human_resource.config import CHUNK_OVERLAP, CHUNK_SIZE


def create_chunker() -> RecursiveCharacterTextSplitter:
    """创建文档切片器实例。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """将文档列表切分为 chunk。

    Args:
        documents: 原始文档列表。

    Returns:
        切分后的 Document 列表，每个 Document 包含 chunk 文本和继承的元数据。
    """
    chunker = create_chunker()
    chunks = chunker.split_documents(documents)

    # 添加 chunk 序号元数据
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    return chunks
