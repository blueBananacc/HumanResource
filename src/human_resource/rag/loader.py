"""文档加载器。

支持 PDF、DOCX、Markdown、纯文本格式。
提取文档内容和元数据（文件名、标题、创建日期、文档类别）。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


# 支持的文件扩展名 → 加载器类型
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def load_document(file_path: str | Path) -> list[Document]:
    """加载单个文档文件。

    Args:
        file_path: 文档文件路径。

    Returns:
        Document 列表（PDF 可能按页拆分为多个 Document）。

    Raises:
        ValueError: 不支持的文件格式。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: {_SUPPORTED_EXTENSIONS}")

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(str(path))
    # elif suffix == ".md":
    #     from langchain_community.document_loaders import UnstructuredMarkdownLoader

    #     loader = UnstructuredMarkdownLoader(str(path))
    else:  # .txt
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()

    # 补充通用元数据
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
        doc.metadata.setdefault("file_path", str(path))

    return docs


def load_directory(dir_path: str | Path) -> list[Document]:
    """加载目录下所有支持格式的文档。

    Args:
        dir_path: 文档目录路径。

    Returns:
        所有文档的 Document 列表。
    """
    directory = Path(dir_path)
    all_docs: list[Document] = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            docs = load_document(file_path)
            all_docs.extend(docs)

    return all_docs
