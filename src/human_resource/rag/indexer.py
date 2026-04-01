"""文档索引入口。

加载 → 切片 → 写入向量库的完整离线索引流程。
支持增量索引：通过文件 hash 跳过已索引的文档。
支持双 collection：按子目录自动映射到 policy_collection / sop_collection。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from human_resource.config import (
    CHROMA_DB_DIR,
    DEFAULT_COLLECTION,
    DIR_COLLECTION_MAP,
    DOCUMENTS_DIR,
)
from human_resource.rag.chunker import chunk_documents
from human_resource.rag.loader import load_document
from human_resource.rag.vectorstore import add_documents

logger = logging.getLogger(__name__)

_INDEX_RECORD_FILE = CHROMA_DB_DIR / "_indexed_files.json"


def _file_hash(file_path: Path) -> str:
    """计算文件 MD5 hash。"""
    h = hashlib.md5()
    h.update(file_path.read_bytes())
    return h.hexdigest()


def _load_index_record() -> dict[str, str]:
    """加载已索引文件记录 {file_path: hash}。"""
    if _INDEX_RECORD_FILE.exists():
        return json.loads(_INDEX_RECORD_FILE.read_text(encoding="utf-8"))
    return {}


def _save_index_record(record: dict[str, str]) -> None:
    """保存已索引文件记录。"""
    _INDEX_RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_RECORD_FILE.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def index_file(
    file_path: str | Path,
    collection_name: str = DEFAULT_COLLECTION,
    force: bool = False,
) -> int:
    """索引单个文档文件。

    Args:
        file_path: 文档文件路径。
        collection_name: 目标 collection 名称。
        force: 是否强制重新索引（忽略 hash 检查）。

    Returns:
        写入的 chunk 数量。
    """
    path = Path(file_path).resolve()
    record = _load_index_record()
    file_key = str(path)
    current_hash = _file_hash(path)

    if not force and record.get(file_key) == current_hash:
        logger.info("文件已索引且未变更，跳过: %s", path.name)
        return 0

    # 加载 → 切片 → 写入
    docs = load_document(path)
    chunks = chunk_documents(docs)

    if not chunks:
        logger.warning("文件切片为空: %s", path.name)
        return 0

    add_documents(chunks, collection_name=collection_name)

    # 更新索引记录
    record[file_key] = current_hash
    _save_index_record(record)

    logger.info("索引完成: %s → %d chunks", path.name, len(chunks))
    return len(chunks)


def index_directory(
    dir_path: str | Path | None = None,
    collection_name: str | None = None,
    force: bool = False,
) -> int:
    """索引目录下所有文档。

    如果指定了 collection_name，则所有文件索引到该 collection。
    如果未指定，则根据子目录名自动映射（policy/ → policy_collection，
    sop/ → sop_collection），顶层文件使用 DEFAULT_COLLECTION。

    Args:
        dir_path: 文档目录路径，默认为 config.DOCUMENTS_DIR。
        collection_name: 目标 collection 名称，None 表示自动映射。
        force: 是否强制重新索引。

    Returns:
        总写入的 chunk 数量。
    """
    directory = Path(dir_path or DOCUMENTS_DIR)
    if not directory.exists():
        logger.warning("文档目录不存在: %s", directory)
        return 0

    from human_resource.rag.loader import _SUPPORTED_EXTENSIONS

    total_chunks = 0
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue

        # 确定目标 collection
        if collection_name:
            target_collection = collection_name
        else:
            # 根据相对于 directory 的第一级子目录名映射
            try:
                rel = file_path.relative_to(directory)
                subdir = rel.parts[0] if len(rel.parts) > 1 else ""
            except ValueError:
                subdir = ""
            target_collection = DIR_COLLECTION_MAP.get(subdir, DEFAULT_COLLECTION)

        try:
            count = index_file(
                file_path, collection_name=target_collection, force=force,
            )
            total_chunks += count
        except Exception:
            logger.exception("索引文件失败: %s", file_path.name)

    logger.info("目录索引完成: %s → 总计 %d chunks", directory, total_chunks)
    return total_chunks
