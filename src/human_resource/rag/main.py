"""RAG 离线索引入口。

将 data/documents/ 下的文档预处理为向量，写入 ChromaDB。
按子目录自动映射到对应 collection：
  - data/documents/policy/ → policy_collection
  - data/documents/sop/   → sop_collection

用法：
    # 索引全部（自动按子目录映射 collection）
    python -m human_resource.rag.main

    # 指定目录和 collection
    python -m human_resource.rag.main --dir data/documents/policy --collection policy_collection

    # 强制重新索引（忽略 hash 缓存）
    python -m human_resource.rag.main --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from human_resource.config import DOCUMENTS_DIR, POLICY_COLLECTION, SOP_COLLECTION
from human_resource.rag.indexer import index_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG 离线索引：将文档转化为向量并写入 ChromaDB",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="文档目录路径（默认: data/documents/，自动按子目录映射 collection）",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="目标 collection 名称（指定后所有文件写入该 collection）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新索引（忽略文件 hash 缓存）",
    )
    args = parser.parse_args(argv)

    dir_path = args.dir or str(DOCUMENTS_DIR)
    collection_name = args.collection

    logger.info("=" * 60)
    logger.info("RAG 离线索引开始")
    logger.info("  文档目录: %s", dir_path)
    logger.info("  目标 collection: %s", collection_name or "自动映射")
    logger.info("  强制重索引: %s", args.force)
    logger.info("=" * 60)

    total = index_directory(
        dir_path=dir_path,
        collection_name=collection_name,
        force=args.force,
    )

    logger.info("=" * 60)
    logger.info("索引完成，共写入 %d 个 chunks", total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
