"""产品使用文档灌知识库(T6,MintHCM BrowseDocumentationTool 对应物)。

把 ``product_help/`` 下的 markdown 手册按 ``##``/``###`` 标题切片,
以 ``category="product_help"`` 写入每个租户的 ``knowledge_articles``:

- 分类不在 ``KnowledgeArticle.CATEGORIES`` 校验器里 → API 侧创建/编辑/删除
  全被拒,产品文档与租户知识天然隔离且只读;
- assistant ``search_product_help`` 工具用 ``category=product_help`` 限定召回;
- 幂等:每次运行先清掉本租户全部 product_help 行再重建(文档随版本刷新)。

用法(容器内)::

    docker compose exec backend python scripts/seed_product_help.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.domain.models import KnowledgeArticle, Tenant  # noqa: E402
from app.infra.db import SessionLocal  # noqa: E402
from app.services.embedding import embed_text  # noqa: E402

DOC_DIR = Path(__file__).resolve().parent / "product_help"
MAX_SECTION = 4000

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def slice_markdown(doc_name: str, text: str) -> list[tuple[str, str]]:
    """按标题层级切片 → [(标题路径, 正文)]。

    ``##`` 开新章(章直属正文归到 "doc > 章"),``###`` 开新节
    ("doc > 章 > 节")。超长段按 MAX_SECTION 再切。
    """
    sections: list[tuple[str, str]] = []
    chapter = ""
    section = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body or not chapter:
            return
        title = (
            f"{doc_name} > {chapter} > {section}" if section else f"{doc_name} > {chapter}"
        )
        for i in range(0, len(body), MAX_SECTION):
            suffix = f"({i // MAX_SECTION + 1})" if i else ""
            sections.append((title + suffix, body[i : i + MAX_SECTION]))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m is None:
            buf.append(line)
            continue
        level, head = len(m.group(1)), m.group(2)
        if level == 1:
            continue  # 文档大标题,不进切片
        flush()
        if level == 2:
            chapter, section = head, ""
        else:
            section = head
        buf.append(line)
    flush()
    return sections


def main() -> None:
    docs = sorted(DOC_DIR.glob("*.md"))
    if not docs:
        print("product_help/ 下没有 markdown 文档")
        return
    all_sections: list[tuple[str, str]] = []
    for d in docs:
        doc_name = d.stem.replace("-", " ").replace("_", " ")
        if d.stem == "customer-guide":
            doc_name = "HROnfire 使用手册"
        all_sections.extend(slice_markdown(doc_name, d.read_text(encoding="utf-8")))
    print(f"切出 {len(all_sections)} 个文档片段")

    db = SessionLocal()
    try:
        tenants = db.scalars(select(Tenant)).all()
        for t in tenants:
            deleted = db.execute(
                delete(KnowledgeArticle).where(
                    KnowledgeArticle.tenant_id == t.id,
                    KnowledgeArticle.category == "product_help",
                )
            ).rowcount
            embedded = 0
            for title, body in all_sections:
                a = KnowledgeArticle(
                    tenant_id=t.id,
                    title=title[:128],
                    category="product_help",
                    content=body,
                    tags=["product_help"],
                    status="published",
                    is_playbook=False,
                )
                db.add(a)
                try:
                    emb = embed_text(db, t.id, f"{a.title}\n{a.content[:2000]}")
                except Exception:
                    emb = None
                if emb:
                    a.embedding = emb
                    embedded += 1
            db.commit()
            print(
                f"tenant {t.name}: 清除 {deleted} 条旧文档,"
                f"灌入 {len(all_sections)} 条(embedding {embedded})"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
