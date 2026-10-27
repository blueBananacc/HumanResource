"""知识库 API —— 制度文档 + assistant RAG 语料。

- 管理 ``GET/POST/PUT/DELETE /knowledge/articles``(manage_employees 写 / view_reports 读)
- ``POST /knowledge/search {q,limit}`` → title+snippet,assistant search_knowledge 走这里
- ESS ``GET /me/knowledge`` published 全文检索
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import String, case, func, or_, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import KnowledgeArticle, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.bm25 import bm25_scores
from app.services.embedding import cosine, embed_text, embed_texts
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    require_permission,
)

router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(require_feature("hr.suite"))],
)
me_router = APIRouter(
    prefix="/me/knowledge",
    tags=["knowledge-ess"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class ArticleIn(BaseModel):
    title: str = Field(min_length=1, max_length=128)
    category: str = "other"
    content: str = Field(min_length=1, max_length=50000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    status: str = "published"
    is_playbook: bool = False

    @field_validator("category")
    @classmethod
    def _v_cat(cls, v: str) -> str:
        if v not in KnowledgeArticle.CATEGORIES:
            raise ValueError(f"未知分类,可选:{','.join(KnowledgeArticle.CATEGORIES)}")
        return v

    @field_validator("status")
    @classmethod
    def _v_status(cls, v: str) -> str:
        if v not in ("draft", "published", "archived"):
            raise ValueError("status 只能是 draft/published/archived")
        return v

    @field_validator("tags")
    @classmethod
    def _v_tags(cls, v: list[str]) -> list[str]:
        return [t[:32] for t in v]


class ArticleOut(BaseModel):
    id: str
    title: str
    category: str
    content: str
    tags: list[str]
    status: str
    is_playbook: bool
    updated_by: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SearchIn(BaseModel):
    q: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=5, ge=1, le=20)
    # T6:按分类限定召回 — assistant 的 search_product_help 只查产品文档
    category: str | None = Field(default=None, max_length=32)


class SearchHit(BaseModel):
    id: str
    title: str
    category: str
    tags: list[str]
    snippet: str
    score: float | None = None
    # P2 引用溯源:命中定位到正文偏移/段落/最近小节标题,便于「有据可依」呈现
    match_field: str | None = None  # title / content / semantic
    char_offset: int | None = None
    paragraph: int | None = None
    section: str | None = None


# ============ 管理端 ============


@router.get("/articles", response_model=PageOut[ArticleOut])
def list_articles(
    status_filter: str | None = None,
    category: str | None = None,
    is_playbook: bool | None = None,
    brief: bool = False,
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, q = paging
    stmt = select(KnowledgeArticle).where(KnowledgeArticle.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(KnowledgeArticle.status == status_filter)
    if category:
        stmt = stmt.where(KnowledgeArticle.category == category)
    if is_playbook is not None:
        stmt = stmt.where(KnowledgeArticle.is_playbook == is_playbook)
    stmt = apply_q_ilike(stmt, q, KnowledgeArticle.title, KnowledgeArticle.content)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(KnowledgeArticle.updated_at.desc()).limit(limit).offset(offset)
    ).all()
    if not brief:
        return PageOut(items=list(rows), total=total, limit=limit, offset=offset)
    # brief=true:content 只回首部摘要 — 给 list_playbooks 这类"只要 id+title"的
    # 调用方省整页正文的带宽,避免工具结果被 8000 字截断后丢 id。
    items = []
    for a in rows:
        o = ArticleOut.model_validate(a)
        o.content = a.content[:300]
        items.append(o)
    return PageOut(items=items, total=total, limit=limit, offset=offset)


@router.post("/articles", response_model=ArticleOut, status_code=status.HTTP_201_CREATED)
def create_article(
    body: ArticleIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    a = KnowledgeArticle(
        tenant_id=current.tenant_id,
        title=body.title,
        category=body.category,
        content=body.content,
        tags=body.tags,
        status=body.status,
        is_playbook=body.is_playbook,
        updated_by=current.id,
    )
    db.add(a)
    db.flush()
    _refresh_embedding(db, a)
    write_audit(db, actor=current, entity_type="knowledge_article", entity_id=a.id, action="create", request=request)
    db.commit()
    return a


def _get(db: Session, aid: str, current: User) -> KnowledgeArticle:
    a = db.get(KnowledgeArticle, aid)
    if a is None or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文章不存在")
    return a


@router.get("/articles/{aid}", response_model=ArticleOut)
def get_article(
    aid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return _get(db, aid, current)


@router.put("/articles/{aid}", response_model=ArticleOut)
def update_article(
    aid: str,
    body: ArticleIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    a = _get(db, aid, current)
    if a.category == "product_help":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "产品文档由系统随版本维护,不支持编辑",
        )
    # 只有检索源(title/content)变了或还没有向量才重算 embedding —
    # 纯元数据保存(status/is_playbook/tags)不该每次付一次 provider 调用。
    need_embed = a.embedding is None or a.title != body.title or a.content != body.content
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    a.updated_by = current.id
    a.updated_at = _now()
    if need_embed:
        _refresh_embedding(db, a)
    write_audit(db, actor=current, entity_type="knowledge_article", entity_id=a.id, action="update", request=request)
    db.commit()
    return a


@router.delete("/articles/{aid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_article(
    aid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    a = _get(db, aid, current)
    if a.category == "product_help":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "产品文档由系统随版本维护,不支持删除",
        )
    write_audit(db, actor=current, entity_type="knowledge_article", entity_id=a.id, action="delete", request=request)
    db.delete(a)
    db.commit()


# ============ 检索(assistant RAG + ESS):trgm 模糊 + embedding 语义混合 ============


def _lex_score(q: str):
    """词面分:title/tags 走 pg_trgm 模糊匹配(长短字段对 trgm 不可靠,content 只
    走 ILIKE 加成),ILIKE 命中给保底分。"""
    pattern = f"%{q}%"
    tags_txt = func.cast(KnowledgeArticle.tags, String)
    return (
        2.0 * func.similarity(KnowledgeArticle.title, q)
        + 0.6 * func.similarity(tags_txt, q)
        + case((KnowledgeArticle.title.ilike(pattern), 1.0), else_=0.0)
        + case((KnowledgeArticle.content.ilike(pattern), 0.4), else_=0.0)
    )


def _embed_source(a: KnowledgeArticle) -> str:
    return f"{a.title}\n{a.content[:2000]}"


def _refresh_embedding(db: Session, a: KnowledgeArticle) -> None:
    """尽力为文章算 embedding;provider 不可用/失败静默跳过,不阻塞写。

    失败时保留旧向量 — 一次 provider 抖动不该把可用的检索索引清掉。
    """
    try:
        emb = embed_text(db, a.tenant_id, _embed_source(a))
    except Exception:
        return
    if emb:
        a.embedding = emb


def _hybrid_search(
    db: Session, tenant_id: str, q: str, limit: int, *, published_only: bool,
    category: str | None = None,
) -> list[tuple[KnowledgeArticle, float]]:
    """召回 = 词面命中(trgm/ILIKE) ∪ 语义池(embedding 可用时全量带向量文章);
    排序 = BM25(应用层,CJK bigram,字段加权) + 0.8*cosine(embedding)。
    无 embedding provider 时自动退化为纯 BM25。
    ``category`` 非空时限定分类(product_help 只查产品文档,不与租户知识混排)。"""
    lex = _lex_score(q).label("lex")
    base = select(KnowledgeArticle, lex).where(KnowledgeArticle.tenant_id == tenant_id)
    if published_only:
        base = base.where(KnowledgeArticle.status == "published")
    if category:
        base = base.where(KnowledgeArticle.category == category)
    pattern = f"%{q}%"
    stmt = base.where(
        or_(
            lex > 0.05,
            KnowledgeArticle.title.ilike(pattern),
            KnowledgeArticle.content.ilike(pattern),
            func.cast(KnowledgeArticle.tags, String).ilike(pattern),
        )
    ).order_by(lex.desc()).limit(50)
    rows = db.execute(stmt).all()
    pool: dict[str, tuple[KnowledgeArticle, float]] = {a.id: (a, float(lx or 0)) for a, lx in rows}

    q_emb = embed_text(db, tenant_id, q)
    if q_emb:
        # 语义召回:词面完全没命中但向量近的文章也要进池(否则同义词类查询永远召回不了)
        emb_stmt = base.where(KnowledgeArticle.embedding.is_not(None))
        if pool:
            emb_stmt = emb_stmt.where(KnowledgeArticle.id.notin_(list(pool)))
        emb_stmt = emb_stmt.order_by(KnowledgeArticle.updated_at.desc()).limit(200)
        for a in db.scalars(emb_stmt).all():
            pool[a.id] = (a, 0.0)
    if not pool:
        return []

    # 排序分:BM25(全语料 IDF,候选打分) — 无词面命中的语义召回项自然得 0
    corpus = db.scalars(
        select(KnowledgeArticle)
        .where(KnowledgeArticle.tenant_id == tenant_id)
        .where(KnowledgeArticle.status == "published")
        .limit(500)
    ).all()
    bm = bm25_scores(
        q,
        [(a.id, a.title, a.tags, a.content) for a in corpus],
        candidate_ids=set(pool),
    )
    scored = [
        (a, bm.get(a.id, 0.0) + (0.8 * cosine(q_emb, a.embedding) if q_emb else 0.0))
        for a, lx in pool.values()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _locate(content: str, q: str) -> tuple[int | None, int | None, str | None]:
    """P2 命中定位:(char_offset, paragraph_idx, 最近的 markdown 小节标题)。

    段落按空行切分(无空行则退化为按行);section 取命中位置之前
    最后一个 `#` 开头标题行 — 让「引用」可回溯到文章具体小节。
    """
    if not q:
        return None, None, None
    idx = content.lower().find(q.lower())
    if idx < 0:
        return None, None, None
    before = content[:idx]
    para = before.count("\n\n") + 1
    if para == 1 and "\n" in before:
        para = before.count("\n") + 1
    section: str | None = None
    for line in before.splitlines():
        ls = line.strip()
        if ls.startswith("#"):
            section = ls.lstrip("#").strip() or None
    return idx, para, section


def _hit(a: KnowledgeArticle, q: str, score: float | None = None) -> SearchHit:
    idx = a.content.lower().find(q.lower()) if q else -1
    if idx >= 0:
        start = max(0, idx - 40)
        snippet = a.content[start : idx + len(q) + 120].strip()
        if start > 0:
            snippet = "…" + snippet
        match_field = "content"
    else:
        snippet = a.content[:160].strip()
        match_field = (
            "title" if q and q.lower() in a.title.lower() else ("semantic" if q else None)
        )
    char_off, para, section = _locate(a.content, q)
    return SearchHit(
        id=a.id,
        title=a.title,
        category=a.category,
        tags=a.tags,
        snippet=snippet,
        score=score,
        match_field=match_field,
        char_offset=char_off,
        paragraph=para,
        section=section,
    )


@router.post("/search", response_model=list[SearchHit])
def search(
    body: SearchIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    rows = _hybrid_search(
        db, current.tenant_id, body.q, body.limit,
        published_only=True, category=body.category,
    )
    return [_hit(a, body.q, round(s, 4)) for a, s in rows]


@me_router.get("", response_model=list[SearchHit])
def me_search(
    q: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not q or not q.strip():
        rows = db.scalars(
            select(KnowledgeArticle)
            .where(
                KnowledgeArticle.tenant_id == current.tenant_id,
                KnowledgeArticle.status == "published",
            )
            .order_by(KnowledgeArticle.updated_at.desc())
            .limit(10)
        ).all()
        return [_hit(a, "") for a in rows]
    q = q.strip()[:128]
    rows = _hybrid_search(db, current.tenant_id, q, 10, published_only=True)
    return [_hit(a, q, round(s, 4)) for a, s in rows]


@router.post("/embeddings/backfill")
def backfill_embeddings(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """为还没有 embedding 的文章批量补齐(每次最多 50 篇,幂等可重复调)。"""
    rows = db.scalars(
        select(KnowledgeArticle)
        .where(KnowledgeArticle.tenant_id == current.tenant_id, KnowledgeArticle.embedding.is_(None))
        .limit(50)
    ).all()
    if not rows:
        return {"embedded": 0, "remaining": 0}
    vectors = embed_texts(db, current.tenant_id, [_embed_source(a) for a in rows])
    embedded = 0
    if vectors:
        # provider 返回行数可能比输入少 — 按较短侧对齐,不炸整个请求
        for a, v in zip(rows, vectors, strict=False):
            a.embedding = v
            embedded += 1
        db.commit()
    remaining = db.scalar(
        select(func.count(KnowledgeArticle.id)).where(
            KnowledgeArticle.tenant_id == current.tenant_id,
            KnowledgeArticle.embedding.is_(None),
        )
    ) or 0
    return {"embedded": embedded, "remaining": remaining}
