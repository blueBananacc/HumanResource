"""知识库 API(SuiteCRM AOK_KnowledgeBase 移植)—— assistant RAG 语料。

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
    updated_by: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SearchIn(BaseModel):
    q: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    id: str
    title: str
    category: str
    tags: list[str]
    snippet: str


# ============ 管理端 ============


@router.get("/articles", response_model=PageOut[ArticleOut])
def list_articles(
    status_filter: str | None = None,
    category: str | None = None,
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
    stmt = apply_q_ilike(stmt, q, KnowledgeArticle.title, KnowledgeArticle.content)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(KnowledgeArticle.updated_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(items=list(rows), total=total, limit=limit, offset=offset)


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
        updated_by=current.id,
    )
    db.add(a)
    db.flush()
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
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    a.updated_by = current.id
    a.updated_at = _now()
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
    write_audit(db, actor=current, entity_type="knowledge_article", entity_id=a.id, action="delete", request=request)
    db.delete(a)
    db.commit()


# ============ 检索(assistant RAG + ESS)============


def _search_stmt(tenant_id: str, q: str, *, published_only: bool):
    pattern = f"%{q}%"
    stmt = (
        select(KnowledgeArticle)
        .where(KnowledgeArticle.tenant_id == tenant_id)
        .where(
            or_(
                KnowledgeArticle.title.ilike(pattern),
                KnowledgeArticle.content.ilike(pattern),
                KnowledgeArticle.category.ilike(pattern),
                func.cast(KnowledgeArticle.tags, String).ilike(pattern),
            )
        )
    )
    if published_only:
        stmt = stmt.where(KnowledgeArticle.status == "published")
    # title 命中排前,category 次之,content 最后
    rank = case(
        (KnowledgeArticle.title.ilike(pattern), 0),
        (KnowledgeArticle.category.ilike(pattern), 1),
        else_=2,
    )
    return stmt.order_by(rank, KnowledgeArticle.updated_at.desc())


def _hit(a: KnowledgeArticle, q: str) -> SearchHit:
    idx = a.content.lower().find(q.lower())
    if idx >= 0:
        start = max(0, idx - 40)
        snippet = a.content[start : idx + len(q) + 120].strip()
        if start > 0:
            snippet = "…" + snippet
    else:
        snippet = a.content[:160].strip()
    return SearchHit(id=a.id, title=a.title, category=a.category, tags=a.tags, snippet=snippet)


@router.post("/search", response_model=list[SearchHit])
def search(
    body: SearchIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    rows = db.scalars(_search_stmt(current.tenant_id, body.q, published_only=True).limit(body.limit)).all()
    return [_hit(a, body.q) for a in rows]


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
    rows = db.scalars(_search_stmt(current.tenant_id, q, published_only=True).limit(10)).all()
    return [_hit(a, q) for a in rows]
