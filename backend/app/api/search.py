"""声明式全局搜索 API(T9)。

- ``GET /api/search`` — 跨实体检索;命中先经 ENTITIES 的 perm/feature 过滤,
  再用各实体 ``build()`` 子查询按 id 集做行级可见性回读(投影不权威)
- ``GET /api/search/facets`` — 各实体命中数
- ``GET/PATCH /api/search/definitions`` — 租户级定义覆盖(manage_settings)
- ``POST /api/search/reindex`` / ``GET /api/search/index-status`` — 运维
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.entities import ENTITIES
from app.domain.models import SearchIndexDefinition, User
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.services import search_index as svc
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_SETTINGS,
    get_org_scope,
    has_permission,
    require_permission,
)

router = APIRouter(prefix="/search", tags=["search"])

MAX_Q = 128


class SearchHit(BaseModel):
    entity: str
    entity_label: str
    id: str
    title: str
    sub: str | None = None
    link: str
    score: int


class SearchOut(BaseModel):
    items: list[SearchHit]
    facets: list[dict[str, Any]]
    total: int


def _visible_ids(
    db: Session, current: User, scope, entity: str, ids: list[str]
) -> dict[str, Any]:
    """经 ENTITIES[entity].build() 回读:返回 {id: row}(label/secondary)。

    build() 自带 perm 口径内的行级 scope(org / 汇报线),命中文档在这里
    被收敛成真实可见集合 —— 索引永远不决定权限。
    """
    spec = ENTITIES.get(entity)
    if spec is None:
        return {}
    if not has_permission(current, spec["perm"]):
        return {}
    if spec.get("feature") and not is_feature_enabled(db, spec["feature"]):
        return {}
    stmt = spec["build"](db, current, scope, None)
    sub = stmt.subquery()
    rows = db.execute(select(sub).where(sub.c.id.in_(ids))).all()
    return {r.id: r for r in rows}


@router.get("", response_model=SearchOut)
def search(
    q: str = Query(..., min_length=1, max_length=MAX_Q),
    entity: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if entity is not None and entity not in svc.SEARCH_SOURCES:
        raise HTTPException(404, "未知实体")
    scope = get_org_scope(db, current)
    hits = svc.doc_hits(db, current.tenant_id, q, entity=entity)
    # 按实体分组做行级可见性回读
    by_entity: dict[str, list[tuple[Any, int]]] = {}
    for doc, score in hits:
        by_entity.setdefault(doc.entity, []).append((doc, score))

    items: list[SearchHit] = []
    facets: list[dict[str, Any]] = []
    for key, group in by_entity.items():
        visible = _visible_ids(db, current, scope, key, [d.entity_id for d, _ in group])
        kept = [(d, s, visible[d.entity_id]) for d, s in group if d.entity_id in visible]
        if not kept:
            continue
        spec = svc.SEARCH_SOURCES[key]
        for d, s, row in kept:
            items.append(
                SearchHit(
                    entity=key,
                    entity_label=spec["label"],
                    id=d.entity_id,
                    title=d.title,
                    sub=getattr(row, "secondary", None),
                    link=spec["link"].format(id=d.entity_id),
                    score=s,
                )
            )
        facets.append(
            {"entity": key, "label": spec["label"], "count": len(kept)}
        )
    items.sort(key=lambda h: h.score, reverse=True)
    return SearchOut(items=items[:limit], facets=facets, total=len(items))


@router.get("/facets")
def search_facets(
    q: str = Query(..., min_length=1, max_length=MAX_Q),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    scope = get_org_scope(db, current)
    hits = svc.doc_hits(db, current.tenant_id, q)
    by_entity: dict[str, list[tuple[Any, int]]] = {}
    for doc, score in hits:
        by_entity.setdefault(doc.entity, []).append((doc, score))
    out = []
    for key, group in by_entity.items():
        visible = _visible_ids(db, current, scope, key, [d.entity_id for d, _ in group])
        n = sum(1 for d, _ in group if d.entity_id in visible)
        if n:
            out.append(
                {"entity": key, "label": svc.SEARCH_SOURCES[key]["label"], "count": n}
            )
    return out


# ---------------- 定义管理 / 重建(manage_settings) ----------------


@router.get(
    "/definitions",
    dependencies=[Depends(require_permission(PERM_MANAGE_SETTINGS))],
)
def list_definitions(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    out = []
    for key in svc.SEARCH_SOURCES:
        spec = svc.SEARCH_SOURCES[key]
        base = svc.effective_definition(db, current.tenant_id, key)
        out.append(
            {
                "entity": key,
                "label": (base or {}).get("label", spec["label"]),
                "is_active": base is not None,
                "fields": (base or {}).get("fields", spec["fields"]),
                "title_template": (base or {}).get("title", spec["title"]),
                "body_template": (base or {}).get("body", spec["body"]),
                "keyword_fields": (base or {}).get(
                    "keyword_fields", spec.get("keyword_fields") or []
                ),
                "overridden": (base or {}).get("overridden", False),
                "version": (base or {}).get("version", 0),
            }
        )
    return out


class DefinitionPatch(BaseModel):
    is_active: bool | None = None
    label: str | None = Field(None, max_length=128)
    fields: list[dict[str, Any]] | None = None
    title_template: str | None = Field(None, max_length=512)
    body_template: str | None = Field(None, max_length=1024)
    keyword_fields: list[str] | None = None


@router.patch(
    "/definitions/{entity}",
    dependencies=[Depends(require_permission(PERM_MANAGE_SETTINGS))],
)
def update_definition(
    entity: str,
    body: DefinitionPatch,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    spec = svc.SEARCH_SOURCES.get(entity)
    if spec is None:
        raise HTTPException(404, "未知实体")
    # 覆盖字段白名单:只允许调注册表已声明字段
    declared = set(spec["fields"])
    if body.fields is not None:
        for f in body.fields:
            if (f or {}).get("name") not in declared:
                raise HTTPException(400, f"字段不在白名单: {(f or {}).get('name')}")
    if body.keyword_fields is not None:
        for k in body.keyword_fields:
            if k not in declared:
                raise HTTPException(400, f"keyword 字段不在白名单: {k}")

    row = db.scalars(
        select(SearchIndexDefinition).where(
            SearchIndexDefinition.tenant_id == current.tenant_id,
            SearchIndexDefinition.entity == entity,
        )
    ).first()
    if row is None:
        row = SearchIndexDefinition(tenant_id=current.tenant_id, entity=entity)
        db.add(row)
    if body.is_active is not None:
        row.is_active = body.is_active
    if body.label is not None:
        row.label = body.label.strip() or None
    if body.fields is not None:
        row.fields = body.fields
    if body.title_template is not None:
        row.title_template = body.title_template.strip() or None
    if body.body_template is not None:
        row.body_template = body.body_template.strip() or None
    if body.keyword_fields is not None:
        row.keyword_fields = body.keyword_fields
    row.version = (row.version or 0) + 1
    write_audit(
        db, actor=current, entity_type="search_index_definition",
        entity_id=entity, action="update",
        detail={"fields": {"entity": entity}}, request=request,
    )
    db.commit()
    return svc.effective_definition(db, current.tenant_id, entity)


@router.post(
    "/reindex",
    dependencies=[Depends(require_permission(PERM_MANAGE_SETTINGS))],
)
def reindex(
    body: dict[str, Any],
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    entity = body.get("entity")
    if entity is not None and entity not in svc.SEARCH_SOURCES:
        raise HTTPException(404, "未知实体")
    result = svc.reindex(db, current.tenant_id, entity)
    write_audit(
        db, actor=current, entity_type="search_index", entity_id=entity or "*",
        action="reindex", detail=result, request=request,
    )
    db.commit()
    return result


@router.get(
    "/index-status",
    dependencies=[Depends(require_permission(PERM_MANAGE_SETTINGS))],
)
def get_index_status(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return svc.index_status(db, current.tenant_id)
