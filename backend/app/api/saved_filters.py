"""列表页保存筛选。

行级隔离 = 仅本人(user_id),不做组织穿透 —— 筛选是个人偏好。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import SavedFilter, User
from app.infra.db import get_db
from app.services.audit import write_audit

router = APIRouter(prefix="/saved-filters", tags=["saved-filters"])


class FilterIn(BaseModel):
    page: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    params: dict[str, Any] = Field(default_factory=dict)


class FilterOut(BaseModel):
    id: str
    page: str
    name: str
    params: dict[str, Any]


def _out(f: SavedFilter) -> FilterOut:
    return FilterOut(id=f.id, page=f.page, name=f.name, params=f.params or {})


@router.get("", response_model=list[FilterOut])
def list_filters(
    page: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    stmt = (
        select(SavedFilter)
        .where(
            SavedFilter.tenant_id == current.tenant_id,
            SavedFilter.user_id == current.id,
        )
        .order_by(SavedFilter.created_at)
    )
    if page:
        stmt = stmt.where(SavedFilter.page == page)
    return [_out(f) for f in db.scalars(stmt)]


@router.post("", response_model=FilterOut, status_code=201)
def create_filter(
    body: FilterIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    f = SavedFilter(
        tenant_id=current.tenant_id,
        user_id=current.id,
        page=body.page,
        name=body.name.strip(),
        params=body.params,
    )
    db.add(f)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "同名筛选已存在")
    write_audit(
        db,
        actor=current,
        entity_type="saved_filter",
        entity_id=f.id,
        action="create",
        detail={"page": f.page, "name": f.name},
        request=request,
    )
    db.commit()
    return _out(f)


@router.delete("/{fid}")
def delete_filter(
    fid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    f = db.get(SavedFilter, fid)
    if f is None or f.tenant_id != current.tenant_id or f.user_id != current.id:
        raise HTTPException(404, "筛选不存在")
    db.delete(f)
    write_audit(
        db,
        actor=current,
        entity_type="saved_filter",
        entity_id=fid,
        action="delete",
        detail={"page": f.page, "name": f.name},
        request=request,
    )
    db.commit()
    return {"ok": True}
