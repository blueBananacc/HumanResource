"""记录星标收藏(MintHCM Favorites 移植)。

行级隔离 = 仅本人;POST 时校验记录存在+租户+org scope(防探测式收藏)。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import Candidate, Employee, Favorite, HrCase, Job, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import get_org_scope

router = APIRouter(prefix="/favorites", tags=["favorites"])

# 可收藏实体白名单 → ORM 模型(无 org_unit_id 的实体不做 scope 校验)。
_FAV_ENTITIES = {
    "talents": Candidate,
    "jobs": Job,
    "employees": Employee,
    "cases": HrCase,
}


class FavoriteIn(BaseModel):
    entity: str = Field(min_length=1, max_length=32)
    entity_id: str = Field(min_length=1, max_length=36)


class FavoriteOut(BaseModel):
    id: str
    entity: str
    entity_id: str


def _check_target(db: Session, current: User, entity: str, entity_id: str) -> None:
    model = _FAV_ENTITIES.get(entity)
    if model is None:
        raise HTTPException(400, f"不支持收藏的实体:{entity};可用:{','.join(_FAV_ENTITIES)}")
    row = db.get(model, entity_id)
    if row is None or row.tenant_id != current.tenant_id:
        raise HTTPException(404, "记录不存在")
    if hasattr(row, "org_unit_id"):
        scope = get_org_scope(db, current)
        if scope is not None and row.org_unit_id not in scope:
            raise HTTPException(404, "记录不存在")


@router.get("", response_model=list[FavoriteOut])
def list_favorites(
    entity: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    stmt = (
        select(Favorite)
        .where(
            Favorite.tenant_id == current.tenant_id,
            Favorite.user_id == current.id,
        )
        .order_by(Favorite.created_at.desc())
    )
    if entity:
        stmt = stmt.where(Favorite.entity == entity)
    return [FavoriteOut(id=f.id, entity=f.entity, entity_id=f.entity_id) for f in db.scalars(stmt)]


@router.post("", response_model=FavoriteOut, status_code=201)
def add_favorite(
    body: FavoriteIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    _check_target(db, current, body.entity, body.entity_id)
    f = Favorite(
        tenant_id=current.tenant_id,
        user_id=current.id,
        entity=body.entity,
        entity_id=body.entity_id,
    )
    db.add(f)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "已收藏")
    write_audit(
        db,
        actor=current,
        entity_type="favorite",
        entity_id=f.id,
        action="create",
        detail={"entity": f.entity, "entity_id": f.entity_id},
        request=request,
    )
    db.commit()
    return FavoriteOut(id=f.id, entity=f.entity, entity_id=f.entity_id)


@router.delete("/{fid}")
def remove_favorite(
    fid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    f = db.get(Favorite, fid)
    if f is None or f.tenant_id != current.tenant_id or f.user_id != current.id:
        raise HTTPException(404, "收藏不存在")
    db.delete(f)
    write_audit(
        db,
        actor=current,
        entity_type="favorite",
        entity_id=fid,
        action="delete",
        detail={"entity": f.entity, "entity_id": f.entity_id},
        request=request,
    )
    db.commit()
    return {"ok": True}
