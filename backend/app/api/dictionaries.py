"""可配置下拉枚举。

- 读 ``GET /dictionaries/{key}`` — 任何登录用户(表单下拉用),只回激活项。
- 写 ``/dictionaries`` CRUD — ``manage_settings``。
- :func:`kind_values` 给 requests/cases/skills 的 kind 校验用:
  字典非空 → 字典激活值;空 → 内置默认(后向兼容,老数据不破)。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import DictionaryItem, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission

router = APIRouter(prefix="/dictionaries", tags=["dictionaries"])

# 可配置 dict_key 注册表 → 内置默认值(字典空时回退)。
DICT_KEYS: dict[str, tuple[str, ...]] = {
    "request_kind": ("general", "onboarding", "benefit", "equipment", "other"),
    "case_kind": ("payroll", "benefit", "relation", "policy", "other"),
    "skill_category": ("tech", "language", "soft", "cert", "other"),
}


def kind_values(db: Session, tenant_id: str, dict_key: str, builtin: tuple[str, ...]) -> set[str]:
    """kind 校验允许值:字典有激活项 → 字典值∪内置;空 → 内置。"""
    vals = set(
        db.scalars(
            select(DictionaryItem.value).where(
                DictionaryItem.tenant_id == tenant_id,
                DictionaryItem.dict_key == dict_key,
                DictionaryItem.is_active.is_(True),
            )
        )
    )
    return vals | set(builtin)


class DictItemIn(BaseModel):
    dict_key: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=128)
    sort: int = 0
    is_active: bool = True


class DictItemPatch(BaseModel):
    label: str | None = Field(default=None, max_length=128)
    sort: int | None = None
    is_active: bool | None = None


class DictItemOut(BaseModel):
    id: str
    dict_key: str
    value: str
    label: str
    sort: int
    is_active: bool


def _out(d: DictionaryItem) -> DictItemOut:
    return DictItemOut(
        id=d.id, dict_key=d.dict_key, value=d.value,
        label=d.label, sort=d.sort, is_active=d.is_active,
    )


@router.get("/keys")
def list_keys(current: User = Depends(get_current_user)):
    return [{"key": k, "builtin": list(v)} for k, v in DICT_KEYS.items()]


@router.get("/{dict_key}/options")
def options(
    dict_key: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """下拉数据源:字典激活项(空则回内置)。"""
    builtin = DICT_KEYS.get(dict_key)
    if builtin is None:
        raise HTTPException(404, "未知字典")
    rows = db.scalars(
        select(DictionaryItem)
        .where(
            DictionaryItem.tenant_id == current.tenant_id,
            DictionaryItem.dict_key == dict_key,
            DictionaryItem.is_active.is_(True),
        )
        .order_by(DictionaryItem.sort, DictionaryItem.label)
    ).all()
    items = [{"value": r.value, "label": r.label} for r in rows]
    if not items:
        items = [{"value": v, "label": v} for v in builtin]
    return items


@router.get("", response_model=list[DictItemOut])
def list_items(
    dict_key: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    stmt = (
        select(DictionaryItem)
        .where(DictionaryItem.tenant_id == current.tenant_id)
        .order_by(DictionaryItem.dict_key, DictionaryItem.sort, DictionaryItem.label)
    )
    if dict_key:
        if dict_key not in DICT_KEYS:
            raise HTTPException(404, "未知字典")
        stmt = stmt.where(DictionaryItem.dict_key == dict_key)
    return [_out(d) for d in db.scalars(stmt)]


@router.post("", response_model=DictItemOut, status_code=201)
def create_item(
    body: DictItemIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    if body.dict_key not in DICT_KEYS:
        raise HTTPException(400, f"未知字典,可用:{','.join(DICT_KEYS)}")
    d = DictionaryItem(
        tenant_id=current.tenant_id,
        dict_key=body.dict_key,
        value=body.value.strip(),
        label=body.label.strip(),
        sort=body.sort,
        is_active=body.is_active,
    )
    db.add(d)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "该值已存在")
    write_audit(
        db, actor=current, entity_type="dictionary_item", entity_id=d.id,
        action="create", detail={"dict_key": d.dict_key, "value": d.value},
        request=request,
    )
    db.commit()
    return _out(d)


@router.patch("/{did}", response_model=DictItemOut)
def update_item(
    did: str,
    body: DictItemPatch,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    d = db.get(DictionaryItem, did)
    if d is None or d.tenant_id != current.tenant_id:
        raise HTTPException(404, "字典项不存在")
    changed: dict = {}
    if body.label is not None and body.label.strip():
        d.label = body.label.strip()
        changed["label"] = d.label
    if body.sort is not None:
        d.sort = body.sort
        changed["sort"] = d.sort
    if body.is_active is not None:
        d.is_active = body.is_active
        changed["is_active"] = d.is_active
    if changed:
        write_audit(
            db, actor=current, entity_type="dictionary_item", entity_id=did,
            action="update", detail={"fields": changed}, request=request,
        )
    db.commit()
    return _out(d)


@router.delete("/{did}")
def delete_item(
    did: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    d = db.get(DictionaryItem, did)
    if d is None or d.tenant_id != current.tenant_id:
        raise HTTPException(404, "字典项不存在")
    db.delete(d)
    write_audit(
        db, actor=current, entity_type="dictionary_item", entity_id=did,
        action="delete", detail={"dict_key": d.dict_key, "value": d.value},
        request=request,
    )
    db.commit()
    return {"ok": True}
