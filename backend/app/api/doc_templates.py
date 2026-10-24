"""文档模板管理 API(T6-4)。

- ``GET  /doc-templates``           合并视图:内置注册表 → 覆盖或默认 + 自定义行
- ``PUT  /doc-templates/{key}``     保存租户模板(内置 key = 覆盖;`custom.*` = 新建)
- ``POST /doc-templates/{key}/reset`` 删除租户行 → 回内置默认(仅内置 key)
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import DocTemplate, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.doc_templates import DOC_SPECS
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission

router = APIRouter(
    prefix="/doc-templates",
    tags=["doc-templates"],
    dependencies=[Depends(require_feature("assistant.agent"))],
)

_DOC_ENTITIES = ("any", "candidates", "employees")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DocTemplateOut(BaseModel):
    key: str
    name: str
    entity: str
    body: str
    is_custom: bool
    is_builtin: bool
    variables: list[dict]
    updated_at: datetime | None
    updated_by: str | None


class DocTemplateSaveIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    entity: str = Field("any")
    body: str = Field(..., min_length=10, max_length=20000)


def _row(db: Session, tenant_id: str, key: str) -> DocTemplate | None:
    return db.scalars(
        select(DocTemplate).where(
            DocTemplate.tenant_id == tenant_id, DocTemplate.key == key
        )
    ).first()


def _out(key: str, name: str, entity: str, body: str, *,
         is_custom: bool, is_builtin: bool,
         row: DocTemplate | None) -> DocTemplateOut:
    spec = DOC_SPECS.get(key)
    return DocTemplateOut(
        key=key, name=name, entity=entity, body=body,
        is_custom=is_custom, is_builtin=is_builtin,
        variables=spec.variables if spec else [],
        updated_at=row.updated_at if row else None,
        updated_by=row.updated_by if row else None,
    )


@router.get("", response_model=list[DocTemplateOut])
def list_doc_templates(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    rows = {
        r.key: r
        for r in db.scalars(
            select(DocTemplate).where(DocTemplate.tenant_id == current.tenant_id)
        )
    }
    out: list[DocTemplateOut] = []
    for key, spec in DOC_SPECS.items():
        r = rows.pop(key, None)
        out.append(_out(
            key, r.name if r else spec.name, r.entity if r else spec.entity,
            r.body if r else spec.body,
            is_custom=r is not None, is_builtin=True, row=r,
        ))
    for key, r in rows.items():
        out.append(_out(key, r.name, r.entity, r.body,
                        is_custom=True, is_builtin=False, row=r))
    return out


@router.put("/{key}", response_model=DocTemplateOut)
def save_doc_template(
    key: str,
    body: DocTemplateSaveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    if body.entity not in _DOC_ENTITIES:
        raise HTTPException(422, f"entity 须为 {'/'.join(_DOC_ENTITIES)}")
    tpl = _row(db, current.tenant_id, key)
    try:
        if tpl is None:
            tpl = DocTemplate(
                tenant_id=current.tenant_id, key=key, name=body.name,
                entity=body.entity, body=body.body, updated_by=current.id,
            )
            db.add(tpl)
            db.flush()
        else:
            tpl.name = body.name
            tpl.entity = body.entity
            tpl.body = body.body
            tpl.is_active = True
            tpl.updated_by = current.id
            tpl.updated_at = _now()
        write_audit(
            db, actor=current, entity_type="doc_template", entity_id=tpl.id,
            action="save", detail={"key": key}, request=request,
        )
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "并发保存冲突,请重试")
    db.refresh(tpl)
    spec = DOC_SPECS.get(key)
    return _out(key, tpl.name, tpl.entity, tpl.body, is_custom=True,
                is_builtin=spec is not None, row=tpl)


@router.post("/{key}/reset", response_model=DocTemplateOut)
def reset_doc_template(
    key: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = DOC_SPECS.get(key)
    if spec is None:
        raise HTTPException(404, "非内置模板请用 DELETE 删除")
    tpl = _row(db, current.tenant_id, key)
    if tpl:
        write_audit(
            db, actor=current, entity_type="doc_template", entity_id=tpl.id,
            action="reset", detail={"key": key}, request=request,
        )
        db.delete(tpl)
        db.commit()
    return _out(key, spec.name, spec.entity, spec.body,
                is_custom=False, is_builtin=True, row=None)


@router.delete("/{key}", status_code=204)
def delete_doc_template(
    key: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    if key in DOC_SPECS:
        raise HTTPException(400, "内置模板不能删除,请用 reset 恢复默认")
    tpl = _row(db, current.tenant_id, key)
    if tpl is None:
        raise HTTPException(404, "模板不存在")
    write_audit(
        db, actor=current, entity_type="doc_template", entity_id=tpl.id,
        action="delete", detail={"key": key}, request=request,
    )
    db.delete(tpl)
    db.commit()
