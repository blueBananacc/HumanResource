"""AI 提示词模板管理 API(设置 → AI 提示词)。

- ``GET  /prompt-templates``               合并视图:注册表每 key → 自定义或默认
- ``PUT  /prompt-templates/{key}``         保存自定义(版本 +1,写版本表)
- ``POST /prompt-templates/{key}/preview`` 用样例变量渲染(不落库)
- ``POST /prompt-templates/{key}/reset``   删除自定义 → 回默认
- ``GET  /prompt-templates/{key}/versions``历史版本列表
- ``POST /prompt-templates/{key}/rollback``回滚到指定版本(仍写新版本)
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import PromptTemplate, PromptTemplateVersion, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission
from app.services.prompting import PROMPT_REGISTRY, render_vars

router = APIRouter(prefix="/prompt-templates", tags=["prompt-templates"])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TemplateOut(BaseModel):
    key: str
    name: str
    description: str
    variables: list[dict]
    content: str
    is_custom: bool
    version: int | None
    updated_at: datetime | None
    updated_by: str | None


class TemplateSaveIn(BaseModel):
    content: str = Field(..., min_length=10, max_length=20000)


class PreviewIn(BaseModel):
    vars: dict[str, str] = Field(default_factory=dict)


class PreviewOut(BaseModel):
    rendered: str


class VersionOut(BaseModel):
    id: str
    version: int
    content: str
    created_by: str | None
    created_at: datetime


class RollbackIn(BaseModel):
    version: int


def _spec_or_404(key: str):
    spec = PROMPT_REGISTRY.get(key)
    if spec is None:
        raise HTTPException(404, f"未知的提示词用途: {key}")
    return spec


def _custom(db: Session, tenant_id: str, key: str) -> PromptTemplate | None:
    return db.scalars(
        select(PromptTemplate).where(
            PromptTemplate.tenant_id == tenant_id,
            PromptTemplate.key == key,
        )
    ).first()


@router.get("", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    customs = {
        t.key: t
        for t in db.scalars(
            select(PromptTemplate).where(PromptTemplate.tenant_id == current.tenant_id)
        )
    }
    out = []
    for key, spec in PROMPT_REGISTRY.items():
        c = customs.get(key)
        out.append(
            TemplateOut(
                key=key,
                name=spec.name,
                description=spec.description,
                variables=spec.variables,
                content=c.content if c else spec.content,
                is_custom=c is not None,
                version=c.version if c else None,
                updated_at=c.updated_at if c else None,
                updated_by=c.updated_by if c else None,
            )
        )
    return out


@router.put("/{key}", response_model=TemplateOut)
def save_template(
    key: str,
    body: TemplateSaveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl is None:
        tpl = PromptTemplate(
            tenant_id=current.tenant_id, key=key, content=body.content,
            version=1, updated_by=current.id,
        )
        db.add(tpl)
        db.flush()
    else:
        tpl.content = body.content
        tpl.version += 1
        tpl.is_active = True
        tpl.updated_by = current.id
        tpl.updated_at = _now()
    db.add(
        PromptTemplateVersion(
            tenant_id=current.tenant_id,
            template_id=tpl.id,
            version=tpl.version,
            content=tpl.content,
            created_by=current.id,
        )
    )
    write_audit(
        db, actor=current, entity_type="prompt_template", entity_id=tpl.id,
        action="save", detail={"key": key, "version": tpl.version},
        request=request,
    )
    db.commit()
    db.refresh(tpl)
    return TemplateOut(
        key=key, name=spec.name, description=spec.description,
        variables=spec.variables, content=tpl.content, is_custom=True,
        version=tpl.version, updated_at=tpl.updated_at, updated_by=tpl.updated_by,
    )


@router.post("/{key}/preview", response_model=PreviewOut)
def preview_template(
    key: str,
    body: PreviewIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    content = tpl.content if tpl else spec.content
    # 预览时给未提供的声明变量塞示例值,让 HR 看到完整渲染效果
    sample = {v["name"]: f"<{v['name']}>" for v in spec.variables}
    sample.update(body.vars)
    return PreviewOut(rendered=render_vars(content, sample))


@router.post("/{key}/reset", response_model=TemplateOut)
def reset_template(
    key: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl:
        write_audit(
            db, actor=current, entity_type="prompt_template", entity_id=tpl.id,
            action="reset", detail={"key": key}, request=request,
        )
        db.delete(tpl)
        db.commit()
    return TemplateOut(
        key=key, name=spec.name, description=spec.description,
        variables=spec.variables, content=spec.content, is_custom=False,
        version=None, updated_at=None, updated_by=None,
    )


@router.get("/{key}/versions", response_model=list[VersionOut])
def list_versions(
    key: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl is None:
        return []
    return [
        VersionOut(
            id=v.id, version=v.version, content=v.content,
            created_by=v.created_by, created_at=v.created_at,
        )
        for v in db.scalars(
            select(PromptTemplateVersion)
            .where(PromptTemplateVersion.template_id == tpl.id)
            .order_by(PromptTemplateVersion.version.desc())
        )
    ]


@router.post("/{key}/rollback", response_model=TemplateOut)
def rollback_template(
    key: str,
    body: RollbackIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl is None:
        raise HTTPException(404, "没有自定义版本可回滚")
    ver = db.scalars(
        select(PromptTemplateVersion).where(
            PromptTemplateVersion.template_id == tpl.id,
            PromptTemplateVersion.version == body.version,
        )
    ).first()
    if ver is None:
        raise HTTPException(404, f"不存在版本 v{body.version}")
    tpl.content = ver.content
    tpl.version += 1
    tpl.updated_by = current.id
    tpl.updated_at = _now()
    db.add(
        PromptTemplateVersion(
            tenant_id=current.tenant_id, template_id=tpl.id,
            version=tpl.version, content=tpl.content, created_by=current.id,
        )
    )
    write_audit(
        db, actor=current, entity_type="prompt_template", entity_id=tpl.id,
        action="rollback", detail={"key": key, "to_version": body.version},
        request=request,
    )
    db.commit()
    db.refresh(tpl)
    return TemplateOut(
        key=key, name=spec.name, description=spec.description,
        variables=spec.variables, content=tpl.content, is_custom=True,
        version=tpl.version, updated_at=tpl.updated_at, updated_by=tpl.updated_by,
    )
