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
from sqlalchemy.exc import IntegrityError
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
    module: str
    variables: list[dict]
    content: str
    is_custom: bool
    # 该 key 的模块级覆盖数量(scope_module 非空的自定义行数)
    scoped_count: int
    version: int | None
    updated_at: datetime | None
    updated_by: str | None


class TemplateSaveIn(BaseModel):
    content: str = Field(..., min_length=10, max_length=20000)
    # 模块级覆盖:"" = 全局(默认);非空 = 仅该模块语境生效(两级回落)
    scope_module: str = Field(default="", max_length=48)


class PreviewIn(BaseModel):
    # 传 content 时预览编辑中的草稿;不传则预览库中已存(或默认)内容
    content: str | None = Field(None, max_length=20000)
    vars: dict[str, str] = Field(default_factory=dict)
    scope_module: str = Field(default="", max_length=48)


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
    scope_module: str = Field(default="", max_length=48)


def _spec_or_404(key: str):
    spec = PROMPT_REGISTRY.get(key)
    if spec is None:
        raise HTTPException(404, f"未知的提示词用途: {key}")
    return spec


def _custom(
    db: Session, tenant_id: str, key: str, scope_module: str = "", *,
    for_update: bool = False,
) -> PromptTemplate | None:
    stmt = select(PromptTemplate).where(
        PromptTemplate.tenant_id == tenant_id,
        PromptTemplate.key == key,
        PromptTemplate.scope_module == scope_module,
    )
    # 保存/回滚时锁行,序列化并发写入 — 否则 version += 1 是丢失更新,
    # 版本表会出现两个相同版本号
    if for_update:
        stmt = stmt.with_for_update()
    return db.scalars(stmt).first()


def _out(key: str, spec, c: PromptTemplate | None, scoped_count: int) -> TemplateOut:
    return TemplateOut(
        key=key,
        name=spec.name,
        description=spec.description,
        module=spec.module,
        variables=spec.variables,
        content=c.content if c else spec.content,
        is_custom=c is not None,
        scoped_count=scoped_count,
        version=c.version if c else None,
        updated_at=c.updated_at if c else None,
        updated_by=c.updated_by if c else None,
    )


@router.get("", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    rows = db.scalars(
        select(PromptTemplate).where(PromptTemplate.tenant_id == current.tenant_id)
    ).all()
    customs = {(t.key, t.scope_module): t for t in rows}
    scoped_counts: dict[str, int] = {}
    for t in rows:
        if t.scope_module:
            scoped_counts[t.key] = scoped_counts.get(t.key, 0) + 1
    out = []
    for key, spec in PROMPT_REGISTRY.items():
        c = customs.get((key, ""))
        out.append(_out(key, spec, c, scoped_counts.get(key, 0)))
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
    scope = body.scope_module.strip()
    tpl = _custom(db, current.tenant_id, key, scope, for_update=True)
    try:
        if tpl is None:
            tpl = PromptTemplate(
                tenant_id=current.tenant_id, key=key, scope_module=scope,
                content=body.content, version=1, updated_by=current.id,
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
            action="save",
            detail={"key": key, "scope_module": scope, "version": tpl.version},
            request=request,
        )
        db.commit()
    except IntegrityError:
        # 并发首建同一 key:唯一约束兜底成 409 而不是 500
        db.rollback()
        raise HTTPException(409, "并发保存冲突,请重试")
    db.refresh(tpl)
    return _out(key, spec, tpl, 0)


@router.post("/{key}/preview", response_model=PreviewOut)
def preview_template(
    key: str,
    body: PreviewIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key, body.scope_module.strip())
    content = (
        body.content
        if body.content is not None
        else (tpl.content if tpl else spec.content)
    )
    # 预览时给未提供的声明变量塞示例值,让 HR 看到完整渲染效果
    sample = {v["name"]: f"<{v['name']}>" for v in spec.variables}
    sample.update(body.vars)
    return PreviewOut(rendered=render_vars(content, sample))


@router.post("/{key}/reset", response_model=TemplateOut)
def reset_template(
    key: str,
    request: Request,
    scope_module: str = "",
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    spec = _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key, scope_module)
    if tpl:
        write_audit(
            db, actor=current, entity_type="prompt_template", entity_id=tpl.id,
            action="reset", detail={"key": key, "scope_module": scope_module},
            request=request,
        )
        db.delete(tpl)
        db.commit()
    return _out(key, spec, None, 0)


@router.get("/{key}/versions", response_model=list[VersionOut])
def list_versions(
    key: str,
    scope_module: str = "",
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key, scope_module)
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
    scope = body.scope_module.strip()
    tpl = _custom(db, current.tenant_id, key, scope, for_update=True)
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
    return _out(key, spec, tpl, 0)
