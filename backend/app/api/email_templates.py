"""邮件模板管理 API(设置 → 邮件模板)。

- ``GET  /email-templates``               合并视图:注册表每 key → 自定义或默认
- ``PUT  /email-templates/{key}``         保存 subject/body_text/body_html
- ``POST /email-templates/{key}/preview`` 样例渲染(不落库)
- ``POST /email-templates/{key}/reset``   删除自定义 → 回默认
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import EmailTemplate, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.mail_templates import EMAIL_SPECS, _rendered_vars
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission
from app.services.prompting import render_vars

router = APIRouter(prefix="/email-templates", tags=["email-templates"])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TemplateOut(BaseModel):
    key: str
    name: str
    description: str
    variables: list[dict]
    subject: str
    body_text: str
    body_html: str | None
    is_custom: bool
    updated_at: datetime | None


class TemplateSaveIn(BaseModel):
    subject: str = Field(..., min_length=1, max_length=256)
    body_text: str = Field(..., min_length=10, max_length=20000)
    body_html: str | None = Field(None, max_length=40000)


class PreviewIn(BaseModel):
    subject: str | None = None
    body_text: str | None = None
    body_html: str | None = None
    vars: dict[str, str] = Field(default_factory=dict)


class PreviewOut(BaseModel):
    subject: str
    text: str
    html: str | None


def _spec_or_404(key: str):
    spec = EMAIL_SPECS.get(key)
    if spec is None:
        raise HTTPException(404, f"未知的邮件模板用途: {key}")
    return spec


def _custom(db: Session, tenant_id: str, key: str) -> EmailTemplate | None:
    return db.scalars(
        select(EmailTemplate).where(
            EmailTemplate.tenant_id == tenant_id,
            EmailTemplate.key == key,
        )
    ).first()


def _out(key: str, tpl: EmailTemplate | None) -> TemplateOut:
    spec = EMAIL_SPECS[key]
    return TemplateOut(
        key=key,
        name=spec.name,
        description=spec.description,
        variables=spec.variables,
        subject=tpl.subject if tpl else spec.subject,
        body_text=tpl.body_text if tpl else spec.body_text,
        body_html=tpl.body_html if tpl else spec.body_html,
        is_custom=tpl is not None,
        updated_at=tpl.updated_at if tpl else None,
    )


@router.get("", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    customs = {
        t.key: t
        for t in db.scalars(
            select(EmailTemplate).where(EmailTemplate.tenant_id == current.tenant_id)
        )
    }
    return [_out(key, customs.get(key)) for key in EMAIL_SPECS]


@router.put("/{key}", response_model=TemplateOut)
def save_template(
    key: str,
    body: TemplateSaveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl is None:
        tpl = EmailTemplate(tenant_id=current.tenant_id, key=key)
        db.add(tpl)
    tpl.subject = body.subject
    tpl.body_text = body.body_text
    tpl.body_html = body.body_html
    tpl.is_active = True
    tpl.updated_by = current.id
    tpl.updated_at = _now()
    write_audit(
        db, actor=current, entity_type="email_template", entity_id=tpl.id or key,
        action="save", detail={"key": key}, request=request,
    )
    db.commit()
    db.refresh(tpl)
    return _out(key, tpl)


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
    subject = body.subject if body.subject is not None else (tpl.subject if tpl else spec.subject)
    text = body.body_text if body.body_text is not None else (tpl.body_text if tpl else spec.body_text)
    html_body = body.body_html if body.body_html is not None else (tpl.body_html if tpl else spec.body_html)
    sample = {v["name"]: f"<{v['name']}>" for v in spec.variables}
    sample.update(body.vars)
    return PreviewOut(
        subject=render_vars(subject, sample),
        text=render_vars(text, sample),
        html=(
            render_vars(html_body, _rendered_vars(sample, escape=True))
            if html_body
            else None
        ),
    )


@router.post("/{key}/reset", response_model=TemplateOut)
def reset_template(
    key: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    _spec_or_404(key)
    tpl = _custom(db, current.tenant_id, key)
    if tpl:
        write_audit(
            db, actor=current, entity_type="email_template", entity_id=tpl.id,
            action="reset", detail={"key": key}, request=request,
        )
        db.delete(tpl)
        db.commit()
    return _out(key, None)
