"""事件驱动自动化规则 API。

- ``GET/POST /automations``            列表/创建(manage_settings)
- ``PATCH/DELETE /automations/{id}``   编辑/启停/删除
- ``GET /automations/{id}/runs``       执行记录
- ``GET /automations/schema``          注册表元数据(实体/字段/算子/收件人)
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.domain.models import AutomationRule, AutomationRun, User
from app.infra.db import get_db
from app.services import automation as svc
from app.services.audit import write_audit
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission
from app.services.timeutil import utc_now

router = APIRouter(prefix="/automations", tags=["automations"])


class RuleOut(BaseModel):
    id: str
    name: str
    entity_type: str
    entity_label: str
    trigger_kind: str
    trigger: str
    schedule: dict[str, Any] | None
    watch_fields: list[str] | None
    conditions: list[dict[str, Any]]
    action: str
    action_config: dict[str, Any] | None
    recipients: list[dict[str, Any]]
    title_tpl: str
    body_tpl: str | None
    is_active: bool
    last_fired_at: Any
    created_at: Any


def _out(r: AutomationRule) -> RuleOut:
    spec = svc.ENTITY_REGISTRY.get(r.entity_type) or {}
    return RuleOut(
        id=r.id, name=r.name, entity_type=r.entity_type,
        entity_label=spec.get("label", r.entity_type),
        trigger_kind=r.trigger_kind or "audit",
        trigger=r.trigger, schedule=r.schedule,
        watch_fields=r.watch_fields,
        conditions=r.conditions or [], action=r.action,
        action_config=r.action_config,
        recipients=r.recipients or [], title_tpl=r.title_tpl,
        body_tpl=r.body_tpl, is_active=r.is_active,
        last_fired_at=r.last_fired_at, created_at=r.created_at,
    )


def _get_rule(db: Session, tenant_id: str, rule_id: str) -> AutomationRule:
    r = db.get(AutomationRule, rule_id)
    if r is None or r.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "规则不存在")
    return r


class RuleIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    entity_type: str = Field(min_length=1, max_length=64)
    trigger_kind: Literal["audit", "date_field"] = "audit"
    trigger: str | None = Field(default=None, max_length=64)
    schedule: dict[str, Any] | None = None
    watch_fields: list[str] | None = None
    conditions: list[dict[str, Any]] = []
    action: Literal["notify", "email", "create_case"] = "notify"
    action_config: dict[str, Any] | None = None
    recipients: list[dict[str, Any]] = []
    title_tpl: str = Field(min_length=1, max_length=256)
    body_tpl: str | None = Field(default=None, max_length=1024)
    is_active: bool = True


class RulePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    trigger_kind: Literal["audit", "date_field"] | None = None
    trigger: str | None = Field(default=None, max_length=64)
    schedule: dict[str, Any] | None = None
    watch_fields: list[str] | None = None
    conditions: list[dict[str, Any]] | None = None
    action: Literal["notify", "email", "create_case"] | None = None
    action_config: dict[str, Any] | None = None
    recipients: list[dict[str, Any]] | None = None
    title_tpl: str | None = Field(default=None, min_length=1, max_length=256)
    body_tpl: str | None = Field(default=None, max_length=1024)
    is_active: bool | None = None


@router.get("/schema")
def get_schema(current: User = Depends(require_permission(PERM_MANAGE_SETTINGS))):
    return svc.schema_meta()


@router.get("", response_model=PageOut[RuleOut])
def list_rules(
    pagination: tuple = Depends(paginate_params),
    entity_type: str | None = None,
    is_active: bool | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    limit, offset, _ = pagination
    stmt = select(AutomationRule).where(AutomationRule.tenant_id == current.tenant_id)
    if entity_type:
        stmt = stmt.where(AutomationRule.entity_type == entity_type)
    if is_active is not None:
        stmt = stmt.where(AutomationRule.is_active.is_(is_active))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(AutomationRule.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(items=[_out(r) for r in rows], total=total, limit=limit, offset=offset)


@router.post("", response_model=RuleOut, status_code=status.HTTP_201_CREATED)
def create_rule(
    body: RuleIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    trigger = body.trigger or (
        "date_field" if body.trigger_kind == "date_field" else "create")
    err = svc.validate_rule_payload(
        body.entity_type, trigger, body.conditions, body.recipients, body.action,
        trigger_kind=body.trigger_kind, schedule=body.schedule,
        action_config=body.action_config,
    )
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
    if body.action in ("notify", "email") and not body.recipients:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "通知类动作至少需要一个收件人")
    r = AutomationRule(
        tenant_id=current.tenant_id, name=body.name, entity_type=body.entity_type,
        trigger_kind=body.trigger_kind, trigger=trigger, schedule=body.schedule,
        watch_fields=body.watch_fields,
        conditions=body.conditions, action=body.action,
        action_config=body.action_config, recipients=body.recipients,
        title_tpl=body.title_tpl, body_tpl=body.body_tpl, is_active=body.is_active,
        created_by=current.id,
    )
    db.add(r)
    db.flush()
    write_audit(
        db, actor=current, entity_type="automation_rule", entity_id=r.id,
        action="create", detail={"name": r.name, "entity_type": r.entity_type},
        request=request,
    )
    db.commit()
    db.refresh(r)
    return _out(r)


@router.patch("/{rule_id}", response_model=RuleOut)
def update_rule(
    rule_id: str,
    body: RulePatch,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _get_rule(db, current.tenant_id, rule_id)
    data = body.model_dump(exclude_unset=True)
    entity_type = r.entity_type  # 实体不可改 —— 改实体请新建规则
    trigger_kind = data.get("trigger_kind", r.trigger_kind)
    trigger = data.get("trigger", r.trigger)
    schedule = data.get("schedule", r.schedule)
    conditions = data.get("conditions", r.conditions)
    recipients = data.get("recipients", r.recipients)
    action = data.get("action", r.action)
    action_config = data.get("action_config", r.action_config)
    err = svc.validate_rule_payload(
        entity_type, trigger, conditions, recipients, action,
        trigger_kind=trigger_kind, schedule=schedule,
        action_config=action_config,
    )
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
    changed = {}
    for k, v in data.items():
        if getattr(r, k) != v:
            setattr(r, k, v)
            changed[k] = v
    r.updated_at = utc_now()
    if changed:
        write_audit(
            db, actor=current, entity_type="automation_rule", entity_id=r.id,
            action="update", detail={"fields": changed}, request=request,
        )
    db.commit()
    db.refresh(r)
    return _out(r)


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rule(
    rule_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _get_rule(db, current.tenant_id, rule_id)
    write_audit(
        db, actor=current, entity_type="automation_rule", entity_id=r.id,
        action="delete", detail={"name": r.name}, request=request,
    )
    db.delete(r)
    db.commit()


class RunOut(BaseModel):
    id: str
    rule_id: str
    entity_type: str
    entity_id: str
    trigger: str
    action: str
    status: str
    recipients: list[Any] | None
    payload: dict[str, Any] | None
    error: str | None
    created_at: Any


@router.get("/{rule_id}/runs", response_model=PageOut[RunOut])
def list_runs(
    rule_id: str,
    pagination: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    _get_rule(db, current.tenant_id, rule_id)
    limit, offset, _ = pagination
    stmt = select(AutomationRun).where(AutomationRun.rule_id == rule_id)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(AutomationRun.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(
        items=[RunOut(
            id=x.id, rule_id=x.rule_id, entity_type=x.entity_type,
            entity_id=x.entity_id, trigger=x.trigger, action=x.action,
            status=x.status, recipients=x.recipients, payload=x.payload,
            error=x.error, created_at=x.created_at,
        ) for x in rows],
        total=total, limit=limit, offset=offset,
    )
