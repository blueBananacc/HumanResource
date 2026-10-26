"""审批中心 API(E3)。

- ``GET /approvals/inbox``                统一待办(策略步 + 旧一级)
- ``POST /approvals/decide``              统一裁决(有实例走多步链)
- ``GET/POST/PUT/DELETE /approvals/policies``  审批策略管理(admin)
- ``PATCH /approvals/delegate``           设置/清除自己的审批代理人
- ``GET /approvals/history``              我处理过的步骤(简单审计视图)

旧散装 decide 端点内部也会先查实例 —— 同一单据不会被两条路径双批。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, require_admin
from app.api.license import require_feature
from app.domain.models import (
    ApprovalInstance,
    ApprovalPolicy,
    ApprovalStep,
    Employee,
    User,
)
from app.infra.db import get_db
from app.services.approvals import (
    CONDITION_FIELDS,
    REQUEST_TYPES,
    decide_on_instance,
    describe_item,
    instance_for,
    pending_for,
    resolve_step_users,
    type_registry,
    validate_policy,
    _obj_for,
)
from app.services.audit import write_audit
from app.services.permissions import (
    can_decide_for_employee,
    has_permission,
)

router = APIRouter(
    prefix="/approvals",
    tags=["approval-center"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ 收件箱 / 裁决 ============


@router.get("/inbox")
def inbox(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """任何登录用户都能查 —— 返回的只会是自己可批的项。"""
    return {"items": pending_for(db, current)}


class DecideIn(BaseModel):
    request_type: str
    request_id: str
    action: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=500)


@router.post("/decide")
def unified_decide(
    body: DecideIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if body.request_type not in REQUEST_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "未知类型")
    inst = instance_for(
        db, current.tenant_id, body.request_type, body.request_id)
    if inst is not None:
        # 策略链路径:实例锁行防并发双批
        inst = db.execute(
            select(ApprovalInstance)
            .where(ApprovalInstance.id == inst.id).with_for_update()
        ).scalar_one()
        step = decide_on_instance(
            db, inst, current, body.action, body.note, request)
        db.commit()
        return {"mode": "step", "instance_status": inst.status,
                "step_idx": step.idx, "step_status": step.status}
    # 旧一级路径:领域权限 + 行级守卫
    conf = type_registry()[body.request_type]
    if not any(has_permission(current, p) for p in conf["perms"]):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无该类型审批权限")
    obj = db.execute(
        select(conf["model"]).where(conf["model"].id == body.request_id)
        .with_for_update()
    ).scalar_one_or_none()
    if obj is None or obj.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "单据不存在")
    if obj.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not can_decide_for_employee(db, current, obj.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
    from app.services.approvals import apply_domain_decision

    apply_domain_decision(
        db, body.request_type, obj, body.action, current, body.note, request)
    db.commit()
    return {"mode": "legacy", "status": body.action}


@router.get("/history")
def history(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """我批过的步骤(含所在实例的单据摘要)。"""
    rows = db.scalars(
        select(ApprovalStep).where(
            ApprovalStep.tenant_id == current.tenant_id,
            ApprovalStep.decided_by_id == current.id,
        ).order_by(ApprovalStep.decided_at.desc()).limit(100)
    ).all()
    insts = {i.id: i for i in db.scalars(
        select(ApprovalInstance).where(
            ApprovalInstance.id.in_([s.instance_id for s in rows] or ["__none__"]))
    )}
    out = []
    for s in rows:
        inst = insts.get(s.instance_id)
        obj = _obj_for(db, inst.request_type, inst.request_id) if inst else None
        emp = db.get(Employee, obj.employee_id) if obj and getattr(obj, "employee_id", None) else None
        out.append({
            "step_id": s.id, "idx": s.idx, "status": s.status,
            "note": s.note, "decided_at": s.decided_at.isoformat() if s.decided_at else None,
            "request_type": inst.request_type if inst else None,
            "request_id": inst.request_id if inst else None,
            "title": describe_item(db, inst.request_type, obj, emp) if inst and obj else "?",
        })
    return {"items": out}


# ============ 策略管理(admin) ============


class PolicyStepIn(BaseModel):
    resolver: Literal["manager", "role", "user"]
    key: str | None = None
    label: str | None = Field(default=None, max_length=64)


class PolicyCondIn(BaseModel):
    field: str
    op: Literal["eq", "ne", "gt", "lt", "ge", "le", "in"]
    value: object


class PolicyIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    request_type: str
    priority: int = 0
    conditions: list[PolicyCondIn] = Field(default_factory=list)
    steps: list[PolicyStepIn] = Field(min_length=1)
    active: bool = True


class PolicyPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    priority: int | None = None
    conditions: list[PolicyCondIn] | None = None
    steps: list[PolicyStepIn] | None = Field(default=None, min_length=1)
    active: bool | None = None


def _policy_out(db: Session, p: ApprovalPolicy) -> dict:
    return {
        "id": p.id, "name": p.name, "request_type": p.request_type,
        "priority": p.priority, "conditions": p.conditions or [],
        "steps": p.steps or [], "active": p.active,
        "condition_fields": list(CONDITION_FIELDS.get(p.request_type, ())),
    }


@router.get("/policies")
def list_policies(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    rows = db.scalars(
        select(ApprovalPolicy).where(
            ApprovalPolicy.tenant_id == current.tenant_id)
        .order_by(ApprovalPolicy.request_type, ApprovalPolicy.priority.desc())
    ).all()
    return {"items": [_policy_out(db, p) for p in rows],
            "request_types": list(REQUEST_TYPES),
            "condition_fields": {k: list(v) for k, v in CONDITION_FIELDS.items()}}


@router.post("/policies", status_code=status.HTTP_201_CREATED)
def create_policy(
    body: PolicyIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
):
    steps = [s.model_dump() for s in body.steps]
    conds = [c.model_dump() for c in body.conditions]
    validate_policy(body.request_type, conds, steps)
    p = ApprovalPolicy(
        tenant_id=current.tenant_id, name=body.name.strip(),
        request_type=body.request_type, priority=body.priority,
        conditions=conds, steps=steps, active=body.active,
        created_by=current.id,
    )
    db.add(p)
    db.flush()
    write_audit(db, actor=current, entity_type="approval_policy",
                entity_id=p.id, action="create",
                detail={"name": p.name, "type": p.request_type}, request=request)
    db.commit()
    db.refresh(p)
    return _policy_out(db, p)


@router.put("/policies/{policy_id}")
def update_policy(
    policy_id: str,
    body: PolicyPatch,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
):
    p = db.get(ApprovalPolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "策略不存在")
    fields = body.model_fields_set
    if "name" in fields:
        p.name = body.name.strip()
    if "priority" in fields:
        p.priority = body.priority
    if "conditions" in fields:
        p.conditions = [c.model_dump() for c in (body.conditions or [])]
    if "steps" in fields:
        p.steps = [s.model_dump() for s in (body.steps or [])]
    if "active" in fields:
        p.active = body.active
    validate_policy(p.request_type, p.conditions, p.steps)
    p.updated_at = _now()
    write_audit(db, actor=current, entity_type="approval_policy",
                entity_id=p.id, action="update", detail={}, request=request)
    db.commit()
    return _policy_out(db, p)


@router.delete("/policies/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_policy(
    policy_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_admin),
):
    p = db.get(ApprovalPolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "策略不存在")
    write_audit(db, actor=current, entity_type="approval_policy",
                entity_id=p.id, action="delete",
                detail={"name": p.name}, request=request)
    db.delete(p)
    db.commit()


# ============ 代理人 ============


@router.get("/delegate")
def get_delegate(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    d = (db.get(User, current.delegate_user_id)
         if current.delegate_user_id else None)
    return {"delegate_user_id": current.delegate_user_id,
            "delegate_email": d.email if d else None}


class DelegateIn(BaseModel):
    delegate_user_id: str | None = None
    delegate_email: str | None = None


@router.patch("/delegate")
def set_delegate(
    body: DelegateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    target_id = body.delegate_user_id
    if not target_id and body.delegate_email:
        d = db.scalars(
            select(User).where(User.tenant_id == current.tenant_id,
                               User.email == body.delegate_email.strip().lower())
        ).first()
        if not d:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "代理人不存在")
        target_id = d.id
    if target_id:
        d = db.get(User, target_id)
        if not d or d.tenant_id != current.tenant_id or d.status != "active":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "代理人不存在")
        if d.id == current.id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能代理给自己")
    # get_current_user 返回的是 expunge 过的游离对象,必须在请求会话内重取
    me = db.get(User, current.id)
    me.delegate_user_id = target_id
    db.commit()
    return {"delegate_user_id": me.delegate_user_id}


@router.get("/instances/{request_type}/{request_id}")
def instance_detail(
    request_type: str,
    request_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """单据的审批链状态(步骤级)——单据详情页/调试可用。"""
    inst = instance_for(db, current.tenant_id, request_type, request_id)
    if inst is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "该单据无审批链")
    steps = db.scalars(
        select(ApprovalStep).where(ApprovalStep.instance_id == inst.id)
        .order_by(ApprovalStep.idx)
    ).all()
    users = {u.id: u.email for u in db.scalars(
        select(User).where(User.tenant_id == current.tenant_id))}
    obj = _obj_for(db, request_type, request_id)
    return {
        "id": inst.id, "status": inst.status,
        "current_step_idx": inst.current_step_idx,
        "title": describe_item(db, request_type, obj) if obj else "?",
        "steps": [{
            "idx": s.idx, "resolver": s.resolver, "label": s.label,
            "status": s.status,
            "resolved": [users.get(u, u) for u in resolve_step_users(db, inst, s)],
            "decided_by": users.get(s.decided_by_id),
            "decided_at": s.decided_at.isoformat() if s.decided_at else None,
            "note": s.note,
        } for s in steps],
    }
