"""审批中心引擎(E3)。

参照 Frappe Workflow 精简版:单据提交时 ``route_request`` 按 request_type
查条件匹配的 ApprovalPolicy(priority 降序,首个命中),命中则建
ApprovalInstance + ApprovalStep 链;单据保持 pending 直到末步批准才落
领域副作用。无策略命中 → 走旧散装一级审批,向后完全兼容。

- resolver: manager(申请人直属上级→绑定账号,无上级兜底 hr 角色)
  / role(该角色全部活跃用户)/ user(指定 user_id);解析结果并入各人
  的 ``delegate_user_id``(代理人代收)
- 条件字段白名单 ``CONDITION_FIELDS``:LLM/UI 都只能挑白名单里的字段,
  无任意字段路径
- 域副作用统一走 ``apply_domain_decision``,里面懒加载各 api 模块抽出的
  ``_apply_*_decision``,与旧端点同一份代码——避免双轨口径漂移
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    ApprovalInstance,
    ApprovalPolicy,
    ApprovalStep,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    LeaveApplication,
    AttendanceRequest,
    Notification,
    OrgUnit,
    OvertimeRequest,
    User,
)

REQUEST_TYPES = (
    "leave", "overtime", "attendance_request", "expense_claim", "employee_request",
)

# 每类型允许做路由条件的字段(值提取器在 _field_value)
CONDITION_FIELDS: dict[str, tuple[str, ...]] = {
    "leave": ("days", "leave_type_id", "org_unit_id"),
    "overtime": ("hours", "leave_type_id", "org_unit_id"),
    "attendance_request": ("kind", "org_unit_id"),
    "expense_claim": ("total", "expense_type_id", "org_unit_id"),
    "employee_request": ("kind", "org_unit_id"),
}

_OPS = ("eq", "ne", "gt", "lt", "ge", "le", "in")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _obj_for(db: Session, request_type: str, request_id: str):
    model = {
        "leave": LeaveApplication,
        "overtime": OvertimeRequest,
        "attendance_request": AttendanceRequest,
        "expense_claim": ExpenseClaim,
        "employee_request": EmployeeRequest,
    }[request_type]
    return db.get(model, request_id)


def _employee_id_of(request_type: str, obj: Any) -> str | None:
    return getattr(obj, "employee_id", None)


def _value_for(db: Session, request_type: str, obj: Any, field: str) -> Any:
    if field == "org_unit_id":
        emp_id = _employee_id_of(request_type, obj)
        emp = db.get(Employee, emp_id) if emp_id else None
        return emp.org_unit_id if emp else None
    return getattr(obj, field, None)


def _cond_ok(db: Session, request_type: str, obj: Any, cond: dict) -> bool:
    field, op, val = cond.get("field"), cond.get("op"), cond.get("value")
    if field not in CONDITION_FIELDS.get(request_type, ()) or op not in _OPS:
        return False
    actual = _value_for(db, request_type, obj, field)
    if op == "eq":
        return actual == val
    if op == "ne":
        return actual != val
    if op == "in":
        return actual in (val if isinstance(val, list) else [val])
    try:
        a, b = float(actual), float(val)
    except (TypeError, ValueError):
        return False
    return {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b}[op]


def validate_policy(request_type: str, conditions: list, steps: list) -> None:
    """建/改策略时的静态校验;不合格抛 422。"""
    if request_type not in REQUEST_TYPES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"未知 request_type,可选:{','.join(REQUEST_TYPES)}")
    for c in conditions or []:
        if (c.get("field") not in CONDITION_FIELDS[request_type]
                or c.get("op") not in _OPS):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"条件不合法:{c.get('field')} {c.get('op')}(可选字段:"
                f"{','.join(CONDITION_FIELDS[request_type])})")
    if not steps:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "steps 不能为空")
    for s in steps:
        r = s.get("resolver")
        if r not in ("manager", "role", "user"):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "resolver 须为 manager/role/user")
        if r in ("role", "user") and not s.get("key"):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"resolver={r} 需要 key")


def instance_for(
    db: Session, tenant_id: str, request_type: str, request_id: str
) -> ApprovalInstance | None:
    return db.scalars(
        select(ApprovalInstance).where(
            ApprovalInstance.tenant_id == tenant_id,
            ApprovalInstance.request_type == request_type,
            ApprovalInstance.request_id == request_id,
        )
    ).first()


def route_request(
    db: Session, tenant_id: str, request_type: str, obj: Any
) -> ApprovalInstance | None:
    """单据提交时调用:有匹配策略则接管(建实例+步骤+通知首步审批人)。"""
    if instance_for(db, tenant_id, request_type, obj.id):
        return None
    policies = db.scalars(
        select(ApprovalPolicy).where(
            ApprovalPolicy.tenant_id == tenant_id,
            ApprovalPolicy.request_type == request_type,
            ApprovalPolicy.active.is_(True),
        ).order_by(ApprovalPolicy.priority.desc())
    ).all()
    pol = next(
        (p for p in policies
         if all(_cond_ok(db, request_type, obj, c) for c in (p.conditions or []))),
        None,
    )
    if pol is None:
        return None
    inst = ApprovalInstance(
        tenant_id=tenant_id, request_type=request_type, request_id=obj.id,
        policy_id=pol.id, status="pending", current_step_idx=0,
    )
    db.add(inst)
    db.flush()
    for i, s in enumerate(pol.steps or []):
        db.add(ApprovalStep(
            tenant_id=tenant_id, instance_id=inst.id, idx=i,
            resolver=s.get("resolver"), resolver_key=s.get("key"),
            label=s.get("label"),
        ))
    db.flush()
    _notify_step_approvers(db, inst, obj)
    return inst


def _requester_user_id(db: Session, request_type: str, obj: Any) -> str | None:
    emp = db.get(Employee, _employee_id_of(request_type, obj))
    return emp.user_id if emp else None


def resolve_step_users(
    db: Session, inst: ApprovalInstance, step: ApprovalStep
) -> list[str]:
    """把 resolver 解析成可审批 user_id 集合(含代理人,去申请人本人)。"""
    obj = _obj_for(db, inst.request_type, inst.request_id)
    requester_uid = _requester_user_id(db, inst.request_type, obj)
    users: list[str] = []
    if step.resolver == "manager":
        emp = db.get(Employee, _employee_id_of(inst.request_type, obj))
        mgr = db.get(Employee, emp.reports_to_id) if emp and emp.reports_to_id else None
        if mgr and mgr.user_id:
            users = [mgr.user_id]
        else:
            # 无上级兜底:hr 角色活跃用户
            users = [u.id for u in db.scalars(
                select(User).where(User.tenant_id == inst.tenant_id,
                                   User.role == "hr", User.status == "active")
            )]
    elif step.resolver == "role":
        users = [u.id for u in db.scalars(
            select(User).where(User.tenant_id == inst.tenant_id,
                               User.role == step.resolver_key,
                               User.status == "active")
        )]
    elif step.resolver == "user":
        users = [step.resolver_key] if step.resolver_key else []
    # 代理人并入:被解析者设置了 delegate_user_id → 代理人也能批
    if users:
        delegates = [r[0] for r in db.execute(
            select(User.delegate_user_id).where(
                User.id.in_(users), User.delegate_user_id.isnot(None))
        ).all()]
        users = list(dict.fromkeys(users + delegates))
    # 自批禁止:申请人本人的账号从可批集合剔除
    return [u for u in users if u != requester_uid]


def _notify(
    db: Session, tenant_id: str, user_id: str, title: str, body: str, link: str
) -> None:
    db.add(Notification(
        tenant_id=tenant_id, user_id=user_id, kind="approval",
        title=title, body=body, link=link,
    ))


def cancel_instance_for(
    db: Session, tenant_id: str, request_type: str, request_id: str
) -> None:
    """申请人撤回单据时关闭挂起的审批实例(步骤一并标记 cancelled)。"""
    inst = instance_for(db, tenant_id, request_type, request_id)
    if inst is None or inst.status != "pending":
        return
    inst.status = "cancelled"
    inst.decided_at = _now()
    for s in db.scalars(
        select(ApprovalStep).where(
            ApprovalStep.instance_id == inst.id,
            ApprovalStep.status == "pending",
        )
    ):
        s.status = "cancelled"
        s.decided_at = _now()


def _step_title(db: Session, inst: ApprovalInstance, obj: Any) -> str:
    emp = db.get(Employee, _employee_id_of(inst.request_type, obj))
    return describe_item(db, inst.request_type, obj, emp)


def _notify_step_approvers(
    db: Session, inst: ApprovalInstance, obj: Any
) -> None:
    step = db.scalars(
        select(ApprovalStep).where(
            ApprovalStep.instance_id == inst.id,
            ApprovalStep.idx == inst.current_step_idx,
        )
    ).first()
    if not step:
        return
    title = _step_title(db, inst, obj)
    for uid in resolve_step_users(db, inst, step):
        db.add(Notification(
            tenant_id=inst.tenant_id, user_id=uid, kind="approval",
            title="有待审批",
            body=f"{title} · 第 {step.idx + 1} 级({step.label or step.resolver})",
            link="/approvals",
        ))


def decide_on_instance(
    db: Session, inst: ApprovalInstance, user: User,
    action: str, note: str | None, request=None,
) -> ApprovalStep:
    """用户在当前步表态。action ∈ approved|rejected。"""
    if inst.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该审批已结束")
    step = db.scalars(
        select(ApprovalStep).where(
            ApprovalStep.instance_id == inst.id,
            ApprovalStep.idx == inst.current_step_idx,
        )
    ).first()
    if not step or step.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前步骤已处理")
    allowed = resolve_step_users(db, inst, step)
    if user.id not in allowed and user.role != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "你不在本步骤的审批人范围内")
    obj = _obj_for(db, inst.request_type, inst.request_id)
    step.status = action
    step.decided_by_id = user.id
    step.decided_at = _now()
    step.note = note
    requester_uid = _requester_user_id(db, inst.request_type, obj)
    title = _step_title(db, inst, obj)
    if action == "rejected":
        inst.status = "rejected"
        inst.decided_at = _now()
        apply_domain_decision(
            db, inst.request_type, obj, "rejected", user, note, request)
        if requester_uid:
            _notify(db, inst.tenant_id, requester_uid, "审批被驳回",
                    f"{title}" + (f":{note}" if note else ""), "/me/requests")
        return step
    # 批准:还有下一步则推进,末步则整单通过
    nxt = db.scalars(
        select(ApprovalStep).where(
            ApprovalStep.instance_id == inst.id,
            ApprovalStep.idx > inst.current_step_idx,
            ApprovalStep.status == "pending",
        ).order_by(ApprovalStep.idx)
    ).first()
    if nxt is None:
        inst.status = "approved"
        inst.decided_at = _now()
        apply_domain_decision(
            db, inst.request_type, obj, "approved", user, note, request)
        if requester_uid:
            _notify(db, inst.tenant_id, requester_uid, "审批已通过", title,
                    "/me/requests")
    else:
        inst.current_step_idx = nxt.idx
        _notify_step_approvers(db, inst, obj)
    return step


def apply_domain_decision(
    db: Session, request_type: str, obj: Any, action: str,
    actor: User, note: str | None, request=None,
) -> None:
    """调用各域抽出的 apply 函数——与散装端点同一份副作用代码。"""
    if request_type == "leave":
        from app.api.leave import _apply_decision
        _apply_decision(db, obj, action, actor, note, request)
    elif request_type == "overtime":
        from app.api.overtime import _apply_decision
        _apply_decision(db, obj, action, actor, note, request)
    elif request_type == "attendance_request":
        from app.api.attendance import _apply_request_decision
        _apply_request_decision(db, obj, action, actor, note, request)
    elif request_type == "expense_claim":
        from app.api.expenses import _apply_claim_decision
        _apply_claim_decision(db, obj, action, actor, note, request)
    elif request_type == "employee_request":
        from app.api.requests import _apply_request_decision
        _apply_request_decision(db, obj, action, actor, note, request)


def describe_item(
    db: Session, request_type: str, obj: Any, emp: Employee | None = None
) -> str:
    """收件箱/通知用的一句话摘要。"""
    name = emp.name if emp else "?"
    if request_type == "leave":
        from app.domain.models import LeaveType
        lt = db.get(LeaveType, obj.leave_type_id)
        return (f"{name} 请假·{lt.name if lt else '?'} "
                f"{obj.start_date.isoformat()}~{obj.end_date.isoformat()} "
                f"{obj.days}天")
    if request_type == "overtime":
        return f"{name} 加班 {obj.work_date.isoformat()} {obj.hours}h"
    if request_type == "attendance_request":
        return f"{name} 补卡 {obj.ts.isoformat()}({obj.kind})"
    if request_type == "expense_claim":
        return f"{name} 报销 ¥{obj.total}"
    if request_type == "employee_request":
        return f"{name} 申请·{obj.kind}:{obj.subject}"
    return f"{name} {request_type}"


# 类型 → (模型, 领域权限元组, 摘要字段) —— inbox/统一 decide 共用
def type_registry() -> dict[str, dict]:
    from app.services.permissions import (
        PERM_APPROVE_EXPENSES,
        PERM_APPROVE_LEAVE,
        PERM_MANAGE_ATTENDANCE,
        PERM_MANAGE_EMPLOYEES,
        PERM_MANAGE_LEAVE,
    )
    return {
        "leave": {"model": LeaveApplication, "perms": (PERM_APPROVE_LEAVE,)},
        "overtime": {"model": OvertimeRequest,
                     "perms": (PERM_MANAGE_LEAVE, PERM_APPROVE_LEAVE)},
        "attendance_request": {"model": AttendanceRequest,
                               "perms": (PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)},
        "expense_claim": {"model": ExpenseClaim, "perms": (PERM_APPROVE_EXPENSES,)},
        "employee_request": {"model": EmployeeRequest,
                             "perms": (PERM_MANAGE_EMPLOYEES,)},
    }


def pending_for(db: Session, user: User) -> list[dict]:
    """统一收件箱:step 待办(我是当前步解析人) + legacy 待办(无可批实例
    且旧守卫通过)。返回 {request_type, request_id, title, mode, step_*}."""
    from app.services.permissions import can_decide_for_employee, has_permission

    items: list[dict] = []
    reg = type_registry()
    emps = {e.id: e for e in db.scalars(
        select(Employee).where(Employee.tenant_id == user.tenant_id))}
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == user.tenant_id))}

    insts = db.scalars(
        select(ApprovalInstance).where(
            ApprovalInstance.tenant_id == user.tenant_id,
            ApprovalInstance.status == "pending",
        )
    ).all()
    handled: set[tuple[str, str]] = set()
    for inst in insts:
        obj = _obj_for(db, inst.request_type, inst.request_id)
        if obj is None or getattr(obj, "status", None) != "pending":
            continue
        handled.add((inst.request_type, inst.request_id))
        step = db.scalars(
            select(ApprovalStep).where(
                ApprovalStep.instance_id == inst.id,
                ApprovalStep.idx == inst.current_step_idx,
            )
        ).first()
        if not step:
            continue
        allowed = resolve_step_users(db, inst, step)
        if user.id not in allowed and user.role != "admin":
            continue
        emp = emps.get(_employee_id_of(inst.request_type, obj))
        total_steps = db.scalar(
            select(ApprovalStep.idx).where(
                ApprovalStep.instance_id == inst.id
            ).order_by(ApprovalStep.idx.desc()).limit(1)
        )
        items.append({
            "mode": "step", "request_type": inst.request_type,
            "request_id": inst.request_id, "instance_id": inst.id,
            "step_idx": inst.current_step_idx,
            "step_count": (total_steps or 0) + 1,
            "step_label": step.label or step.resolver,
            "employee_name": emp.name if emp else "?",
            "org_name": orgs.get(emp.org_unit_id) if emp else None,
            "title": describe_item(db, inst.request_type, obj, emp),
            "created_at": obj.created_at.isoformat() if obj.created_at else None,
        })

    for rtype, conf in reg.items():
        if not any(has_permission(user, p) for p in conf["perms"]):
            continue
        rows = db.scalars(
            select(conf["model"]).where(
                conf["model"].tenant_id == user.tenant_id,
                conf["model"].status == "pending",
            ).order_by(conf["model"].created_at.desc()).limit(200)
        ).all()
        for obj in rows:
            if (rtype, obj.id) in handled:
                continue
            emp_id = _employee_id_of(rtype, obj)
            if not can_decide_for_employee(db, user, emp_id):
                continue
            emp = emps.get(emp_id)
            items.append({
                "mode": "legacy", "request_type": rtype, "request_id": obj.id,
                "employee_name": emp.name if emp else "?",
                "org_name": orgs.get(emp.org_unit_id) if emp else None,
                "title": describe_item(db, rtype, obj, emp),
                "created_at": obj.created_at.isoformat() if obj.created_at else None,
            })
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items
