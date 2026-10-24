"""HR 工单 API(SuiteCRM Cases 移植)。

- ESS ``/me/cases``:提交/列表/公开留言(行级=本人员工档案)
- HR ``/cases``:队列/指派/状态/留言(manage_employees)
- ``case_no`` 租户内递增 — insert 时锁 tenant 行取 max+1。
- 状态变更 / 指派 → 写 Notification 给提交人账号。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    HrCase,
    HrCaseUpdate,
    Notification,
    Tenant,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    has_permission,
    require_linked_employee,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(
    prefix="/cases",
    tags=["hr-cases"],
    dependencies=[Depends(require_feature("hr.suite"))],
)
me_router = APIRouter(
    prefix="/me/cases",
    tags=["hr-cases-ess"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_PRIORITIES = ("low", "normal", "high", "urgent")
_OPEN_STATUSES = ("new", "assigned", "in_progress")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class CaseIn(BaseModel):
    subject: str = Field(min_length=1, max_length=128)
    kind: str = "other"
    priority: str = "normal"
    description: str | None = Field(default=None, max_length=10000)

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in HrCase.KINDS:
            raise ValueError(f"未知类型,可选:{','.join(HrCase.KINDS)}")
        return v

    @field_validator("priority")
    @classmethod
    def _v_priority(cls, v: str) -> str:
        if v not in _PRIORITIES:
            raise ValueError(f"未知优先级,可选:{','.join(_PRIORITIES)}")
        return v


class UpdateIn(BaseModel):
    body: str = Field(min_length=1, max_length=3000)
    is_internal: bool = False


class AssignIn(BaseModel):
    assignee_user_id: str | None


class StatusIn(BaseModel):
    status: Literal["new", "assigned", "in_progress", "resolved", "closed"]
    resolution: str | None = Field(default=None, max_length=3000)


class CaseUpdateOut(BaseModel):
    id: str
    author_user_id: str | None
    author_email: str | None = None
    body: str
    is_internal: bool
    created_at: datetime


class CaseOut(BaseModel):
    id: str
    case_no: int
    subject: str
    kind: str
    priority: str
    status: str
    description: str | None
    requester_employee_id: str | None
    requester_name: str | None = None
    assignee_user_id: str | None
    assignee_email: str | None = None
    resolution: str | None
    resolved_at: datetime | None
    updates: list[CaseUpdateOut] = []
    created_at: datetime
    updated_at: datetime


# ============ helpers ============


def _out(db: Session, c: HrCase, *, include_internal: bool, with_updates: bool = True) -> CaseOut:
    emp = db.get(Employee, c.requester_employee_id) if c.requester_employee_id else None
    assignee = db.get(User, c.assignee_user_id) if c.assignee_user_id else None
    updates: list[CaseUpdateOut] = []
    if with_updates:
        stmt = (
            select(HrCaseUpdate, User.email)
            .join(User, User.id == HrCaseUpdate.author_user_id, isouter=True)
            .where(HrCaseUpdate.case_id == c.id)
            .order_by(HrCaseUpdate.created_at)
        )
        if not include_internal:
            stmt = stmt.where(HrCaseUpdate.is_internal.is_(False))
        updates = [
            CaseUpdateOut(
                id=u.id, author_user_id=u.author_user_id, author_email=email,
                body=u.body, is_internal=u.is_internal, created_at=u.created_at,
            )
            for u, email in db.execute(stmt).all()
        ]
    return CaseOut(
        id=c.id,
        case_no=c.case_no,
        subject=c.subject,
        kind=c.kind,
        priority=c.priority,
        status=c.status,
        description=c.description,
        requester_employee_id=c.requester_employee_id,
        requester_name=emp.name if emp else None,
        assignee_user_id=c.assignee_user_id,
        assignee_email=assignee.email if assignee else None,
        resolution=c.resolution,
        resolved_at=c.resolved_at,
        updates=updates,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _next_case_no(db: Session, tenant_id: str) -> int:
    """租户内序列:锁 tenant 行后取 max+1,低并发下序列化足够。"""
    db.execute(select(Tenant.id).where(Tenant.id == tenant_id).with_for_update()).first()
    n = db.execute(
        select(func.max(HrCase.case_no)).where(HrCase.tenant_id == tenant_id)
    ).scalar_one()
    return (n or 0) + 1


def _notify_requester(db: Session, c: HrCase, title: str, body: str) -> None:
    emp = db.get(Employee, c.requester_employee_id) if c.requester_employee_id else None
    if emp and emp.user_id:
        db.add(
            Notification(
                tenant_id=c.tenant_id,
                user_id=emp.user_id,
                kind="case",
                title=title,
                body=body,
                link="/me/cases",
            )
        )


def _visible_requester_ids(db: Session, current: User) -> list[str] | None:
    return visible_employee_ids(db, current)


# ============ ESS ============


@me_router.get("", response_model=list[CaseOut])
def me_list(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    emp_id = require_linked_employee(db, current)
    rows = db.scalars(
        select(HrCase)
        .where(
            HrCase.tenant_id == current.tenant_id,
            HrCase.requester_employee_id == emp_id,
        )
        .order_by(HrCase.created_at.desc())
        .limit(100)
    ).all()
    return [_out(db, c, include_internal=False) for c in rows]


@me_router.post("", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
def me_create(
    body: CaseIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp_id = require_linked_employee(db, current)
    c = HrCase(
        tenant_id=current.tenant_id,
        case_no=_next_case_no(db, current.tenant_id),
        subject=body.subject,
        kind=body.kind,
        priority=body.priority,
        description=body.description,
        requester_employee_id=emp_id,
    )
    db.add(c)
    db.flush()
    write_audit(db, actor=current, entity_type="hr_case", entity_id=c.id, action="create", request=request)
    db.commit()
    return _out(db, c, include_internal=False)


def _own_case(db: Session, cid: str, current: User) -> HrCase:
    emp_id = require_linked_employee(db, current)
    c = db.get(HrCase, cid)
    if c is None or c.tenant_id != current.tenant_id or c.requester_employee_id != emp_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "工单不存在")
    return c


@me_router.post("/{cid}/updates", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
def me_update_comment(
    cid: str,
    body: UpdateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    c = _own_case(db, cid, current)
    if c.status == "closed":
        raise HTTPException(status.HTTP_409_CONFLICT, "已关闭工单不能留言")
    db.add(
        HrCaseUpdate(
            tenant_id=current.tenant_id,
            case_id=c.id,
            author_user_id=current.id,
            body=body.body,
            is_internal=False,
        )
    )
    c.updated_at = _now()
    db.commit()
    return _out(db, c, include_internal=False)


# ============ HR 队列 ============


@router.get("", response_model=PageOut[CaseOut])
def list_cases(
    status_filter: str | None = None,
    kind: str | None = None,
    priority: str | None = None,
    mine: bool = False,
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    limit, offset, q = paging
    stmt = select(HrCase).where(HrCase.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(HrCase.status == status_filter)
    if kind:
        stmt = stmt.where(HrCase.kind == kind)
    if priority:
        stmt = stmt.where(HrCase.priority == priority)
    if mine:
        stmt = stmt.where(HrCase.assignee_user_id == current.id)
    stmt = apply_q_ilike(stmt, q, HrCase.subject)
    emp_ids = _visible_requester_ids(db, current)
    if emp_ids is not None:
        stmt = stmt.where(HrCase.requester_employee_id.in_(emp_ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(HrCase.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(
        items=[_out(db, c, include_internal=True, with_updates=False) for c in rows],
        total=total, limit=limit, offset=offset,
    )


def _hr_case(db: Session, cid: str, current: User, *, check_scope: bool = False) -> HrCase:
    c = db.get(HrCase, cid)
    if c is None or c.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "工单不存在")
    if check_scope:
        emp_ids = _visible_requester_ids(db, current)
        if emp_ids is not None and c.requester_employee_id not in emp_ids:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "工单不存在")
    return c


@router.get("/{cid}", response_model=CaseOut)
def get_case(
    cid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    return _out(db, _hr_case(db, cid, current, check_scope=True), include_internal=True)


@router.post("/{cid}/assign", response_model=CaseOut)
def assign_case(
    cid: str,
    body: AssignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    c = _hr_case(db, cid, current, check_scope=True)
    if body.assignee_user_id:
        u = db.get(User, body.assignee_user_id)
        if u is None or u.tenant_id != current.tenant_id or u.status != "active":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "受理人不存在或已停用")
        # 受理人必须能看工单队列 —— 派给无权限角色等于派给了看不到的人
        if not has_permission(u, PERM_MANAGE_EMPLOYEES):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "受理人需要 manage_employees 权限",
            )
    old_assignee = c.assignee_user_id
    c.assignee_user_id = body.assignee_user_id
    if c.status == "new" and body.assignee_user_id:
        c.status = "assigned"
    c.updated_at = _now()
    # 只在受理人真的变化时通知 —— 重复指派同一人 / unassign 不打扰提交人
    if body.assignee_user_id and body.assignee_user_id != old_assignee:
        _notify_requester(db, c, "工单已受理", f"工单 HR-{c.case_no}「{c.subject}」已分配受理人")
    write_audit(db, actor=current, entity_type="hr_case", entity_id=c.id, action="assign", request=request)
    db.commit()
    return _out(db, c, include_internal=True)


@router.post("/{cid}/status", response_model=CaseOut)
def set_status(
    cid: str,
    body: StatusIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    c = _hr_case(db, cid, current, check_scope=True)
    if body.status == "resolved" and not (body.resolution or c.resolution):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "解决时需填写 resolution")
    old_status = c.status
    c.status = body.status
    if body.resolution:
        c.resolution = body.resolution
    c.resolved_at = _now() if body.status in ("resolved", "closed") else None
    c.updated_at = _now()
    # 状态真变化(或新写了 resolution)才通知 —— 重复提交同一状态不打扰提交人
    if body.status != old_status or body.resolution:
        _notify_requester(
            db, c, "工单状态更新",
            f"工单 HR-{c.case_no}「{c.subject}」→ {body.status}"
            + (f" · {body.resolution[:80]}" if body.resolution else ""),
        )
    write_audit(
        db, actor=current, entity_type="hr_case", entity_id=c.id,
        action=f"status:{body.status}", request=request,
    )
    db.commit()
    return _out(db, c, include_internal=True)


@router.post("/{cid}/updates", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
def hr_update(
    cid: str,
    body: UpdateIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    c = _hr_case(db, cid, current, check_scope=True)
    db.add(
        HrCaseUpdate(
            tenant_id=current.tenant_id,
            case_id=c.id,
            author_user_id=current.id,
            body=body.body,
            is_internal=body.is_internal,
        )
    )
    c.updated_at = _now()
    if not body.is_internal:
        _notify_requester(db, c, "工单有新回复", f"工单 HR-{c.case_no}「{c.subject}」有新进展")
    db.commit()
    return _out(db, c, include_internal=True)
