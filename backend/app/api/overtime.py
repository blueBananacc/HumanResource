"""加班申请审批 API(HR/主管端)。

员工端在 ``hr_me.py``(``/me/overtime-requests``)。批准时把加班小时按
``WorkCalendar.workday_hours`` 折算成天数(0.5 天粒度、不足 0.5 按 0.5 计),
结算到 ``leave_type_id`` 指定的调休类型:upsert 当年 allocation + 台账 +delta
(ref_type="overtime"),员工再按正常请假流程消耗。

- ``GET  /overtime/requests``                列表(scope=all/team/mine)
- ``POST /overtime/requests/{id}/approve``   批准 → 调休入账
- ``POST /overtime/requests/{id}/reject``    驳回
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    LeaveAllocation,
    LeaveLedgerEntry,
    LeaveType,
    OrgUnit,
    OvertimeRequest,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_APPROVE_LEAVE,
    PERM_MANAGE_LEAVE,
    can_decide_for_employee,
    get_linked_employee_id,
    require_permission,
    team_ids_with_self,
    visible_employee_ids,
)
from app.services.work_calendar import workday_hours

router = APIRouter(prefix="/overtime", tags=["hris-overtime"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class OvertimeRequestOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    work_date: str
    hours: float
    leave_type_id: str
    leave_type_name: str
    comp_days: float
    reason: str
    status: str
    approver_id: str | None
    approver_email: str | None
    decided_at: datetime | None
    decision_note: str | None
    own_request: bool
    created_at: datetime


class DecisionIn(BaseModel):
    note: str | None = None


def _comp_days(db: Session, tenant_id: str, hours: float) -> float:
    """加班小时 → 调休天数:按日标准工时折算,0.5 天粒度,不足 0.5 按 0.5 计。"""
    wh = workday_hours(db, tenant_id) or 8.0
    return max(0.5, round(hours / wh * 2) / 2)


def _out(
    db: Session, r: OvertimeRequest, emps: dict, types: dict,
    users: dict, orgs: dict, me_id: str | None = None,
) -> OvertimeRequestOut:
    e = emps.get(r.employee_id)
    return OvertimeRequestOut(
        id=r.id, employee_id=r.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        work_date=r.work_date.isoformat(), hours=r.hours,
        leave_type_id=r.leave_type_id,
        leave_type_name=types[r.leave_type_id].name if r.leave_type_id in types else "?",
        # 已批准的行读写入库的 comp_days;pending 时按当前日历预算(预览)
        comp_days=r.comp_days if r.comp_days is not None
        else _comp_days(db, r.tenant_id, r.hours),
        reason=r.reason, status=r.status,
        approver_id=r.approver_id,
        approver_email=users.get(r.approver_id) if r.approver_id else None,
        decided_at=r.decided_at, decision_note=r.decision_note,
        own_request=bool(e and e.user_id == me_id),
        created_at=r.created_at,
    )


def _maps(db: Session, tenant_id: str):
    emps = {e.id: e for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id))}
    types = {t.id: t for t in db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == tenant_id))}
    users = {u.id: u.email for u in db.scalars(
        select(User).where(User.tenant_id == tenant_id))}
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id))}
    return emps, types, users, orgs


def _locked(db: Session, req_id: str) -> OvertimeRequest | None:
    """FOR UPDATE:批准要写台账,并发双击会重复入账。"""
    return db.execute(
        select(OvertimeRequest).where(OvertimeRequest.id == req_id).with_for_update()
    ).scalar_one_or_none()


@router.get("/requests", response_model=PageOut[OvertimeRequestOut])
def list_requests(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    scope: Literal["all", "team", "mine"] = "all",
    employee_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE, PERM_APPROVE_LEAVE)),
):
    limit, offset, _ = pagination
    stmt = select(OvertimeRequest).where(
        OvertimeRequest.tenant_id == current.tenant_id
    )
    if status_filter:
        stmt = stmt.where(OvertimeRequest.status == status_filter)
    if employee_id:
        stmt = stmt.where(OvertimeRequest.employee_id == employee_id)
    if scope == "mine":
        own = get_linked_employee_id(db, current)
        stmt = stmt.where(OvertimeRequest.employee_id == (own or "__none__"))
    elif scope == "team":
        stmt = stmt.where(
            OvertimeRequest.employee_id.in_(
                team_ids_with_self(db, current) or ["__none__"]
            )
        )
    elif current.role not in ("admin", "hr"):
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(OvertimeRequest.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(OvertimeRequest.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps, types, users, orgs = _maps(db, current.tenant_id)
    return PageOut(
        items=[_out(db, r, emps, types, users, orgs, current.id) for r in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/requests/{req_id}/approve", response_model=OvertimeRequestOut)
def approve_request(
    req_id: str,
    body: DecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE, PERM_APPROVE_LEAVE)),
):
    r = _locked(db, req_id)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")

    comp_days = _comp_days(db, r.tenant_id, r.hours)
    # 折算结果写死在申请行上:撤销回冲与后续展示都以入账值为准,
    # 审批后改过日标准工时也不会回算错数
    r.comp_days = comp_days
    year = r.work_date.year
    # 调休入账:当年 allocation 直接加天(不存在则建),台账记 +delta。
    # 已有额度行走 SQL 原子自增 — 并发批同一员工两单不丢增量
    # (Python 侧 read-modify-write 会互相覆盖)。
    alloc = db.scalars(
        select(LeaveAllocation).where(
            LeaveAllocation.employee_id == r.employee_id,
            LeaveAllocation.leave_type_id == r.leave_type_id,
            LeaveAllocation.year == year,
        )
    ).first()
    if alloc:
        alloc.days = LeaveAllocation.days + comp_days
    else:
        alloc = LeaveAllocation(
            tenant_id=current.tenant_id, employee_id=r.employee_id,
            leave_type_id=r.leave_type_id, year=year, days=comp_days,
            note="加班调休",
        )
        db.add(alloc)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status.HTTP_409_CONFLICT, "额度写入冲突,请重试") from None
    db.add(LeaveLedgerEntry(
        tenant_id=current.tenant_id, employee_id=r.employee_id,
        leave_type_id=r.leave_type_id, year=year, delta_days=comp_days,
        ref_type="overtime", ref_id=r.id,
        note=f"加班 {r.work_date.isoformat()} {r.hours}h → 调休 {comp_days} 天",
    ))
    r.status = "approved"
    r.approver_id = current.id
    r.decided_at = _now()
    r.decision_note = body.note
    write_audit(
        db, actor=current, entity_type="overtime_request", entity_id=r.id,
        action="approve", detail={"hours": r.hours, "comp_days": comp_days},
        request=request,
    )
    db.commit()
    emps, types, users, orgs = _maps(db, current.tenant_id)
    return _out(db, r, emps, types, users, orgs, current.id)


@router.post("/requests/{req_id}/reject", response_model=OvertimeRequestOut)
def reject_request(
    req_id: str,
    body: DecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE, PERM_APPROVE_LEAVE)),
):
    r = _locked(db, req_id)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
    r.status = "rejected"
    r.approver_id = current.id
    r.decided_at = _now()
    r.decision_note = body.note
    write_audit(
        db, actor=current, entity_type="overtime_request", entity_id=r.id,
        action="reject", detail={}, request=request,
    )
    db.commit()
    emps, types, users, orgs = _maps(db, current.tenant_id)
    return _out(db, r, emps, types, users, orgs, current.id)
