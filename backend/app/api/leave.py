"""假期管理 API(HR 管理端)。

- ``GET/POST/PUT/DELETE /leave/types``            假期类型
- ``GET/POST/DELETE /leave/allocations``          额度发放(写台账 +)
- ``GET /leave/applications``                     申请列表(scope=team 仅下属+自己)
- ``POST /leave/applications/{id}/approve|reject`` 审批(approve_leave;HM 仅下属)
- ``GET /leave/balances?employee_id=&year=``       余额(也可传 all 看全员汇总)
- ``GET /leave/ledger``                           台账流水

员工自助端点在 ``hr_me.py``(/me/leave-applications 等)。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Attendance,
    Employee,
    LeaveAllocation,
    LeaveApplication,
    LeaveLedgerEntry,
    LeaveType,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_APPROVE_LEAVE,
    PERM_MANAGE_LEAVE,
    PERM_VIEW_REPORTS,
    get_linked_employee_id,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(prefix="/leave", tags=["hris-leave"], dependencies=[Depends(require_feature("hr.suite"))])


# ============ schemas ============


class LeaveTypeIn(BaseModel):
    name: str
    annual_days: float = Field(0, ge=0, le=365)
    is_paid: bool = True

    @field_validator("name")
    @classmethod
    def _v(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v


class LeaveTypeOut(BaseModel):
    id: str
    name: str
    annual_days: float
    is_paid: bool
    allocated_days: float = 0
    used_days: float = 0


class AllocationIn(BaseModel):
    employee_id: str
    leave_type_id: str
    year: int = Field(..., ge=2000, le=2100)
    days: float = Field(..., ge=0, le=365)
    note: str | None = None


class AllocationUpdateIn(BaseModel):
    days: float = Field(..., ge=0, le=365)
    note: str | None = None


class AllocationOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    leave_type_id: str
    leave_type_name: str
    year: int
    days: float
    note: str | None
    created_at: datetime


class ApplicationOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    leave_type_id: str
    leave_type_name: str
    start_date: str
    end_date: str
    days: float
    reason: str
    status: str
    approver_id: str | None
    approver_email: str | None
    decided_at: datetime | None
    decision_note: str | None
    created_at: datetime


class DecisionIn(BaseModel):
    note: str | None = None


class BalanceOut(BaseModel):
    allocation_id: str
    employee_id: str
    employee_name: str
    leave_type_id: str
    leave_type_name: str
    year: int
    allocated: float
    used: float
    pending: float
    balance: float


class LedgerOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    leave_type_name: str
    year: int
    delta_days: float
    ref_type: str
    ref_id: str
    note: str | None
    created_at: datetime


# ============ helpers ============


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _emp_map(db: Session, tenant_id: str):
    return {
        e.id: e
        for e in db.scalars(select(Employee).where(Employee.tenant_id == tenant_id))
    }


def _type_map(db: Session, tenant_id: str):
    return {
        t.id: t
        for t in db.scalars(select(LeaveType).where(LeaveType.tenant_id == tenant_id))
    }


def _user_map(db: Session, tenant_id: str):
    return {
        u.id: u.email
        for u in db.scalars(select(User).where(User.tenant_id == tenant_id))
    }


def _app_out(
    a: LeaveApplication, emps: dict, types: dict, users: dict, orgs: dict
) -> ApplicationOut:
    e = emps.get(a.employee_id)
    return ApplicationOut(
        id=a.id,
        employee_id=a.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        leave_type_id=a.leave_type_id,
        leave_type_name=types.get(a.leave_type_id).name if a.leave_type_id in types else "?",
        start_date=a.start_date.isoformat(),
        end_date=a.end_date.isoformat(),
        days=a.days,
        reason=a.reason,
        status=a.status,
        approver_id=a.approver_id,
        approver_email=users.get(a.approver_id) if a.approver_id else None,
        decided_at=a.decided_at,
        decision_note=a.decision_note,
        created_at=a.created_at,
    )


def _can_approve(db: Session, user: User, app: LeaveApplication) -> bool:
    """admin/hr 都能批;hiring_manager 仅直属下属。"""
    if user.role in ("admin", "hr"):
        return True
    if user.role == "hiring_manager":
        mgr_emp = get_linked_employee_id(db, user)
        target = db.get(Employee, app.employee_id)
        return bool(mgr_emp and target and target.reports_to_id == mgr_emp)
    return False


def _write_ledger(
    db: Session, *, tenant_id: str, employee_id: str, leave_type_id: str,
    year: int, delta: float, ref_type: str, ref_id: str, note: str | None = None,
) -> None:
    db.add(LeaveLedgerEntry(
        tenant_id=tenant_id, employee_id=employee_id, leave_type_id=leave_type_id,
        year=year, delta_days=delta, ref_type=ref_type, ref_id=ref_id, note=note,
    ))


def _mark_attendance_leave(
    db: Session, *, tenant_id: str, employee_id: str, start, end, on_leave: bool
) -> None:
    """请假审批通过/撤销时联动 attendance 状态。"""
    from datetime import timedelta

    day = start
    while day <= end:
        att = db.scalars(
            select(Attendance).where(
                Attendance.employee_id == employee_id, Attendance.day == day
            )
        ).first()
        if on_leave:
            if att is None:
                att = Attendance(
                    tenant_id=tenant_id, employee_id=employee_id, day=day,
                    status="leave", source="leave",
                )
                db.add(att)
            elif att.status in ("present", "absent") and att.first_in is None:
                att.status = "leave"
                att.source = "leave"
        else:
            if att and att.source == "leave" and att.status == "leave":
                att.status = "absent" if att.first_in is None else "present"
                att.source = "checkin"
        day += timedelta(days=1)


# ============ types ============


@router.get("/types", response_model=list[LeaveTypeOut])
def list_types(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    types = db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == current.tenant_id)
    ).all()
    year = datetime.now().year
    out = []
    for t in types:
        alloc = db.scalar(
            select(func.coalesce(func.sum(LeaveAllocation.days), 0)).where(
                LeaveAllocation.leave_type_id == t.id, LeaveAllocation.year == year
            )
        ) or 0
        used = db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.leave_type_id == t.id,
                LeaveApplication.status == "approved",
            )
        ) or 0
        out.append(LeaveTypeOut(
            id=t.id, name=t.name, annual_days=t.annual_days, is_paid=t.is_paid,
            allocated_days=alloc, used_days=used,
        ))
    return out


@router.post("/types", response_model=LeaveTypeOut, status_code=status.HTTP_201_CREATED)
def create_type(
    body: LeaveTypeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    t = LeaveType(tenant_id=current.tenant_id, name=body.name,
                  annual_days=body.annual_days, is_paid=body.is_paid)
    db.add(t)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "假期类型已存在")
    return LeaveTypeOut(id=t.id, name=t.name, annual_days=t.annual_days, is_paid=t.is_paid)


@router.put("/types/{type_id}", response_model=LeaveTypeOut)
def update_type(
    type_id: str,
    body: LeaveTypeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    t = db.get(LeaveType, type_id)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期类型不存在")
    t.name, t.annual_days, t.is_paid = body.name, body.annual_days, body.is_paid
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "假期类型已存在")
    return LeaveTypeOut(id=t.id, name=t.name, annual_days=t.annual_days, is_paid=t.is_paid)


@router.delete("/types/{type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_type(
    type_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    t = db.get(LeaveType, type_id)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期类型不存在")
    used = db.scalars(
        select(LeaveApplication.id).where(LeaveApplication.leave_type_id == t.id).limit(1)
    ).first()
    if used:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该类型下已有请假申请,不能删除")
    db.query(LeaveAllocation).filter(LeaveAllocation.leave_type_id == t.id).delete()
    db.delete(t)
    db.commit()


# ============ allocations ============


@router.get("/allocations", response_model=PageOut[AllocationOut])
def list_allocations(
    pagination: tuple = Depends(paginate_params),
    employee_id: str | None = None,
    year: int | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(LeaveAllocation).where(LeaveAllocation.tenant_id == current.tenant_id)
    if employee_id:
        stmt = stmt.where(LeaveAllocation.employee_id == employee_id)
    if year:
        stmt = stmt.where(LeaveAllocation.year == year)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(LeaveAllocation.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps, types = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id)
    return PageOut(
        items=[AllocationOut(
            id=a.id, employee_id=a.employee_id,
            employee_name=emps[a.employee_id].name if a.employee_id in emps else "?",
            leave_type_id=a.leave_type_id,
            leave_type_name=types[a.leave_type_id].name if a.leave_type_id in types else "?",
            year=a.year, days=a.days, note=a.note, created_at=a.created_at,
        ) for a in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/allocations", response_model=AllocationOut, status_code=status.HTTP_201_CREATED)
def create_allocation(
    body: AllocationIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    lt = db.get(LeaveType, body.leave_type_id)
    if not lt or lt.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期类型不存在")
    a = LeaveAllocation(
        tenant_id=current.tenant_id, employee_id=emp.id, leave_type_id=lt.id,
        year=body.year, days=body.days, note=body.note,
    )
    db.add(a)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "该员工本年度此类型额度已存在,可在列表中调整或收回")
    _write_ledger(
        db, tenant_id=current.tenant_id, employee_id=emp.id, leave_type_id=lt.id,
        year=body.year, delta=body.days, ref_type="allocation", ref_id=a.id,
        note=body.note or f"{lt.name} {body.year} 年度额度",
    )
    db.commit()
    db.refresh(a)
    write_audit(
        db, actor=current, entity_type="leave_allocation", entity_id=a.id,
        action="create", detail={"employee_id": emp.id, "days": a.days},
        request=request,
    )
    db.commit()
    return AllocationOut(
        id=a.id, employee_id=a.employee_id, employee_name=emp.name,
        leave_type_id=a.leave_type_id, leave_type_name=lt.name,
        year=a.year, days=a.days, note=a.note, created_at=a.created_at,
    )


@router.put("/allocations/{alloc_id}", response_model=AllocationOut)
def update_allocation(
    alloc_id: str,
    body: AllocationUpdateIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    """调整额度天数 — 以台账 delta 记账,余额接口照常重算。"""
    a = db.get(LeaveAllocation, alloc_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "额度不存在")
    delta = body.days - a.days
    if delta != 0:
        _write_ledger(
            db, tenant_id=current.tenant_id, employee_id=a.employee_id,
            leave_type_id=a.leave_type_id, year=a.year, delta=delta,
            ref_type="allocation_adjust", ref_id=a.id,
            note=body.note or "额度调整",
        )
    a.days = body.days
    if body.note is not None:
        a.note = body.note
    write_audit(
        db, actor=current, entity_type="leave_allocation", entity_id=a.id,
        action="update", detail={"days": a.days, "delta": delta},
        request=request,
    )
    db.commit()
    db.refresh(a)
    emps, types = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id)
    return AllocationOut(
        id=a.id, employee_id=a.employee_id,
        employee_name=emps[a.employee_id].name if a.employee_id in emps else "?",
        leave_type_id=a.leave_type_id,
        leave_type_name=types[a.leave_type_id].name if a.leave_type_id in types else "?",
        year=a.year, days=a.days, note=a.note, created_at=a.created_at,
    )


@router.delete("/allocations/{alloc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_allocation(
    alloc_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    a = db.get(LeaveAllocation, alloc_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "额度不存在")
    _write_ledger(
        db, tenant_id=current.tenant_id, employee_id=a.employee_id,
        leave_type_id=a.leave_type_id, year=a.year, delta=-a.days,
        ref_type="allocation_revoke", ref_id=a.id, note="额度收回",
    )
    write_audit(
        db, actor=current, entity_type="leave_allocation", entity_id=a.id,
        action="delete", detail={"days": a.days}, request=request,
    )
    db.delete(a)
    db.commit()


# ============ applications ============


@router.get("/applications", response_model=PageOut[ApplicationOut])
def list_applications(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    scope: Literal["all", "team", "mine"] = "all",
    employee_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_LEAVE)),
):
    limit, offset, q = pagination
    stmt = select(LeaveApplication).where(
        LeaveApplication.tenant_id == current.tenant_id
    )
    if status_filter:
        stmt = stmt.where(LeaveApplication.status == status_filter)
    if employee_id:
        stmt = stmt.where(LeaveApplication.employee_id == employee_id)
    if scope == "team":
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(LeaveApplication.employee_id.in_(ids or ["__none__"]))
    elif scope == "mine":
        own = get_linked_employee_id(db, current)
        stmt = stmt.where(
            LeaveApplication.employee_id == (own or "__none__")
        )
    stmt = apply_q_ilike(stmt, q, LeaveApplication.reason)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(LeaveApplication.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps, types, users = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id), _user_map(db, current.tenant_id)
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)
    )}
    return PageOut(
        items=[_app_out(a, emps, types, users, orgs) for a in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/applications/{app_id}/approve", response_model=ApplicationOut)
def approve_application(
    app_id: str,
    body: DecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_LEAVE)),
):
    a = db.get(LeaveApplication, app_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not _can_approve(db, current, a):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请")
    a.status = "approved"
    a.approver_id = current.id
    a.decided_at = _now()
    a.decision_note = body.note
    year = a.start_date.year
    _write_ledger(
        db, tenant_id=current.tenant_id, employee_id=a.employee_id,
        leave_type_id=a.leave_type_id, year=year, delta=-a.days,
        ref_type="application", ref_id=a.id,
        note=f"请假 {a.start_date.isoformat()}~{a.end_date.isoformat()}",
    )
    _mark_attendance_leave(
        db, tenant_id=current.tenant_id, employee_id=a.employee_id,
        start=a.start_date, end=a.end_date, on_leave=True,
    )
    write_audit(
        db, actor=current, entity_type="leave_application", entity_id=a.id,
        action="approve", detail={"days": a.days}, request=request,
    )
    db.commit()
    db.refresh(a)
    emps, types, users = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id), _user_map(db, current.tenant_id)
    return _app_out(a, emps, types, users, {})


@router.post("/applications/{app_id}/reject", response_model=ApplicationOut)
def reject_application(
    app_id: str,
    body: DecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_LEAVE)),
):
    a = db.get(LeaveApplication, app_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not _can_approve(db, current, a):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请")
    a.status = "rejected"
    a.approver_id = current.id
    a.decided_at = _now()
    a.decision_note = body.note
    write_audit(
        db, actor=current, entity_type="leave_application", entity_id=a.id,
        action="reject", detail={}, request=request,
    )
    db.commit()
    db.refresh(a)
    emps, types, users = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id), _user_map(db, current.tenant_id)
    return _app_out(a, emps, types, users, {})


# ============ balances / ledger ============


@router.get("/balances", response_model=list[BalanceOut])
def balances(
    employee_id: str | None = None,
    year: int | None = None,
    scope: Literal["all", "team"] = "all",
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    y = year or datetime.now().year
    emp_ids: list[str] | None
    if scope == "team":
        emp_ids = visible_employee_ids(db, current)
    else:
        if current.role in ("admin", "hr"):
            emp_ids = None
        else:
            emp_ids = visible_employee_ids(db, current)
    stmt = select(LeaveAllocation).where(
        LeaveAllocation.tenant_id == current.tenant_id, LeaveAllocation.year == y
    )
    if employee_id:
        stmt = stmt.where(LeaveAllocation.employee_id == employee_id)
    if emp_ids is not None:
        stmt = stmt.where(LeaveAllocation.employee_id.in_(emp_ids or ["__none__"]))
    allocs = db.scalars(stmt).all()
    emps, types = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id)
    out: list[BalanceOut] = []
    for a in allocs:
        used = db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == a.employee_id,
                LeaveApplication.leave_type_id == a.leave_type_id,
                LeaveApplication.status == "approved",
                func.extract("year", LeaveApplication.start_date) == y,
            )
        ) or 0
        pending = db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == a.employee_id,
                LeaveApplication.leave_type_id == a.leave_type_id,
                LeaveApplication.status == "pending",
                func.extract("year", LeaveApplication.start_date) == y,
            )
        ) or 0
        out.append(BalanceOut(
            allocation_id=a.id,
            employee_id=a.employee_id,
            employee_name=emps[a.employee_id].name if a.employee_id in emps else "?",
            leave_type_id=a.leave_type_id,
            leave_type_name=types[a.leave_type_id].name if a.leave_type_id in types else "?",
            year=y, allocated=a.days, used=used, pending=pending,
            balance=a.days - used - pending,
        ))
    return out


@router.get("/ledger", response_model=PageOut[LedgerOut])
def ledger(
    pagination: tuple = Depends(paginate_params),
    employee_id: str | None = None,
    leave_type_id: str | None = None,
    year: int | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    limit, offset, _ = pagination
    stmt = select(LeaveLedgerEntry).where(
        LeaveLedgerEntry.tenant_id == current.tenant_id
    )
    if employee_id:
        stmt = stmt.where(LeaveLedgerEntry.employee_id == employee_id)
    if leave_type_id:
        stmt = stmt.where(LeaveLedgerEntry.leave_type_id == leave_type_id)
    if year:
        stmt = stmt.where(LeaveLedgerEntry.year == year)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(LeaveLedgerEntry.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps, types = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id)
    return PageOut(
        items=[LedgerOut(
            id=e.id, employee_id=e.employee_id,
            employee_name=emps[e.employee_id].name if e.employee_id in emps else "?",
            leave_type_name=types[e.leave_type_id].name if e.leave_type_id in types else "?",
            year=e.year, delta_days=e.delta_days, ref_type=e.ref_type,
            ref_id=e.ref_id, note=e.note, created_at=e.created_at,
        ) for e in rows],
        total=total, limit=limit, offset=offset,
    )
