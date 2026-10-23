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
from app.api.license import require_feature
from app.domain.models import (
    Attendance,
    Employee,
    LeaveAllocation,
    LeaveApplication,
    LeaveLedgerEntry,
    LeavePolicy,
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
    PERM_VIEW_REPORTS,
    can_decide_for_employee,
    get_linked_employee_id,
    require_permission,
    team_ids_with_self,
    visible_employee_ids,
)
from app.services.work_calendar import holiday_map, is_workday, workday_set

router = APIRouter(prefix="/leave", tags=["hris-leave"], dependencies=[Depends(require_feature("hr.suite"))])


# ============ schemas ============


class LeaveTypeIn(BaseModel):
    name: str
    annual_days: float = Field(0, ge=0, le=365)
    is_paid: bool = True
    # 调休类型:加班申请批准后把额度结算到这类类型(额度来自加班而非年度发放)
    is_comp: bool = False

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
    is_comp: bool = False
    allocated_days: float = 0
    used_days: float = 0


def _validate_half_day(v: float) -> float:
    if (v * 2) != int(v * 2):
        raise ValueError("天数须以 0.5 天为步长")
    return v


class AllocationIn(BaseModel):
    employee_id: str
    leave_type_id: str
    year: int = Field(..., ge=2000, le=2100)
    days: float = Field(..., ge=0, le=365)
    note: str | None = None

    @field_validator("days")
    @classmethod
    def _v_days(cls, v: float) -> float:
        return _validate_half_day(v)


class AllocationUpdateIn(BaseModel):
    days: float = Field(..., ge=0, le=365)
    note: str | None = None

    @field_validator("days")
    @classmethod
    def _v_days(cls, v: float) -> float:
        return _validate_half_day(v)


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
    own_request: bool
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
    a: LeaveApplication, emps: dict, types: dict, users: dict, orgs: dict,
    me_id: str | None = None,
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
        own_request=bool(e and e.user_id == me_id),
        created_at=a.created_at,
    )


def _locked_app(db: Session, app_id: str) -> LeaveApplication | None:
    """FOR UPDATE 锁申请行 — 审批/撤销写台账,并发双击会记两次流水。"""
    return db.execute(
        select(LeaveApplication).where(LeaveApplication.id == app_id).with_for_update()
    ).scalar_one_or_none()


def _can_approve(db: Session, user: User, app: LeaveApplication) -> bool:
    """admin/hr 都能批;hiring_manager 仅直属下属;任何人不能批自己的申请。
    与补卡/加班审批共用 services.permissions.can_decide_for_employee 同一口径。"""
    return can_decide_for_employee(db, user, app.employee_id)


def _write_ledger(
    db: Session, *, tenant_id: str, employee_id: str, leave_type_id: str,
    year: int, delta: float, ref_type: str, ref_id: str, note: str | None = None,
) -> None:
    db.add(LeaveLedgerEntry(
        tenant_id=tenant_id, employee_id=employee_id, leave_type_id=leave_type_id,
        year=year, delta_days=delta, ref_type=ref_type, ref_id=ref_id, note=note,
    ))


def _mark_attendance_leave(
    db: Session, *, tenant_id: str, employee_id: str, start, end
) -> None:
    """请假审批通过时把区间内考勤标记为 leave(撤销路径在 hr_me 删行重算)。

    只写工作日:周末/节假日不算请假,保持台账扣的天数与考勤标记的天数
    口径一致(请假校验已按工作日数卡 days)。
    """
    from datetime import timedelta

    wd = workday_set(db, tenant_id)
    hols = holiday_map(db, tenant_id, start, end)
    day = start
    while day <= end:
        if not is_workday(day, wd, hols):
            day += timedelta(days=1)
            continue
        att = db.scalars(
            select(Attendance).where(
                Attendance.employee_id == employee_id, Attendance.day == day
            )
        ).first()
        if att is None:
            att = Attendance(
                tenant_id=tenant_id, employee_id=employee_id, day=day,
                status="leave", source="leave",
            )
            db.add(att)
        elif att.status in ("present", "absent") and att.first_in is None:
            att.status = "leave"
            att.source = "leave"
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
            is_comp=t.is_comp, allocated_days=alloc, used_days=used,
        ))
    return out


@router.post("/types", response_model=LeaveTypeOut, status_code=status.HTTP_201_CREATED)
def create_type(
    body: LeaveTypeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    t = LeaveType(tenant_id=current.tenant_id, name=body.name,
                  annual_days=body.annual_days, is_paid=body.is_paid,
                  is_comp=body.is_comp)
    db.add(t)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "假期类型已存在")
    return LeaveTypeOut(id=t.id, name=t.name, annual_days=t.annual_days,
                        is_paid=t.is_paid, is_comp=t.is_comp)


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
    # 改成调休类型前先确认没被政策引用:政策包里出现调休类型是死配置
    # (run 会跳过),不如直接拦住让管理员先解引用
    if body.is_comp and not t.is_comp and any(
        it.get("leave_type_id") == t.id
        for p in db.scalars(
            select(LeavePolicy).where(LeavePolicy.tenant_id == current.tenant_id)
        )
        for it in (p.items or [])
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "该类型已被假期政策引用,先从政策中移除")
    t.name, t.annual_days, t.is_paid, t.is_comp = (
        body.name, body.annual_days, body.is_paid, body.is_comp
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "假期类型已存在")
    return LeaveTypeOut(id=t.id, name=t.name, annual_days=t.annual_days,
                        is_paid=t.is_paid, is_comp=t.is_comp)


@router.delete("/types/{type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_type(
    type_id: str,
    request: Request,
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
    # 调休类型挂的加班申请也是 FK CASCADE:不拦的话删除类型会把审批
    # 历史连同已入账的额度一起抹掉,与"有申请不能删"的语义不一致。
    if db.scalars(
        select(OvertimeRequest.id)
        .where(OvertimeRequest.leave_type_id == t.id).limit(1)
    ).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该类型下已有加班申请,不能删除")
    if any(
        it.get("leave_type_id") == t.id
        for p in db.scalars(
            select(LeavePolicy).where(LeavePolicy.tenant_id == current.tenant_id)
        )
        for it in (p.items or [])
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该类型已被假期政策引用,先从政策中移除")
    # 该类型下没有申请才允许删;其额度与台账条目随 leave_types.id 的
    # ON DELETE CASCADE 一并清掉,子台账自洽不留孤儿行。
    db.query(LeaveAllocation).filter(LeaveAllocation.leave_type_id == t.id).delete()
    db.delete(t)
    write_audit(
        db, actor=current, entity_type="leave_type", entity_id=t.id,
        action="delete", detail={"name": t.name}, request=request,
    )
    db.commit()


# ============ allocations ============


@router.get("/allocations", response_model=PageOut[AllocationOut])
def list_allocations(
    pagination: tuple = Depends(paginate_params),
    employee_id: str | None = None,
    year: int | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
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
    if scope == "mine":
        own = get_linked_employee_id(db, current)
        stmt = stmt.where(
            LeaveApplication.employee_id == (own or "__none__")
        )
    elif scope == "team":
        stmt = stmt.where(
            LeaveApplication.employee_id.in_(
                team_ids_with_self(db, current) or ["__none__"]
            )
        )
    elif current.role not in ("admin", "hr"):
        # scope=all:非管理角色仍按汇报线过滤(主管只见下属+自己)
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(LeaveApplication.employee_id.in_(ids or ["__none__"]))
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
        items=[_app_out(a, emps, types, users, orgs, current.id) for a in rows],
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
    a = _locked_app(db, app_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not _can_approve(db, current, a):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
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
        start=a.start_date, end=a.end_date,
    )
    write_audit(
        db, actor=current, entity_type="leave_application", entity_id=a.id,
        action="approve", detail={"days": a.days}, request=request,
    )
    db.commit()
    db.refresh(a)
    emps, types, users = _emp_map(db, current.tenant_id), _type_map(db, current.tenant_id), _user_map(db, current.tenant_id)
    return _app_out(a, emps, types, users, {}, current.id)


@router.post("/applications/{app_id}/reject", response_model=ApplicationOut)
def reject_application(
    app_id: str,
    body: DecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_LEAVE)),
):
    a = _locked_app(db, app_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    if not _can_approve(db, current, a):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
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
    return _app_out(a, emps, types, users, {}, current.id)


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
        emp_ids = team_ids_with_self(db, current)
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


# ============ policies(假期政策:自动发放 + 年度结转)============


class PolicyItemIn(BaseModel):
    leave_type_id: str
    days: float = Field(..., ge=0, le=365)

    @field_validator("days")
    @classmethod
    def _v_days(cls, v: float) -> float:
        return _validate_half_day(v)


class PolicyIn(BaseModel):
    name: str
    items: list[PolicyItemIn] = Field(..., min_length=1)
    # 结转到下一年的额度上限;None 表示不结转
    carry_forward: float | None = Field(None, ge=0, le=365)

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v


class PolicyItemOut(BaseModel):
    leave_type_id: str
    leave_type_name: str
    days: float


class PolicyOut(BaseModel):
    id: str
    name: str
    items: list[PolicyItemOut]
    carry_forward: float | None
    assigned_count: int
    created_at: datetime


class PolicyAssignIn(BaseModel):
    employee_ids: list[str] = Field(..., min_length=1)


class PolicyRunIn(BaseModel):
    year: int = Field(..., ge=2000, le=2100)


def _check_policy_types(db: Session, tenant_id: str, items: list[PolicyItemIn]) -> None:
    for it in items:
        lt = db.get(LeaveType, it.leave_type_id)
        if not lt or lt.tenant_id != tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "政策包含不存在的假期类型")
        # 调休额度只能来自加班结算:政策 run 是 set-total 语义,放进调休类型
        # 会把员工加班攒下的额度按目标天数静默清零
        if lt.is_comp:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "调休类型的额度只能来自加班结算,不能放进政策包",
            )


def _policy_out(db: Session, p: LeavePolicy, tenant_id: str) -> PolicyOut:
    types = _type_map(db, tenant_id)
    n = db.scalar(
        select(func.count()).select_from(Employee).where(
            Employee.leave_policy_id == p.id
        )
    ) or 0
    return PolicyOut(
        id=p.id, name=p.name,
        items=[PolicyItemOut(
            leave_type_id=it["leave_type_id"],
            leave_type_name=types[it["leave_type_id"]].name
            if it["leave_type_id"] in types else "?",
            days=it["days"],
        ) for it in (p.items or [])],
        carry_forward=p.carry_forward, assigned_count=n, created_at=p.created_at,
    )


def _apply_policy(
    db: Session, tenant_id: str, employee_id: str, policy: LeavePolicy, year: int
) -> None:
    """按政策对齐某员工的当年额度。

    set-total 语义:目标天数 = 政策天数 + 上年结转(carry_forward 封顶),
    差额写台账 ``policy_run`` —— 重复 run 幂等(delta=0 不再记账)。
    """
    for item in policy.items or []:
        lt_id, days = item["leave_type_id"], item["days"]
        lt = db.get(LeaveType, lt_id)
        if not lt or lt.tenant_id != tenant_id:
            continue
        # 防御:存量政策项的类型后来被改成调休 — 跳过,不清加班攒的额度
        if lt.is_comp:
            continue
        carry = 0.0
        if policy.carry_forward:
            prev = db.scalars(
                select(LeaveAllocation).where(
                    LeaveAllocation.employee_id == employee_id,
                    LeaveAllocation.leave_type_id == lt_id,
                    LeaveAllocation.year == year - 1,
                )
            ).first()
            if prev:
                used = db.scalar(
                    select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                        LeaveApplication.employee_id == employee_id,
                        LeaveApplication.leave_type_id == lt_id,
                        LeaveApplication.status == "approved",
                        func.extract("year", LeaveApplication.start_date) == year - 1,
                    )
                ) or 0
                carry = min(max(0.0, prev.days - used), policy.carry_forward)
        target = days + carry
        # 已休部分不回收:目标低于当年已批用量时按用量对齐,余额不为负
        used_y = float(db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == employee_id,
                LeaveApplication.leave_type_id == lt_id,
                LeaveApplication.status == "approved",
                func.extract("year", LeaveApplication.start_date) == year,
            )
        ) or 0)
        target = max(target, used_y)
        alloc = db.scalars(
            select(LeaveAllocation).where(
                LeaveAllocation.employee_id == employee_id,
                LeaveAllocation.leave_type_id == lt_id,
                LeaveAllocation.year == year,
            )
        ).first()
        if alloc:
            delta = target - alloc.days
            alloc.days = target
        else:
            alloc = LeaveAllocation(
                tenant_id=tenant_id, employee_id=employee_id,
                leave_type_id=lt_id, year=year, days=target,
                note=f"政策「{policy.name}」发放",
            )
            db.add(alloc)
            db.flush()
            delta = target
        if delta != 0:
            _write_ledger(
                db, tenant_id=tenant_id, employee_id=employee_id,
                leave_type_id=lt_id, year=year, delta=delta,
                ref_type="policy_run", ref_id=policy.id,
                note=f"政策「{policy.name}」{year} 年发放"
                     + (f"(含上年结转 {carry})" if carry else ""),
            )


@router.get("/policies", response_model=list[PolicyOut])
def list_policies(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    rows = db.scalars(
        select(LeavePolicy).where(LeavePolicy.tenant_id == current.tenant_id)
        .order_by(LeavePolicy.created_at.desc())
    ).all()
    return [_policy_out(db, p, current.tenant_id) for p in rows]


@router.post("/policies", response_model=PolicyOut, status_code=status.HTTP_201_CREATED)
def create_policy(
    body: PolicyIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    _check_policy_types(db, current.tenant_id, body.items)
    p = LeavePolicy(
        tenant_id=current.tenant_id, name=body.name,
        items=[{"leave_type_id": it.leave_type_id, "days": it.days}
               for it in body.items],
        carry_forward=body.carry_forward,
    )
    db.add(p)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "假期政策已存在") from None
    return _policy_out(db, p, current.tenant_id)


@router.put("/policies/{policy_id}", response_model=PolicyOut)
def update_policy(
    policy_id: str,
    body: PolicyIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    p = db.get(LeavePolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期政策不存在")
    _check_policy_types(db, current.tenant_id, body.items)
    p.name = body.name
    p.items = [{"leave_type_id": it.leave_type_id, "days": it.days}
               for it in body.items]
    p.carry_forward = body.carry_forward
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "假期政策已存在") from None
    return _policy_out(db, p, current.tenant_id)


@router.delete("/policies/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_policy(
    policy_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    p = db.get(LeavePolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期政策不存在")
    bound = db.scalars(
        select(Employee.id).where(Employee.leave_policy_id == p.id).limit(1)
    ).first()
    if bound:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "仍有员工绑定该政策,先解除绑定再删除")
    write_audit(
        db, actor=current, entity_type="leave_policy", entity_id=p.id,
        action="delete", detail={"name": p.name}, request=request,
    )
    db.delete(p)
    db.commit()


@router.post("/policies/{policy_id}/assign")
def assign_policy(
    policy_id: str,
    body: PolicyAssignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    """给员工挂政策并立即按当年发放额度。"""
    p = db.get(LeavePolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期政策不存在")
    year = datetime.now().year
    assigned = 0
    for emp_id in body.employee_ids:
        emp = db.get(Employee, emp_id)
        if not emp or emp.tenant_id != current.tenant_id:
            continue
        emp.leave_policy_id = p.id
        _apply_policy(db, current.tenant_id, emp.id, p, year)
        assigned += 1
    write_audit(
        db, actor=current, entity_type="leave_policy", entity_id=p.id,
        action="assign", detail={"employee_ids": body.employee_ids, "year": year},
        request=request,
    )
    db.commit()
    return {"assigned": assigned, "year": year}


@router.post("/policies/{policy_id}/unassign")
def unassign_policy(
    policy_id: str,
    body: PolicyAssignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    """解除绑定(已发放的额度保留,不回冲)。"""
    p = db.get(LeavePolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期政策不存在")
    n = 0
    for emp_id in body.employee_ids:
        emp = db.get(Employee, emp_id)
        if emp and emp.tenant_id == current.tenant_id and emp.leave_policy_id == p.id:
            emp.leave_policy_id = None
            n += 1
    write_audit(
        db, actor=current, entity_type="leave_policy", entity_id=p.id,
        action="unassign", detail={"employee_ids": body.employee_ids},
        request=request,
    )
    db.commit()
    return {"unassigned": n}


@router.post("/policies/{policy_id}/run")
def run_policy(
    policy_id: str,
    body: PolicyRunIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LEAVE)),
):
    """年度发放:对全部绑定员工按政策对齐 year 的额度(含上年结转)。"""
    p = db.get(LeavePolicy, policy_id)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期政策不存在")
    emps = db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id,
            Employee.leave_policy_id == p.id,
        )
    ).all()
    for emp in emps:
        _apply_policy(db, current.tenant_id, emp.id, p, body.year)
    write_audit(
        db, actor=current, entity_type="leave_policy", entity_id=p.id,
        action="run", detail={"year": body.year, "employees": len(emps)},
        request=request,
    )
    db.commit()
    return {"applied": len(emps), "year": body.year}
