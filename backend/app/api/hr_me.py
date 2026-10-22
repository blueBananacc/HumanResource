"""员工自助(ESS)API — 对应 Frappe HRMS 的 "Employee Self Service"。

所有端点以 ``/me`` 开头,行级过滤到当前登录用户绑定的员工档案
(``require_linked_employee``)。员工账号(role=employee)没有任何业务权限项,
但都能访问这里;其他角色(admin/hr)若绑定了员工档案也能用,
这样管理员自己也可是员工。

- ``GET/PATCH /me/profile``           本人档案(仅可改联系方式/个人字段)
- ``GET /me/leave-balances``          本人当年假期余额
- ``GET/POST /me/leave-applications`` 本人请假单 / 新建申请
- ``POST /me/leave-applications/{id}/cancel``  撤销 pending 申请
- ``GET /me/leave-types``             可申请的类型
- ``POST /me/checkins``               打卡(in/out)
- ``GET /me/checkins``                今日流水
- ``GET /me/attendance``              本人考勤(按月)
- ``GET /me/salary-slips``            本人已发放工资条
- ``GET /me/salary-slips/{id}``       工资条详情
- ``GET/POST /me/expense-claims``     本人报销单 / 新建
- ``GET /me/expense-claims/{id}``     报销单详情
- ``GET /me/goals`` ``PATCH /me/goals/{id}`` 本人目标(只改 progress)
- ``GET /me/appraisals`` ``POST /me/appraisals/{id}/submit-self`` 考核自评
- ``GET /me/lifecycle``               本人生命周期事件+任务
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    AppraisalCycle,
    Attendance,
    Employee,
    EmployeeCheckin,
    ExpenseClaim,
    ExpenseItem,
    ExpenseType,
    Goal,
    LeaveApplication,
    LeaveLedgerEntry,
    LeaveType,
    LifecycleEvent,
    LifecycleTask,
    SalarySlip,
    LeaveAllocation,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.permissions import require_linked_employee

router = APIRouter(prefix="/me", tags=["hris-ess"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _me(db: Session, user: User) -> Employee:
    emp_id = require_linked_employee(db, user)
    return db.get(Employee, emp_id)


# ============ profile ============


class MeProfileOut(BaseModel):
    id: str
    employee_no: str
    name: str
    org_unit_name: str | None
    designation: str | None
    reports_to_name: str | None
    work_email: str | None
    phone: str | None
    hire_date: date | None
    employment_type: str
    status: str
    personal: dict | None


class MeProfileIn(BaseModel):
    phone: str | None = None
    personal: dict | None = None


@router.get("/profile", response_model=MeProfileOut)
def me_profile(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    e = _me(db, current)
    org = db.get(OrgUnit, e.org_unit_id) if e.org_unit_id else None
    mgr = db.get(Employee, e.reports_to_id) if e.reports_to_id else None
    return MeProfileOut(
        id=e.id, employee_no=e.employee_no, name=e.name,
        org_unit_name=org.name if org else None, designation=e.designation,
        reports_to_name=mgr.name if mgr else None,
        work_email=e.work_email, phone=e.phone, hire_date=e.hire_date,
        employment_type=e.employment_type, status=e.status, personal=e.personal,
    )


@router.patch("/profile", response_model=MeProfileOut)
def me_profile_update(
    body: MeProfileIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    if body.phone is not None:
        e.phone = body.phone
    if body.personal is not None:
        merged = dict(e.personal or {})
        merged.update(body.personal)
        e.personal = merged
    e.updated_at = _now()
    db.commit()
    return me_profile(db, current)


# ============ leave ============


class MeLeaveTypeOut(BaseModel):
    id: str
    name: str
    is_paid: bool


@router.get("/leave-types", response_model=list[MeLeaveTypeOut])
def me_leave_types(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    _me(db, current)
    rows = db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == current.tenant_id)
    ).all()
    return [MeLeaveTypeOut(id=t.id, name=t.name, is_paid=t.is_paid) for t in rows]


class MeBalanceOut(BaseModel):
    leave_type_id: str
    leave_type_name: str
    allocated: float
    used: float
    pending: float
    balance: float


@router.get("/leave-balances", response_model=list[MeBalanceOut])
def me_leave_balances(
    year: int | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    y = year or datetime.now().year
    allocs = db.scalars(
        select(LeaveAllocation).where(
            LeaveAllocation.employee_id == e.id, LeaveAllocation.year == y
        )
    ).all()
    out = []
    for a in allocs:
        lt = db.get(LeaveType, a.leave_type_id)
        used = db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == e.id,
                LeaveApplication.leave_type_id == a.leave_type_id,
                LeaveApplication.status == "approved",
                func.extract("year", LeaveApplication.start_date) == y,
            )
        ) or 0
        pending = db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == e.id,
                LeaveApplication.leave_type_id == a.leave_type_id,
                LeaveApplication.status == "pending",
                func.extract("year", LeaveApplication.start_date) == y,
            )
        ) or 0
        out.append(MeBalanceOut(
            leave_type_id=a.leave_type_id,
            leave_type_name=lt.name if lt else "?",
            allocated=a.days, used=used, pending=pending,
            balance=a.days - used - pending,
        ))
    return out


class MeApplicationIn(BaseModel):
    leave_type_id: str
    start_date: date
    end_date: date
    days: float = Field(..., gt=0, le=365)
    reason: str = ""

    @field_validator("end_date")
    @classmethod
    def _v_range(cls, v: date, info):
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("结束日期不能早于开始日期")
        return v


class MeApplicationOut(BaseModel):
    id: str
    leave_type_id: str
    leave_type_name: str
    start_date: date
    end_date: date
    days: float
    reason: str
    status: str
    decided_at: datetime | None
    decision_note: str | None
    created_at: datetime


def _app_me_out(a: LeaveApplication, types: dict) -> MeApplicationOut:
    return MeApplicationOut(
        id=a.id, leave_type_id=a.leave_type_id,
        leave_type_name=types[a.leave_type_id].name if a.leave_type_id in types else "?",
        start_date=a.start_date, end_date=a.end_date, days=a.days,
        reason=a.reason, status=a.status, decided_at=a.decided_at,
        decision_note=a.decision_note, created_at=a.created_at,
    )


@router.get("/leave-applications", response_model=list[MeApplicationOut])
def me_leave_applications(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    e = _me(db, current)
    rows = db.scalars(
        select(LeaveApplication)
        .where(LeaveApplication.employee_id == e.id)
        .order_by(LeaveApplication.created_at.desc())
        .limit(100)
    ).all()
    types = {t.id: t for t in db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == current.tenant_id)
    )}
    return [_app_me_out(a, types) for a in rows]


@router.post("/leave-applications", response_model=MeApplicationOut,
             status_code=status.HTTP_201_CREATED)
def me_leave_create(
    body: MeApplicationIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    lt = db.get(LeaveType, body.leave_type_id)
    if not lt or lt.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期类型不存在")
    span = (body.end_date - body.start_date).days + 1
    if body.days > span:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "请假天数超过日期区间")
    overlap = db.scalars(
        select(LeaveApplication.id).where(
            LeaveApplication.employee_id == e.id,
            LeaveApplication.status.in_(["pending", "approved"]),
            LeaveApplication.start_date <= body.end_date,
            LeaveApplication.end_date >= body.start_date,
        ).limit(1)
    ).first()
    if overlap:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "与已有请假申请日期重叠")
    a = LeaveApplication(
        tenant_id=current.tenant_id, employee_id=e.id,
        leave_type_id=lt.id, start_date=body.start_date, end_date=body.end_date,
        days=body.days, reason=body.reason, status="pending",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    types = {lt.id: lt}
    return _app_me_out(a, types)


@router.post("/leave-applications/{app_id}/cancel", response_model=MeApplicationOut)
def me_leave_cancel(
    app_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    a = db.get(LeaveApplication, app_id)
    if not a or a.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status not in ("pending", "approved"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前状态不能撤销")
    was_approved = a.status == "approved"
    a.status = "cancelled"
    if was_approved:
        _cancel_ledger_and_attendance(db, a)
    db.commit()
    db.refresh(a)
    lt = db.get(LeaveType, a.leave_type_id)
    return _app_me_out(a, {a.leave_type_id: lt})


def _cancel_ledger_and_attendance(db: Session, a: LeaveApplication) -> None:
    """撤销已批准申请:回写台账 +,恢复考勤。"""
    from app.api.attendance import summarize_day
    from datetime import timedelta

    db.add(LeaveLedgerEntry(
        tenant_id=a.tenant_id, employee_id=a.employee_id,
        leave_type_id=a.leave_type_id, year=a.start_date.year,
        delta_days=a.days, ref_type="application_cancel", ref_id=a.id,
        note="请假撤销返还",
    ))
    day = a.start_date
    while day <= a.end_date:
        att = db.scalars(
            select(Attendance).where(
                Attendance.employee_id == a.employee_id, Attendance.day == day
            )
        ).first()
        if att and att.source == "leave":
            db.delete(att)
            summarize_day(
                db, tenant_id=a.tenant_id, employee_id=a.employee_id, day=day
            )
        day += timedelta(days=1)


# ============ checkins / attendance ============


class CheckinIn(BaseModel):
    kind: Literal["in", "out"]
    note: str | None = None


class CheckinOut(BaseModel):
    id: str
    ts: datetime
    kind: str
    method: str
    note: str | None


@router.post("/checkins", response_model=CheckinOut, status_code=status.HTTP_201_CREATED)
def me_checkin(
    body: CheckinIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    c = EmployeeCheckin(
        tenant_id=current.tenant_id, employee_id=e.id, ts=_now(),
        kind=body.kind, method="web", note=body.note,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    # 顺手刷新当日 attendance
    from app.api.attendance import summarize_day
    summarize_day(
        db, tenant_id=current.tenant_id, employee_id=e.id, day=c.ts.date()
    )
    db.commit()
    return CheckinOut(id=c.id, ts=c.ts, kind=c.kind, method=c.method, note=c.note)


@router.get("/checkins", response_model=list[CheckinOut])
def me_checkins(
    day: date | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    d = day or _now().date()
    start = datetime.combine(d, datetime.min.time())
    rows = db.scalars(
        select(EmployeeCheckin).where(
            EmployeeCheckin.employee_id == e.id,
            EmployeeCheckin.ts >= start,
            EmployeeCheckin.ts < start + timedelta(days=1),
        ).order_by(EmployeeCheckin.ts)
    ).all()
    return [CheckinOut(id=c.id, ts=c.ts, kind=c.kind, method=c.method, note=c.note)
            for c in rows]


class MeAttendanceOut(BaseModel):
    day: date
    status: str
    first_in: datetime | None
    last_out: datetime | None
    work_hours: float | None


@router.get("/attendance", response_model=list[MeAttendanceOut])
def me_attendance(
    month: str | None = Query(None, description="YYYY-MM"),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    now = _now()
    if month:
        try:
            y, m = month.split("-")
            start = date(int(y), int(m), 1)
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "month 格式应为 YYYY-MM")
    else:
        start = date(now.year, now.month, 1)
    end = (date(start.year, start.month + 1, 1) if start.month < 12
           else date(start.year + 1, 1, 1))
    rows = db.scalars(
        select(Attendance).where(
            Attendance.employee_id == e.id,
            Attendance.day >= start, Attendance.day < end,
        ).order_by(Attendance.day)
    ).all()
    return [MeAttendanceOut(
        day=a.day, status=a.status, first_in=a.first_in,
        last_out=a.last_out, work_hours=a.work_hours,
    ) for a in rows]


# ============ salary slips ============


class SlipItem(BaseModel):
    label: str
    kind: str
    amount: float


class MeSlipOut(BaseModel):
    id: str
    period: str
    gross: float
    deductions: float
    net: float
    currency: str
    items: list[SlipItem] | None
    issued_at: datetime | None
    note: str | None


@router.get("/salary-slips", response_model=list[MeSlipOut])
def me_slips(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    e = _me(db, current)
    rows = db.scalars(
        select(SalarySlip).where(
            SalarySlip.employee_id == e.id, SalarySlip.status == "issued"
        ).order_by(SalarySlip.period.desc())
    ).all()
    return [MeSlipOut(
        id=s.id, period=s.period, gross=s.gross, deductions=s.deductions,
        net=s.net, currency=s.currency, items=s.items, issued_at=s.issued_at,
        note=s.note,
    ) for s in rows]


@router.get("/salary-slips/{slip_id}", response_model=MeSlipOut)
def me_slip_detail(
    slip_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    s = db.get(SalarySlip, slip_id)
    if not s or s.employee_id != e.id or s.status != "issued":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "工资条不存在")
    return MeSlipOut(
        id=s.id, period=s.period, gross=s.gross, deductions=s.deductions,
        net=s.net, currency=s.currency, items=s.items, issued_at=s.issued_at,
        note=s.note,
    )


# ============ expense claims ============


class ExpenseItemIn(BaseModel):
    type_id: str | None = None
    occurred_on: date
    amount: float = Field(..., gt=0)
    note: str | None = None


class MeClaimIn(BaseModel):
    title: str
    items: list[ExpenseItemIn] = Field(..., min_length=1)


class MeClaimItemOut(BaseModel):
    id: str
    type_id: str | None
    type_name: str | None
    occurred_on: date
    amount: float
    note: str | None


class MeClaimOut(BaseModel):
    id: str
    title: str
    total: float
    status: str
    items: list[MeClaimItemOut]
    decided_at: datetime | None
    decision_note: str | None
    created_at: datetime


@router.get("/expense-types", response_model=list[MeLeaveTypeOut])
def me_expense_types(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    _me(db, current)
    rows = db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == current.tenant_id)
    ).all()
    return [MeLeaveTypeOut(id=t.id, name=t.name, is_paid=True) for t in rows]


@router.get("/expense-claims", response_model=list[MeClaimOut])
def me_claims(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    e = _me(db, current)
    rows = db.scalars(
        select(ExpenseClaim).where(ExpenseClaim.employee_id == e.id)
        .order_by(ExpenseClaim.created_at.desc()).limit(100)
    ).all()
    types = {t.id: t for t in db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == current.tenant_id)
    )}
    items = {i.claim_id: i for i in db.scalars(
        select(ExpenseItem).where(
            ExpenseItem.claim_id.in_([c.id for c in rows] or ["__none__"])
        )
    )}
    # group items
    by_claim: dict[str, list[ExpenseItem]] = {}
    for i in items.values():
        by_claim.setdefault(i.claim_id, []).append(i)
    out = []
    for c in rows:
        out.append(MeClaimOut(
            id=c.id, title=c.title, total=c.total, status=c.status,
            items=[MeClaimItemOut(
                id=i.id, type_id=i.type_id,
                type_name=types[i.type_id].name if i.type_id in types else None,
                occurred_on=i.occurred_on, amount=i.amount, note=i.note,
            ) for i in sorted(by_claim.get(c.id, []), key=lambda x: x.occurred_on)],
            decided_at=c.decided_at, decision_note=c.decision_note,
            created_at=c.created_at,
        ))
    return out


@router.get("/expense-claims/{claim_id}", response_model=MeClaimOut)
def me_claim_detail(
    claim_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    c = db.get(ExpenseClaim, claim_id)
    if not c or c.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "报销单不存在")
    types = {t.id: t for t in db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == current.tenant_id)
    )}
    items = db.scalars(
        select(ExpenseItem).where(ExpenseItem.claim_id == c.id)
    ).all()
    return MeClaimOut(
        id=c.id, title=c.title, total=c.total, status=c.status,
        items=[MeClaimItemOut(
            id=i.id, type_id=i.type_id,
            type_name=types[i.type_id].name if i.type_id in types else None,
            occurred_on=i.occurred_on, amount=i.amount, note=i.note,
        ) for i in sorted(items, key=lambda x: x.occurred_on)],
        decided_at=c.decided_at, decision_note=c.decision_note,
        created_at=c.created_at,
    )


@router.post("/expense-claims", response_model=MeClaimOut,
             status_code=status.HTTP_201_CREATED)
def me_claim_create(
    body: MeClaimIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    title = body.title.strip()
    if not title:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "标题不能为空")
    for it in body.items:
        if it.type_id:
            t = db.get(ExpenseType, it.type_id)
            if not t or t.tenant_id != current.tenant_id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "报销类型不存在")
    c = ExpenseClaim(
        tenant_id=current.tenant_id, employee_id=e.id, title=title,
        total=sum(i.amount for i in body.items), status="pending",
    )
    db.add(c)
    db.flush()
    for it in body.items:
        db.add(ExpenseItem(
            tenant_id=current.tenant_id, claim_id=c.id, type_id=it.type_id,
            occurred_on=it.occurred_on, amount=it.amount, note=it.note,
        ))
    db.commit()
    return me_claim_detail(c.id, db, current)


# ============ goals / appraisals ============


class MeGoalOut(BaseModel):
    id: str
    title: str
    description: str
    due_date: date | None
    progress: int
    status: str


@router.get("/goals", response_model=list[MeGoalOut])
def me_goals(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    e = _me(db, current)
    rows = db.scalars(
        select(Goal).where(Goal.employee_id == e.id).order_by(Goal.created_at.desc())
    ).all()
    return [MeGoalOut(
        id=g.id, title=g.title, description=g.description, due_date=g.due_date,
        progress=g.progress, status=g.status,
    ) for g in rows]


class MeGoalIn(BaseModel):
    progress: int = Field(..., ge=0, le=100)


@router.patch("/goals/{goal_id}", response_model=MeGoalOut)
def me_goal_update(
    goal_id: str,
    body: MeGoalIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    g = db.get(Goal, goal_id)
    if not g or g.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "目标不存在")
    g.progress = body.progress
    if g.status in ("active", "done"):
        g.status = "done" if body.progress >= 100 else "active"
    db.commit()
    return MeGoalOut(
        id=g.id, title=g.title, description=g.description, due_date=g.due_date,
        progress=g.progress, status=g.status,
    )


class MeAppraisalOut(BaseModel):
    id: str
    cycle_id: str
    cycle_name: str
    status: str
    self_review: str | None
    self_rating: int | None
    manager_review: str | None
    manager_rating: int | None


@router.get("/appraisals", response_model=list[MeAppraisalOut])
def me_appraisals(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    e = _me(db, current)
    rows = db.scalars(
        select(Appraisal).where(Appraisal.employee_id == e.id)
        .order_by(Appraisal.created_at.desc())
    ).all()
    cycles = {c.id: c for c in db.scalars(
        select(AppraisalCycle).where(AppraisalCycle.tenant_id == current.tenant_id)
    )}
    return [MeAppraisalOut(
        id=a.id, cycle_id=a.cycle_id,
        cycle_name=cycles[a.cycle_id].name if a.cycle_id in cycles else "?",
        status=a.status, self_review=a.self_review, self_rating=a.self_rating,
        manager_review=a.manager_review if a.status == "reviewed" else None,
        manager_rating=a.manager_rating if a.status == "reviewed" else None,
    ) for a in rows]


class SelfReviewIn(BaseModel):
    self_review: str
    self_rating: int = Field(..., ge=1, le=5)


@router.post("/appraisals/{appraisal_id}/submit-self", response_model=MeAppraisalOut)
def me_appraisal_submit(
    appraisal_id: str,
    body: SelfReviewIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    a = db.get(Appraisal, appraisal_id)
    if not a or a.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "考核表不存在")
    if a.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已提交,不能重复自评")
    a.self_review = body.self_review.strip()
    a.self_rating = body.self_rating
    a.status = "self_submitted"
    a.updated_at = _now()
    db.commit()
    cycle = db.get(AppraisalCycle, a.cycle_id)
    return MeAppraisalOut(
        id=a.id, cycle_id=a.cycle_id, cycle_name=cycle.name if cycle else "?",
        status=a.status, self_review=a.self_review, self_rating=a.self_rating,
        manager_review=None, manager_rating=None,
    )


# ============ lifecycle ============


class MeTaskOut(BaseModel):
    id: str
    title: str
    done: bool
    done_at: datetime | None
    sort: int


class MeEventOut(BaseModel):
    id: str
    kind: str
    status: str
    effective_date: date | None
    note: str | None
    tasks: list[MeTaskOut]


@router.get("/lifecycle", response_model=list[MeEventOut])
def me_lifecycle(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    e = _me(db, current)
    events = db.scalars(
        select(LifecycleEvent).where(LifecycleEvent.employee_id == e.id)
        .order_by(LifecycleEvent.created_at.desc())
    ).all()
    tasks = db.scalars(
        select(LifecycleTask).where(
            LifecycleTask.event_id.in_([ev.id for ev in events] or ["__none__"])
        ).order_by(LifecycleTask.sort)
    ).all()
    by_event: dict[str, list[LifecycleTask]] = {}
    for t in tasks:
        by_event.setdefault(t.event_id, []).append(t)
    return [MeEventOut(
        id=ev.id, kind=ev.kind, status=ev.status,
        effective_date=ev.effective_date, note=ev.note,
        tasks=[MeTaskOut(id=t.id, title=t.title, done=t.done,
                         done_at=t.done_at, sort=t.sort)
               for t in by_event.get(ev.id, [])],
    ) for ev in events]
