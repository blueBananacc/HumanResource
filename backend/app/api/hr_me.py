"""员工自助(ESS)API。

所有端点以 ``/me`` 开头,行级过滤到当前登录用户绑定的员工档案
(``require_linked_employee``)。员工账号(role=employee)没有任何业务权限项,
但都能访问这里;其他角色(admin/hr)若绑定了员工档案也能用,
这样管理员自己也可是员工。

- ``GET/PATCH /me/profile``           本人档案(仅可改联系方式/个人字段)
- ``GET /me/leave-balances``          本人当年假期余额
- ``GET/POST /me/leave-applications`` 本人请假单 / 新建申请
- ``POST /me/leave-applications/{id}/cancel``  撤销 pending 申请
- ``GET /me/leave-types``             可申请的类型
- ``GET /me/workday-count``           区间工作日数(请假表单按它算天数)
- ``GET/POST /me/overtime-requests``  加班申请(批准 → 折算调休入账)
- ``POST /me/overtime-requests/{id}/cancel``   撤回/撤销加班申请
- ``POST /me/checkins``               打卡(in/out)
- ``GET /me/checkins``                今日流水
- ``GET /me/attendance``              本人考勤(按月)
- ``GET/POST /me/attendance-requests``补卡申请(批准 → 补写打卡重算)
- ``POST /me/attendance-requests/{id}/cancel`` 撤回/撤销补卡申请
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

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    AppraisalCycle,
    Attendance,
    AttendanceFlag,
    AttendanceRequest,
    Employee,
    EmployeeCheckin,
    ExpenseClaim,
    ExpenseItem,
    ExpenseType,
    Goal,
    LeaveAllocation,
    LeaveApplication,
    LeaveLedgerEntry,
    LeaveType,
    LifecycleEvent,
    LifecycleTask,
    OrgUnit,
    OvertimeRequest,
    SalarySlip,
    User,
)
from app.infra.db import get_db
from app.services.attendance_policy import client_ip, get_policy, ip_allowed
from app.services.audit import write_audit_denied
from app.services.permissions import require_linked_employee
from app.services.work_calendar import count_workdays, workday_hours

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
    # model_fields_set:显式传 null = 清空;不传字段才不覆盖
    if "phone" in body.model_fields_set:
        e.phone = body.phone
    if "personal" in body.model_fields_set:
        e.personal = body.personal
    e.updated_at = _now()
    db.commit()
    return me_profile(db, current)


# ============ leave ============


class MeLeaveTypeOut(BaseModel):
    id: str
    name: str
    is_paid: bool
    is_comp: bool = False


@router.get("/leave-types", response_model=list[MeLeaveTypeOut])
def me_leave_types(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    _me(db, current)
    rows = db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == current.tenant_id)
    ).all()
    return [MeLeaveTypeOut(id=t.id, name=t.name, is_paid=t.is_paid,
                           is_comp=t.is_comp) for t in rows]


@router.get("/workday-count")
def me_workday_count(
    start: date,
    end: date,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """区间内工作日数 — 请假表单按它自动算天数(节假日历感知)。"""
    _me(db, current)
    if end < start:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "结束日期不能早于开始日期")
    if (end - start).days > 366:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "区间最长 366 天")
    return {"workdays": count_workdays(db, current.tenant_id, start, end)}


class MeBalanceOut(BaseModel):
    leave_type_id: str
    leave_type_name: str
    allocated: float
    used: float
    pending: float
    balance: float


def _used_pending_days(
    db: Session, *, employee_id: str, leave_type_id: str, year: int
) -> tuple[float, float]:
    """某年某类型 (已批天数, 待批天数) — 余额与撤销回冲共用口径,
    balance 一律按 allocated - used - pending 算。"""

    def _sum(st: str) -> float:
        return float(db.scalar(
            select(func.coalesce(func.sum(LeaveApplication.days), 0)).where(
                LeaveApplication.employee_id == employee_id,
                LeaveApplication.leave_type_id == leave_type_id,
                LeaveApplication.status == st,
                func.extract("year", LeaveApplication.start_date) == year,
            )
        ) or 0)

    return _sum("approved"), _sum("pending")


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
        used, pending = _used_pending_days(
            db, employee_id=e.id, leave_type_id=a.leave_type_id, year=y
        )
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
    # 不传则由服务端按工作日历自动计算(推荐;客户端算不准节假日)
    days: float | None = Field(None, gt=0, le=365)
    reason: str = ""

    @field_validator("end_date")
    @classmethod
    def _v_range(cls, v: date, info):
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("结束日期不能早于开始日期")
        return v

    @field_validator("days")
    @classmethod
    def _v_days(cls, v: float | None) -> float | None:
        if v is not None and (v * 2) != int(v * 2):
            raise ValueError("请假天数须以 0.5 天为步长")
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
    # 天数按工作日历校验:周末/节假日不算请假,考勤也只标记工作日,
    # 台账扣的天数与考勤标记的天数口径一致。days 只允许等于工作日数
    # 或少一个半天 —— 允许更小值就能用半天额度换整段休假考勤。
    workdays = count_workdays(db, current.tenant_id, body.start_date, body.end_date)
    if workdays == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "日期区间内没有工作日")
    days = body.days if body.days is not None else float(workdays)
    if days > workdays:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "请假天数超过区间内工作日数")
    if days < workdays - 0.5:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "请假天数须与工作日一致(至多相差一个半天)",
        )
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
        days=days, reason=body.reason, status="pending",
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
    # FOR UPDATE:approved 撤销会写返还台账,与审批并发时必须先锁行
    a = db.execute(
        select(LeaveApplication).where(LeaveApplication.id == app_id).with_for_update()
    ).scalar_one_or_none()
    if not a or a.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if a.status not in ("pending", "approved"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前状态不能撤销")
    was_approved = a.status == "approved"
    a.status = "cancelled"
    if was_approved:
        # autoflush 关闭:先把 cancelled 落库,_cancel_ledger_and_attendance
        # 里的 summarize_day 才查不到 approved 请假(on_leave=False),否则
        # 重算又会把这行标回 leave。
        db.flush()
        _cancel_ledger_and_attendance(db, a)
    db.commit()
    db.refresh(a)
    lt = db.get(LeaveType, a.leave_type_id)
    return _app_me_out(a, {a.leave_type_id: lt})


def _cancel_ledger_and_attendance(db: Session, a: LeaveApplication) -> None:
    """撤销已批准申请:回写台账 +,恢复考勤。"""
    from datetime import timedelta

    from app.api.attendance import summarize_day

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
            # 就地重算而不是 delete+重算:autoflush 关闭时 delete 未落库,
            # summarize_day 读到的是同一行(已标删除),改的状态在 flush 时
            # 被 DELETE 吞掉 —— 原实现实际只留了"行被删光"的效果。
            summarize_day(
                db, tenant_id=a.tenant_id, employee_id=a.employee_id, day=day
            )
        day += timedelta(days=1)


# ============ overtime requests(加班 → 调休)============


class MeOvertimeIn(BaseModel):
    leave_type_id: str  # 须为 is_comp=True 的调休类型
    work_date: date
    hours: float = Field(..., gt=0, le=24)
    reason: str = ""

    @field_validator("hours")
    @classmethod
    def _v_hours(cls, v: float) -> float:
        if (v * 2) != int(v * 2):
            raise ValueError("加班时长须以 0.5 小时为步长")
        return v


class MeOvertimeOut(BaseModel):
    id: str
    work_date: date
    hours: float
    leave_type_id: str
    leave_type_name: str
    comp_days: float
    reason: str
    status: str
    decided_at: datetime | None
    decision_note: str | None
    created_at: datetime


def _me_ot_out(db: Session, r: OvertimeRequest, types: dict) -> MeOvertimeOut:
    # 已批准的行读写入库的 comp_days;未批准的按当前日历预算(预览)
    if r.comp_days is not None:
        comp_days = r.comp_days
    else:
        wh = workday_hours(db, r.tenant_id) or 8.0
        comp_days = max(0.5, round(r.hours / wh * 2) / 2)
    return MeOvertimeOut(
        id=r.id, work_date=r.work_date, hours=r.hours,
        leave_type_id=r.leave_type_id,
        leave_type_name=types[r.leave_type_id].name
        if r.leave_type_id in types else "?",
        comp_days=comp_days,
        reason=r.reason, status=r.status, decided_at=r.decided_at,
        decision_note=r.decision_note, created_at=r.created_at,
    )


@router.get("/overtime-requests", response_model=list[MeOvertimeOut])
def me_overtime_list(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    e = _me(db, current)
    rows = db.scalars(
        select(OvertimeRequest)
        .where(OvertimeRequest.employee_id == e.id)
        .order_by(OvertimeRequest.created_at.desc())
        .limit(100)
    ).all()
    types = {t.id: t for t in db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == current.tenant_id)
    )}
    return [_me_ot_out(db, r, types) for r in rows]


@router.post("/overtime-requests", response_model=MeOvertimeOut,
             status_code=status.HTTP_201_CREATED)
def me_overtime_create(
    body: MeOvertimeIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    lt = db.get(LeaveType, body.leave_type_id)
    if not lt or lt.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "假期类型不存在")
    if not lt.is_comp:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "该类型不是调休类型,不能结算加班")
    # 加班是事后申报,日期不能在未来(与补卡申请的未来时刻拦截同口径)
    if body.work_date > _now().date():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "加班日期不能是未来")
    # 同一加班日只允许一条在途申请,与补卡申请同口径
    dup = db.scalars(
        select(OvertimeRequest.id).where(
            OvertimeRequest.employee_id == e.id,
            OvertimeRequest.status == "pending",
            OvertimeRequest.work_date == body.work_date,
        ).limit(1)
    ).first()
    if dup:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "该日期已有待审批的加班申请")
    r = OvertimeRequest(
        tenant_id=current.tenant_id, employee_id=e.id,
        work_date=body.work_date, hours=body.hours,
        leave_type_id=lt.id, reason=body.reason, status="pending",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return _me_ot_out(db, r, {lt.id: lt})


@router.post("/overtime-requests/{req_id}/cancel", response_model=MeOvertimeOut)
def me_overtime_cancel(
    req_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    r = db.execute(
        select(OvertimeRequest).where(OvertimeRequest.id == req_id).with_for_update()
    ).scalar_one_or_none()
    if not r or r.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status not in ("pending", "approved"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前状态不能撤销")
    was_approved = r.status == "approved"
    r.status = "cancelled"
    if was_approved:
        # 回冲调休:allocation -comp_days + 台账 -delta。comp_days 用批准时
        # 写入行的值 — 审批后改过日标准工时,重算会回冲错数。
        if r.comp_days is not None:
            comp_days = r.comp_days
        else:
            wh = workday_hours(db, current.tenant_id) or 8.0
            comp_days = max(0.5, round(r.hours / wh * 2) / 2)
        year = r.work_date.year
        # FOR UPDATE 锁额度行:校验余额与扣减必须原子
        alloc = db.execute(
            select(LeaveAllocation).where(
                LeaveAllocation.employee_id == e.id,
                LeaveAllocation.leave_type_id == r.leave_type_id,
                LeaveAllocation.year == year,
            ).with_for_update()
        ).scalar_one_or_none()
        if alloc is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "调休额度已使用,不能撤销")
        # 剩余额度(alloc - 已批 - 待批)不够回冲则拒绝 — 只比 alloc.days
        # 会把已花掉的天数又吐回来,余额变负。
        used, pending = _used_pending_days(
            db, employee_id=e.id, leave_type_id=r.leave_type_id, year=year
        )
        if alloc.days - used - pending - comp_days < 0:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "调休额度已使用,不能撤销")
        alloc.days -= comp_days
        db.add(LeaveLedgerEntry(
            tenant_id=current.tenant_id, employee_id=e.id,
            leave_type_id=r.leave_type_id, year=year,
            delta_days=-comp_days, ref_type="overtime_cancel", ref_id=r.id,
            note="加班撤销回冲调休",
        ))
    db.commit()
    db.refresh(r)
    lt = db.get(LeaveType, r.leave_type_id)
    return _me_ot_out(db, r, {r.leave_type_id: lt})


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
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    ip = client_ip(request)
    policy = get_policy(db, current.tenant_id)
    if not ip_allowed(policy, ip):
        write_audit_denied(
            db, actor=current, entity_type="employee_checkin",
            entity_id=e.id, action="create",
            reason="ip_not_allowed", detail={"ip": ip}, request=request,
        )
        db.commit()
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "当前网络环境不允许打卡")
    c = EmployeeCheckin(
        tenant_id=current.tenant_id, employee_id=e.id, ts=_now(),
        kind=body.kind, method="web", note=body.note, ip=ip,
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


class MeFlagOut(BaseModel):
    id: str
    day: date
    kind: str
    minutes: int
    detail: dict | None
    excused: bool
    excused_note: str | None


@router.get("/attendance-flags", response_model=list[MeFlagOut])
def me_attendance_flags(
    month: str | None = Query(None, description="YYYY-MM"),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    stmt = select(AttendanceFlag).where(
        AttendanceFlag.employee_id == e.id)
    if month:
        try:
            y, m = month.split("-")
            start = date(int(y), int(m), 1)
            end = (date(int(y), int(m) + 1, 1) if int(m) < 12
                   else date(int(y) + 1, 1, 1))
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "month 格式应为 YYYY-MM")
        stmt = stmt.where(AttendanceFlag.day >= start, AttendanceFlag.day < end)
    rows = db.scalars(stmt.order_by(AttendanceFlag.day.desc())).all()
    return [MeFlagOut(
        id=f.id, day=f.day, kind=f.kind, minutes=f.minutes,
        detail=f.detail, excused=f.excused, excused_note=f.excused_note,
    ) for f in rows]


@router.get("/attendance-summary")
def me_attendance_summary(
    month: str = Query(..., description="YYYY-MM"),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    from app.api.attendance import monthly_summary_rows

    rows = monthly_summary_rows(db, current.tenant_id, month, [e.id])
    return rows[0] if rows else None


# ============ attendance requests(补卡申请)============


class MeAttRequestIn(BaseModel):
    ts: datetime  # 要补打的时刻
    kind: Literal["in", "out"]
    reason: str = ""


class MeAttRequestOut(BaseModel):
    id: str
    ts: datetime
    kind: str
    reason: str
    status: str
    decided_at: datetime | None
    decision_note: str | None
    created_at: datetime


def _me_atr_out(r: AttendanceRequest) -> MeAttRequestOut:
    return MeAttRequestOut(
        id=r.id, ts=r.ts, kind=r.kind, reason=r.reason, status=r.status,
        decided_at=r.decided_at, decision_note=r.decision_note,
        created_at=r.created_at,
    )


@router.get("/attendance-requests", response_model=list[MeAttRequestOut])
def me_att_requests(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    e = _me(db, current)
    rows = db.scalars(
        select(AttendanceRequest)
        .where(AttendanceRequest.employee_id == e.id)
        .order_by(AttendanceRequest.created_at.desc())
        .limit(100)
    ).all()
    return [_me_atr_out(r) for r in rows]


@router.post("/attendance-requests", response_model=MeAttRequestOut,
             status_code=status.HTTP_201_CREATED)
def me_att_request_create(
    body: MeAttRequestIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    # 入参可能带时区(ISO Z),统一压成与 DB 一致的 naive UTC
    ts = body.ts
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    # 补卡只补过去/当下的打卡时刻(留 5 分钟钟差余量)
    if ts > _now() + timedelta(minutes=5):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "补卡时刻不能是未来")
    day_start = datetime.combine(ts.date(), datetime.min.time())
    dup = db.scalars(
        select(AttendanceRequest.id).where(
            AttendanceRequest.employee_id == e.id,
            AttendanceRequest.status == "pending",
            AttendanceRequest.kind == body.kind,
            AttendanceRequest.ts >= day_start,
            AttendanceRequest.ts < day_start + timedelta(days=1),
        ).limit(1)
    ).first()
    if dup:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "该时段已有待审批的同向补卡申请")
    r = AttendanceRequest(
        tenant_id=current.tenant_id, employee_id=e.id, ts=ts,
        kind=body.kind, reason=body.reason, status="pending",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return _me_atr_out(r)


@router.post("/attendance-requests/{req_id}/cancel", response_model=MeAttRequestOut)
def me_att_request_cancel(
    req_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    e = _me(db, current)
    r = db.execute(
        select(AttendanceRequest)
        .where(AttendanceRequest.id == req_id).with_for_update()
    ).scalar_one_or_none()
    if not r or r.employee_id != e.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status not in ("pending", "approved"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "当前状态不能撤销")
    was_approved = r.status == "approved"
    r.status = "cancelled"
    if was_approved:
        # 批准时补写的那条 checkin(method=request)删掉,再重算当天
        from app.api.attendance import summarize_day

        ck = db.scalars(
            select(EmployeeCheckin).where(
                EmployeeCheckin.employee_id == e.id,
                EmployeeCheckin.ts == r.ts,
                EmployeeCheckin.kind == r.kind,
                EmployeeCheckin.method == "request",
            ).limit(1)
        ).first()
        if ck:
            db.delete(ck)
            db.flush()
        summarize_day(
            db, tenant_id=current.tenant_id, employee_id=e.id, day=r.ts.date()
        )
    db.commit()
    db.refresh(r)
    return _me_atr_out(r)


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


class MeExpenseTypeOut(BaseModel):
    id: str
    name: str


@router.get("/expense-types", response_model=list[MeExpenseTypeOut])
def me_expense_types(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    _me(db, current)
    rows = db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == current.tenant_id)
    ).all()
    return [MeExpenseTypeOut(id=t.id, name=t.name) for t in rows]


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
    # group items — dict keyed by claim_id would keep only the last row per claim
    by_claim: dict[str, list[ExpenseItem]] = {}
    for i in db.scalars(
        select(ExpenseItem).where(
            ExpenseItem.claim_id.in_([c.id for c in rows] or ["__none__"])
        )
    ):
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
    cycle = db.get(AppraisalCycle, a.cycle_id)
    if not cycle or cycle.status != "open":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "考核周期未开启或已关闭")
    a.self_review = body.self_review.strip()
    a.self_rating = body.self_rating
    a.status = "self_submitted"
    a.updated_at = _now()
    db.commit()
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
