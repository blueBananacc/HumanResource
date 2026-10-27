"""考勤 API(HR 管理端)。

- ``GET /attendance``                    考勤记录(按月/员工/状态过滤)
- ``PUT /attendance/{id}``               手工修正状态(manage_attendance)
- ``POST /attendance/summarize``         把指定日期段的 checkin 汇总成 attendance
- ``GET /attendance/checkins``           打卡流水
- ``GET /attendance/requests``           补卡申请列表(scope=all/team/mine)
- ``POST /attendance/requests/{id}/approve|reject`` 审批 → 补写 checkin 重算
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Attendance,
    AttendanceConfirmation,
    AttendanceFlag,
    AttendancePolicy,
    AttendanceRequest,
    Employee,
    EmployeeCheckin,
    LeaveApplication,
    OrgUnit,
    OvertimeRequest,
    User,
    WorkCalendar,
)
from app.infra.db import get_db
from app.services.attendance_policy import evaluate_flags, get_policy, validate_cidrs
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_APPROVE_LEAVE,
    PERM_MANAGE_ATTENDANCE,
    PERM_VIEW_REPORTS,
    can_decide_for_employee,
    get_linked_employee_id,
    require_permission,
    team_ids_with_self,
    visible_employee_ids,
)
from app.services.work_calendar import (
    holiday_map,
    is_workday,
    is_workday_day,
    workday_set,
)

router = APIRouter(prefix="/attendance", tags=["hris-attendance"], dependencies=[Depends(require_feature("hr.suite"))])


class AttendanceOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    day: date
    status: str
    first_in: datetime | None
    last_out: datetime | None
    work_hours: float | None
    source: str


class CheckinOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    ts: datetime
    kind: str
    method: str
    note: str | None


class AttendanceIn(BaseModel):
    status: Literal["present", "absent", "leave", "remote", "holiday"]
    first_in: datetime | None = None
    last_out: datetime | None = None


class SummarizeIn(BaseModel):
    start: date
    end: date


def _maps(db: Session, tenant_id: str):
    emps = {
        e.id: e
        for e in db.scalars(select(Employee).where(Employee.tenant_id == tenant_id))
    }
    orgs = {
        o.id: o.name
        for o in db.scalars(select(OrgUnit).where(OrgUnit.tenant_id == tenant_id))
    }
    return emps, orgs


def _out(a: Attendance, emps: dict, orgs: dict) -> AttendanceOut:
    e = emps.get(a.employee_id)
    return AttendanceOut(
        id=a.id, employee_id=a.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        day=a.day, status=a.status, first_in=a.first_in, last_out=a.last_out,
        work_hours=a.work_hours, source=a.source,
    )


def summarize_day(
    db: Session, *, tenant_id: str, employee_id: str, day: date,
    workdays: frozenset[int] | set[int] | None = None,
    hols: dict | None = None,
    policy: "AttendancePolicy | None" = None,
    calendar: "WorkCalendar | None" = None,
) -> Attendance:
    """单日汇总:checkin → first_in/last_out;approved 请假覆盖的工作日 → leave
    (非工作日即使被请假区间覆盖也只算 holiday,与台账只扣工作日同口径);
    工作日无记录 → absent;否则 present(有 checkin)。

    ``workdays``/``hols``/``policy``/``calendar`` 传预先按整段拉好的值
    (批量 summarize 防 N+1);不传则按单日查(is_workday_day)。
    """
    day_start = datetime.combine(day, datetime.min.time())
    day_end = day_start + timedelta(days=1)
    checkins = db.scalars(
        select(EmployeeCheckin).where(
            EmployeeCheckin.employee_id == employee_id,
            EmployeeCheckin.ts >= day_start,
            EmployeeCheckin.ts < day_end,
        ).order_by(EmployeeCheckin.ts)
    ).all()
    first_in = next((c.ts for c in checkins if c.kind == "in"), None)
    last_out = next((c.ts for c in reversed(checkins) if c.kind == "out"), None)

    on_leave = db.scalars(
        select(LeaveApplication.id).where(
            LeaveApplication.tenant_id == tenant_id,
            LeaveApplication.employee_id == employee_id,
            LeaveApplication.status == "approved",
            LeaveApplication.start_date <= day,
            LeaveApplication.end_date >= day,
        ).limit(1)
    ).first()

    att = db.scalars(
        select(Attendance).where(
            Attendance.employee_id == employee_id, Attendance.day == day
        )
    ).first()
    if att is None:
        att = Attendance(
            tenant_id=tenant_id, employee_id=employee_id, day=day, source="checkin"
        )
        db.add(att)
        db.flush()  # flag 关联需要 att.id 外键可引用
    wd: bool | None = None
    if att.source != "manual":
        if not on_leave and first_in:
            # 最常见路径:有打卡无请假 → present,完全不用查日历
            att.status = "present"
            att.source = "checkin"
        else:
            # 日历查询只在这一分支发生:leave/absent/holiday 都要判工作日。
            # 请假覆盖的非工作日不能标 leave —— 台账只扣工作日,
            # 否则整段重算又会把周末/节假日补成 leave,口径两边对不上。
            if workdays is not None and hols is not None:
                wd = is_workday(day, workdays, hols)
            else:
                wd = is_workday_day(db, tenant_id, day)
            if on_leave and wd:
                att.status = "leave"
                att.source = "leave"
            elif first_in:
                att.status = "present"
                att.source = "checkin"
            else:
                att.status = "absent" if wd else "holiday"
                att.source = "checkin"
        # 直接赋计算值而非 or 合并:补卡撤销删 checkin 后重算,旧 first_in /
        # last_out / work_hours 必须清掉,否则 absent 行上还挂着已撤销的时刻
        att.first_in = first_in
        att.last_out = last_out
        att.work_hours = (
            round((last_out - first_in).total_seconds() / 3600, 2)
            if first_in and last_out and last_out > first_in
            else None
        )
    # 迟到/早退异常台账:present + 工作日才评;其余状态清理可能过期的旧 flag。
    # 工作日口径已知则复用(免二次查询);fast path 需要时才补查一次。
    if att.status == "present" and (att.first_in or att.last_out):
        if wd is None:
            if workdays is not None and hols is not None:
                wd = is_workday(day, workdays, hols)
            else:
                wd = is_workday_day(db, tenant_id, day)
        evaluate_flags(
            db, att=att, is_workday=bool(wd), calendar=calendar, policy=policy
        )
    else:
        evaluate_flags(
            db, att=att, is_workday=False, calendar=calendar, policy=policy
        )
    return att


@router.get("", response_model=PageOut[AttendanceOut])
def list_attendance(
    pagination: tuple = Depends(paginate_params),
    month: str | None = Query(None, description="YYYY-MM"),
    employee_id: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    scope: Literal["all", "team"] = "all",
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(Attendance).where(Attendance.tenant_id == current.tenant_id)
    if month:
        try:
            y, m = month.split("-")
            start = date(int(y), int(m), 1)
            end = (date(int(y), int(m) + 1, 1) if int(m) < 12
                   else date(int(y) + 1, 1, 1))
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "month 格式应为 YYYY-MM")
        stmt = stmt.where(Attendance.day >= start, Attendance.day < end)
    if employee_id:
        stmt = stmt.where(Attendance.employee_id == employee_id)
    if status_filter:
        stmt = stmt.where(Attendance.status == status_filter)
    if scope == "team":
        # 显式"我管理的":直属下属+自己,admin/hr 也按汇报线过滤
        own = db.execute(
            select(Employee.id).where(
                Employee.tenant_id == current.tenant_id,
                Employee.user_id == current.id,
            )
        ).scalar()
        ids = [
            r[0]
            for r in db.execute(
                select(Employee.id).where(
                    Employee.tenant_id == current.tenant_id,
                    Employee.reports_to_id == (own or "__none__"),
                )
            ).all()
        ]
        if own:
            ids.append(own)
        stmt = stmt.where(Attendance.employee_id.in_(ids or ["__none__"]))
    else:
        emp_ids = visible_employee_ids(db, current)
        if emp_ids is not None:
            stmt = stmt.where(Attendance.employee_id.in_(emp_ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Attendance.day.desc()).limit(limit).offset(offset)
    ).all()
    emps, orgs = _maps(db, current.tenant_id)
    return PageOut(
        items=[_out(a, emps, orgs) for a in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/summarize")
def summarize(
    body: SummarizeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    if body.end < body.start or (body.end - body.start).days > 92:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "日期范围不合法(最多 92 天)")
    emp_ids = [e.id for e in db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id, Employee.status == "active"
        )
    )]
    # 日历/策略按整段拉一次传给 summarize_day,避免每个员工每天再查
    wd = workday_set(db, current.tenant_id)
    hols = holiday_map(db, current.tenant_id, body.start, body.end)
    policy = get_policy(db, current.tenant_id)
    cal = db.scalars(
        select(WorkCalendar).where(WorkCalendar.tenant_id == current.tenant_id)
    ).first()
    day = body.start
    n = 0
    while day <= body.end:
        for emp_id in emp_ids:
            summarize_day(
                db, tenant_id=current.tenant_id, employee_id=emp_id, day=day,
                workdays=wd, hols=hols, policy=policy, calendar=cal,
            )
            n += 1
        day += timedelta(days=1)
    db.commit()
    return {"summarized": n}


@router.get("/checkins", response_model=PageOut[CheckinOut])
def list_checkins(
    pagination: tuple = Depends(paginate_params),
    employee_id: str | None = None,
    day: date | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(EmployeeCheckin).where(
        EmployeeCheckin.tenant_id == current.tenant_id
    )
    if employee_id:
        stmt = stmt.where(EmployeeCheckin.employee_id == employee_id)
    if day:
        start = datetime.combine(day, datetime.min.time())
        stmt = stmt.where(
            EmployeeCheckin.ts >= start, EmployeeCheckin.ts < start + timedelta(days=1)
        )
    emp_ids = visible_employee_ids(db, current)
    if emp_ids is not None:
        stmt = stmt.where(EmployeeCheckin.employee_id.in_(emp_ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(EmployeeCheckin.ts.desc()).limit(limit).offset(offset)
    ).all()
    emps = {
        e.id: e
        for e in db.scalars(select(Employee).where(Employee.tenant_id == current.tenant_id))
    }
    return PageOut(
        items=[CheckinOut(
            id=c.id, employee_id=c.employee_id,
            employee_name=emps[c.employee_id].name if c.employee_id in emps else "?",
            ts=c.ts, kind=c.kind, method=c.method, note=c.note,
        ) for c in rows],
        total=total, limit=limit, offset=offset,
    )


# ============ 补卡申请审批(发起端在 hr_me:/me/attendance-requests)============


class AttendanceRequestOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    ts: datetime
    kind: str
    reason: str
    status: str
    approver_id: str | None
    approver_email: str | None
    decided_at: datetime | None
    decision_note: str | None
    own_request: bool
    created_at: datetime


class ReqDecisionIn(BaseModel):
    note: str | None = None


def _req_out(
    r: AttendanceRequest, emps: dict, orgs: dict, users: dict,
    me_id: str | None = None,
) -> AttendanceRequestOut:
    e = emps.get(r.employee_id)
    return AttendanceRequestOut(
        id=r.id, employee_id=r.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        ts=r.ts, kind=r.kind, reason=r.reason, status=r.status,
        approver_id=r.approver_id,
        approver_email=users.get(r.approver_id) if r.approver_id else None,
        decided_at=r.decided_at, decision_note=r.decision_note,
        own_request=bool(e and e.user_id == me_id),
        created_at=r.created_at,
    )


def _req_maps(db: Session, tenant_id: str):
    emps, orgs = _maps(db, tenant_id)
    users = {
        u.id: u.email
        for u in db.scalars(select(User).where(User.tenant_id == tenant_id))
    }
    return emps, orgs, users


def _locked_req(db: Session, req_id: str) -> AttendanceRequest | None:
    """FOR UPDATE:批准会写 checkin,并发双击会记两次。"""
    return db.execute(
        select(AttendanceRequest)
        .where(AttendanceRequest.id == req_id)
        .with_for_update()
    ).scalar_one_or_none()


@router.get("/requests", response_model=PageOut[AttendanceRequestOut])
def list_requests(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    scope: Literal["all", "team", "mine"] = "all",
    employee_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    limit, offset, _ = pagination
    stmt = select(AttendanceRequest).where(
        AttendanceRequest.tenant_id == current.tenant_id
    )
    if status_filter:
        stmt = stmt.where(AttendanceRequest.status == status_filter)
    if employee_id:
        stmt = stmt.where(AttendanceRequest.employee_id == employee_id)
    if scope == "mine":
        own = get_linked_employee_id(db, current)
        stmt = stmt.where(
            AttendanceRequest.employee_id == (own or "__none__"))
    elif scope == "team":
        stmt = stmt.where(
            AttendanceRequest.employee_id.in_(
                team_ids_with_self(db, current) or ["__none__"]
            )
        )
    elif current.role not in ("admin", "hr"):
        # scope=all:非管理角色仍按汇报线过滤(主管只见下属+自己)
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(
            AttendanceRequest.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(AttendanceRequest.created_at.desc())
        .limit(limit).offset(offset)
    ).all()
    emps, orgs, users = _req_maps(db, current.tenant_id)
    return PageOut(
        items=[_req_out(r, emps, orgs, users, current.id) for r in rows],
        total=total, limit=limit, offset=offset,
    )


def _apply_request_decision(
    db: Session, r: AttendanceRequest, action: str, actor: User,
    note: str | None, request: Request | None = None,
) -> None:
    """批准 = 补写打卡 + 重算当日考勤;驳回只改状态。审批中心引擎与
    本模块散装端点共用 —— 不 commit。"""
    r.status = action
    r.approver_id = actor.id
    r.decided_at = datetime.now(UTC).replace(tzinfo=None)
    r.decision_note = note
    if action == "approved":
        # 补写一条打卡(method=request 标记来源),再重算当日考勤
        db.add(EmployeeCheckin(
            tenant_id=r.tenant_id, employee_id=r.employee_id,
            ts=r.ts, kind=r.kind, method="request",
            # reason 是 Text 无上限,checkin.note 只有 VARCHAR(256) — 截断防超长
            note=(r.reason or "补卡审批通过")[:256],
        ))
        # autoflush 关闭,先落库让 summarize_day 能读到这条 checkin
        db.flush()
        summarize_day(
            db, tenant_id=r.tenant_id, employee_id=r.employee_id,
            day=r.ts.date(),
        )
    write_audit(
        db, actor=actor, entity_type="attendance_request", entity_id=r.id,
        action="approve" if action == "approved" else "reject",
        detail={"employee_id": r.employee_id, "ts": r.ts.isoformat(),
                "kind": r.kind} if action == "approved" else {},
        request=request,
    )


def _decide_via_engine_or_self(
    db: Session, r: AttendanceRequest, current: User, action: str,
    note: str | None, request: Request,
) -> None:
    from app.services.approvals import decide_on_instance, instance_for

    inst = instance_for(db, current.tenant_id, "attendance_request", r.id)
    if inst is not None:
        decide_on_instance(db, inst, current, action, note, request)
        return
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
    _apply_request_decision(db, r, action, current, note, request)


@router.post("/requests/{req_id}/approve", response_model=AttendanceRequestOut)
def approve_request(
    req_id: str,
    body: ReqDecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    r = _locked_req(db, req_id)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    _decide_via_engine_or_self(db, r, current, "approved", body.note, request)
    db.commit()
    emps, orgs, users = _req_maps(db, current.tenant_id)
    return _req_out(r, emps, orgs, users, current.id)


@router.post("/requests/{req_id}/reject", response_model=AttendanceRequestOut)
def reject_request(
    req_id: str,
    body: ReqDecisionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    r = _locked_req(db, req_id)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该申请已处理")
    _decide_via_engine_or_self(db, r, current, "rejected", body.note, request)
    db.commit()
    emps, orgs, users = _req_maps(db, current.tenant_id)
    return _req_out(r, emps, orgs, users, current.id)


# ============ 考勤策略 / 迟到早退异常 / 月度汇总 ============


class PolicyOut(BaseModel):
    grace_in_minutes: int
    grace_out_minutes: int
    track_late_early: bool
    ip_check_enabled: bool
    allowed_ips: list[str]
    notify_on_flag: bool


class PolicyIn(BaseModel):
    grace_in_minutes: int = 0
    grace_out_minutes: int = 0
    track_late_early: bool = True
    ip_check_enabled: bool = False
    allowed_ips: list[str] = []
    notify_on_flag: bool = True


def _policy_out(p: AttendancePolicy) -> PolicyOut:
    return PolicyOut(
        grace_in_minutes=p.grace_in_minutes,
        grace_out_minutes=p.grace_out_minutes,
        track_late_early=p.track_late_early,
        ip_check_enabled=p.ip_check_enabled,
        allowed_ips=list(p.allowed_ips or []),
        notify_on_flag=p.notify_on_flag,
    )


@router.get("/policy", response_model=PolicyOut)
def read_policy(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    return _policy_out(get_policy(db, current.tenant_id))


@router.put("/policy", response_model=PolicyOut)
def update_policy(
    body: PolicyIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    if body.grace_in_minutes < 0 or body.grace_out_minutes < 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "宽限分钟数不能为负")
    err = validate_cidrs(body.allowed_ips)
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
    p = get_policy(db, current.tenant_id)
    p.grace_in_minutes = body.grace_in_minutes
    p.grace_out_minutes = body.grace_out_minutes
    p.track_late_early = body.track_late_early
    p.ip_check_enabled = body.ip_check_enabled
    p.allowed_ips = [s.strip() for s in body.allowed_ips if s.strip()]
    p.notify_on_flag = body.notify_on_flag
    p.updated_at = datetime.now(UTC).replace(tzinfo=None)
    write_audit(
        db, actor=current, entity_type="attendance_policy",
        entity_id=p.id, action="update",
        detail=body.model_dump(), request=request,
    )
    db.commit()
    return _policy_out(p)


class FlagOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    day: date
    kind: str
    minutes: int
    detail: dict | None
    excused: bool
    excused_note: str | None
    created_at: datetime


def _flag_out(f: AttendanceFlag, emps: dict, orgs: dict) -> FlagOut:
    e = emps.get(f.employee_id)
    return FlagOut(
        id=f.id, employee_id=f.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        day=f.day, kind=f.kind, minutes=f.minutes, detail=f.detail,
        excused=f.excused, excused_note=f.excused_note, created_at=f.created_at,
    )


@router.get("/flags", response_model=PageOut[FlagOut])
def list_flags(
    pagination: tuple = Depends(paginate_params),
    month: str | None = Query(None, description="YYYY-MM"),
    day: date | None = Query(None, description="精确到某天,优先级高于 month"),
    employee_id: str | None = None,
    kind: Literal["late_in", "early_out"] | None = None,
    excused: bool | None = None,
    scope: Literal["all", "team", "mine"] = "all",
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    limit, offset, _ = pagination
    stmt = select(AttendanceFlag).where(
        AttendanceFlag.tenant_id == current.tenant_id)
    if month:
        try:
            y, m = month.split("-")
            start = date(int(y), int(m), 1)
            end = (date(int(y), int(m) + 1, 1) if int(m) < 12
                   else date(int(y) + 1, 1, 1))
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "month 格式应为 YYYY-MM")
        stmt = stmt.where(AttendanceFlag.day >= start, AttendanceFlag.day < end)
    if day:
        stmt = stmt.where(AttendanceFlag.day == day)
    if employee_id:
        stmt = stmt.where(AttendanceFlag.employee_id == employee_id)
    if kind:
        stmt = stmt.where(AttendanceFlag.kind == kind)
    if excused is not None:
        stmt = stmt.where(AttendanceFlag.excused.is_(excused))
    if scope == "mine":
        own = get_linked_employee_id(db, current)
        stmt = stmt.where(AttendanceFlag.employee_id == (own or "__none__"))
    elif scope == "team":
        stmt = stmt.where(
            AttendanceFlag.employee_id.in_(
                team_ids_with_self(db, current) or ["__none__"]))
    elif current.role not in ("admin", "hr"):
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(
            AttendanceFlag.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(AttendanceFlag.day.desc(), AttendanceFlag.created_at.desc())
        .limit(limit).offset(offset)
    ).all()
    emps, orgs = _maps(db, current.tenant_id)
    return PageOut(
        items=[_flag_out(f, emps, orgs) for f in rows],
        total=total, limit=limit, offset=offset,
    )


class ExcuseIn(BaseModel):
    note: str | None = None


def _flag_scope_guard(db: Session, current: User, flag: AttendanceFlag) -> None:
    """免计权限:manage_attendance 或能对下属做决定的审批人。"""
    if not can_decide_for_employee(db, current, flag.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能免计直属下属或具备考勤管理权限")


@router.post("/flags/{flag_id}/excuse", response_model=FlagOut)
def excuse_flag(
    flag_id: str,
    body: ExcuseIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    f = db.get(AttendanceFlag, flag_id)
    if not f or f.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "异常记录不存在")
    _flag_scope_guard(db, current, f)
    f.excused = True
    f.excused_by_id = current.id
    f.excused_note = body.note
    f.excused_at = datetime.now(UTC).replace(tzinfo=None)
    f.updated_at = f.excused_at
    write_audit(
        db, actor=current, entity_type="attendance_flag", entity_id=f.id,
        action="excuse",
        detail={"employee_id": f.employee_id, "day": str(f.day), "kind": f.kind},
        request=request,
    )
    db.commit()
    emps, orgs = _maps(db, current.tenant_id)
    return _flag_out(f, emps, orgs)


class BulkExcuseIn(BaseModel):
    flag_ids: list[str]
    note: str | None = None


class BulkExcuseOut(BaseModel):
    excused: int
    skipped: int
    denied: int
    items: list[FlagOut]


@router.post("/flags/bulk-excuse", response_model=BulkExcuseOut)
def bulk_excuse_flags(
    body: BulkExcuseIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    """批量免计考勤异常 — 助手「核对今天 N 条异常」的落点。

    已免计的跳过(天然幂等),无权决定的计入 denied 而不是整体 403:
    批量场景里部分失败不该废掉整批。"""
    if len(body.flag_ids) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "单批最多 100 条")
    rows = db.scalars(
        select(AttendanceFlag).where(
            AttendanceFlag.tenant_id == current.tenant_id,
            AttendanceFlag.id.in_(body.flag_ids),
        )
    ).all()
    emps, orgs = _maps(db, current.tenant_id)
    now = datetime.now(UTC).replace(tzinfo=None)
    items: list[FlagOut] = []
    skipped = denied = 0
    for f in rows:
        if f.excused:
            skipped += 1
            continue
        if not can_decide_for_employee(db, current, f.employee_id):
            denied += 1
            continue
        f.excused = True
        f.excused_by_id = current.id
        f.excused_note = body.note
        f.excused_at = now
        f.updated_at = now
        items.append(_flag_out(f, emps, orgs))
        write_audit(
            db, actor=current, entity_type="attendance_flag", entity_id=f.id,
            action="excuse",
            detail={"employee_id": f.employee_id, "day": str(f.day),
                    "kind": f.kind, "bulk": True},
            request=request,
        )
    db.commit()
    return BulkExcuseOut(
        excused=len(items), skipped=skipped, denied=denied, items=items)


@router.post("/flags/{flag_id}/unexcuse", response_model=FlagOut)
def unexcuse_flag(
    flag_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(
        require_permission(PERM_MANAGE_ATTENDANCE, PERM_APPROVE_LEAVE)),
):
    f = db.get(AttendanceFlag, flag_id)
    if not f or f.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "异常记录不存在")
    _flag_scope_guard(db, current, f)
    f.excused = False
    f.excused_by_id = None
    f.excused_note = None
    f.excused_at = None
    f.updated_at = datetime.now(UTC).replace(tzinfo=None)
    write_audit(
        db, actor=current, entity_type="attendance_flag", entity_id=f.id,
        action="unexcuse", detail={}, request=request,
    )
    db.commit()
    emps, orgs = _maps(db, current.tenant_id)
    return _flag_out(f, emps, orgs)


class MonthlyRow(BaseModel):
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    present: int
    absent: int
    leave: int
    remote: int
    holiday: int
    work_hours: float
    late_count: int
    early_count: int
    late_minutes: int
    overtime_hours: float


def _month_bounds(month: str) -> tuple[date, date]:
    try:
        y, m = month.split("-")
        start = date(int(y), int(m), 1)
        end = (date(int(y), int(m) + 1, 1) if int(m) < 12
               else date(int(y) + 1, 1, 1))
    except (ValueError, IndexError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "month 格式应为 YYYY-MM")
    return start, end


def monthly_summary_rows(
    db: Session, tenant_id: str, month: str, emp_ids: list[str]
) -> list[MonthlyRow]:
    """月度汇总:考勤状态计数 + 工时 + 未免计异常 + 批准加班。供 HR/ESS 复用。"""
    start, end = _month_bounds(month)
    if not emp_ids:
        return []
    rows = db.execute(
        select(
            Attendance.employee_id,
            Attendance.status,
            func.count(Attendance.id),
            func.coalesce(func.sum(Attendance.work_hours), 0),
        ).where(
            Attendance.tenant_id == tenant_id,
            Attendance.employee_id.in_(emp_ids),
            Attendance.day >= start,
            Attendance.day < end,
        ).group_by(Attendance.employee_id, Attendance.status)
    ).all()
    flags = db.execute(
        select(
            AttendanceFlag.employee_id,
            AttendanceFlag.kind,
            func.count(AttendanceFlag.id),
            func.coalesce(func.sum(AttendanceFlag.minutes), 0),
        ).where(
            AttendanceFlag.tenant_id == tenant_id,
            AttendanceFlag.employee_id.in_(emp_ids),
            AttendanceFlag.day >= start,
            AttendanceFlag.day < end,
            AttendanceFlag.excused.is_(False),
        ).group_by(AttendanceFlag.employee_id, AttendanceFlag.kind)
    ).all()
    ots = db.execute(
        select(
            OvertimeRequest.employee_id,
            func.coalesce(func.sum(OvertimeRequest.hours), 0),
        ).where(
            OvertimeRequest.tenant_id == tenant_id,
            OvertimeRequest.employee_id.in_(emp_ids),
            OvertimeRequest.status == "approved",
            OvertimeRequest.work_date >= start,
            OvertimeRequest.work_date < end,
        ).group_by(OvertimeRequest.employee_id)
    ).all()
    emps, orgs = _maps(db, tenant_id)
    agg: dict[str, MonthlyRow] = {}
    for emp_id in emp_ids:
        e = emps.get(emp_id)
        agg[emp_id] = MonthlyRow(
            employee_id=emp_id, employee_name=e.name if e else "?",
            org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
            present=0, absent=0, leave=0, remote=0, holiday=0,
            work_hours=0.0, late_count=0, early_count=0,
            late_minutes=0, overtime_hours=0.0,
        )
    for emp_id, st, cnt, wh in rows:
        r = agg.get(emp_id)
        if r is None:
            continue
        if st in ("present", "absent", "leave", "remote", "holiday"):
            setattr(r, st, int(cnt))
        r.work_hours = round(r.work_hours + float(wh or 0), 2)
    for emp_id, kind, cnt, mins in flags:
        r = agg.get(emp_id)
        if r is None:
            continue
        if kind == "late_in":
            r.late_count = int(cnt)
            r.late_minutes = int(mins or 0)
        else:
            r.early_count = int(cnt)
    for emp_id, hours in ots:
        r = agg.get(emp_id)
        if r is not None:
            r.overtime_hours = round(float(hours or 0), 2)
    return [agg[e] for e in emp_ids]


@router.get("/monthly-summary", response_model=list[MonthlyRow])
def monthly_summary(
    month: str = Query(..., description="YYYY-MM"),
    scope: Literal["all", "team"] = "all",
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    if scope == "team":
        emp_ids = team_ids_with_self(db, current) or []
    else:
        ids = visible_employee_ids(db, current)
        emp_ids = ids if ids is not None else [
            e.id for e in db.scalars(
                select(Employee).where(
                    Employee.tenant_id == current.tenant_id,
                    Employee.status == "active",
                )
            )
        ]
    return monthly_summary_rows(db, current.tenant_id, month, emp_ids)


# ============ 考勤结果在线确认(HR 侧,Moka 同款) ============


class ConfirmationRow(BaseModel):
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    status: str | None  # None=未确认
    note: str | None
    resolved_note: str | None
    updated_at: datetime | None


@router.get("/confirmations", response_model=list[ConfirmationRow])
def attendance_confirmations(
    month: str = Query(..., description="YYYY-MM"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    """全员月度考勤确认状态:确认/异议/未确认,异议排最前。"""
    emp_ids = visible_employee_ids(db, current)
    stmt = select(Employee).where(
        Employee.tenant_id == current.tenant_id,
        Employee.status == "active",
    )
    if emp_ids is not None:
        stmt = stmt.where(Employee.id.in_(emp_ids))
    emps = db.scalars(stmt.order_by(Employee.name)).all()
    if not emps:
        return []
    confs = {
        c.employee_id: c
        for c in db.scalars(
            select(AttendanceConfirmation).where(
                AttendanceConfirmation.tenant_id == current.tenant_id,
                AttendanceConfirmation.month == month,
                AttendanceConfirmation.employee_id.in_([e.id for e in emps]),
            )
        )
    }
    orgs = {
        o.id: o.name
        for o in db.scalars(select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id))
    }
    rows = [
        ConfirmationRow(
            employee_id=e.id,
            employee_name=e.name,
            org_unit_name=orgs.get(e.org_unit_id) if e.org_unit_id else None,
            status=confs[e.id].status if e.id in confs else None,
            note=confs[e.id].note if e.id in confs else None,
            resolved_note=confs[e.id].resolved_note if e.id in confs else None,
            updated_at=confs[e.id].updated_at if e.id in confs else None,
        )
        for e in emps
    ]
    order = {"disputed": 0, "resolved": 1, "confirmed": 2}
    rows.sort(key=lambda r: (order.get(r.status or "", 3), r.employee_name))
    return rows


class ResolveIn(BaseModel):
    month: str
    note: str | None = None


@router.post("/confirmations/{employee_id}/resolve", response_model=ConfirmationRow)
def resolve_confirmation(
    employee_id: str,
    body: ResolveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    """HR 处理考勤异议 → status=resolved;处理人可对考勤记录另行修正。"""
    e = db.get(Employee, employee_id)
    if not e or e.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    c = db.scalars(
        select(AttendanceConfirmation).where(
            AttendanceConfirmation.employee_id == employee_id,
            AttendanceConfirmation.month == body.month,
        )
    ).first()
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "该员工本月没有确认记录")
    if c.status != "disputed":
        raise HTTPException(409, f"当前状态 {c.status} 无需处理(仅 disputed 可解决)")
    c.status = "resolved"
    c.resolved_by_id = current.id
    c.resolved_note = (body.note or "").strip() or None
    write_audit(
        db, actor=current, entity_type="attendance_confirmation", entity_id=c.id,
        action="resolve",
        detail={"employee_id": employee_id, "month": body.month, "note": body.note},
        request=request,
    )
    db.commit()
    org = db.get(OrgUnit, e.org_unit_id) if e.org_unit_id else None
    return ConfirmationRow(
        employee_id=e.id, employee_name=e.name,
        org_unit_name=org.name if org else None,
        status=c.status, note=c.note, resolved_note=c.resolved_note,
        updated_at=c.updated_at,
    )


# 参数路由沉底 —— 若放在 /policy 等字面量路由之前,PUT /policy 会被
# {att_id} 吃掉并命中 AttendanceIn 校验(422),FastAPI 按声明序匹配。
@router.put("/{att_id}", response_model=AttendanceOut)
def update_attendance(
    att_id: str,
    body: AttendanceIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    a = db.get(Attendance, att_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "考勤记录不存在")
    a.status = body.status
    a.source = "manual"
    # model_fields_set:显式传 null = 清空时间;不传字段则不覆盖
    if "first_in" in body.model_fields_set:
        a.first_in = body.first_in
    if "last_out" in body.model_fields_set:
        a.last_out = body.last_out
    a.work_hours = (
        round((a.last_out - a.first_in).total_seconds() / 3600, 2)
        if a.first_in and a.last_out and a.last_out > a.first_in
        else None
    )
    write_audit(
        db, actor=current, entity_type="attendance", entity_id=a.id,
        action="manual_correct",
        detail={
            "employee_id": a.employee_id, "day": str(a.day),
            "status": a.status,
            "first_in": a.first_in.isoformat() if a.first_in else None,
            "last_out": a.last_out.isoformat() if a.last_out else None,
        },
        request=request,
    )
    db.commit()
    db.refresh(a)
    emps, orgs = _maps(db, current.tenant_id)
    return _out(a, emps, orgs)
