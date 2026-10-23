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
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Attendance,
    AttendanceRequest,
    Employee,
    EmployeeCheckin,
    LeaveApplication,
    OrgUnit,
    User,
)
from app.infra.db import get_db
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
) -> Attendance:
    """单日汇总:checkin → first_in/last_out;approved 请假覆盖的工作日 → leave
    (非工作日即使被请假区间覆盖也只算 holiday,与台账只扣工作日同口径);
    工作日无记录 → absent;否则 present(有 checkin)。

    ``workdays``/``hols`` 传预先按整段拉好的日历(批量 summarize 防 N+1);
    不传则按单日查(is_workday_day)。
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
    # 日历按整段拉一次传给 summarize_day,避免每个员工每天再查两次
    wd = workday_set(db, current.tenant_id)
    hols = holiday_map(db, current.tenant_id, body.start, body.end)
    day = body.start
    n = 0
    while day <= body.end:
        for emp_id in emp_ids:
            summarize_day(
                db, tenant_id=current.tenant_id, employee_id=emp_id, day=day,
                workdays=wd, hols=hols,
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
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
    r.status = "approved"
    r.approver_id = current.id
    r.decided_at = datetime.now(UTC).replace(tzinfo=None)
    r.decision_note = body.note
    # 补写一条打卡(method=request 标记来源),再重算当日考勤
    db.add(EmployeeCheckin(
        tenant_id=current.tenant_id, employee_id=r.employee_id,
        ts=r.ts, kind=r.kind, method="request",
        # reason 是 Text 无上限,checkin.note 只有 VARCHAR(256) — 截断防超长
        note=(r.reason or "补卡审批通过")[:256],
    ))
    # autoflush 关闭,先落库让 summarize_day 能读到这条 checkin
    db.flush()
    summarize_day(
        db, tenant_id=current.tenant_id, employee_id=r.employee_id,
        day=r.ts.date(),
    )
    write_audit(
        db, actor=current, entity_type="attendance_request", entity_id=r.id,
        action="approve",
        detail={"employee_id": r.employee_id, "ts": r.ts.isoformat(),
                "kind": r.kind},
        request=request,
    )
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
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "只能审批直属下属的申请,不能自批")
    r.status = "rejected"
    r.approver_id = current.id
    r.decided_at = datetime.now(UTC).replace(tzinfo=None)
    r.decision_note = body.note
    write_audit(
        db, actor=current, entity_type="attendance_request", entity_id=r.id,
        action="reject", detail={}, request=request,
    )
    db.commit()
    emps, orgs, users = _req_maps(db, current.tenant_id)
    return _req_out(r, emps, orgs, users, current.id)
