"""考勤 API(HR 管理端)。

- ``GET /attendance``                    考勤记录(按月/员工/状态过滤)
- ``PUT /attendance/{id}``               手工修正状态(manage_attendance)
- ``POST /attendance/summarize``         把指定日期段的 checkin 汇总成 attendance
- ``GET /attendance/checkins``           打卡流水
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Attendance,
    Employee,
    EmployeeCheckin,
    LeaveApplication,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_ATTENDANCE,
    PERM_VIEW_REPORTS,
    require_permission,
    visible_employee_ids,
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


def summarize_day(db: Session, *, tenant_id: str, employee_id: str, day: date) -> Attendance:
    """单日汇总:checkin → first_in/last_out;approved 请假覆盖 → leave;
    工作日无记录 → absent;否则 present(有 checkin)。"""
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
        if on_leave:
            att.status = "leave"
            att.source = "leave"
        elif first_in:
            att.status = "present"
            att.source = "checkin"
        elif day.weekday() >= 5:
            att.status = "holiday"
            att.source = "checkin"
        else:
            att.status = "absent"
            att.source = "checkin"
        att.first_in = first_in or att.first_in
        att.last_out = last_out or att.last_out
        if att.first_in and att.last_out and att.last_out > att.first_in:
            att.work_hours = round(
                (att.last_out - att.first_in).total_seconds() / 3600, 2
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
    if body.first_in is not None:
        a.first_in = body.first_in
    if body.last_out is not None:
        a.last_out = body.last_out
    if a.first_in and a.last_out and a.last_out > a.first_in:
        a.work_hours = round((a.last_out - a.first_in).total_seconds() / 3600, 2)
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
    day = body.start
    n = 0
    while day <= body.end:
        for emp_id in emp_ids:
            summarize_day(db, tenant_id=current.tenant_id, employee_id=emp_id, day=day)
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
