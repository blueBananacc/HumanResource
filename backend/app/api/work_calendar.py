"""工作日历 / 节假日 API(HR 管理端)。

- ``GET/PUT /work-calendar``            每周工作日 + 日标准工时(manage_attendance 写)
- ``GET/POST /work-calendar/holidays``  节假日 / 调休补班日
- ``DELETE /work-calendar/holidays/{id}``
- ``GET /work-calendar/count``          区间内工作日数(排期/请假估算共用)

员工端工作日查询在 ``hr_me.py``(``/me/workday-count``)。
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.license import require_feature
from app.domain.models import Holiday, User, WorkCalendar
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_ATTENDANCE,
    PERM_VIEW_REPORTS,
    require_permission,
)
from app.services.work_calendar import (
    DEFAULT_WORKDAY_HOURS,
    DEFAULT_WORKDAYS,
    count_workdays,
    get_calendar,
)

router = APIRouter(prefix="/work-calendar", tags=["hris-work-calendar"], dependencies=[Depends(require_feature("hr.suite"))])


# ============ schemas ============


class CalendarOut(BaseModel):
    workdays: list[int]
    workday_hours: float
    work_start: time
    work_end: time


class CalendarIn(BaseModel):
    workdays: list[int] = Field(..., min_length=1)
    workday_hours: float = Field(8, gt=0, le=24)
    work_start: time = time(9, 0)
    work_end: time = time(18, 0)

    @field_validator("workdays")
    @classmethod
    def _v(cls, v: list[int]) -> list[int]:
        days = sorted(set(v))
        if any(d < 1 or d > 7 for d in days):
            raise ValueError("工作日须为 1-7(周一到周日)")
        return days


class HolidayIn(BaseModel):
    day: date
    name: str
    kind: str = "holiday"

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in ("holiday", "workday"):
            raise ValueError("kind 须为 holiday(放假)或 workday(调休补班)")
        return v


class HolidayOut(BaseModel):
    id: str
    day: date
    name: str
    kind: str


def _cal_out(cal: WorkCalendar | None) -> CalendarOut:
    return CalendarOut(
        workdays=sorted(cal.workdays) if cal else sorted(DEFAULT_WORKDAYS),
        workday_hours=cal.workday_hours if cal else DEFAULT_WORKDAY_HOURS,
        work_start=cal.work_start if cal else time(9, 0),
        work_end=cal.work_end if cal else time(18, 0),
    )


# ============ calendar ============


@router.get("", response_model=CalendarOut)
def read_calendar(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return _cal_out(get_calendar(db, current.tenant_id))


@router.put("", response_model=CalendarOut)
def update_calendar(
    body: CalendarIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    cal = get_calendar(db, current.tenant_id)
    if cal is None:
        cal = WorkCalendar(tenant_id=current.tenant_id)
        db.add(cal)
    cal.workdays = body.workdays
    cal.workday_hours = body.workday_hours
    cal.work_start = body.work_start
    cal.work_end = body.work_end
    cal.updated_at = datetime.now(UTC).replace(tzinfo=None)
    write_audit(
        db, actor=current, entity_type="work_calendar", entity_id=current.tenant_id,
        action="update",
        detail={
            "workdays": body.workdays, "workday_hours": body.workday_hours,
            "work_start": body.work_start.isoformat(),
            "work_end": body.work_end.isoformat(),
        },
        request=request,
    )
    db.commit()
    return _cal_out(cal)


# ============ holidays ============


@router.get("/holidays", response_model=list[HolidayOut])
def list_holidays(
    year: int | None = Query(None, ge=2000, le=2100),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    stmt = select(Holiday).where(Holiday.tenant_id == current.tenant_id)
    if year:
        stmt = stmt.where(
            Holiday.day >= date(year, 1, 1), Holiday.day <= date(year, 12, 31)
        )
    rows = db.scalars(stmt.order_by(Holiday.day)).all()
    return [HolidayOut(id=h.id, day=h.day, name=h.name, kind=h.kind) for h in rows]


@router.post("/holidays", response_model=HolidayOut, status_code=status.HTTP_201_CREATED)
def create_holiday(
    body: HolidayIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    h = Holiday(
        tenant_id=current.tenant_id, day=body.day, name=body.name, kind=body.kind
    )
    db.add(h)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "该日期已有节假日条目") from None
    write_audit(
        db, actor=current, entity_type="holiday", entity_id=h.id,
        action="create",
        detail={"day": str(h.day), "name": h.name, "kind": h.kind},
        request=request,
    )
    db.commit()
    return HolidayOut(id=h.id, day=h.day, name=h.name, kind=h.kind)


@router.delete("/holidays/{holiday_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_holiday(
    holiday_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    h = db.get(Holiday, holiday_id)
    if not h or h.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "节假日条目不存在")
    write_audit(
        db, actor=current, entity_type="holiday", entity_id=h.id,
        action="delete", detail={"day": str(h.day), "name": h.name},
        request=request,
    )
    db.delete(h)
    db.commit()


# ============ helpers ============


@router.get("/count")
def workday_count(
    start: date,
    end: date,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    if end < start:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "结束日期不能早于开始日期")
    if (end - start).days > 366:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "区间最长 366 天")
    return {"workdays": count_workdays(db, current.tenant_id, start, end)}
