"""工作日历服务:租户级工作日判定。

判定顺序:
1. ``Holiday`` 逐日覆盖 —— kind=holiday 放假 / kind=workday 调休补班
2. 否则看 ``WorkCalendar.workdays``(ISO weekday,1=周一 … 7=周日)
3. 租户没配日历时回退默认周一~周五、每日 8 小时

请假校验、考勤汇总、加班折算都走这里,保证"工作日"口径全局一致。
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import Holiday, WorkCalendar

DEFAULT_WORKDAYS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
DEFAULT_WORKDAY_HOURS = 8.0


def get_calendar(db: Session, tenant_id: str) -> WorkCalendar | None:
    return db.scalars(
        select(WorkCalendar).where(WorkCalendar.tenant_id == tenant_id)
    ).first()


def workday_set(db: Session, tenant_id: str) -> frozenset[int]:
    """每周工作日集合(ISO weekday)。"""
    cal = get_calendar(db, tenant_id)
    if cal and cal.workdays:
        return frozenset(cal.workdays)
    return DEFAULT_WORKDAYS


def workday_hours(db: Session, tenant_id: str) -> float:
    cal = get_calendar(db, tenant_id)
    return cal.workday_hours if cal and cal.workday_hours else DEFAULT_WORKDAY_HOURS


def holiday_map(
    db: Session, tenant_id: str, start: date, end: date
) -> dict[date, Holiday]:
    rows = db.scalars(
        select(Holiday).where(
            Holiday.tenant_id == tenant_id,
            Holiday.day >= start,
            Holiday.day <= end,
        )
    ).all()
    return {h.day: h for h in rows}


def is_workday(
    day: date, workdays: frozenset[int] | set[int], hols: dict[date, Holiday]
) -> bool:
    h = hols.get(day)
    if h is not None:
        return h.kind == "workday"
    return day.isoweekday() in workdays


def workday_dates(
    db: Session, tenant_id: str, start: date, end: date
) -> list[date]:
    """[start, end] 内全部工作日(含 holiday 覆盖规则)。"""
    if end < start:
        return []
    wd = workday_set(db, tenant_id)
    hols = holiday_map(db, tenant_id, start, end)
    out = []
    day = start
    while day <= end:
        if is_workday(day, wd, hols):
            out.append(day)
        day += timedelta(days=1)
    return out


def count_workdays(db: Session, tenant_id: str, start: date, end: date) -> int:
    return len(workday_dates(db, tenant_id, start, end))


def is_workday_day(db: Session, tenant_id: str, day: date) -> bool:
    """单日判定(summarize_day 这类每天一次的调用走它,避免重复拉整段)。"""
    h = db.scalars(
        select(Holiday).where(Holiday.tenant_id == tenant_id, Holiday.day == day)
    ).first()
    if h is not None:
        return h.kind == "workday"
    return day.isoweekday() in workday_set(db, tenant_id)


__all__ = [
    "DEFAULT_WORKDAYS",
    "DEFAULT_WORKDAY_HOURS",
    "count_workdays",
    "get_calendar",
    "holiday_map",
    "is_workday",
    "is_workday_day",
    "workday_dates",
    "workday_hours",
    "workday_set",
]
