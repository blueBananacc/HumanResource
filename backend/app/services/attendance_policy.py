"""考勤策略:宽限时间、迟到/早退异常判定、打卡 IP 白名单。

时区口径:checkin ``ts`` 是 UTC naive;``work_start``/``work_end`` 是部署
时区(``app_timezone``)墙钟 —— 判定前先把打卡时刻换算到本地再取
time-of-day 比较。``attendance.day`` 的既有分桶口径不变。
"""
from __future__ import annotations

import ipaddress
import logging
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.models import (
    Attendance,
    AttendanceFlag,
    AttendancePolicy,
    Employee,
    Notification,
    WorkCalendar,
)
from app.services.timeutil import utc_now

logger = logging.getLogger(__name__)

_LATE = "late_in"
_EARLY = "early_out"


def get_policy(db: Session, tenant_id: str) -> AttendancePolicy:
    """每租户一行;不存在则建默认行(不开 IP 校验、宽限 0、追踪开)。"""
    p = db.scalars(
        select(AttendancePolicy).where(AttendancePolicy.tenant_id == tenant_id)
    ).first()
    if p is None:
        p = AttendancePolicy(tenant_id=tenant_id)
        db.add(p)
        db.flush()
    return p


def validate_cidrs(ips: list[str]) -> str | None:
    for raw in ips:
        try:
            ipaddress.ip_network(str(raw).strip(), strict=False)
        except ValueError:
            return f"无效的 IP/CIDR:{raw}"
    return None


def client_ip(request: Request | None) -> str | None:
    """与 audit 同一口径:X-Forwarded-For 首跳 → client.host。"""
    if request is None:
        return None
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else None
    if not ip and request.client is not None:
        ip = request.client.host
    return ip or None


def ip_allowed(policy: AttendancePolicy, ip: str | None) -> bool:
    """IP 白名单判定。开关关/列表空 → 放行;开关开 → 必须命中一条 CIDR。"""
    if not policy.ip_check_enabled or not policy.allowed_ips:
        return True
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for raw in policy.allowed_ips:
        try:
            if addr in ipaddress.ip_network(str(raw).strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def _to_local(dt: datetime) -> datetime:
    """UTC naive → 部署时区 naive(与 work_start/work_end 同一墙钟口径)。"""
    return dt.replace(tzinfo=UTC).astimezone(ZoneInfo(settings.app_timezone)).replace(
        tzinfo=None
    )


def _expected(day: date, t: time) -> datetime:
    return datetime.combine(day, t)


def _upsert_flag(
    db: Session,
    *,
    att: Attendance,
    kind: str,
    minutes: int,
    detail: dict,
    notify: bool,
) -> bool:
    """按 (attendance_id, kind) upsert;保留 excused;返回是否新建。"""
    f = db.scalars(
        select(AttendanceFlag).where(
            AttendanceFlag.attendance_id == att.id, AttendanceFlag.kind == kind
        )
    ).first()
    created = False
    if f is None:
        f = AttendanceFlag(
            tenant_id=att.tenant_id, attendance_id=att.id,
            employee_id=att.employee_id, day=att.day, kind=kind,
        )
        db.add(f)
        created = True
    f.minutes = minutes
    f.detail = detail
    f.updated_at = utc_now()
    if created and notify:
        emp = db.get(Employee, att.employee_id)
        if emp is not None and emp.user_id:
            label = "迟到" if kind == _LATE else "早退"
            db.add(Notification(
                tenant_id=att.tenant_id, user_id=emp.user_id,
                kind="attendance_flag",
                title=f"{label} {minutes} 分钟",
                body=(
                    f"{att.day.isoformat()} {detail.get('label', '')}:"
                    f"期望 {detail.get('expected')},实际 {detail.get('actual')}"
                )[:512],
                link="/me/attendance",
            ))
    return created


def _delete_flag(db: Session, attendance_id: str, kind: str) -> None:
    f = db.scalars(
        select(AttendanceFlag).where(
            AttendanceFlag.attendance_id == attendance_id,
            AttendanceFlag.kind == kind,
        )
    ).first()
    if f is not None:
        db.delete(f)


def evaluate_flags(
    db: Session,
    *,
    att: Attendance,
    is_workday: bool,
    calendar: WorkCalendar | None = None,
    policy: AttendancePolicy | None = None,
) -> None:
    """对一条已算好的 attendance 做迟到/早退判定(幂等,可重复跑)。

    仅 ``status=="present"`` 且工作日才评;考勤事实不再满足时删掉旧 flag,
    台账与事实保持一致。不 commit —— 由 summarize_day 的调用方统一提交。
    """
    policy = policy or get_policy(db, att.tenant_id)
    if not policy.track_late_early:
        return
    if calendar is None:
        calendar = db.scalars(
            select(WorkCalendar).where(WorkCalendar.tenant_id == att.tenant_id)
        ).first()
    cal = calendar or WorkCalendar(
        tenant_id=att.tenant_id,
        workdays=[1, 2, 3, 4, 5],
        workday_hours=8.0,
        work_start=time(9, 0),
        work_end=time(18, 0),
    )

    late = early = None
    if att.status == "present" and is_workday and att.first_in:
        actual_in = _to_local(att.first_in)
        expected_in = _expected(att.day, cal.work_start)
        late_minutes = int((actual_in - expected_in).total_seconds() // 60)
        if late_minutes > policy.grace_in_minutes:
            late = (
                late_minutes,
                {"label": "上班", "expected": cal.work_start.strftime("%H:%M"),
                 "actual": actual_in.strftime("%H:%M")},
            )
    if att.status == "present" and is_workday and att.last_out:
        actual_out = _to_local(att.last_out)
        expected_out = _expected(att.day, cal.work_end)
        early_minutes = int((expected_out - actual_out).total_seconds() // 60)
        if early_minutes > policy.grace_out_minutes:
            early = (
                early_minutes,
                {"label": "下班", "expected": cal.work_end.strftime("%H:%M"),
                 "actual": actual_out.strftime("%H:%M")},
            )

    notify = policy.notify_on_flag
    if late is not None:
        _upsert_flag(db, att=att, kind=_LATE, minutes=late[0],
                     detail=late[1], notify=notify)
    else:
        _delete_flag(db, att.id, _LATE)
    if early is not None:
        _upsert_flag(db, att=att, kind=_EARLY, minutes=early[0],
                     detail=early[1], notify=notify)
    else:
        _delete_flag(db, att.id, _EARLY)


__all__ = [
    "client_ip",
    "evaluate_flags",
    "get_policy",
    "ip_allowed",
    "validate_cidrs",
]
