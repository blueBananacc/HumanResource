"""人事提醒中心 + 人事日历(慧人力智能待办/2号人事部人事日历同款)。

数据源实时聚合,不落库:
- ``Employee.personal`` JSON 的 ``birth_date``(生日,年年复现)、
  ``probation_end``(转正到期)、``contract_end``(合同到期)、
  ``cert_expire``(证件到期)
- ``Employee.hire_date`` → 入职周年(年年复现,计算第 N 年)
- ``LifecycleEvent`` in_progress + effective_date → 待入职/待离职

``run_people_reminders`` 挂 ``_watch_loop`` 每日扫:
- deadline 类(转正/合同/证件/待入职/待离职):进入 7 天窗口即提醒 HR
- 关怀类(生日/入职周年):当天提醒 HR,同时给员工本人送祝福通知
同日同事件按 ``people.{kind}:{emp}:{date}`` link fragment 去重。
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Employee,
    LifecycleEvent,
    Notification,
    OrgUnit,
    Tenant,
    User,
)
from app.services.timeutil import local_day_start_utc, local_now

logger = logging.getLogger(__name__)

# 事件类型 → (中文名, 类别:deadline 需提前准备 / celebrate 当天关怀)
EVENT_META: dict[str, tuple[str, str]] = {
    "birthday": ("生日", "celebrate"),
    "anniversary": ("入职周年", "celebrate"),
    "probation_end": ("转正到期", "deadline"),
    "contract_end": ("合同到期", "deadline"),
    "cert_expire": ("证件到期", "deadline"),
    "onboarding": ("待入职", "deadline"),
    "separation": ("待离职", "deadline"),
}
EVENT_KINDS = tuple(EVENT_META.keys())

DEADLINE_WINDOW_DAYS = 7  # deadline 类提前提醒窗口


def _parse_day(v: Any) -> date | None:
    """personal JSON 里的日期值 → date;容忍 str/date/datetime,非法即 None。"""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def _recur_on(md: tuple[int, int], year: int) -> date | None:
    """某年的 MM-DD(处理 2/29 → 平年 2/28)。"""
    m, d = md
    try:
        return date(year, m, d)
    except ValueError:
        return date(year, m, 28) if (m, d) == (2, 29) else None


def collect_events(
    db: Session, tenant_id: str, start: date, end: date,
) -> list[dict[str, Any]]:
    """窗口 [start, end] 内的全部人事事件(含周期性事件按年展开)。

    每项: {kind, label, category, employee_id, employee_name, org_unit_name,
           day (date), days_until, link, meta?}
    """
    emps = db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)
    ).all()
    orgs = {
        o.id: o.name
        for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
        )
    }
    today = local_now().date()
    out: list[dict[str, Any]] = []

    def _push(emp: Employee, kind: str, day: date, meta: dict | None = None) -> None:
        label, category = EVENT_META[kind]
        out.append({
            "kind": kind,
            "label": label,
            "category": category,
            "employee_id": emp.id,
            "employee_name": emp.name,
            "org_unit_name": orgs.get(emp.org_unit_id),
            "day": day,
            "days_until": (day - today).days,
            "link": f"/employees/{emp.id}",
            "meta": meta or {},
        })

    for e in emps:
        p = e.personal or {}
        birth = _parse_day(p.get("birth_date"))
        if birth and e.status != "terminated":
            for y in {start.year, end.year}:
                d = _recur_on((birth.month, birth.day), y)
                if d and start <= d <= end:
                    _push(e, "birthday", d, {"age": y - birth.year or None})
        if e.hire_date and e.status != "terminated":
            for y in {start.year, end.year}:
                if y <= e.hire_date.year:
                    continue  # 入职当年不算"周年"
                d = _recur_on((e.hire_date.month, e.hire_date.day), y)
                if d and start <= d <= end:
                    _push(e, "anniversary", d, {"years": y - e.hire_date.year})
        for key, kind in (
            ("probation_end", "probation_end"),
            ("contract_end", "contract_end"),
            ("cert_expire", "cert_expire"),
        ):
            d = _parse_day(p.get(key))
            if d and e.status != "terminated" and start <= d <= end:
                _push(e, kind, d)

    # 生命周期:在办的入职/离职 → 待入职/待离职(effective_date 过期仍在办 → 逾期)
    evs = db.scalars(
        select(LifecycleEvent).where(
            LifecycleEvent.tenant_id == tenant_id,
            LifecycleEvent.kind.in_(("onboarding", "separation")),
            LifecycleEvent.status == "in_progress",
            LifecycleEvent.effective_date.is_not(None),
        )
    ).all()
    emp_map = {e.id: e for e in emps}
    for ev in evs:
        e = emp_map.get(ev.employee_id)
        if not e:
            continue
        d = ev.effective_date
        if start <= d <= end or d < start:  # 过期的也要(逾期清单)
            kind = "onboarding" if ev.kind == "onboarding" else "separation"
            _push(e, kind, d, {"event_id": ev.id, "overdue": d < today})

    out.sort(key=lambda x: (x["day"], x["employee_name"]))
    return out


def _notified_today(db: Session, user_id: str, link: str) -> bool:
    return (
        db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == user_id,
                Notification.kind == "people.reminder",
                Notification.link == link,
                Notification.created_at >= local_day_start_utc(),
            )
        ).scalar_one()
        > 0
    )


def _hr_user_ids(db: Session, tenant_id: str) -> list[str]:
    return [
        u.id
        for u in db.scalars(
            select(User).where(
                User.tenant_id == tenant_id,
                User.role.in_(("admin", "hr")),
                User.status == "active",
            )
        )
    ]


def _dedupe_link(ev: dict[str, Any]) -> str:
    return f"{ev['link']}#people:{ev['kind']}:{ev['employee_id']}:{ev['day']}"


def run_people_reminders(db: Session, today: date | None = None) -> int:
    """每日扫一轮全租户人事事件 → Notification。返回发出条数。"""
    today = today or local_now().date()
    fired = 0
    for t in db.scalars(select(Tenant)).all():
        events = collect_events(
            db, t.id, today - timedelta(days=30),
            today + timedelta(days=DEADLINE_WINDOW_DAYS),
        )
        if not events:
            continue
        hr_ids = _hr_user_ids(db, t.id)
        emp_user = {e.id: e.user_id for e in db.scalars(
            select(Employee).where(Employee.tenant_id == t.id)
        )}
        for ev in events:
            link = _dedupe_link(ev)
            du = ev["days_until"]
            if ev["category"] == "deadline":
                if du > DEADLINE_WINDOW_DAYS:
                    continue
                when = "已逾期" if du < 0 else ("今天" if du == 0 else f"{du} 天后")
                for uid in hr_ids:
                    if _notified_today(db, uid, link):
                        continue
                    db.add(Notification(
                        tenant_id=t.id, user_id=uid, kind="people.reminder",
                        title=f"{ev['label']}:{ev['employee_name']} · {when}",
                        body=f"{ev['day'].isoformat()} {ev['label']}"
                             + ("(逾期未办结)" if du < 0 else ""),
                        link=link,
                    ))
                    fired += 1
            else:  # celebrate:当天才提,HR + 员工本人祝福
                if du != 0:
                    continue
                for uid in hr_ids:
                    if _notified_today(db, uid, link):
                        continue
                    db.add(Notification(
                        tenant_id=t.id, user_id=uid, kind="people.reminder",
                        title=f"今天是 {ev['employee_name']} 的{ev['label']}",
                        body="可在「人事提醒」一键送达祝福",
                        link=link,
                    ))
                    fired += 1
                euid = emp_user.get(ev["employee_id"])
                if euid and not _notified_today(db, euid, link):
                    years = ev["meta"].get("years")
                    extra = f"入职 {years} 周年快乐" if years else "生日快乐"
                    db.add(Notification(
                        tenant_id=t.id, user_id=euid, kind="people.reminder",
                        title=extra,
                        body="来自公司与 HR 的祝福 🎉",
                        link=link,
                    ))
                    fired += 1
    db.commit()
    return fired
