"""人事异动月报 — 入/离/调/升聚合(report kind ``hr_movements``)。

纯确定性、零 LLM:
- hires     = 期间 ``Employee.hire_date`` 落窗
- departures= 期间 ``LifecycleEvent(kind='leave')`` effective_date 落窗
- transfers / promotions = 同表 kind='transfer'/'promotion'
- 无 lifecycle 事件的租户也至少有 hires(hire_date 是员工主数据)
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import Employee, LifecycleEvent, OrgUnit, User
from app.services.permissions import apply_org_filter, get_org_scope

_KIND_LABELS = {
    "join": "入职",
    "leave": "离职",
    "transfer": "调动",
    "promotion": "晋升",
}


def _month_range(month: str | None) -> tuple[date, date, str]:
    """``YYYY-MM`` → (首日, 下月首日, 规范化串);空参默认上月。"""
    today = date.today()
    if not month:
        first = today.replace(day=1)
        prev_end = first.fromordinal(first.toordinal() - 1)
        month = f"{prev_end.year:04d}-{prev_end.month:02d}"
    try:
        y, m = (int(x) for x in month.split("-"))
        start = date(y, m, 1)
        end = date(y + (m == 12), (m % 12) + 1, 1)
    except (ValueError, TypeError):
        # LLM 侧自由参数 — 格式错回退上月,不炸 500
        first = today.replace(day=1)
        prev_end = first.fromordinal(first.toordinal() - 1)
        return _month_range(f"{prev_end.year:04d}-{prev_end.month:02d}")
    return start, end, f"{start.year:04d}-{start.month:02d}"


def build_movements_report(db: Session, u: User, month: str | None = None) -> dict:
    start, end, norm = _month_range(month)
    scope = get_org_scope(db, u)

    emps = {
        e.id: e
        for e in db.scalars(select(Employee).where(Employee.tenant_id == u.tenant_id)).all()
    }
    org_names = {
        o.id: o.name
        for o in db.scalars(select(OrgUnit).where(OrgUnit.tenant_id == u.tenant_id)).all()
    }

    def _org(emp: Employee | None) -> str:
        return org_names.get(emp.org_unit_id, "未分配") if emp else "未分配"

    def _item(emp: Employee | None, kind: str, day) -> dict:
        return {
            "employee_id": emp.id if emp else None,
            "name": emp.name if emp else "?",
            "org": _org(emp),
            "kind": kind,
            "kind_label": _KIND_LABELS.get(kind, kind),
            "date": day.isoformat() if day else None,
        }

    items: list[dict] = []

    # 入职:hire_date 口径(lifecycle 事件可能未配置模板,员工主数据是兜底)
    hire_stmt = select(Employee).where(
        Employee.tenant_id == u.tenant_id,
        Employee.hire_date >= start,
        Employee.hire_date < end,
    )
    hire_stmt = apply_org_filter(hire_stmt, org_column=Employee.org_unit_id, scope=scope)
    for e in db.scalars(hire_stmt).all():
        items.append(_item(e, "join", e.hire_date))

    # 离/调/升:lifecycle 事件为准(effective_date 落窗,取消的事件不算)
    ev_stmt = select(LifecycleEvent).where(
        LifecycleEvent.tenant_id == u.tenant_id,
        LifecycleEvent.status != "cancelled",
        LifecycleEvent.effective_date >= start,
        LifecycleEvent.effective_date < end,
        LifecycleEvent.kind.in_(("join", "leave", "transfer", "promotion")),
    )
    for ev in db.scalars(ev_stmt).all():
        emp = emps.get(ev.employee_id)
        if emp is None:
            continue
        if scope is not None and emp.org_unit_id not in scope:
            continue
        if ev.kind == "join" and emp.hire_date and start <= emp.hire_date < end:
            continue  # hire_date 已计入,避免 join 事件双算
        items.append(_item(emp, ev.kind, ev.effective_date))

    items.sort(key=lambda i: (i["date"] or "", i["kind"]))

    counts = {k: sum(1 for i in items if i["kind"] == k) for k in _KIND_LABELS}
    hires = counts["join"]
    departures = counts["leave"]

    # 期初在册 = 当前在册 - 期间净变化(近似:期末在册按 status active 数)
    active_now_stmt = select(func.count(Employee.id)).where(
        Employee.tenant_id == u.tenant_id, Employee.status == "active"
    )
    active_now_stmt = apply_org_filter(
        active_now_stmt, org_column=Employee.org_unit_id, scope=scope)
    active_now = db.scalar(active_now_stmt) or 0
    headcount_start = max(active_now - hires + departures, 0)

    by_org: dict[str, dict] = {}
    for i in items:
        b = by_org.setdefault(
            i["org"], {"org": i["org"], "join": 0, "leave": 0, "transfer": 0, "promotion": 0})
        b[i["kind"]] += 1

    return {
        "month": norm,
        "hires": hires,
        "departures": departures,
        "transfers": counts["transfer"],
        "promotions": counts["promotion"],
        "net_change": hires - departures,
        "headcount_start": headcount_start,
        "headcount_end": active_now,
        "departure_rate": (
            round(departures / headcount_start * 100, 1) if headcount_start else 0.0
        ),
        "by_org": sorted(by_org.values(), key=lambda b: -b["join"] - b["leave"]),
        "items": items[:50],
        "truncated": len(items) > 50,
    }
