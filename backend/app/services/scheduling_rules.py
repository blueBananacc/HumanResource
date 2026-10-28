"""排班合规校验(喔趣:排班环节前置发现违规,而非结算时才暴露)。

规则存 ``ShiftRule``(每租户一行,缺省用默认值):
- ``consecutive_days`` 自然日连续排班段超过上限 → block
- ``weekly_hours``     ISO 周班次时长总和超过上限 → block
- ``rest_gap``         相邻两日班次间隔(下班→次日上班)不足 → block
- ``availability``     排班日落在员工「不可排」偏好日 → block;
  「偏好日」只 warn 不阻断

``violations_for_window`` 对窗口内全部(或指定组织)员工出体检报告;
``precheck`` 校验一条候选排班引入的违规,挂到手工新建/替班/换班批准/
模板套用的入口处。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Employee,
    EmployeeAvailability,
    Shift,
    ShiftAssignment,
    ShiftRule,
)

Block = "block"
Warn = "warn"


def get_rules(db: Session, tenant_id: str) -> ShiftRule:
    """每租户一行;不存在则建默认行。"""
    r = db.scalars(
        select(ShiftRule).where(ShiftRule.tenant_id == tenant_id)
    ).first()
    if r is None:
        r = ShiftRule(tenant_id=tenant_id)
        db.add(r)
        db.flush()
    return r


def rules_out(r: ShiftRule) -> dict:
    return {
        "max_consecutive_days": r.max_consecutive_days,
        "max_weekly_hours": r.max_weekly_hours,
        "min_rest_minutes": r.min_rest_minutes,
        "respect_availability": r.respect_availability,
    }


def shift_hours(s: Shift) -> float:
    """班次时长(小时);跨天班 end<start 按隔夜计。"""
    dur = s.end_min - s.start_min
    if dur <= 0:
        dur += 1440
    return dur / 60.0


def _avail_map(
    db: Session, tenant_id: str, employee_ids: list[str]
) -> dict[str, dict[int, str]]:
    """employee_id → {weekday → preference}。"""
    rows = db.scalars(
        select(EmployeeAvailability).where(
            EmployeeAvailability.tenant_id == tenant_id,
            EmployeeAvailability.employee_id.in_(employee_ids),
        )
    ).all()
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for a in rows:
        out[a.employee_id][a.weekday] = a.preference
    return out


def evaluate_pairs(
    pairs: list[tuple[date, Shift]],
    rules: ShiftRule,
    avail: dict[int, str],
) -> list[dict]:
    """对一名员工的 (day, shift) 序列跑全部规则;返回违规列表。

    纯函数 —— 方便测试与「现有排班 + 候选排班」的混合校验。
    """
    out: list[dict] = []
    days = sorted(d for d, _ in pairs)
    by_day = dict(pairs)

    # 连续出勤天数
    run = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            run += 1
            if run > rules.max_consecutive_days:
                out.append({
                    "rule": "consecutive_days", "severity": Block,
                    "day": days[i].isoformat(),
                    "detail": f"连续排班 {run} 天,超过上限 {rules.max_consecutive_days} 天",
                })
        else:
            run = 1

    # ISO 周工时
    weekly: dict[tuple[int, int], float] = defaultdict(float)
    for d, s in pairs:
        iso = d.isocalendar()
        weekly[(iso[0], iso[1])] += shift_hours(s)
    for wk, hrs in weekly.items():
        if hrs > rules.max_weekly_hours:
            week_days = [d for d in days if d.isocalendar()[:2] == wk]
            out.append({
                "rule": "weekly_hours", "severity": Block,
                "day": max(week_days).isoformat(),
                "detail": (
                    f"第 {wk[1]} 周排班 {hrs:.1f}h,"
                    f"超过上限 {rules.max_weekly_hours}h"),
            })

    # 相邻班次休息间隔
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days != 1:
            continue
        prev_s, s = by_day[days[i - 1]], by_day[days[i]]
        gap = (1440 - prev_s.end_min) + s.start_min
        if gap < rules.min_rest_minutes:
            out.append({
                "rule": "rest_gap", "severity": Block,
                "day": days[i].isoformat(),
                "detail": (
                    f"{prev_s.name}下班至{s.name}上班仅休息 "
                    f"{gap // 60}h{gap % 60}m,不足 "
                    f"{rules.min_rest_minutes // 60}h"),
            })

    # 员工可用性
    if rules.respect_availability:
        for d in days:
            pref = avail.get(d.weekday())
            if pref == "unavailable":
                out.append({
                    "rule": "availability", "severity": Block,
                    "day": d.isoformat(),
                    "detail": "员工登记该 weekday 为不可排",
                })
            elif pref == "preferred":
                out.append({
                    "rule": "availability", "severity": Warn,
                    "day": d.isoformat(),
                    "detail": "员工登记该 weekday 为偏好日(非硬约束)",
                })
    return out


def _load_pairs(
    db: Session, tenant_id: str, emp_ids: list[str], start: date, end: date
) -> tuple[dict[str, list[tuple[date, Shift]]], dict[str, Shift]]:
    rows = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == tenant_id,
            ShiftAssignment.employee_id.in_(emp_ids),
            ShiftAssignment.day >= start,
            ShiftAssignment.day <= end,
        )
    ).all()
    shift_ids = {a.shift_id for a in rows}
    shifts = {
        s.id: s for s in db.scalars(
            select(Shift).where(Shift.id.in_(shift_ids or {""}))).all()
    }
    pairs: dict[str, list[tuple[date, Shift]]] = defaultdict(list)
    for a in rows:
        s = shifts.get(a.shift_id)
        if s is not None:
            pairs[a.employee_id].append((a.day, s))
    return pairs, shifts


def violations_for_window(
    db: Session,
    tenant_id: str,
    *,
    start: date,
    end: date,
    org_unit_id: str | None = None,
) -> list[dict]:
    """窗口内全员(或某部门)合规体检;按员工分组返回。"""
    emp_q = select(Employee).where(
        Employee.tenant_id == tenant_id, Employee.status == "active")
    if org_unit_id:
        emp_q = emp_q.where(Employee.org_unit_id == org_unit_id)
    emps = db.scalars(emp_q).all()
    if not emps:
        return []
    rules = get_rules(db, tenant_id)
    pairs, _shifts = _load_pairs(db, tenant_id, [e.id for e in emps], start, end)
    avail = _avail_map(db, tenant_id, [e.id for e in emps])
    out: list[dict] = []
    for e in emps:
        for v in evaluate_pairs(pairs.get(e.id, []), rules,
                                avail.get(e.id, {})):
            out.append({"employee_id": e.id, "employee_name": e.name, **v})
    out.sort(key=lambda v: (v["day"], v["employee_name"]))
    return out


def violations_for_employee(
    db: Session,
    tenant_id: str,
    employee_id: str,
    *,
    start: date,
    end: date,
    extra_pairs: list[tuple[date, Shift]] | None = None,
) -> list[dict]:
    """单员工窗口体检(可与额外候选排班合并)。"""
    rules = get_rules(db, tenant_id)
    pairs, _ = _load_pairs(db, tenant_id, [employee_id], start, end)
    merged = list(pairs.get(employee_id, [])) + list(extra_pairs or [])
    avail = _avail_map(db, tenant_id, [employee_id])
    return evaluate_pairs(merged, rules, avail.get(employee_id, {}))


def precheck(
    db: Session,
    tenant_id: str,
    employee_id: str,
    day: date,
    shift: Shift,
    *,
    extra_pairs: list[tuple[date, Shift]] | None = None,
) -> list[dict]:
    """校验「给 emp 在 day 加一条 shift」会引入的违规。

    窗口取 day±8 天(覆盖连续段与周边界);只返回 day 及以后产生的
    违规 —— 新排班只可能向后延伸违规。
    """
    rules = get_rules(db, tenant_id)
    start = day - timedelta(days=8)
    end = day + timedelta(days=8)
    pairs, _ = _load_pairs(db, tenant_id, [employee_id], start, end)
    merged = list(pairs.get(employee_id, []))
    merged.append((day, shift))
    if extra_pairs:
        merged.extend(extra_pairs)
    avail = _avail_map(db, tenant_id, [employee_id])
    vs = evaluate_pairs(merged, rules, avail.get(employee_id, {}))
    return [v for v in vs if v["day"] >= day.isoformat()]


def availability_map_for(
    db: Session, tenant_id: str, employee_ids: list[str]
) -> dict[str, dict[int, str]]:
    return _avail_map(db, tenant_id, employee_ids)
