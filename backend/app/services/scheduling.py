"""排班引擎(H3)— 确定性求解,零 LLM。

- ``generate_schedule``:按 WorkCalendar 工作日 × Shift.required 填空位
  硬约束:已批请假不排 / 一人一岗一天 / 已有排班不重排
  软约束:按当月累计班次最少的员工优先(公平轮转)
- ``find_gaps``:逐日逐班核对覆盖缺口(含"排了班但人在请假"的虚覆盖)
- ``suggest_cover`` / ``auto_cover``:替班候选排序 + 自动换班
"""
from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Employee,
    LeaveApplication,
    Shift,
    ShiftAssignment,
)
from app.services.scheduling_rules import (
    availability_map_for,
    get_rules,
)
from app.services.work_calendar import workday_dates


def month_range(month: str | None) -> tuple[date, date]:
    """YYYY-MM → (首日, 末日);空/非法默认当月。"""
    today = date.today()
    try:
        y, m = (int(x) for x in (month or "").split("-"))
        start = date(y, m, 1)
    except (ValueError, TypeError):
        start = date(today.year, today.month, 1)
    end = start.replace(day=monthrange(start.year, start.month)[1])
    return start, end


def _leave_days(
    db: Session, tenant_id: str, start: date, end: date
) -> dict[str, set[date]]:
    """employee_id → 期间内已批请假的日期集合。"""
    apps = db.scalars(
        select(LeaveApplication).where(
            LeaveApplication.tenant_id == tenant_id,
            LeaveApplication.status == "approved",
            LeaveApplication.start_date <= end,
            LeaveApplication.end_date >= start,
        )
    ).all()
    out: dict[str, set[date]] = defaultdict(set)
    for a in apps:
        d = max(a.start_date, start)
        while d <= min(a.end_date, end):
            out[a.employee_id].add(d)
            d += timedelta(days=1)
    return out


def _load(db: Session, tenant_id: str, start: date, end: date,
          org_unit_id: str | None, shift_ids: list[str] | None):
    shifts = db.scalars(
        select(Shift).where(Shift.tenant_id == tenant_id, Shift.active.is_(True))
    ).all()
    if shift_ids:
        shifts = [s for s in shifts if s.id in set(shift_ids)]
    emp_q = select(Employee).where(
        Employee.tenant_id == tenant_id, Employee.status == "active")
    if org_unit_id:
        emp_q = emp_q.where(Employee.org_unit_id == org_unit_id)
    employees = db.scalars(emp_q).all()
    assigns = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == tenant_id,
            ShiftAssignment.day >= start,
            ShiftAssignment.day <= end,
        )
    ).all()
    return shifts, employees, assigns


def _busy_map(assigns: list[ShiftAssignment]) -> dict[str, set[date]]:
    m: dict[str, set[date]] = defaultdict(set)
    for a in assigns:
        m[a.employee_id].add(a.day)
    return m


def _fill_map(assigns: list[ShiftAssignment]) -> dict[tuple[date, str], int]:
    m: dict[tuple[date, str], int] = defaultdict(int)
    for a in assigns:
        m[(a.day, a.shift_id)] += 1
    return m


def generate_schedule(
    db: Session,
    tenant_id: str,
    *,
    start: date,
    end: date,
    org_unit_id: str | None = None,
    shift_ids: list[str] | None = None,
    dry_run: bool = True,
    actor_id: str | None = None,
) -> dict:
    """生成排班计划。dry_run=True 只返回计划不落库(确认门预览用)。"""
    shifts, employees, assigns = _load(
        db, tenant_id, start, end, org_unit_id, shift_ids)
    workdays = set(workday_dates(db, tenant_id, start, end))
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    leave = _leave_days(db, tenant_id, start, end)
    busy = _busy_map(assigns)
    filled = _fill_map(assigns)
    rules = get_rules(db, tenant_id)
    avail = availability_map_for(db, tenant_id, [e.id for e in employees])
    assigned_cnt: dict[str, int] = defaultdict(int)
    for a in assigns:
        assigned_cnt[a.employee_id] += 1

    plan: list[dict] = []
    skipped: list[dict] = []
    for d in days:
        if d not in workdays:
            continue
        for s in shifts:
            need = s.required - filled[(d, s.id)]
            if need <= 0:
                continue
            cands = [
                e for e in employees
                if d not in leave.get(e.id, set()) and d not in busy.get(e.id, set())
                and (not rules.respect_availability
                     or avail.get(e.id, {}).get(d.weekday()) != "unavailable")
            ]
            # 公平轮转:当月累计最少优先,同数按工号稳定排序
            cands.sort(key=lambda e: (assigned_cnt[e.id], e.employee_no))
            picked, cands = cands[:need], cands[need:]
            for e in picked:
                plan.append({
                    "day": d.isoformat(), "shift_id": s.id, "shift_name": s.name,
                    "employee_id": e.id, "employee_name": e.name, "source": "auto",
                })
                busy[e.id].add(d)
                filled[(d, s.id)] += 1
                assigned_cnt[e.id] += 1
            lack = need - len(picked)
            if lack > 0:
                skipped.append({
                    "day": d.isoformat(), "shift_id": s.id, "shift_name": s.name,
                    "missing": lack, "reason": "可排员工不足(请假/已排满)",
                })

    if not dry_run:
        for p in plan:
            db.add(ShiftAssignment(
                tenant_id=tenant_id, employee_id=p["employee_id"],
                shift_id=p["shift_id"], day=date.fromisoformat(p["day"]),
                source="auto", created_by=actor_id,
            ))
        db.commit()

    return {
        "planned": len(plan), "plan": plan,
        "understaffed": skipped,
        "workdays": len(workdays & set(days)),
        "shifts": len(shifts), "employees": len(employees),
    }


def find_gaps(
    db: Session, tenant_id: str, *, start: date, end: date,
    org_unit_id: str | None = None,
) -> list[dict]:
    """逐日逐班覆盖缺口。

    覆盖口径 = 排班员工当天无已批请假;排了班但人在请假算「待替班」,
    不占覆盖数。"""
    shifts, _emps, assigns = _load(db, tenant_id, start, end, org_unit_id, None)
    emp_names = {
        e.id: e.name for e in db.scalars(
            select(Employee).where(Employee.tenant_id == tenant_id)).all()
    }
    leave = _leave_days(db, tenant_id, start, end)
    by_slot: dict[tuple[date, str], dict] = {}
    for a in assigns:
        slot = by_slot.setdefault((a.day, a.shift_id), {"ok": [], "on_leave": []})
        (slot["on_leave"] if a.day in leave.get(a.employee_id, set())
         else slot["ok"]).append(a)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    out: list[dict] = []
    for d in days:
        for s in shifts:
            slot = by_slot.get((d, s.id), {"ok": [], "on_leave": []})
            covered = len(slot["ok"])
            on_leave = [
                {"assignment_id": a.id, "employee_id": a.employee_id,
                 "employee_name": emp_names.get(a.employee_id, "?")}
                for a in slot["on_leave"]
            ]
            missing = max(s.required - covered - len(on_leave), 0)
            if missing > 0 or on_leave:
                out.append({
                    "day": d.isoformat(), "shift_id": s.id, "shift_name": s.name,
                    "required": s.required, "covered": covered,
                    "on_leave": on_leave, "missing": missing,
                })
    return out


def cover_candidates(
    db: Session, tenant_id: str, *, day: date, shift_id: str | None = None,
    org_unit_id: str | None = None, limit: int = 5,
) -> list[dict]:
    """某天可替班候选人:在职、当天无假无班;同部门优先、当月累计少优先。"""
    month_start = day.replace(day=1)
    _, employees, assigns = _load(db, tenant_id, month_start, day, None, None)
    leave = _leave_days(db, tenant_id, day, day)
    busy = _busy_map(assigns)
    assigned_cnt: dict[str, int] = defaultdict(int)
    for a in assigns:
        assigned_cnt[a.employee_id] += 1
    cands = []
    for e in employees:
        if day in leave.get(e.id, set()) or day in busy.get(e.id, set()):
            continue
        cands.append({
            "employee_id": e.id, "employee_name": e.name,
            "org_unit_id": e.org_unit_id,
            "same_org": bool(org_unit_id and e.org_unit_id == org_unit_id),
            "assigned_this_month": assigned_cnt[e.id],
        })
    cands.sort(key=lambda c: (not c["same_org"], c["assigned_this_month"],
                              c["employee_name"]))
    return cands[:limit]


def auto_cover(
    db: Session, tenant_id: str, *, start: date, end: date,
    actor_id: str | None = None, dry_run: bool = False,
) -> dict:
    """自动替班:先补「排班但请假」的换班,再填覆盖不足的槽位。"""
    gaps = find_gaps(db, tenant_id, start=start, end=end)
    emp_orgs = {
        e.id: e.org_unit_id for e in db.scalars(
            select(Employee).where(Employee.tenant_id == tenant_id)).all()
    }
    covered: list[dict] = []
    filled: list[dict] = []
    remaining: list[dict] = []

    for g in gaps:
        day = date.fromisoformat(g["day"])
        # 替班:请假员工的 assignment 转给候选人
        for ol in g["on_leave"]:
            cands = cover_candidates(
                db, tenant_id, day=day, shift_id=g["shift_id"],
                org_unit_id=emp_orgs.get(ol["employee_id"]), limit=1)
            if not cands:
                remaining.append({"day": g["day"], "shift_name": g["shift_name"],
                                  "reason": f"{ol['employee_name']}请假,无人可替"})
                continue
            sub = cands[0]
            covered.append({
                "day": g["day"], "shift_name": g["shift_name"],
                "from": ol["employee_name"], "to": sub["employee_name"],
                "assignment_id": ol["assignment_id"],
            })
            if not dry_run:
                a = db.get(ShiftAssignment, ol["assignment_id"])
                a.employee_id = sub["employee_id"]
                a.source = "cover"
                a.note = f"替{ol['employee_name']}班(请假)"
        # 填缺口
        for _ in range(g["missing"]):
            cands = cover_candidates(
                db, tenant_id, day=day, shift_id=g["shift_id"], limit=5)
            taken = {f["employee_id"] for f in filled if f["day"] == g["day"]}
            cands = [c for c in cands if c["employee_id"] not in taken]
            if not cands:
                remaining.append({"day": g["day"], "shift_name": g["shift_name"],
                                  "reason": "覆盖不足且无可排员工"})
                break
            sub = cands[0]
            filled.append({"day": g["day"], "shift_id": g["shift_id"],
                           "shift_name": g["shift_name"],
                           "employee_id": sub["employee_id"],
                           "employee_name": sub["employee_name"]})
            if not dry_run:
                db.add(ShiftAssignment(
                    tenant_id=tenant_id, employee_id=sub["employee_id"],
                    shift_id=g["shift_id"], day=day, source="cover",
                    note="自动补缺", created_by=actor_id))
    if not dry_run:
        db.commit()
    return {"covered": covered, "filled": filled, "remaining": remaining,
            "dry_run": dry_run}
