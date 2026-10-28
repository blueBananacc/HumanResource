"""排班 API(H3 + 喔趣排班 2.0)。

- ``GET/POST/PUT/DELETE /shifts``            班次定义(早/晚班 + 每班需人数)
- ``GET/POST/DELETE /shifts/assignments``    排班明细(按月/员工过滤)
- ``POST /shifts/generate``                  生成排班(dry_run 预览 / 落库)
- ``GET  /shifts/gaps``                      覆盖缺口(含「排班但请假」虚覆盖)
- ``GET  /shifts/cover-candidates``          某天可替班候选人
- ``POST /shifts/auto-cover``                自动替班 + 补缺
- ``POST /shifts/assignments/{id}/reassign`` 手工指定替班人
- ``GET/PUT /shifts/rules``                  合规规则(连班/周工时/休息间隔/可用性)
- ``GET  /shifts/compliance``                全员排班合规体检
- ``GET/POST/PUT/DELETE /shifts/templates``  排班模板/循环规则
- ``POST /shifts/templates/{id}/apply``      模板套用生成排班
- ``GET  /shifts/swaps``                     换班申请列表(HR 审批队列)
- ``POST /shifts/swaps/{id}/decide``         批准(执行交换)/驳回
"""
from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    Notification,
    ScheduleTemplate,
    Shift,
    ShiftAssignment,
    ShiftSwapRequest,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_ATTENDANCE,
    PERM_VIEW_REPORTS,
    require_permission,
)
from app.services.scheduling import (
    _busy_map,
    _leave_days,
    auto_cover,
    cover_candidates,
    find_gaps,
    generate_schedule,
    month_range,
)
from app.services.scheduling_rules import (
    _avail_map,
    _load_pairs,
    evaluate_pairs,
    get_rules,
    precheck,
    rules_out,
    violations_for_employee,
    violations_for_window,
)
from app.services.work_calendar import workday_dates


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)

router = APIRouter(prefix="/shifts", tags=["hris-shifts"],
                 dependencies=[Depends(require_feature("hr.suite"))])


def _min_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


class ShiftOut(BaseModel):
    id: str
    name: str
    start_min: int
    end_min: int
    time_range: str
    required: int
    color: str | None
    active: bool


def _shift_out(s: Shift) -> ShiftOut:
    return ShiftOut(
        id=s.id, name=s.name, start_min=s.start_min, end_min=s.end_min,
        time_range=f"{_min_to_hhmm(s.start_min)}-{_min_to_hhmm(s.end_min)}",
        required=s.required, color=s.color, active=s.active,
    )


class ShiftIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    start_min: int = Field(540, ge=0, lt=1440)
    end_min: int = Field(1020, ge=0, le=1440)
    required: int = Field(1, ge=1, le=99)
    color: str | None = None
    active: bool = True


@router.get("", response_model=list[ShiftOut])
def list_shifts(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    rows = db.scalars(
        select(Shift).where(Shift.tenant_id == current.tenant_id)
        .order_by(Shift.start_min)).all()
    return [_shift_out(s) for s in rows]


@router.post("", response_model=ShiftOut, status_code=status.HTTP_201_CREATED)
def create_shift(
    body: ShiftIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = Shift(tenant_id=current.tenant_id, **body.model_dump())
    db.add(s)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(409, "同名班次已存在")
    db.refresh(s)
    return _shift_out(s)


# ===== 合规规则 / 体检(喔趣「从源头规避用工风险」) =====
# 注意:以下静态路径必须注册在 ``/{shift_id}`` 路由之前。

class RulesIn(BaseModel):
    max_consecutive_days: int = Field(6, ge=1, le=14)
    max_weekly_hours: float = Field(44, ge=1, le=80)
    min_rest_minutes: int = Field(720, ge=0, le=1440)
    respect_availability: bool = True


@router.get("/rules")
def shift_rules_get(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return rules_out(get_rules(db, current.tenant_id))


@router.put("/rules")
def shift_rules_put(
    body: RulesIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    r = get_rules(db, current.tenant_id)
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    write_audit(db, actor=current, entity_type="shift_rules",
                entity_id=r.id, action="update",
                detail=body.model_dump(), request=request)
    db.commit()
    return rules_out(r)


@router.get("/employees")
def plannable_employees(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """可排班员工下拉数据(排班经理未必有员工管理权限,单独给轻量列表)。"""
    rows = db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id,
            Employee.status == "active")
        .order_by(Employee.employee_no)).all()
    return [{"id": e.id, "name": e.name, "employee_no": e.employee_no,
             "org_unit_id": e.org_unit_id} for e in rows]


@router.get("/compliance")
def compliance(
    month: str | None = Query(None, description="YYYY-MM"),
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    start, end = month_range(month)
    return {"month": start.isoformat()[:7],
            "violations": violations_for_window(
                db, current.tenant_id, start=start, end=end,
                org_unit_id=org_unit_id)}


# ===== AI 排班助手(盖雅「时间智能体」:解释排班/多方案对比) =====
#
# 三个只读端点把时间智能体的「为什么」与「换谁更好」做成确定性数据:
# - ``/board``       某日各班次覆盖看板(谁上班/谁请假/缺几人)
# - ``/explain``     某员工某天排班全景:在排什么、为什么没排、若排某班会违规吗
# - ``/plan-compare`` 某槽位候选方案对比:逐人合规预判 + 公平/偏好优劣项


@router.get("/board")
def day_board(
    day: date,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """某日排班看板:班次 × 已排员工(含请假标记)× 缺口。"""
    shifts = db.scalars(
        select(Shift).where(
            Shift.tenant_id == current.tenant_id, Shift.active.is_(True))
        .order_by(Shift.start_min)).all()
    rows = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == current.tenant_id,
            ShiftAssignment.day == day)).all()
    emps, _ = _name_maps(db, current.tenant_id)
    leave = _leave_days(db, current.tenant_id, day, day)
    slots = []
    for s in shifts:
        assigned = [
            {
                "assignment_id": a.id,
                "employee_id": a.employee_id,
                "employee_name": emps.get(a.employee_id, "?"),
                "source": a.source,
                "on_leave": day in leave.get(a.employee_id, set()),
            }
            for a in rows if a.shift_id == s.id
        ]
        on_leave = sum(1 for x in assigned if x["on_leave"])
        covered = len(assigned) - on_leave
        slots.append({
            "shift_id": s.id, "shift_name": s.name,
            "time_range": f"{_min_to_hhmm(s.start_min)}-{_min_to_hhmm(s.end_min)}",
            "required": s.required, "covered": covered,
            "missing": max(s.required - covered, 0),
            "assigned": assigned,
        })
    return {"day": day.isoformat(), "slots": slots}


def _consecutive_streak(pairs: list[tuple[date, Shift]], day: date) -> int:
    """截至 day(含)的连续排班天数。"""
    days = {d for d, _ in pairs}
    n, cur = 0, day
    while cur in days:
        n += 1
        cur -= timedelta(days=1)
    return n


@router.get("/explain")
def explain_schedule(
    employee_id: str,
    day: date,
    shift_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """解释某员工某天的排班状态与可排性(AI 排班助手核心)。

    - 已排:返回班次+来源+是否请假中
    - 未排:列出原因(请假/非工作日无班次需求/登记不可排)
    - ``shift_id`` 指定时附加「假设排此班」的合规预判 violations
    """
    emp = db.get(Employee, employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    emps, shift_names = _name_maps(db, current.tenant_id)
    assignment = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == current.tenant_id,
            ShiftAssignment.employee_id == employee_id,
            ShiftAssignment.day == day)).first()
    on_leave = day in _leave_days(db, current.tenant_id, day, day).get(
        employee_id, set())
    avail = _avail_map(db, current.tenant_id, [employee_id]).get(
        employee_id, {})
    pref = avail.get(day.weekday())
    pairs, _shifts = _load_pairs(
        db, current.tenant_id, [employee_id],
        day.replace(day=1), day)
    month_cnt = len(pairs.get(employee_id, []))
    reasons: list[str] = []
    if assignment:
        reasons.append(
            f"当天已排「{shift_names.get(assignment.shift_id, '?')}」"
            f"(来源 {assignment.source})")
        if on_leave:
            reasons.append("但当天有已批请假 — 属于待替班项")
    else:
        if on_leave:
            reasons.append("当天有已批请假,不可排班")
        if pref == "unavailable":
            reasons.append("员工登记该 weekday 为不可排")
        if not reasons:
            reasons.append("当天空闲可排(无排班/无请假)")
    if pref == "preferred":
        reasons.append("员工登记该 weekday 为偏好日")
    # 假设合规预判
    hypothetical: dict | None = None
    target = None
    if shift_id:
        target = db.get(Shift, shift_id)
        if not target or target.tenant_id != current.tenant_id:
            raise HTTPException(404, "班次不存在")
    if target is not None:
        vs = precheck(db, current.tenant_id, employee_id, day, target)
        hypothetical = {
            "shift_id": target.id, "shift_name": target.name,
            "ok": not any(v["severity"] == "block" for v in vs),
            "violations": vs,
        }
    pending_swaps = db.scalars(
        select(ShiftSwapRequest).where(
            ShiftSwapRequest.tenant_id == current.tenant_id,
            ShiftSwapRequest.day == day,
            ShiftSwapRequest.status.in_(
                ["pending_peer", "pending_approval"]),
            (ShiftSwapRequest.requester_employee_id == employee_id)
            | (ShiftSwapRequest.target_employee_id == employee_id),
        )).all()
    return {
        "employee_id": employee_id, "employee_name": emp.name,
        "day": day.isoformat(), "weekday": day.weekday(),
        "is_workday": day in set(
            workday_dates(db, current.tenant_id, day, day)),
        "availability": pref or "unset",
        "assignment": (
            {"id": assignment.id, "shift_id": assignment.shift_id,
             "shift_name": shift_names.get(assignment.shift_id, "?"),
             "source": assignment.source, "on_leave": on_leave}
            if assignment else None),
        "on_leave": on_leave,
        "month_assigned_count": month_cnt,
        "consecutive_days_to_date": _consecutive_streak(
            pairs.get(employee_id, []), day),
        "pending_swaps": [
            {"id": s.id, "status": s.status, "mode": s.mode}
            for s in pending_swaps],
        "hypothetical": hypothetical,
        "reasons": reasons,
    }


@router.get("/plan-compare")
def plan_compare(
    day: date,
    shift_id: str,
    limit: int = Query(5, ge=1, le=15),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """某槽位候选方案对比(盖雅:多种微调方案 + 解释各方案优劣)。

    对每个可排候选人给出:合规预判(若排此班引入的违规)、当月已排数
    (公平性)、可用性偏好、同部门标记,并汇总确定性优劣项供 LLM 讲解。
    """
    shift = db.get(Shift, shift_id)
    if not shift or shift.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    month_start = day.replace(day=1)
    employees = db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id,
            Employee.status == "active").order_by(Employee.employee_no)).all()
    assigns = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == current.tenant_id,
            ShiftAssignment.day >= month_start,
            ShiftAssignment.day <= day)).all()
    busy = _busy_map(assigns)
    cnt: dict[str, int] = defaultdict(int)
    org_by_emp: dict[str, str | None] = {}
    for a in assigns:
        cnt[a.employee_id] += 1
    leave = _leave_days(db, current.tenant_id, day, day)
    avail = _avail_map(db, current.tenant_id, [e.id for e in employees])
    for e in employees:
        org_by_emp[e.id] = e.org_unit_id
    # 当日该班已排员工的部门,作为 same_org 参照(取多数)
    day_orgs = [
        org_by_emp.get(a.employee_id)
        for a in assigns if a.day == day and a.shift_id == shift_id]
    anchor_org = day_orgs[0] if day_orgs else None

    plans: list[dict] = []
    ineligible: list[dict] = []
    for e in employees:
        if day in leave.get(e.id, set()):
            ineligible.append({"employee_id": e.id, "employee_name": e.name,
                               "reason": "当天有已批请假"})
            continue
        if day in busy.get(e.id, set()):
            ineligible.append({"employee_id": e.id, "employee_name": e.name,
                               "reason": "当天已有排班"})
            continue
        vs = precheck(db, current.tenant_id, e.id, day, shift)
        blocks = [v for v in vs if v["severity"] == "block"]
        warns = [v for v in vs if v["severity"] == "warn"]
        pref = avail.get(e.id, {}).get(day.weekday())
        pros: list[str] = []
        cons: list[str] = []
        if not blocks:
            pros.append("无合规阻断")
        for v in blocks:
            cons.append(f"违规:{v['detail']}")
        for v in warns:
            cons.append(f"提醒:{v['detail']}")
        if pref == "preferred":
            pros.append("员工登记该日为偏好日")
        elif pref == "unavailable":
            cons.append("员工登记该日为不可排")
        if anchor_org and e.org_unit_id == anchor_org:
            pros.append("与当班同事同部门")
        plans.append({
            "employee_id": e.id, "employee_name": e.name,
            "org_unit_id": e.org_unit_id,
            "assigned_this_month": cnt[e.id],
            "eligible": not blocks,
            "violations": vs, "pros": pros, "cons": cons,
        })
    # 排序:合规优先 → 当月已排少优先(公平) → 偏好日加权
    plans.sort(key=lambda p: (
        not p["eligible"], p["assigned_this_month"],
        0 if avail.get(p["employee_id"], {}).get(day.weekday())
        == "preferred" else 1,
        p["employee_name"]))
    return {
        "day": day.isoformat(), "shift_id": shift_id,
        "shift_name": shift.name,
        "plans": plans[:limit],
        "ineligible": ineligible[:10],
        "pool_size": len(plans),
    }


# ===== 排班模板 / 循环规则(喔趣:固定班次·多班次循环模板) =====

class TemplateIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    period_days: int = Field(7, ge=1, le=28)
    pattern: list[dict] = Field(default_factory=list)
    active: bool = True


def _check_pattern(db: Session, tenant_id: str, body: TemplateIn) -> None:
    offsets: set[int] = set()
    shift_ids: set[str] = set()
    for p in body.pattern:
        try:
            off = int(p.get("offset"))
        except (TypeError, ValueError):
            raise HTTPException(422, "pattern.offset 必须是整数")
        if not 0 <= off < body.period_days:
            raise HTTPException(422, f"offset {off} 超出周期 {body.period_days}")
        if off in offsets:
            raise HTTPException(422, f"offset {off} 重复")
        offsets.add(off)
        if p.get("shift_id"):
            shift_ids.add(str(p["shift_id"]))
    if shift_ids:
        found = {
            s.id for s in db.scalars(
                select(Shift).where(
                    Shift.tenant_id == tenant_id, Shift.id.in_(shift_ids))).all()
        }
        missing = shift_ids - found
        if missing:
            raise HTTPException(422, "模板引用了不存在的班次")


class TemplateOut(BaseModel):
    id: str
    name: str
    period_days: int
    pattern: list[dict]
    active: bool


def _tpl_out(t: ScheduleTemplate) -> TemplateOut:
    return TemplateOut(id=t.id, name=t.name, period_days=t.period_days,
                       pattern=t.pattern or [], active=t.active)


@router.get("/templates", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    rows = db.scalars(
        select(ScheduleTemplate).where(
            ScheduleTemplate.tenant_id == current.tenant_id)
        .order_by(ScheduleTemplate.created_at)).all()
    return [_tpl_out(t) for t in rows]


@router.post("/templates", response_model=TemplateOut,
             status_code=status.HTTP_201_CREATED)
def create_template(
    body: TemplateIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    _check_pattern(db, current.tenant_id, body)
    t = ScheduleTemplate(tenant_id=current.tenant_id, **body.model_dump())
    db.add(t)
    db.commit()
    db.refresh(t)
    return _tpl_out(t)


@router.put("/templates/{tid}", response_model=TemplateOut)
def update_template(
    tid: str,
    body: TemplateIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    t = db.get(ScheduleTemplate, tid)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(404, "模板不存在")
    _check_pattern(db, current.tenant_id, body)
    for k, v in body.model_dump().items():
        setattr(t, k, v)
    db.commit()
    return _tpl_out(t)


@router.delete("/templates/{tid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(
    tid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    t = db.get(ScheduleTemplate, tid)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(404, "模板不存在")
    db.delete(t)
    db.commit()


class TemplateApplyIn(BaseModel):
    employee_ids: list[str] | None = None
    org_unit_id: str | None = None
    start: date
    cycles: int = Field(1, ge=1, le=12)
    dry_run: bool = True


@router.post("/templates/{tid}/apply")
def apply_template(
    tid: str,
    body: TemplateApplyIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    t = db.get(ScheduleTemplate, tid)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(404, "模板不存在")
    pats: dict[int, str | None] = {}
    for p in t.pattern or []:
        try:
            pats[int(p.get("offset"))] = (
                str(p["shift_id"]) if p.get("shift_id") else None)
        except (TypeError, ValueError):
            continue
    days = [body.start + timedelta(days=i)
            for i in range(t.period_days * body.cycles)]
    end = days[-1]
    if body.org_unit_id:
        emps = db.scalars(
            select(Employee).where(
                Employee.tenant_id == current.tenant_id,
                Employee.org_unit_id == body.org_unit_id,
                Employee.status == "active")).all()
    elif body.employee_ids:
        emps = db.scalars(
            select(Employee).where(
                Employee.tenant_id == current.tenant_id,
                Employee.id.in_(body.employee_ids),
                Employee.status == "active")).all()
        if len(emps) != len(set(body.employee_ids)):
            raise HTTPException(404, "存在无效员工(非在职/跨租户)")
    else:
        raise HTTPException(422, "employee_ids 与 org_unit_id 至少提供一个")
    sid_needed = {v for v in pats.values() if v}
    shifts = {
        s.id: s for s in db.scalars(
            select(Shift).where(Shift.id.in_(sid_needed or {""}))).all()
    }
    existing = {
        (a.employee_id, a.day)
        for a in db.scalars(
            select(ShiftAssignment).where(
                ShiftAssignment.tenant_id == current.tenant_id,
                ShiftAssignment.day >= body.start,
                ShiftAssignment.day <= end)).all()
    }
    leave = _leave_days(db, current.tenant_id, body.start, end)
    plan: list[dict] = []
    skipped: list[dict] = []
    seen = set(existing)
    for e in emps:
        for d in days:
            sid = pats.get((d - body.start).days % t.period_days)
            if not sid:
                continue
            s = shifts.get(sid)
            if s is None or not s.active:
                skipped.append({"employee_id": e.id, "day": d.isoformat(),
                                "reason": "班次不存在或已停用"})
                continue
            if (e.id, d) in seen:
                skipped.append({"employee_id": e.id, "day": d.isoformat(),
                                "reason": "当天已有排班"})
                continue
            if d in leave.get(e.id, set()):
                skipped.append({"employee_id": e.id, "day": d.isoformat(),
                                "reason": "当天有已批请假"})
                continue
            plan.append({"employee_id": e.id, "employee_name": e.name,
                         "day": d.isoformat(), "shift_id": sid,
                         "shift_name": s.name})
            seen.add((e.id, d))
    # 合规体检(合并候选排班,告警不阻断)
    warnings: list[dict] = []
    if plan:
        rules = get_rules(db, current.tenant_id)
        pairs, _ = _load_pairs(
            db, current.tenant_id, [e.id for e in emps], body.start, end)
        avail = _avail_map(db, current.tenant_id, [e.id for e in emps])
        names = {e.id: e.name for e in emps}
        by_emp: dict[str, list] = {}
        for p in plan:
            by_emp.setdefault(p["employee_id"], []).append(
                (date.fromisoformat(p["day"]), shifts[p["shift_id"]]))
        for eid, extra in by_emp.items():
            for v in evaluate_pairs(
                    pairs.get(eid, []) + extra, rules, avail.get(eid, {})):
                warnings.append(
                    {"employee_id": eid, "employee_name": names[eid], **v})
    if not body.dry_run:
        for p in plan:
            db.add(ShiftAssignment(
                tenant_id=current.tenant_id, employee_id=p["employee_id"],
                shift_id=p["shift_id"], day=date.fromisoformat(p["day"]),
                source="template", created_by=current.id))
        write_audit(db, actor=current, entity_type="schedule_template",
                    entity_id=t.id, action="apply",
                    detail={"start": body.start.isoformat(),
                            "cycles": body.cycles,
                            "planned": len(plan)},
                    request=request)
        db.commit()
    return {"planned": len(plan), "plan": plan, "skipped": skipped,
            "warnings": warnings, "dry_run": body.dry_run}


# ===== 换班申请(HR 审批队列) =====

class SwapOut(BaseModel):
    id: str
    day: date
    requester_employee_id: str
    requester_name: str
    target_employee_id: str
    target_name: str
    mode: str
    status: str
    requester_note: str | None
    peer_note: str | None
    decision_note: str | None
    created_at: str


def _swap_out(r: ShiftSwapRequest, names: dict[str, str]) -> SwapOut:
    return SwapOut(
        id=r.id, day=r.day,
        requester_employee_id=r.requester_employee_id,
        requester_name=names.get(r.requester_employee_id, "?"),
        target_employee_id=r.target_employee_id,
        target_name=names.get(r.target_employee_id, "?"),
        mode=r.mode, status=r.status,
        requester_note=r.requester_note, peer_note=r.peer_note,
        decision_note=r.decision_note,
        created_at=r.created_at.isoformat())


def _emp_names(db: Session, tenant_id: str) -> dict[str, str]:
    return {e.id: e.name for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)).all()}


def _notify(db: Session, tenant_id: str, emp_id: str, *,
            kind: str, title: str, body: str | None = None) -> None:
    emp = db.get(Employee, emp_id)
    if emp is not None and emp.user_id:
        db.add(Notification(
            tenant_id=tenant_id, user_id=emp.user_id, kind=kind,
            title=title[:128], body=(body or "")[:512],
            link="/me/shifts"))


@router.get("/swaps", response_model=list[SwapOut])
def list_swaps(
    status: str | None = None,
    month: str | None = Query(None, description="YYYY-MM"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    stmt = select(ShiftSwapRequest).where(
        ShiftSwapRequest.tenant_id == current.tenant_id)
    if status:
        stmt = stmt.where(ShiftSwapRequest.status == status)
    if month:
        start, end = month_range(month)
        stmt = stmt.where(ShiftSwapRequest.day >= start,
                          ShiftSwapRequest.day <= end)
    rows = db.scalars(
        stmt.order_by(ShiftSwapRequest.created_at.desc())).all()
    names = _emp_names(db, current.tenant_id)
    return [_swap_out(r, names) for r in rows]


class SwapDecideIn(BaseModel):
    approve: bool
    note: str | None = None


@router.post("/swaps/{swap_id}/decide")
def decide_swap(
    swap_id: str,
    body: SwapDecideIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    r = db.get(ShiftSwapRequest, swap_id)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(404, "换班申请不存在")
    if r.status != "pending_approval":
        raise HTTPException(409, "申请不在待审批状态")
    warnings: list[dict] = []
    names = _emp_names(db, current.tenant_id)
    if body.approve:
        ra = db.scalars(
            select(ShiftAssignment).where(
                ShiftAssignment.tenant_id == current.tenant_id,
                ShiftAssignment.employee_id == r.requester_employee_id,
                ShiftAssignment.day == r.day)).first()
        if ra is None:
            raise HTTPException(409, "发起方当天排班已变化,无法交换")
        if r.mode == "exchange":
            ta = db.scalars(
                select(ShiftAssignment).where(
                    ShiftAssignment.tenant_id == current.tenant_id,
                    ShiftAssignment.employee_id == r.target_employee_id,
                    ShiftAssignment.day == r.day)).first()
            if ta is None:
                raise HTTPException(409, "对方当天排班已变化,无法交换")
            ra.shift_id, ta.shift_id = ta.shift_id, ra.shift_id
            ra.source = ta.source = "swap"
        else:
            clash = db.scalars(
                select(ShiftAssignment).where(
                    ShiftAssignment.tenant_id == current.tenant_id,
                    ShiftAssignment.employee_id == r.target_employee_id,
                    ShiftAssignment.day == r.day)).first()
            if clash is not None:
                raise HTTPException(409, "对方当天已有排班,无法转入")
            ra.employee_id = r.target_employee_id
            ra.source = "swap"
            ra.note = f"换班(转给{names.get(r.target_employee_id, '?')})"
        r.status = "approved"
        for eid in (r.requester_employee_id, r.target_employee_id):
            _notify(db, current.tenant_id, eid, kind="shift.swap",
                    title="换班申请已批准",
                    body=f"{r.day.isoformat()} 的换班已生效,请查看新班次")
        # 事后体检:交换引入的违规做告警返回(双方已同意,不阻断)
        for eid in (r.requester_employee_id, r.target_employee_id):
            for v in violations_for_employee(
                    db, current.tenant_id, eid,
                    start=r.day - timedelta(days=8),
                    end=r.day + timedelta(days=8)):
                if v["day"] >= r.day.isoformat():
                    warnings.append(
                        {"employee_id": eid,
                         "employee_name": names.get(eid, "?"), **v})
    else:
        r.status = "rejected"
        _notify(db, current.tenant_id, r.requester_employee_id,
                kind="shift.swap", title="换班申请被驳回",
                body=body.note or "审批未通过")
    r.decided_at = _now()
    r.approver_id = current.id
    r.decision_note = body.note
    write_audit(db, actor=current, entity_type="shift_swap",
                entity_id=r.id,
                action="approve" if body.approve else "reject",
                detail={"day": r.day.isoformat(), "mode": r.mode},
                request=request)
    db.commit()
    return {"swap": _swap_out(r, names), "warnings": warnings}


@router.put("/{shift_id}", response_model=ShiftOut)
def update_shift(
    shift_id: str,
    body: ShiftIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = db.get(Shift, shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    for k, v in body.model_dump().items():
        setattr(s, k, v)
    db.commit()
    return _shift_out(s)


@router.delete("/{shift_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shift(
    shift_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = db.get(Shift, shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    has = db.scalars(
        select(ShiftAssignment.id).where(ShiftAssignment.shift_id == shift_id)
        .limit(1)).first()
    if has:
        # 有排班引用的班次不物理删,停用保住历史
        s.active = False
        db.commit()
        return
    db.delete(s)
    db.commit()


# ===== 排班明细 =====

class AssignmentOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    shift_id: str
    shift_name: str
    day: date
    source: str
    note: str | None


def _assign_out(a: ShiftAssignment, emps: dict, shifts: dict) -> AssignmentOut:
    return AssignmentOut(
        id=a.id, employee_id=a.employee_id,
        employee_name=emps.get(a.employee_id, "?"),
        shift_id=a.shift_id, shift_name=shifts.get(a.shift_id, "?"),
        day=a.day, source=a.source, note=a.note,
    )


def _name_maps(db: Session, tenant_id: str):
    emps = {e.id: e.name for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)).all()}
    shifts = {s.id: s.name for s in db.scalars(
        select(Shift).where(Shift.tenant_id == tenant_id)).all()}
    return emps, shifts


@router.get("/assignments", response_model=PageOut[AssignmentOut])
def list_assignments(
    pagination: tuple = Depends(paginate_params),
    month: str | None = Query(None, description="YYYY-MM"),
    employee_id: str | None = None,
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    start, end = month_range(month)
    stmt = select(ShiftAssignment).where(
        ShiftAssignment.tenant_id == current.tenant_id,
        ShiftAssignment.day >= start, ShiftAssignment.day <= end)
    if employee_id:
        stmt = stmt.where(ShiftAssignment.employee_id == employee_id)
    if org_unit_id:
        emp_ids = select(Employee.id).where(
            Employee.tenant_id == current.tenant_id,
            Employee.org_unit_id == org_unit_id)
        stmt = stmt.where(ShiftAssignment.employee_id.in_(emp_ids))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(ShiftAssignment.day, ShiftAssignment.shift_id)
        .limit(limit).offset(offset)).all()
    emps, shifts = _name_maps(db, current.tenant_id)
    return PageOut(
        items=[_assign_out(a, emps, shifts) for a in rows],
        total=total, limit=limit, offset=offset)


class AssignmentIn(BaseModel):
    employee_id: str
    shift_id: str
    day: date
    note: str | None = None
    force: bool = False


def _assert_compliant(
    db: Session, tenant_id: str, employee_id: str, day: date,
    shift: Shift, force: bool,
) -> None:
    """前置合规校验:有阻断级违规且未确认放行 → 409 携带违规明细。"""
    if force:
        return
    vs = [v for v in precheck(db, tenant_id, employee_id, day, shift)
          if v["severity"] == "block"]
    if vs:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"message": "排班违反合规规则", "violations": vs})


@router.post("/assignments", response_model=AssignmentOut,
             status_code=status.HTTP_201_CREATED)
def create_assignment(
    body: AssignmentIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    s = db.get(Shift, body.shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    _assert_compliant(
        db, current.tenant_id, body.employee_id, body.day, s, body.force)
    a = ShiftAssignment(
        tenant_id=current.tenant_id, source="manual",
        **body.model_dump(exclude={"force"}))
    db.add(a)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(409, "该员工当天已有排班")
    emps, shifts = _name_maps(db, current.tenant_id)
    return _assign_out(a, emps, shifts)


@router.delete("/assignments/{assign_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assignment(
    assign_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    a = db.get(ShiftAssignment, assign_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(404, "排班不存在")
    db.delete(a)
    db.commit()


# ===== 生成 / 缺口 / 替班 =====

class GenerateIn(BaseModel):
    month: str | None = Field(None, description="YYYY-MM,默认当月")
    org_unit_id: str | None = None
    shift_ids: list[str] | None = None
    dry_run: bool = True


@router.post("/generate")
def generate(
    body: GenerateIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    start, end = month_range(body.month)
    out = generate_schedule(
        db, current.tenant_id, start=start, end=end,
        org_unit_id=body.org_unit_id, shift_ids=body.shift_ids,
        dry_run=body.dry_run, actor_id=current.id)
    if not body.dry_run:
        write_audit(db, actor=current, entity_type="shift_assignment",
                    entity_id="-", action="generate",
                    detail={"month": start.isoformat()[:7],
                            "planned": out["planned"]},
                    request=request)
        db.commit()
    out["month"] = start.isoformat()[:7]
    return out


@router.get("/gaps")
def gaps(
    month: str | None = Query(None, description="YYYY-MM"),
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    start, end = month_range(month)
    return {"month": start.isoformat()[:7],
            "gaps": find_gaps(db, current.tenant_id, start=start, end=end,
                              org_unit_id=org_unit_id)}


@router.get("/cover-candidates")
def candidates(
    day: date,
    shift_id: str | None = None,
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return {"day": day.isoformat(),
            "candidates": cover_candidates(
                db, current.tenant_id, day=day, shift_id=shift_id,
                org_unit_id=org_unit_id)}


class AutoCoverIn(BaseModel):
    month: str | None = Field(None, description="YYYY-MM,默认当月")
    dry_run: bool = False


@router.post("/auto-cover")
def cover(
    body: AutoCoverIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    start, end = month_range(body.month)
    out = auto_cover(db, current.tenant_id, start=start, end=end,
                     actor_id=current.id, dry_run=body.dry_run)
    if not body.dry_run:
        write_audit(db, actor=current, entity_type="shift_assignment",
                    entity_id="-", action="auto_cover",
                    detail={"month": start.isoformat()[:7],
                            "covered": len(out["covered"]),
                            "filled": len(out["filled"])},
                    request=request)
        db.commit()
    out["month"] = start.isoformat()[:7]
    return out


class ReassignIn(BaseModel):
    employee_id: str
    note: str | None = None
    force: bool = False


@router.post("/assignments/{assign_id}/reassign", response_model=AssignmentOut)
def reassign(
    assign_id: str,
    body: ReassignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    a = db.get(ShiftAssignment, assign_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(404, "排班不存在")
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id or emp.status != "active":
        raise HTTPException(404, "替班员工不存在或已离职")
    s = db.get(Shift, a.shift_id)
    if s is not None:
        _assert_compliant(
            db, current.tenant_id, body.employee_id, a.day, s, body.force)
    clash = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == current.tenant_id,
            ShiftAssignment.employee_id == body.employee_id,
            ShiftAssignment.day == a.day, ShiftAssignment.id != a.id)
    ).first()
    if clash:
        raise HTTPException(409, "替班员工当天已有排班")
    write_audit(db, actor=current, entity_type="shift_assignment",
                entity_id=a.id, action="reassign",
                detail={"day": str(a.day), "from": a.employee_id,
                        "to": body.employee_id},
                request=request)
    a.employee_id = body.employee_id
    a.source = "cover"
    a.note = body.note or "手工替班"
    db.commit()
    emps, shifts = _name_maps(db, current.tenant_id)
    return _assign_out(a, emps, shifts)
