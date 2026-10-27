"""人才流失预警(规则版,对标 Moka 四层信号体系)。

不做 ML 模型——demo 数据密度下,**可解释的规则引擎**更合适:
每条预警都带 signals[] 明细(哪个信号、什么数据、权重多少),HRBP
拿去做关怀对话的触发点,而不是管控名单(预警仅 manage_employees 可见)。

四层信号映射(Moka 模型):
  L1 行为/敬业度  — 近 14 天无打卡、工时环比下滑
  L2 绩效发展     — 目标逾期未结/停滞、近两期上级评分下滑、长期无晋升
  L3 考勤请假     — 缺勤天数、请假环比激增
  L4 组织关系     — 90 天无过程反馈(便签)、不在任何在途项目、技能成长停滞

- ``GET /flight-risk``               组织范围内员工风险榜(score desc)
- ``GET /flight-risk/{employee_id}`` 单人信号明细
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    Attendance,
    Employee,
    EmployeeCheckin,
    EmployeeSkill,
    Goal,
    InsightNote,
    LifecycleEvent,
    OrgUnit,
    Project,
    ProjectMember,
)
from app.infra.db import get_db
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(
    prefix="/flight-risk", tags=["flight-risk"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_TODAY = date.today


class RiskSignal(BaseModel):
    key: str
    layer: str              # behavior|performance|attendance|relation
    label: str
    detail: str
    weight: int


class RiskOut(BaseModel):
    employee_id: str
    employee_name: str
    designation: str | None = None
    org_unit: str | None = None
    score: int
    level: str              # low|medium|high
    signals: list[RiskSignal] = Field(default_factory=list)


def _level(score: int) -> str:
    return "high" if score >= 60 else "medium" if score >= 30 else "low"


def _signals_for_employee(
    db: Session, tenant_id: str, emp: Employee, *,
    attend: dict[str, list[Attendance]],
    checkins: dict[str, list[datetime]],
    goals: dict[str, list[Goal]],
    appraisals: dict[str, list[Appraisal]],
    notes: dict[str, datetime],
    promotions: dict[str, datetime | None],
    projects: dict[str, int],
    skills_recent: dict[str, int],
) -> list[RiskSignal]:
    today = _TODAY()
    d30 = today - timedelta(days=30)
    d14 = today - timedelta(days=14)
    d90 = today - timedelta(days=90)
    out: list[RiskSignal] = []

    # ---- L1 行为/敬业度 ----
    recent_ins = [t for t in checkins.get(emp.id, []) if t.date() >= d14]
    if checkins.get(emp.id) and not recent_ins:
        out.append(RiskSignal(
            key="no_checkin_14d", layer="behavior", label="近 14 天无打卡",
            detail="打卡流水近两周为空(可能长期缺勤/停用)", weight=15,
        ))
    att = attend.get(emp.id, [])
    cur = [a for a in att if a.day >= d30]
    prev = [a for a in att if d30 - timedelta(days=30) <= a.day < d30]
    cur_h = [a.work_hours for a in cur if a.work_hours]
    prev_h = [a.work_hours for a in prev if a.work_hours]
    if cur_h and prev_h:
        drop = (sum(prev_h) / len(prev_h)) - (sum(cur_h) / len(cur_h))
        if drop >= 1.5:
            out.append(RiskSignal(
                key="hours_drop", layer="behavior", label="工时明显下滑",
                detail=f"近30天日均 {sum(cur_h)/len(cur_h):.1f}h,"
                       f"环比降 {drop:.1f}h", weight=15,
            ))

    # ---- L2 绩效发展 ----
    glist = goals.get(emp.id, [])
    overdue = [g for g in glist if g.status == "active" and g.due_date and g.due_date < today]
    if overdue:
        out.append(RiskSignal(
            key="goals_overdue", layer="performance", label="目标逾期未结",
            detail=" · ".join(g.title[:20] for g in overdue[:3]), weight=20,
        ))
    stalled = [g for g in glist if g.status == "active" and g.progress < 30]
    if len(glist) >= 2 and len(stalled) == len([g for g in glist if g.status == "active"]) and stalled:
        out.append(RiskSignal(
            key="goals_stalled", layer="performance", label="目标整体停滞",
            detail=f"{len(stalled)} 个在途目标进度均低于 30%", weight=15,
        ))
    appr = sorted(
        [a for a in appraisals.get(emp.id, []) if a.manager_rating is not None],
        key=lambda a: a.updated_at or a.created_at, reverse=True,
    )
    if len(appr) >= 2 and appr[0].manager_rating < appr[1].manager_rating:
        out.append(RiskSignal(
            key="rating_decline", layer="performance", label="绩效评分下滑",
            detail=f"上期 {appr[1].manager_rating}/5 → 本期 {appr[0].manager_rating}/5",
            weight=20,
        ))
    if emp.hire_date and (today - emp.hire_date).days >= 730:
        promo = promotions.get(emp.id)
        if not promo:
            out.append(RiskSignal(
                key="no_promotion_2y", layer="performance", label="司龄≥2年无晋升",
                detail=f"入职 {emp.hire_date.isoformat()} 以来无晋升事件",
                weight=10,
            ))
    if skills_recent.get(emp.id, 0) == 0:
        out.append(RiskSignal(
            key="skill_stagnant", layer="performance", label="技能档案 90 天无新增",
            detail="能力成长停滞(无新确认技能)", weight=10,
        ))

    # ---- L3 考勤请假 ----
    absent = [a for a in cur if a.status == "absent"]
    leave = [a for a in cur if a.status == "leave"]
    prev_leave = [a for a in prev if a.status == "leave"]
    if len(absent) >= 2:
        out.append(RiskSignal(
            key="absent_days", layer="attendance", label="缺勤天数偏多",
            detail=f"近30天缺勤 {len(absent)} 天", weight=20,
        ))
    if len(leave) >= 2 * max(len(prev_leave), 1) and len(leave) >= 3:
        out.append(RiskSignal(
            key="leave_spike", layer="attendance", label="请假频次激增",
            detail=f"近30天请假 {len(leave)} 天(前30天 {len(prev_leave)} 天)",
            weight=15,
        ))

    # ---- L4 组织关系 ----
    last_note = notes.get(emp.id)
    if last_note is None or last_note.date() < d90:
        out.append(RiskSignal(
            key="no_feedback_90d", layer="relation", label="90 天无过程反馈",
            detail="主管便签/复盘/里程碑均无更新,关注断档", weight=15,
        ))
    if projects.get(emp.id, 0) == 0:
        out.append(RiskSignal(
            key="no_project", layer="relation", label="不在任何在途项目",
            detail="组织连接度低", weight=10,
        ))
    return out


def _score(signals: list[RiskSignal]) -> int:
    return min(100, sum(s.weight for s in signals))


def _load_maps(db: Session, tenant_id: str, emp_ids: list[str]):
    """一次取齐信号源,避免 N+1(demo 规模可接受的全量聚合)。"""
    d60 = _TODAY() - timedelta(days=60)
    d90 = _TODAY() - timedelta(days=90)

    attend: dict[str, list[Attendance]] = {}
    for a in db.scalars(
        select(Attendance).where(
            Attendance.tenant_id == tenant_id,
            Attendance.employee_id.in_(emp_ids),
            Attendance.day >= d60,
        )
    ).all():
        attend.setdefault(a.employee_id, []).append(a)

    checkins: dict[str, list[datetime]] = {}
    for c in db.scalars(
        select(EmployeeCheckin).where(
            EmployeeCheckin.tenant_id == tenant_id,
            EmployeeCheckin.employee_id.in_(emp_ids),
            EmployeeCheckin.ts >= datetime.combine(d60, datetime.min.time()),
        )
    ).all():
        checkins.setdefault(c.employee_id, []).append(c.ts)

    goals: dict[str, list[Goal]] = {}
    for g in db.scalars(
        select(Goal).where(
            Goal.tenant_id == tenant_id, Goal.employee_id.in_(emp_ids),
        )
    ).all():
        goals.setdefault(g.employee_id, []).append(g)

    appraisals: dict[str, list[Appraisal]] = {}
    for a in db.scalars(
        select(Appraisal).where(
            Appraisal.tenant_id == tenant_id, Appraisal.employee_id.in_(emp_ids),
        )
    ).all():
        appraisals.setdefault(a.employee_id, []).append(a)

    notes: dict[str, datetime] = {}
    for n in db.scalars(
        select(InsightNote).where(
            InsightNote.tenant_id == tenant_id,
            InsightNote.employee_id.in_(emp_ids),
        )
    ).all():
        cur = notes.get(n.employee_id)
        if cur is None or n.created_at > cur:
            notes[n.employee_id] = n.created_at

    promotions: dict[str, datetime | None] = {}
    for e in db.scalars(
        select(LifecycleEvent).where(
            LifecycleEvent.tenant_id == tenant_id,
            LifecycleEvent.employee_id.in_(emp_ids),
            LifecycleEvent.kind == "promotion",
        )
    ).all():
        prev = promotions.get(e.employee_id)
        if prev is None or e.created_at > prev:
            promotions[e.employee_id] = e.created_at

    projects: dict[str, int] = {}
    for pm, _p in db.execute(
        select(ProjectMember, Project)
        .join(Project, Project.id == ProjectMember.project_id)
        .where(
            ProjectMember.tenant_id == tenant_id,
            ProjectMember.employee_id.in_(emp_ids),
            Project.status == "active",
        )
    ).all():
        projects[pm.employee_id] = projects.get(pm.employee_id, 0) + 1

    skills_recent: dict[str, int] = {}
    for es in db.scalars(
        select(EmployeeSkill).where(
            EmployeeSkill.tenant_id == tenant_id,
            EmployeeSkill.employee_id.in_(emp_ids),
            EmployeeSkill.status == "confirmed",
            EmployeeSkill.created_at >= datetime.combine(d90, datetime.min.time()),
        )
    ).all():
        skills_recent[es.employee_id] = skills_recent.get(es.employee_id, 0) + 1

    return dict(
        attend=attend, checkins=checkins, goals=goals, appraisals=appraisals,
        notes=notes, promotions=promotions, projects=projects,
        skills_recent=skills_recent,
    )


def _risk_rows(
    db: Session, tenant_id: str, emps: list[Employee]
) -> list[tuple[Employee, int, list[RiskSignal]]]:
    if not emps:
        return []
    maps = _load_maps(db, tenant_id, [e.id for e in emps])
    orgs = {
        o.id: o.name
        for o in db.scalars(select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)).all()
    }
    out = []
    for e in emps:
        sigs = _signals_for_employee(db, tenant_id, e, **maps)
        out.append((e, _score(sigs), sigs))
    return out


def _to_out(emp: Employee, score: int, sigs: list[RiskSignal], org_name: str | None):
    return RiskOut(
        employee_id=emp.id,
        employee_name=emp.name,
        designation=emp.designation,
        org_unit=org_name,
        score=score,
        level=_level(score),
        signals=sigs,
    )


@router.get("", response_model=list[RiskOut])
def flight_risk_board(
    level: str | None = Query(None, pattern="^(low|medium|high)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """组织范围内流失风险榜(仅 HR 可见 — 预警是关怀触发点,不是监控名单)。"""
    scope = visible_employee_ids(db, current)
    q = select(Employee).where(
        Employee.tenant_id == current.tenant_id, Employee.status == "active"
    )
    if scope is not None:
        if not scope:
            return []
        q = q.where(Employee.id.in_(list(scope)))
    emps = db.scalars(q).all()

    rows = _risk_rows(db, current.tenant_id, emps)
    orgs = {
        o.id: o.name
        for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)
        ).all()
    }
    out = [
        _to_out(e, s, sigs, orgs.get(e.org_unit_id))
        for e, s, sigs in sorted(rows, key=lambda r: r[1], reverse=True)
        if s > 0
    ]
    if level:
        out = [o for o in out if o.level == level]
    return out[:limit]


@router.get("/{employee_id}", response_model=RiskOut)
def flight_risk_one(
    employee_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    emp = db.get(Employee, employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    scope = visible_employee_ids(db, current)
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在")
    rows = _risk_rows(db, current.tenant_id, [emp])
    e, score, sigs = rows[0]
    org = db.get(OrgUnit, e.org_unit_id) if e.org_unit_id else None
    return _to_out(e, score, sigs, org.name if org else None)
