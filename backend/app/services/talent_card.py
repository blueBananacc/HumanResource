"""P 批次 — 人才画像聚合卡(只读视图,真相散在各业务域)。

聚合:基本信息 + 技能(confirmed/suggested 分组)+ 面试评价(沿
source_candidate 回溯)+ 项目经历 + 绩效摘要 + 洞察时间线。

可见性规则(Moka 概览页同款):
- 员工本人看自己的卡:隐藏 suggested 技能与 manager_only 洞察
- HR/主管:全量可见,含待确认队列
"""
from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Appraisal, Employee, EmployeeSkill, InsightNote, Interview, OrgUnit,
    Project, ProjectMember, Skill, SkillRating, SkillRatingRequest,
)


def _names(db: Session, ids: list[str], model, attr="name") -> dict[str, str]:
    if not ids:
        return {}
    rows = db.execute(select(model.id, getattr(model, attr)).where(model.id.in_(ids))).all()
    return {r[0]: r[1] for r in rows}


def build_card(
    db: Session, employee: Employee, *, viewer_employee_id: str | None = None,
) -> dict[str, Any]:
    """聚合员工画像卡。``viewer_employee_id`` = 查看者绑定的员工 id(用于本人脱敏)。"""
    tid = employee.tenant_id
    is_self = viewer_employee_id is not None and viewer_employee_id == employee.id

    org = db.get(OrgUnit, employee.org_unit_id) if employee.org_unit_id else None
    mgr = db.get(Employee, employee.reports_to_id) if employee.reports_to_id else None
    hire_days = (date.today() - employee.hire_date).days if employee.hire_date else None

    # ---- 技能 ----
    rows = db.execute(
        select(EmployeeSkill, Skill)
        .join(Skill, Skill.id == EmployeeSkill.skill_id)
        .where(EmployeeSkill.employee_id == employee.id)
        .order_by(EmployeeSkill.updated_at.desc())
    ).all()
    confirmed, suggested = [], []
    for es, sk in rows:
        item = {
            "id": es.id, "skill_id": sk.id, "name": sk.name,
            "category": sk.category, "level": es.level, "source": es.source,
            "evidence": es.evidence, "note": es.note, "status": es.status,
        }
        (suggested if es.status == "suggested" else confirmed).append(item)

    # ---- 多评价者评分:每技能挂各来源分数 + 分歧标记 ----
    rrows = db.execute(
        select(SkillRating, Employee)
        .join(Employee, Employee.id == SkillRating.rater_employee_id)
        .where(SkillRating.employee_id == employee.id)
        .order_by(SkillRating.created_at.desc())
    ).all()
    ratings_by_skill: dict[str, list[dict[str, Any]]] = {}
    for r, rater in rrows:
        ratings_by_skill.setdefault(r.skill_id, []).append({
            "id": r.id, "rater_kind": r.rater_kind, "rater_name": rater.name,
            "level": r.level, "comment": r.comment,
            "created_at": str(r.created_at or "")[:16],
        })
    for item in confirmed + suggested:
        rs = ratings_by_skill.get(item["skill_id"], [])
        item["ratings"] = rs
        levels = [x["level"] for x in rs]
        # 分歧:≥2 个来源且极差 ≥2 级(自评5 vs 经理评2 → 校准信号)
        item["divergent"] = len(rs) >= 2 and (max(levels) - min(levels)) >= 2

    # 进行中的评分请求(本人视图也可见 —— 让员工知道"已催办/在等谁")
    pend_reqs = db.scalars(
        select(SkillRatingRequest).where(
            SkillRatingRequest.employee_id == employee.id,
            SkillRatingRequest.status == "pending",
        ).order_by(SkillRatingRequest.created_at.desc())
    ).all()
    rater_names = _names(db, [q.rater_employee_id for q in pend_reqs], Employee)
    skill_names = _names(db, [q.skill_id for q in pend_reqs], Skill)
    pending_ratings = [{
        "id": q.id, "skill_id": q.skill_id,
        "skill_name": skill_names.get(q.skill_id),
        "rater_name": rater_names.get(q.rater_employee_id),
        "rater_kind": q.rater_kind, "created_at": str(q.created_at or "")[:16],
    } for q in pend_reqs]

    # ---- 面试评价(沿录用来源回溯)----
    interviews: list[dict[str, Any]] = []
    if employee.source_candidate_id:
        ints = db.scalars(
            select(Interview).where(
                Interview.tenant_id == tid,
                Interview.candidate_id == employee.source_candidate_id,
                Interview.status == "done",
            ).order_by(Interview.created_at.desc()).limit(5)
        ).all()
        for iv in ints:
            s = iv.summary or {}
            interviews.append({
                "id": iv.id, "created_at": str(iv.created_at or "")[:10],
                "recommendation": s.get("recommendation"),
                "dimension_scores": s.get("dimension_scores"),
            })

    # ---- 项目经历 ----
    prows = db.execute(
        select(ProjectMember, Project)
        .join(Project, Project.id == ProjectMember.project_id)
        .where(ProjectMember.employee_id == employee.id)
        .order_by(ProjectMember.joined_at.desc())
    ).all()
    projects = [{
        "project_id": p.id, "name": p.name, "status": p.status,
        "role": m.role, "allocation_pct": m.allocation_pct,
        "joined_via": m.joined_via, "contribution": m.contribution_summary,
        "joined_at": str(m.joined_at or "")[:10],
        "start_date": str(p.start_date or ""), "end_date": str(p.end_date or ""),
    } for m, p in prows]

    # ---- 绩效摘要(近 3 次)----
    aps = db.scalars(
        select(Appraisal).where(Appraisal.employee_id == employee.id)
        .order_by(Appraisal.updated_at.desc()).limit(3)
    ).all()
    appraisals = [{
        "id": a.id, "status": a.status,
        "self_rating": a.self_rating, "manager_rating": a.manager_rating,
        "updated_at": str(a.updated_at or "")[:10],
    } for a in aps]

    # ---- 洞察时间线 ----
    q = select(InsightNote).where(InsightNote.employee_id == employee.id)
    if is_self:
        q = q.where(InsightNote.visibility == "employee_visible")
    notes = db.scalars(q.order_by(InsightNote.created_at.desc()).limit(20)).all()
    insights = [{
        "id": n.id, "kind": n.kind, "title": n.title,
        "body": n.body[:300], "visibility": n.visibility,
        "source_ref": n.source_ref, "created_at": str(n.created_at or "")[:16],
    } for n in notes]

    card = {
        "employee": {
            "id": employee.id, "name": employee.name,
            "employee_no": employee.employee_no,
            "designation": employee.designation,
            "org_name": org.name if org else None,
            "manager_name": mgr.name if mgr else None,
            "hire_date": str(employee.hire_date or ""),
            "hire_days": hire_days,
            "status": employee.status,
        },
        "skills_confirmed": confirmed,
        "suggested_count": 0 if is_self else len(suggested),
        "interviews": interviews,
        "projects": projects,
        "appraisals": appraisals,
        "insights": insights,
        "skills_suggested": [] if is_self else suggested,
        "pending_rating_requests": pending_ratings,
    }
    return card
