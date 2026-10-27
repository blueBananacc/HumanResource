"""内部人才匹配 — 岗位机会 × 员工技能画像的双向匹配(北森内部人才市场)。

三要素对标:
- 员工个人资料 = ``EmployeeSkill``(confirmed,多源证据链)
- 工作机会    = 开放岗位 ``Job`` + 其 ``JobSkill`` 结构化要求
- 匹配选择    = 确定性评分(非 LLM):覆盖率×权重 + 技能等级 +
                **图谱兑现**(子技能可向上兑现父级要求,0.8 衰减——
                Workday related-skills 的轻量版)

输出坚持证据链传统:每个命中给出 level/source/via(direct|child),缺失
项单列 —— HR 看到的不只是分数,是"凭什么匹配"。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Employee, EmployeeSkill, Job, JobSkill, OrgUnit, Skill, User,
)
from app.infra.db import get_db
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    get_linked_employee_id,
    has_permission,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(
    prefix="/internal-match", tags=["internal-match"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

CHILD_DECAY = 0.8  # 子技能向上兑现父级要求的衰减系数


class MatchedSkill(BaseModel):
    skill_id: str
    name: str
    level: float          # 兑现后等级(child 兑现已折 0.8)
    via: str              # direct | child
    via_name: str | None = None   # via=child 时:员工实际持有的子技能名
    source: str | None = None     # EmployeeSkill.source(证据来源)


class RequiredSkill(BaseModel):
    skill_id: str
    name: str
    importance: str


def _requirements(db: Session, tenant_id: str, job_id: str) -> list[RequiredSkill]:
    rows = db.execute(
        select(JobSkill, Skill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.tenant_id == tenant_id, JobSkill.job_id == job_id)
    ).all()
    return [
        RequiredSkill(skill_id=s.id, name=s.name, importance=js.importance)
        for js, s in rows
    ]


def _children_map(db: Session, tenant_id: str, parent_ids: set[str]) -> dict[str, list[str]]:
    """parent_id → [child skill_id] 一级图谱边。"""
    if not parent_ids:
        return {}
    rows = db.execute(
        select(Skill.id, Skill.parent_id).where(
            Skill.tenant_id == tenant_id, Skill.parent_id.in_(parent_ids)
        )
    ).all()
    m: dict[str, list[str]] = {}
    for cid, pid in rows:
        m.setdefault(pid, []).append(cid)
    return m


def _ancestor_map(db: Session, tenant_id: str, skill_ids: set[str]) -> dict[str, str]:
    """skill_id → parent_id(一级祖先),供「持父级兑现子要求」方向。"""
    if not skill_ids:
        return {}
    rows = db.execute(
        select(Skill.id, Skill.parent_id).where(
            Skill.tenant_id == tenant_id,
            Skill.id.in_(skill_ids),
            Skill.parent_id.isnot(None),
        )
    ).all()
    return {sid: pid for sid, pid in rows}


def _emp_skill_index(
    db: Session, tenant_id: str, emp_ids: list[str]
) -> dict[str, dict[str, EmployeeSkill]]:
    if not emp_ids:
        return {}
    rows = db.scalars(
        select(EmployeeSkill).where(
            EmployeeSkill.tenant_id == tenant_id,
            EmployeeSkill.employee_id.in_(emp_ids),
            EmployeeSkill.status == "confirmed",
        )
    ).all()
    idx: dict[str, dict[str, EmployeeSkill]] = {}
    for es in rows:
        idx.setdefault(es.employee_id, {})[es.skill_id] = es
    return idx


def _skill_names(db: Session, ids: set[str]) -> dict[str, str]:
    if not ids:
        return {}
    return dict(
        db.execute(select(Skill.id, Skill.name).where(Skill.id.in_(ids))).all()
    )


def _score_one(
    reqs: list[RequiredSkill],
    emp_skills: dict[str, EmployeeSkill],
    children: dict[str, list[str]],
    ancestors: dict[str, str],
    names: dict[str, str],
) -> dict[str, Any] | None:
    must = [r for r in reqs if r.importance == "must"]
    nice = [r for r in reqs if r.importance != "must"]
    matched: list[MatchedSkill] = []
    missing: list[RequiredSkill] = []

    for r in reqs:
        direct = emp_skills.get(r.skill_id)
        if direct:
            matched.append(MatchedSkill(
                skill_id=r.skill_id, name=r.name, level=float(direct.level),
                via="direct", source=direct.source,
            ))
            continue
        # 图谱兑现(双向一级):员工持要求的某个子技能(更专精→兑现),或
        # 持要求的父技能(更泛→弱兑现),统一 CHILD_DECAY 衰减。
        best: tuple[float, EmployeeSkill, str] | None = None  # (eff, row, via)
        for cid in children.get(r.skill_id, []):
            cs = emp_skills.get(cid)
            if cs:
                eff = cs.level * CHILD_DECAY
                if best is None or eff > best[0]:
                    best = (eff, cs, "child")
        pid = ancestors.get(r.skill_id)
        if pid and (ps := emp_skills.get(pid)):
            eff = ps.level * CHILD_DECAY
            if best is None or eff > best[0]:
                best = (eff, ps, "parent")
        if best:
            matched.append(MatchedSkill(
                skill_id=r.skill_id, name=r.name, level=round(best[0], 2),
                via=best[2], via_name=names.get(best[1].skill_id),
                source=best[1].source,
            ))
        else:
            missing.append(r)

    if not matched:
        return None
    must_cov = (
        sum(1 for m in matched if m.skill_id in {r.skill_id for r in must})
        / len(must) if must else 1.0
    )
    nice_cov = (
        sum(1 for m in matched if m.skill_id in {r.skill_id for r in nice})
        / len(nice) if nice else 1.0
    )
    avg_level = sum(m.level for m in matched) / len(matched)
    score = round(must_cov * 70 + nice_cov * 20 + (avg_level / 5) * 10)
    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "must_hit": sum(1 for m in matched if m.skill_id in {r.skill_id for r in must}),
        "must_total": len(must),
    }


@router.get("/jobs/{job_id}")
def match_employees_for_job(
    job_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """岗位 → 内部员工排行:谁最适配这个机会(含图谱兑现)。"""
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    reqs = _requirements(db, current.tenant_id, job_id)
    req_ids = {r.skill_id for r in reqs}
    children = _children_map(db, current.tenant_id, req_ids)
    ancestors = _ancestor_map(db, current.tenant_id, req_ids)

    scope = visible_employee_ids(db, current)
    q = select(Employee).where(
        Employee.tenant_id == current.tenant_id, Employee.status == "active"
    )
    if scope is not None:
        q = q.where(Employee.id.in_(scope or ["__none__"]))
    emps = db.scalars(q.limit(500)).all()

    idx = _emp_skill_index(db, current.tenant_id, [e.id for e in emps])
    extra_ids = {c for ids in children.values() for c in ids} | set(ancestors.values())
    names = _skill_names(db, extra_ids)
    org_ids = {e.org_unit_id for e in emps if e.org_unit_id}
    org_names = dict(
        db.execute(select(OrgUnit.id, OrgUnit.name).where(OrgUnit.id.in_(org_ids))).all()
    ) if org_ids else {}

    items = []
    for e in emps:
        res = _score_one(reqs, idx.get(e.id, {}), children, ancestors, names)
        if res is None:
            continue
        items.append({
            "employee": {
                "id": e.id, "name": e.name,
                "designation": e.designation,
                "org_name": org_names.get(e.org_unit_id),
            },
            **res,
        })
    items.sort(key=lambda x: -x["score"])
    return {
        "job": {"id": job.id, "title": job.title, "status": job.status},
        "requirements": reqs,
        "items": items[:limit],
        "evaluated": len(emps),
    }


@router.get("/employees/{emp_id}")
def match_jobs_for_employee(
    emp_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """员工 → 开放岗位排行:他能适配哪些机会(北森「系统推荐」视角)。

    员工本人(linked)可看自己的;HR/管理者按行级范围看下属的。
    """
    emp = db.get(Employee, emp_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    if get_linked_employee_id(db, current) != emp.id:
        if not has_permission(current, PERM_VIEW_REPORTS):
            raise HTTPException(403, "无权查看")
        scope = visible_employee_ids(db, current)
        if scope is not None and emp.id not in scope:
            raise HTTPException(404, "员工不存在或越权")

    emp_skills = {
        es.skill_id: es
        for es in db.scalars(
            select(EmployeeSkill).where(
                EmployeeSkill.employee_id == emp.id,
                EmployeeSkill.tenant_id == current.tenant_id,
                EmployeeSkill.status == "confirmed",
            )
        )
    }

    jobs = db.scalars(
        select(Job).where(
            Job.tenant_id == current.tenant_id, Job.status == "open"
        ).limit(200)
    ).all()
    all_reqs = {j.id: _requirements(db, current.tenant_id, j.id) for j in jobs}
    req_ids = {r.skill_id for rs in all_reqs.values() for r in rs}
    children = _children_map(db, current.tenant_id, req_ids)
    ancestors = _ancestor_map(db, current.tenant_id, req_ids)
    names = _skill_names(
        db, {c for ids in children.values() for c in ids} | set(ancestors.values())
    )

    items = []
    for j in jobs:
        reqs = all_reqs[j.id]
        if not reqs:
            continue
        res = _score_one(reqs, emp_skills, children, ancestors, names)
        if res is None:
            continue
        items.append({
            "job": {"id": j.id, "title": j.title},
            **{k: res[k] for k in ("score", "matched", "missing", "must_hit", "must_total")},
        })
    items.sort(key=lambda x: -x["score"])
    return {
        "employee": {"id": emp.id, "name": emp.name, "designation": emp.designation},
        "skills_count": len(emp_skills),
        "items": items[:limit],
    }
