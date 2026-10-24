"""记录级 AI 摘要 — 上下文聚合(T6-2)。

``build_summary_context(db, u, scope, entity, rid)`` 按实体收集**限界**上下文
dict 供 summarize 提示词渲染:列表 ≤10 项、文本截断,避免把整库塞进 prompt。

权限/范围:实体 perm 由调用方(api/ai.py SUMMARIZABLE)校验;本模块只做
租户隔离 + org scope(`get_org_scope`/`ensure_can_see`/`visible_employee_ids`,
与 entities.py 同口径)。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    CandidateNote,
    CandidateSkill,
    Employee,
    EmployeeRequest,
    Goal,
    HrCase,
    HrCaseUpdate,
    Interview,
    Job,
    JobApplication,
    LeaveApplication,
    OrgUnit,
    Resume,
    ResumeJobMatch,
    Skill,
    User,
)
from app.services.permissions import (
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    visible_employee_ids,
)


def _s(v: Any, n: int = 200) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s[:n] + ("…" if len(s) > n else "")


def _d(v: Any) -> str | None:
    return v.strftime("%Y-%m-%d") if v is not None else None


def get_scoped_row(db: Session, u: User, scope, entity: str, rid: str):
    """租户 + scope 可见性校验,越界一律 404(防探测)。"""
    from fastapi import HTTPException

    if entity == "candidates":
        row = db.get(Candidate, rid)
        if row is None or row.tenant_id != u.tenant_id:
            raise HTTPException(404, "记录不存在")
        ensure_can_see(scope, row.org_unit_id)
        return row
    if entity == "employees":
        row = db.get(Employee, rid)
        if row is None or row.tenant_id != u.tenant_id:
            raise HTTPException(404, "记录不存在")
        ensure_can_see(scope, row.org_unit_id)
        return row
    if entity == "cases":
        row = db.get(HrCase, rid)
        if row is None or row.tenant_id != u.tenant_id:
            raise HTTPException(404, "记录不存在")
        emp_ids = visible_employee_ids(db, u)
        if emp_ids is not None and (row.requester_employee_id or "__none__") not in emp_ids:
            raise HTTPException(404, "记录不存在")
        return row
    raise HTTPException(400, f"不支持摘要的实体: {entity}")


def _candidate_ctx(db: Session, u: User, scope, c: Candidate) -> dict[str, Any]:
    skills = db.execute(
        select(Skill.name, CandidateSkill.level)
        .join(CandidateSkill, CandidateSkill.skill_id == Skill.id)
        .where(CandidateSkill.tenant_id == u.tenant_id, CandidateSkill.candidate_id == c.id)
        .limit(20)
    ).all()
    apps_stmt = (
        select(Job.title, JobApplication.stage)
        .join(Job, Job.id == JobApplication.job_id)
        .where(JobApplication.tenant_id == u.tenant_id, JobApplication.candidate_id == c.id)
    )
    apps = db.execute(apply_org_filter(apps_stmt, org_column=Job.org_unit_id, scope=scope).limit(10)).all()
    ivs_stmt = (
        select(Job.title, Interview.status, Interview.summary)
        .join(Job, Job.id == Interview.job_id)
        .where(Interview.tenant_id == u.tenant_id, Interview.candidate_id == c.id)
    )
    ivs = db.execute(
        apply_org_filter(ivs_stmt, org_column=Job.org_unit_id, scope=scope)
        .order_by(Interview.started_at.desc()).limit(10)
    ).all()
    matches_stmt = (
        select(Job.title, ResumeJobMatch.score, ResumeJobMatch.comment)
        .join(Resume, Resume.id == ResumeJobMatch.resume_id)
        .join(Job, Job.id == ResumeJobMatch.job_id)
        .join(Candidate, Candidate.id == Resume.candidate_id)
        .where(
            ResumeJobMatch.tenant_id == u.tenant_id,
            Resume.candidate_id == c.id,
            ResumeJobMatch.score.is_not(None),
        )
    )
    matches = db.execute(
        apply_org_filter(matches_stmt, org_column=Job.org_unit_id, scope=scope)
        .order_by(ResumeJobMatch.score.desc()).limit(5)
    ).all()
    notes = db.scalars(
        select(CandidateNote)
        .where(CandidateNote.tenant_id == u.tenant_id, CandidateNote.candidate_id == c.id)
        .order_by(CandidateNote.created_at.desc()).limit(5)
    ).all()
    resumes_n = db.scalar(
        select(func.count(Resume.id)).where(
            Resume.tenant_id == u.tenant_id, Resume.candidate_id == c.id
        )
    ) or 0
    return {
        "name": c.name,
        "status": c.status,
        "tags": c.tags or [],
        "is_blacklisted": c.is_blacklisted,
        "skills": [{"name": n, "level": lv} for n, lv in skills],
        "applications": [{"job": t, "stage": s} for t, s in apps],
        "interviews": [
            {
                "job": t,
                "status": s,
                "recommendation": (sm or {}).get("recommendation"),
            }
            for t, s, sm in ivs
        ],
        "top_matches": [{"job": t, "score": sc, "comment": _s(cm, 120)} for t, sc, cm in matches],
        "resume_count": resumes_n,
        "recent_notes": [
            {"at": _d(n.created_at), "content": _s(n.content, 200)} for n in notes
        ],
    }


def _employee_ctx(db: Session, u: User, scope, e: Employee) -> dict[str, Any]:
    org = db.get(OrgUnit, e.org_unit_id) if e.org_unit_id else None
    mgr = db.get(Employee, e.reports_to_id) if e.reports_to_id else None
    reqs = db.scalars(
        select(EmployeeRequest)
        .where(EmployeeRequest.tenant_id == u.tenant_id, EmployeeRequest.employee_id == e.id)
        .order_by(EmployeeRequest.created_at.desc()).limit(10)
    ).all()
    cases = db.scalars(
        select(HrCase)
        .where(HrCase.tenant_id == u.tenant_id, HrCase.requester_employee_id == e.id)
        .order_by(HrCase.created_at.desc()).limit(10)
    ).all()
    leaves = db.scalars(
        select(LeaveApplication)
        .where(LeaveApplication.tenant_id == u.tenant_id, LeaveApplication.employee_id == e.id)
        .order_by(LeaveApplication.created_at.desc()).limit(5)
    ).all()
    goals = db.scalars(
        select(Goal)
        .where(Goal.tenant_id == u.tenant_id, Goal.employee_id == e.id)
        .order_by(Goal.created_at.desc()).limit(10)
    ).all()
    return {
        "name": e.name,
        "employee_no": e.employee_no,
        "designation": e.designation,
        "org": org.name if org else None,
        "manager": mgr.name if mgr else None,
        "status": e.status,
        "employment_type": e.employment_type,
        "hire_date": _d(e.hire_date),
        "requests": [{"subject": r.subject, "kind": r.kind, "status": r.status} for r in reqs],
        "cases": [{"subject": c.subject, "status": c.status} for c in cases],
        "leave": [
            {"status": l.status, "days": l.days, "start": _d(l.start_date), "end": _d(l.end_date)}
            for l in leaves
        ],
        "goals": [{"title": g.title, "status": g.status, "progress": g.progress} for g in goals],
    }


def _case_ctx(db: Session, u: User, scope, c: HrCase) -> dict[str, Any]:
    requester = db.get(Employee, c.requester_employee_id) if c.requester_employee_id else None
    assignee = db.get(User, c.assignee_user_id) if c.assignee_user_id else None
    updates = db.scalars(
        select(HrCaseUpdate)
        .where(HrCaseUpdate.tenant_id == u.tenant_id, HrCaseUpdate.case_id == c.id)
        .order_by(HrCaseUpdate.created_at.desc()).limit(10)
    ).all()
    return {
        "case_no": c.case_no,
        "subject": c.subject,
        "kind": c.kind,
        "priority": c.priority,
        "status": c.status,
        "description": _s(c.description, 800),
        "resolution": _s(c.resolution, 500),
        "requester": requester.name if requester else None,
        "assignee": assignee.email if assignee else None,
        "created_at": _d(c.created_at),
        "updates": [
            {"at": _d(x.created_at), "internal": x.is_internal, "body": _s(x.body, 300)}
            for x in updates
        ],
    }


_BUILDERS = {
    "candidates": _candidate_ctx,
    "employees": _employee_ctx,
    "cases": _case_ctx,
}


def build_summary_context(
    db: Session, u: User, entity: str, rid: str
) -> dict[str, Any]:
    """入口:scope 解析 + 行可见性 + 限界上下文。"""
    scope = get_org_scope(db, u)
    row = get_scoped_row(db, u, scope, entity, rid)
    builder = _BUILDERS.get(entity)
    if builder is None:
        from fastapi import HTTPException

        raise HTTPException(400, f"不支持摘要的实体: {entity}")
    return builder(db, u, scope, row)
