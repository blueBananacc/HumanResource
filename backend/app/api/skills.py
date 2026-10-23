"""技能库 API(参考 MintHCM Competencies 收敛)。

- ``GET/POST/PUT/DELETE /skills``                  技能字典(manage_settings)
- ``GET/PUT /jobs/{id}/skills``                    岗位技能要求(must/nice)
- ``GET/PUT /candidates/{id}/skills``              候选人技能(level 1-5)
- ``POST  /candidates/{id}/skills/import-resume``  从简历解析合并导入
- ``GET   /jobs/{id}/skill-coverage``              岗位要求 × 候选池覆盖度

``PUT`` 均为整组替换语义;``items`` 可传 ``skill_id`` 或 ``name``
(name 不存在则自动入字典)。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import (
    Candidate,
    CandidateSkill,
    Job,
    JobApplication,
    JobSkill,
    Resume,
    Skill,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_SETTINGS,
    PERM_VIEW_REPORTS,
    PERM_WRITE_JOBS,
    PERM_WRITE_RESUMES,
    require_permission,
)

router = APIRouter(prefix="/skills", tags=["skills"])
job_router = APIRouter(prefix="/jobs", tags=["skills"])
cand_router = APIRouter(prefix="/candidates", tags=["skills"])

CATEGORIES = Skill.CATEGORIES
IMPORTANCES = ("must", "nice")


# ============ schemas ============


class SkillIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    category: str = "other"
    description: str | None = Field(None, max_length=500)


class SkillOut(BaseModel):
    id: str
    name: str
    category: str
    description: str | None


class JobSkillItem(BaseModel):
    skill_id: str | None = None
    name: str | None = None
    importance: str = "must"


class CandidateSkillItem(BaseModel):
    skill_id: str | None = None
    name: str | None = None
    level: int = Field(3, ge=1, le=5)


class JobSkillsIn(BaseModel):
    items: list[JobSkillItem] = Field(default_factory=list, max_length=50)


class CandidateSkillsIn(BaseModel):
    items: list[CandidateSkillItem] = Field(default_factory=list, max_length=50)


class JobSkillOut(BaseModel):
    skill_id: str
    name: str
    category: str
    importance: str


class CandidateSkillOut(BaseModel):
    skill_id: str
    name: str
    category: str
    level: int
    source: str


class CoverageItem(BaseModel):
    skill_id: str
    name: str
    importance: str
    pool_count: int
    pipeline_count: int


# ============ helpers ============


def _skill_or_404(db: Session, tenant_id: str, skill_id: str) -> Skill:
    s = db.get(Skill, skill_id)
    if not s or s.tenant_id != tenant_id:
        raise HTTPException(404, "技能不存在")
    return s


def _resolve_skill(db: Session, tenant_id: str, item: dict) -> Skill:
    """按 skill_id 或 name 解析;name 不存在则自动入字典。"""
    sid = item.get("skill_id")
    if sid:
        return _skill_or_404(db, tenant_id, sid)
    name = (item.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "需要 skill_id 或 name")
    s = db.scalars(
        select(Skill).where(Skill.tenant_id == tenant_id, Skill.name == name)
    ).first()
    if s is None:
        s = Skill(tenant_id=tenant_id, name=name, category="other")
        db.add(s)
        db.flush()
    return s


def _job_or_404(db: Session, tenant_id: str, job_id: str) -> Job:
    j = db.get(Job, job_id)
    if not j or j.tenant_id != tenant_id:
        raise HTTPException(404, "岗位不存在")
    return j


def _cand_or_404(db: Session, tenant_id: str, cid: str) -> Candidate:
    c = db.get(Candidate, cid)
    if not c or c.tenant_id != tenant_id:
        raise HTTPException(404, "候选人不存在")
    return c


def _job_skills_out(db: Session, tenant_id: str, job_id: str) -> list[JobSkillOut]:
    rows = db.execute(
        select(JobSkill, Skill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.tenant_id == tenant_id, JobSkill.job_id == job_id)
        .order_by(Skill.name)
    ).all()
    return [
        JobSkillOut(
            skill_id=s.id, name=s.name, category=s.category, importance=js.importance
        )
        for js, s in rows
    ]


def _cand_skills_out(db: Session, tenant_id: str, cid: str) -> list[CandidateSkillOut]:
    rows = db.execute(
        select(CandidateSkill, Skill)
        .join(Skill, Skill.id == CandidateSkill.skill_id)
        .where(CandidateSkill.tenant_id == tenant_id, CandidateSkill.candidate_id == cid)
        .order_by(Skill.name)
    ).all()
    return [
        CandidateSkillOut(
            skill_id=s.id, name=s.name, category=s.category,
            level=cs.level, source=cs.source,
        )
        for cs, s in rows
    ]


# ============ 技能字典 ============


@router.get("", response_model=list[SkillOut])
def list_skills(
    q: str | None = Query(None, max_length=64),
    category: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    stmt = select(Skill).where(Skill.tenant_id == current.tenant_id)
    if q and q.strip():
        stmt = stmt.where(Skill.name.ilike(f"%{q.strip()}%"))
    if category:
        stmt = stmt.where(Skill.category == category)
    return [
        SkillOut(id=s.id, name=s.name, category=s.category, description=s.description)
        for s in db.scalars(stmt.order_by(Skill.name).limit(limit))
    ]


@router.post("", response_model=SkillOut, status_code=201)
def create_skill(
    body: SkillIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    if body.category not in CATEGORIES:
        raise HTTPException(400, f"category 必须是 {CATEGORIES} 之一")
    name = body.name.strip()
    exists = db.scalars(
        select(Skill).where(Skill.tenant_id == current.tenant_id, Skill.name == name)
    ).first()
    if exists:
        raise HTTPException(409, "同名技能已存在")
    s = Skill(
        tenant_id=current.tenant_id, name=name,
        category=body.category, description=body.description,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    write_audit(db, actor=current, entity_type="skill", entity_id=s.id,
                action="create", detail={"name": name}, request=request)
    db.commit()
    return SkillOut(id=s.id, name=s.name, category=s.category, description=s.description)


@router.put("/{skill_id}", response_model=SkillOut)
def update_skill(
    skill_id: str,
    body: SkillIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    s = _skill_or_404(db, current.tenant_id, skill_id)
    if body.category not in CATEGORIES:
        raise HTTPException(400, f"category 必须是 {CATEGORIES} 之一")
    s.name = body.name.strip()
    s.category = body.category
    s.description = body.description
    write_audit(db, actor=current, entity_type="skill", entity_id=s.id,
                action="update", detail={"name": s.name}, request=request)
    db.commit()
    return SkillOut(id=s.id, name=s.name, category=s.category, description=s.description)


@router.delete("/{skill_id}", status_code=204)
def delete_skill(
    skill_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    s = _skill_or_404(db, current.tenant_id, skill_id)
    db.delete(s)
    write_audit(db, actor=current, entity_type="skill", entity_id=skill_id,
                action="delete", detail={"name": s.name}, request=request)
    db.commit()


# ============ 岗位技能 ============


@job_router.get("/{job_id}/skills", response_model=list[JobSkillOut])
def get_job_skills(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    _job_or_404(db, current.tenant_id, job_id)
    return _job_skills_out(db, current.tenant_id, job_id)


@job_router.put("/{job_id}/skills", response_model=list[JobSkillOut])
def put_job_skills(
    job_id: str,
    body: JobSkillsIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_JOBS)),
):
    job = _job_or_404(db, current.tenant_id, job_id)
    seen: set[str] = set()
    new_rows: list[JobSkill] = []
    for item in body.items:
        if item.importance not in IMPORTANCES:
            raise HTTPException(400, "importance 必须是 must / nice")
        s = _resolve_skill(db, current.tenant_id, item.model_dump())
        if s.id in seen:
            continue
        seen.add(s.id)
        new_rows.append(
            JobSkill(
                tenant_id=current.tenant_id, job_id=job.id,
                skill_id=s.id, importance=item.importance,
            )
        )
    db.query(JobSkill).filter(
        JobSkill.tenant_id == current.tenant_id, JobSkill.job_id == job.id
    ).delete()
    db.add_all(new_rows)
    write_audit(db, actor=current, entity_type="job", entity_id=job.id,
                action="skills_update", detail={"count": len(new_rows)}, request=request)
    db.commit()
    return _job_skills_out(db, current.tenant_id, job_id)


@job_router.get("/{job_id}/skill-coverage", response_model=list[CoverageItem])
def skill_coverage(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    job = _job_or_404(db, current.tenant_id, job_id)
    reqs = _job_skills_out(db, current.tenant_id, job.id)
    if not reqs:
        return []
    req_ids = [r.skill_id for r in reqs]

    # 全租户具备该技能的候选人数
    pool_rows = db.execute(
        select(CandidateSkill.skill_id, func.count(func.distinct(CandidateSkill.candidate_id)))
        .where(
            CandidateSkill.tenant_id == current.tenant_id,
            CandidateSkill.skill_id.in_(req_ids),
        )
        .group_by(CandidateSkill.skill_id)
    ).all()
    pool = {sid: int(n) for sid, n in pool_rows}

    # 该岗位管道中具备该技能的候选人数
    pipe_rows = db.execute(
        select(CandidateSkill.skill_id, func.count(func.distinct(CandidateSkill.candidate_id)))
        .join(
            JobApplication,
            (JobApplication.candidate_id == CandidateSkill.candidate_id)
            & (JobApplication.tenant_id == current.tenant_id)
            & (JobApplication.job_id == job.id),
        )
        .where(
            CandidateSkill.tenant_id == current.tenant_id,
            CandidateSkill.skill_id.in_(req_ids),
        )
        .group_by(CandidateSkill.skill_id)
    ).all()
    pipe = {sid: int(n) for sid, n in pipe_rows}

    return [
        CoverageItem(
            skill_id=r.skill_id, name=r.name, importance=r.importance,
            pool_count=pool.get(r.skill_id, 0),
            pipeline_count=pipe.get(r.skill_id, 0),
        )
        for r in reqs
    ]


# ============ 候选人技能 ============


@cand_router.get("/{candidate_id}/skills", response_model=list[CandidateSkillOut])
def get_candidate_skills(
    candidate_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    _cand_or_404(db, current.tenant_id, candidate_id)
    return _cand_skills_out(db, current.tenant_id, candidate_id)


@cand_router.put("/{candidate_id}/skills", response_model=list[CandidateSkillOut])
def put_candidate_skills(
    candidate_id: str,
    body: CandidateSkillsIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_RESUMES)),
):
    cand = _cand_or_404(db, current.tenant_id, candidate_id)
    seen: set[str] = set()
    new_rows: list[CandidateSkill] = []
    for item in body.items:
        s = _resolve_skill(db, current.tenant_id, item.model_dump())
        if s.id in seen:
            continue
        seen.add(s.id)
        new_rows.append(
            CandidateSkill(
                tenant_id=current.tenant_id, candidate_id=cand.id,
                skill_id=s.id, level=item.level, source="manual",
            )
        )
    db.query(CandidateSkill).filter(
        CandidateSkill.tenant_id == current.tenant_id,
        CandidateSkill.candidate_id == cand.id,
    ).delete()
    db.add_all(new_rows)
    write_audit(db, actor=current, entity_type="candidate", entity_id=cand.id,
                action="skills_update", detail={"count": len(new_rows)}, request=request)
    db.commit()
    return _cand_skills_out(db, current.tenant_id, candidate_id)


@cand_router.post("/{candidate_id}/skills/import-resume")
def import_resume_skills(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_RESUMES)),
):
    """把候选人最新解析完成简历的 ``parsed_data.skills`` 合并进技能档案。"""
    cand = _cand_or_404(db, current.tenant_id, candidate_id)
    resume = db.scalars(
        select(Resume)
        .where(
            Resume.candidate_id == cand.id,
            Resume.parse_status == "done",
        )
        .order_by(Resume.created_at.desc())
        .limit(1)
    ).first()
    if not resume:
        raise HTTPException(404, "没有已解析的简历")
    names = [
        str(s).strip()
        for s in (resume.parsed_data or {}).get("skills") or []
        if str(s).strip()
    ]
    if not names:
        return {"imported": 0, "total": 0}

    existing = {
        row.skill_id
        for row in db.scalars(
            select(CandidateSkill).where(
                CandidateSkill.tenant_id == current.tenant_id,
                CandidateSkill.candidate_id == cand.id,
            )
        )
    }
    imported = 0
    for name in names:
        s = _resolve_skill(db, current.tenant_id, {"name": name})
        if s.id in existing:
            continue
        existing.add(s.id)
        db.add(
            CandidateSkill(
                tenant_id=current.tenant_id, candidate_id=cand.id,
                skill_id=s.id, level=3, source="resume",
            )
        )
        imported += 1
    write_audit(db, actor=current, entity_type="candidate", entity_id=cand.id,
                action="skills_import", detail={"imported": imported}, request=request)
    db.commit()
    return {"imported": imported, "total": len(names)}
