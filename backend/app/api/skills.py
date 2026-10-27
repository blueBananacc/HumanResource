"""技能库 API。

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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.dictionaries import kind_values
from app.domain.models import (
    Candidate,
    CandidateSkill,
    EmployeeSkill,
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
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    has_permission,
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
    parent_id: str | None = None
    aliases: list[str] | None = Field(None, max_length=20)


class SkillOut(BaseModel):
    id: str
    name: str
    category: str
    description: str | None
    parent_id: str | None = None
    aliases: list[str] = []


class JobSkillItem(BaseModel):
    skill_id: str | None = None
    name: str | None = Field(None, max_length=64)
    importance: str = "must"


class CandidateSkillItem(BaseModel):
    skill_id: str | None = None
    name: str | None = Field(None, max_length=64)
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


def _locked_skill_or_404(db: Session, tenant_id: str, skill_id: str) -> Skill:
    """FOR UPDATE 锁技能行 — delete 与引用方(_resolve_skill)都走这里,
    保证「数引用 → 删除」与「建引用」互斥,否则新引用会被 CASCADE 静默抹掉。"""
    s = db.get(Skill, skill_id, with_for_update=True)
    if not s or s.tenant_id != tenant_id:
        raise HTTPException(404, "技能不存在")
    return s


def _resolve_skill(
    db: Session, tenant_id: str, item: dict, *, allow_create: bool
) -> Skill:
    """按 skill_id 或 name 解析。

    name 不存在时 ``allow_create=True`` 才自动入字典 — 字典本身归
    ``manage_settings`` 管,岗位/候选人技能写权限不应绕过它随意造词条;
    ``import-resume`` 例外(简历抽取本来就要扩充字典)。

    命中既有词条时锁该行 — 与 delete_skill 的行锁互斥,防「对方删了
    词条但刚给它建的引用被 CASCADE 带走」。
    """
    sid = item.get("skill_id")
    if sid:
        return _locked_skill_or_404(db, tenant_id, sid)
    name = (item.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "需要 skill_id 或 name")
    s = db.scalars(
        select(Skill)
        .where(Skill.tenant_id == tenant_id, Skill.name == name)
        .with_for_update()
    ).first()
    if s is None:
        # 别名归并:「Golang」提交时命中字典里 aliases 含 "Golang" 的「Go」,
        # 按 canonical 词条归一 —— 图谱的同义词边(Workday related-skills 轻量版)。
        low = name.lower()
        s = next(
            (
                row for row in db.scalars(
                    select(Skill).where(Skill.tenant_id == tenant_id).with_for_update()
                )
                if any(low == a.strip().lower() for a in (row.aliases or []))
            ),
            None,
        )
    if s is None:
        if not allow_create:
            raise HTTPException(
                400, f"技能「{name[:32]}」不在字典中,需管理员先创建"
            )
        s = Skill(tenant_id=tenant_id, name=name, category="other")
        db.add(s)
        db.flush()
    return s


def _skill_out(s: Skill) -> SkillOut:
    return SkillOut(
        id=s.id, name=s.name, category=s.category, description=s.description,
        parent_id=s.parent_id, aliases=list(s.aliases or []),
    )


def _apply_graph_fields(
    db: Session, tenant_id: str, s: Skill, body: SkillIn
) -> None:
    """parent_id 自引用校验(同租户、非自身、无环)+ aliases 清洗。"""
    if body.parent_id:
        if body.parent_id == s.id:
            raise HTTPException(400, "父技能不能是自身")
        p = db.get(Skill, body.parent_id)
        if not p or p.tenant_id != tenant_id:
            raise HTTPException(400, "父技能不存在")
        # 环检测:沿 parent 链向上走,撞到自己即成环
        seen, cur = {s.id}, p
        while cur is not None and cur.parent_id:
            if cur.parent_id in seen:
                raise HTTPException(400, "父技能链存在循环")
            seen.add(cur.parent_id)
            cur = db.get(Skill, cur.parent_id)
        s.parent_id = body.parent_id
    else:
        s.parent_id = None
    if body.aliases is not None:
        cleaned = []
        for a in body.aliases:
            a = a.strip()[:64]
            if a and a.lower() != body.name.strip().lower() and a not in cleaned:
                cleaned.append(a)
        s.aliases = cleaned or None


def _job_or_404(
    db: Session, tenant_id: str, job_id: str, *, lock: bool = False
) -> Job:
    j = db.get(Job, job_id, with_for_update=lock)
    if not j or j.tenant_id != tenant_id:
        raise HTTPException(404, "岗位不存在")
    return j


def _cand_or_404(
    db: Session, tenant_id: str, cid: str, *, lock: bool = False
) -> Candidate:
    c = db.get(Candidate, cid, with_for_update=lock)
    if not c or c.tenant_id != tenant_id:
        raise HTTPException(404, "候选人不存在")
    return c


def _job_visible(db: Session, current: User, job: Job) -> Job:
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    return job


def _cand_visible(db: Session, current: User, cand: Candidate) -> Candidate:
    ensure_can_see(get_org_scope(db, current), cand.org_unit_id)
    return cand


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
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    stmt = select(Skill).where(Skill.tenant_id == current.tenant_id)
    if q and q.strip():
        stmt = stmt.where(Skill.name.ilike(f"%{q.strip()}%"))
    if category:
        stmt = stmt.where(Skill.category == category)
    return [_skill_out(s) for s in db.scalars(stmt.order_by(Skill.name).limit(limit))]


@router.post("", response_model=SkillOut, status_code=201)
def create_skill(
    body: SkillIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    allowed = kind_values(db, current.tenant_id, "skill_category", CATEGORIES)
    if body.category not in allowed:
        raise HTTPException(400, f"category 必须是 {sorted(allowed)} 之一")
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
    db.flush()
    _apply_graph_fields(db, current.tenant_id, s, body)
    try:
        db.commit()
    except IntegrityError:
        # exists-check 与插入之间可能有并发写入 — 唯一约束兜底成 409
        db.rollback()
        raise HTTPException(409, "同名技能已存在")
    db.refresh(s)
    write_audit(db, actor=current, entity_type="skill", entity_id=s.id,
                action="create", detail={"name": name}, request=request)
    db.commit()
    return _skill_out(s)


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
    allowed = kind_values(db, current.tenant_id, "skill_category", CATEGORIES)
    if body.category not in allowed:
        raise HTTPException(400, f"category 必须是 {sorted(allowed)} 之一")
    name = body.name.strip()
    dup = db.scalars(
        select(Skill).where(
            Skill.tenant_id == current.tenant_id,
            Skill.name == name,
            Skill.id != s.id,
        )
    ).first()
    if dup:
        raise HTTPException(409, "同名技能已存在")
    s.name = name
    s.category = body.category
    s.description = body.description
    _apply_graph_fields(db, current.tenant_id, s, body)
    write_audit(db, actor=current, entity_type="skill", entity_id=s.id,
                action="update", detail={"name": s.name}, request=request)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "同名技能已存在")
    return _skill_out(s)


@router.delete("/{skill_id}", status_code=204)
def delete_skill(
    skill_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    # 先拿技能行锁再数引用 — 与 _resolve_skill 的锁互斥,防计数后插入
    # 的新引用被 CASCADE 静默删除
    s = _locked_skill_or_404(db, current.tenant_id, skill_id)
    # 被岗位/候选人引用时不允许硬删 — CASCADE 会连坐抹掉技能档案
    refs = (
        db.scalar(select(func.count()).select_from(JobSkill).where(JobSkill.skill_id == s.id)) or 0
    ) + (
        db.scalar(select(func.count()).select_from(CandidateSkill).where(CandidateSkill.skill_id == s.id)) or 0
    )
    if refs:
        raise HTTPException(409, f"该技能被 {refs} 处岗位/候选人引用,先移除引用")
    db.delete(s)
    write_audit(db, actor=current, entity_type="skill", entity_id=skill_id,
                action="delete", detail={"name": s.name}, request=request)
    db.commit()


class GraphNode(BaseModel):
    id: str
    name: str
    category: str
    parent_id: str | None
    aliases: list[str]
    job_refs: int
    candidate_refs: int
    employee_refs: int


class SkillGraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[dict[str, str]]  # {from: child_id, to: parent_id}


@router.get("/graph", response_model=SkillGraphOut)
def skill_graph(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """技能图谱视图:节点=技能词条(带引用热度),边=parent 层级 + 别名归并。

    Workday Skills Cloud 的轻量映射——不做全量本体,只暴露
    租户自己维护的关系,供管理端图谱可视化与匹配扩展用。
    """
    rows = db.scalars(
        select(Skill).where(Skill.tenant_id == current.tenant_id).order_by(Skill.name)
    ).all()
    cnt = lambda model, col: dict(
        db.execute(select(col, func.count()).group_by(col)).all()
    )
    job_c = cnt(JobSkill, JobSkill.skill_id)
    cand_c = cnt(CandidateSkill, CandidateSkill.skill_id)
    emp_c = cnt(EmployeeSkill, EmployeeSkill.skill_id)
    nodes, edges = [], []
    for s in rows:
        nodes.append(GraphNode(
            id=s.id, name=s.name, category=s.category, parent_id=s.parent_id,
            aliases=list(s.aliases or []),
            job_refs=job_c.get(s.id, 0), candidate_refs=cand_c.get(s.id, 0),
            employee_refs=emp_c.get(s.id, 0),
        ))
        if s.parent_id:
            edges.append({"from": s.id, "to": s.parent_id})
    return SkillGraphOut(nodes=nodes, edges=edges)


# ============ 岗位技能 ============


@job_router.get("/{job_id}/skills", response_model=list[JobSkillOut])
def get_job_skills(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    job = _job_or_404(db, current.tenant_id, job_id)
    _job_visible(db, current, job)
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
    # 锁岗位行:整组替换是「删+插」,并发 PUT 否则可能交叠出
    # 唯一键冲突(500)或合并出非预期结果
    job = _job_or_404(db, current.tenant_id, job_id, lock=True)
    _job_visible(db, current, job)
    allow_create = has_permission(current, PERM_MANAGE_SETTINGS)
    seen: set[str] = set()
    new_rows: list[JobSkill] = []
    # 按解析键排序后取锁,所有并发 PUT 走同一锁序,防环状死锁
    for item in sorted(body.items, key=lambda i: (i.skill_id or i.name or "")):
        if item.importance not in IMPORTANCES:
            raise HTTPException(400, "importance 必须是 must / nice")
        s = _resolve_skill(db, current.tenant_id, item.model_dump(), allow_create=allow_create)
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
    _job_visible(db, current, job)
    reqs = _job_skills_out(db, current.tenant_id, job.id)
    if not reqs:
        return []
    req_ids = [r.skill_id for r in reqs]
    scope = get_org_scope(db, current)

    # 具备该技能的候选人数(按当前用户数据范围过滤)
    pool_stmt = (
        select(CandidateSkill.skill_id, func.count(func.distinct(CandidateSkill.candidate_id)))
        .join(Candidate, Candidate.id == CandidateSkill.candidate_id)
        .where(
            CandidateSkill.tenant_id == current.tenant_id,
            CandidateSkill.skill_id.in_(req_ids),
        )
        .group_by(CandidateSkill.skill_id)
    )
    pool_stmt = apply_org_filter(
        pool_stmt, org_column=Candidate.org_unit_id, scope=scope
    )
    pool_rows = db.execute(pool_stmt).all()
    pool = {sid: int(n) for sid, n in pool_rows}

    # 该岗位管道中具备该技能的候选人数(同样按数据范围过滤)
    pipe_stmt = (
        select(CandidateSkill.skill_id, func.count(func.distinct(CandidateSkill.candidate_id)))
        .join(Candidate, Candidate.id == CandidateSkill.candidate_id)
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
    )
    pipe_stmt = apply_org_filter(
        pipe_stmt, org_column=Candidate.org_unit_id, scope=scope
    )
    pipe_rows = db.execute(pipe_stmt).all()
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
    cand = _cand_or_404(db, current.tenant_id, candidate_id)
    _cand_visible(db, current, cand)
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
    cand = _cand_or_404(db, current.tenant_id, candidate_id, lock=True)
    _cand_visible(db, current, cand)
    allow_create = has_permission(current, PERM_MANAGE_SETTINGS)
    seen: set[str] = set()
    new_rows: list[CandidateSkill] = []
    for item in sorted(body.items, key=lambda i: (i.skill_id or i.name or "")):
        s = _resolve_skill(db, current.tenant_id, item.model_dump(), allow_create=allow_create)
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
    # 锁候选人行:与 put_candidate_skills 互斥,防合并/替换交叠出唯一键冲突
    cand = _cand_or_404(db, current.tenant_id, candidate_id, lock=True)
    _cand_visible(db, current, cand)
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
    # 技能名上限 64(字典列宽),超长词不入库
    names = [
        n
        for s in (resume.parsed_data or {}).get("skills") or []
        if (n := str(s).strip()) and len(n) <= 64
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
    for name in sorted(names):
        s = _resolve_skill(db, current.tenant_id, {"name": name}, allow_create=True)
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
