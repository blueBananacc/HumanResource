"""申请管道 API:候选人 × 岗位的阶段卡 + 看板。

- ``GET    /applications``                列表(job_id / stage / q 候选人名)
- ``POST   /applications``                建卡(同候选人同岗位唯一,409)
- ``GET    /applications/{id}``           详情 + 变迁历史
- ``POST   /applications/{id}/transition``换阶段(记 event + audit)
- ``DELETE /applications/{id}``           删卡
- ``GET    /jobs/{job_id}/pipeline``      看板:按 stage 分列的卡片
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Candidate,
    Job,
    JobApplication,
    JobApplicationEvent,
    Resume,
    ResumeJobMatch,
    User,
)
from app.infra.db import get_db
from app.services.applications import STAGE_ORDER, record_event
from app.services.audit import write_audit
from app.services.decision_learning import record_decision
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    PERM_WRITE_JOBS,
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(
    prefix="/applications",
    tags=["hris-applications"],
    dependencies=[Depends(require_feature("hr.suite"))],
)
pipeline_router = APIRouter(
    prefix="/jobs",
    tags=["hris-applications"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

ALL_STAGES = list(JobApplication.STAGES)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class ApplicationIn(BaseModel):
    candidate_id: str
    job_id: str
    stage: str = "applied"
    note: str | None = Field(None, max_length=500)


class TransitionIn(BaseModel):
    to_stage: str
    note: str | None = Field(None, max_length=500)
    rejected_reason: str | None = Field(None, max_length=500)


class ApplicationOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    stage: str
    source: str
    owner_user_id: str | None
    owner_email: str | None
    note: str | None
    rejected_reason: str | None
    match_score: int | None
    stage_changed_at: datetime
    created_at: datetime


class EventOut(BaseModel):
    id: str
    from_stage: str | None
    to_stage: str
    actor_email: str | None
    note: str | None
    created_at: datetime


class ApplicationDetailOut(ApplicationOut):
    events: list[EventOut]


class PipelineCard(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    stage: str
    match_score: int | None
    owner_email: str | None
    days_in_stage: int


class PipelineColumn(BaseModel):
    stage: str
    count: int
    cards: list[PipelineCard]


class PipelineOut(BaseModel):
    job_id: str
    job_title: str
    columns: list[PipelineColumn]
    total: int


# ============ helpers ============


def _match_scores(db: Session, tenant_id: str, job_id: str, cand_ids: list[str]) -> dict[str, int]:
    """每个候选人在该岗位下的最高 AI 匹配分(取其所有简历的 done 匹配)。"""
    if not cand_ids:
        return {}
    rows = db.execute(
        select(Resume.candidate_id, func.max(ResumeJobMatch.score))
        .join(ResumeJobMatch, ResumeJobMatch.resume_id == Resume.id)
        .where(
            ResumeJobMatch.tenant_id == tenant_id,
            ResumeJobMatch.job_id == job_id,
            ResumeJobMatch.status == "done",
            Resume.candidate_id.in_(cand_ids),
        )
        .group_by(Resume.candidate_id)
    ).all()
    return {cid: int(score) for cid, score in rows if score is not None}


def _to_out(
    app: JobApplication,
    cands: dict[str, str],
    jobs: dict[str, str],
    users: dict[str, str],
    scores: dict[str, int],
) -> ApplicationOut:
    return ApplicationOut(
        id=app.id,
        candidate_id=app.candidate_id,
        candidate_name=cands.get(app.candidate_id, "?"),
        job_id=app.job_id,
        job_title=jobs.get(app.job_id, "?"),
        stage=app.stage,
        source=app.source,
        owner_user_id=app.owner_user_id,
        owner_email=users.get(app.owner_user_id) if app.owner_user_id else None,
        note=app.note,
        rejected_reason=app.rejected_reason,
        match_score=scores.get(app.candidate_id),
        stage_changed_at=app.stage_changed_at,
        created_at=app.created_at,
    )


def _maps(db: Session, tenant_id: str, apps: list[JobApplication]):
    cand_ids = {a.candidate_id for a in apps}
    job_ids = {a.job_id for a in apps}
    owner_ids = {a.owner_user_id for a in apps if a.owner_user_id}
    cands = {}
    if cand_ids:
        cands = {
            c.id: c.name
            for c in db.scalars(select(Candidate).where(Candidate.id.in_(cand_ids)))
        }
    jobs = {}
    if job_ids:
        jobs = {
            j.id: j.title for j in db.scalars(select(Job).where(Job.id.in_(job_ids)))
        }
    users = {}
    if owner_ids:
        users = {
            u.id: u.email for u in db.scalars(select(User).where(User.id.in_(owner_ids)))
        }
    return cands, jobs, users


def _get_app(db: Session, app_id: str, tenant_id: str) -> JobApplication:
    app = db.get(JobApplication, app_id)
    if not app or app.tenant_id != tenant_id:
        raise HTTPException(404, "申请不存在")
    return app


def _check_scope(db: Session, current: User, job: Job, cand: Candidate) -> None:
    scope = get_org_scope(db, current)
    ensure_can_see(scope, job.org_unit_id)
    ensure_can_see(scope, cand.org_unit_id)


def _get_app_in_scope(
    db: Session, app_id: str, current: User
) -> JobApplication:
    """单卡读/写共用的可见性守卫:租户命中后还要求岗位与候选人
    都在当前用户数据范围内,否则按 404 处理(不暴露存在性)。"""
    app = _get_app(db, app_id, current.tenant_id)
    job = db.get(Job, app.job_id)
    cand = db.get(Candidate, app.candidate_id)
    if job:
        ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    if cand:
        ensure_can_see(get_org_scope(db, current), cand.org_unit_id)
    return app


# ============ routes ============


@router.get("", response_model=PageOut[ApplicationOut])
def list_applications(
    request: Request,
    job_id: str | None = Query(None),
    stage: str | None = Query(None),
    page: tuple[int, int, str | None] = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, q = page
    stmt = (
        select(JobApplication, Candidate.name)
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .join(Job, Job.id == JobApplication.job_id)
        .where(JobApplication.tenant_id == current.tenant_id)
    )
    if job_id:
        stmt = stmt.where(JobApplication.job_id == job_id)
    if stage:
        if stage not in ALL_STAGES:
            raise HTTPException(400, f"stage 必须是 {ALL_STAGES} 之一")
        stmt = stmt.where(JobApplication.stage == stage)
    scope = get_org_scope(db, current)
    stmt = apply_org_filter(stmt, org_column=Job.org_unit_id, scope=scope)
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    stmt = apply_q_ilike(stmt, q, Candidate.name)
    stmt = stmt.order_by(JobApplication.stage_changed_at.desc())
    total = count_total(db, stmt)
    rows = db.execute(stmt.limit(limit).offset(offset)).all()

    apps = [r[0] for r in rows]
    cands, jobs, users = _maps(db, current.tenant_id, apps)
    scores: dict[str, int] = {}
    by_job: dict[str, list[str]] = {}
    for a in apps:
        by_job.setdefault(a.job_id, []).append(a.candidate_id)
    for jid, cand_ids in by_job.items():
        scores.update(_match_scores(db, current.tenant_id, jid, cand_ids))
    return PageOut(
        items=[_to_out(a, cands, jobs, users, scores) for a in apps],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
def create_application(
    body: ApplicationIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_JOBS)),
):
    if body.stage not in ALL_STAGES:
        raise HTTPException(400, f"stage 必须是 {ALL_STAGES} 之一")
    job = db.get(Job, body.job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    cand = db.get(Candidate, body.candidate_id)
    if not cand or cand.tenant_id != current.tenant_id:
        raise HTTPException(404, "候选人不存在")
    _check_scope(db, current, job, cand)

    app = JobApplication(
        tenant_id=current.tenant_id,
        candidate_id=cand.id,
        job_id=job.id,
        stage=body.stage,
        source="manual",
        owner_user_id=current.id,
        note=body.note,
    )
    db.add(app)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "该候选人已在此岗位的管道中") from None
    db.add(
        JobApplicationEvent(
            tenant_id=current.tenant_id,
            application_id=app.id,
            from_stage=None,
            to_stage=body.stage,
            actor_user_id=current.id,
            note=body.note or "建卡",
        )
    )
    db.commit()
    db.refresh(app)

    write_audit(
        db, actor=current, entity_type="job_application", entity_id=app.id,
        action="create",
        detail={"job_id": job.id, "candidate_id": cand.id, "stage": app.stage},
        request=request,
    )
    db.commit()

    cands, jobs, users = _maps(db, current.tenant_id, [app])
    scores = _match_scores(db, current.tenant_id, app.job_id, [app.candidate_id])
    return _to_out(app, cands, jobs, users, scores)


@router.get("/{app_id}", response_model=ApplicationDetailOut)
def get_application(
    app_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    app = _get_app_in_scope(db, app_id, current)
    cands, jobs, users = _maps(db, current.tenant_id, [app])
    scores = _match_scores(db, current.tenant_id, app.job_id, [app.candidate_id])
    events = db.scalars(
        select(JobApplicationEvent)
        .where(JobApplicationEvent.application_id == app.id)
        .order_by(JobApplicationEvent.created_at)
    ).all()
    actor_ids = {e.actor_user_id for e in events if e.actor_user_id}
    actors = {}
    if actor_ids:
        actors = {
            u.id: u.email
            for u in db.scalars(select(User).where(User.id.in_(actor_ids)))
        }
    out = _to_out(app, cands, jobs, users, scores)
    return ApplicationDetailOut(
        **out.model_dump(),
        events=[
            EventOut(
                id=e.id,
                from_stage=e.from_stage,
                to_stage=e.to_stage,
                actor_email=actors.get(e.actor_user_id) if e.actor_user_id else None,
                note=e.note,
                created_at=e.created_at,
            )
            for e in events
        ],
    )


@router.post("/{app_id}/transition", response_model=ApplicationOut)
def transition(
    app_id: str,
    body: TransitionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_JOBS)),
):
    app = _get_app_in_scope(db, app_id, current)
    if body.to_stage not in ALL_STAGES:
        raise HTTPException(400, f"to_stage 必须是 {ALL_STAGES} 之一")
    if body.to_stage == app.stage:
        raise HTTPException(409, f"已在 {body.to_stage} 阶段")

    from_stage = app.stage
    record_event(db, app, body.to_stage, actor_user_id=current.id, note=body.note)
    # P3 记忆中枢:决策埋点 — 快照当时 AI 匹配评分与逐维 level
    record_decision(
        db,
        tenant_id=current.tenant_id,
        application=app,
        from_stage=from_stage,
        to_stage=body.to_stage,
        actor_id=current.id,
    )
    if body.to_stage == "rejected":
        app.rejected_reason = body.rejected_reason
    elif app.rejected_reason is not None:
        app.rejected_reason = None
    db.commit()
    db.refresh(app)

    write_audit(
        db, actor=current, entity_type="job_application", entity_id=app.id,
        action="transition",
        detail={"from": from_stage, "to": body.to_stage},
        request=request,
    )
    db.commit()

    cands, jobs, users = _maps(db, current.tenant_id, [app])
    scores = _match_scores(db, current.tenant_id, app.job_id, [app.candidate_id])
    return _to_out(app, cands, jobs, users, scores)


@router.delete("/{app_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_application(
    app_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_JOBS)),
):
    app = _get_app_in_scope(db, app_id, current)
    db.delete(app)
    write_audit(
        db, actor=current, entity_type="job_application", entity_id=app_id,
        action="delete", request=request,
    )
    db.commit()


@pipeline_router.get("/{job_id}/pipeline", response_model=PipelineOut)
def get_pipeline(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    scope = get_org_scope(db, current)
    ensure_can_see(scope, job.org_unit_id)

    # 看板卡片同样按候选人数据范围过滤:能看见岗位不等于能看所有候选人
    apps = db.scalars(
        apply_org_filter(
            select(JobApplication)
            .join(Candidate, Candidate.id == JobApplication.candidate_id)
            .where(
                JobApplication.tenant_id == current.tenant_id,
                JobApplication.job_id == job_id,
            )
            .order_by(JobApplication.stage_changed_at.desc()),
            org_column=Candidate.org_unit_id,
            scope=scope,
        )
    ).all()
    cand_ids = [a.candidate_id for a in apps]
    cands, _, users = _maps(db, current.tenant_id, apps)
    scores = _match_scores(db, current.tenant_id, job_id, cand_ids)

    now = _now()
    buckets: dict[str, list[PipelineCard]] = {s: [] for s in ALL_STAGES}
    for a in apps:
        buckets[a.stage].append(
            PipelineCard(
                id=a.id,
                candidate_id=a.candidate_id,
                candidate_name=cands.get(a.candidate_id, "?"),
                stage=a.stage,
                match_score=scores.get(a.candidate_id),
                owner_email=users.get(a.owner_user_id) if a.owner_user_id else None,
                days_in_stage=max(0, (now - a.stage_changed_at).days),
            )
        )
    ordered = STAGE_ORDER + ["rejected", "withdrawn"]
    return PipelineOut(
        job_id=job.id,
        job_title=job.title,
        columns=[
            PipelineColumn(stage=s, count=len(buckets[s]), cards=buckets[s])
            for s in ordered
        ],
        total=len(apps),
    )
