"""结构化面试评分卡 API(T9)。

人类面试官的结构化评价载体:考察维度(criteria)+ 指派面试官 +
逐维评分 + 录用表态 + AI 聚合。与 AI ``/api/interviews`` 互补。

- HR/管理员:create/edit/close/delete/synthesize — ``write_interview``
- 面试官本人:``GET /mine`` + ``POST /{id}/feedback`` — 按 user.id 匹配
- 查看:view_reports + 候选人 org scope;被指派的面试官无需该权限
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import String, func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.idempotency import idempotent
from app.domain.models import (
    Candidate,
    Interview,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    User,
)
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.services import scorecards as svc
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    PERM_WRITE_INTERVIEW,
    ensure_can_see,
    get_org_scope,
    has_permission,
    require_permission,
)

router = APIRouter(prefix="/scorecards", tags=["scorecards"])


# ---------------------------- Schemas -----------------------------


class ScorecardIn(BaseModel):
    candidate_id: str
    job_id: str | None = None
    interview_id: str | None = None
    title: str = Field(..., min_length=1, max_length=128)
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    interviewer_ids: list[str] = Field(default_factory=list)
    scheduled_at: datetime | None = None
    location: str | None = Field(None, max_length=256)
    note: str | None = Field(None, max_length=2000)


class ScorecardPatch(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=128)
    criteria: list[dict[str, Any]] | None = None
    interviewer_ids: list[str] | None = None
    scheduled_at: datetime | None = None
    location: str | None = Field(None, max_length=256)
    note: str | None = Field(None, max_length=2000)
    status: str | None = None
    base_updated_at: datetime | None = None


class FeedbackIn(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    verdict: str = Field(...)
    scores: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = Field(None, max_length=4000)


class FeedbackOut(BaseModel):
    id: str
    interviewer_user_id: str
    interviewer_name: str
    rating: int
    verdict: str
    scores: list[dict[str, Any]]
    notes: str | None
    submitted_at: datetime


class ScorecardOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    job_id: str | None
    job_title: str | None
    interview_id: str | None
    title: str
    criteria: list[dict[str, Any]]
    status: str
    scheduled_at: datetime | None
    location: str | None
    note: str | None
    interviewers: list[dict[str, Any]]
    feedbacks: list[FeedbackOut]
    aggregate: dict[str, Any]
    ai_summary: dict[str, Any] | None
    created_by: str | None
    created_at: datetime
    updated_at: datetime
    deletable: bool = True


# ---------------------------- Helpers -----------------------------


def _get_scoped(
    db: Session, current: User, sid: str, *, for_feedback: bool = False
) -> InterviewScorecard:
    """取卡 + 可见性校验。

    可见:(view_reports 且候选 org scope 内)或(本人是被指派面试官)。
    """
    sc = db.get(InterviewScorecard, sid)
    if sc is None or sc.tenant_id != current.tenant_id:
        raise HTTPException(404, "评分卡不存在")
    if has_permission(current, PERM_VIEW_REPORTS):
        cand = db.get(Candidate, sc.candidate_id)
        if cand is not None:
            ensure_can_see(get_org_scope(db, current), cand.org_unit_id)
    elif current.id not in (sc.interviewer_ids or []):
        raise HTTPException(404, "评分卡不存在")
    return sc


def _interviewer_rows(
    db: Session, sc: InterviewScorecard, fb_map: dict[str, InterviewFeedback]
) -> list[dict[str, Any]]:
    names = svc.user_display_names(db, sc.tenant_id, list(sc.interviewer_ids or []))
    return [
        {
            "user_id": uid,
            "name": names.get(uid, "(已删除)"),
            "submitted": uid in fb_map,
        }
        for uid in (sc.interviewer_ids or [])
    ]


def _full_out(db: Session, sc: InterviewScorecard) -> ScorecardOut:
    cand = db.get(Candidate, sc.candidate_id)
    job = db.get(Job, sc.job_id) if sc.job_id else None
    feedbacks = list(
        db.scalars(
            select(InterviewFeedback)
            .where(InterviewFeedback.scorecard_id == sc.id)
            .order_by(InterviewFeedback.submitted_at)
        ).all()
    )
    fb_map = {f.interviewer_user_id: f for f in feedbacks}
    names = svc.user_display_names(
        db, sc.tenant_id, [f.interviewer_user_id for f in feedbacks]
    )
    return ScorecardOut(
        id=sc.id,
        candidate_id=sc.candidate_id,
        candidate_name=cand.name if cand else "(已删除)",
        job_id=sc.job_id,
        job_title=job.title if job else None,
        interview_id=sc.interview_id,
        title=sc.title,
        criteria=list(sc.criteria or []),
        status=sc.status,
        scheduled_at=sc.scheduled_at,
        location=sc.location,
        note=sc.note,
        interviewers=_interviewer_rows(db, sc, fb_map),
        feedbacks=[
            FeedbackOut(
                id=f.id,
                interviewer_user_id=f.interviewer_user_id,
                interviewer_name=names.get(f.interviewer_user_id, "?"),
                rating=f.rating,
                verdict=f.verdict,
                scores=list(f.scores or []),
                notes=f.notes,
                submitted_at=f.submitted_at,
            )
            for f in feedbacks
        ],
        aggregate=svc.aggregate(sc, feedbacks),
        ai_summary=sc.ai_summary,
        created_by=sc.created_by,
        created_at=sc.created_at,
        updated_at=sc.updated_at,
    )


def _check_interviewers(db: Session, tenant_id: str, ids: list[str]) -> list[str]:
    """面试官必须是本租户激活用户;去重保序。返回清洗后的 user_id 列表。"""
    out = list(dict.fromkeys(i for i in ids if i))
    if not out:
        return out
    valid = set(
        db.scalars(
            select(User.id).where(
                User.tenant_id == tenant_id,
                User.status == "active",
                User.id.in_(out),
            )
        ).all()
    )
    bad = [i for i in out if i not in valid]
    if bad:
        raise HTTPException(400, f"面试官不是本租户用户: {','.join(bad[:3])}")
    return out


# ---------------------------- Routes -----------------------------


@router.post(
    "",
    response_model=ScorecardOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission(PERM_WRITE_INTERVIEW))],
)
@idempotent(scope="scorecards.create", resource_type="interview_scorecard")
def create_scorecard(
    body: ScorecardIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    cand = db.get(Candidate, body.candidate_id)
    if cand is None or cand.tenant_id != current.tenant_id:
        raise HTTPException(404, "候选人不存在")
    ensure_can_see(get_org_scope(db, current), cand.org_unit_id)
    job = None
    if body.job_id:
        job = db.get(Job, body.job_id)
        if job is None or job.tenant_id != current.tenant_id:
            raise HTTPException(404, "岗位不存在")
        ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    if body.interview_id:
        iv = db.get(Interview, body.interview_id)
        if iv is None or iv.tenant_id != current.tenant_id:
            raise HTTPException(404, "面试不存在")
        if iv.candidate_id != cand.id:
            raise HTTPException(400, "面试与候选人不匹配")
    try:
        criteria = svc.normalize_criteria(body.criteria)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    interviewers = _check_interviewers(db, current.tenant_id, body.interviewer_ids)

    sc = InterviewScorecard(
        tenant_id=current.tenant_id,
        candidate_id=cand.id,
        job_id=job.id if job else None,
        interview_id=body.interview_id,
        title=body.title.strip(),
        criteria=criteria,
        interviewer_ids=interviewers,
        scheduled_at=body.scheduled_at,
        location=(body.location or "").strip() or None,
        note=(body.note or "").strip() or None,
        created_by=current.id,
    )
    db.add(sc)
    db.flush()
    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="create",
        detail={"title": sc.title, "candidate_id": cand.id,
                "fields": {"title": sc.title, "status": sc.status}},
        request=request,
    )
    svc.notify_assigned(db, sc, interviewers)
    db.commit()
    return _full_out(db, sc)


@router.get("/mine", response_model=list[ScorecardOut])
def my_scorecards(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """我被指派为面试官的评分卡(含我是否已提交)。"""
    rows = db.scalars(
        select(InterviewScorecard)
        .where(
            InterviewScorecard.tenant_id == current.tenant_id,
            InterviewScorecard.interviewer_ids.cast(String).like(f'%"{current.id}"%'),
        )
        .order_by(InterviewScorecard.created_at.desc())
    ).all()
    return [_full_out(db, sc) for sc in rows]


# ---------------------------- 面试官工作台 ----------------------------


class WorkbenchItem(BaseModel):
    id: str
    title: str
    candidate_id: str
    candidate_name: str
    job_id: str | None
    job_title: str | None
    scheduled_at: datetime | None
    location: str | None
    status: str
    criteria: list[dict[str, Any]]
    my_submitted: bool
    submitted_count: int
    interviewer_count: int
    overdue: bool


class WorkbenchOut(BaseModel):
    """面试官待办台:待我评价(含逾期) / 即将到来的面试 / 我已提交。"""

    todo: list[WorkbenchItem]
    upcoming: list[WorkbenchItem]
    done: list[WorkbenchItem]


def _wb_item(
    db: Session, sc: InterviewScorecard, fb_users: set[str], current: User
) -> WorkbenchItem:
    cand = db.get(Candidate, sc.candidate_id)
    job = db.get(Job, sc.job_id) if sc.job_id else None
    return WorkbenchItem(
        id=sc.id,
        title=sc.title,
        candidate_id=sc.candidate_id,
        candidate_name=cand.name if cand else "(已删除)",
        job_id=sc.job_id,
        job_title=job.title if job else None,
        scheduled_at=sc.scheduled_at,
        location=sc.location,
        status=sc.status,
        criteria=list(sc.criteria or []),
        my_submitted=current.id in fb_users,
        submitted_count=len(fb_users),
        interviewer_count=len(sc.interviewer_ids or []),
        overdue=svc.scorecard_overdue(sc),
    )


@router.get("/workbench", response_model=WorkbenchOut)
def interviewer_workbench(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """面试官工作台 — 我被指派的所有评分卡按待办分桶。

    - ``todo``:open 且我未提交(``overdue`` 标记面试已过 24h 未交);
    - ``upcoming``:open 且 scheduled_at 在未来(面试还没开始,先看上下文);
    - ``done``:我已提交或卡已 closed。
    """
    rows = db.scalars(
        select(InterviewScorecard)
        .where(
            InterviewScorecard.tenant_id == current.tenant_id,
            InterviewScorecard.interviewer_ids.cast(String).like(f'%"{current.id}"%'),
        )
        .order_by(InterviewScorecard.scheduled_at.asc().nullslast())
    ).all()
    fb_users_map: dict[str, set[str]] = {}
    if rows:
        for sid, uid in db.execute(
            select(
                InterviewFeedback.scorecard_id,
                InterviewFeedback.interviewer_user_id,
            ).where(InterviewFeedback.scorecard_id.in_([r.id for r in rows]))
        ).all():
            fb_users_map.setdefault(sid, set()).add(uid)

    now = datetime.now(UTC).replace(tzinfo=None)
    todo: list[WorkbenchItem] = []
    upcoming: list[WorkbenchItem] = []
    done: list[WorkbenchItem] = []
    for sc in rows:
        item = _wb_item(db, sc, fb_users_map.get(sc.id, set()), current)
        sched = sc.scheduled_at
        sched_naive = sched.replace(tzinfo=None) if sched and sched.tzinfo else sched
        if not item.my_submitted and sc.status == "open":
            if sched_naive is None or sched_naive <= now:
                todo.append(item)
            else:
                upcoming.append(item)
        else:
            done.append(item)
    # 逾期优先排最前
    todo.sort(key=lambda i: (not i.overdue, i.scheduled_at or datetime.max))
    return WorkbenchOut(todo=todo, upcoming=upcoming, done=done)


@router.post(
    "/{sid}/nudge",
    dependencies=[Depends(require_permission(PERM_WRITE_INTERVIEW))],
)
def nudge_scorecard(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """HR 手动催办:给该卡未交反馈的面试官发提醒(cooldown 去重)。"""
    sc = _get_scoped(db, current, sid)
    if sc.status != "open":
        raise HTTPException(400, "评分卡已关闭")
    sent = svc.nudge_scorecard(db, sc)
    names = svc.user_display_names(db, sc.tenant_id, sent)
    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="nudge",
        detail={"sent_to": sent, "pending": svc.pending_interviewers(db, sc)},
        request=request,
    )
    db.commit()
    return {
        "sent": [{"user_id": u, "name": names.get(u, "?")} for u in sent],
        "pending_count": len(svc.pending_interviewers(db, sc)),
    }


@router.get(
    "",
    response_model=PageOut[ScorecardOut],
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def list_scorecards(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    candidate_id: str | None = Query(None),
    job_id: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    limit, offset, _q = p
    stmt = (
        select(InterviewScorecard)
        .join(Candidate, Candidate.id == InterviewScorecard.candidate_id)
        .where(InterviewScorecard.tenant_id == current.tenant_id)
    )
    scope = get_org_scope(db, current)
    if scope is not None:
        stmt = stmt.where(
            (Candidate.org_unit_id.in_(scope)) | (Candidate.org_unit_id.is_(None))
        )
    if candidate_id:
        stmt = stmt.where(InterviewScorecard.candidate_id == candidate_id)
    if job_id:
        stmt = stmt.where(InterviewScorecard.job_id == job_id)
    if status_filter:
        stmt = stmt.where(InterviewScorecard.status == status_filter)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(InterviewScorecard.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut[ScorecardOut](
        items=[_full_out(db, sc) for sc in rows],
        total=total, limit=limit, offset=offset,
    )


@router.get("/{sid}", response_model=ScorecardOut)
def get_scorecard(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    sc = _get_scoped(db, current, sid)
    return _full_out(db, sc)


@router.patch(
    "/{sid}",
    response_model=ScorecardOut,
    dependencies=[Depends(require_permission(PERM_WRITE_INTERVIEW))],
)
def update_scorecard(
    sid: str,
    body: ScorecardPatch,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    sc = _get_scoped(db, current, sid)
    if body.base_updated_at is not None and sc.updated_at:
        if sc.updated_at.replace(microsecond=0) != body.base_updated_at.replace(microsecond=0):
            raise HTTPException(409, "记录已被他人修改,请刷新后重试")

    changed: dict[str, Any] = {}
    has_feedback = db.scalars(
        select(InterviewFeedback.id).where(InterviewFeedback.scorecard_id == sc.id).limit(1)
    ).first() is not None

    if body.criteria is not None:
        if has_feedback:
            raise HTTPException(400, "已有面试官反馈,考察维度已冻结不可修改")
        try:
            sc.criteria = svc.normalize_criteria(body.criteria)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        changed["criteria"] = sc.criteria
    newly_assigned: list[str] = []
    if body.interviewer_ids is not None:
        before = set(sc.interviewer_ids or [])
        sc.interviewer_ids = _check_interviewers(db, current.tenant_id, body.interviewer_ids)
        newly_assigned = [u for u in sc.interviewer_ids if u not in before]
        changed["interviewer_ids"] = sc.interviewer_ids
    if body.title is not None:
        sc.title = body.title.strip()
        changed["title"] = sc.title
    if body.location is not None:
        sc.location = body.location.strip() or None
        changed["location"] = sc.location
    if body.note is not None:
        sc.note = body.note.strip() or None
        changed["note"] = sc.note
    if body.scheduled_at is not None:
        sc.scheduled_at = body.scheduled_at
        changed["scheduled_at"] = body.scheduled_at.isoformat()
    if body.status is not None:
        if body.status not in ("open", "closed"):
            raise HTTPException(400, "status 取值 open/closed")
        sc.status = body.status
        changed["status"] = sc.status

    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="update", detail={"fields": changed}, request=request,
    )
    svc.notify_assigned(db, sc, newly_assigned)
    db.commit()
    return _full_out(db, sc)


@router.delete(
    "/{sid}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission(PERM_WRITE_INTERVIEW))],
)
def delete_scorecard(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    sc = _get_scoped(db, current, sid)
    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="delete", detail={"title": sc.title}, request=request,
    )
    db.delete(sc)
    db.commit()


@router.post(
    "/{sid}/feedback",
    response_model=ScorecardOut,
)
@idempotent(scope="scorecards.feedback", resource_type="interview_scorecard")
def submit_feedback(
    sid: str,
    body: FeedbackIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    sc = _get_scoped(db, current, sid, for_feedback=True)
    if sc.status != "open":
        raise HTTPException(400, "评分卡已关闭,不能再提交反馈")
    if current.id not in (sc.interviewer_ids or []):
        raise HTTPException(403, "你不是本场面试的指派面试官")
    if body.verdict not in svc.VERDICTS:
        raise HTTPException(400, f"verdict 取值 {svc.VERDICTS}")
    try:
        scores = svc.validate_scores(sc.criteria or [], body.scores)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    fb = svc.submit_feedback(
        db, scorecard=sc, interviewer_user_id=current.id, rating=body.rating,
        verdict=body.verdict, notes=(body.notes or "").strip() or None,
        scores=scores,
    )
    svc.clear_nudges_on_feedback(db, sc, current.id)
    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="feedback",
        detail={"user_id": current.id, "verdict": fb.verdict, "rating": fb.rating},
        request=request,
    )
    db.commit()
    return _full_out(db, sc)


@router.post("/{sid}/synthesize", response_model=ScorecardOut)
@idempotent(scope="scorecards.synthesize", resource_type="interview_scorecard")
def synthesize_scorecard(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_WRITE_INTERVIEW):
        raise HTTPException(403, "权限不足")
    if not is_feature_enabled(db, "assistant.agent"):
        raise HTTPException(402, "功能未启用: assistant.agent")
    sc = _get_scoped(db, current, sid)
    feedbacks = list(
        db.scalars(
            select(InterviewFeedback).where(InterviewFeedback.scorecard_id == sc.id)
        ).all()
    )
    if not feedbacks:
        raise HTTPException(400, "尚无面试官反馈,无法聚合")
    sc.ai_summary = svc.synthesize(db, sc, feedbacks)
    write_audit(
        db, actor=current, entity_type="interview_scorecard", entity_id=sc.id,
        action="synthesize", detail={"feedback_count": len(feedbacks)},
        request=request,
    )
    db.commit()
    return _full_out(db, sc)


@router.post("/ai-criteria")
def suggest_criteria(
    body: dict[str, Any],
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """AI 建议考察维度 — 只回草稿不落库。"""
    if not has_permission(current, PERM_WRITE_INTERVIEW):
        raise HTTPException(403, "权限不足")
    if not is_feature_enabled(db, "assistant.agent"):
        raise HTTPException(402, "功能未启用: assistant.agent")
    job = None
    if body.get("job_id"):
        job = db.get(Job, str(body["job_id"]))
        if job is None or job.tenant_id != current.tenant_id:
            raise HTTPException(404, "岗位不存在")
    cand = None
    if body.get("candidate_id"):
        cand = db.get(Candidate, str(body["candidate_id"]))
        if cand is None or cand.tenant_id != current.tenant_id:
            raise HTTPException(404, "候选人不存在")
    items = svc.suggest_criteria(db, current.tenant_id, job, cand)
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id="scorecard.criteria",
        action="field_assist",
        detail={"job_id": job.id if job else None,
                "candidate_id": cand.id if cand else None,
                "suggested": len(items)},
        request=request,
    )
    db.commit()
    return {"criteria": items}
