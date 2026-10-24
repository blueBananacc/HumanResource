"""人才再激活外联 API。

- ``GET/POST /campaigns``:创建时把 filter 快照成 sends 行(无邮箱→skipped)
- ``GET /campaigns/{id}``:详情 + sends
- ``POST /campaigns/{id}/send``:celery 逐条 SMTP
- ``POST /campaigns/{id}/cancel``:未投递的标记 cancelled
- filter:{group_id?, skill_ids?, query?} — group_id 走人才分组,skill_ids
  走 candidate_skills join,query 走姓名/邮箱 ILIKE。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Candidate,
    CandidateGroup,
    CandidateGroupMember,
    CandidateSkill,
    TalentCampaign,
    TalentCampaignSend,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    PERM_WRITE_RESUMES,
    apply_org_filter,
    get_org_scope,
    require_permission,
)

router = APIRouter(
    prefix="/campaigns",
    tags=["talent-campaigns"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class FilterIn(BaseModel):
    group_id: str | None = None
    skill_ids: list[str] = Field(default_factory=list, max_length=50)
    query: str | None = Field(default=None, max_length=128)


class CampaignIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    filter: FilterIn = Field(default_factory=FilterIn)
    subject: str = Field(min_length=1, max_length=128)
    body_text: str = Field(min_length=1, max_length=20000)


class SendOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str | None = None
    to_email: str
    status: str
    error: str | None
    sent_at: datetime | None


class CampaignOut(BaseModel):
    id: str
    name: str
    filter: dict
    subject: str
    body_text: str
    status: str
    total: int
    sent_count: int
    failed_count: int
    created_by: str | None
    started_at: datetime | None
    done_at: datetime | None
    created_at: datetime
    sends: list[SendOut] = []


def _out(db: Session, c: TalentCampaign, *, with_sends: bool = False) -> CampaignOut:
    sends: list[SendOut] = []
    if with_sends:
        rows = db.execute(
            select(TalentCampaignSend, Candidate.name)
            .join(Candidate, Candidate.id == TalentCampaignSend.candidate_id)
            .where(TalentCampaignSend.campaign_id == c.id)
            .order_by(TalentCampaignSend.created_at)
            .limit(200)
        ).all()
        sends = [
            SendOut(
                id=s.id, candidate_id=s.candidate_id, candidate_name=name,
                to_email=s.to_email, status=s.status, error=s.error, sent_at=s.sent_at,
            )
            for s, name in rows
        ]
    return CampaignOut(
        id=c.id,
        name=c.name,
        filter=c.filter,
        subject=c.subject,
        body_text=c.body_text,
        status=c.status,
        total=c.total,
        sent_count=c.sent_count,
        failed_count=c.failed_count,
        created_by=c.created_by,
        started_at=c.started_at,
        done_at=c.done_at,
        created_at=c.created_at,
        sends=sends,
    )


def _get(db: Session, cid: str, current: User) -> TalentCampaign:
    c = db.get(TalentCampaign, cid)
    if c is None or c.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "触达活动不存在")
    return c


def _resolve_candidates(db: Session, f: FilterIn, current: User, scope) -> list[Candidate]:
    # 两段式:id distinct → 实体 IN;candidates.tags 是 JSON 列,
    # DISTINCT 全行会撞 postgres 的 "no equality operator for json"
    stmt = select(Candidate.id).where(Candidate.tenant_id == current.tenant_id)
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if f.group_id:
        stmt = stmt.join(
            CandidateGroupMember,
            (CandidateGroupMember.candidate_id == Candidate.id)
            & (CandidateGroupMember.group_id == f.group_id),
        )
    if f.skill_ids:
        stmt = stmt.join(
            CandidateSkill, CandidateSkill.candidate_id == Candidate.id
        ).where(CandidateSkill.skill_id.in_(f.skill_ids))
    if f.query:
        pattern = f"%{f.query}%"
        stmt = stmt.where(
            (Candidate.name.ilike(pattern))
            | (Candidate.display_email.ilike(pattern))
        )
    ids = db.scalars(stmt.distinct().limit(500)).all()
    if not ids:
        return []
    return db.scalars(select(Candidate).where(Candidate.id.in_(ids))).all()


# ============ endpoints ============


@router.get("", response_model=PageOut[CampaignOut])
def list_campaigns(
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _q = paging
    stmt = select(TalentCampaign).where(TalentCampaign.tenant_id == current.tenant_id)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(TalentCampaign.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(items=[_out(db, c) for c in rows], total=total, limit=limit, offset=offset)


@router.post("", response_model=CampaignOut, status_code=status.HTTP_201_CREATED)
def create_campaign(
    body: CampaignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_RESUMES)),
):
    scope = get_org_scope(db, current)
    if body.filter.group_id:
        g = db.get(CandidateGroup, body.filter.group_id)
        if g is None or g.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "分组不存在")
    cands = _resolve_candidates(db, body.filter, current, scope)
    c = TalentCampaign(
        tenant_id=current.tenant_id,
        name=body.name,
        filter=body.filter.model_dump(),
        subject=body.subject,
        body_text=body.body_text,
        total=len(cands),
        created_by=current.id,
    )
    db.add(c)
    db.flush()
    for cand in cands:
        email = (cand.display_email or "").strip()
        db.add(
            TalentCampaignSend(
                tenant_id=current.tenant_id,
                campaign_id=c.id,
                candidate_id=cand.id,
                to_email=email or "",
                status="queued" if email else "skipped_no_email",
            )
        )
    write_audit(
        db, actor=current, entity_type="talent_campaign", entity_id=c.id,
        action="create", detail={"total": c.total}, request=request,
    )
    db.commit()
    return _out(db, c, with_sends=True)


@router.get("/{cid}", response_model=CampaignOut)
def get_campaign(
    cid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return _out(db, _get(db, cid, current), with_sends=True)


@router.post("/{cid}/send", response_model=CampaignOut)
def send_campaign(
    cid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_RESUMES)),
):
    c = _get(db, cid, current)
    # worker crash/重启会把活动留在 sending —— 超 1h 无进展视为陈旧,允许重发
    # (幂等:任务只投递 queued 行,已 sent 的不会重发)
    stale_sending = (
        c.status == "sending"
        and (c.started_at is None or c.started_at < _now() - timedelta(hours=1))
    )
    if c.status not in ("draft", "failed") and not stale_sending:
        raise HTTPException(status.HTTP_409_CONFLICT, "该活动已在发送或已完成")
    if c.status == "failed":
        # 整批失败后的重试:把 failed 的 sends 重置回 queued 再投递
        db.query(TalentCampaignSend).filter(
            TalentCampaignSend.campaign_id == c.id,
            TalentCampaignSend.status == "failed",
        ).update({"status": "queued", "error": None})
    queued = count_total(
        db,
        select(TalentCampaignSend).where(
            TalentCampaignSend.campaign_id == c.id,
            TalentCampaignSend.status == "queued",
        ),
    )
    if stale_sending and queued == 0:
        # worker 在最后一条 send 落库后、置终态前挂掉 —— 现场按任务尾部
        # 同一口径重算计数并落终态,否则活动永远卡在 sending(只能 cancel)
        c.sent_count = int(
            db.execute(
                select(func.count(TalentCampaignSend.id)).where(
                    TalentCampaignSend.campaign_id == c.id,
                    TalentCampaignSend.status == "sent",
                )
            ).scalar_one()
        )
        c.failed_count = int(
            db.execute(
                select(func.count(TalentCampaignSend.id)).where(
                    TalentCampaignSend.campaign_id == c.id,
                    TalentCampaignSend.status.in_(("failed", "skipped_no_email")),
                )
            ).scalar_one()
        )
        c.status = "done" if c.sent_count > 0 else "failed"
        c.done_at = _now()
        write_audit(
            db, actor=current, entity_type="talent_campaign", entity_id=c.id,
            action="send", request=request,
        )
        db.commit()
        return _out(db, c, with_sends=True)
    if queued == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "没有待投递的收件人")
    c.status = "sending"
    c.started_at = _now()
    write_audit(db, actor=current, entity_type="talent_campaign", entity_id=c.id, action="send", request=request)
    db.commit()
    from app.workers.tasks.notify import send_talent_campaign

    try:
        send_talent_campaign.delay(str(c.id))
    except Exception:
        # broker 不可达 → 回滚到 draft,否则活动永远卡在 sending
        c.status = "draft"
        c.started_at = None
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "发送任务排队失败,请稍后重试"
        ) from None
    return _out(db, c, with_sends=True)


@router.post("/{cid}/cancel", response_model=CampaignOut)
def cancel_campaign(
    cid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_RESUMES)),
):
    c = _get(db, cid, current)
    if c.status in ("done",):
        raise HTTPException(status.HTTP_409_CONFLICT, "已发送完成的活动不能取消")
    db.query(TalentCampaignSend).filter(
        TalentCampaignSend.campaign_id == c.id,
        TalentCampaignSend.status == "queued",
    ).update({"status": "cancelled"})
    c.status = "cancelled"
    c.done_at = _now()
    write_audit(db, actor=current, entity_type="talent_campaign", entity_id=c.id, action="cancel", request=request)
    db.commit()
    return _out(db, c, with_sends=True)
