"""主动陪伴:招聘早报(对标 Moka「主动推进,跟进不等人」)。

``GET /api/briefing/today`` 聚合「昨日至今新入库简历 / 今日面试 /
待跟进 Offer / 滞留申请 / 记忆中枢学习进度」,一屏交付。

跳过逻辑:``has_items`` 为 False 时前端不打扰(只留一条静默提示)。
所有数据租户隔离;聚合只用计数+少量条目,不返回敏感全文。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import (
    Candidate,
    InterviewScorecard,
    JobApplication,
    JobOffer,
    Job,
    Resume,
    User,
)
from app.infra.db import get_db
from app.services.decision_learning import weight_suggestions

router = APIRouter(prefix="/briefing", tags=["briefing"])

# 滞留阈值:处于在途阶段且超过 N 天未流转,视为该人工跟进
_STALE_DAYS = 5


def _today_start() -> datetime:
    now = datetime.now(UTC).replace(tzinfo=None)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@router.get("/today")
def today_briefing(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    t = current.tenant_id
    now = datetime.now(UTC).replace(tzinfo=None)
    day0 = _today_start()

    # 近 24h 新入库简历(取最近 5 条预览)
    res_q = (
        select(Resume, Candidate.name)
        .join(Candidate, Candidate.id == Resume.candidate_id)
        .where(Resume.tenant_id == t, Resume.created_at >= day0 - timedelta(days=1))
        .order_by(Resume.created_at.desc())
        .limit(5)
    )
    res_rows = db.execute(res_q).all()
    res_total = db.scalar(
        select(func.count(Resume.id)).where(
            Resume.tenant_id == t, Resume.created_at >= day0 - timedelta(days=1)
        )
    ) or 0

    # 今日面试(评分卡 scheduled_at 落在今天,仍 open)
    iv_q = (
        select(InterviewScorecard, Candidate.name, Job.title)
        .join(Candidate, Candidate.id == InterviewScorecard.candidate_id)
        .outerjoin(Job, Job.id == InterviewScorecard.job_id)
        .where(
            InterviewScorecard.tenant_id == t,
            InterviewScorecard.status == "open",
            InterviewScorecard.scheduled_at >= day0,
            InterviewScorecard.scheduled_at < day0 + timedelta(days=1),
        )
        .order_by(InterviewScorecard.scheduled_at)
        .limit(8)
    )
    iv_rows = db.execute(iv_q).all()

    # 待跟进 Offer:draft 待发出 / sent 已发未回
    of_q = (
        select(JobOffer, Candidate.name)
        .join(Candidate, Candidate.id == JobOffer.candidate_id)
        .where(JobOffer.tenant_id == t, JobOffer.status.in_(["draft", "sent"]))
        .order_by(JobOffer.created_at.desc())
        .limit(5)
    )
    of_rows = db.execute(of_q).all()

    # 滞留申请:在途阶段 & 超期未动
    stale_q = (
        select(JobApplication, Candidate.name, Job.title)
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .join(Job, Job.id == JobApplication.job_id)
        .where(
            JobApplication.tenant_id == t,
            JobApplication.stage.in_(["applied", "screening", "interview", "offer"]),
            JobApplication.stage_changed_at < now - timedelta(days=_STALE_DAYS),
        )
        .order_by(JobApplication.stage_changed_at)
        .limit(5)
    )
    st_rows = db.execute(stale_q).all()

    # 记忆中枢:全租户决策学习总览(复用 P3 引擎,job_id=None 全量)
    learned = weight_suggestions(db, t, days=90)
    n = learned["decisions_total"]
    confidence = "high" if n >= 20 else ("medium" if n >= 5 else "low")
    top_dims = [
        {
            "name": d["name"],
            "current_weight": d["current_weight"],
            "suggested_weight": d["suggested_weight"],
            "gap": d["gap"],
        }
        for d in learned["dimensions"][:3]
        if d.get("suggested_weight") is not None
    ]

    sections = [
        {
            "key": "new_resumes",
            "title": "新入库简历",
            "count": res_total,
            "items": [
                {"candidate_id": str(r.Resume.candidate_id), "name": r.name}
                for r in res_rows
            ],
            "link": "/talents",
        },
        {
            "key": "today_interviews",
            "title": "今日面试",
            "count": len(iv_rows),
            "items": [
                {
                    "id": str(r.InterviewScorecard.id),
                    "title": r.InterviewScorecard.title,
                    "candidate": r.name,
                    "job": r[2],
                    "scheduled_at": r.InterviewScorecard.scheduled_at.isoformat()
                    if r.InterviewScorecard.scheduled_at
                    else None,
                }
                for r in iv_rows
            ],
            "link": "/interviews/schedule",
        },
        {
            "key": "pending_offers",
            "title": "待跟进 Offer",
            "count": len(of_rows),
            "items": [
                {
                    "id": str(r.JobOffer.id),
                    "candidate": r.name,
                    "position": r.JobOffer.position,
                    "status": r.JobOffer.status,
                }
                for r in of_rows
            ],
            "link": "/offers",
        },
        {
            "key": "stale_apps",
            "title": f"滞留超 {_STALE_DAYS} 天",
            "count": len(st_rows),
            "items": [
                {
                    "id": str(r.JobApplication.id),
                    "candidate": r.name,
                    "job": r[2],
                    "stage": r.JobApplication.stage,
                }
                for r in st_rows
            ],
            "link": "/pipeline",
        },
    ]

    return {
        "date": day0.date().isoformat(),
        "generated_at": now.isoformat(),
        "sections": sections,
        "memory": {
            "decisions_total": n,
            "positive": learned["positive"],
            "negative": learned["negative"],
            "confidence": confidence,
            "top_dimensions": top_dims,
        },
        "has_items": any(s["count"] > 0 for s in sections),
    }
