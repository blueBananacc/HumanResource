"""P3 记忆中枢:决策埋点 + 「学到了什么」权重建议。

- ``record_decision``:JobApplication stage 流转时快照 AI 匹配评分,
  把「HR 真实选择」与「AI 当时的判断」绑定留存。
- ``weight_suggestions``:按维度聚合 正/负 决策的 level 均值差,
  给出 ScoringRule 维度权重调整建议 — 「我从你的选择里学到了什么」。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.domain.models import (
    DecisionFeedback,
    JobApplication,
    Resume,
    ResumeJobMatch,
)

# to_stage → decision 语义;applied/withdrawn 不算 HR 决策,不埋点
_STAGE_DECISION = {
    "screening": "advanced",
    "interview": "advanced",
    "offer": "offered",
    "hired": "hired",
    "rejected": "rejected",
}
_NEGATIVE = ("rejected",)


def record_decision(
    db: Session,
    *,
    tenant_id: str,
    application: JobApplication,
    from_stage: str | None,
    to_stage: str,
    actor_id: str | None,
) -> DecisionFeedback | None:
    """管道流转埋点;非决策性流转(applied/withdrawn)返回 None 不记。"""
    decision = _STAGE_DECISION.get(to_stage)
    if decision is None:
        return None
    match = _latest_match(db, tenant_id, application)
    fb = DecisionFeedback(
        tenant_id=tenant_id,
        job_id=application.job_id,
        candidate_id=application.candidate_id,
        application_id=application.id,
        decision=decision,
        from_stage=from_stage,
        to_stage=to_stage,
        match_score=match.score if match else None,
        criteria_snapshot=match.criteria if match else None,
        decided_by=actor_id,
    )
    db.add(fb)
    return fb


def _latest_match(
    db: Session, tenant_id: str, application: JobApplication
) -> ResumeJobMatch | None:
    """找该申请 (job × 候选人最新简历) 最近一次完成的匹配评估。"""
    return db.scalar(
        select(ResumeJobMatch)
        .join(Resume, Resume.id == ResumeJobMatch.resume_id)
        .where(
            ResumeJobMatch.tenant_id == tenant_id,
            ResumeJobMatch.job_id == application.job_id,
            Resume.candidate_id == application.candidate_id,
        )
        .order_by(
            desc(ResumeJobMatch.finished_at),
            desc(ResumeJobMatch.created_at),
        )
        .limit(1)
    )


def weight_suggestions(
    db: Session, tenant_id: str, *, job_id: str | None = None, days: int = 90
) -> dict[str, Any]:
    """「我从你的选择里学到了什么」:按维度给出权重调整建议。

    算法(可解释、零 LLM):
    - 取近 ``days`` 天带 criteria_snapshot 的决策,按维度 key 分组;
    - 分别算 正决策(advanced/offered/hired)与负决策(rejected)下该维
      level 均值(0-4);
    - gap = pos_avg - neg_avg:gap 越大 → 该维越能区分好坏 → 建议提权;
      gap 为负 → 该维在淘汰者上反而更高 → 建议降权;
    - 输出维度级明细 + 归一化建议权重(相对当前权重 ±20% 内)。
    """
    q = select(DecisionFeedback).where(
        DecisionFeedback.tenant_id == tenant_id,
        DecisionFeedback.criteria_snapshot.is_not(None),
    )
    if job_id:
        q = q.where(DecisionFeedback.job_id == job_id)
    if days:
        from datetime import UTC, datetime, timedelta

        q = q.where(
            DecisionFeedback.created_at
            >= datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        )
    rows = db.scalars(q.order_by(desc(DecisionFeedback.created_at)).limit(500)).all()

    # dim_key → {name, pos:[level], neg:[level], weight}
    agg: dict[str, dict[str, Any]] = {}
    total_pos = total_neg = 0
    for fb in rows:
        pos = fb.decision not in _NEGATIVE
        total_pos += 1 if pos else 0
        total_neg += 0 if pos else 1
        for c in fb.criteria_snapshot or []:
            key = str(c.get("key") or c.get("name") or "")
            if not key:
                continue
            entry = agg.setdefault(
                key,
                {"name": c.get("name") or key, "pos": [], "neg": [], "weight": c.get("weight")},
            )
            lvl = c.get("level")
            if isinstance(lvl, (int, float)):
                (entry["pos"] if pos else entry["neg"]).append(float(lvl))
            if c.get("weight") is not None:
                entry["weight"] = c["weight"]

    dims = []
    for key, e in agg.items():
        pos_avg = sum(e["pos"]) / len(e["pos"]) if e["pos"] else None
        neg_avg = sum(e["neg"]) / len(e["neg"]) if e["neg"] else None
        gap = (pos_avg - neg_avg) if (pos_avg is not None and neg_avg is not None) else None
        cur_w = float(e["weight"]) if e.get("weight") is not None else None
        suggestion = None
        if gap is not None and cur_w is not None:
            # gap∈[-4,4] → 建议幅度 ∝ gap/4,±20% 封顶
            delta = max(-0.2, min(0.2, (gap / 4.0) * 0.4))
            suggestion = round(cur_w * (1 + delta), 3)
        if pos_avg is not None or neg_avg is not None:
            dims.append(
                {
                    "key": key,
                    "name": e["name"],
                    "current_weight": cur_w,
                    "suggested_weight": suggestion,
                    "pos_avg_level": round(pos_avg, 2) if pos_avg is not None else None,
                    "neg_avg_level": round(neg_avg, 2) if neg_avg is not None else None,
                    "gap": round(gap, 2) if gap is not None else None,
                    "samples_pos": len(e["pos"]),
                    "samples_neg": len(e["neg"]),
                }
            )
    dims.sort(key=lambda d: abs(d.get("gap") or 0), reverse=True)
    return {
        "decisions_total": len(rows),
        "positive": total_pos,
        "negative": total_neg,
        "days": days,
        "dimensions": dims,
    }
