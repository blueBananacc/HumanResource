"""结构化面试评分卡服务(T9)。

与 AI ``Interview`` 互补:Interview 是机器面试会话,评分卡承载**人类
面试官**的逐维评分与录用表态。criteria 存 JSONB [{key,name,dimension,
weight}]——key 创建时生成,反馈按 key 引用,改显示名不断链。
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    CandidateSkill,
    Employee,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    JobSkill,
    Notification,
    Skill,
    User,
)


def user_display_names(
    db: Session, tenant_id: str, user_ids: list[str]
) -> dict[str, str]:
    """user_id → 展示名:绑定员工档案的姓名优先,否则邮箱。"""
    ids = [i for i in dict.fromkeys(user_ids) if i]
    if not ids:
        return {}
    users = {
        u.id: u
        for u in db.scalars(select(User).where(User.id.in_(ids))).all()
    }
    emps = {
        e.user_id: e.name
        for e in db.scalars(
            select(Employee).where(
                Employee.tenant_id == tenant_id,
                Employee.user_id.in_(ids),
            )
        ).all()
        if e.user_id
    }
    return {
        uid: emps.get(uid) or (users[uid].email if uid in users else "?")
        for uid in ids
    }

logger = logging.getLogger(__name__)

DIMENSIONS = ("tech", "soft", "general")
VERDICTS = ("recommend", "hold", "reject")
MIN_RATING, MAX_RATING = 1, 5


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def normalize_criteria(raw: list | None) -> list[dict[str, Any]]:
    """入参 [{name,dimension?,weight?,key?}] → 规范 [{key,name,dimension,weight}]。

    PATCH 路径允许带 key 保留引用;新条目补生成 key。空名/未知 dimension 拒绝。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        name = str((item or {}).get("name") or "").strip()[:64]
        if not name:
            raise ValueError("考察维度 name 不能为空")
        dimension = str((item or {}).get("dimension") or "general").strip()
        if dimension not in DIMENSIONS:
            raise ValueError(f"dimension 必须是 {DIMENSIONS} 之一")
        try:
            weight = int((item or {}).get("weight", 1))
        except (TypeError, ValueError):
            raise ValueError("weight 必须是整数") from None
        if not 1 <= weight <= 10:
            raise ValueError("weight 取值 1-10")
        key = str((item or {}).get("key") or uuid.uuid4().hex[:8])
        if key in seen:
            raise ValueError("criteria 存在重复 key")
        seen.add(key)
        out.append(
            {"key": key, "name": name, "dimension": dimension, "weight": weight}
        )
    return out


def validate_scores(
    criteria: list[dict[str, Any]], raw: list | None
) -> list[dict[str, Any]]:
    """校验逐维评分:criterion 必须命中评分卡声明的 key,score ∈ 1-5。"""
    valid_keys = {c["key"] for c in criteria}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        ck = str((item or {}).get("criterion") or "")
        if ck not in valid_keys:
            raise ValueError(f"未知考察维度: {ck or '(空)'}")
        if ck in seen:
            raise ValueError("同一维度重复评分")
        seen.add(ck)
        try:
            score = int((item or {}).get("score"))
        except (TypeError, ValueError):
            raise ValueError("score 必须是整数") from None
        if not MIN_RATING <= score <= MAX_RATING:
            raise ValueError(f"score 取值 {MIN_RATING}-{MAX_RATING}")
        comment = str((item or {}).get("comment") or "").strip()[:500] or None
        out.append({"criterion": ck, "score": score, "comment": comment})
    return out


def aggregate(
    scorecard: InterviewScorecard, feedbacks: list[InterviewFeedback]
) -> dict[str, Any]:
    """聚合:每维度 均分/极值/分歧,总均分,表态计数,分歧清单。

    分歧标记:任一维度跨面试官极差 >= 2,或表态同时出现 recommend 与 reject。
    """
    crit = {c["key"]: c for c in (scorecard.criteria or [])}
    per_crit: dict[str, list[int]] = {k: [] for k in crit}
    ratings: list[int] = []
    tally = {v: 0 for v in VERDICTS}
    for fb in feedbacks:
        ratings.append(fb.rating)
        if fb.verdict in tally:
            tally[fb.verdict] += 1
        for s in fb.scores or []:
            k = s.get("criterion")
            if k in per_crit and isinstance(s.get("score"), int):
                per_crit[k].append(s["score"])

    criteria_out = []
    disagreements: list[dict[str, Any]] = []
    for k, c in crit.items():
        vals = per_crit[k]
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0
        avg = round(sum(vals) / len(vals), 2) if vals else None
        criteria_out.append(
            {
                "key": k,
                "name": c["name"],
                "dimension": c["dimension"],
                "weight": c["weight"],
                "avg": avg,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "count": len(vals),
            }
        )
        if spread >= 2:
            disagreements.append(
                {"kind": "criterion", "name": c["name"], "spread": spread}
            )
    if tally["recommend"] and tally["reject"]:
        disagreements.append(
            {
                "kind": "verdict",
                "name": "录用表态分歧",
                "spread": tally["recommend"] + tally["reject"],
            }
        )
    return {
        "feedback_count": len(feedbacks),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "verdict_tally": tally,
        "criteria": criteria_out,
        "disagreements": disagreements,
    }


def submit_feedback(
    db: Session,
    *,
    scorecard: InterviewScorecard,
    interviewer_user_id: str,
    rating: int,
    verdict: str,
    notes: str | None,
    scores: list[dict[str, Any]],
) -> InterviewFeedback:
    """upsert:同一面试官重复提交 = 更新自己的卡。"""
    fb = db.scalars(
        select(InterviewFeedback).where(
            InterviewFeedback.scorecard_id == scorecard.id,
            InterviewFeedback.interviewer_user_id == interviewer_user_id,
        )
    ).first()
    if fb is None:
        fb = InterviewFeedback(
            tenant_id=scorecard.tenant_id,
            scorecard_id=scorecard.id,
            interviewer_user_id=interviewer_user_id,
            rating=rating,
            verdict=verdict,
            scores=scores,
            notes=notes,
            submitted_at=_utcnow(),
        )
        db.add(fb)
    else:
        fb.rating = rating
        fb.verdict = verdict
        fb.scores = scores
        fb.notes = notes
        fb.submitted_at = _utcnow()
    return fb


# ---------------------------------------------------------------------------
# 面评催办(北森"催促面试评价"同款)
# ---------------------------------------------------------------------------

# 面试结束(scheduled_at)后超过该时长仍未提交反馈 → 视为逾期,可催办
OVERDUE_HOURS = 24
# 同一面试官对同一卡:距上次催办不足该时长不重复发(防轰炸)
NUDGE_COOLDOWN_HOURS = 20


def pending_interviewers(
    db: Session, sc: InterviewScorecard
) -> list[str]:
    """已指派但尚未提交反馈的面试官 user_id。"""
    submitted = set(
        db.scalars(
            select(InterviewFeedback.interviewer_user_id).where(
                InterviewFeedback.scorecard_id == sc.id
            )
        ).all()
    )
    return [u for u in (sc.interviewer_ids or []) if u not in submitted]


def scorecard_overdue(sc: InterviewScorecard, now: datetime | None = None) -> bool:
    """面试已过 N 小时且卡仍 open — 反馈进入逾期态。"""
    if sc.status != "open" or sc.scheduled_at is None:
        return False
    now = now or _utcnow()
    sched = sc.scheduled_at
    if sched.tzinfo is not None:
        sched = sched.replace(tzinfo=None)
    return (now - sched).total_seconds() > OVERDUE_HOURS * 3600


def _nudge_link(sc: InterviewScorecard) -> str:
    return f"/interviews/workbench#nudge:{sc.id}"


def notify_assigned(
    db: Session, sc: InterviewScorecard, user_ids: list[str]
) -> int:
    """面试官被指派时的任务通知(北森:任务含简历/评估上下文)。"""
    if not user_ids:
        return 0
    cand = db.get(Candidate, sc.candidate_id)
    when = (
        sc.scheduled_at.strftime("%m-%d %H:%M") if sc.scheduled_at else "时间待定"
    )
    n = 0
    for uid in user_ids:
        db.add(
            Notification(
                tenant_id=sc.tenant_id,
                user_id=uid,
                kind="scorecard.assign",
                title="你收到一场面试评价任务",
                body=f"「{sc.title}」候选人 {cand.name if cand else '?'} · {when}",
                link=f"/interviews/workbench#assign:{sc.id}",
            )
        )
        n += 1
    return n


def _recently_nudged(
    db: Session, user_id: str, sc: InterviewScorecard, now: datetime
) -> bool:
    """cooldown 内已对该用户催过这张卡。"""
    from datetime import timedelta

    cutoff = now - timedelta(hours=NUDGE_COOLDOWN_HOURS)
    return (
        db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == user_id,
                Notification.kind == "scorecard.nudge",
                Notification.link == _nudge_link(sc),
                Notification.created_at >= cutoff,
            )
        ).scalar_one()
        > 0
    )


def nudge_scorecard(
    db: Session, sc: InterviewScorecard, *, now: datetime | None = None
) -> list[str]:
    """对该卡所有未交反馈的面试官发催办通知(cooldown 去重)。

    返回实际发出通知的 user_id 列表。
    """
    now = now or _utcnow()
    cand = db.get(Candidate, sc.candidate_id)
    sent: list[str] = []
    for uid in pending_interviewers(db, sc):
        if _recently_nudged(db, uid, sc, now):
            continue
        db.add(
            Notification(
                tenant_id=sc.tenant_id,
                user_id=uid,
                kind="scorecard.nudge",
                title="面试反馈待提交",
                body=(
                    f"「{sc.title}」候选人 {cand.name if cand else '?'}"
                    " 的面试已结束,请尽快填写评分卡"
                ),
                link=_nudge_link(sc),
            )
        )
        sent.append(uid)
    return sent


def run_scorecard_nudges(db: Session, *, now: datetime | None = None) -> int:
    """全租户自动催办扫描:_watch_loop 每 tick 调用。

    条件:评分卡 open + scheduled_at 已过 OVERDUE_HOURS + 面试官未交。
    通知按 cooldown 去重,天然幂等。返回发出数。
    """
    now = now or _utcnow()
    cards = db.scalars(
        select(InterviewScorecard).where(InterviewScorecard.status == "open")
    ).all()
    sent = 0
    for sc in cards:
        if not scorecard_overdue(sc, now):
            continue
        sent += len(nudge_scorecard(db, sc, now=now))
    if sent:
        db.commit()
    return sent


def clear_nudges_on_feedback(
    db: Session, sc: InterviewScorecard, user_id: str
) -> int:
    """面试官提交反馈后,把该卡的催办通知标记已读(ack)。"""
    rows = db.scalars(
        select(Notification).where(
            Notification.user_id == user_id,
            Notification.kind.in_(("scorecard.nudge", "scorecard.assign")),
            Notification.link.like(f"%{sc.id}%"),
            Notification.read_at.is_(None),
        )
    ).all()
    now = _utcnow()
    for n in rows:
        n.read_at = now
    return len(rows)


# ---------------------------------------------------------------------------
# AI:维度建议 / 聚合分析
# ---------------------------------------------------------------------------


def _render(key: str, db: Session, tenant_id: str, module: str, **vars_) -> str:
    from app.services.prompting import (
        PROMPT_REGISTRY, guarded, render_prompt, render_vars,
    )

    prompt = render_prompt(db, tenant_id, key, module=module, **vars_)
    if prompt is None:
        prompt = render_vars(PROMPT_REGISTRY[key].content, vars_)
    # 上下文含候选人资料/面试官评语 — 不可信内容,注入防御
    return guarded(prompt)


def suggest_criteria(
    db: Session, tenant_id: str, job: Job | None, candidate: Candidate | None
) -> list[dict[str, Any]]:
    """按岗位技能+JD 建议考察维度。**只回草稿不落库**;LLM 失败返回 []。"""
    from app.infra.license.state import is_feature_enabled
    from app.integrations.llm.provider import LLMError, call_llm_json

    if not is_feature_enabled(db, "assistant.agent"):
        return []
    ctx: dict[str, Any] = {}
    if job is not None:
        skills = db.execute(
            select(Skill.name, Skill.category, JobSkill.importance)
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .where(JobSkill.job_id == job.id, JobSkill.tenant_id == tenant_id)
        ).all()
        ctx["job"] = {
            "title": job.title,
            "level": job.level,
            "description": (job.description or "")[:1500],
            "skills": [
                {"name": n, "category": c, "importance": i} for n, c, i in skills
            ],
        }
    if candidate is not None:
        cskills = db.execute(
            select(Skill.name, CandidateSkill.level)
            .join(CandidateSkill, CandidateSkill.skill_id == Skill.id)
            .where(CandidateSkill.candidate_id == candidate.id)
            .limit(20)
        ).all()
        ctx["candidate"] = {
            "name": candidate.name,
            "skills": [{"name": n, "level": lv} for n, lv in cskills],
        }
    vars_ = {"context": json.dumps(ctx, ensure_ascii=False, default=str)[:6000]}
    try:
        data = call_llm_json(
            "scorecard.suggest",
            [{"role": "system", "content": _render(
                "scorecard.criteria", db, tenant_id, "interviews", **vars_
            )}],
            db=db, tenant_id=tenant_id,
            prompt_key="scorecard.criteria",
        )
        raw = data.get("criteria") or []
        out = []
        for item in raw[:12]:
            name = str((item or {}).get("name") or "").strip()[:64]
            if not name:
                continue
            dim = str((item or {}).get("dimension") or "general")
            out.append(
                {
                    "name": name,
                    "dimension": dim if dim in DIMENSIONS else "general",
                    "weight": 1,
                }
            )
        return out
    except (LLMError, Exception) as e:  # noqa: BLE001 — 建议降级为空,不炸表单
        logger.warning("scorecard criteria 建议失败: %s", e)
        return []


def synthesize(
    db: Session, scorecard: InterviewScorecard, feedbacks: list[InterviewFeedback]
) -> dict[str, Any]:
    """LLM 聚合全部反馈 → ai_summary dict。LLM 不可用时回落机械聚合文本。"""
    from app.integrations.llm.provider import LLMError, call_llm_json

    names = user_display_names(
        db, scorecard.tenant_id, [f.interviewer_user_id for f in feedbacks]
    )
    crit = {c["key"]: c["name"] for c in (scorecard.criteria or [])}
    ctx = {
        "title": scorecard.title,
        "criteria": [
            {"name": c["name"], "dimension": c["dimension"]}
            for c in (scorecard.criteria or [])
        ],
        "feedbacks": [
            {
                "interviewer": names.get(f.interviewer_user_id, "?"),
                "rating": f.rating,
                "verdict": f.verdict,
                "notes": (f.notes or "")[:800],
                "scores": [
                    {
                        "criterion": crit.get(s.get("criterion"), s.get("criterion")),
                        "score": s.get("score"),
                        "comment": (s.get("comment") or "")[:200],
                    }
                    for s in (f.scores or [])
                ],
            }
            for f in feedbacks
        ],
    }
    agg = aggregate(scorecard, feedbacks)
    ctx["aggregate"] = agg
    # 回落默认文案:LLM 不可用/返回空结果时仍给出可用摘要
    tally = agg["verdict_tally"]
    fallback = (
        f"共 {len(feedbacks)} 位面试官提交反馈,总均分 "
        f"{agg['avg_rating'] if agg['avg_rating'] is not None else '—'}/5;"
        f"表态:推荐 {tally['recommend']}、保留 {tally['hold']}、"
        f"不推荐 {tally['reject']}。"
    )
    if agg["disagreements"]:
        fallback += "存在评分分歧,建议评审会讨论。"
    summary: dict[str, Any] = {
        "summary": fallback,
        "themes": [],
        "disagreements": [d.get("name") for d in agg["disagreements"]],
        "recommendation": "",
        "feedback_count": len(feedbacks),
        "generated_at": _utcnow().isoformat(),
    }
    try:
        data = call_llm_json(
            "scorecard.synthesize",
            [{"role": "system", "content": _render(
                "scorecard.synthesize",
                db, scorecard.tenant_id, "interviews",
                context=json.dumps(ctx, ensure_ascii=False, default=str)[:10000],
            )}],
            db=db, tenant_id=scorecard.tenant_id,
            prompt_key="scorecard.synthesize",
        )
        if data.get("summary"):
            summary["summary"] = str(data["summary"])[:2000]
        summary["themes"] = [str(t)[:120] for t in (data.get("themes") or [])][:8]
        ds = data.get("disagreements")
        if isinstance(ds, list) and ds:
            summary["disagreements"] = [str(d)[:120] for d in ds][:8]
        summary["recommendation"] = str(data.get("recommendation") or "")[:500]
    except (LLMError, Exception) as e:  # noqa: BLE001 — 降级机械聚合
        logger.warning("scorecard synthesize LLM 失败,回落机械聚合: %s", e)
    return summary
