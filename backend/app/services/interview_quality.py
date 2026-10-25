"""组织级面试质量聚合(T17)— 纯确定性,零 LLM。

数据源:
- AI 面试 ``Interview.summary.dimension_scores {维度:分}`` + ``recommendation``
- 人类评分卡 ``InterviewScorecard.criteria [{key,name,dimension}]`` +
  ``InterviewFeedback.scores [{criterion,score,comment}]`` / ``verdict``

口径(全部可解释):
- **评估事件** = 窗口内 ``status='done'`` 且 ``mode!='self_test'`` 的 AI 面试,
  ∪ 窗口内有反馈提交的独立评分卡(未挂面试的人工轮次)。
- **维度全集** = 事件内 AI dimension_scores keys ∪ 相关评分卡 criteria[].name,
  经 ``normalize_skill_key`` 归一化合并近名维度。
- **单维 coverage** = 该维被实际评到的事件数 / 评估事件总数。
- **维度覆盖率 KPI** = Σ各维命中 / (|维度全集| × 事件数) —— 该考的维度里考了多少。
- **通过率** = 结论为推荐的事件 / 有结论的事件(AI: recommendation 含 hire/推荐
  且不含"不推荐";人工: feedback verdict=recommend,hold/reject 计入分母)。
- **missed** = coverage < 50 的维度 —— "经常被遗漏的考察项"。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Interview,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    User,
)
from app.services.permissions import apply_org_filter, get_org_scope
from app.services.text_match import normalize_skill_key

_MISSED_THRESHOLD = 50


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _rec_positive(rec: object) -> bool | None:
    """summary.recommendation → True=推荐 / False=不推荐 / None=无结论。"""
    s = str(rec or "").strip().lower()
    if not s:
        return None
    if "不推荐" in s or "reject" in s:
        return False
    if "推荐" in s or "hire" in s or "strong" in s:
        return True
    return None


def _pct(num: int, den: int) -> int:
    return round(num * 100 / den) if den else 0


def _delta(cur: int, prev: int) -> str | None:
    if prev == 0:
        return None
    d = round((cur - prev) * 100 / prev)
    return f"+{d}%" if d >= 0 else f"{d}%"


def build_quality_report(db: Session, user: User, days: int = 30) -> dict:
    """窗口内组织面试质量报告;days 自由参数 clamp 到 [1,365]。"""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 30
    days = max(1, min(days, 365))
    now = _now()
    since = now - timedelta(days=days)
    prev_start = since - timedelta(days=days)
    scope = get_org_scope(db, user)
    tid = user.tenant_id

    def _iv_stmt(start: datetime, end: datetime | None):
        stmt = (
            select(Interview)
            .join(Candidate, Candidate.id == Interview.candidate_id)
            .where(
                Interview.tenant_id == tid,
                Interview.status == "done",
                Interview.mode != "self_test",
                Interview.finished_at >= start,
            )
        )
        if end is not None:
            stmt = stmt.where(Interview.finished_at < end)
        return apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)

    ivs = list(db.scalars(_iv_stmt(since, None)).all())
    prev_total = len(db.scalars(_iv_stmt(prev_start, since)).all())
    total = len(ivs)

    # 维度归一 → 首次出现的原始名做展示名
    dim_display: dict[str, str] = {}

    def _reg_dim(name: str) -> str:
        norm = normalize_skill_key(name)
        if norm and norm not in dim_display:
            dim_display[norm] = name.strip()
        return norm

    # ---- 评估事件:AI 面试 + 独立评分卡反馈 ----
    # events: {key: {"dims": set, "outcome": bool|None, "job_id": str|None, "mins": float|None}}
    events: dict[str, dict] = {}
    for iv in ivs:
        dims = set()
        scores = (iv.summary or {}).get("dimension_scores") or {}
        for k in scores:
            if str(k).strip():
                dims.add(_reg_dim(str(k)))
        events[f"iv:{iv.id}"] = {
            "dims": dims,
            "outcome": _rec_positive((iv.summary or {}).get("recommendation")),
            "job_id": iv.job_id,
            "mins": (
                (iv.finished_at - iv.started_at).total_seconds() / 60
                if iv.started_at and iv.finished_at
                else None
            ),
        }

    # 窗口内评分卡(挂面试的 + 独立的;criteria 计划维度进全集)
    cards = list(
        db.scalars(
            select(InterviewScorecard).where(InterviewScorecard.tenant_id == tid)
        ).all()
    )
    card_by_id = {c.id: c for c in cards}
    # criteria key → 归一维度名(评分卡内 criterion 反馈按 key 引用)
    key_to_dim: dict[tuple[str, str], str] = {}

    feedbacks = list(
        db.scalars(
            select(InterviewFeedback).where(
                InterviewFeedback.tenant_id == tid,
                InterviewFeedback.submitted_at >= since,
            )
        ).all()
    )
    # 有反馈的独立评分卡 → 评估事件;挂面试卡片的反馈并入该面试事件
    fb_by_card: dict[str, list[InterviewFeedback]] = {}
    for fb in feedbacks:
        fb_by_card.setdefault(fb.scorecard_id, []).append(fb)

    for card in cards:
        crit = {c.get("key"): _reg_dim(str(c.get("name") or c.get("dimension") or ""))
                for c in (card.criteria or []) if isinstance(c, dict)}
        for k, v in crit.items():
            if k:
                key_to_dim[(card.id, str(k))] = v
        fbs = fb_by_card.get(card.id) or []
        if card.interview_id and f"iv:{card.interview_id}" in events:
            ev = events[f"iv:{card.interview_id}"]
        elif fbs:
            # 独立评分卡反馈 = 一个评估事件(反馈在窗口内才算)
            if not any(f.submitted_at >= since for f in fbs):
                continue
            ev = events.setdefault(f"sc:{card.id}", {
                "dims": set(), "outcome": None, "job_id": card.job_id, "mins": None,
            })
        else:
            continue
        for fb in fbs:
            for s in fb.scores or []:
                if not isinstance(s, dict):
                    continue
                dim = key_to_dim.get((card.id, str(s.get("criterion") or "")))
                if dim:
                    ev["dims"].add(dim)
            if fb.verdict in ("recommend", "reject") and ev["outcome"] is None:
                ev["outcome"] = fb.verdict == "recommend"
        # hold 表态计入"有结论"但不计通过 —— 用单独标志
        if fbs and ev.get("_decided") is None:
            ev["_decided"] = any(f.verdict in ("recommend", "reject", "hold") for f in fbs)

    # ---- 维度聚合 ----
    # 维度全集:事件命中维度 ∪ 窗口相关评分卡 criteria 名
    universe: set[str] = set(dim_display.keys()) | {
        d for ev in events.values() for d in ev["dims"]
    }
    n_events = len(events)
    dim_rows = []
    total_hits = 0
    # 单维均分:AI dimension_scores(原分)+ feedback score×2 对齐 10 分制
    dim_scores: dict[str, list[float]] = {}
    for iv in ivs:
        for k, v in ((iv.summary or {}).get("dimension_scores") or {}).items():
            norm = normalize_skill_key(str(k))
            try:
                dim_scores.setdefault(norm, []).append(float(v))
            except (TypeError, ValueError):
                pass
    for fb in feedbacks:
        card = card_by_id.get(fb.scorecard_id)
        if not card:
            continue
        for s in fb.scores or []:
            if not isinstance(s, dict):
                continue
            dim = key_to_dim.get((card.id, str(s.get("criterion") or "")))
            try:
                if dim and s.get("score") is not None:
                    dim_scores.setdefault(dim, []).append(float(s["score"]) * 2)
            except (TypeError, ValueError):
                pass

    for norm in sorted(universe):
        hits = sum(1 for ev in events.values() if norm in ev["dims"])
        total_hits += hits
        scores = dim_scores.get(norm) or []
        dim_rows.append({
            "name": dim_display.get(norm) or norm,
            "coverage": _pct(hits, n_events),
            "scored_in": hits,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        })
    dim_rows.sort(key=lambda r: r["coverage"], reverse=True)

    # ---- KPI ----
    decided = [ev for ev in events.values() if ev["outcome"] is not None or ev.get("_decided")]
    passed = sum(1 for ev in decided if ev["outcome"] is True)
    mins = [ev["mins"] for ev in events.values() if ev["mins"] is not None]
    kpis = {
        "interviews_total": total,
        "interviews_delta": _delta(total, prev_total),
        "dimension_coverage": _pct(total_hits, len(universe) * n_events),
        "pass_rate": _pct(passed, len(decided)),
        "avg_duration_min": round(sum(mins) / len(mins)) if mins else None,
        "assessed_events": n_events,
        "decided": len(decided),
    }

    # ---- 岗位下钻 ----
    job_names = {j.id: j.title for j in db.scalars(
        select(Job).where(Job.tenant_id == tid)
    ).all()}
    by_job: dict[str, dict] = {}
    for ev in events.values():
        jid = ev["job_id"]
        if not jid:
            continue
        b = by_job.setdefault(jid, {"job_id": jid, "title": job_names.get(jid, ""),
                                    "interviews": 0, "decided": 0, "passed": 0})
        b["interviews"] += 1
        if ev["outcome"] is not None or ev.get("_decided"):
            b["decided"] += 1
            b["passed"] += 1 if ev["outcome"] is True else 0
    by_job_rows = [
        {**b, "pass_rate": _pct(b["passed"], b["decided"])}
        for b in sorted(by_job.values(), key=lambda x: -x["interviews"])
    ]
    for b in by_job_rows:
        b.pop("decided", None)
        b.pop("passed", None)

    return {
        "days": days,
        "generated_at": now.isoformat(),
        "kpis": kpis,
        "dimensions": dim_rows,
        "missed": [d for d in dim_rows if d["coverage"] < _MISSED_THRESHOLD],
        "by_job": by_job_rows,
    }
