"""T16 人才库扫描(sourcing)—— 画像驱动的跨岗位候选人筛选。

流程(零 LLM 预筛 → LLM 精评 topK,成本控制):

    候选人最新已解析简历
      → gate-lite:min_years(结构化/文本年限)+ education(学历等级)
      → 预筛分 = 0.6*coverage.score + 0.4*keywords 命中率 → 取 top K
      → LLM 逐人精评(score + highlights + gaps,fail-open 回落预筛分)
      → 排序返回 + SourcingRun 留痕 + profile use_count/last_used_at
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import Candidate, Resume, SourcingRun, TalentProfile
from app.services.keyword_coverage import compute_coverage
from app.services.scoring_rules import _structured_work_years, estimate_experience_years
from app.services.text_match import keyword_in_text

logger = logging.getLogger(__name__)

PREFILTER_TOPK = 20
_EDU_ORDER = {"高中": 0, "大专": 1, "本科": 2, "学士": 2, "硕士": 3, "MBA": 3, "博士": 4}


def normalize_spec(raw: dict[str, Any] | None) -> dict[str, Any]:
    """画像 spec 归一化:{skills[], keywords[], nice_to_have[], min_years, education}"""
    raw = raw or {}

    def _list(k: str, cap: int = 30) -> list[str]:
        out = []
        for v in raw.get(k) or []:
            s = str(v).strip()
            if s and s not in out:
                out.append(s[:60])
        return out[:cap]

    try:
        min_years = max(0.0, float(raw.get("min_years") or 0))
    except (TypeError, ValueError):
        min_years = 0.0
    return {
        "skills": _list("skills", 20),
        "keywords": _list("keywords", 20),
        "nice_to_have": _list("nice_to_have"),
        "min_years": min_years,
        "education": str(raw.get("education") or "").strip()[:20],
    }


def _resume_years(r: Resume) -> float:
    yrs = _structured_work_years(r)
    if yrs is not None:
        return yrs
    return estimate_experience_years(r.parsed_text or "")


def _resume_edu_level(r: Resume) -> int:
    """简历最高学历等级(结构化 educations 优先,全文关键词兜底)。"""
    secs = ((r.parsed_data or {}).get("sections") or {})
    best = -1
    for e in secs.get("educations") or []:
        deg = str(e.get("degree") or "")
        for k, lv in _EDU_ORDER.items():
            if k in deg:
                best = max(best, lv)
    if best < 0:
        text = r.parsed_text or ""
        for k, lv in sorted(_EDU_ORDER.items(), key=lambda x: -x[1]):
            if k in text:
                return lv
    return best


def _gate_lite(r: Resume, spec: dict[str, Any]) -> str | None:
    """返回淘汰原因;通过返回 None。"""
    req_years = float(spec.get("min_years") or 0)
    if req_years > 0 and _resume_years(r) < req_years:
        return "years"
    req_edu = _EDU_ORDER.get(str(spec.get("education") or ""), -1)
    if req_edu >= 0 and _resume_edu_level(r) < req_edu:
        return "education"
    return None


def _prefilter_score(r: Resume, spec: dict[str, Any]) -> tuple[float, dict, float]:
    cov = compute_coverage(r, spec.get("skills") or [])
    cov_score = (cov or {}).get("score", 0)
    kws = spec.get("keywords") or []
    text = " ".join([
        r.parsed_text or "",
        " ".join((r.parsed_data or {}).get("skills") or []),
    ])
    kw_pct = (
        sum(1 for k in kws if keyword_in_text(k, text)) / len(kws) * 100 if kws else 0.0
    )
    # 无关键词时权重全给 coverage;两者皆无(空 spec)给中间分 50
    if kws:
        score = 0.6 * cov_score + 0.4 * kw_pct
    elif (cov or {}).get("total"):
        score = float(cov_score)
    else:
        score = 50.0
    return score, (cov or {"score": 0, "hit": [], "missing": [], "total": 0}), kw_pct


def _candidate_resumes(db: Session, tenant_id: str) -> list[tuple[Candidate, Resume]]:
    """每位候选人取最新一份已解析简历。"""
    cands = list(
        db.scalars(
            select(Candidate).where(
                Candidate.tenant_id == tenant_id,
                Candidate.is_blacklisted.is_(False),
            )
        ).all()
    )
    out: list[tuple[Candidate, Resume]] = []
    for c in cands:
        r = db.scalars(
            select(Resume)
            .where(
                Resume.tenant_id == tenant_id,
                Resume.candidate_id == c.id,
                Resume.parse_status == "done",
            )
            .order_by(Resume.created_at.desc())
        ).first()
        if r:
            out.append((c, r))
    return out


def _llm_score(
    db: Session, tenant_id: str, spec: dict[str, Any],
    cand: Candidate, r: Resume,
) -> dict[str, Any] | None:
    """单候选精评;LLM 失败返回 None(调用方回落预筛分)。"""
    from app.integrations.llm.provider import LLMError, call_llm_json

    sys_p = (
        "你是人才库筛选评估器。按画像 spec 评估候选人,只输出 JSON:\n"
        '{"score": 0-100, "highlights": ["核心匹配点≤3条,每条≤20字"], '
        '"gaps": ["主要缺口≤2条"]}\n'
        "评分口径:技能覆盖为主,经验年限与领域契合为辅;证据不足的维度给中位分。\n"
        "禁止 JSON 之外的文字。"
    )
    user = (
        f"画像 spec: {spec}\n"
        f"候选人: {cand.name}\n"
        f"简历技能: {(r.parsed_data or {}).get('skills') or []}\n"
        f"简历文本(截断): {(r.parsed_text or '')[:2000]}"
    )
    try:
        raw = call_llm_json(
            "sourcing.score",
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user}],
            db=db, tenant_id=tenant_id,
            required=("score",), temperature=0.2,
            prompt_key="sourcing.score",
        )
        return {
            "score": max(0, min(100, int(raw.get("score") or 50))),
            "highlights": [str(h)[:60] for h in (raw.get("highlights") or [])][:3],
            "gaps": [str(g)[:60] for g in (raw.get("gaps") or [])][:2],
        }
    except LLMError as e:
        logger.warning("sourcing 精评失败 cand=%s: %s", cand.id, e)
        return None


def scan(
    db: Session,
    *,
    tenant_id: str,
    spec: dict[str, Any],
    profile: TalentProfile | None = None,
    limit: int = 5,
    llm_top_k: int = 10,
    created_by: str | None = None,
) -> dict[str, Any]:
    """人才库扫描 → 排序短名单。返回 {items, run, spec}。"""
    t0 = time.time()
    spec = normalize_spec(spec)
    rows = _candidate_resumes(db, tenant_id)

    survivors: list[dict[str, Any]] = []
    for cand, r in rows:
        if _gate_lite(r, spec):
            continue
        pre, cov, kw_pct = _prefilter_score(r, spec)
        survivors.append({
            "candidate": cand, "resume": r,
            "pre_score": pre, "coverage": cov, "kw_pct": kw_pct,
        })
    survivors.sort(key=lambda x: -x["pre_score"])
    top = survivors[:PREFILTER_TOPK]

    llm_n = 0
    for item in top[:llm_top_k]:
        res = _llm_score(db, tenant_id, spec, item["candidate"], item["resume"])
        llm_n += 1
        if res:
            item["final"] = res["score"]
            item["highlights"] = res["highlights"]
            item["gaps"] = res["gaps"]
        else:
            item["final"] = round(item["pre_score"])
            item["highlights"] = _fallback_highlights(item)
            item["gaps"] = [f"缺少 {m}" for m in (item["coverage"].get("missing") or [])][:2]
    rest = top[llm_top_k:]
    for item in rest:
        item["final"] = round(item["pre_score"])
        item["highlights"] = _fallback_highlights(item)
        item["gaps"] = [f"缺少 {m}" for m in (item["coverage"].get("missing") or [])][:2]

    top.sort(key=lambda x: -x["final"])
    items = [{
        "candidate_id": it["candidate"].id,
        "name": it["candidate"].name,
        "resume_id": it["resume"].id,
        "score": it["final"],
        "coverage": it["coverage"],
        "years": round(_resume_years(it["resume"]), 1),
        "education": _edu_label(_resume_edu_level(it["resume"])),
        "skills": ((it["resume"].parsed_data or {}).get("skills") or [])[:8],
        "highlights": it["highlights"],
        "gaps": it["gaps"],
    } for it in top[:limit]]

    run = SourcingRun(
        tenant_id=tenant_id,
        profile_id=profile.id if profile else None,
        spec_snapshot=spec,
        scanned_total=len(rows),
        prefiltered=len(survivors),
        llm_scored=llm_n,
        returned=len(items),
        duration_ms=int((time.time() - t0) * 1000),
        created_by=created_by,
    )
    db.add(run)
    if profile is not None:
        profile.use_count = (profile.use_count or 0) + 1
        profile.last_used_at = datetime.utcnow()
    db.commit()

    return {"items": items, "run_id": run.id, "spec": spec,
            "scanned": len(rows), "prefiltered": len(survivors)}


def _fallback_highlights(item: dict[str, Any]) -> list[str]:
    hits = (item["coverage"].get("hit") or [])[:3]
    yrs = _resume_years(item["resume"])
    out = [f"具备 {h} 经验" for h in hits]
    if yrs > 0:
        out.append(f"约{round(yrs, 1)}年经验")
    return (out or ["画像部分维度可评"])[:3]


def _edu_label(lv: int) -> str:
    return {-1: "", 0: "高中", 1: "大专", 2: "本科", 3: "硕士", 4: "博士"}.get(lv, "")


def generate_profile_draft(db: Session, tenant_id: str, prompt: str) -> dict[str, Any]:
    """一句话 JD → AI 起草画像 spec;LLM 失败回落关键词启发式。"""
    from app.integrations.llm.provider import LLMError, call_llm_json

    sys_p = (
        "你是人才画像设计器。按招聘意图输出画像 spec,只输出 JSON:\n"
        '{"name": "画像名≤30字", "skills": ["必备技能≤10"], '
        '"keywords": ["加分关键词≤10"], "nice_to_have": ["加分项≤5"], '
        '"min_years": 数字, "education": "本科|硕士|博士|大专|高中|空串"}\n'
        "技能词取 JD 里的核心技术词;无明确年限要求给 0。禁止 JSON 之外的文字。"
    )
    try:
        raw = call_llm_json(
            "talent_profile.generate",
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": f"招聘意图: {prompt[:1500]}"}],
            db=db, tenant_id=tenant_id,
            required=("skills",), temperature=0.3,
            prompt_key="talent_profile.generate",
        )
        return {
            "name": str(raw.get("name") or prompt[:30])[:200],
            "spec": normalize_spec(raw),
        }
    except LLMError as e:
        logger.warning("画像生成失败,启发式兜底: %s", e)
        words = [w for w in (prompt or "").replace(",", " ").split() if len(w) >= 2][:10]
        return {"name": prompt[:30] or "新画像",
                "spec": normalize_spec({"skills": words, "keywords": []})}
