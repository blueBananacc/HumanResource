"""相似候选人召回(iCIMS Talent Match / silver medalist)。

iCIMS Similarity = skills match + experience match(对 model candidate);
我们在此之上加可选的语义 embedding 融合与「银牌候选人」标记:

    model 候选人
      ├─ 技能集:CandidateSkill(level 加权)∪ parsed_data.skills
      ├─ 经验画像:最近职位 title(work_experiences[0])+ 总年限 + 学历等级
      ↓
    每个有已解析简历的在库候选人:
      skill_score = 加权命中率(精确 skill_id 优先,关键词兜底,CJK 感知)
      exp_score   = 0.55×职位相似 + 0.45×年限接近
      final       = 0.6×skill + 0.4×exp;有 embedding 时 0.8×final + 0.2×cos
      silver      = 曾走到 interview/offer 后被拒/撤回 → +5 分召回加成

全部确定性信号优先;embedding 仅当租户配了 embedding model 时参与,
任何失败 fail-open 不影响结果。
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    CandidateSkill,
    Job,
    JobApplication,
    JobApplicationEvent,
    Resume,
    Skill,
)
from app.services.scoring_rules import _structured_work_years, estimate_experience_years
from app.services.sourcing import _EDU_ORDER
from app.services.text_match import keyword_in_text, normalize_skill_key

logger = logging.getLogger(__name__)

_SILVER_DEEP = {"interview", "offer"}
_SILVER_END = {"rejected", "withdrawn"}

# 职位名归一化时的噪声词(中英文都剥掉,只留职能核心)
_TITLE_NOISE = re.compile(
    r"(高级|资深|初级|中级|senior|junior|sr\.?|jr\.?|lead|principal|staff|实习|intern)",
    re.IGNORECASE,
)


def _latest_resume(db: Session, tenant_id: str, candidate_id: str) -> Resume | None:
    return db.scalars(
        select(Resume)
        .where(
            Resume.tenant_id == tenant_id,
            Resume.candidate_id == candidate_id,
            Resume.parse_status == "done",
        )
        .order_by(Resume.created_at.desc())
    ).first()


def _corpus(r: Resume | None) -> str:
    """打分语料:全文 + 抽取技能 + 技能段文本。"""
    if r is None:
        return ""
    pd = r.parsed_data or {}
    parts = [r.parsed_text or ""]
    skills = pd.get("skills") or []
    if skills:
        parts.append(" ".join(str(s) for s in skills))
    stext = ((pd.get("sections") or {}).get("skills_text") or "")
    if stext:
        parts.append(stext)
    return " ".join(parts)


def _work_years(r: Resume | None) -> float:
    if r is None:
        return 0.0
    yrs = _structured_work_years(r)
    if yrs is not None:
        return yrs
    return estimate_experience_years(r.parsed_text or "")


def _edu_level(r: Resume | None) -> int:
    if r is None:
        return -1
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


def _latest_title(r: Resume | None) -> str:
    """最近职位名:结构化 work_experiences 里 date_range 含「至今」或排最前的。"""
    if r is None:
        return ""
    works = ((r.parsed_data or {}).get("sections") or {}).get("work_experiences") or []
    if not works:
        return ""
    current = next(
        (w for w in works if re.search(r"至今|今|present|now", str(w.get("date_range") or ""), re.I)),
        works[0],
    )
    return str(current.get("title") or "")


def _norm_title(t: str) -> str:
    t = _TITLE_NOISE.sub(" ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _title_sim(a: str, b: str) -> float:
    """职位名相似度:整串命中 > 共享职能词;无数据给中性 50。"""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 50.0
    if na == nb:
        return 100.0
    if keyword_in_text(na, nb) or keyword_in_text(nb, na):
        return 90.0
    # token Jaccard(拉丁词 + CJK 连续段)
    def _toks(s: str) -> set[str]:
        words = set(re.findall(r"[a-z][a-z0-9+#.]*", s))
        words.update(re.findall(r"[一-鿿]{2,}", s))
        return {w for w in words if len(w) >= 2}

    ta, tb = _toks(na), _toks(nb)
    if not ta or not tb:
        return 50.0
    jac = len(ta & tb) / len(ta | tb)
    return round(jac * 100, 1)


def _candidate_skills(db: Session, tenant_id: str, candidate_ids: list[str]) -> dict[str, dict[str, float]]:
    """{candidate_id: {skill_id: level}}——精确技能锚点。"""
    if not candidate_ids:
        return {}
    rows = db.execute(
        select(CandidateSkill.candidate_id, CandidateSkill.skill_id, CandidateSkill.level)
        .where(
            CandidateSkill.tenant_id == tenant_id,
            CandidateSkill.candidate_id.in_(candidate_ids),
        )
    ).all()
    out: dict[str, dict[str, float]] = {}
    for cid, sid, lv in rows:
        out.setdefault(cid, {})[sid] = float(lv or 3)
    return out


def _model_skill_list(
    db: Session, tenant_id: str, model: Candidate, resume: Resume | None
) -> list[dict[str, Any]]:
    """model 技能清单:[{skill_id?, name, weight}];CandidateSkill level 加权 + 解析技能。"""
    rows = db.execute(
        select(CandidateSkill.skill_id, CandidateSkill.level, Skill.name)
        .join(Skill, Skill.id == CandidateSkill.skill_id)
        .where(
            CandidateSkill.tenant_id == tenant_id,
            CandidateSkill.candidate_id == model.id,
        )
    ).all()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sid, lv, name in rows:
        key = normalize_skill_key(name or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"skill_id": sid, "name": name, "weight": max(1.0, float(lv or 3))})
    for s in (resume.parsed_data or {}).get("skills") or [] if resume else []:
        name = str(s).strip()
        key = normalize_skill_key(name)
        if not name or key in seen:
            continue
        seen.add(key)
        out.append({"skill_id": None, "name": name, "weight": 2.5})
    return out[:30]


def _silver_map(db: Session, tenant_id: str, candidate_ids: list[str]) -> dict[str, str]:
    """{candidate_id: job_title}——曾深入流程(≥interview)后被拒/撤回的银牌候选人。"""
    if not candidate_ids:
        return {}
    apps = db.execute(
        select(JobApplication.id, JobApplication.candidate_id, Job.title)
        .join(Job, Job.id == JobApplication.job_id)
        .where(
            JobApplication.tenant_id == tenant_id,
            JobApplication.candidate_id.in_(candidate_ids),
            JobApplication.stage.in_(_SILVER_END),
        )
    ).all()
    if not apps:
        return {}
    app_ids = [a.id for a in apps]
    deep = set(
        db.scalars(
            select(JobApplicationEvent.application_id).where(
                JobApplicationEvent.tenant_id == tenant_id,
                JobApplicationEvent.application_id.in_(app_ids),
                JobApplicationEvent.to_stage.in_(_SILVER_DEEP),
            )
        ).all()
    )
    return {a.candidate_id: a.title for a in apps if a.id in deep}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_scores(db: Session, tenant_id: str, corpus: list[str]) -> list[float] | None:
    """cos×100 列表;embedding 不可用/失败 → None。corpus[0] 是 model。"""
    try:
        from app.services.embedding import embed_texts

        vecs = embed_texts(db, tenant_id, corpus)
        if not vecs or len(vecs) != len(corpus):
            return None
        return [_cos(vecs[0], v) * 100 for v in vecs[1:]]
    except Exception as e:  # noqa: BLE001
        logger.warning("[similar] embedding fail-open: %s", e)
        return None


def similar_candidates(
    db: Session,
    *,
    tenant_id: str,
    model: Candidate,
    pool: list[Candidate],
    limit: int = 10,
    min_score: int = 0,
) -> dict[str, Any]:
    """model 候选人 vs pool(已过权限过滤)→ 相似榜。纯读、无持久化。"""
    m_resume = _latest_resume(db, tenant_id, model.id)
    m_skills = _model_skill_list(db, tenant_id, model, m_resume)
    m_title = _latest_title(m_resume)
    m_years = _work_years(m_resume)
    m_edu = _edu_level(m_resume)

    cand_ids = [c.id for c in pool]
    resumes = {c.id: _latest_resume(db, tenant_id, c.id) for c in pool}
    resumes = {k: v for k, v in resumes.items() if v is not None}
    skill_map = _candidate_skills(db, tenant_id, cand_ids)
    silvers = _silver_map(db, tenant_id, cand_ids)

    scored: list[dict[str, Any]] = []
    for c in pool:
        if c.id == model.id or c.id not in resumes:
            continue
        r = resumes[c.id]
        corpus = _corpus(r)
        t_skills = skill_map.get(c.id, {})

        # ---- skills match(加权命中)----
        hit_names: list[str] = []
        w_hit = w_all = 0.0
        for s in m_skills:
            w_all += s["weight"]
            if s["skill_id"] and s["skill_id"] in t_skills:
                w_hit += s["weight"]
                hit_names.append(s["name"])
            elif keyword_in_text(s["name"], corpus):
                w_hit += s["weight"] * 0.85  # 关键词命中略降权
                hit_names.append(s["name"])
        skill_score = (w_hit / w_all * 100) if w_all else 50.0

        # ---- experience match ----
        t_title = _latest_title(r)
        t_years = _work_years(r)
        t_edu = _edu_level(r)
        tsim = _title_sim(m_title, t_title)
        if m_years > 0 and t_years > 0:
            ysim = max(0.0, 100.0 - 15.0 * abs(m_years - t_years))
        else:
            ysim = 50.0
        edu_sim = 100.0 - 25.0 * abs(m_edu - t_edu) if (m_edu >= 0 and t_edu >= 0) else 75.0
        exp_score = 0.45 * tsim + 0.35 * ysim + 0.20 * edu_sim

        silver_job = silvers.get(c.id)
        det = 0.6 * skill_score + 0.4 * exp_score
        if silver_job:
            det += 5.0
        scored.append(
            {
                "candidate": c,
                "resume": r,
                "skill_score": round(skill_score, 1),
                "exp_score": round(exp_score, 1),
                "det": det,
                "shared_skills": hit_names[:8],
                "title": t_title or None,
                "years": round(t_years, 1) if t_years else None,
                "edu_level": t_edu,
                "silver_job": silver_job,
            }
        )

    # ---- 可选语义融合:embedding cos×100 占 20% ----
    emb_corpus = [_corpus(m_resume)[:800]] + [_corpus(it["resume"])[:800] for it in scored]
    emb = _embed_scores(db, tenant_id, emb_corpus) if scored else None
    for i, it in enumerate(scored):
        if emb is not None:
            it["score"] = round(min(100.0, 0.8 * it["det"] + 0.2 * emb[i]))
        else:
            it["score"] = round(min(100.0, it["det"]))
        it.pop("det")

    scored.sort(key=lambda x: -x["score"])
    items = [it for it in scored if it["score"] >= min_score][:limit]
    return {
        "model": {
            "candidate_id": model.id,
            "name": model.name,
            "title": m_title or None,
            "years": round(m_years, 1) if m_years else None,
            "skills": [s["name"] for s in m_skills][:10],
        },
        "items": [
            {
                "candidate_id": it["candidate"].id,
                "name": it["candidate"].name,
                "score": it["score"],
                "skill_score": it["skill_score"],
                "exp_score": it["exp_score"],
                "shared_skills": it["shared_skills"],
                "title": it["title"],
                "years": it["years"],
                "education_level": it["edu_level"],
                "silver": bool(it["silver_job"]),
                "silver_job_title": it["silver_job"],
                "tags": it["candidate"].tags or [],
                "status": it["candidate"].status,
                "display_email": it["candidate"].display_email,
                "resume_id": it["resume"].id,
            }
            for it in items
        ],
        "scanned": len(scored),
        "semantic": emb is not None,
    }
