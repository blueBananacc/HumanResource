"""简历↔岗位 LLM 匹配评估服务(T12 两段式)。

外部接口:``evaluate_match(db, resume, job) -> dict``。

流水线(移植 resume-matcher screenCandidate:Gate first, score second):

1. **门槛 Gate** — :mod:`app.services.match_gate`:按岗位惰性派生的二值
   门槛题逐候选判定;``FAIL ∧ severity≥7`` → 直接淘汰,score=NULL
   ("被拒"≠"零分"),不再花评分调用。
2. **证据先行评分** — 逐维 ``{evidence: 原文引用, level: 0-4 锚定级}``,
   代码侧按 JD 权重加权聚合出 0-100 + 红旗;模型旧格式输出(无
   criteria)自动兼容,用其直接给的 score。

LLM 失败时给确定性兜底(基于 skill 重叠度),保证产品体验不全断。
任何上游 LLMError 在本函数内吞掉,worker 不会走 failed 状态。

输入:简历 ``parsed_text``(前 3000 字)+ 岗位 title/level/skills/description
+ T11 岗位知识源上下文。
输出 dict:
- ``score`` int 0-100,gated 时为 ``None``
- ``strengths`` / ``gaps`` list[str] ≤5
- ``comment`` str
- ``gate`` dict | None — {passed, answers, disqualifiers}
- ``criteria`` list | None — [{key,name,weight,level,evidence}]
- ``red_flags`` dict | None — {"major": [...], "minor": [...]}
- ``gated`` bool — 被门槛淘汰
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.domain.models import Job, Resume
from app.integrations.llm.provider import LLMError, Message, call_llm_json
from app.services import match_gate
from app.services.prompting import DATA_LABEL, guarded, render_prompt

logger = logging.getLogger(__name__)


def _resume_excerpt(resume: Resume) -> str:
    """简历正文给 LLM(前 3000 字,与 :mod:`question_generator` 共用切片大小)。"""
    if resume.parsed_text:
        return resume.parsed_text[:3000]
    pd = resume.parsed_data or {}
    skills = pd.get("skills") or []
    return (
        f"(简历正文为空,只能依据元数据)\n"
        f"姓名: {pd.get('name_hint') or '(未识别)'}\n"
        f"邮箱: {pd.get('email') or '(未识别)'}\n"
        f"技能: {', '.join(skills) if skills else '(未识别)'}\n"
    )


def _system_prompt() -> str:
    return (
        "你是高级招聘评估师,做简历与岗位的**匹配评估**。\n"
        "对每个评估维度:**先引用简历原文作证据,再按锚定量表定级**。\n"
        "\n"
        "**只输出 JSON**,字段:\n"
        '  {"criteria": {\n'
        '     "skills":         {"evidence": "简历原文引用或\\"未提及\\"", "level": 0-4},\n'
        '     "experience":     {"evidence": "...", "level": 0-4},\n'
        '     "project_depth":  {"evidence": "...", "level": 0-4},\n'
        '     "domain":         {"evidence": "...", "level": 0-4},\n'
        '     "stability":      {"evidence": "...", "level": 0-4}},\n'
        '   "strengths": ["匹配亮点 1", ...],   // <=5 条, 每条 <=30 字\n'
        '   "gaps":      ["关键短板 1", ...],   // <=5 条\n'
        '   "comment":   "总结 (<=120 字)"}\n'
        "\n"
        "锚定量表(评证据强度,不评感觉):\n"
        "  4=有直接、具体、可验证的证据,完全满足或超出要求\n"
        "  3=有明确证据接近匹配,仅小缺口\n"
        "  2=有主张但未证实(如只写'熟悉'却无项目支撑)\n"
        "  1=仅弱相关/擦边证据\n"
        "  0=简历未提及,或明确违反岗位约束\n"
        "\n"
        "评估原则:\n"
        "- 简历没写的一律 evidence=\"未提及\" + level=0,不猜不补\n"
        "- 岗位禁止项在简历中明确出现时,该维度 level=0(负面匹配)\n"
        "- 不评判候选人种族 / 性别 / 婚育 / 户籍 / 学历背景\n"
        "禁止 JSON 之外的任何文字。"
    )


def _user_message(resume: Resume, job: Job, knowledge: str = "") -> str:
    skills_label = "、".join(job.skills) if job.skills else "(未填)"
    return (
        f"岗位: {job.title}\n"
        f"等级: {job.level}\n"
        f"关键技能: {skills_label}\n"
        f"JD: {(job.description or '')[:800]}\n"
        + (f"{knowledge}\n" if knowledge else "")
        + f"\n"
        f"候选人简历摘录{DATA_LABEL}:\n"
        f"{_resume_excerpt(resume)}\n"
        f"\n"
        f"请按要求输出 JSON。"
    )


def _clamp_score(v: Any) -> int:
    try:
        n = int(v)
    except (ValueError, TypeError):
        n = 60
    return max(0, min(100, n))


def _normalize_str_list(raw: Any, *, item_max: int = 80, max_items: int = 5) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()[:item_max]
        if s:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def evaluate_match(
    db: Session,
    *,
    resume: Resume,
    job: Job,
) -> dict[str, Any]:
    """单对 (resume, job) 评估,返回结构化结果(见模块 docstring)。

    Gate 先跑:被硬约束淘汰 → ``gated=True, score=None``;
    评分失败给 skills 重叠兜底,**不抛**到调用方。
    """
    # T11:岗位知识源(ready 状态)注入评估上下文;无源时为空串,零回归
    from app.services.job_knowledge import knowledge_context_text

    knowledge = knowledge_context_text(
        db, job, query=f"{job.title} {' '.join(job.skills or [])}", limit=4
    )
    excerpt = _resume_excerpt(resume)

    # T15:确定性关键词覆盖子分(零 LLM)—— gated/scored/兜底路径都附
    from app.services.keyword_coverage import compute_coverage

    coverage = None
    try:
        coverage = compute_coverage(resume, job.skills or [])
    except Exception as e:  # noqa: BLE001
        logger.warning("coverage 计算失败忽略 job=%s: %s", job.id, e)

    # ---- 1) 门槛 Gate:能派生出画像才判;任何异常 fail-open 进评分 ----
    gate_result: dict[str, Any] | None = None
    try:
        profile = match_gate.derive_gate_profile(db, job)
        questions = profile.get("questions") or []
        emphasis = match_gate.emphasis_of(profile)
        if questions:
            gate_result = match_gate.run_gate(
                db, resume_text=excerpt, questions=questions,
                tenant_id=job.tenant_id,
            )
        else:
            gate_result = {"passed": True, "answers": [], "disqualifiers": [],
                           "skipped": "no_questions"}
    except Exception as e:  # noqa: BLE001 — 画像/判定任何异常都不阻断评分
        logger.warning("Gate 链路异常 fail-open job=%s: %s", job.id, e)
        emphasis = {c["key"]: c["default_weight"] for c in match_gate.CRITERIA}

    if gate_result and not gate_result.get("passed"):
        return {
            "score": None,
            "strengths": [],
            "gaps": [f"门槛未过: {d['question']}" for d in gate_result["disqualifiers"]],
            "comment": "硬性门槛不满足,未进入评分。",
            "gate": gate_result,
            "criteria": None,
            "red_flags": None,
            "rule_snapshot": None,
            "coverage": coverage,
            "gated": True,
        }

    # ---- 2a) T13 声明式评分规则:岗位挂了规则走规则引擎 ----
    if job.scoring_rule_id:
        from app.domain.models import ScoringRule
        from app.services import scoring_rules as sr

        rule = db.get(ScoringRule, job.scoring_rule_id)
        if rule and rule.tenant_id == job.tenant_id:
            try:
                result = sr.evaluate_with_rule(db, resume=resume, job=job, rule=rule)
                result.update({"gate": gate_result, "coverage": coverage, "gated": False})
                return result
            except Exception as e:  # noqa: BLE001 — 规则路径整体异常回落 T12 路径
                logger.warning(
                    "规则引擎异常,回落五维路径 rule=%s job=%s: %s",
                    rule.id, job.id, e,
                )

    # ---- 2) 证据先行评分 ----
    messages: list[Message] = [
        {
            "role": "system",
            "content": guarded(
                render_prompt(db, job.tenant_id, "match.evaluate")
                or _system_prompt()
            ),
        },
        {"role": "user", "content": _user_message(resume, job, knowledge)},
    ]
    try:
        parsed = call_llm_json(
            "match.evaluate",
            messages,
            db=db,
            tenant_id=job.tenant_id,
            required=(),
            temperature=0.2,
            prompt_key="match.evaluate",
        )
    except LLMError as e:
        logger.warning("LLM 匹配评估失败,使用兜底: %s", e)
        fb = _fallback_match(resume, job)
        fb.update({"gate": gate_result, "criteria": None,
                   "red_flags": None, "rule_snapshot": None,
                   "coverage": coverage, "gated": False})
        return fb

    strengths = _normalize_str_list(parsed.get("strengths"))
    gaps = _normalize_str_list(parsed.get("gaps"))
    comment = str(parsed.get("comment") or "").strip()[:200]

    # 新格式:逐维 evidence+level → 代码侧加权;旧格式:用模型直给的 score
    criteria_rows = None
    red_flags = None
    if isinstance(parsed.get("criteria"), dict) and parsed["criteria"]:
        score, criteria_rows, red_flags = match_gate.aggregate_score(
            parsed["criteria"], emphasis
        )
    else:
        score = _clamp_score(parsed.get("score", 60))
    if not comment:
        comment = _comment_for_score(score)

    return {
        "score": score,
        "strengths": strengths,
        "gaps": gaps,
        "comment": comment,
        "gate": gate_result,
        "criteria": criteria_rows,
        "red_flags": red_flags,
        "rule_snapshot": None,
        "coverage": coverage,
        "gated": False,
    }


def _comment_for_score(score: int) -> str:
    if score >= 80:
        return "高度匹配,建议优先安排面试。"
    if score >= 65:
        return "基本符合,可纳入候选池进一步评估。"
    if score >= 50:
        return "部分匹配,需结合 JD 重点核对短板。"
    return "匹配度较低,建议跳过或换岗位评估。"


def _fallback_match(resume: Resume, job: Job) -> dict[str, Any]:
    """LLM 不可达 / 解析失败时的确定性兜底:基于技能词重叠度估分。

    粗粒度:岗位 skills 在简历正文 / parsed_data.skills 里命中比例。
    无新依赖,纯字符串包含。
    """
    job_skills = [s.strip() for s in (job.skills or []) if s and s.strip()]
    if not job_skills:
        # 岗位没填技能,无从匹配 — 给中性分
        return {
            "score": 60,
            "strengths": [],
            "gaps": ["岗位未填关键技能,匹配度无法精确评估"],
            "comment": _comment_for_score(60),
        }

    from app.services.keyword_coverage import compute_coverage

    cov = compute_coverage(resume, job_skills)
    hits = (cov or {}).get("hit") or []
    miss = (cov or {}).get("missing") or []
    overlap = len(hits) / max(1, len(job_skills))
    # 简单线性映射:0% → 40 分,100% → 90 分
    score = int(round(40 + overlap * 50))
    return {
        "score": score,
        "strengths": [f"命中: {s}" for s in hits[:5]],
        "gaps": [f"缺少: {s}" for s in miss[:5]],
        "comment": _comment_for_score(score),
    }
