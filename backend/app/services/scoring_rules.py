"""T13 声明式评分规则引擎。

移植 hr-assistant ``scoring-utils.ts`` / ``scoring-rule-analysis-utils.ts`` 的
核心概念,叠加 HROnfire 的多租户 / LLM 网关留痕 / T12 证据先行体系。

规则 = 维度列表,每维声明一个评估器方法:

- ``keyword``  — 简历文本子串命中(0 token)
- ``duration`` — 文本正则抽取工作年限,线性插值(0 token)
- ``boolean``  — 特征存在性检查(0 token)
- ``range``    — 数值区间线性映射(0 token)
- ``ai``       — 批量 LLM 一次调用,沿用 T12 ``{evidence, level 0-4}``
  证据先行格式(level×25 → 0-100)

评估入口 :func:`evaluate_with_rule` 在 :func:`app.services.match_evaluator
.evaluate_match` 中 Gate 通过后分支调用;确定性维产出统一
``{score, evidence}``,与 ai 维一起加权聚合。规则评估时
:func:`rule_snapshot` 冻结进 match 记录 —— 规则被改/删后历史分数
仍可回溯到当时的规则版本。
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.domain.models import Job, Resume, ScoringRule
from app.integrations.llm.provider import LLMError, Message, call_llm_json
from app.services.prompting import DATA_LABEL, INJECTION_GUARD

logger = logging.getLogger(__name__)

VALID_METHODS = ("keyword", "duration", "ai", "boolean", "range")
VALID_MATCH_MODES = ("any", "all")
BOOLEAN_FIELDS = ("education", "experience", "projects", "skills")
RANGE_FIELDS = ("experience_years",)
MAX_DIMENSIONS = 8
MAX_KEYWORDS = 20

#: 红旗阈值(与 match_gate 同口径):权重 ≥30 且 level=0 → major;≥15 → minor
RED_FLAG_MAJOR_W = 30
RED_FLAG_MINOR_W = 15


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ---------------------------------------------------------------------------
# 规则规范化
# ---------------------------------------------------------------------------


def normalize_dimensions(raw: Any) -> list[dict[str, Any]]:
    """把(用户输入 / LLM 生成的)维度数组归一化为合法 schema。

    - 非法/空维度丢弃,id 去重,总量上限 MAX_DIMENSIONS
    - weight 钳 1-100,**不强制合计=100**(UI 侧提示;聚合按权重归一化)
    - 每维只保留该方法用到的 evaluator 参数
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, d in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()[:60]
        if not name:
            continue
        did = str(d.get("id") or f"dim-{i + 1}").strip()[:40]
        if did in seen:
            continue
        seen.add(did)

        ev = d.get("evaluator") if isinstance(d.get("evaluator"), dict) else {}
        method = str(ev.get("method") or "ai").strip().lower()
        if method not in VALID_METHODS:
            method = "ai"
        evaluator = _normalize_evaluator(method, ev)

        out.append(
            {
                "id": did,
                "name": name,
                "weight": int(_clamp(d.get("weight"), 1, 100, 25)),
                "description": str(d.get("description") or "").strip()[:200],
                "evaluator": evaluator,
            }
        )
        if len(out) >= MAX_DIMENSIONS:
            break
    return out


def _normalize_evaluator(method: str, ev: dict) -> dict:
    """只保留该方法有意义的参数,并做类型/范围钳制。"""
    if method == "keyword":
        kws = [str(k).strip()[:40] for k in (ev.get("keywords") or []) if str(k).strip()][
            :MAX_KEYWORDS
        ]
        mode = str(ev.get("matchMode") or "any")
        return {
            "method": "keyword",
            "keywords": kws,
            "matchMode": mode if mode in VALID_MATCH_MODES else "any",
        }
    if method == "duration":
        min_y = _clamp(ev.get("minYears"), 0, 50, 0)
        pref_y = _clamp(ev.get("preferredYears"), 0, 50, min_y)
        return {"method": "duration", "minYears": min_y, "preferredYears": max(pref_y, min_y)}
    if method == "boolean":
        field = str(ev.get("booleanField") or "skills")
        return {
            "method": "boolean",
            "booleanField": field if field in BOOLEAN_FIELDS else "skills",
        }
    if method == "range":
        field = str(ev.get("field") or "experience_years")
        lo = _clamp(ev.get("minValue"), 0, 100, 0)
        hi = _clamp(ev.get("maxValue"), lo, 100, 100)
        return {
            "method": "range",
            "field": field if field in RANGE_FIELDS else "experience_years",
            "minValue": lo,
            "maxValue": max(hi, lo),
        }
    # ai
    return {
        "method": "ai",
        "aiPrompt": str(ev.get("aiPrompt") or "").strip()[:300],
    }


def weight_total(dimensions: list[dict]) -> int:
    return sum(int(d.get("weight") or 0) for d in dimensions)


def rule_snapshot(rule: ScoringRule) -> dict[str, Any]:
    """评估时冻结的规则快照(深拷贝维度)。"""
    import copy

    return {
        "rule_id": rule.id,
        "name": rule.name,
        "version": rule.version,
        "dimensions": copy.deepcopy(rule.dimensions or []),
        "snapshot_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


# ---------------------------------------------------------------------------
# 简历文本抽取(我方 parsed_data 只有 email/phone/skills —— 确定性评估以
# parsed_text 正则为主)
# ---------------------------------------------------------------------------


def _resume_text(resume: Resume) -> str:
    pd = resume.parsed_data or {}
    parts = [resume.parsed_text or ""]
    skills = pd.get("skills") or []
    if skills:
        parts.append("技能: " + ", ".join(str(s) for s in skills))
    # T14:结构化工作/教育段摘要,给 AI 维更干净的上下文
    secs = pd.get("sections") or {}
    works = secs.get("work_experiences") or []
    if works:
        lines = [
            f"- {w.get('title') or '?'} @ {w.get('company') or '?'}({w.get('date_range') or '?'})"
            for w in works[:8]
        ]
        parts.append("工作经历(结构化):\n" + "\n".join(lines))
    edus = secs.get("educations") or []
    if edus:
        parts.append(
            "教育经历(结构化): "
            + "; ".join(
                f"{e.get('school') or '?'} {e.get('degree') or ''}({e.get('date_range') or '?'})".strip()
                for e in edus[:4]
            )
        )
    return "\n".join(parts)


def _sections(resume: Resume) -> dict:
    return (resume.parsed_data or {}).get("sections") or {}


def _structured_work_years(resume: Resume) -> float | None:
    """T14:优先用结构化工作段求总年限(按段 years 合计,去重为按段已切)。

    返回 None 表示结构化数据不可用,调用方回退全文正则。
    """
    works = _sections(resume).get("work_experiences")
    if not works:
        return None
    total = sum(float(w.get("years") or 0) for w in works)
    # 防御:单段解析误差累加后可能虚高,50 年封顶
    return min(total, 50.0)


def estimate_experience_years(text: str) -> float:
    """从简历正文估计工作年限。

    两个信号取较大值:
    - ``N年`` / ``N+年`` 字面声明(如 '5年经验')取最大
    - ``YYYY[.MM]-(YYYY|至今)`` 日期区间时长合计(去重按起止年,不精确到月)
    """
    years = 0.0
    for m in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*\+?\s*年", text):
        try:
            years = max(years, float(m.group(1)))
        except ValueError:
            continue

    now_y = datetime.now(UTC).year
    total = 0.0
    for m in re.finditer(
        r"(\d{4})(?:[.\-/年]\s*\d{1,2})?\s*(?:-|~|–|—|至|到)\s*(\d{4}|至今|今)", text
    ):
        try:
            start = int(m.group(1))
            end = now_y if m.group(2) in ("至今", "今") else int(m.group(2))
        except ValueError:
            continue
        if 1950 <= start <= now_y and end >= start:
            total += min(end, now_y) - start
    return max(years, total)


# ---------------------------------------------------------------------------
# 确定性评估器(纯函数,产出统一 {score:0-100, evidence})
# ---------------------------------------------------------------------------


def _eval_keyword(text: str, ev: dict) -> dict[str, Any]:
    kws = ev.get("keywords") or []
    if not kws:
        return {"score": 50, "evidence": "未配置关键词"}
    lower = text.lower()
    matched = [k for k in kws if str(k).lower() in lower]
    missing = [k for k in kws if str(k).lower() not in lower]
    if ev.get("matchMode") == "all":
        score = round(len(matched) / len(kws) * 100)
    else:
        score = min(round(len(matched) / len(kws) * 100) + 20, 100)
    parts = []
    if matched:
        parts.append("命中: " + "、".join(matched[:6]))
    if missing:
        parts.append("缺失: " + "、".join(missing[:6]))
    return {"score": score, "evidence": "; ".join(parts) or "无关键词命中"}


def _eval_duration(resume: Resume, text: str, ev: dict) -> dict[str, Any]:
    yrs_struct = _structured_work_years(resume)
    if yrs_struct is not None:
        yrs = yrs_struct
        src = "结构化工作段"
    else:
        yrs = estimate_experience_years(text)
        src = "文本估计"
    min_y = float(ev.get("minYears") or 0)
    pref_y = float(ev.get("preferredYears") or min_y)
    if pref_y <= min_y:
        score = 100 if yrs >= min_y else 0
    elif yrs >= pref_y:
        score = 100
    elif yrs >= min_y:
        score = round(60 + (yrs - min_y) / (pref_y - min_y) * 40)
    else:
        score = round(yrs / min_y * 50) if min_y > 0 else 50
    return {
        "score": int(_clamp(score, 0, 100, 0)),
        "evidence": f"{src}经验 {yrs:.1f} 年(要求 {min_y}-{pref_y} 年)",
    }


_BOOL_PATTERNS = {
    "education": ["本科", "硕士", "博士", "大专", "学历", "MBA", "bachelor", "master"],
    "experience": ["工作", "经历", "任职", "入职", "职位", "岗位"],
    "projects": ["项目", "作品", "案例"],
    "skills": [],  # skills 用 parsed_data 判定
}


def _eval_boolean(resume: Resume, text: str, ev: dict) -> dict[str, Any]:
    field = ev.get("booleanField") or "skills"
    secs = _sections(resume)
    if field == "skills":
        has = bool((resume.parsed_data or {}).get("skills"))
        detail = "parsed_data.skills"
    elif field == "education" and secs.get("educations") is not None:
        has = bool(secs.get("educations"))
        detail = "结构化教育段"
    elif field == "experience" and secs.get("work_experiences") is not None:
        has = bool(secs.get("work_experiences"))
        detail = "结构化工作段"
    elif field == "projects" and secs.get("projects") is not None:
        has = bool(secs.get("projects"))
        detail = "结构化项目段"
    else:
        pats = _BOOL_PATTERNS.get(field) or []
        lower = text.lower()
        has = any(p.lower() in lower for p in pats)
        detail = "简历正文特征词"
    return {
        "score": 100 if has else 0,
        "evidence": f"{field}: {'具备' if has else '未检出'}({detail})",
    }


def _eval_range(resume: Resume, text: str, ev: dict) -> dict[str, Any]:
    # 目前唯一支持的数值字段
    value = _structured_work_years(resume)
    if value is None:
        value = estimate_experience_years(text)
    lo = float(ev.get("minValue") or 0)
    hi = float(ev.get("maxValue") or 100)
    if hi <= lo:
        score = 100 if value >= lo else 0
    else:
        score = round(_clamp((value - lo) / (hi - lo) * 100, 0, 100, 0))
    return {"score": int(score), "evidence": f"数值 {value:.1f} ∈ [{lo}, {hi}]"}


def _eval_deterministic(resume: Resume, text: str, dim: dict) -> dict[str, Any]:
    ev = dim.get("evaluator") or {}
    method = ev.get("method")
    if method == "keyword":
        return _eval_keyword(text, ev)
    if method == "duration":
        return _eval_duration(resume, text, ev)
    if method == "boolean":
        return _eval_boolean(resume, text, ev)
    if method == "range":
        return _eval_range(resume, text, ev)
    return {"score": 50, "evidence": "未知评估器"}


# ---------------------------------------------------------------------------
# AI 维度批量评估(一次调用评所有 ai 维,证据先行)
# ---------------------------------------------------------------------------


def _eval_ai_dimensions(
    db: Session,
    *,
    tenant_id: str,
    job: Job,
    resume_text: str,
    ai_dims: list[dict],
) -> dict[str, dict[str, Any]]:
    """所有 ai 维合并为一次 call_llm_json;返回 {dim_id: {score, evidence}}。

    LLM 失败 → 每维 score=50 + 'AI评估暂不可用'(单维 fail-open,
    不阻断已算出的确定性维)。
    """
    if not ai_dims:
        return {}
    dims_payload = [
        {
            "id": d["id"],
            "name": d["name"],
            "description": d.get("description") or "",
            "rubric": (d.get("evaluator") or {}).get("aiPrompt") or "",
        }
        for d in ai_dims
    ]
    sys_prompt = (
        "你是资深招聘评估师,按给定维度逐一评估候选人。\n"
        "对每个维度:**先引用简历原文作证据,再按 0-4 锚定级定级**。\n"
        "锚定量表:4=有直接可验证证据且超出要求;3=明确证据接近匹配;"
        "2=有主张但未证实;1=仅弱相关;0=未提及或明确违反。\n"
        f"{INJECTION_GUARD}\n"
        "**只输出 JSON**:"
        '{"dims": {"<dim_id>": {"evidence": "简历原文或\\"未提及\\"", "level": 0-4}}}\n'
        "每个给定维度必须恰答一次,用给定 id。"
    )
    user = (
        f"岗位: {job.title}\n等级: {job.level}\n"
        f"关键技能: {'、'.join(job.skills or []) or '(未填)'}\n"
        f"JD: {(job.description or '')[:1200]}\n\n"
        f"评估维度:\n{__import__('json').dumps(dims_payload, ensure_ascii=False, indent=1)}\n\n"
        f"候选人简历摘录{DATA_LABEL}:\n{resume_text[:3000]}\n"
    )
    try:
        raw = call_llm_json(
            "scoring_rule.evaluate",
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            db=db,
            tenant_id=tenant_id,
            required=("dims",),
            temperature=0.2,
            prompt_key="scoring_rule.evaluate",
        )
    except LLMError as e:
        logger.warning("规则 AI 维评估失败 job=%s: %s", job.id, e)
        return {
            d["id"]: {"score": 50, "evidence": "AI评估暂不可用"} for d in ai_dims
        }

    raw_dims = raw.get("dims") if isinstance(raw.get("dims"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for d in ai_dims:
        cell = raw_dims.get(d["id"]) if isinstance(raw_dims.get(d["id"]), dict) else {}
        level = int(_clamp(cell.get("level"), 0, 4, 0))
        evidence = str(cell.get("evidence") or "未提及")[:400]
        out[d["id"]] = {"score": level * 25, "evidence": evidence, "level": level}
    return out


# ---------------------------------------------------------------------------
# 规则评估主入口
# ---------------------------------------------------------------------------


def evaluate_with_rule(
    db: Session,
    *,
    resume: Resume,
    job: Job,
    rule: ScoringRule,
) -> dict[str, Any]:
    """按声明式规则评估候选人。

    返回与 :func:`match_evaluator.evaluate_match` 兼容的 dict:
    ``score / strengths / gaps / comment / criteria / red_flags / rule_snapshot``。
    criteria 行 = ``{key,name,weight,method,level,evidence}``。
    """
    text = _resume_text(resume)
    dimensions = rule.dimensions or []
    ai_dims = [d for d in dimensions if (d.get("evaluator") or {}).get("method") == "ai"]
    ai_results = _eval_ai_dimensions(
        db, tenant_id=job.tenant_id, job=job, resume_text=text, ai_dims=ai_dims
    )

    rows: list[dict[str, Any]] = []
    major: list[str] = []
    minor: list[str] = []
    total_w = 0
    acc = 0.0
    for d in dimensions:
        ev = d.get("evaluator") or {}
        method = ev.get("method") or "ai"
        w = int(d.get("weight") or 0)
        if w <= 0:
            continue
        if method == "ai":
            res = ai_results.get(d["id"]) or {"score": 50, "evidence": "AI评估未返回"}
        else:
            try:
                res = _eval_deterministic(resume, text, d)
            except Exception as e:  # noqa: BLE001 — 单维异常不拖垮整评
                logger.warning("确定性维评估异常 dim=%s: %s", d.get("id"), e)
                res = {"score": 50, "evidence": "评估异常"}
        score = int(_clamp(res.get("score"), 0, 100, 50))
        level = int(res.get("level")) if "level" in res else int(round(score / 25))
        level = int(_clamp(level, 0, 4, 0))
        acc += score * w
        total_w += w
        rows.append(
            {
                "key": d["id"],
                "name": d["name"],
                "weight": w,
                "method": method,
                "level": level,
                "score": score,
                "evidence": str(res.get("evidence") or "")[:400],
            }
        )
        if score == 0 and w >= RED_FLAG_MAJOR_W:
            major.append(d["name"])
        elif score == 0 and w >= RED_FLAG_MINOR_W:
            minor.append(d["name"])

    final = round(acc / total_w) if total_w else 50
    strengths = [f"{r['name']}: {r['evidence']}" for r in rows if r["score"] >= 75][:5]
    gaps = [f"{r['name']}: {r['evidence']}" for r in rows if r["score"] < 50][:5]
    return {
        "score": final,
        "strengths": strengths,
        "gaps": gaps,
        "comment": f"按规则「{rule.name}」v{rule.version} 评估,{total_w} 权重合计 {final} 分。",
        "criteria": rows,
        "red_flags": {"major": major, "minor": minor} if (major or minor) else None,
        "rule_snapshot": rule_snapshot(rule),
    }


# ---------------------------------------------------------------------------
# AI 规则生成 / 规则自分析(移植 scoring-rule-analysis-utils)
# ---------------------------------------------------------------------------


def generate_rule_draft(db: Session, job: Job) -> dict[str, Any]:
    """按岗位 JD 让 LLM 起草规则(维度+评估器+权重),normalize 后返回草案。

    LLM 失败 → 返回基于岗位 skills 的启发式草案(keyword+duration+ai),
    保证"AI 生成"按钮永远有响应。
    """
    sys_prompt = (
        "你是资深招聘策略顾问,根据岗位 JD 设计一套声明式评分规则。\n"
        "**只输出 JSON**:\n"
        '{"name": "规则名", "description": "适用场景",\n'
        ' "dimensions": [{"id": "kebab-case", "name": "维度名", "weight": 1-100,\n'
        '   "description": "维度说明",\n'
        '   "evaluator": {"method": "keyword|duration|ai|boolean|range",\n'
        '     "keywords": ["keyword 法必填"], "matchMode": "any|all",\n'
        '     "minYears": 0, "preferredYears": 0,\n'
        '     "aiPrompt": "ai 法的评估要点", "booleanField": "education|experience|projects|skills",\n'
        '     "field": "experience_years", "minValue": 0, "maxValue": 100}}]}\n'
        "要求:\n"
        "- 4~6 个维度,权重合计尽量=100\n"
        "- 能用确定性方法的关键词/年限/有无字段就不要用 ai 维(ai 维成本最高)\n"
        "- keyword 维的 keywords 从 JD 技能要求里提取\n"
        "- ai 维用于需要语义判断的维度(如项目深度/领域契合),aiPrompt 写清评估要点\n"
        "禁止 JSON 之外的任何文字。"
    )
    user = (
        f"岗位: {job.title}\n等级: {job.level}\n"
        f"关键技能: {'、'.join(job.skills or []) or '(未填)'}\n"
        f"JD: {(job.description or '')[:2500]}\n"
    )
    try:
        raw = call_llm_json(
            "scoring_rule.generate",
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            db=db,
            tenant_id=job.tenant_id,
            required=("dimensions",),
            temperature=0.3,
            prompt_key="scoring_rule.generate",
        )
        dims = normalize_dimensions(raw.get("dimensions"))
        if not dims:
            raise LLMError("empty dimensions")
        return {
            "name": str(raw.get("name") or f"{job.title}评分规则")[:200],
            "description": str(raw.get("description") or "")[:500],
            "dimensions": dims,
        }
    except LLMError as e:
        logger.warning("规则生成失败 job=%s,启发式兜底: %s", job.id, e)
        return _heuristic_draft(job)


def _heuristic_draft(job: Job) -> dict[str, Any]:
    dims: list[dict[str, Any]] = [
        {
            "id": "skills",
            "name": "关键技能匹配",
            "weight": 40,
            "description": "JD 技能要求的关键词命中度",
            "evaluator": {
                "method": "keyword",
                "keywords": [str(s)[:40] for s in (job.skills or [])][:MAX_KEYWORDS],
                "matchMode": "any",
            },
        },
        {
            "id": "experience",
            "name": "工作年限",
            "weight": 30,
            "description": "从简历文本估计的相关经验年限",
            "evaluator": {
                "method": "duration",
                "minYears": {"entry": 0, "intermediate": 2, "advanced": 5}.get(job.level, 2),
                "preferredYears": {"entry": 2, "intermediate": 5, "advanced": 8}.get(job.level, 5),
            },
        },
        {
            "id": "fit",
            "name": "综合契合度",
            "weight": 30,
            "description": "项目深度/领域契合等需语义判断的维度",
            "evaluator": {
                "method": "ai",
                "aiPrompt": "评估项目深度、领域契合与岗位的匹配程度",
            },
        },
    ]
    return {
        "name": f"{job.title}评分规则",
        "description": "启发式兜底草案(LLM 不可用时生成)",
        "dimensions": normalize_dimensions(dims),
    }


def analyze_rule(db: Session, rule: ScoringRule, linked_jobs: list[Job]) -> dict[str, Any]:
    """规则 + 关联岗位 JD → 结构化分析(覆盖缺口/权重建议/可落地草案)。"""
    rule_payload = {
        "name": rule.name,
        "version": rule.version,
        "dimensions": rule.dimensions,
    }
    jobs_payload = [
        {
            "title": j.title,
            "level": j.level,
            "skills": j.skills or [],
            "description": (j.description or "")[:800],
        }
        for j in linked_jobs[:10]
    ]
    sys_prompt = (
        "你是资深招聘策略顾问,审查一套简历评分规则是否合理。\n"
        "**只输出 JSON**:\n"
        '{"summary": "整体判断",\n'
        ' "coverageGaps": ["未被维度覆盖的岗位要求"],\n'
        ' "weightAdjustments": ["权重失衡建议"],\n'
        ' "industrySignals": ["行业门槛信号"],\n'
        ' "recommendedChanges": ["建议动作"],\n'
        ' "proposedRule": {"name": "规则名", "description": "...",\n'
        '   "dimensions": [{"id": "kebab", "name": "...", "weight": 1-100,\n'
        '     "evaluator": {"method": "keyword|duration|ai|boolean|range", ...}}]}}\n'
        "要求:结合关联岗位 JD 给具体建议;proposedRule 必须是可落地的完整草案,"
        "维度总权重尽量=100,确定性方法优先。"
    )
    import json as _json

    user = (
        f"当前规则:\n{_json.dumps(rule_payload, ensure_ascii=False, indent=1)}\n\n"
        f"关联岗位({len(jobs_payload)} 个):\n{_json.dumps(jobs_payload, ensure_ascii=False, indent=1)}\n"
    )
    raw = call_llm_json(
        "scoring_rule.analyze",
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        db=db,
        tenant_id=rule.tenant_id,
        required=("summary", "proposedRule"),
        temperature=0.3,
        prompt_key="scoring_rule.analyze",
    )
    proposed = raw.get("proposedRule") if isinstance(raw.get("proposedRule"), dict) else {}
    return {
        "summary": str(raw.get("summary") or "")[:500],
        "coverageGaps": _str_list(raw.get("coverageGaps")),
        "weightAdjustments": _str_list(raw.get("weightAdjustments")),
        "industrySignals": _str_list(raw.get("industrySignals")),
        "recommendedChanges": _str_list(raw.get("recommendedChanges")),
        "proposedRule": {
            "name": str(proposed.get("name") or rule.name)[:200],
            "description": str(proposed.get("description") or "")[:500],
            "dimensions": normalize_dimensions(proposed.get("dimensions")),
        },
        "analyzed_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def _str_list(raw: Any, max_items: int = 8) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    out = [str(x).strip()[:200] for x in (raw if isinstance(raw, list) else [])]
    return [x for x in out if x][:max_items]


def apply_analysis(rule: ScoringRule, analysis: dict) -> None:
    """把 analyze_rule 返回的 proposedRule 应用到规则(version+1)。

    只覆盖 name/description/dimensions,不动 id/tenant。
    """
    proposed = analysis.get("proposedRule") if isinstance(analysis, dict) else {}
    if not isinstance(proposed, dict):
        return
    dims = normalize_dimensions(proposed.get("dimensions"))
    if dims:
        rule.dimensions = dims
    if proposed.get("name"):
        rule.name = str(proposed["name"])[:200]
    if proposed.get("description"):
        rule.description = str(proposed["description"])[:500]
    rule.version = (rule.version or 1) + 1
    rule.updated_at = datetime.now(UTC).replace(tzinfo=None)
