"""硬性门槛 Gate(T12):先判"能不能做",再评"做得好不好"。

两段式(移植 resume-matcher gate.ts):

1. :func:`derive_gate_profile` — 每岗位惰性派生一次,缓存到 ``Job.gate_profile``;
   一次 LLM 调用同时产出**门槛题**(二值 YES/NO)+**评分权重 emphasis**。
   JD 内容指纹变化自动重派生。
2. :func:`run_gate` — 逐候选小调用:PASS / FAIL / UNCERTAIN + 原文证据;
   纯函数 :func:`evaluate_gate` 裁决,``FAIL ∧ severity≥BLOCKING_SEVERITY`` 才淘汰。

设计细节(刻意为之):
- severity/why **不给答题模型看** —— 让模型看到 FAIL 的后果会诱导软化高
  严重度判定,判定与后果分离;
- UNCERTAIN 永不淘汰 —— 初筛误拒损失数月价值,误纳只浪费一次面试,recall 优先;
- 淘汰者 score=NULL 不是 0 —— "被拒"和"零分"是两种语义;
- 派生/判定调用失败都 fail-open(继续评分)+ 响亮告警 —— 与"跑过且全过"可区分。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.domain.models import Job
from app.integrations.llm.provider import LLMError, Message, call_llm_json
from app.services.prompting import DATA_LABEL, INJECTION_GUARD

logger = logging.getLogger(__name__)

#: FMEA severity ≥ 此值的 FAIL 才阻断候选人;以下是"强烈偏好"仅记录。
BLOCKING_SEVERITY = 7

#: 每岗位门槛题上限(同 source 的 MAX_GATE_QUESTIONS)。
MAX_GATE_QUESTIONS = 5

#: 评分维度固定集(权重由 emphasis 按 JD 覆盖,缺失回落默认)。
CRITERIA: list[dict[str, Any]] = [
    {"key": "skills", "name": "关键技能匹配", "default_weight": 40},
    {"key": "experience", "name": "经验与年限", "default_weight": 25},
    {"key": "project_depth", "name": "项目深度与业务规模", "default_weight": 15},
    {"key": "domain", "name": "行业/领域契合", "default_weight": 10},
    {"key": "stability", "name": "履历稳定性与成长", "default_weight": 10},
]

#: level=0 且权重 ≥ 此值 → major 红旗;≥15 → minor。
RED_FLAG_MAJOR_W = 30
RED_FLAG_MINOR_W = 15


def _jd_hash(job: Job) -> str:
    """JD 内容指纹 — 内容变 → 缓存失效自动重派生。"""
    blob = json.dumps(
        {
            "t": job.title,
            "l": job.level,
            "s": sorted(job.skills or []),
            "d": (job.description or "")[:2000],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _normalize_profile(raw: dict) -> dict:
    """派生输出归一化:同 id 只留首个(防 advisory 题继承 blocking severity)、
    severity 钳 1-10、题量上限、权重钳 5-100 回落默认。"""
    seen: set[str] = set()
    questions: list[dict] = []
    for q in (raw.get("questions") or [])[: MAX_GATE_QUESTIONS * 2]:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "").strip()[:64]
        text = str(q.get("question") or "").strip()[:200]
        if not qid or not text or qid in seen:
            continue
        seen.add(qid)
        questions.append(
            {
                "id": qid,
                "question": text,
                "severity": _clamp_int(q.get("severity"), 1, 10, 5),
                "why": str(q.get("why") or "").strip()[:300],
            }
        )
        if len(questions) >= MAX_GATE_QUESTIONS:
            break

    raw_w = raw.get("emphasis") if isinstance(raw.get("emphasis"), dict) else {}
    emphasis: dict[str, int] = {}
    for c in CRITERIA:
        w = _clamp_int(raw_w.get(c["key"]), 5, 100, c["default_weight"])
        emphasis[c["key"]] = w

    return {
        "questions": questions,
        "emphasis": emphasis,
        "derived_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def derive_gate_profile(db: Session, job: Job, *, force: bool = False) -> dict:
    """取(或惰性派生)岗位筛选画像:门槛题 + 评分权重。

    缓存键 ``jd_hash``;LLM 失败 → 返回空画像(questions=[])并记 warning,
    调用方继续评分(fail-open)。
    """
    h = _jd_hash(job)
    cached = job.gate_profile or {}
    if not force and cached.get("jd_hash") == h and "questions" in cached:
        return cached

    sys_prompt = (
        "你是资深招聘顾问,从岗位 JD 提取**硬性筛选门槛**与评分权重。\n"
        "\n"
        "**只输出 JSON**:\n"
        '  {"questions": [\n'
        '     {"id": "kebab-case 标识",\n'
        '      "question": "二值问题,YES 表示候选人可接受",\n'
        '      "severity": 1-10,\n'
        '      "why": "JD 中支持该门槛的原文片段"}],\n'
        '   "emphasis": {"skills": 5-50, "experience": 5-50, "project_depth": 5-50,\n'
        '                "domain": 5-50, "stability": 5-50}}\n'
        "\n"
        "门槛题规则:\n"
        "- 2~5 个,每个都必须能只凭简历回答 YES/NO\n"
        "- 只收硬约束:工作地点/到岗方式、工作许可、法定执业证照、"
        "JD 明确写'必须'的不可协商项\n"
        "- 严禁分级技能:'有 React 经验'是评分项不是门槛;"
        "'能在德国合法工作'才是门槛\n"
        "- severity: 10=完全不可能录用;7-9=JD 明示硬门槛;4-6=强烈偏好;"
        "1-3=加分项。只有 ≥7 会淘汰,留给真正的硬约束\n"
        "- JD 没有硬约束时,只给 1 个最站得住的 severity≤4 的题,"
        "不要为凑数编造门槛\n"
        "- emphasis 表示该岗位各维度的相对权重(技能/经验/项目/领域/稳定性)\n"
        "禁止输出 JSON 之外的任何文字。"
    )
    skills_label = "、".join(job.skills or []) or "(未填)"
    user = (
        f"岗位: {job.title}\n等级: {job.level}\n关键技能: {skills_label}\n"
        f"JD: {(job.description or '')[:3000]}\n"
    )
    try:
        raw = call_llm_json(
            "match.gate_derive",
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            db=db,
            tenant_id=job.tenant_id,
            required=("questions",),
            temperature=0.2,
            prompt_key="match.gate_derive",
        )
        profile = _normalize_profile(raw)
    except LLMError as e:
        logger.warning("门槛画像派生失败 job=%s,fail-open: %s", job.id, e)
        profile = _normalize_profile({"questions": [], "emphasis": {}})
    profile["jd_hash"] = h
    job.gate_profile = profile
    db.commit()
    return profile


def evaluate_gate(questions: list[dict], answers: list[dict]) -> dict:
    """纯裁决:FAIL ∧ severity≥BLOCKING → disqualifier;UNCERTAIN/缺答放行。

    同 id 重复作答只记一次(去重,防同一问题被读成两个独立缺陷)。
    """
    by_id = {q["id"]: q for q in questions}
    disqualifiers: list[dict] = []
    flagged: set[str] = set()
    norm_answers: list[dict] = []

    for a in answers or []:
        if not isinstance(a, dict):
            continue
        q = by_id.get(str(a.get("id") or ""))
        if q is None:
            continue
        verdict = str(a.get("verdict") or "UNCERTAIN").upper()
        if verdict not in ("PASS", "FAIL", "UNCERTAIN"):
            verdict = "UNCERTAIN"
        norm_answers.append(
            {
                "id": q["id"],
                "question": q["question"],
                "verdict": verdict,
                "evidence": str(a.get("evidence") or "未提及")[:400],
            }
        )
        if verdict != "FAIL" or q["severity"] < BLOCKING_SEVERITY:
            continue
        if q["id"] in flagged:
            continue
        flagged.add(q["id"])
        disqualifiers.append(
            {
                "id": q["id"],
                "question": q["question"],
                "severity": q["severity"],
                "evidence": norm_answers[-1]["evidence"],
            }
        )

    return {
        "passed": not disqualifiers,
        "answers": norm_answers,
        "disqualifiers": disqualifiers,
    }


def run_gate(
    db: Session,
    *,
    resume_text: str,
    questions: list[dict],
    tenant_id: str,
) -> dict:
    """逐候选门槛判定。无题 → fail-open;调用失败 → fail-open(gate=None 语义由调用方处理)。

    返回 ``{passed, answers, disqualifiers}`` 或失败时 ``{"passed": True, "skipped": "error"}``。
    """
    if not questions:
        return {"passed": True, "answers": [], "disqualifiers": [], "skipped": "no_questions"}

    sys_prompt = (
        "你是招聘初筛员,只回答门槛问题,不做整体评价。\n"
        "对每个问题给出判定:\n"
        "  PASS      = 简历显示候选人**满足**该约束\n"
        "  FAIL      = 简历显示候选人**不满足**该约束\n"
        "  UNCERTAIN = 简历没有提及(绝不猜;没写就答 UNCERTAIN,不答 FAIL)\n"
        "每个判定引用简历原文作证据,未提及写 \"未提及\"。每题恰答一次,用给定 id。\n"
        f"{INJECTION_GUARD}\n"
        "**只输出 JSON**:"
        '{"answers": [{"id": "...", "verdict": "PASS|FAIL|UNCERTAIN", "evidence": "..."}]}'
    )
    # severity/why 故意不下发 —— 判定与后果分离
    qs = [{"id": q["id"], "question": q["question"]} for q in questions]
    user = (
        f"门槛问题:\n{json.dumps(qs, ensure_ascii=False, indent=1)}\n\n"
        f"候选人简历摘录{DATA_LABEL}:\n{resume_text[:3000]}\n"
    )
    try:
        raw = call_llm_json(
            "match.gate_answer",
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            db=db,
            tenant_id=tenant_id,
            required=("answers",),
            temperature=0.0,
            prompt_key="match.gate_answer",
        )
    except LLMError as e:
        logger.warning("门槛判定调用失败,fail-open: %s", e)
        return {"passed": True, "answers": [], "disqualifiers": [], "skipped": "error"}
    return evaluate_gate(questions, raw.get("answers") or [])


def emphasis_of(profile: dict) -> dict[str, int]:
    """画像里的评分权重(缺失维度回落默认)。"""
    out: dict[str, int] = {}
    em = profile.get("emphasis") or {}
    for c in CRITERIA:
        out[c["key"]] = _clamp_int(em.get(c["key"]), 5, 100, c["default_weight"])
    return out


def aggregate_score(
    raw_criteria: dict[str, Any], emphasis: dict[str, int]
) -> tuple[int, list[dict], dict]:
    """纯函数:模型输出的逐维 {evidence, level} → 加权百分制 + 红旗。

    ``raw_criteria`` = ``{key: {"evidence": str, "level": 0-4}}``。
    返回 ``(score, criteria_rows, red_flags)``;rows 含 key/name/weight/
    level/evidence 供落库与展示;red_flags = ``{"major": [...], "minor": [...]}``
    (level=0 且权重 ≥30 major,≥15 minor)。
    """
    total_w = 0
    acc = 0.0
    rows: list[dict] = []
    major: list[str] = []
    minor: list[str] = []
    for c in CRITERIA:
        key = c["key"]
        cell = raw_criteria.get(key) if isinstance(raw_criteria.get(key), dict) else {}
        level = _clamp_int(cell.get("level"), 0, 4, 0)
        evidence = str(cell.get("evidence") or "未提及")[:400]
        w = emphasis.get(key, c["default_weight"])
        acc += round(level / 4 * 100) * w
        total_w += w
        rows.append(
            {"key": key, "name": c["name"], "weight": w, "level": level, "evidence": evidence}
        )
        if level == 0 and w >= RED_FLAG_MAJOR_W:
            major.append(c["name"])
        elif level == 0 and w >= RED_FLAG_MINOR_W:
            minor.append(c["name"])
    score = round(acc / total_w) if total_w else 0
    return score, rows, {"major": major, "minor": minor}
