"""AI 产品化端点(T6):字段级助手 / 记录摘要 / 文档起草。

- ``POST /ai/field-assist``  — 表单字段 ✨按钮:按 key ``field.<target>.<intent>``
  渲染提示词 → LLM → 返回文本(**只回草稿,不落库**,前端填入供人工改)
- ``GET  /ai/summarize/{entity}/{id}`` — 记录 360° 摘要(candidates/employees/cases),
  上下文聚合见 ``app.services.ai_summary``
- ``POST /ai/doc-draft``     — 文档模板 + 记录 → 预填变量 → LLM 补全草稿

权限:按目标独立映射(与 entities.py 同口径);所有调用写 ``ai_call`` 审计。
license 统一要求 ``assistant.agent``。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import DocTemplate, User
from app.infra.db import get_db
from app.integrations.llm.provider import LLMError, call_llm
from app.services.ai_summary import build_summary_context
from app.services.audit import write_audit
from app.services.doc_templates import DOC_SPECS, build_doc_vars
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_MANAGE_PERFORMANCE,
    PERM_VIEW_REPORTS,
    PERM_WRITE_JOBS,
    PERM_WRITE_RESUMES,
    get_org_scope,
    has_permission,
    visible_employee_ids,
)
from app.services.prompting import (
    PROMPT_REGISTRY, guarded, render_prompt, render_vars,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    dependencies=[Depends(require_feature("assistant.agent"))],
)

_MAX_TEXT = 20_000  # LLM 返回草稿的截断上限

# field-assist target → 所需权限(写类字段用写权限,通用用查看权限)
_FIELD_PERMS = {
    "jobs": PERM_WRITE_JOBS,
    "announcements": PERM_MANAGE_EMPLOYEES,
    "cases": PERM_MANAGE_EMPLOYEES,
    "campaigns": PERM_WRITE_RESUMES,
}

_SUMMARY_PERMS = {
    "candidates": PERM_VIEW_REPORTS,
    "employees": PERM_MANAGE_EMPLOYEES,
    "cases": PERM_MANAGE_EMPLOYEES,
}

_SUMMARY_KEYS = {
    "candidates": "summarize.candidate",
    "employees": "summarize.employee",
    "cases": "summarize.case",
}


class FieldAssistIn(BaseModel):
    # target 形如 "jobs.description" / "cases.resolution" / "generic"
    target: str = Field(..., min_length=1, max_length=64)
    intent: str = Field("draft", pattern="^(draft|polish|expand|shorten)$")
    fields: dict[str, str] = Field(default_factory=dict)
    instruction: str = Field("", max_length=1000)
    current_text: str = Field("", max_length=_MAX_TEXT)


class FieldAssistOut(BaseModel):
    text: str
    key: str


class SummarizeOut(BaseModel):
    entity: str
    id: str
    summary: str


class DocTemplateOptionOut(BaseModel):
    key: str
    name: str
    entity: str


class DocDraftIn(BaseModel):
    key: str = Field(..., min_length=1, max_length=64)
    entity: str = Field(..., pattern="^(candidates|employees)$")
    record_id: str = Field(..., min_length=1, max_length=36)
    instruction: str = Field("", max_length=1000)


class DocDraftOut(BaseModel):
    draft: str
    vars: dict[str, str]


def _chat_or_503(
    messages, db: Session, tenant_id: str, *, call_type: str, prompt_key: str | None = None,
    user_id: str | None = None,
) -> str:
    try:
        return call_llm(
            call_type, messages, db=db, tenant_id=tenant_id,
            prompt_key=prompt_key, user_id=user_id,
        )
    except LLMError as e:
        raise HTTPException(503, str(e)) from e


@router.post("/field-assist", response_model=FieldAssistOut)
def field_assist(
    body: FieldAssistIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    target = body.target.strip()
    module = target.split(".", 1)[0]
    perm = _FIELD_PERMS.get(module, PERM_VIEW_REPORTS)
    if not has_permission(current, perm):
        raise HTTPException(403, "没有该模块的 AI 助手权限")

    key = f"field.{target}.{body.intent}"
    if key not in PROMPT_REGISTRY:
        key = f"field.generic.{body.intent}"
    spec = PROMPT_REGISTRY[key]

    variables = dict(body.fields)
    variables.setdefault("instruction", body.instruction or "(无)")
    variables.setdefault("current_text", body.current_text)
    variables.setdefault(
        "context", json.dumps(body.fields, ensure_ascii=False)[:4000]
    )
    prompt = render_prompt(db, current.tenant_id, key, module=spec.module, **variables)
    if prompt is None:
        prompt = render_vars(spec.content, variables)

    text = _chat_or_503(
        [{"role": "system", "content": guarded(prompt)}], db, current.tenant_id,
        call_type="ai.field_assist", prompt_key=key, user_id=current.id,
    ).strip()[:_MAX_TEXT]
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=key,
        action="field_assist",
        detail={"target": target, "intent": body.intent},
        request=request,
    )
    db.commit()
    return FieldAssistOut(text=text, key=key)


@router.get("/summarize/{entity}/{rid}", response_model=SummarizeOut)
def summarize(
    entity: str,
    rid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    perm = _SUMMARY_PERMS.get(entity)
    if perm is None:
        raise HTTPException(400, f"不支持摘要的实体: {entity}")
    if not has_permission(current, perm):
        raise HTTPException(403, "没有该模块权限")

    ctx = build_summary_context(db, current, entity, rid)
    ctx_json = json.dumps(ctx, ensure_ascii=False, default=str)[:12_000]
    key = _SUMMARY_KEYS[entity]
    spec = PROMPT_REGISTRY[key]
    prompt = render_prompt(
        db, current.tenant_id, key, module=spec.module, context=ctx_json
    )
    if prompt is None:
        prompt = render_vars(spec.content, {"context": ctx_json})

    text = _chat_or_503(
        [{"role": "system", "content": guarded(prompt)}], db, current.tenant_id,
        call_type="ai.summarize", prompt_key=key, user_id=current.id,
    ).strip()[:_MAX_TEXT]
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=rid,
        action="summarize", detail={"entity": entity},
        request=request,
    )
    db.commit()
    return SummarizeOut(entity=entity, id=rid, summary=text)


@router.get("/doc-templates", response_model=list[DocTemplateOptionOut])
def list_doc_template_options(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """起草用模板下拉(轻量,不带 body;管理走 /api/doc-templates)。"""
    customs = {
        r.key: r
        for r in db.scalars(
            select(DocTemplate).where(
                DocTemplate.tenant_id == current.tenant_id,
                DocTemplate.is_active.is_(True),
            )
        )
    }
    out: list[DocTemplateOptionOut] = []
    for key, spec in DOC_SPECS.items():
        r = customs.pop(key, None)
        out.append(DocTemplateOptionOut(
            key=key, name=r.name if r else spec.name,
            entity=r.entity if r else spec.entity,
        ))
    for key, r in customs.items():
        out.append(DocTemplateOptionOut(key=key, name=r.name, entity=r.entity))
    return out


def _get_doc_template(db: Session, tenant_id: str, key: str) -> tuple[str, str, str]:
    """返回 (name, entity, body):租户覆盖行 > 内置注册表 > 自定义行已含在上面。"""
    row = db.scalars(
        select(DocTemplate).where(
            DocTemplate.tenant_id == tenant_id,
            DocTemplate.key == key,
            DocTemplate.is_active.is_(True),
        )
    ).first()
    if row is not None:
        return row.name, row.entity, row.body
    spec = DOC_SPECS.get(key)
    if spec is None:
        raise HTTPException(404, "文档模板不存在")
    return spec.name, spec.entity, spec.body


@router.post("/doc-draft", response_model=DocDraftOut)
def doc_draft(
    body: DocDraftIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_VIEW_REPORTS):
        raise HTTPException(403, "没有权限")
    name, tpl_entity, tpl_body = _get_doc_template(db, current.tenant_id, body.key)
    if tpl_entity not in ("any", body.entity):
        raise HTTPException(
            400, f"模板「{name}」只适用于 {tpl_entity},不适用于 {body.entity}"
        )

    scope = get_org_scope(db, current)
    vars_map = build_doc_vars(db, current, scope, body.entity, body.record_id)
    prefilled = render_vars(tpl_body, vars_map)
    ctx_json = json.dumps(vars_map, ensure_ascii=False, default=str)[:4000]

    spec = PROMPT_REGISTRY["doc.draft"]
    prompt = render_prompt(
        db, current.tenant_id, "doc.draft",
        template=prefilled, context=ctx_json,
        instruction=body.instruction or "(无)",
    )
    if prompt is None:
        prompt = render_vars(
            spec.content,
            {"template": prefilled, "context": ctx_json,
             "instruction": body.instruction or "(无)"},
        )
    draft = _chat_or_503(
        [{"role": "system", "content": guarded(prompt)}], db, current.tenant_id,
        call_type="ai.doc_draft", prompt_key=body.key, user_id=current.id,
    ).strip()[:_MAX_TEXT]
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=body.record_id,
        action="doc_draft",
        detail={"template": body.key, "entity": body.entity},
        request=request,
    )
    db.commit()
    return DocDraftOut(draft=draft, vars={k: str(v) for k, v in vars_map.items()})


# ==================== JD 合规审查(对标 Moka AI 审核 JD)====================


class JdReviewIn(BaseModel):
    job_id: str | None = None
    jd_text: str = Field("", max_length=_MAX_TEXT)


class JdIssue(BaseModel):
    dimension: str = ""
    severity: str = "low"
    quote: str = ""
    suggestion: str = ""


class JdReviewOut(BaseModel):
    score: int = 0
    issues: list[JdIssue] = Field(default_factory=list)
    summary: str = ""
    degraded: bool = False  # LLM 输出无法解析时的降级标记


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出里抠出第一个 JSON 对象(容忍 ```json 围栏/前导废话)。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except (ValueError, TypeError):
        return None


@router.post("/jd-review", response_model=JdReviewOut)
def jd_review(
    body: JdReviewIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_WRITE_JOBS) and not has_permission(
        current, PERM_VIEW_REPORTS
    ):
        raise HTTPException(403, "没有 JD 审查权限")
    from app.domain.models import Job

    title, jd = "岗位", body.jd_text.strip()
    if body.job_id:
        job = db.scalar(
            select(Job).where(Job.id == body.job_id, Job.tenant_id == current.tenant_id)
        )
        if not job:
            raise HTTPException(404, "岗位不存在")
        title = job.title
        jd = jd or (job.description or "").strip()
    if not jd:
        raise HTTPException(400, "JD 内容为空")

    spec = PROMPT_REGISTRY["jd.review"]
    prompt = render_prompt(
        db, current.tenant_id, "jd.review", module="jobs",
        title=title, jd_text=jd[:8000],
    ) or render_vars(spec.content, {"title": title, "jd_text": jd[:8000]})

    raw = _chat_or_503(
        [{"role": "system", "content": guarded(prompt)}], db, current.tenant_id,
        call_type="ai.jd_review", prompt_key="jd.review", user_id=current.id,
    )
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=body.job_id or "adhoc",
        action="jd_review", detail={"title": title}, request=request,
    )
    db.commit()

    data = _extract_json(raw)
    if data is None:
        return JdReviewOut(score=0, issues=[], summary=raw[:2000], degraded=True)
    issues = [
        JdIssue(
            dimension=str(i.get("dimension", "")),
            severity=i.get("severity", "low") if i.get("severity") in ("high", "medium", "low") else "low",
            quote=str(i.get("quote", ""))[:500],
            suggestion=str(i.get("suggestion", ""))[:500],
        )
        for i in (data.get("issues") or [])[:12]
        if isinstance(i, dict)
    ]
    try:
        score = max(0, min(100, int(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    return JdReviewOut(score=score, issues=issues, summary=str(data.get("summary", ""))[:1000])


# ==================== 面试教练·面试官副驾(对标 Moka 实时护航)====================


class CopilotIn(BaseModel):
    scorecard_id: str | None = None
    answer_text: str = Field(..., min_length=1, max_length=_MAX_TEXT)
    asked_dimensions: list[str] = Field(default_factory=list)


class CopilotFollowup(BaseModel):
    dimension: str = ""
    question: str = ""
    intent: str = ""


class CopilotOut(BaseModel):
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    followups: list[CopilotFollowup] = Field(default_factory=list)
    note: str = ""
    degraded: bool = False


@router.post("/interview-copilot", response_model=CopilotOut)
def interview_copilot(
    body: CopilotIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_VIEW_REPORTS):
        raise HTTPException(403, "没有面试副驾权限")
    from app.domain.models import InterviewScorecard

    # 加载评分卡考察维度(可选——没有评分卡时退化为通用副驾)
    criteria: list[str] = []
    if body.scorecard_id:
        sc = db.scalar(
            select(InterviewScorecard).where(
                InterviewScorecard.id == body.scorecard_id,
                InterviewScorecard.tenant_id == current.tenant_id,
            )
        )
        if not sc:
            raise HTTPException(404, "评分卡不存在")
        criteria = [
            str(c.get("name", "")).strip()
            for c in (sc.criteria or [])
            if isinstance(c, dict) and c.get("name")
        ]

    # 轻量覆盖判断:维度名(或其关键片段)出现在已问维度/候选人回答相关上下文
    asked = set(body.asked_dimensions)
    covered = [c for c in criteria if c in asked]
    missing = [c for c in criteria if c not in asked]

    spec = PROMPT_REGISTRY["interview.copilot"]
    variables = {
        "criteria": json.dumps(criteria or ["通用能力", "专业技能", "文化契合"], ensure_ascii=False),
        "covered": "、".join(covered) or "无",
        "answer_text": body.answer_text[:4000],
    }
    prompt = render_prompt(
        db, current.tenant_id, "interview.copilot", module="interviews", **variables
    ) or render_vars(spec.content, variables)

    try:
        raw = call_llm(
            "ai.interview_copilot",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="interview.copilot", user_id=current.id,
        )
    except LLMError:
        raw = ""

    write_audit(
        db, actor=current, entity_type="ai_call",
        entity_id=body.scorecard_id or "adhoc",
        action="interview_copilot", detail={}, request=request,
    )
    db.commit()

    data = _extract_json(raw) if raw else None
    if data is None:
        # 降级:LLM 不可用/输出不可解析 → 规则化兜底(不瘫)
        fallback_dims = missing or criteria[:3] or ["通用能力"]
        return CopilotOut(
            covered=covered,
            missing=missing,
            followups=[
                CopilotFollowup(
                    dimension=d,
                    question=f"能举个具体例子说明你在「{d}」方面的表现吗?当时的目标、行动和结果分别是什么?",
                    intent=f"验证{d}的真实深度(STAR 行为面)",
                )
                for d in fallback_dims[:3]
            ],
            note="AI 副驾暂不可用,已按考察维度给出通用追问建议。" if not raw else "AI 输出未能解析,已按考察维度给出通用追问建议。",
            degraded=True,
        )

    followups = [
        CopilotFollowup(
            dimension=str(f.get("dimension", "")),
            question=str(f.get("question", ""))[:300],
            intent=str(f.get("intent", ""))[:300],
        )
        for f in (data.get("followups") or [])[:3]
        if isinstance(f, dict) and f.get("question")
    ]
    llm_missing = [
        str(m) for m in (data.get("missing") or []) if str(m) in criteria or not criteria
    ]
    return CopilotOut(
        covered=covered,
        missing=llm_missing or missing,
        followups=followups,
        note=str(data.get("note", ""))[:500],
    )


# ==================== AI 绩效助手(对标北森 AI 绩效助手三段式)====================
# 目标建议 → 评语起草(+校准/面谈) → 反馈分析;信号源全部走库内聚合。

from datetime import date  # noqa: E402

from app.domain.models import (  # noqa: E402
    Appraisal,
    AppraisalCycle,
    Employee,
    EmployeeSkill,
    Goal,
    InsightNote,
    OrgUnit,
    Skill,
    Survey,
    SurveyQuestion,
    SurveyResponse,
    SurveyResponseAnswer,
)


def _employee_brief(db: Session, tenant_id: str, emp: Employee) -> dict:
    org = db.get(OrgUnit, emp.org_unit_id) if emp.org_unit_id else None
    tenure = (
        round((date.today() - emp.hire_date).days / 365.25, 1)
        if emp.hire_date else None
    )
    return {
        "name": emp.name,
        "designation": emp.designation or "",
        "org_unit": org.name if org else "",
        "tenure_years": tenure,
    }


def _confirmed_skills(db: Session, tenant_id: str, employee_id: str) -> list[str]:
    rows = db.execute(
        select(EmployeeSkill, Skill)
        .join(Skill, Skill.id == EmployeeSkill.skill_id)
        .where(
            EmployeeSkill.tenant_id == tenant_id,
            EmployeeSkill.employee_id == employee_id,
            EmployeeSkill.status == "confirmed",
        )
    ).all()
    return [f"{s.name}(L{es.level})" for es, s in rows][:12]


class GoalSuggestIn(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=36)
    instruction: str = Field("", max_length=500)
    count: int = Field(4, ge=1, le=6)


class GoalSuggestion(BaseModel):
    title: str = ""
    description: str = ""


class GoalSuggestOut(BaseModel):
    goals: list[GoalSuggestion] = Field(default_factory=list)
    degraded: bool = False


@router.post("/goal-suggest", response_model=GoalSuggestOut)
def goal_suggest(
    body: GoalSuggestIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_MANAGE_PERFORMANCE):
        raise HTTPException(403, "没有目标管理权限")
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")

    existing = [
        g.title
        for g in db.scalars(
            select(Goal).where(
                Goal.tenant_id == current.tenant_id,
                Goal.employee_id == emp.id,
                Goal.status == "active",
            )
        ).all()
    ][:8]
    variables = {
        "employee": json.dumps(_employee_brief(db, current.tenant_id, emp), ensure_ascii=False),
        "skills": "、".join(_confirmed_skills(db, current.tenant_id, emp.id)) or "暂无技能档案",
        "existing_goals": "、".join(existing) or "无",
        "instruction": body.instruction or "(无)",
    }
    spec = PROMPT_REGISTRY["goal.suggest"]
    prompt = render_prompt(
        db, current.tenant_id, "goal.suggest", module="performance", **variables
    ) or render_vars(spec.content, variables)

    try:
        raw = call_llm(
            "ai.goal_suggest",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="goal.suggest", user_id=current.id,
        )
    except LLMError:
        raw = ""

    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=emp.id,
        action="goal_suggest", detail={"employee": emp.name}, request=request,
    )
    db.commit()

    data = _extract_json(raw) if raw else None
    goals = [
        GoalSuggestion(
            title=str(g.get("title", ""))[:60],
            description=str(g.get("description", ""))[:200],
        )
        for g in ((data or {}).get("goals") or [])[: body.count]
        if isinstance(g, dict) and g.get("title")
    ]
    if not goals:
        # 降级:按岗位给通用 SMART 骨架
        title_hint = emp.designation or "本岗位"
        goals = [
            GoalSuggestion(
                title=f"提升{title_hint}核心交付质量",
                description="本周期内关键交付物零返工;按时完成率 ≥95%。",
            ),
            GoalSuggestion(
                title="沉淀一项可复用的工作方法",
                description="产出 1 份文档/SOP 并在团队内分享一次。",
            ),
            GoalSuggestion(
                title="补强一项岗位相关技能",
                description="结合技能档案短板,完成学习并在实际任务中应用一次。",
            ),
        ]
    return GoalSuggestOut(goals=goals, degraded=data is None)


class AppraisalAssistIn(BaseModel):
    appraisal_id: str = Field(..., min_length=1, max_length=36)


class AppraisalAssistOut(BaseModel):
    suggested_rating: int | None = None
    review_draft: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    calibration_note: str = ""
    talking_points: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False


@router.post("/appraisal-assist", response_model=AppraisalAssistOut)
def appraisal_assist(
    body: AppraisalAssistIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_MANAGE_PERFORMANCE):
        raise HTTPException(403, "没有绩效评价权限")
    a = db.get(Appraisal, body.appraisal_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(404, "考核表不存在")
    cycle = db.get(AppraisalCycle, a.cycle_id)
    emp = db.get(Employee, a.employee_id)
    if not cycle or not emp:
        raise HTTPException(404, "考核上下文缺失")

    # 周期内目标达成
    goals = db.scalars(
        select(Goal).where(
            Goal.tenant_id == current.tenant_id,
            Goal.employee_id == emp.id,
        )
    ).all()
    in_cycle = [
        {
            "title": g.title,
            "progress": g.progress,
            "status": g.status,
            "due": str(g.due_date) if g.due_date else None,
        }
        for g in goals
        if g.due_date is None
        or (cycle.period_start <= g.due_date <= cycle.period_end)
        or g.status == "active"
    ][:10]
    # 周期内过程记录
    notes = [
        {
            "kind": n.kind,
            "title": n.title or "",
            "date": n.created_at.date().isoformat() if n.created_at else "",
            "excerpt": (n.body or "")[:120],
        }
        for n in db.scalars(
            select(InsightNote)
            .where(
                InsightNote.tenant_id == current.tenant_id,
                InsightNote.employee_id == emp.id,
            )
            .order_by(InsightNote.created_at.desc())
        ).all()
        if n.created_at and n.created_at.date() <= cycle.period_end
    ][:8]

    variables = {
        "employee": json.dumps(_employee_brief(db, current.tenant_id, emp), ensure_ascii=False),
        "goals": json.dumps(in_cycle, ensure_ascii=False) or "[]",
        "skills": "、".join(_confirmed_skills(db, current.tenant_id, emp.id)) or "暂无",
        "notes": json.dumps(notes, ensure_ascii=False) or "[]",
        "self_review": (a.self_review or "(未填写)")[:2000],
        "self_rating": a.self_rating if a.self_rating is not None else "未评",
    }
    spec = PROMPT_REGISTRY["appraisal.assist"]
    prompt = render_prompt(
        db, current.tenant_id, "appraisal.assist", module="performance", **variables
    ) or render_vars(spec.content, variables)

    try:
        raw = call_llm(
            "ai.appraisal_assist",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="appraisal.assist", user_id=current.id,
        )
    except LLMError:
        raw = ""

    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=a.id,
        action="appraisal_assist", detail={"employee": emp.name, "cycle": cycle.name},
        request=request,
    )
    db.commit()

    evidence = {
        "goal_count": len(in_cycle),
        "note_count": len(notes),
        "avg_progress": (
            round(sum(g["progress"] for g in in_cycle) / len(in_cycle))
            if in_cycle else None
        ),
    }
    data = _extract_json(raw) if raw else None
    if data is None:
        # 降级:按目标达成模板化草稿
        done = [g for g in in_cycle if g["status"] == "done" or g["progress"] >= 80]
        draft = (
            f"本周期 {len(in_cycle)} 项目标中 {len(done)} 项达成/接近达成"
            f"(平均进度 {evidence['avg_progress'] or 0}%)。"
            + ("自评较积极,建议面谈中核对关键交付物。" if a.self_rating and a.self_rating >= 4 else "")
        )
        return AppraisalAssistOut(
            suggested_rating=None,
            review_draft=draft,
            strengths=[g["title"] for g in done[:2]],
            improvements=["过程记录不足,建议补充事实依据" if not notes else ""],
            calibration_note="AI 暂不可用,以上为按目标数据生成的模板草稿,请人工修订。",
            talking_points=["核对目标完成口径", "确认下周期目标"],
            evidence=evidence,
            degraded=True,
        )

    try:
        rating = int(data.get("suggested_rating"))
        rating = rating if 1 <= rating <= 5 else None
    except (TypeError, ValueError):
        rating = None
    return AppraisalAssistOut(
        suggested_rating=rating,
        review_draft=str(data.get("review_draft", ""))[:800],
        strengths=[str(s)[:80] for s in (data.get("strengths") or [])][:4],
        improvements=[str(s)[:80] for s in (data.get("improvements") or [])][:4],
        calibration_note=str(data.get("calibration_note", ""))[:300],
        talking_points=[str(t)[:120] for t in (data.get("talking_points") or [])][:4],
        evidence=evidence,
    )


# ==================== 反馈分析(对标北森/Moka 的员工域反馈洞察)====================


class FeedbackAnalyzeIn(BaseModel):
    target: str = Field(..., pattern="^(survey|employee)$")
    survey_id: str | None = None
    employee_id: str | None = None


class FeedbackTheme(BaseModel):
    theme: str = ""
    support: str = ""
    quote: str = ""
    count: int | None = None
    polarity: str = ""


class FeedbackAnalyzeOut(BaseModel):
    target: str
    sentiment: str = ""
    themes: list[FeedbackTheme] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    trajectory: str = ""
    summary: str = ""
    response_count: int | None = None
    degraded: bool = False


@router.post("/feedback-analyze", response_model=FeedbackAnalyzeOut)
def feedback_analyze(
    body: FeedbackAnalyzeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if body.target == "survey":
        return _survey_feedback(body, request, db, current)
    return _employee_feedback(body, request, db, current)


def _survey_feedback(
    body: FeedbackAnalyzeIn, request: Request, db: Session, current: User
) -> FeedbackAnalyzeOut:
    if not has_permission(current, PERM_VIEW_REPORTS):
        raise HTTPException(403, "没有问卷分析权限")
    if not body.survey_id:
        raise HTTPException(400, "缺少 survey_id")
    s = db.get(Survey, body.survey_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "问卷不存在")

    questions = db.scalars(
        select(SurveyQuestion)
        .where(SurveyQuestion.survey_id == s.id)
        .order_by(SurveyQuestion.order)
    ).all()
    responses = db.scalars(
        select(SurveyResponse).where(SurveyResponse.survey_id == s.id)
    ).all()
    answers = db.execute(
        select(SurveyResponseAnswer, SurveyResponse)
        .join(SurveyResponse, SurveyResponse.id == SurveyResponseAnswer.response_id)
        .where(
            SurveyResponseAnswer.tenant_id == current.tenant_id,
            SurveyResponse.survey_id == s.id,
        )
    ).all()
    by_q: dict[str, list] = {}
    for ans, _r in answers:
        by_q.setdefault(ans.question_id, []).append(ans)

    stats: list[dict] = []
    text_samples: list[str] = []
    for q in questions:
        qans = by_q.get(q.id, [])
        item: dict[str, Any] = {"q": q.text, "type": q.type, "answered": len(qans)}
        if q.type in ("single_choice", "multi_choice"):
            counts: dict[str, int] = {}
            for a in qans:
                vals = a.value if isinstance(a.value, list) else [a.value]
                for v in vals:
                    counts[str(v)] = counts.get(str(v), 0) + 1
            item["options"] = counts
        elif q.type == "rating":
            nums = [a.value for a in qans if isinstance(a.value, int | float)]
            if nums:
                item["avg"] = round(sum(nums) / len(nums), 2)
        else:
            samples = [str(a.value) for a in qans if a.value][:6]
            item["samples"] = samples
            text_samples.extend(samples)
        stats.append(item)

    happy = [r.happiness for r in responses if r.happiness is not None]
    stats.append({
        "q": "整体幸福感(0-10)", "type": "rating",
        "answered": len(happy),
        "avg": round(sum(happy) / len(happy), 2) if happy else None,
    })

    spec = PROMPT_REGISTRY["feedback.survey"]
    variables = {
        "title": s.title,
        "stats": json.dumps(stats, ensure_ascii=False)[:8000],
    }
    prompt = render_prompt(
        db, current.tenant_id, "feedback.survey", module="performance", **variables
    ) or render_vars(spec.content, variables)
    try:
        raw = call_llm(
            "ai.feedback_survey",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="feedback.survey", user_id=current.id,
        )
    except LLMError:
        raw = ""

    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=s.id,
        action="feedback_analyze", detail={"survey": s.title}, request=request,
    )
    db.commit()

    data = _extract_json(raw) if raw else None
    if data is None:
        # 降级:低分题即风险
        low = [
            i["q"] for i in stats
            if i.get("type") == "rating" and i.get("avg") is not None and i["avg"] < 3
        ]
        return FeedbackAnalyzeOut(
            target="survey",
            sentiment="mixed",
            themes=[],
            risks=[f"低分题:{q}" for q in low[:3]],
            actions=["AI 暂不可用;建议人工查看文本回答原文"],
            summary=f"共 {len(responses)} 份回收;评分均值偏低题目 {len(low)} 个。",
            response_count=len(responses),
            degraded=True,
        )
    themes = [
        FeedbackTheme(
            theme=str(t.get("theme", ""))[:60],
            support=str(t.get("support", "")),
            quote=str(t.get("quote", ""))[:200],
        )
        for t in (data.get("themes") or [])[:6]
        if isinstance(t, dict) and t.get("theme")
    ]
    return FeedbackAnalyzeOut(
        target="survey",
        sentiment=str(data.get("sentiment", "")),
        themes=themes,
        risks=[str(r)[:120] for r in (data.get("risks") or [])][:5],
        actions=[str(a)[:120] for a in (data.get("actions") or [])][:5],
        response_count=len(responses),
    )


def _employee_feedback(
    body: FeedbackAnalyzeIn, request: Request, db: Session, current: User
) -> FeedbackAnalyzeOut:
    if not has_permission(current, PERM_MANAGE_EMPLOYEES):
        raise HTTPException(403, "没有员工反馈分析权限")
    if not body.employee_id:
        raise HTTPException(400, "缺少 employee_id")
    scope = visible_employee_ids(db, current)
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在")

    notes = db.scalars(
        select(InsightNote)
        .where(
            InsightNote.tenant_id == current.tenant_id,
            InsightNote.employee_id == emp.id,
        )
        .order_by(InsightNote.created_at.desc())
    ).all()[:30]
    if not notes:
        return FeedbackAnalyzeOut(
            target="employee",
            summary="该员工暂无过程反馈记录(便签/复盘/里程碑)。",
            degraded=True,
        )

    note_items = [
        {
            "kind": n.kind,
            "date": n.created_at.date().isoformat() if n.created_at else "",
            "excerpt": (n.body or "")[:200],
        }
        for n in notes
    ]
    spec = PROMPT_REGISTRY["feedback.employee"]
    variables = {
        "employee": json.dumps(_employee_brief(db, current.tenant_id, emp), ensure_ascii=False),
        "notes": json.dumps(note_items, ensure_ascii=False)[:8000],
    }
    prompt = render_prompt(
        db, current.tenant_id, "feedback.employee", module="performance", **variables
    ) or render_vars(spec.content, variables)
    try:
        raw = call_llm(
            "ai.feedback_employee",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="feedback.employee", user_id=current.id,
        )
    except LLMError:
        raw = ""

    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=emp.id,
        action="feedback_analyze", detail={"employee": emp.name, "notes": len(notes)},
        request=request,
    )
    db.commit()

    data = _extract_json(raw) if raw else None
    if data is None:
        return FeedbackAnalyzeOut(
            target="employee",
            summary=f"共 {len(notes)} 条过程记录;AI 暂不可用,请人工查阅明细。",
            degraded=True,
        )
    themes = [
        FeedbackTheme(
            theme=str(t.get("theme", ""))[:60],
            count=t.get("count") if isinstance(t.get("count"), int) else None,
            polarity=str(t.get("polarity", "")),
        )
        for t in (data.get("themes") or [])[:6]
        if isinstance(t, dict) and t.get("theme")
    ]
    return FeedbackAnalyzeOut(
        target="employee",
        sentiment=str(data.get("sentiment", "")),
        themes=themes,
        actions=[str(a)[:120] for a in (data.get("actions") or [])][:4],
        trajectory=str(data.get("trajectory", "")),
        summary=str(data.get("summary", ""))[:300],
    )


# ==================== 会议纪要整理(对标 Moka 智能纪要)====================


class MeetingMinutesIn(BaseModel):
    note_id: str | None = None        # 持久化模式:生成并写入 note.source_ref.minutes
    body: str = Field("", max_length=_MAX_TEXT)  # 草稿模式:仅返回结构化纪要
    title: str = Field("", max_length=200)


class ActionItem(BaseModel):
    item: str = ""
    owner: str = ""
    due: str = ""


class MeetingMinutesOut(BaseModel):
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    sentiment: str = ""
    note_id: str | None = None
    saved: bool = False
    degraded: bool = False


@router.post("/meeting-minutes", response_model=MeetingMinutesOut)
def meeting_minutes(
    body: MeetingMinutesIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, PERM_MANAGE_EMPLOYEES):
        raise HTTPException(403, "没有会议纪要整理权限")

    note: InsightNote | None = None
    title, raw_body = body.title.strip(), body.body.strip()
    if body.note_id:
        note = db.get(InsightNote, body.note_id)
        if not note or note.tenant_id != current.tenant_id:
            raise HTTPException(404, "便签不存在")
        scope = visible_employee_ids(db, current)
        if scope is not None and note.employee_id not in scope:
            raise HTTPException(404, "便签不存在")
        title = title or (note.title or "")
        raw_body = raw_body or note.body
    if not raw_body:
        raise HTTPException(400, "会议记录内容为空")

    spec = PROMPT_REGISTRY["meeting.minutes"]
    variables = {"title": title or "(未命名会议)", "body": raw_body[:12_000]}
    prompt = render_prompt(
        db, current.tenant_id, "meeting.minutes", module="performance", **variables
    ) or render_vars(spec.content, variables)
    try:
        raw = call_llm(
            "ai.meeting_minutes",
            [{"role": "system", "content": guarded(prompt)}],
            db=db, tenant_id=current.tenant_id,
            prompt_key="meeting.minutes", user_id=current.id,
        )
    except LLMError:
        raw = ""

    data = _extract_json(raw) if raw else None
    if data is None:
        # 降级:首段做摘要,分号/换行切要点
        first = raw_body.strip().split("\n")[0][:120]
        points = [
            ln.strip(" -•\t")[:80]
            for ln in raw_body.split("\n")[1:8]
            if ln.strip(" -•\t")
        ]
        out = MeetingMinutesOut(
            summary=first or raw_body[:120],
            key_points=points[:5],
            note_id=note.id if note else None,
            degraded=True,
        )
    else:
        out = MeetingMinutesOut(
            summary=str(data.get("summary", ""))[:400],
            key_points=[str(p)[:120] for p in (data.get("key_points") or [])][:6],
            decisions=[str(d)[:120] for d in (data.get("decisions") or [])][:6],
            action_items=[
                ActionItem(
                    item=str(a.get("item", ""))[:120],
                    owner=str(a.get("owner", "") or "")[:40],
                    due=str(a.get("due", "") or "")[:40],
                )
                for a in (data.get("action_items") or [])[:8]
                if isinstance(a, dict) and a.get("item")
            ],
            sentiment=str(data.get("sentiment", "")),
            note_id=note.id if note else None,
        )

    saved = False
    if note is not None:
        ref = dict(note.source_ref or {})
        ref["minutes"] = {
            "summary": out.summary,
            "key_points": out.key_points,
            "decisions": out.decisions,
            "action_items": [a.model_dump() for a in out.action_items],
            "sentiment": out.sentiment,
        }
        note.source_ref = ref
        saved = True
    write_audit(
        db, actor=current, entity_type="ai_call",
        entity_id=note.id if note else "adhoc",
        action="meeting_minutes",
        detail={"title": title, "saved": saved}, request=request,
    )
    db.commit()
    out.saved = saved
    return out
