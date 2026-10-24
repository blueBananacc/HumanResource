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
from app.integrations.llm.provider import LLMError, chat
from app.services.ai_summary import build_summary_context
from app.services.audit import write_audit
from app.services.doc_templates import DOC_SPECS, build_doc_vars
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    PERM_WRITE_JOBS,
    PERM_WRITE_RESUMES,
    get_org_scope,
    has_permission,
)
from app.services.prompting import PROMPT_REGISTRY, render_prompt, render_vars

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


def _chat_or_503(messages, db: Session, tenant_id: str) -> str:
    try:
        return chat(messages, db=db, tenant_id=tenant_id)
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
        [{"role": "system", "content": prompt}], db, current.tenant_id
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
        [{"role": "system", "content": prompt}], db, current.tenant_id
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
        [{"role": "system", "content": prompt}], db, current.tenant_id
    ).strip()[:_MAX_TEXT]
    write_audit(
        db, actor=current, entity_type="ai_call", entity_id=body.record_id,
        action="doc_draft",
        detail={"template": body.key, "entity": body.entity},
        request=request,
    )
    db.commit()
    return DocDraftOut(draft=draft, vars={k: str(v) for k, v in vars_map.items()})
