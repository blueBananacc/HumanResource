"""T13 声明式评分规则 API。

端点(读写均走 ``match.evaluate`` license feature,与匹配评估同源):
- GET    /api/scoring-rules                     列表(附每规则挂的岗位数)
- POST   /api/scoring-rules                     创建(维度 normalize)
- GET    /api/scoring-rules/{id}                详情 + 关联岗位
- PUT    /api/scoring-rules/{id}                修改(version+1)
- DELETE /api/scoring-rules/{id}                删除(岗位列 SET NULL)
- POST   /api/scoring-rules/generate            按岗位 AI 起草规则草案
- POST   /api/scoring-rules/{id}/analyze        规则×关联岗位 → 结构化分析
- POST   /api/scoring-rules/{id}/apply-analysis 应用分析草案(version+1)
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import Job, ScoringRule, User
from app.infra.db import get_db
from app.services import scoring_rules as sr

router = APIRouter(prefix="/scoring-rules", tags=["scoring-rules"])
logger = logging.getLogger(__name__)


# --------------------------- Schemas ---------------------------


class RuleIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    dimensions: list[dict[str, Any]] = Field(default_factory=list)


class RuleOut(BaseModel):
    id: str
    name: str
    description: str | None
    dimensions: list[dict[str, Any]]
    version: int
    weight_total: int
    job_count: int = 0


class GenerateIn(BaseModel):
    job_id: str


class ApplyIn(BaseModel):
    analysis: dict[str, Any]


# --------------------------- Helpers ---------------------------


def _get_rule(db: Session, rid: str, current: User) -> ScoringRule:
    r = db.get(ScoringRule, rid)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(404, "评分规则不存在")
    return r


def _job_count(db: Session, rule_id: str, tenant_id: str) -> int:
    return int(
        db.scalar(
            select(func.count(Job.id)).where(
                Job.scoring_rule_id == rule_id,
                Job.tenant_id == tenant_id,
            )
        )
        or 0
    )


def _to_out(db: Session, r: ScoringRule) -> RuleOut:
    return RuleOut(
        id=r.id,
        name=r.name,
        description=r.description,
        dimensions=list(r.dimensions or []),
        version=r.version,
        weight_total=sr.weight_total(r.dimensions or []),
        job_count=_job_count(db, r.id, r.tenant_id),
    )


def _linked_jobs(db: Session, rule_id: str, tenant_id: str) -> list[Job]:
    return list(
        db.scalars(
            select(Job).where(
                Job.scoring_rule_id == rule_id,
                Job.tenant_id == tenant_id,
            )
        ).all()
    )


# --------------------------- Routes ---------------------------


@router.get("", response_model=list[RuleOut])
def list_rules(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[RuleOut]:
    rows = db.scalars(
        select(ScoringRule)
        .where(ScoringRule.tenant_id == current.tenant_id)
        .order_by(ScoringRule.updated_at.desc())
    ).all()
    return [_to_out(db, r) for r in rows]


@router.post(
    "",
    response_model=RuleOut,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def create_rule(
    body: RuleIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RuleOut:
    dims = sr.normalize_dimensions(body.dimensions)
    if not dims:
        raise HTTPException(400, "至少需要一个有效维度")
    rule = ScoringRule(
        tenant_id=current.tenant_id,
        name=body.name.strip(),
        description=body.description,
        dimensions=dims,
        version=1,
        created_by=current.id,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return _to_out(db, rule)


@router.get("/{rule_id}", response_model=dict)
def get_rule(
    rule_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    r = _get_rule(db, rule_id, current)
    jobs = _linked_jobs(db, r.id, current.tenant_id)
    return {
        **_to_out(db, r).model_dump(),
        "jobs": [{"id": j.id, "title": j.title, "status": j.status} for j in jobs],
    }


@router.get("/{rule_id}/learned", response_model=dict)
def learned_insights(
    rule_id: str,
    days: int = 90,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    """P3「我从你的选择里学到了什么」:按维度聚合近期决策反馈,
    给出权重调整建议(建议仅展示,不自动改规则)。"""
    from app.services.decision_learning import weight_suggestions

    r = _get_rule(db, rule_id, current)
    raw = weight_suggestions(db, current.tenant_id, days=days)
    rule_dims = {str(d.get("id") or d.get("name")): d for d in (r.dimensions or [])}
    name_to_id = {
        str(d.get("name")): str(d.get("id") or d.get("name"))
        for d in (r.dimensions or [])
    }
    dims = []
    for d in raw["dimensions"]:
        rd = rule_dims.get(d["key"]) or rule_dims.get(name_to_id.get(d["key"], ""))
        dims.append({**d, "in_rule": rd is not None,
                     "current_weight": rd.get("weight", d["current_weight"]) if rd else d["current_weight"]})
    return {
        **raw,
        "rule": {"id": r.id, "name": r.name, "version": r.version},
        "dimensions": dims,
    }


@router.put(
    "/{rule_id}",
    response_model=RuleOut,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def update_rule(
    rule_id: str,
    body: RuleIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RuleOut:
    r = _get_rule(db, rule_id, current)
    dims = sr.normalize_dimensions(body.dimensions)
    if not dims:
        raise HTTPException(400, "至少需要一个有效维度")
    r.name = body.name.strip()
    r.description = body.description
    r.dimensions = dims
    r.version = (r.version or 1) + 1
    db.commit()
    db.refresh(r)
    return _to_out(db, r)


@router.delete(
    "/{rule_id}",
    status_code=204,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def delete_rule(
    rule_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    r = _get_rule(db, rule_id, current)
    db.delete(r)  # jobs.scoring_rule_id ON DELETE SET NULL
    db.commit()


@router.post(
    "/generate",
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def generate_rule(
    body: GenerateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    """按岗位 JD 起草规则草案(不落库,前端确认后走 POST 保存)。"""
    job = db.get(Job, body.job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    draft = sr.generate_rule_draft(db, job)
    return {**draft, "weight_total": sr.weight_total(draft["dimensions"])}


@router.post(
    "/{rule_id}/analyze",
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def analyze(
    rule_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    r = _get_rule(db, rule_id, current)
    jobs = _linked_jobs(db, r.id, current.tenant_id)
    if not jobs:
        raise HTTPException(400, "该规则未挂任何岗位,无法对照分析")
    return sr.analyze_rule(db, r, jobs)


@router.post(
    "/{rule_id}/apply-analysis",
    response_model=RuleOut,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def apply(
    rule_id: str,
    body: ApplyIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RuleOut:
    r = _get_rule(db, rule_id, current)
    proposed = (body.analysis or {}).get("proposedRule") or {}
    if not sr.normalize_dimensions(proposed.get("dimensions")):
        raise HTTPException(400, "分析草案无有效维度")
    sr.apply_analysis(r, body.analysis)
    db.commit()
    db.refresh(r)
    return _to_out(db, r)
