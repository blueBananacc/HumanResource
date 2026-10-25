"""T16 人才库扫描 API。

- POST /api/sourcing/scan   {profile_id | spec, limit, llm_top_k} → 排序短名单
- GET  /api/sourcing/runs   扫描留痕列表
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import SourcingRun, TalentProfile, User
from app.infra.db import get_db
from app.services import sourcing

router = APIRouter(prefix="/sourcing", tags=["sourcing"])
logger = logging.getLogger(__name__)


class ScanIn(BaseModel):
    profile_id: str | None = None
    spec: dict[str, Any] | None = None
    limit: int = Field(default=5, ge=1, le=20)
    llm_top_k: int = Field(default=10, ge=0, le=20)


class RunOut(BaseModel):
    id: str
    profile_id: str | None
    scanned_total: int
    prefiltered: int
    llm_scored: int
    returned: int
    duration_ms: int
    created_at: str


@router.post(
    "/scan",
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def scan(
    body: ScanIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    profile: TalentProfile | None = None
    if body.profile_id:
        profile = db.get(TalentProfile, body.profile_id)
        if not profile or profile.tenant_id != current.tenant_id:
            raise HTTPException(404, "人才画像不存在")
    spec = profile.spec if profile else (body.spec or {})
    if profile and body.spec:
        spec = {**profile.spec, **{k: v for k, v in body.spec.items() if v}}
    result = sourcing.scan(
        db,
        tenant_id=current.tenant_id,
        spec=spec,
        profile=profile,
        limit=body.limit,
        llm_top_k=body.llm_top_k,
        created_by=current.id,
    )
    if profile:
        result["profile"] = {"id": profile.id, "name": profile.name}
    return result


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    limit: int = 20,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[RunOut]:
    rows = db.scalars(
        select(SourcingRun)
        .where(SourcingRun.tenant_id == current.tenant_id)
        .order_by(SourcingRun.created_at.desc())
        .limit(min(limit, 100))
    ).all()
    return [
        RunOut(
            id=r.id, profile_id=r.profile_id,
            scanned_total=r.scanned_total, prefiltered=r.prefiltered,
            llm_scored=r.llm_scored, returned=r.returned,
            duration_ms=r.duration_ms,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]
