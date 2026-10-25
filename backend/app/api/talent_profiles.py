"""T16 人才画像 API —— 「招什么样的人」可复用模板。

端点(读写走 ``match.evaluate`` license feature):
- GET    /api/talent-profiles                列表(名称模糊 q,last_used_at 优先)
- POST   /api/talent-profiles                创建(spec normalize)
- GET    /api/talent-profiles/{id}           详情
- PUT    /api/talent-profiles/{id}           修改
- DELETE /api/talent-profiles/{id}           删除
- POST   /api/talent-profiles/generate       一句话 → AI 起草 spec(不落库)
- POST   /api/talent-profiles/from-job       岗位 → 画像草稿(不落库)
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
from app.domain.models import Job, TalentProfile, User
from app.infra.db import get_db
from app.services import sourcing

router = APIRouter(prefix="/talent-profiles", tags=["talent-profiles"])
logger = logging.getLogger(__name__)


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    spec: dict[str, Any] = Field(default_factory=dict)
    scoring_rule_id: str | None = None
    source_job_id: str | None = None


class ProfileOut(BaseModel):
    id: str
    name: str
    spec: dict[str, Any]
    scoring_rule_id: str | None
    source_job_id: str | None
    last_used_at: str | None
    use_count: int
    status: str


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)


class FromJobIn(BaseModel):
    job_id: str


def _get_profile(db: Session, pid: str, current: User) -> TalentProfile:
    p = db.get(TalentProfile, pid)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "人才画像不存在")
    return p


def _to_out(p: TalentProfile) -> ProfileOut:
    return ProfileOut(
        id=p.id,
        name=p.name,
        spec=dict(p.spec or {}),
        scoring_rule_id=p.scoring_rule_id,
        source_job_id=p.source_job_id,
        last_used_at=p.last_used_at.isoformat() if p.last_used_at else None,
        use_count=p.use_count or 0,
        status=p.status,
    )


@router.get("", response_model=list[ProfileOut])
def list_profiles(
    q: str = "",
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[ProfileOut]:
    stmt = select(TalentProfile).where(
        TalentProfile.tenant_id == current.tenant_id,
        TalentProfile.status == "active",
    )
    if q.strip():
        stmt = stmt.where(TalentProfile.name.ilike(f"%{q.strip()}%"))
    rows = db.scalars(
        stmt.order_by(TalentProfile.last_used_at.desc().nullslast(),
                      TalentProfile.updated_at.desc())
    ).all()
    return [_to_out(p) for p in rows]


@router.post(
    "",
    response_model=ProfileOut,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def create_profile(
    body: ProfileIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProfileOut:
    spec = sourcing.normalize_spec(body.spec)
    p = TalentProfile(
        tenant_id=current.tenant_id,
        name=body.name.strip(),
        spec=spec,
        scoring_rule_id=body.scoring_rule_id,
        source_job_id=body.source_job_id,
        created_by=current.id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _to_out(p)


@router.get("/{pid}", response_model=ProfileOut)
def get_profile(
    pid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProfileOut:
    return _to_out(_get_profile(db, pid, current))


@router.put(
    "/{pid}",
    response_model=ProfileOut,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def update_profile(
    pid: str,
    body: ProfileIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> ProfileOut:
    p = _get_profile(db, pid, current)
    p.name = body.name.strip()
    p.spec = sourcing.normalize_spec(body.spec)
    if body.scoring_rule_id is not None:
        p.scoring_rule_id = body.scoring_rule_id
    if body.source_job_id is not None:
        p.source_job_id = body.source_job_id
    db.commit()
    db.refresh(p)
    return _to_out(p)


@router.delete(
    "/{pid}",
    status_code=204,
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def delete_profile(
    pid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    db.delete(_get_profile(db, pid, current))
    db.commit()


@router.post(
    "/generate",
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def generate_profile(
    body: GenerateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    """一句话 → AI 起草画像 spec(不落库,确认后走 POST 保存)。"""
    return sourcing.generate_profile_draft(db, current.tenant_id, body.prompt)


@router.post(
    "/from-job",
    dependencies=[Depends(require_feature("match.evaluate"))],
)
def profile_from_job(
    body: FromJobIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict:
    """岗位 → 画像草稿(不落库):skills/description 抽取。"""
    job = db.get(Job, body.job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    return {
        "name": f"{job.title}画像",
        "spec": {
            "skills": list(job.skills or [])[:10],
            "keywords": [],
            "nice_to_have": [],
            "min_years": {"entry": 0, "intermediate": 2, "advanced": 5}.get(job.level or "", 2),
            "education": "",
        },
        "source_job_id": job.id,
    }
