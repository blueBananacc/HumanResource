"""后台任务运维 API(只读)。

``GET /scheduler/jobs`` — 注册任务 + 最近一次运行;
``GET /scheduler/jobs/{name}/runs`` — 运行历史。
打点见 ``app.services.jobruns.record_run``。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import JobRun, User
from app.infra.db import get_db
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

# 注册的任务清单(打点 name → 展示元数据)。
REGISTERED_JOBS = [
    {"name": "email_sync_loop", "label": "邮箱同步轮询", "source": "fastapi-loop"},
    {"name": "watch_loop", "label": "触发器+报表订阅调度", "source": "fastapi-loop"},
    {"name": "app.workers.tasks.resume.parse_resume", "label": "简历解析", "source": "celery"},
    {
        "name": "app.workers.tasks.match.evaluate_matches_for_job",
        "label": "岗位匹配评估",
        "source": "celery",
    },
    {
        "name": "app.workers.tasks.notify.send_talent_campaign",
        "label": "外联活动发送",
        "source": "celery",
    },
    {
        "name": "app.workers.tasks.notify.send_report_subscription",
        "label": "报表订阅发送",
        "source": "celery",
    },
    {"name": "idempotency_purge", "label": "幂等键过期清扫", "source": "fastapi-loop"},
]


class JobOut(BaseModel):
    name: str
    label: str
    source: str
    last_status: str | None = None
    last_started_at: str | None = None
    last_duration_ms: int | None = None
    last_message: str | None = None


class RunOut(BaseModel):
    id: str
    name: str
    started_at: str
    duration_ms: int
    status: str
    message: str | None = None
    tenant_id: str | None = None


def _iso(dt) -> str:
    return dt.isoformat() if dt else ""


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    # 每个任务最近一次 run(name 分组取 max started_at 的整行)
    latest_started = (
        select(JobRun.name, func.max(JobRun.started_at).label("mx"))
        .group_by(JobRun.name)
        .subquery()
    )
    rows = db.execute(
        select(JobRun).join(
            latest_started,
            (JobRun.name == latest_started.c.name)
            & (JobRun.started_at == latest_started.c.mx),
        )
    ).scalars().all()
    last = {r.name: r for r in rows}
    out: list[JobOut] = []
    for j in REGISTERED_JOBS:
        r = last.get(j["name"])
        out.append(
            JobOut(
                name=j["name"],
                label=j["label"],
                source=j["source"],
                last_status=r.status if r else None,
                last_started_at=_iso(r.started_at) if r else None,
                last_duration_ms=r.duration_ms if r else None,
                last_message=r.message if r else None,
            )
        )
    return out


@router.get("/jobs/{name}/runs", response_model=list[RunOut])
def job_runs(
    name: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    rows = db.scalars(
        select(JobRun)
        .where(JobRun.name == name)
        .order_by(JobRun.started_at.desc())
        .limit(limit)
    )
    return [
        RunOut(
            id=r.id, name=r.name, started_at=_iso(r.started_at),
            duration_ms=r.duration_ms, status=r.status,
            message=r.message, tenant_id=r.tenant_id,
        )
        for r in rows
    ]
