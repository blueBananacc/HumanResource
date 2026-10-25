"""模型调用留痕运维 API(T11)。

- ``GET  /api/model-call-logs``         分页+过滤(manage_settings)
- ``GET  /api/model-call-logs/stats``   总量/错误率/分组/延迟分位
- ``POST /api/model-call-logs/report``  助手 sidecar 回报每轮调用(tool_token=JWT)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, paginate_params
from app.api.auth import get_current_user
from app.domain.models import ModelCallLog, User
from app.infra.db import get_db
from app.services.model_calls import write_model_call
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission

router = APIRouter(prefix="/model-call-logs", tags=["model-call-logs"])


class LogOut(BaseModel):
    id: str
    call_type: str
    provider: str
    model: str
    prompt_key: str | None
    prompt_version: str | None
    schema_version: str
    input_digest: str
    output: dict | list | None
    raw_output: str | None
    error_kind: str
    error_message: str | None
    latency_ms: int
    user_id: str | None
    created_at: datetime


def _out(r: ModelCallLog) -> LogOut:
    return LogOut(
        id=r.id,
        call_type=r.call_type,
        provider=r.provider,
        model=r.model,
        prompt_key=r.prompt_key,
        prompt_version=r.prompt_version,
        schema_version=r.schema_version,
        input_digest=r.input_digest,
        output=r.output,
        raw_output=r.raw_output,
        error_kind=r.error_kind,
        error_message=r.error_message,
        latency_ms=r.latency_ms,
        user_id=r.user_id,
        created_at=r.created_at,
    )


@router.get("/", response_model=PageOut[LogOut])
def list_logs(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    call_type: str | None = Query(None, max_length=48),
    error_kind: str | None = Query(None, max_length=32),
    has_error: bool | None = Query(None),
    hours: int | None = Query(None, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
) -> PageOut[LogOut]:
    limit, offset, _q = p
    stmt = select(ModelCallLog).where(ModelCallLog.tenant_id == current.tenant_id)
    if call_type:
        stmt = stmt.where(ModelCallLog.call_type == call_type)
    if error_kind:
        stmt = stmt.where(ModelCallLog.error_kind == error_kind)
    if has_error is True:
        stmt = stmt.where(ModelCallLog.error_kind != "")
    elif has_error is False:
        stmt = stmt.where(ModelCallLog.error_kind == "")
    if hours:
        stmt = stmt.where(
            ModelCallLog.created_at >= datetime.utcnow() - timedelta(hours=hours)
        )
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(desc(ModelCallLog.created_at)).limit(limit).offset(offset)
    ).all()
    return PageOut[LogOut](
        items=[_out(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get("/stats")
def stats(
    hours: int = Query(24 * 7, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    base = select(ModelCallLog).where(
        ModelCallLog.tenant_id == current.tenant_id,
        ModelCallLog.created_at >= since,
    )
    rows = db.scalars(base).all()

    total = len(rows)
    errors = sum(1 for r in rows if r.error_kind)
    latencies = sorted(r.latency_ms for r in rows)
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    avg = round(sum(latencies) / total) if total else 0

    by_type: dict[str, dict] = {}
    for r in rows:
        t = by_type.setdefault(
            r.call_type, {"count": 0, "errors": 0, "avg_latency_ms": 0, "_lat": 0}
        )
        t["count"] += 1
        t["errors"] += 1 if r.error_kind else 0
        t["_lat"] += r.latency_ms
    for t in by_type.values():
        t["avg_latency_ms"] = round(t.pop("_lat") / t["count"])

    by_day: dict[str, int] = {}
    for r in rows:
        day = r.created_at.strftime("%Y-%m-%d")
        by_day[day] = by_day.get(day, 0) + 1

    return {
        "window_hours": hours,
        "total": total,
        "errors": errors,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "avg_latency_ms": avg,
        "p95_latency_ms": p95,
        "by_type": [
            {"call_type": k, **v}
            for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["count"])
        ],
        "by_day": [{"day": k, "count": by_day[k]} for k in sorted(by_day)],
    }


class ReportIn(BaseModel):
    """助手 sidecar 每轮调用回报(带 tool_token → get_current_user)。"""

    call_type: str = Field("assistant.turn", max_length=48)
    model: str = Field("", max_length=128)
    provider: str = Field("", max_length=64)
    input_digest: str = Field("", max_length=16)
    latency_ms: int = Field(0, ge=0, le=3600_000)
    output: dict | None = None
    error_kind: str = Field("", max_length=32)
    error_message: str | None = Field(None, max_length=500)


@router.post("/report", status_code=201)
def report_call(
    body: ReportIn,
    current: User = Depends(get_current_user),
) -> dict:
    row_id = write_model_call(
        tenant_id=current.tenant_id,
        user_id=current.id,
        call_type=body.call_type,
        provider=body.provider,
        model=body.model,
        input_digest=body.input_digest,
        output=body.output,
        error_kind=body.error_kind,
        error_message=body.error_message,
        latency_ms=body.latency_ms,
    )
    if row_id is None:
        raise HTTPException(500, "留痕写入失败")
    return {"id": row_id}
