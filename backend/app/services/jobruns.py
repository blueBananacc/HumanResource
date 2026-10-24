"""后台任务运行打点(MintHCM SchedulersJobs 移植)。

``record_run`` 绝不抛异常 —— 打点失败不能拖垮被点任务本身。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.domain.models import JobRun

logger = logging.getLogger(__name__)


def record_run(
    db: Session,
    name: str,
    started_at: datetime,
    status: str,
    message: str | None = None,
    tenant_id: str | None = None,
) -> None:
    try:
        finished = datetime.now(UTC)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        db.add(
            JobRun(
                tenant_id=tenant_id,
                name=name,
                started_at=started_at.replace(tzinfo=None),
                duration_ms=max(0, int((finished - started_at).total_seconds() * 1000)),
                status=status,
                message=(message or "")[:256] or None,
            )
        )
        db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("job_run 打点失败 name=%s: %s", name, e)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
