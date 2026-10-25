"""幂等键内核(T10) —— 服务层。

参考 Gauzy ``idempotency.service``:一把 key 一次副作用。

- ``claim`` 是唯一入口:插入即锁((tenant_id, scope, key) 唯一元组),
  并发同 key 只有一个能插进去;输家读回行再裁决(replay / in-flight /
  reused / 接管)。
- ``claim`` **用自己的 SessionLocal** —— 锁行必须立即提交才对其它
  并发事务可见,不能与调用方的事务纠缠。
- 锁接管:``locked_at`` 超过 ``STALE_LOCK``(默认 120s)视为死锁,
  ``FOR UPDATE`` 行锁内重校验后续约 —— 进程中途死掉不永久占锁。
- 过期:``expires_at`` 到点由 claim 惰性清理 + ``_watch_loop`` 定时
  ``purge_expired`` 双管。
- ``complete``/``fail`` 对已终态行幂等(settle 崩溃安全);
  ``release`` 拒绝活锁(409)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.models import IdempotencyKey
from app.infra.db import SessionLocal

# 回放窗(默认 24h)与死锁窗口(默认 120s)— 与 Gauzy 同。
DEFAULT_RETENTION_SECONDS = 24 * 60 * 60
MIN_RETENTION_SECONDS = 60
MAX_RETENTION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_STALE_LOCK_MS = 2 * 60 * 1000

Outcome = Literal["claimed", "replayed", "in_flight", "reused_key"]


@dataclass
class ClaimResult:
    outcome: Outcome
    record: IdempotencyKey | None = None
    # replay 时携带
    replay_status: int = 200
    replay_body: Any = None
    replayed_at: datetime | None = None
    # in_flight 时携带
    retry_after_ms: int = 0
    # reused 时携带(只给前缀,不泄漏请求体)
    expected_hash_prefix: str = ""
    actual_hash_prefix: str = ""


def clamp_ttl(seconds: int | None) -> int:
    """per-route ttl 覆盖,clamp 到 [60s, 7d]。"""
    if not isinstance(seconds, (int, float)) or seconds is None:
        return DEFAULT_RETENTION_SECONDS
    return int(min(MAX_RETENTION_SECONDS, max(MIN_RETENTION_SECONDS, seconds)))


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _find_by_key(db: Session, tenant_id: str, scope: str, key: str) -> IdempotencyKey | None:
    return db.scalars(
        select(IdempotencyKey).where(
            IdempotencyKey.tenant_id == tenant_id,
            IdempotencyKey.scope == scope,
            IdempotencyKey.key == key,
        )
    ).first()


def _is_lease_live(row: IdempotencyKey, stale_ms: int) -> bool:
    """行是否仍被活跃请求持有:in_progress 且锁未过 stale 窗。"""
    if row.status != "in_progress" or not row.locked_at:
        return False
    return (_now() - row.locked_at).total_seconds() * 1000 <= stale_ms


def _takeover(db: Session, row: IdempotencyKey, stale_ms: int) -> bool:
    """FOR UPDATE 内重校验死锁 → 续约 locked_at/expires_at。"""
    locked = db.scalars(
        select(IdempotencyKey)
        .where(IdempotencyKey.id == row.id, IdempotencyKey.status == "in_progress")
        .with_for_update()
    ).first()
    if not locked:
        return False
    if locked.locked_at and (_now() - locked.locked_at).total_seconds() * 1000 <= stale_ms:
        return False  # 持锁者还活着
    locked.locked_at = _now()
    locked.expires_at = _now() + timedelta(seconds=DEFAULT_RETENTION_SECONDS)
    db.flush()
    row.locked_at = locked.locked_at
    row.expires_at = locked.expires_at
    return True


def _resolve_existing(
    db: Session,
    row: IdempotencyKey,
    request_hash: str,
    stale_ms: int,
) -> ClaimResult:
    """既有行的裁决 —— 与 Gauzy resolveExisting 同序。"""
    if row.request_hash != request_hash:
        return ClaimResult(
            outcome="reused_key",
            record=row,
            expected_hash_prefix=(row.request_hash or "")[:8],
            actual_hash_prefix=(request_hash or "")[:8],
        )
    if row.status in ("completed", "failed"):
        # 终态行回放:status 缺失时按行状态兜底(failed→500, completed→200)
        return ClaimResult(
            outcome="replayed",
            record=row,
            replay_status=row.response_status
            if row.response_status is not None
            else (500 if row.status == "failed" else 200),
            replay_body=row.response_body,
            replayed_at=row.created_at,
        )
    # in_progress:活锁 → 让调用方等;死锁 → FOR UPDATE 接管
    if _is_lease_live(row, stale_ms):
        held_ms = (_now() - row.locked_at).total_seconds() * 1000 if row.locked_at else 0
        return ClaimResult(
            outcome="in_flight",
            record=row,
            retry_after_ms=max(0, int(stale_ms - held_ms)),
        )
    if _takeover(db, row, stale_ms):
        return ClaimResult(outcome="claimed", record=row)
    return ClaimResult(
        outcome="in_flight", record=row, retry_after_ms=stale_ms
    )


def claim(
    *,
    tenant_id: str,
    user_id: str | None,
    scope: str,
    key: str,
    request_hash: str,
    ttl_seconds: int | None = None,
    resource_type: str | None = None,
    stale_lock_ms: int = DEFAULT_STALE_LOCK_MS,
) -> ClaimResult:
    """认领一把 key,或给出为什么不能领的答案。"""
    retention = clamp_ttl(ttl_seconds)
    db = SessionLocal()
    try:
        existing = _find_by_key(db, tenant_id, scope, key)
        # 惰性过期:过期且锁死 → 删行后按"从未出现"处理
        if (
            existing is not None
            and existing.expires_at <= _now()
            and not _is_lease_live(existing, stale_lock_ms)
        ):
            db.delete(existing)
            db.commit()
            existing = None

        if existing is not None:
            result = _resolve_existing(db, existing, request_hash, stale_lock_ms)
            db.commit()
            return result

        row = IdempotencyKey(
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,
            key=key,
            request_hash=request_hash,
            status="in_progress",
            resource_type=resource_type,
            expires_at=_now() + timedelta(seconds=retention),
            locked_at=_now(),
        )
        try:
            db.add(row)
            db.commit()
            return ClaimResult(outcome="claimed", record=row)
        except IntegrityError:
            db.rollback()
            raced = _find_by_key(db, tenant_id, scope, key)
            if raced is None:
                raise  # 赢家在行可见前被删 —— 抛出原错误,调用方重试即领锁
            result = _resolve_existing(db, raced, request_hash, stale_lock_ms)
            db.commit()
            return result
    finally:
        db.close()


def _settle(
    record_id: str,
    status: str,
    *,
    response_status: int | None,
    response_body: Any,
    resource_id: str | None = None,
    resource_type: str | None = None,
) -> None:
    """写终态 —— 对已终态行幂等(崩溃安全,不覆盖首个答案)。"""
    db = SessionLocal()
    try:
        row = db.get(IdempotencyKey, record_id)
        if row is None or row.status != "in_progress":
            return
        row.status = status
        row.response_status = (
            response_status if response_status is not None else row.response_status
        )
        row.response_body = (
            response_body if response_body is not None else row.response_body
        )
        row.resource_id = resource_id or row.resource_id
        row.resource_type = resource_type or row.resource_type
        db.commit()
    finally:
        db.close()


def complete(
    record_id: str,
    *,
    response_status: int,
    response_body: Any = None,
    resource_id: str | None = None,
    resource_type: str | None = None,
) -> None:
    _settle(
        record_id,
        "completed",
        response_status=response_status,
        response_body=response_body,
        resource_id=resource_id,
        resource_type=resource_type,
    )


def fail(record_id: str, *, response_status: int, response_body: Any = None) -> None:
    """失败也占住 key —— 坏请求被回放同样的拒绝,而不是无限重试。"""
    _settle(record_id, "failed", response_status=response_status, response_body=response_body)


def release(record_id: str, tenant_id: str, stale_lock_ms: int = DEFAULT_STALE_LOCK_MS) -> IdempotencyKey:
    """操作员释放一把 key(真删 —— 标 released 仍占唯一元组)。

    活锁拒绝(409 由 API 层翻译)。返回被删的行供审计。"""
    db = SessionLocal()
    try:
        row = db.scalars(
            select(IdempotencyKey).where(
                IdempotencyKey.id == record_id,
                IdempotencyKey.tenant_id == tenant_id,
            )
        ).first()
        if row is None:
            raise LookupError("not found")
        if _is_lease_live(row, stale_lock_ms):
            held_ms = (_now() - row.locked_at).total_seconds() * 1000 if row.locked_at else 0
            raise RuntimeError(f"in_flight:{int(stale_ms - held_ms)}")
        db.delete(row)
        db.commit()
        return row
    finally:
        db.close()


def purge_expired(limit: int = 500, stale_lock_ms: int = DEFAULT_STALE_LOCK_MS) -> int:
    """有界清扫:终态过期行 + 锁死过期行。谓词双写(选一次、删时再断言一次)。"""
    db = SessionLocal()
    try:
        now = _now()
        abandoned_before = now - timedelta(milliseconds=stale_lock_ms)
        terminal = db.scalars(
            select(IdempotencyKey.id).where(
                IdempotencyKey.expires_at < now,
                IdempotencyKey.status.in_(["completed", "failed"]),
            ).limit(limit)
        ).all()
        remaining = limit - len(terminal)
        abandoned: list[str] = []
        if remaining > 0:
            abandoned = db.scalars(
                select(IdempotencyKey.id).where(
                    IdempotencyKey.expires_at < now,
                    IdempotencyKey.status == "in_progress",
                    IdempotencyKey.locked_at < abandoned_before,
                ).limit(remaining)
            ).all()
        if terminal:
            db.execute(
                delete(IdempotencyKey).where(
                    IdempotencyKey.id.in_(terminal),
                    IdempotencyKey.expires_at < now,
                    IdempotencyKey.status.in_(["completed", "failed"]),
                )
            )
        if abandoned:
            db.execute(
                delete(IdempotencyKey).where(
                    IdempotencyKey.id.in_(abandoned),
                    IdempotencyKey.expires_at < now,
                    IdempotencyKey.status == "in_progress",
                    IdempotencyKey.locked_at < abandoned_before,
                )
            )
        db.commit()
        return len(terminal) + len(abandoned)
    finally:
        db.close()


def stats(db: Session, tenant_id: str) -> dict:
    """运维视图:按状态计数 + 最近 24h 新建数。"""
    rows = db.execute(
        select(IdempotencyKey.status, func.count())
        .where(IdempotencyKey.tenant_id == tenant_id)
        .group_by(IdempotencyKey.status)
    ).all()
    by_status = {s: c for s, c in rows}
    recent = db.scalar(
        select(func.count())
        .select_from(IdempotencyKey)
        .where(
            IdempotencyKey.tenant_id == tenant_id,
            IdempotencyKey.created_at >= _now() - timedelta(hours=24),
        )
    ) or 0
    return {
        "by_status": by_status,
        "total": sum(by_status.values()),
        "recent_24h": recent,
    }
