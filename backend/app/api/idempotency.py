"""幂等键(T10) —— ``@idempotent`` 端点装饰器 + 运维 API。

语义移植自 Gauzy ``IdempotencyInterceptor`` + ``IdempotencyService``:

- 装饰器直包端点函数;无 key 且非 required → 直通,行为与声明前一致。
- claim 走 ``services.idempotency``(独立 SessionLocal,锁行即提)。
- REPLAY:<400 原样回放 body + ``Idempotency-Replayed: true`` +
  ``Idempotency-Original-Request``;>=400 重建 HTTPException,
  让重试看到与首次相同的错误信封。
- IN_FLIGHT → 409 + ``Retry-After``;REUSED_KEY → 409 + 请求指纹前缀对。
- settle 失败只记日志 —— 工作已做完,不能让"存不上响应"诱发重试。

``key_field``:兜底键位 —— 端点 body 上的某字段天然唯一时(如
assistant confirm 的 ``call_id``),客户端不传 header 也受保护。
"""
from __future__ import annotations

import functools
import inspect
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import IdempotencyKey, User
from app.infra.db import get_db
from app.services import idempotency as svc
from app.services.idempotency_policy import (
    IDEMPOTENCY_KEY_HEADER,
    ORIGINAL_REQUEST_HEADER,
    REPLAYED_HEADER,
    RETRY_AFTER_HEADER,
    build_request_hash,
    hash_prefix,
    normalize_key,
    serialize_for_storage,
    validate_key,
)
from app.services.permissions import PERM_MANAGE_SETTINGS, require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/idempotency-keys", tags=["idempotency"])


def _resolve_user(kwargs: dict[str, Any]) -> User | None:
    """从端点 kwargs 里捞出已解析的当前用户(依赖先于函数体解析完毕)。"""
    for v in kwargs.values():
        if isinstance(v, User):
            return v
    return None


def _fallback_key(kwargs: dict[str, Any], field: str) -> str | None:
    """在端点参数里找兜底键(如 body.call_id / body.idempotency_key)。"""
    for v in kwargs.values():
        if v is None:
            continue
        val = getattr(v, field, None)
        if val is None and isinstance(v, dict):
            val = v.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _error_body(exc: Exception) -> Any:
    """把端点抛出的异常存成回放体 —— FastAPI 对 HTTPException 的信封是 {"detail": ...}。"""
    detail = getattr(exc, "detail", None)
    if detail is not None:
        return {"detail": detail}
    return {"detail": "Internal Server Error"}


def _quiet(fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        logger.error("idempotency settle failed: %s", e)


def idempotent(
    *,
    scope: str,
    required: bool = False,
    ttl_seconds: int | None = None,
    resource_type: str | None = None,
    key_field: str | None = None,
):
    """声明端点可在 ``Idempotency-Key`` 下安全重试。

    ``key_field``:header 缺失时从端点参数取兜底键(确认卡的 call_id)。
    """

    def deco(fn):
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        for name, ann in (("request", Request), ("response", Response)):
            if name not in sig.parameters:
                params.append(
                    inspect.Parameter(
                        name, inspect.Parameter.KEYWORD_ONLY,
                        annotation=ann, default=None,
                    )
                )

        # 装饰器注入、原函数并不声明的参数 —— 调 fn 前必须剥掉,
        # 否则 TypeError: fn() got an unexpected keyword argument 'response'
        injected = {
            p.name for p in params
        } - set(sig.parameters)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            request: Request | None = kwargs.get("request")
            response: Response | None = kwargs.get("response")
            call_kwargs = {k: v for k, v in kwargs.items() if k not in injected}
            method = (request.method if request else "").upper()
            if method in ("GET", "HEAD", "OPTIONS"):
                return await _call(fn, *args, **call_kwargs)

            key = normalize_key(request.headers.get(IDEMPOTENCY_KEY_HEADER) if request else None)
            if key is None and key_field:
                fb = _fallback_key(kwargs, key_field)
                if fb is not None:
                    # 命名空间前缀:兜底键(如 call_id)与客户端 UUID 分域,
                    # 且保证 ≥8 字符("confirm:call-1" = 14)
                    key = f"{key_field}:{fb}"
            if key is None:
                if required:
                    raise HTTPException(
                        400,
                        f"该操作需要 {IDEMPOTENCY_KEY_HEADER} 请求头以安全重试",
                    )
                return await _call(fn, *args, **call_kwargs)
            bad = validate_key(key)
            if bad:
                raise HTTPException(
                    400, f"{IDEMPOTENCY_KEY_HEADER} 不可用({bad},需 8–255 字符)"
                )

            raw_body = await request.body() if request is not None else None
            request_hash = build_request_hash(
                method=method,
                path=request.url.path if request else "",
                query=request.url.query if request else "",
                raw_body=raw_body,
            )
            user = _resolve_user(kwargs)
            result = svc.claim(
                tenant_id=user.tenant_id if user else "public",
                user_id=user.id if user else None,
                scope=scope,
                key=key,
                request_hash=request_hash,
                ttl_seconds=ttl_seconds,
                resource_type=resource_type,
            )

            if result.outcome == "replayed":
                if response is not None:
                    response.headers[REPLAYED_HEADER] = "true"
                    if result.replayed_at:
                        response.headers[ORIGINAL_REQUEST_HEADER] = result.replayed_at.isoformat()
                if result.replay_status and result.replay_status >= 400:
                    body = result.replay_body
                    detail = body.get("detail") if isinstance(body, dict) else body
                    # HTTPException 绕过注入的 Response —— 回放标记走异常 headers
                    err_headers = {REPLAYED_HEADER: "true"}
                    if result.replayed_at:
                        err_headers[ORIGINAL_REQUEST_HEADER] = (
                            result.replayed_at.isoformat()
                        )
                    raise HTTPException(
                        status_code=result.replay_status,
                        detail=detail,
                        headers=err_headers,
                    )
                if response is not None:
                    response.status_code = result.replay_status or 200
                return result.replay_body

            if result.outcome == "in_flight":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="同键请求仍在执行中,请稍后重试",
                    headers={RETRY_AFTER_HEADER: str(max(1, result.retry_after_ms // 1000))},
                )

            if result.outcome == "reused_key":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "IDEMPOTENCY_KEY_REUSED",
                        "message": "该幂等键已用于另一请求",
                        "expected": result.expected_hash_prefix,
                        "actual": result.actual_hash_prefix,
                    },
                )

            # claimed → 执行
            record = result.record
            try:
                value = await _call(fn, *args, **call_kwargs)
            except Exception as e:  # noqa: BLE001
                stored, _ = serialize_for_storage(_error_body(e))
                _quiet(
                    lambda: svc.fail(
                        record.id,
                        response_status=getattr(e, "status_code", 500),
                        response_body=stored,
                    )
                )
                raise

            out_status = getattr(response, "status_code", None) or (
                201 if method == "POST" else 200
            )
            if isinstance(value, Response):
                out_status = value.status_code
                try:
                    body_val = jsonable_encoder(
                        __import__("json").loads(value.body.decode("utf-8"))
                    )
                except Exception:  # noqa: BLE001
                    body_val = None
            else:
                body_val = jsonable_encoder(value)
            stored, dropped = serialize_for_storage(body_val)
            if dropped:
                logger.warning("idempotency %s: response too large, status-only replay", scope)
            rid = None
            if isinstance(body_val, dict) and isinstance(body_val.get("id"), str):
                rid = body_val["id"]
            _quiet(
                lambda: svc.complete(
                    record.id,
                    response_status=out_status,
                    response_body=stored,
                    resource_id=rid,
                    resource_type=resource_type,
                )
            )
            return value

        wrapper.__signature__ = sig.replace(parameters=params)
        return wrapper

    return deco


async def _call(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await run_in_threadpool(fn, *args, **kwargs)


# ─────────────────────────── 运维 API ───────────────────────────


class KeyOut(BaseModel):
    id: str
    scope: str
    key: str
    status: str
    response_status: int | None
    resource_type: str | None
    resource_id: str | None
    user_id: str | None
    request_hash: str
    expires_at: datetime | None
    locked_at: datetime | None
    created_at: datetime | None


def _out(k: IdempotencyKey) -> KeyOut:
    return KeyOut(
        id=k.id, scope=k.scope, key=k.key, status=k.status,
        response_status=k.response_status, resource_type=k.resource_type,
        resource_id=k.resource_id, user_id=k.user_id,
        request_hash=hash_prefix(k.request_hash),  # 只给前缀,不泄漏指纹全量
        expires_at=k.expires_at, locked_at=k.locked_at, created_at=k.created_at,
    )


@router.get("", response_model=list[KeyOut])
def list_keys(
    scope: str | None = Query(default=None),
    status_: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    stmt = select(IdempotencyKey).where(IdempotencyKey.tenant_id == current.tenant_id)
    if scope:
        stmt = stmt.where(IdempotencyKey.scope == scope)
    if status_:
        stmt = stmt.where(IdempotencyKey.status == status_)
    if q:
        stmt = stmt.where(IdempotencyKey.key.ilike(f"%{q}%"))
    rows = db.scalars(
        stmt.order_by(IdempotencyKey.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return [_out(k) for k in rows]


@router.get("/stats")
def key_stats(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    return svc.stats(db, current.tenant_id)


@router.get("/{kid}", response_model=KeyOut)
def get_key(
    kid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    row = db.scalars(
        select(IdempotencyKey).where(
            IdempotencyKey.id == kid, IdempotencyKey.tenant_id == current.tenant_id
        )
    ).first()
    if row is None:
        raise HTTPException(404, "幂等键不存在")
    return _out(row)


@router.delete("/{kid}")
def release_key(
    kid: str,
    current: User = Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    try:
        svc.release(kid, current.tenant_id)
    except LookupError:
        raise HTTPException(404, "幂等键不存在") from None
    except RuntimeError as e:
        msg = str(e)
        retry_ms = int(msg.split(":", 1)[1]) if msg.startswith("in_flight:") else 0
        raise HTTPException(
            status_code=409,
            detail="该键仍在执行中,不能释放",
            headers={RETRY_AFTER_HEADER: str(max(1, retry_ms // 1000))},
        ) from None
    return {"ok": True}
