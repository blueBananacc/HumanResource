"""MCP(Model Context Protocol)接入 — 外部 AI 客户端的令牌管理与协议代理。

- ``/me/mcp-tokens``:用户自助签发/吊销个人访问令牌(PAT 模式,sha256 入库,
  明文仅签发时返回一次 —— 与 survey_links/invite_tokens 同一惯例)
- ``POST /mcp/token-exchange``:把 mcp 令牌兑换成等价 tool_token + 用户
  上下文。**仅 sidecar 可调** —— 需带 ``X-Internal-Token`` 内部密钥(由
  JWT_SECRET 派生),否则 PAT 会被放大成全量用户 JWT(范围越权 + 吊销不
  生效 + 可再签新 PAT 持久化)。
- ``POST /mcp``:JSON-RPC 代理到 sidecar(保持 sidecar 只在 docker 网内)
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.config import settings
from app.domain.models import McpToken, User
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.infra.security import create_access_token
from app.services.audit import write_audit

router = APIRouter(prefix="/mcp", tags=["mcp"])
me_router = APIRouter(
    prefix="/me/mcp-tokens",
    tags=["mcp-tokens"],
    dependencies=[Depends(require_feature("assistant.agent"))],
)

_TOKEN_PREFIX = "hrm_"
# 每用户同时有效的令牌上限 —— PAT 是长期凭证,数量失控会放大泄漏面。
_MAX_ACTIVE_TOKENS = 20

# token-exchange 的 sidecar↔backend 内部共享密钥的 HMAC 消息串。
_INTERNAL_MSG = b"hronfire/mcp/token-exchange"
# 兑换出的 tool_token 寿命:sidecar 会话缓存只存 ~10s 就会重新兑换,
# 短寿命把"已吊销 PAT 但旧 JWT 仍可用"的窗口压到分钟级。
_TOOL_TOKEN_MINUTES = 15


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _internal_key() -> str:
    """内部共享密钥,由 JWT_SECRET 派生 —— 不引入新环境变量。

    威胁模型对齐:能算出该值的攻击者已持有 JWT_SECRET,本来就能直接伪造
    任意 JWT,派生值不新增攻击面;JWT_SECRET 轮换时内部密钥同步轮换。
    """
    return hmac.new(
        settings.jwt_secret.encode(), _INTERNAL_MSG, hashlib.sha256
    ).hexdigest()


def _require_internal(request: Request) -> None:
    got = request.headers.get("x-internal-token", "")
    if not secrets.compare_digest(got, _internal_key()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "仅限内部调用")


class TokenIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name 不能为空白")
        return v


class TokenOut(BaseModel):
    id: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None

    model_config = {"from_attributes": True}


class TokenCreatedOut(TokenOut):
    token: str


class ExchangeIn(BaseModel):
    token: str = Field(min_length=8, max_length=128)


@me_router.get("", response_model=list[TokenOut])
def list_tokens(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    rows = db.scalars(
        select(McpToken)
        .where(McpToken.user_id == current.id)
        .order_by(McpToken.created_at.desc())
    ).all()
    return rows


@me_router.post("", response_model=TokenCreatedOut, status_code=status.HTTP_201_CREATED)
def create_token(
    body: TokenIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    active = db.scalar(
        select(func.count(McpToken.id)).where(
            McpToken.user_id == current.id, McpToken.revoked_at.is_(None)
        )
    ) or 0
    if active >= _MAX_ACTIVE_TOKENS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"最多同时持有 {_MAX_ACTIVE_TOKENS} 个有效令牌,请先吊销不用的",
        )
    plaintext = _TOKEN_PREFIX + secrets.token_hex(24)
    row = McpToken(
        tenant_id=current.tenant_id,
        user_id=current.id,
        name=body.name,
        token_hash=_hash(plaintext),
    )
    db.add(row)
    db.flush()
    write_audit(db, actor=current, entity_type="mcp_token", entity_id=row.id, action="create", request=request)
    db.commit()
    return TokenCreatedOut(
        id=row.id,
        name=row.name,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        token=plaintext,
    )


@me_router.delete("/{tid}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_token(
    tid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    row = db.get(McpToken, tid)
    if row is None or row.tenant_id != current.tenant_id or row.user_id != current.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "令牌不存在")
    if row.revoked_at is None:
        row.revoked_at = _now()
        write_audit(db, actor=current, entity_type="mcp_token", entity_id=row.id, action="revoke", request=request)
        db.commit()


@router.post("/token-exchange", dependencies=[Depends(_require_internal)])
def token_exchange(body: ExchangeIn, db: Session = Depends(get_db)):
    """sidecar 内部:把 mcp 令牌兑换成等价 tool_token + 用户上下文。

    不做 JWT 鉴权,但要求 ``X-Internal-Token`` 内部密钥 —— 公开兑换会让
    PAT 越出 MCP 工具面(兑换的 JWT 能打全部业务 API、可再签新 PAT 持久化,
    且已发 JWT 在 PAT 吊销后仍存活到期)。兑换的 tool_token 短寿命
    (``_TOOL_TOKEN_MINUTES``),sidecar 会话缓存到期后重新兑换,
    使吊销在分钟级内真正生效。
    """
    row = db.scalars(
        select(McpToken).where(McpToken.token_hash == _hash(body.token))
    ).first()
    if row is None or row.revoked_at is not None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "无效或已吊销的令牌")
    if not is_feature_enabled(db, "assistant.agent"):
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "assistant.agent 未开通")
    user = db.get(User, row.user_id)
    if user is None or user.status == "disabled" or user.tenant_id != row.tenant_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌所属用户不可用")
    # 节流更新 last_used_at(60s 内不重复写)
    now = _now()
    if row.last_used_at is None or (now - row.last_used_at).total_seconds() > 60:
        row.last_used_at = now
        db.commit()
    return {
        "tool_token": create_access_token(
            subject=user.id, via="mcp", expires_minutes=_TOOL_TOKEN_MINUTES
        ),
        "user": {
            "tenant_id": user.tenant_id,
            "id": user.id,
            "role": user.role,
            "name": user.email,
        },
    }


# 与 sidecar express.json(1mb)同口径:代理先拒绝超大 body,避免未认证请求
# 在鉴权前就消耗 backend 内存 + 一次 sidecar 往返。略大于 1mb 留头部余量。
_MCP_MAX_BODY = 1_200_000


@router.post("")
async def mcp_proxy(request: Request):
    """JSON-RPC 直通代理 — sidecar 不出网,外部 MCP 客户端只到这里。

    转发 body + Authorization;sidecar 负责令牌兑换/工具执行/协议应答。
    """
    # 声明长度先挡(chunked/谎报长度再按实际挡),两个都不在 413 前鉴权
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > _MCP_MAX_BODY:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "请求体过大")
    payload = await request.body()
    if len(payload) > _MCP_MAX_BODY:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "请求体过大")
    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{settings.assistant_url}/mcp",
                content=payload,
                headers={**headers, "content-type": "application/json"},
            )
        except httpx.HTTPError as e:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32603, "message": f"助手服务不可达: {e}"}},
                status_code=502,
            )
    # 202 无响应体(MCP notifications 应答)要原样透传,不能塞 {}
    if not resp.content:
        return Response(status_code=resp.status_code)
    try:
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except ValueError:
        return Response(content=resp.content, status_code=resp.status_code)
