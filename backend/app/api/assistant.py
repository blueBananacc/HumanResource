"""AI 助手 — FastAPI ↔ pi sidecar 之间的代理层。

前端只跟这里说话,sidecar 在 docker 网内不对外。这层负责:
- 认证(get_current_user)+ license feature 校验(``assistant.agent``)
- 解析当前租户的 LLMProvider,把 key/model 下发给 sidecar(不经过前端、不落盘)
- 签发与当前用户等价的 tool_token,sidecar 调业务工具时原样回带 —
  租户隔离 / 权限 / license 全部由既有中间件判定,sidecar 不做权限决策
- SSE 透传(text_delta / tool_start / tool_end / confirm_required / done / error)
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.config import settings
from app.domain.models import User
from app.infra.db import SessionLocal
from app.infra.security import create_access_token
from app.integrations.llm.registry import resolve_provider

router = APIRouter(
    prefix="/assistant",
    tags=["assistant"],
    dependencies=[Depends(require_feature("assistant.agent"))],
)

_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


class ConfirmIn(BaseModel):
    call_id: str
    approved: bool


def _llm_payload(db: Session, user: User) -> dict:
    """租户当前 provider → sidecar 用的扁平配置。"""
    p = resolve_provider(db, user.tenant_id)
    if p.is_mock:
        return {"model": "mock", "mock": True}
    # model 存的是 LiteLLM 标识(deepseek/deepseek-flash),pi 只需要裸模型名;
    # base_url 原样透传(openai-completions api 会在其后拼 /chat/completions)
    return {
        "model": p.model.split("/", 1)[-1],
        "base_url": p.base_url,
        "api_key": p.api_key,
        "mock": False,
    }


def _user_payload(user: User) -> dict:
    return {
        "tenant_id": user.tenant_id,
        "id": user.id,
        "role": user.role,
        "name": user.email,
    }


def _resolve_llm(user: User) -> dict:
    """短事务里取 LLM 配置 — session 在返回 StreamingResponse 前就关闭,
    避免池连接被 SSE 流(最长 300s)持有。"""
    with SessionLocal() as db:
        return _llm_payload(db, user)


@router.post("/chat")
async def chat(
    body: ChatIn,
    current: User = Depends(get_current_user),
) -> StreamingResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    llm = await run_in_threadpool(_resolve_llm, current)
    payload = {
        "conversation_id": body.conversation_id,
        "message": message,
        "llm": llm,
        "tool_token": create_access_token(subject=current.id),
        "user": _user_payload(current),
    }

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream(
                    "POST", f"{settings.assistant_url}/chat", json=payload
                ) as resp:
                    if resp.status_code != 200:
                        text = (await resp.aread()).decode("utf-8", "replace")[:500]
                        yield (
                            "event: error\n"
                            f'data: {{"message": "助手服务异常({resp.status_code}): {text}"}}\n\n'
                        )
                        return
                    async for line in resp.aiter_lines():
                        yield line + "\n"
        except httpx.HTTPError as e:
            yield (
                "event: error\n"
                f'data: {{"message": "无法连接 AI 助手服务: {e}"}}\n\n'
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/confirm")
async def confirm(
    body: ConfirmIn,
    current: User = Depends(get_current_user),
) -> dict:
    """前端对写操作确认卡的裁决,转发给 sidecar 解除 beforeToolCall 停放。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{settings.assistant_url}/confirm",
                json={
                    "call_id": body.call_id,
                    "approved": body.approved,
                    # 附带身份让 sidecar 校验 call_id 归属,防止越权裁决他人确认
                    "tenant_id": current.tenant_id,
                    "user_id": current.id,
                },
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"助手服务不可达: {e}") from e
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="该操作已超时或被处理")
    resp.raise_for_status()
    return {"ok": True}


@router.get("/conversations")
async def list_conversations(
    current: User = Depends(get_current_user),
) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                f"{settings.assistant_url}/conversations",
                params={"tenant_id": current.tenant_id, "user_id": current.id},
            )
        except httpx.HTTPError:
            return {"items": []}
    return resp.json() if resp.status_code == 200 else {"items": []}


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: str,
    current: User = Depends(get_current_user),
) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                f"{settings.assistant_url}/conversations/{conv_id}",
                params={"tenant_id": current.tenant_id, "user_id": current.id},
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"助手服务不可达: {e}") from e
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="会话不存在")
    resp.raise_for_status()
    return resp.json()


@router.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    current: User = Depends(get_current_user),
) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.delete(
                f"{settings.assistant_url}/conversations/{conv_id}",
                params={"tenant_id": current.tenant_id, "user_id": current.id},
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"助手服务不可达: {e}") from e
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="会话不存在")
    resp.raise_for_status()
