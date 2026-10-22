"""AI 助手代理层测试。

sidecar(pi-agent-core Node 服务)不在测试里起 —— 用 ``httpx.MockTransport``
替换 :mod:`app.api.assistant` 里的 ``AsyncClient``,验证:

1. /chat 未认证 → 401
2. /chat SSE 帧原样透传,且转发 payload 里带租户/用户信息、tool_token、LLM 配置
3. /confirm 转发裁决;sidecar 404 → 代理也 404
4. license feature 关闭(assistant.agent)→ 402
"""
from __future__ import annotations

import json
import os
import types
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-assistant")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-assistant"

from app.api import assistant as assistant_api  # noqa: E402
from app.api import license as license_api  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import hash_password  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def user(db_session):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(name=f"T-{suffix}")
    db_session.add(t)
    db_session.flush()
    u = User(
        tenant_id=t.id,
        email=f"hr-{suffix}@t.dev",
        role="admin",
        password_hash=hash_password("pw"),
    )
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


def _login(client, user) -> dict:
    r = client.post("/api/auth/login", json={"email": user.email, "password": "pw"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _mock_httpx(handler):
    """把 assistant 模块里的 httpx.AsyncClient 换成 MockTransport 版本。"""

    class _Client(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler))

    return types.SimpleNamespace(
        AsyncClient=_Client,
        HTTPError=httpx.HTTPError,
        Timeout=httpx.Timeout,
    )


def test_chat_requires_auth(client):
    r = client.post("/api/assistant/chat", json={"message": "hi"})
    assert r.status_code == 401


def test_chat_streams_sse_and_forwards_context(client, user, monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        body = (
            "event: conversation\n"
            'data: {"conversation_id":"conv-1"}\n\n'
            "event: text_delta\n"
            'data: {"delta":"你好"}\n\n'
            "event: done\n"
            'data: {"conversation_id":"conv-1"}\n\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(assistant_api, "httpx", _mock_httpx(handler))
    headers = _login(client, user)
    r = client.post(
        "/api/assistant/chat",
        json={"message": "你好"},
        headers=headers,
    )
    assert r.status_code == 200
    assert "event: text_delta" in r.text
    assert "你好" in r.text

    # sidecar 收到的上下文:租户/用户、工具 token、LLM(mock)
    assert captured["url"].endswith("/chat")
    p = captured["payload"]
    assert p["message"] == "你好"
    assert p["user"]["tenant_id"] == user.tenant_id
    assert p["user"]["id"] == user.id
    assert p["tool_token"]
    # 未配置 provider 时走 mock
    assert p["llm"]["mock"] is True


def test_confirm_forwards_and_maps_404(client, user, monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(404, json={"ok": False})

    monkeypatch.setattr(assistant_api, "httpx", _mock_httpx(handler))
    headers = _login(client, user)
    r = client.post(
        "/api/assistant/confirm",
        json={"call_id": "call-1", "approved": True},
        headers=headers,
    )
    assert r.status_code == 404
    # sidecar 按 (call_id, tenant_id, user_id) 三元组校验归属,防跨用户裁决
    assert seen["payload"] == {
        "call_id": "call-1",
        "approved": True,
        "tenant_id": user.tenant_id,
        "user_id": user.id,
    }


def test_feature_gate_blocks_when_disabled(client, user, monkeypatch):
    monkeypatch.setattr(license_api, "is_feature_enabled", lambda db, f: False)
    headers = _login(client, user)
    r = client.post(
        "/api/assistant/chat",
        json={"message": "hi"},
        headers=headers,
    )
    assert r.status_code == 402
    assert "assistant.agent" in r.json()["detail"]
