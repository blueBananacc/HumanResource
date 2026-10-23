"""AI 提示词模板测试。

覆盖:
- render_vars 替换 / 未知变量保留
- GET 合并视图(默认 + is_custom)
- PUT 保存 → 版本 +1 + 版本表;render_prompt 命中自定义
- preview 渲染;reset 回默认;rollback 回历史版本
- render_prompt 停用后返回 None(服务回落默认)
- 权限与非注册 key 404
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-prompt")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    PromptTemplate,
    PromptTemplateVersion,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.prompting import (  # noqa: E402
    PROMPT_REGISTRY,
    render_prompt,
    render_vars,
)
from sqlalchemy import select  # noqa: E402


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
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"pt-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _user(db, t: Tenant, *, role: str = "admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        db.query(PromptTemplateVersion).filter(
            PromptTemplateVersion.tenant_id == t.id
        ).delete()
        db.query(PromptTemplate).filter(PromptTemplate.tenant_id == t.id).delete()
        db.query(User).filter(User.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def test_render_vars():
    out = render_vars("你好 {{name}},岗位 {{job}}。", {"name": "张三", "job": "工程师"})
    assert out == "你好 张三,岗位 工程师。"
    # 未提供的变量原样保留
    out = render_vars("{{a}} + {{b}}", {"a": "1"})
    assert out == "1 + {{b}}"


def test_list_returns_registry_defaults(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    try:
        r = client.get("/api/prompt-templates", headers=_hdr(u))
        assert r.status_code == 200, r.text
        items = {x["key"]: x for x in r.json()}
        assert set(items) == set(PROMPT_REGISTRY)
        assert all(not x["is_custom"] for x in items.values())
        assert items["match.evaluate"]["content"] == PROMPT_REGISTRY["match.evaluate"].content
        # 带变量的模板返回变量文档
        gen_vars = {v["name"] for v in items["question.generate"]["variables"]}
        assert {"count", "level", "job_clause", "kinds_clause"} <= gen_vars
    finally:
        _cleanup(db_session, t)


def test_save_preview_reset_flow(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        custom = "自定义匹配提示词 v1,只输出 JSON。"
        r = client.put(
            "/api/prompt-templates/match.evaluate", headers=hdr,
            json={"content": custom},
        )
        assert r.status_code == 200, r.text
        assert r.json()["is_custom"] is True
        assert r.json()["version"] == 1

        # render_prompt 命中自定义
        assert render_prompt(db_session, t.id, "match.evaluate") == custom

        # 再保存 → version 2 + 两条版本记录
        custom2 = "自定义匹配提示词 v2,只输出 JSON。"
        client.put("/api/prompt-templates/match.evaluate", headers=hdr, json={"content": custom2})
        assert render_prompt(db_session, t.id, "match.evaluate") == custom2
        vers = db_session.scalars(
            select(PromptTemplateVersion)
            .join(PromptTemplate, PromptTemplateVersion.template_id == PromptTemplate.id)
            .where(PromptTemplate.tenant_id == t.id)
        ).all()
        assert [v.version for v in vers] == [1, 2]

        # preview 用声明变量的示例值渲染
        r = client.post(
            "/api/prompt-templates/interview.question/preview", headers=hdr,
            json={"vars": {"job_title": "测试岗"}},
        )
        assert r.status_code == 200
        assert "测试岗" in r.json()["rendered"]
        assert "<level>" in r.json()["rendered"]

        # reset → 回默认
        r = client.post("/api/prompt-templates/match.evaluate/reset", headers=hdr)
        assert r.json()["is_custom"] is False
        assert render_prompt(db_session, t.id, "match.evaluate") is None
    finally:
        _cleanup(db_session, t)


def test_rollback(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        client.put("/api/prompt-templates/match.evaluate", headers=hdr,
                   json={"content": "第一版内容,十个字符以上"})
        client.put("/api/prompt-templates/match.evaluate", headers=hdr,
                   json={"content": "第二版内容,十个字符以上"})
        r = client.post(
            "/api/prompt-templates/match.evaluate/rollback", headers=hdr,
            json={"version": 1},
        )
        assert r.status_code == 200
        assert r.json()["content"] == "第一版内容,十个字符以上"
        assert r.json()["version"] == 3
        # 历史版本列表(带 content,前端可在回滚前预览旧版本内容)
        r = client.get("/api/prompt-templates/match.evaluate/versions", headers=hdr)
        assert [v["version"] for v in r.json()] == [3, 2, 1]
        assert r.json()[0]["content"] == "第一版内容,十个字符以上"
        # 不存在的版本 → 404
        r = client.post(
            "/api/prompt-templates/match.evaluate/rollback", headers=hdr,
            json={"version": 99},
        )
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_inactive_template_falls_back(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    try:
        db_session.add(PromptTemplate(
            tenant_id=t.id, key="match.evaluate",
            content="停用中的模板", is_active=False,
        ))
        db_session.commit()
        assert render_prompt(db_session, t.id, "match.evaluate") is None
        # 未注册 key 永远 None
        assert render_prompt(db_session, t.id, "no.such.key") is None
    finally:
        _cleanup(db_session, t)


def test_permission_and_unknown_key(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t, role="viewer")
    admin = _user(db_session, t)
    try:
        r = client.get("/api/prompt-templates", headers=_hdr(u))
        assert r.status_code == 403
        r = client.put("/api/prompt-templates/bogus.key", headers=_hdr(admin),
                       json={"content": "x" * 20})
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)
