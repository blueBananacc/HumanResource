"""邮件模板测试。

覆盖:
- GET 合并视图(默认 subject/text/html)
- PUT 保存 → render_email 命中;html 侧变量被 escape
- preview(未落库内容可直接预览);reset 回默认
- render_email 无模板返回 None;停用返回 None
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-mail")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import Base, EmailTemplate, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.mail_templates import EMAIL_SPECS, render_email  # noqa: E402


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
    t = Tenant(name=f"et-{uuid.uuid4().hex[:8]}")
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
        db.query(EmailTemplate).filter(EmailTemplate.tenant_id == t.id).delete()
        db.query(User).filter(User.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def test_list_returns_defaults(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    try:
        r = client.get("/api/email-templates", headers=_hdr(u))
        assert r.status_code == 200, r.text
        items = {x["key"]: x for x in r.json()}
        assert set(items) == set(EMAIL_SPECS)
        inv = items["interview.invite"]
        assert inv["is_custom"] is False
        assert "{{job_title}}" in inv["subject"]
        assert "{{candidate_name}}" in inv["body_text"]
        assert inv["body_html"] is not None
    finally:
        _cleanup(db_session, t)


def test_save_render_reset(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        # render_email 无模板 → None(走内置文案)
        assert render_email(db_session, t.id, "interview.invite", {}) is None

        r = client.put(
            "/api/email-templates/interview.invite", headers=hdr,
            json={
                "subject": "定制主题:{{job_title}}",
                "body_text": "{{candidate_name}} 你好,链接 {{invite_url}}",
                "body_html": "<p>{{candidate_name}} <b>你好</b></p>",
            },
        )
        assert r.status_code == 200
        assert r.json()["is_custom"] is True

        out = render_email(
            db_session, t.id, "interview.invite",
            {
                "candidate_name": "张<b>三</b>",
                "job_title": "工程师",
                "invite_url": "https://x/i/t?t=1&y=2",
            },
        )
        assert out is not None
        subject, text, html = out
        assert subject == "定制主题:工程师"
        assert "张<b>三</b>" in text  # 纯文本不 escape
        assert "张&lt;b&gt;三&lt;/b&gt;" in html  # html 侧 escape

        # 停用 → 回落 None
        tpl = db_session.query(EmailTemplate).filter_by(tenant_id=t.id).first()
        tpl.is_active = False
        db_session.commit()
        assert render_email(db_session, t.id, "interview.invite", {}) is None

        r = client.post("/api/email-templates/interview.invite/reset", headers=hdr)
        assert r.json()["is_custom"] is False
    finally:
        _cleanup(db_session, t)


def test_preview_without_save(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    try:
        r = client.post(
            "/api/email-templates/interview.done/preview", headers=_hdr(u),
            json={
                "subject": "预览 {{job_title}}",
                "body_text": "正文 {{candidate_name}}",
                "body_html": "<p>{{candidate_name}}</p>",
                "vars": {"job_title": "岗位A", "candidate_name": "李<x>"},
            },
        )
        assert r.status_code == 200
        assert r.json()["subject"] == "预览 岗位A"
        assert "李<x>" in r.json()["text"]
        assert "李&lt;x&gt;" in r.json()["html"]
    finally:
        _cleanup(db_session, t)


def test_viewer_denied_and_unknown_key(client, db_session):
    t = _tenant(db_session)
    v = _user(db_session, t, role="viewer")
    admin = _user(db_session, t)
    try:
        r = client.get("/api/email-templates", headers=_hdr(v))
        assert r.status_code == 403
        r = client.put("/api/email-templates/bogus", headers=_hdr(admin),
                       json={"subject": "s", "body_text": "x" * 20})
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)
