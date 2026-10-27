# -*- coding: utf-8 -*-
"""招聘早报聚合端点测试:分节计数/条目、跳过逻辑、记忆中枢块、租户隔离。"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-briefing")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    DecisionFeedback,
    InterviewScorecard,
    Job,
    JobApplication,
    JobOffer,
    Resume,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def client():
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"brf-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t) -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@x.com",
        password_hash=hash_password("p1234567"),
        role="admin",
        status="active",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u):
    return {"Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"}


def _seed(db, t):
    """造:1 新简历 / 1 今日面试 / 1 待跟进 Offer / 1 滞留申请。"""
    job = Job(tenant_id=t.id, title="后端工程师", status="open")
    cand = Candidate(tenant_id=t.id, name="张三")
    db.add_all([job, cand])
    db.commit()
    db.refresh(job)
    db.refresh(cand)

    db.add(
        Resume(
            tenant_id=t.id,
            candidate_id=cand.id,
            file_name="张三.pdf",
            file_size=1024,
            file_mime="application/pdf",
            storage_key="test/t1.pdf",
            source="upload",
        )
    )
    db.add(
        InterviewScorecard(
            tenant_id=t.id,
            candidate_id=cand.id,
            job_id=job.id,
            title="一面",
            scheduled_at=datetime.now(UTC).replace(tzinfo=None),
            status="open",
        )
    )
    db.add(JobOffer(tenant_id=t.id, candidate_id=cand.id, job_id=job.id, position="后端工程师", status="sent"))
    app_ = JobApplication(
        tenant_id=t.id,
        candidate_id=cand.id,
        job_id=job.id,
        stage="interview",
        stage_changed_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10),
    )
    db.add(app_)
    db.commit()
    db.refresh(app_)
    # 决策反馈 → 记忆中枢
    db.add(
        DecisionFeedback(
            tenant_id=t.id,
            candidate_id=cand.id,
            job_id=job.id,
            application_id=app_.id,
            decision="hired",
            to_stage="hired",
            criteria_snapshot=[{"key": "k1", "name": "沟通协作", "level": 3, "weight": 30}],
        )
    )
    db.commit()
    return job, cand


def test_briefing_empty_tenant_quiet(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.get("/api/briefing/today", headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["has_items"] is False
    assert all(s["count"] == 0 for s in body["sections"])
    assert body["memory"]["decisions_total"] == 0
    assert body["memory"]["confidence"] == "low"


def test_briefing_full_sections(client, db):
    t = _tenant(db)
    u = _user(db, t)
    _seed(db, t)
    r = client.get("/api/briefing/today", headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["has_items"] is True
    counts = {s["key"]: s["count"] for s in body["sections"]}
    assert counts["new_resumes"] == 1
    assert counts["today_interviews"] == 1
    assert counts["pending_offers"] == 1
    assert counts["stale_apps"] == 1
    sec = {s["key"]: s for s in body["sections"]}
    assert sec["new_resumes"]["items"][0]["name"] == "张三"
    assert sec["today_interviews"]["items"][0]["title"] == "一面"
    assert sec["pending_offers"]["items"][0]["status"] == "sent"
    assert sec["stale_apps"]["items"][0]["stage"] == "interview"
    # 每条 section 都带跳转链接
    assert all(s["link"] for s in body["sections"])
    # 记忆中枢
    assert body["memory"]["decisions_total"] == 1
    assert body["memory"]["positive"] == 1


def test_briefing_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    _seed(db, t2)
    r = client.get("/api/briefing/today", headers=_hdr(u1))
    assert r.status_code == 200
    assert r.json()["has_items"] is False


def test_briefing_stale_threshold(client, db):
    """3 天前流转的在途申请不算滞留(阈值 5 天)。"""
    t = _tenant(db)
    u = _user(db, t)
    job = Job(tenant_id=t.id, title="产品", status="open")
    cand = Candidate(tenant_id=t.id, name="李四")
    db.add_all([job, cand])
    db.commit()
    db.refresh(job)
    db.refresh(cand)
    db.add(
        JobApplication(
            tenant_id=t.id,
            candidate_id=cand.id,
            job_id=job.id,
            stage="screening",
            stage_changed_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3),
        )
    )
    db.commit()
    r = client.get("/api/briefing/today", headers=_hdr(u))
    stale = next(s for s in r.json()["sections"] if s["key"] == "stale_apps")
    assert stale["count"] == 0
