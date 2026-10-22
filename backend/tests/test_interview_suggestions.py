"""面试页"AI 推荐发起 + 面试删除"测试。

走 ``CELERY_TASK_ALWAYS_EAGER=true``,与 :mod:`tests.test_interview_invite`
同一套 fixture 风格,直接打真 PG。

覆盖:
1. GET /api/interviews/suggested — 仅 status=done 且 score>90 的匹配出现;
   90 分 / pending / 已忽略 / 已有面试记录 / 非 open 岗位 都排除
2. POST /api/matches/{id}/dismiss-suggestion — 标记后建议消失
3. DELETE /api/interviews/{id} — abandoned / 已过期且无评分 可删;
   进行中 / 有评分 一律 409
4. deletable 字段随列表返回
5. 跨租户隔离
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-suggest")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-suggest"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Interview,
    InterviewTurn,
    Job,
    Resume,
    ResumeJobMatch,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import hash_password  # noqa: E402
from app.main import create_app  # noqa: E402

# ---------------------------- fixtures ----------------------------


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
def tenant(db_session):
    t = Tenant(name=f"suggest-test-{uuid.uuid4().hex[:8]}")
    db_session.add(t)
    db_session.commit()
    db_session.refresh(t)
    yield t
    db_session.query(InterviewTurn).filter(
        InterviewTurn.interview_id.in_(
            db_session.query(Interview.id).filter(Interview.tenant_id == t.id)
        )
    ).delete(synchronize_session=False)
    db_session.query(Interview).filter(Interview.tenant_id == t.id).delete()
    db_session.query(ResumeJobMatch).filter(
        ResumeJobMatch.tenant_id == t.id
    ).delete(synchronize_session=False)
    db_session.query(Resume).filter(Resume.tenant_id == t.id).delete()
    db_session.query(Job).filter(Job.tenant_id == t.id).delete()
    db_session.query(Candidate).filter(Candidate.tenant_id == t.id).delete()
    db_session.query(User).filter(User.tenant_id == t.id).delete()
    db_session.delete(t)
    db_session.commit()


@pytest.fixture
def admin_and_token(db_session, tenant, client):
    email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
    u = User(
        tenant_id=tenant.id,
        email=email,
        password_hash=hash_password("test1234"),
        role="admin",
    )
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    resp = client.post(
        "/api/auth/login", json={"email": email, "password": "test1234"}
    )
    assert resp.status_code == 200, resp.text
    return u, resp.json()["access_token"]


# ---------------------------- helpers -----------------------------


def _make_job(db, tenant, *, status="open"):
    j = Job(
        tenant_id=tenant.id,
        title=f"岗位-{uuid.uuid4().hex[:6]}",
        level="intermediate",
        skills=["Python"],
        description="desc",
        status=status,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _make_candidate_with_resume(db, tenant):
    cand = Candidate(
        tenant_id=tenant.id,
        name=f"候选人-{uuid.uuid4().hex[:6]}",
        display_email="cand@example.com",
    )
    db.add(cand)
    db.commit()
    db.refresh(cand)
    r = Resume(
        tenant_id=tenant.id,
        candidate_id=cand.id,
        file_name="r.pdf",
        file_size=1024,
        file_mime="application/pdf",
        storage_key=f"test-{uuid.uuid4().hex}",
        source="upload",
        parse_status="done",
        parsed_text="5 年 Python 经验",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return cand, r


def _make_match(db, tenant, resume, job, *, score=95, status="done"):
    m = ResumeJobMatch(
        tenant_id=tenant.id,
        resume_id=resume.id,
        job_id=job.id,
        status=status,
        score=score,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _make_interview(db, tenant, job, cand, **overrides):
    iv = Interview(
        tenant_id=tenant.id,
        job_id=job.id,
        candidate_id=cand.id,
        mode="remote",
        status=overrides.get("status", "in_progress"),
        level="intermediate",
        question_count=3,
        kinds=["tech"],
        delivery="link",
        expires_at=overrides.get("expires_at"),
        invite_token_hash=overrides.get("invite_token_hash"),
        summary=overrides.get("summary"),
    )
    db.add(iv)
    db.commit()
    db.refresh(iv)
    return iv


def _suggested(client, token):
    r = client.get(
        "/api/interviews/suggested",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------- suggested ---------------------------


def test_suggested_lists_high_score_match(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, resume = _make_candidate_with_resume(db_session, tenant)
    m = _make_match(db_session, tenant, resume, job, score=92)

    items = _suggested(client, token)
    assert len(items) == 1
    s = items[0]
    assert s["match_id"] == m.id
    assert s["score"] == 92
    assert s["candidate_id"] == cand.id
    assert s["job_id"] == job.id
    assert s["job_title"] == job.title


def test_suggested_threshold_is_strictly_above_90(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    _, r1 = _make_candidate_with_resume(db_session, tenant)
    _, r2 = _make_candidate_with_resume(db_session, tenant)
    _make_match(db_session, tenant, r1, job, score=90)  # 恰好 90 — 不算
    _make_match(db_session, tenant, r2, job, score=91)  # 91 — 出现

    items = _suggested(client, token)
    scores = {s["score"] for s in items}
    assert scores == {91}


def test_suggested_excludes_pending_and_failed(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    _, r1 = _make_candidate_with_resume(db_session, tenant)
    _, r2 = _make_candidate_with_resume(db_session, tenant)
    _make_match(db_session, tenant, r1, job, score=99, status="pending")
    _make_match(db_session, tenant, r2, job, score=99, status="failed")
    assert _suggested(client, token) == []


def test_suggested_excludes_when_interview_exists(
    client, db_session, tenant, admin_and_token
):
    """同一 (job, candidate) 已有面试记录(含 abandoned)不再推荐。"""
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, resume = _make_candidate_with_resume(db_session, tenant)
    _make_match(db_session, tenant, resume, job, score=95)
    _make_interview(db_session, tenant, job, cand, status="abandoned")

    assert _suggested(client, token) == []


def test_suggested_excludes_non_open_job(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant, status="closed")
    _, resume = _make_candidate_with_resume(db_session, tenant)
    _make_match(db_session, tenant, resume, job, score=95)
    assert _suggested(client, token) == []


def test_dismiss_suggestion_hides_match(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    _, resume = _make_candidate_with_resume(db_session, tenant)
    m = _make_match(db_session, tenant, resume, job, score=95)
    assert len(_suggested(client, token)) == 1

    r = client.post(
        f"/api/matches/{m.id}/dismiss-suggestion",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 204, r.text
    assert _suggested(client, token) == []

    db_session.refresh(m)
    assert m.suggestion_dismissed_at is not None


def test_suggested_cross_tenant_isolation(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    other = Tenant(name=f"other-{uuid.uuid4().hex[:6]}")
    db_session.add(other)
    db_session.commit()
    db_session.refresh(other)
    try:
        job = _make_job(db_session, other)
        _, resume = _make_candidate_with_resume(db_session, other)
        _make_match(db_session, other, resume, job, score=99)
        assert _suggested(client, token) == []
    finally:
        db_session.query(ResumeJobMatch).filter(
            ResumeJobMatch.tenant_id == other.id
        ).delete(synchronize_session=False)
        db_session.query(Resume).filter(Resume.tenant_id == other.id).delete()
        db_session.query(Job).filter(Job.tenant_id == other.id).delete()
        db_session.query(Candidate).filter(
            Candidate.tenant_id == other.id
        ).delete()
        db_session.delete(other)
        db_session.commit()


# ---------------------------- delete ------------------------------


def test_delete_abandoned_interview_ok(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv = _make_interview(db_session, tenant, job, cand, status="abandoned")

    r = client.delete(
        f"/api/interviews/{iv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 204, r.text
    assert (
        db_session.query(Interview).filter(Interview.id == iv.id).first()
        is None
    )


def test_delete_expired_unscored_ok_and_turns_cascaded(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv = _make_interview(
        db_session,
        tenant,
        job,
        cand,
        expires_at=datetime.now() - timedelta(hours=1),
    )
    turn = InterviewTurn(
        interview_id=iv.id,
        idx=1,
        level="intermediate",
        question="介绍一下自己",
        answer="我答了但没评分",
        answered_at=datetime.now(),
        score_status="pending",
    )
    db_session.add(turn)
    db_session.commit()
    tid = turn.id

    r = client.delete(
        f"/api/interviews/{iv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 204, r.text
    assert (
        db_session.query(Interview).filter(Interview.id == iv.id).first()
        is None
    )
    assert (
        db_session.query(InterviewTurn)
        .filter(InterviewTurn.id == tid)
        .first()
        is None
    )


def test_delete_active_interview_rejected(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv = _make_interview(
        db_session,
        tenant,
        job,
        cand,
        expires_at=datetime.now() + timedelta(hours=48),
        invite_token_hash="x" * 64,
    )

    r = client.delete(
        f"/api/interviews/{iv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409, r.text
    assert db_session.get(Interview, iv.id) is not None


def test_delete_expired_but_scored_rejected(
    client, db_session, tenant, admin_and_token
):
    """过期但有评分的面试不能删 — 评分记录是决策依据。"""
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv = _make_interview(
        db_session,
        tenant,
        job,
        cand,
        expires_at=datetime.now() - timedelta(hours=1),
    )
    turn = InterviewTurn(
        interview_id=iv.id,
        idx=1,
        level="intermediate",
        question="介绍一下自己",
        answer="答案",
        answered_at=datetime.now(),
        scores={"accuracy": 80},
        score_status="done",
    )
    db_session.add(turn)
    db_session.commit()

    r = client.delete(
        f"/api/interviews/{iv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409, r.text
    assert db_session.get(Interview, iv.id) is not None


def test_delete_done_with_summary_rejected(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv = _make_interview(
        db_session,
        tenant,
        job,
        cand,
        status="done",
        summary={"recommendation": "推荐进入下一轮"},
    )

    r = client.delete(
        f"/api/interviews/{iv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409, r.text
    assert db_session.get(Interview, iv.id) is not None


def test_delete_cross_tenant_not_found(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    other = Tenant(name=f"other-{uuid.uuid4().hex[:6]}")
    db_session.add(other)
    db_session.commit()
    db_session.refresh(other)
    try:
        job = _make_job(db_session, other)
        cand, _ = _make_candidate_with_resume(db_session, other)
        iv = _make_interview(db_session, other, job, cand, status="abandoned")

        r = client.delete(
            f"/api/interviews/{iv.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404
        assert db_session.get(Interview, iv.id) is not None
    finally:
        db_session.query(Interview).filter(
            Interview.tenant_id == other.id
        ).delete()
        db_session.query(Resume).filter(Resume.tenant_id == other.id).delete()
        db_session.query(Job).filter(Job.tenant_id == other.id).delete()
        db_session.query(Candidate).filter(
            Candidate.tenant_id == other.id
        ).delete()
        db_session.delete(other)
        db_session.commit()


def test_list_interviews_returns_deletable_flag(
    client, db_session, tenant, admin_and_token
):
    _admin, token = admin_and_token
    job = _make_job(db_session, tenant)
    cand, _ = _make_candidate_with_resume(db_session, tenant)
    iv_dead = _make_interview(
        db_session, tenant, job, cand, status="abandoned"
    )
    cand2, _ = _make_candidate_with_resume(db_session, tenant)
    iv_live = _make_interview(
        db_session,
        tenant,
        job,
        cand2,
        expires_at=datetime.now() + timedelta(hours=48),
        invite_token_hash="y" * 64,
    )

    r = client.get(
        "/api/interviews/", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, r.text
    flags = {i["id"]: i["deletable"] for i in r.json()["items"]}
    assert flags[iv_dead.id] is True
    assert flags[iv_live.id] is False
