"""申请管道(job_applications)+ 联动钩子测试。

覆盖:
- 建卡 / 列表 / 详情(含 events)/ 换阶段 / 删除 / 409 去重
- 看板 /jobs/{id}/pipeline 分列 + match_score
- offer 钩子:send → offer、accepted → hired、declined → rejected、
  withdraw → 回退 interview
- 发起 remote 面试 → 无卡自动补建并推进 interview
- 权限:viewer 写操作 403
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-apps")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Job,
    JobApplication,
    JobApplicationEvent,
    ResumeJobMatch,
    Resume,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
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
    t = Tenant(name=f"app-{uuid.uuid4().hex[:8]}")
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


def _job_cand(db, t: Tenant, u: User):
    job = Job(tenant_id=t.id, title="测试岗位", level="intermediate",
              description="d", created_by=u.id)
    cand = Candidate(tenant_id=t.id, name="张三")
    db.add_all([job, cand])
    db.commit()
    db.refresh(job)
    db.refresh(cand)
    return job, cand


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        for model in (
            JobApplicationEvent, JobApplication, ResumeJobMatch, Resume,
            Candidate, Job, User,
        ):
            db.query(model).filter(model.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def test_application_crud_and_transitions(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        # 建卡
        r = client.post(
            "/api/applications", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id, "stage": "applied"},
        )
        assert r.status_code == 201, r.text
        app_id = r.json()["id"]
        assert r.json()["stage"] == "applied"
        assert r.json()["candidate_name"] == "张三"

        # 同候选人同岗位 → 409
        r2 = client.post(
            "/api/applications", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id},
        )
        assert r2.status_code == 409

        # 前进
        r = client.post(
            f"/api/applications/{app_id}/transition", headers=hdr,
            json={"to_stage": "screening"},
        )
        assert r.status_code == 200 and r.json()["stage"] == "screening"

        # 回退也允许(人工操作不锁死)
        r = client.post(
            f"/api/applications/{app_id}/transition", headers=hdr,
            json={"to_stage": "applied"},
        )
        assert r.status_code == 200 and r.json()["stage"] == "applied"

        # 非法 stage → 400;原地 → 409
        r = client.post(
            f"/api/applications/{app_id}/transition", headers=hdr,
            json={"to_stage": "bogus"},
        )
        assert r.status_code == 400
        r = client.post(
            f"/api/applications/{app_id}/transition", headers=hdr,
            json={"to_stage": "applied"},
        )
        assert r.status_code == 409

        # rejected 记原因
        r = client.post(
            f"/api/applications/{app_id}/transition", headers=hdr,
            json={"to_stage": "rejected", "rejected_reason": "不匹配"},
        )
        assert r.json()["rejected_reason"] == "不匹配"

        # 详情带事件历史:建卡 + 3 次换阶段
        r = client.get(f"/api/applications/{app_id}", headers=hdr)
        assert r.status_code == 200
        events = r.json()["events"]
        assert [e["to_stage"] for e in events] == [
            "applied", "screening", "applied", "rejected",
        ]
        assert events[0]["from_stage"] is None

        # 删除
        r = client.delete(f"/api/applications/{app_id}", headers=hdr)
        assert r.status_code == 204
        r = client.get(f"/api/applications/{app_id}", headers=hdr)
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_pipeline_kanban_with_match_score(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        # 两张卡不同阶段 + 一条 done 匹配分
        r = client.post(
            "/api/applications", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id, "stage": "screening"},
        )
        assert r.status_code == 201
        resume = Resume(
            tenant_id=t.id, candidate_id=cand.id, file_name="r.pdf",
            file_size=1, file_mime="application/pdf", storage_key="k",
            parse_status="done",
        )
        db_session.add(resume)
        db_session.flush()
        db_session.add(ResumeJobMatch(
            tenant_id=t.id, resume_id=resume.id, job_id=job.id,
            status="done", score=88,
        ))
        db_session.commit()

        r = client.get(f"/api/jobs/{job.id}/pipeline", headers=hdr)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        cols = {c["stage"]: c for c in data["columns"]}
        assert cols["screening"]["count"] == 1
        card = cols["screening"]["cards"][0]
        assert card["candidate_name"] == "张三"
        assert card["match_score"] == 88
        assert cols["applied"]["count"] == 0

        # 列表过滤
        r = client.get(f"/api/applications?job_id={job.id}&stage=screening", headers=hdr)
        assert r.json()["total"] == 1
        r = client.get("/api/applications?q=张", headers=hdr)
        assert r.json()["total"] == 1
    finally:
        _cleanup(db_session, t)


def test_offer_hooks_drive_pipeline(client, db_session):
    """offer send → offer;accepted → hired;declined → rejected。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        # 管道卡先停在 interview
        client.post(
            "/api/applications", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id, "stage": "interview"},
        )
        r = client.post(
            "/api/offers", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id, "position": "工程师"},
        )
        assert r.status_code == 201, r.text
        oid = r.json()["id"]

        r = client.post(f"/api/offers/{oid}/send", headers=hdr)
        assert r.status_code == 200
        app_row = db_session.scalars(
            select(JobApplication).where(JobApplication.candidate_id == cand.id)
        ).first()
        assert app_row.stage == "offer"

        r = client.post(f"/api/offers/{oid}/respond", headers=hdr, json={"accepted": True})
        assert r.status_code == 200
        db_session.expire_all()
        app_row = db_session.scalars(
            select(JobApplication).where(JobApplication.candidate_id == cand.id)
        ).first()
        assert app_row.stage == "hired"
    finally:
        _cleanup(db_session, t)


def test_offer_decline_and_withdraw_hooks(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        # declined → rejected(允许从 offer 回退到旁路)
        r = client.post(
            "/api/offers", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id},
        )
        oid = r.json()["id"]
        client.post(f"/api/offers/{oid}/send", headers=hdr)
        client.post(f"/api/offers/{oid}/respond", headers=hdr, json={"accepted": False})
        db_session.expire_all()
        app_row = db_session.scalars(
            select(JobApplication).where(JobApplication.candidate_id == cand.id)
        ).first()
        assert app_row.stage == "rejected"

        # withdrawn → 回退 interview
        r = client.post(
            "/api/offers", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id},
        )
        oid2 = r.json()["id"]
        client.post(f"/api/offers/{oid2}/send", headers=hdr)
        client.post(f"/api/offers/{oid2}/withdraw", headers=hdr)
        db_session.expire_all()
        app_row = db_session.scalars(
            select(JobApplication).where(JobApplication.candidate_id == cand.id)
        ).first()
        assert app_row.stage == "interview"
    finally:
        _cleanup(db_session, t)


def test_interview_start_auto_advances(client, db_session):
    """无管道卡时发起 remote 面试 → 自动补建并停在 interview。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        r = client.post(
            "/api/interviews/start", headers=hdr,
            json={
                "candidate_id": cand.id, "job_id": job.id,
                "question_count": 3, "kinds": ["tech"],
                "expires_in_hours": 48, "delivery": "link",
            },
        )
        assert r.status_code == 201, r.text
        db_session.expire_all()
        app_row = db_session.scalars(
            select(JobApplication).where(JobApplication.candidate_id == cand.id)
        ).first()
        assert app_row is not None
        assert app_row.stage == "interview"
        assert app_row.source == "system"
    finally:
        _cleanup(db_session, t)


def test_viewer_cannot_write(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t, role="viewer")
    job, cand = _job_cand(db_session, t, u)
    hdr = _hdr(u)
    try:
        r = client.post(
            "/api/applications", headers=hdr,
            json={"candidate_id": cand.id, "job_id": job.id},
        )
        assert r.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_tenant_isolation(client, db_session):
    """租户 B 看不到租户 A 的卡。"""
    t1 = _tenant(db_session)
    t2 = _tenant(db_session)
    u1 = _user(db_session, t1)
    u2 = _user(db_session, t2)
    job, cand = _job_cand(db_session, t1, u1)
    try:
        r = client.post(
            "/api/applications", headers=_hdr(u1),
            json={"candidate_id": cand.id, "job_id": job.id},
        )
        app_id = r.json()["id"]
        r = client.get(f"/api/applications/{app_id}", headers=_hdr(u2))
        assert r.status_code == 404
        r = client.get("/api/applications", headers=_hdr(u2))
        assert r.json()["total"] == 0
    finally:
        _cleanup(db_session, t1, t2)
