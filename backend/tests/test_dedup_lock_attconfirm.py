"""两级简历查重 + 候选人锁定/释放 + 考勤在线确认路测。

查重:通道分级(exact/suspect)/ 入库钩子挂标 / 队列两级 / dismiss / 租户隔离。
锁定:lock/unlock / 过期自动释放 / 非持锁人变更被 409 / 管道流转被挡。
考勤确认:员工 confirm/dispute / HR 列表排序 / resolve 状态机 / 通知投递。
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.domain.models import (  # noqa: E402
    Attendance,
    AttendanceConfirmation,
    Base,
    Candidate,
    Employee,
    Job,
    JobApplication,
    Notification,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.candidates import _hash as c_hash  # noqa: E402


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
    t = Tenant(name=f"dd-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t, role="admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@x.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u):
    return {"Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"}


def _cand(db, t, name="张三", email=None, phone=None) -> Candidate:
    c = Candidate(
        tenant_id=t.id, name=name,
        email_hash=c_hash(email, t.id) if email else None,
        phone_hash=c_hash(phone, t.id) if phone else None,
        display_email=email, display_phone=phone,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


# ---------------- 两级查重 ----------------


def test_dedup_tiers_and_queue(db, client):
    """name+email = 精准级;仅 email / 仅 name = 疑似级。"""
    t = _tenant(db)
    u = _user(db, t)
    a = _cand(db, t, "重复甲", email="a@x.com", phone="13800000001")
    _cand(db, t, "重复甲", email="a@x.com", phone="999")  # name+email → exact
    _cand(db, t, "疑似乙", email="a@x.com")  # email-only → suspect
    _cand(db, t, "重复甲", phone="777")  # name-only → suspect
    _cand(db, t, "干净丙", email="c@x.com")

    r = client.get("/api/talents/dedup-queue", headers=_hdr(u))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["scanned"] >= 5
    assert len(data["exact"]) == 1
    assert {tuple(sorted(p["channels"])) for p in data["exact"]} == {
        ("email", "name")
    }
    assert len(data["suspect"]) == 2
    # 疑似对里的 candidate 挂标
    flagged = {p["candidate"]["id"] for p in data["suspect"]} | {
        p["peer"]["id"] for p in data["suspect"]
    }
    db.refresh(a)
    flagged_cands = db.scalars(
        select(Candidate).where(
            Candidate.tenant_id == t.id, Candidate.dedup_flag == "suspect"
        )
    ).all()
    assert {c.id for c in flagged_cands} & flagged


def test_dedup_ingest_hook_and_dismiss(db, client):
    """upsert_candidate 新建 → name 撞车挂疑标;dismiss 摘标。"""
    t = _tenant(db)
    u = _user(db, t)
    _cand(db, t, "同名丁", email="d1@x.com")

    from app.services.candidates import upsert_candidate

    c2 = upsert_candidate(
        db, tenant_id=t.id, name="同名丁", email="d2@x.com", phone=None
    )
    db.commit()
    db.refresh(c2)
    assert c2.dedup_flag == "suspect"
    assert c2.dedup_peer_id is not None

    # dismiss → 摘标 + 审计
    r = client.post(f"/api/talents/{c2.id}/dedup-dismiss", headers=_hdr(u))
    assert r.status_code == 200, r.text
    db.refresh(c2)
    assert c2.dedup_flag is None

    # 再扫不重复挂标(对端仍在,scan 会重新挂 — 这是"待审队列"语义)
    r = client.post("/api/talents/dedup-scan", headers=_hdr(u))
    assert r.status_code == 200
    assert r.json()["suspect"] >= 1


def test_dedup_queue_scan_tenant_isolation(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u1, u2 = _user(db, t1), _user(db, t2)
    _cand(db, t1, "跨租户", email="x@x.com")
    _cand(db, t2, "跨租户", email="x@x.com")  # 同租户外同名同邮不算重复

    r = client.get("/api/talents/dedup-queue", headers=_hdr(u1))
    pairs = r.json()["exact"] + r.json()["suspect"]
    assert all(p["candidate"]["name"] != "跨租户" or "跨" not in p["peer"]["name"] for p in pairs) or pairs == []
    # t2 用户看不到 t1 的候选人
    t1_cand = db.scalars(select(Candidate).where(Candidate.tenant_id == t1.id)).first()
    r = client.post(f"/api/talents/{t1_cand.id}/dedup-dismiss", headers=_hdr(u2))
    assert r.status_code == 404


# ---------------- 候选人锁定 ----------------


def test_lock_unlock_flow(db, client):
    t = _tenant(db)
    u = _user(db, t)
    u2 = _user(db, t)
    c = _cand(db, t, "锁定戊", email="lock@x.com")

    # u 锁定 7 天
    r = client.post(f"/api/talents/{c.id}/lock", json={"days": 7}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    assert r.json()["locked"] is True
    until = r.json()["locked_until"]
    assert until is not None

    # u2 再锁 → 409;改标签 → 409;建申请 → 409
    r = client.post(f"/api/talents/{c.id}/lock", json={"days": 3}, headers=_hdr(u2))
    assert r.status_code == 409
    r = client.put(f"/api/talents/{c.id}/tags", json={"tags": ["x"]}, headers=_hdr(u2))
    assert r.status_code == 409, r.text
    job = Job(tenant_id=t.id, title="岗", status="open")
    db.add(job)
    db.commit()
    r = client.post("/api/applications", json={
        "job_id": job.id, "candidate_id": c.id,
    }, headers=_hdr(u2))
    assert r.status_code == 409, (r.status_code, r.text)

    # 持锁人本人可操作
    r = client.put(f"/api/talents/{c.id}/tags", json={"tags": ["ok"]}, headers=_hdr(u))
    assert r.status_code == 200

    # u2 解锁 → 403;u 解锁 → 200
    r = client.post(f"/api/talents/{c.id}/unlock", headers=_hdr(u2))
    assert r.status_code == 403
    r = client.post(f"/api/talents/{c.id}/unlock", headers=_hdr(u))
    assert r.status_code == 200
    r = client.put(f"/api/talents/{c.id}/tags", json={"tags": ["free"]}, headers=_hdr(u2))
    assert r.status_code == 200


def test_lock_expired_auto_release_and_transition_guard(db, client):
    t = _tenant(db)
    u = _user(db, t)
    u2 = _user(db, t)
    c = _cand(db, t, "过期己", email="exp@x.com")
    job = Job(tenant_id=t.id, title="岗2", status="open")
    db.add(job)
    db.commit()
    app_ = JobApplication(
        tenant_id=t.id, candidate_id=c.id, job_id=job.id, stage="applied",
    )
    db.add(app_)
    db.commit()

    # u 锁 → u2 流转被 409
    r = client.post(f"/api/talents/{c.id}/lock", json={"days": 7}, headers=_hdr(u))
    assert r.status_code == 200
    r = client.post(f"/api/applications/{app_.id}/transition",
                    json={"to_stage": "screening"}, headers=_hdr(u2))
    assert r.status_code == 409, (r.status_code, r.text)

    # 手动把 locked_until 拨到过去 → 自动释放,u2 可流转
    c.locked_until = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    db.commit()
    r = client.post(f"/api/applications/{app_.id}/transition",
                    json={"to_stage": "screening"}, headers=_hdr(u2))
    assert r.status_code == 200, r.text

    # 过期锁可被他人接管(重新 lock)
    r = client.post(f"/api/talents/{c.id}/lock", json={"days": 1}, headers=_hdr(u2))
    assert r.status_code == 200


# ---------------- 考勤在线确认 ----------------


def _employee(db, t, u, name="员工A") -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name, user_id=u.id, status="active",
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def test_attendance_confirm_dispute_resolve(db, client):
    t = _tenant(db)
    hr = _user(db, t, "admin")
    hr2 = _user(db, t, "hr")
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu, "考勤庚")

    # 先造两天考勤
    for d in ("2026-10-01", "2026-10-02"):
        db.add(Attendance(
            tenant_id=t.id, employee_id=e.id,
            day=datetime.strptime(d, "%Y-%m-%d").date(), status="present",
        ))
    db.commit()

    # 初始:未确认
    r = client.get("/api/me/attendance-confirmation?month=2026-10", headers=_hdr(eu))
    assert r.status_code == 200
    assert r.json()["status"] is None

    # 确认
    r = client.post("/api/me/attendance/confirm", json={"month": "2026-10"}, headers=_hdr(eu))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "confirmed"

    # HR 列表:confirmed
    r = client.get("/api/attendance/confirmations?month=2026-10", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    row = next(x for x in r.json() if x["employee_id"] == e.id)
    assert row["status"] == "confirmed"

    # 员工反悔 → dispute → HR 收到通知,列表 disputed 置顶
    r = client.post("/api/me/attendance/dispute",
                    json={"month": "2026-10", "note": "10-01 缺卡是我外出"},
                    headers=_hdr(eu))
    assert r.status_code == 200
    assert r.json()["status"] == "disputed"
    notif = db.scalars(
        select(Notification).where(
            Notification.tenant_id == t.id,
            Notification.kind == "attendance.dispute",
        )
    ).all()
    assert len(notif) == 1
    assert notif[0].user_id == hr2.id  # hr 角色收通知(admin 不收)

    r = client.get("/api/attendance/confirmations?month=2026-10", headers=_hdr(hr))
    rows = r.json()
    assert rows[0]["status"] == "disputed" and rows[0]["note"].startswith("10-01")

    # HR resolve → resolved;重复 resolve → 409
    r = client.post(f"/api/attendance/confirmations/{e.id}/resolve",
                    json={"month": "2026-10", "note": "已补卡"}, headers=_hdr(hr))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "resolved" and r.json()["resolved_note"] == "已补卡"
    r = client.post(f"/api/attendance/confirmations/{e.id}/resolve",
                    json={"month": "2026-10"}, headers=_hdr(hr))
    assert r.status_code == 409

    # resolve 后员工再确认 → confirmed(resolved_note 清空)
    r = client.post("/api/me/attendance/confirm", json={"month": "2026-10"}, headers=_hdr(eu))
    assert r.json()["status"] == "confirmed"
    assert r.json()["resolved_note"] is None


def test_attendance_confirm_tenant_scope_and_validation(db, client):
    t = _tenant(db)
    eu = _user(db, t, "employee")
    _employee(db, t, eu, "校验辛")
    hr = _user(db, t, "manager")  # manager 无 manage_attendance → 403

    # month 格式非法 → 400
    r = client.post("/api/me/attendance/confirm", json={"month": "2026/10"}, headers=_hdr(eu))
    assert r.status_code == 400
    # 无员工档案的用户 → 4xx(require_linked_employee)
    orphan = _user(db, t, "employee")
    r = client.post("/api/me/attendance/confirm", json={"month": "2026-10"}, headers=_hdr(orphan))
    assert r.status_code in (400, 403, 404)
    # manager 不可读 HR 确认列表
    r = client.get("/api/attendance/confirmations?month=2026-10", headers=_hdr(hr))
    assert r.status_code == 403
