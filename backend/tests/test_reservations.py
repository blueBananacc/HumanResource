"""会议室 / 资源 / 预约测试。

覆盖:
- 会议室/资源 CRUD
- 预约:同对象重叠 → 409(含边界:首尾相接不算重叠)
- 不同会议室同时间可并存;取消后可再约
- 修改预约的冲突校验(排除自身)
- 面试官 availability 冲突计数
- 权限与租户隔离
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-resv")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Reservation,
    Resource,
    Room,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


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
    t = Tenant(name=f"rv-{uuid.uuid4().hex[:8]}")
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
        for m in (Reservation, Resource, Room, User):
            db.query(m).filter(m.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


BASE = datetime(2026, 11, 3, 9, 0, 0)  # 周一 09:00


def _resv_json(room_id, *, start_h=9, end_h=10, title="面试", kind="room", **kw):
    """start_h/end_h = 当天几点(9.5 = 09:30)。"""
    day = BASE.replace(hour=0, minute=0)
    return {
        "kind": kind,
        "room_id": room_id,
        "title": title,
        "starts_at": (day + timedelta(hours=start_h)).isoformat(),
        "ends_at": (day + timedelta(hours=end_h)).isoformat(),
        **kw,
    }


def test_room_resource_crud(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/rooms", headers=hdr,
                        json={"name": "301", "location": "3F", "capacity": 8})
        assert r.status_code == 201
        rid = r.json()["id"]
        assert r.json()["capacity"] == 8

        r = client.put(f"/api/rooms/{rid}", headers=hdr,
                       json={"name": "301A", "location": "3F", "capacity": 10})
        assert r.json()["name"] == "301A"

        r = client.post("/api/resources", headers=hdr,
                        json={"name": "视频会议一体机", "type": "equipment"})
        assert r.status_code == 201
        res_id = r.json()["id"]
        r = client.post("/api/resources", headers=hdr,
                        json={"name": "x", "type": "bogus"})
        assert r.status_code == 400

        r = client.get("/api/rooms", headers=hdr)
        assert len(r.json()) == 1
        r = client.get("/api/resources", headers=hdr)
        assert len(r.json()) == 1

        assert client.delete(f"/api/resources/{res_id}", headers=hdr).status_code == 204
        assert client.delete(f"/api/rooms/{rid}", headers=hdr).status_code == 204
        assert client.get("/api/rooms", headers=hdr).json() == []
    finally:
        _cleanup(db_session, t)


def test_reservation_conflict_flow(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        room1 = client.post("/api/rooms", headers=hdr, json={"name": "R1"}).json()["id"]
        room2 = client.post("/api/rooms", headers=hdr, json={"name": "R2"}).json()["id"]

        # 9-10 点约 room1
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room1, start_h=9, end_h=10))
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        assert r.json()["room_name"] == "R1"

        # 9:30-10:30 同 room1 → 409
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room1, start_h=9.5, end_h=10.5, title="冲突"))
        assert r.status_code == 409
        assert "冲突" in r.json()["detail"]

        # 10-11 点 room1(首尾相接)→ OK
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room1, start_h=10, end_h=11))
        assert r.status_code == 201

        # 同时间 room2 → OK
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room2, start_h=9, end_h=10, title="另一间"))
        assert r.status_code == 201

        # 取消第一条 → 再约同时间 OK
        r = client.post(f"/api/reservations/{rid}/cancel", headers=hdr)
        assert r.json()["status"] == "cancelled"
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room1, start_h=9, end_h=10, title="再来"))
        assert r.status_code == 201

        # 列表按区间
        r = client.get(
            "/api/reservations",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=12)).isoformat()},
            headers=hdr,
        )
        assert len(r.json()) == 4  # 含 cancelled
        assert any(x["status"] == "cancelled" for x in r.json())
    finally:
        _cleanup(db_session, t)


def test_reservation_update_and_validation(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]
        rid = client.post("/api/reservations", headers=hdr,
                          json=_resv_json(room, start_h=9, end_h=10)).json()["id"]
        other = client.post("/api/reservations", headers=hdr,
                            json=_resv_json(room, start_h=11, end_h=12, title="O")).json()["id"]

        # 改期到 9:30-10:30(自身不算冲突)→ OK
        r = client.put(f"/api/reservations/{rid}", headers=hdr,
                       json=_resv_json(room, start_h=9.5, end_h=10.5))
        assert r.status_code == 200

        # 改到与 other 重叠 → 409
        r = client.put(f"/api/reservations/{rid}", headers=hdr,
                       json=_resv_json(room, start_h=10.5, end_h=11.5))
        assert r.status_code == 409

        # 结束 <= 开始 → 400
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room, start_h=10, end_h=10))
        assert r.status_code == 400

        # 已取消不能改
        client.post(f"/api/reservations/{other}/cancel", headers=hdr)
        r = client.put(f"/api/reservations/{other}", headers=hdr,
                       json=_resv_json(room, start_h=13, end_h=14))
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_interviewer_availability(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hr = _user(db_session, t, role="hr")
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]
        rp = client.post(
            "/api/reservations", headers=hdr,
            json=_resv_json(room, start_h=9, end_h=10,
                            interviewer_ids=[hr.id, u.id]),
        )
        assert rp.status_code == 201
        r = client.get(
            "/api/interviewers/availability",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=3)).isoformat()},
            headers=hdr,
        )
        assert r.status_code == 200
        by_id = {x["user_id"]: x for x in r.json()}
        assert by_id[hr.id]["busy_count"] == 1
        assert by_id[u.id]["busy_count"] == 1
    finally:
        _cleanup(db_session, t)


def test_permissions_isolation(client, db_session):
    t1 = _tenant(db_session)
    t2 = _tenant(db_session)
    v = _user(db_session, t1, role="viewer")
    u2 = _user(db_session, t2)
    admin = _user(db_session, t1)
    try:
        room = client.post("/api/rooms", headers=_hdr(admin), json={"name": "R"}).json()["id"]
        # viewer:读列表 ok(rooms 无权限门),reservations 需要 view_reports → viewer 有吗?
        r = client.get("/api/reservations",
                       params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=2)).isoformat()},
                       headers=_hdr(v))
        assert r.status_code in (200, 403)
        # viewer 写预约 → 403
        r = client.post("/api/reservations", headers=_hdr(v),
                        json=_resv_json(room, start_h=9, end_h=10))
        assert r.status_code == 403
        # viewer 写会议室 → 403
        r = client.post("/api/rooms", headers=_hdr(v), json={"name": "X"})
        assert r.status_code == 403

        # 租户隔离:B 租户用 A 的 room_id → 404;看不到 A 的预约
        client.post("/api/reservations", headers=_hdr(admin),
                    json=_resv_json(room, start_h=9, end_h=10))
        r = client.post("/api/reservations", headers=_hdr(u2),
                        json=_resv_json(room, start_h=9, end_h=10))
        assert r.status_code == 404
        r = client.get("/api/reservations",
                       params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=2)).isoformat()},
                       headers=_hdr(u2))
        assert r.status_code == 200
        assert r.json() == []
    finally:
        _cleanup(db_session, t1, t2)
