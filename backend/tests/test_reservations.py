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
    OrgUnit,
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

        # 非法 kind 过滤 → 400
        r = client.get(
            "/api/reservations",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=12)).isoformat(),
                    "kind": "bogus"},
            headers=hdr,
        )
        assert r.status_code == 400
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
        # viewer:读列表 OK(viewer 有 view_reports)
        r = client.get("/api/reservations",
                       params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=2)).isoformat()},
                       headers=_hdr(v))
        assert r.status_code == 200
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


def test_org_scoped_user_list_and_scope_enforcement(client, db_session):
    """绑 org 的非 admin 用户:list_rooms/list_reservations 不能 500,且按数据范围过滤。"""
    t = _tenant(db_session)
    org_a = OrgUnit(tenant_id=t.id, name="A")
    org_b = OrgUnit(tenant_id=t.id, name="B")
    db_session.add_all([org_a, org_b])
    db_session.commit()
    db_session.refresh(org_a)
    db_session.refresh(org_b)
    admin = _user(db_session, t)
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    db_session.commit()
    hadm, hhr = _hdr(admin), _hdr(hr)
    try:
        room_b = client.post("/api/rooms", headers=hadm,
                             json={"name": "B会议室", "org_unit_id": org_b.id}).json()["id"]
        room_pub = client.post("/api/rooms", headers=hadm, json={"name": "公共"}).json()["id"]

        # org-A 用户:list 不 500,只看到 org NULL 的公共会议室
        r = client.get("/api/rooms", headers=hhr)
        assert r.status_code == 200
        assert [x["id"] for x in r.json()] == [room_pub]

        # org-A 用户:不能约 org-B 的会议室
        r = client.post("/api/reservations", headers=hhr,
                        json=_resv_json(room_b, start_h=9, end_h=10))
        assert r.status_code == 404

        # admin 在 org-B 会议室约一条 + 一条纯面试官预约(不占对象)
        client.post("/api/reservations", headers=hadm,
                    json=_resv_json(room_b, start_h=9, end_h=10))
        iv = _resv_json(None, start_h=9, end_h=10, kind="interview", title="纯面试官")
        r = client.post("/api/reservations", headers=hhr, json=iv)
        assert r.status_code == 201  # 无对象 → 不与 B 会议室的预约误报冲突

        # org-A 用户看预约列表:不含 org-B 会议室的那条,含自己的 interview 条
        r = client.get("/api/reservations", headers=hhr,
                       params={"start": BASE.isoformat(),
                               "end": (BASE + timedelta(hours=3)).isoformat()})
        assert r.status_code == 200
        titles = [x["title"] for x in r.json()]
        assert titles == ["纯面试官"]

        # org-A 用户不能改/取消 org-B 会议室上的预约
        resv_b = client.get("/api/reservations", headers=hadm,
                            params={"start": BASE.isoformat(),
                                    "end": (BASE + timedelta(hours=3)).isoformat(),
                                    "room_id": room_b}).json()[0]
        r = client.put(f"/api/reservations/{resv_b['id']}", headers=hhr,
                       json=_resv_json(room_pub, start_h=11, end_h=12))
        assert r.status_code == 404
        r = client.post(f"/api/reservations/{resv_b['id']}/cancel", headers=hhr)
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_room_resource_or_conflict(client, db_session):
    """同时占会议室+资源的预约:同会议室不同资源依然冲突(OR 语义)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]
        res_b = client.post("/api/resources", headers=hdr,
                            json={"name": "B", "type": "equipment"}).json()["id"]
        res_c = client.post("/api/resources", headers=hdr,
                            json={"name": "C", "type": "equipment"}).json()["id"]
        body1 = _resv_json(room, start_h=9, end_h=10, resource_id=res_b)
        assert client.post("/api/reservations", headers=hdr, json=body1).status_code == 201
        # 同会议室、不同资源、时间重叠 → 仍 409
        body2 = _resv_json(room, start_h=9.5, end_h=10.5, resource_id=res_c)
        r = client.post("/api/reservations", headers=hdr, json=body2)
        assert r.status_code == 409
    finally:
        _cleanup(db_session, t)


def test_reservation_input_validation(client, db_session):
    """面试官必须本租户;update 复用完整校验(非法 kind / 停用对象)。"""
    t = _tenant(db_session)
    t2 = _tenant(db_session)
    u = _user(db_session, t)
    u2 = _user(db_session, t2)
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]

        # 跨租户面试官 → 400
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room, start_h=9, end_h=10,
                                        interviewer_ids=[u2.id]))
        assert r.status_code == 400

        rid = client.post("/api/reservations", headers=hdr,
                          json=_resv_json(room, start_h=9, end_h=10)).json()["id"]

        # 非法 kind → 400
        r = client.put(f"/api/reservations/{rid}", headers=hdr,
                       json=_resv_json(room, start_h=11, end_h=12, kind="bogus"))
        assert r.status_code == 400

        # 改到已停用的会议室 → 400
        client.put(f"/api/rooms/{room}", headers=hdr,
                   json={"name": "R", "is_active": False})
        r = client.put(f"/api/reservations/{rid}", headers=hdr,
                       json=_resv_json(room, start_h=11, end_h=12))
        assert r.status_code == 400

        # 无对象的非 interview 预约 → 400
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(None, start_h=13, end_h=14))
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t, t2)


def test_inactive_interviewer_rejected(client, db_session):
    """禁用用户不能当面试官 — 之前只校验租户归属。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    disabled = _user(db_session, t, role="hr")
    disabled.status = "disabled"
    db_session.commit()
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]
        r = client.post("/api/reservations", headers=hdr,
                        json=_resv_json(room, start_h=9, end_h=10,
                                        interviewer_ids=[disabled.id]))
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_out_of_scope_room_always_404(client, db_session):
    """范围外会议室不论启用/停用都 404 — 不通过 400 外泄停用状态。"""
    t = _tenant(db_session)
    org_a = OrgUnit(tenant_id=t.id, name="A")
    org_b = OrgUnit(tenant_id=t.id, name="B")
    db_session.add_all([org_a, org_b])
    db_session.commit()
    db_session.refresh(org_a)
    db_session.refresh(org_b)
    admin = _user(db_session, t)
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    db_session.commit()
    try:
        room_b = client.post("/api/rooms", headers=_hdr(admin),
                             json={"name": "B", "org_unit_id": org_b.id}).json()["id"]
        # 停用状态下范围外用户依然只配看到 404(PUT 整组替换,org 要带上)
        client.put(f"/api/rooms/{room_b}", headers=_hdr(admin),
                   json={"name": "B", "is_active": False, "org_unit_id": org_b.id})
        r = client.post("/api/reservations", headers=_hdr(hr),
                        json=_resv_json(room_b, start_h=9, end_h=10))
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_delete_room_in_use_guard(client, db_session):
    """有预约记录(含已取消)的会议室/资源不能硬删 — 会连坐抹掉台账。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        room = client.post("/api/rooms", headers=hdr, json={"name": "R"}).json()["id"]
        res = client.post("/api/resources", headers=hdr,
                          json={"name": "RES", "type": "equipment"}).json()["id"]
        rid = client.post(
            "/api/reservations", headers=hdr,
            json=_resv_json(room, start_h=9, end_h=10, resource_id=res),
        ).json()["id"]
        client.post(f"/api/reservations/{rid}/cancel", headers=hdr)

        r = client.delete(f"/api/rooms/{room}", headers=hdr)
        assert r.status_code == 409
        assert "停用" in r.json()["detail"]
        r = client.delete(f"/api/resources/{res}", headers=hdr)
        assert r.status_code == 409

        # 没有预约的空对象可以删
        empty_room = client.post("/api/rooms", headers=hdr, json={"name": "E"}).json()["id"]
        assert client.delete(f"/api/rooms/{empty_room}", headers=hdr).status_code == 204
    finally:
        _cleanup(db_session, t)


def test_availability_title_masking(client, db_session):
    """org-scoped 用户看纯面试官预约(不占对象):数量可见,标题打码防泄露。"""
    t = _tenant(db_session)
    org_a = OrgUnit(tenant_id=t.id, name="A")
    db_session.add(org_a)
    db_session.commit()
    db_session.refresh(org_a)
    admin = _user(db_session, t)
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    hr2 = _user(db_session, t, role="hr")
    db_session.commit()
    try:
        # admin 建一条不占对象的预约,面试官是 hr2(hr 不在参与者里,也不是创建者)
        client.post(
            "/api/reservations", headers=_hdr(admin),
            json=_resv_json(None, start_h=9, end_h=10, kind="interview",
                            title="张三终面", interviewer_ids=[hr2.id]),
        )
        # org-scoped 的 hr:能看到 busy_count,但标题打码
        r = client.get(
            "/api/interviewers/availability",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=3)).isoformat()},
            headers=_hdr(hr),
        )
        assert r.status_code == 200
        by_id = {x["user_id"]: x for x in r.json()}
        assert by_id[hr2.id]["busy_count"] == 1
        assert by_id[hr2.id]["busy_titles"] == ["(受限预约)"]

        # admin 不受限:看到真标题
        r = client.get(
            "/api/interviewers/availability",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=3)).isoformat()},
            headers=_hdr(admin),
        )
        by_id = {x["user_id"]: x for x in r.json()}
        assert by_id[hr2.id]["busy_titles"] == ["张三终面"]

        # 参与面试官本人可以看到自己的标题
        r = client.get(
            "/api/interviewers/availability",
            params={"start": BASE.isoformat(), "end": (BASE + timedelta(hours=3)).isoformat()},
            headers=_hdr(hr2),
        )
        by_id = {x["user_id"]: x for x in r.json()}
        assert by_id[hr2.id]["busy_titles"] == ["张三终面"]
    finally:
        _cleanup(db_session, t)
