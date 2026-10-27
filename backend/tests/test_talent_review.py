"""人才盘点九宫格 + 人才池路测。

评分聚合为确定性规则;校准为人工覆盖;覆盖 CRUD/流转/隔离。
"""
from __future__ import annotations

import os
import uuid
from datetime import date

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Appraisal,
    AppraisalCycle,
    Base,
    Employee,
    EmployeeSkill,
    Goal,
    Skill,
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
    t = Tenant(name=f"tr-{uuid.uuid4().hex[:8]}")
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


def _emp(db, t, **kw) -> Employee:
    e = Employee(
        tenant_id=t.id,
        employee_no=kw.pop("no", f"E{uuid.uuid4().hex[:6]}"),
        name=kw.pop("name", "张三"),
        designation=kw.pop("designation", "工程师"),
        status="active",
        **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _seed_strong(db, t, e):
    """高绩效高潜力:评分5 + 技能L5 + 目标100。"""
    cyc = AppraisalCycle(tenant_id=t.id, name="C1", status="open",
                         period_start=date(2026, 1, 1), period_end=date(2026, 12, 31))
    db.add(cyc)
    db.flush()
    db.add(Appraisal(
        tenant_id=t.id, cycle_id=cyc.id, employee_id=e.id,
        manager_rating=5, status="reviewed",
    ))
    sk = Skill(tenant_id=t.id, name="Python", category="tech")
    db.add(sk)
    db.flush()
    db.add(EmployeeSkill(
        tenant_id=t.id, employee_id=e.id, skill_id=sk.id,
        level=5, source="manual", status="confirmed",
    ))
    db.add(Goal(tenant_id=t.id, employee_id=e.id, title="G", progress=100))
    db.commit()


def _seed_weak(db, t, e):
    cyc = AppraisalCycle(tenant_id=t.id, name="C1", status="open",
                         period_start=date(2026, 1, 1), period_end=date(2026, 12, 31))
    db.add(cyc)
    db.flush()
    db.add(Appraisal(
        tenant_id=t.id, cycle_id=cyc.id, employee_id=e.id,
        manager_rating=1, status="reviewed",
    ))
    db.add(Goal(tenant_id=t.id, employee_id=e.id, title="G", progress=10))
    db.commit()


def test_nine_box_auto_placement(client, db):
    t = _tenant(db)
    u = _user(db, t)
    strong = _emp(db, t, name="强者")
    weak = _emp(db, t, name="弱者")
    empty = _emp(db, t, name="白板")
    _seed_strong(db, t, strong)
    _seed_weak(db, t, weak)

    r = client.get("/api/talent-review/nine-box", headers=_hdr(u))
    assert r.status_code == 200, r.text
    items = {i["name"]: i for i in r.json()["employees"]}
    assert items["强者"]["perf_band"] == 3 and items["强者"]["potential_band"] == 3
    assert items["强者"]["calibrated"] is False
    assert items["弱者"]["perf_band"] == 1
    # 白板:无数据→中档+estimated
    assert items["白板"]["estimated"] is True


def test_calibrate_override_and_reset(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t, name="待校准")
    _seed_weak(db, t, e)  # 自动:1_x

    r = client.post(
        f"/api/talent-review/nine-box/{e.id}/calibrate",
        json={"perf_band": 3, "potential_band": 3, "note": "校准会:重点项目立功"},
        headers=_hdr(u),
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["perf_band"] == 3 and b["calibrated"] is True

    grid = client.get("/api/talent-review/nine-box", headers=_hdr(u)).json()
    it = next(i for i in grid["employees"] if i["id"] == e.id)
    assert it["calibrated"] is True and it["auto_perf_band"] == 1

    # 空 bands → 重置回自动
    r2 = client.post(
        f"/api/talent-review/nine-box/{e.id}/calibrate",
        json={}, headers=_hdr(u),
    )
    assert r2.json()["calibrated"] is False
    assert r2.json()["perf_band"] == 1


def test_pool_crud_members_flow(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e1, e2 = _emp(db, t, name="甲"), _emp(db, t, name="乙")

    p = client.post("/api/talent-pools", json={
        "name": "后端继任池", "kind": "succession",
        "target_designation": "架构师",
    }, headers=_hdr(u))
    assert p.status_code == 201, p.text
    pid = p.json()["id"]

    # 一键入池(批量)
    m = client.post(f"/api/talent-pools/{pid}/members", json={
        "employee_ids": [e1.id, e2.id], "readiness": 2,
    }, headers=_hdr(u))
    assert m.status_code == 201 and m.json()["added"] == 2
    assert m.json()["member_count"] == 2

    # 阶段流转 in→developing→ready→exited
    for stage in ("developing", "ready", "exited"):
        r = client.patch(
            f"/api/talent-pools/{pid}/members/{e1.id}",
            json={"stage": stage}, headers=_hdr(u),
        )
        assert r.status_code == 200 and r.json()["stage"] == stage

    detail = client.get(f"/api/talent-pools/{pid}", headers=_hdr(u)).json()
    assert detail["saturation"]["exited"] == 1
    assert detail["member_count"] == 1  # exited 不计入活跃

    # 移除成员
    d = client.delete(
        f"/api/talent-pools/{pid}/members/{e2.id}", headers=_hdr(u))
    assert d.status_code == 204


def test_pool_member_nine_box_bands(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t, name="明星")
    _seed_strong(db, t, e)
    pid = client.post("/api/talent-pools", json={
        "name": "高潜池", "kind": "high_potential",
    }, headers=_hdr(u)).json()["id"]
    client.post(f"/api/talent-pools/{pid}/members",
                json={"employee_ids": [e.id]}, headers=_hdr(u))
    # 先触发九宫格落位,再读池详情应带 band
    client.get("/api/talent-review/nine-box", headers=_hdr(u))
    detail = client.get(f"/api/talent-pools/{pid}", headers=_hdr(u)).json()
    mem = detail["members"][0]
    assert mem["perf_band"] == 3 and mem["potential_band"] == 3


def test_isolation_and_perms(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    e2 = _emp(db, t2)

    # 跨租户校准 404
    r = client.post(f"/api/talent-review/nine-box/{e2.id}/calibrate",
                    json={"perf_band": 3}, headers=_hdr(u1))
    assert r.status_code == 404
    # 跨租户池操作 404
    p2 = client.post("/api/talent-pools", json={"name": "P"},
                     headers=_hdr(_user(db, t2))).json()
    r2 = client.post(f"/api/talent-pools/{p2['id']}/members",
                     json={"employee_ids": [e2.id]}, headers=_hdr(u1))
    assert r2.status_code == 404
    # 无权限角色 403
    viewer = _user(db, t1, role="employee")
    r3 = client.get("/api/talent-review/nine-box", headers=_hdr(viewer))
    assert r3.status_code == 403
