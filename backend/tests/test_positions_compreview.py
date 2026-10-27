"""职位管理 + 调薪矩阵路测。

职位:占坑/释放/状态流转/空缺率;矩阵:绩效×compa-ratio 落位与决策流。
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
    SalaryBand,
    SalarySlip,
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
    t = Tenant(name=f"pc-{uuid.uuid4().hex[:8]}")
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


def _mk_position(client, u, title="高级工程师"):
    r = client.post("/api/positions", json={
        "title": title, "designation": "工程师",
    }, headers=_hdr(u))
    assert r.status_code == 201, r.text
    return r.json()


# ---------------- 职位管理 ----------------


def test_position_fill_vacate_flow(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    pos = _mk_position(client, u)
    assert pos["status"] == "open"

    # 占坑
    f = client.post(f"/api/positions/{pos['id']}/fill",
                    json={"employee_id": e.id}, headers=_hdr(u))
    assert f.status_code == 200, f.text
    assert f.json()["status"] == "filled" and f.json()["incumbent_name"] == "张三"
    db.refresh(e)
    assert e.position_id == pos["id"]

    # 重复占坑 409
    e2 = _emp(db, t, name="李四")
    f2 = client.post(f"/api/positions/{pos['id']}/fill",
                     json={"employee_id": e2.id}, headers=_hdr(u))
    assert f2.status_code == 409

    # 释放
    v = client.post(f"/api/positions/{pos['id']}/vacate", headers=_hdr(u))
    assert v.json()["status"] == "open"
    db.refresh(e)
    assert e.position_id is None


def test_position_transfer_and_stats(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    p1 = _mk_position(client, u, "P1")
    p2 = _mk_position(client, u, "P2")
    client.post(f"/api/positions/{p1['id']}/fill",
                json={"employee_id": e.id}, headers=_hdr(u))
    # 换岗:P1 自动释放变 open
    client.post(f"/api/positions/{p2['id']}/fill",
                json={"employee_id": e.id}, headers=_hdr(u))
    lst = client.get("/api/positions", headers=_hdr(u)).json()
    stats = lst["stats"]
    assert stats["open"] == 1 and stats["filled"] == 1
    assert stats["vacancy_rate"] == 50.0


def test_position_frozen_and_closed(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    pos = _mk_position(client, u)
    # 冻结后不可占坑
    client.patch(f"/api/positions/{pos['id']}",
                 json={"status": "frozen"}, headers=_hdr(u))
    r = client.post(f"/api/positions/{pos['id']}/fill",
                    json={"employee_id": e.id}, headers=_hdr(u))
    assert r.status_code == 400
    # 解冻→占坑→closed 拒绝(有人在编)
    client.patch(f"/api/positions/{pos['id']}",
                 json={"status": "open"}, headers=_hdr(u))
    client.post(f"/api/positions/{pos['id']}/fill",
                json={"employee_id": e.id}, headers=_hdr(u))
    # 在编职位:open/frozen/closed 一律拒绝,只能 vacate
    for s in ("open", "frozen", "closed"):
        c = client.patch(f"/api/positions/{pos['id']}",
                         json={"status": s}, headers=_hdr(u))
        assert c.status_code == 400, s


# ---------------- 调薪矩阵 ----------------


def _seed_comp(db, t, e, rating, gross, band_min, band_max):
    cyc = AppraisalCycle(tenant_id=t.id, name="C", status="open",
                         period_start=date(2026, 1, 1), period_end=date(2026, 12, 31))
    db.add(cyc)
    db.flush()
    db.add(Appraisal(tenant_id=t.id, cycle_id=cyc.id, employee_id=e.id,
                     manager_rating=rating, status="reviewed"))
    db.add(SalarySlip(tenant_id=t.id, employee_id=e.id, period="2026-10",
                      gross=gross, status="issued"))
    if not db.query(SalaryBand).filter_by(tenant_id=t.id, name="工程带宽").first():
        db.add(SalaryBand(tenant_id=t.id, name="工程带宽", designation="工程师",
                          gross_min=band_min, gross_max=band_max, is_active=True))
    db.commit()


def test_matrix_placement_and_decision(client, db):
    t = _tenant(db)
    u = _user(db, t)
    # 高绩效(5)×低带宽比(8000/中点15000=0.53)→建议12%
    hi = _emp(db, t, name="高绩效")
    _seed_comp(db, t, hi, rating=5, gross=8000, band_min=10000, band_max=20000)
    # 低绩效(1)×高带宽比(25000/15000=1.67)→建议0%
    lo = _emp(db, t, name="低绩效")
    _seed_comp(db, t, lo, rating=1, gross=25000, band_min=10000, band_max=20000)
    # 无薪资数据→中档+estimated
    _emp(db, t, name="白板")

    m = client.get("/api/comp-review/matrix", headers=_hdr(u))
    assert m.status_code == 200, m.text
    b = m.json()
    emps = {i["name"]: i for i in b["employees"]}
    assert emps["高绩效"]["perf_band"] == 3
    assert emps["高绩效"]["ratio_band"] == "low"
    assert emps["高绩效"]["suggested_pct"] == 12.0
    assert emps["低绩效"]["suggested_pct"] == 0.0
    assert emps["白板"]["estimated"] is True
    assert b["stats"]["est_cost"] == 8000 * 0.12

    # 记录决策(调 15%)
    d = client.post("/api/comp-review/decisions", json={
        "employee_id": hi.id, "final_pct": 15.0, "note": "破格",
    }, headers=_hdr(u))
    assert d.status_code == 201, d.text
    did = d.json()["id"]
    assert d.json()["suggested_pct"] == 12.0 and d.json()["final_pct"] == 15.0

    # 批准→不可再改
    a = client.post(f"/api/comp-review/decisions/{did}/approve", headers=_hdr(u))
    assert a.json()["status"] == "approved"
    r = client.post("/api/comp-review/decisions", json={
        "employee_id": hi.id, "final_pct": 20.0,
    }, headers=_hdr(u))
    assert r.status_code == 409


def test_isolation_and_perms(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    # 跨租户职位/员工 404
    p2 = client.post("/api/positions", json={"title": "x"},
                     headers=_hdr(_user(db, t2))).json()
    assert client.post(f"/api/positions/{p2['id']}/vacate",
                       headers=_hdr(u1)).status_code == 404
    # 无权限角色
    viewer = _user(db, t1, role="employee")
    assert client.get("/api/positions", headers=_hdr(viewer)).status_code == 403
    assert client.get("/api/comp-review/matrix", headers=_hdr(viewer)).status_code in (403, 404)
