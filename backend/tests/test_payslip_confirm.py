"""工资条电子签收(薪人薪事同款):确认/撤回/催签收路测。

- 员工 confirm:仅 issued 可见可签,幂等;draft 与他人工资条 404。
- 发放推送:issue → payslip.issued 通知。
- 催签收:nudge → payslip.nudge 通知,20h 去重;已签收 409;draft 400。
- 撤回:retract 清签收态,重发需重新签收(薪人薪事:收回重发需重新确认)。
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Employee,
    Notification,
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
    t = Tenant(name=f"ps-{uuid.uuid4().hex[:8]}")
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


def _employee(db, t, u, name="员工甲") -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name, user_id=u.id, status="active",
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _make_slip(client, hr, emp, period="2026-10"):
    r = client.post(
        "/api/payroll/slips",
        json={
            "employee_id": emp.id,
            "period": period,
            "items": [
                {"label": "基本工资", "kind": "earning", "amount": 10000},
                {"label": "社保", "kind": "deduction", "amount": 800},
            ],
        },
        headers=_hdr(hr),
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_confirm_flow_idempotent(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)
    slip = _make_slip(client, hr, e)

    # draft:员工不可见不可签
    r = client.post(
        f"/api/me/salary-slips/{slip['id']}/confirm", headers=_hdr(eu)
    )
    assert r.status_code == 404

    r = client.post(f"/api/payroll/slips/{slip['id']}/issue", headers=_hdr(hr))
    assert r.status_code == 200

    # 员工列表出现且未签收
    r = client.get("/api/me/salary-slips", headers=_hdr(eu))
    mine = next(x for x in r.json() if x["id"] == slip["id"])
    assert mine["confirmed_at"] is None

    # 签收 + 幂等
    r = client.post(
        f"/api/me/salary-slips/{slip['id']}/confirm", headers=_hdr(eu)
    )
    assert r.status_code == 200, r.text
    ts1 = r.json()["confirmed_at"]
    assert ts1
    r = client.post(
        f"/api/me/salary-slips/{slip['id']}/confirm", headers=_hdr(eu)
    )
    assert r.status_code == 200
    assert r.json()["confirmed_at"] == ts1  # 幂等不更新时间

    # HR 侧可见 confirmed_at
    r = client.get(f"/api/payroll/slips/{slip['id']}", headers=_hdr(hr))
    assert r.json()["confirmed_at"] == ts1

    # HR 侧再催签收 → 409
    r = client.post(
        f"/api/payroll/slips/{slip['id']}/nudge", headers=_hdr(hr)
    )
    assert r.status_code == 409


def test_retract_requires_reconfirm(db, client):
    """收回清签收态,重发后需重新签收。"""
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)
    slip = _make_slip(client, hr, e, period="2026-09")

    client.post(f"/api/payroll/slips/{slip['id']}/issue", headers=_hdr(hr))
    client.post(f"/api/me/salary-slips/{slip['id']}/confirm", headers=_hdr(eu))

    # 收回:清签收 + 员工不可见
    r = client.post(f"/api/payroll/slips/{slip['id']}/retract", headers=_hdr(hr))
    assert r.status_code == 200
    assert r.json()["confirmed_at"] is None
    r = client.get("/api/me/salary-slips", headers=_hdr(eu))
    assert not [x for x in r.json() if x["id"] == slip["id"]]

    # 重发:需重新签收
    client.post(f"/api/payroll/slips/{slip['id']}/issue", headers=_hdr(hr))
    r = client.get("/api/me/salary-slips", headers=_hdr(eu))
    mine = next(x for x in r.json() if x["id"] == slip["id"])
    assert mine["confirmed_at"] is None


def test_issue_notifies_and_nudge_cooldown(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)
    slip = _make_slip(client, hr, e, period="2026-08")

    client.post(f"/api/payroll/slips/{slip['id']}/issue", headers=_hdr(hr))
    n = db.scalars(
        select(Notification).where(
            Notification.user_id == eu.id,
            Notification.kind == "payslip.issued",
        )
    ).all()
    assert len(n) == 1
    assert slip["id"] in n[0].link

    # 催签收:发出 + cooldown 去重
    r = client.post(f"/api/payroll/slips/{slip['id']}/nudge", headers=_hdr(hr))
    assert r.status_code == 200 and r.json()["sent"] == 1
    r = client.post(f"/api/payroll/slips/{slip['id']}/nudge", headers=_hdr(hr))
    assert r.status_code == 200 and r.json()["sent"] == 0
    nudges = db.scalars(
        select(Notification).where(
            Notification.user_id == eu.id,
            Notification.kind == "payslip.nudge",
        )
    ).all()
    assert len(nudges) == 1

    # draft 不能催
    slip2 = _make_slip(client, hr, e, period="2026-07")
    r = client.post(
        f"/api/payroll/slips/{slip2['id']}/nudge", headers=_hdr(hr)
    )
    assert r.status_code == 400


def test_confirm_only_own_slip(db, client):
    """别人的工资条 → 404;跨租户同理。"""
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    other_u = _user(db, t, "employee")
    e = _employee(db, t, eu)
    _employee(db, t, other_u, "员工乙")
    slip = _make_slip(client, hr, e, period="2026-06")
    client.post(f"/api/payroll/slips/{slip['id']}/issue", headers=_hdr(hr))

    r = client.post(
        f"/api/me/salary-slips/{slip['id']}/confirm", headers=_hdr(other_u)
    )
    assert r.status_code == 404
    r = client.get(
        f"/api/me/salary-slips/{slip['id']}", headers=_hdr(other_u)
    )
    assert r.status_code == 404
