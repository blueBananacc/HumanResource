"""人事提醒中心 + 人事日历路测(慧人力智能待办/2号人事日历同款)。

- collect_events:birthday/anniversary 周期展开、probation/contract/cert
  一次性事件、onboarding/separation 生命周期(含逾期)、terminated 排除。
- GET /hr/reminders:分组 + 类型过滤 + 逾期 bucket。
- GET /hr/calendar:月展开(2/29 → 平年 2/28)。
- POST /hr/reminders/greet:送达员工通知、同日幂等、无账号 400。
- run_people_reminders:deadline 7 天窗口发 HR、celebrate 当天 HR+员工祝福、去重。
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Employee,
    LifecycleEvent,
    Notification,
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
    t = Tenant(name=f"pe-{uuid.uuid4().hex[:8]}")
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


def _employee(db, t, name, personal=None, hire=None, status="active", user=None) -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name, status=status, personal=personal, hire_date=hire,
        user_id=user.id if user else None,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def test_reminders_grouping_and_kinds(db, client):
    t = _tenant(db)
    u = _user(db, t)
    today = date.today()
    _employee(db, t, "生日君", personal={
        "birth_date": f"1990-{today.month:02d}-{today.day:02d}",  # 今天生日
        "probation_end": (today + timedelta(days=3)).isoformat(),  # 本周转正
        "contract_end": (today + timedelta(days=20)).isoformat(),  # 本月合同
    })
    _employee(db, t, "周年君", hire=today.replace(year=today.year - 3))
    _employee(db, t, "离职君", personal={
        "cert_expire": (today - timedelta(days=5)).isoformat(),  # 已过期证件
    })
    _employee(db, t, "已离职", personal={
        "birth_date": f"1990-{today.month:02d}-{today.day:02d}",
    }, status="terminated")  # terminated 不出生日

    r = client.get("/api/hr/reminders?days=30", headers=_hdr(u))
    assert r.status_code == 200, r.text
    d = r.json()
    kinds_today = {e["kind"] for e in d["today"]}
    assert "birthday" in kinds_today and "anniversary" in kinds_today
    assert {e["employee_name"] for e in d["today"] if e["kind"] == "birthday"} == {"生日君"}
    # 同年窗口内周期事件不重复展开
    bday_rows = [e for e in d["today"]
                 if e["kind"] == "birthday" and e["employee_name"] == "生日君"]
    assert len(bday_rows) == 1
    kinds_week = {e["kind"] for e in d["week"]}
    assert "probation_end" in kinds_week
    assert any(e["kind"] == "contract_end" for e in d["later"])
    # 逾期 bucket:cert_expire 5 天前
    assert any(e["kind"] == "cert_expire" and e["employee_name"] == "离职君"
               for e in d["overdue"])
    # 类型过滤
    r2 = client.get("/api/hr/reminders?days=30&kind=birthday", headers=_hdr(u))
    d2 = r2.json()
    allk = {e["kind"] for g in ("today", "week", "later", "overdue") for e in d2[g]}
    assert allk <= {"birthday"}


def test_calendar_month_and_leap(db, client):
    t = _tenant(db)
    u = _user(db, t)
    # 闰年生人 → 平年 2/28
    _employee(db, t, "闰年君", personal={"birth_date": "2000-02-29"})
    _employee(db, t, "三月君", personal={"contract_end": "2026-03-15"})
    r = client.get("/api/hr/calendar?month=2026-02", headers=_hdr(u))
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    feb29 = [i for i in items if i["kind"] == "birthday" and i["employee_name"] == "闰年君"]
    assert feb29 and feb29[0]["day"] == "2026-02-28"
    assert not any(i["employee_name"] == "三月君" for i in items)
    r = client.get("/api/hr/calendar?month=2026-03", headers=_hdr(u))
    assert any(i["kind"] == "contract_end" for i in r.json()["items"])


def test_lifecycle_onboarding_separation(db, client):
    t = _tenant(db)
    u = _user(db, t)
    today = date.today()
    e1 = _employee(db, t, "待入职君")
    db.add(LifecycleEvent(
        tenant_id=t.id, employee_id=e1.id, kind="onboarding",
        status="in_progress", effective_date=today + timedelta(days=5),
    ))
    e2 = _employee(db, t, "逾期离职君")
    db.add(LifecycleEvent(
        tenant_id=t.id, employee_id=e2.id, kind="separation",
        status="in_progress", effective_date=today - timedelta(days=2),
    ))
    db.commit()
    r = client.get("/api/hr/reminders?days=30", headers=_hdr(u))
    d = r.json()
    assert any(e["kind"] == "onboarding" and e["employee_name"] == "待入职君"
               for e in d["week"])
    assert any(e["kind"] == "separation" and e["employee_name"] == "逾期离职君"
               for e in d["overdue"])


def test_greet_idempotent_and_400(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, "寿星", user=eu)
    noacc = _employee(db, t, "无账号君")

    r = client.post("/api/hr/reminders/greet", json={
        "employee_id": e.id, "kind": "birthday"}, headers=_hdr(hr))
    assert r.status_code == 200 and r.json()["sent"] == 1
    n = db.scalars(select(Notification).where(
        Notification.user_id == eu.id, Notification.kind == "people.greeting")).all()
    assert len(n) == 1 and "生日" in n[0].title
    # 同日幂等
    r = client.post("/api/hr/reminders/greet", json={
        "employee_id": e.id, "kind": "birthday"}, headers=_hdr(hr))
    assert r.json()["sent"] == 0
    # 无账号 → 400;非法 kind → 422;他人员工 → 404(租户隔离由 _employee 同租户保证,用不存在的 id)
    r = client.post("/api/hr/reminders/greet", json={
        "employee_id": noacc.id, "kind": "birthday"}, headers=_hdr(hr))
    assert r.status_code == 400
    r = client.post("/api/hr/reminders/greet", json={
        "employee_id": e.id, "kind": "contract_end"}, headers=_hdr(hr))
    assert r.status_code == 422
    r = client.post("/api/hr/reminders/greet", json={
        "employee_id": "nonexistent", "kind": "birthday"}, headers=_hdr(hr))
    assert r.status_code == 404


def test_people_reminders_sweep(db, client):
    """每日扫:deadline 进 7 天窗口发 HR;生日当天 HR+员工双通知;日内去重。"""
    from app.services.hr_events import run_people_reminders

    t = _tenant(db)
    hr = _user(db, t, "hr")
    eu = _user(db, t, "employee")
    today = date.today()
    _employee(db, t, "转正君", personal={
        "probation_end": (today + timedelta(days=5)).isoformat(),
    })
    _employee(db, t, "寿星君", personal={
        "birth_date": f"1992-{today.month:02d}-{today.day:02d}",
    }, user=eu)
    _employee(db, t, "远期君", personal={
        "contract_end": (today + timedelta(days=60)).isoformat(),  # 超窗
    })

    fired = run_people_reminders(db, today=today)
    assert fired >= 3  # 转正→hr, 生日→hr + 员工祝福
    hr_notes = db.scalars(select(Notification).where(
        Notification.user_id == hr.id, Notification.kind == "people.reminder")).all()
    titles = [n.title for n in hr_notes]
    assert any("转正到期" in x and "转正君" in x for x in titles)
    assert any("生日" in x and "寿星君" in x for x in titles)
    assert not any("远期君" in x for x in titles)
    emp_notes = db.scalars(select(Notification).where(
        Notification.user_id == eu.id, Notification.kind == "people.reminder")).all()
    assert any("生日快乐" in n.title for n in emp_notes)
    # 再跑一轮 → 去重不再发
    fired2 = run_people_reminders(db, today=today)
    assert fired2 == 0
