"""排班 2.0(喔趣):合规校验 + 员工自助(班次/可用性/换班)+ 模板循环。

- 规则:连班/周工时/休息间隔/可用性;create/reassign 阻断 409,force 放行。
- 员工侧:我的班次、可用性维护、换班(确认→审批→执行交换/转出)、通知。
- 模板:周期 pattern 套用,跳过已有排班/请假,合规告警不阻断。
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
    Notification,
    ShiftAssignment,
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
    t = Tenant(name=f"ss-{uuid.uuid4().hex[:8]}")
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


def _employee(db, t, u=None, name="员工甲") -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name, user_id=u.id if u else None, status="active",
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _shift(client, hr, name="早班", start=540, end=1020):
    r = client.post("/api/shifts", json={
        "name": f"{name}-{uuid.uuid4().hex[:4]}",
        "start_min": start, "end_min": end, "required": 1,
    }, headers=_hdr(hr))
    assert r.status_code == 201, r.text
    return r.json()


def _assign(client, hr, emp_id, shift_id, day, force=False):
    r = client.post("/api/shifts/assignments", json={
        "employee_id": emp_id, "shift_id": shift_id,
        "day": day.isoformat(), "force": force,
    }, headers=_hdr(hr))
    return r


# ---------- 合规规则 ----------


def test_rules_crud_and_availability_block(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)

    # 默认规则
    r = client.get("/api/shifts/rules", headers=_hdr(hr))
    assert r.status_code == 200
    assert r.json()["max_consecutive_days"] == 6

    # 收紧:连班 ≤ 2 天触发 block
    r = client.put("/api/shifts/rules", json={
        "max_consecutive_days": 2, "max_weekly_hours": 40,
        "min_rest_minutes": 720, "respect_availability": True,
    }, headers=_hdr(hr))
    assert r.status_code == 200

    s = _shift(client, hr)
    d0 = date.today() + timedelta(days=30)
    assert _assign(client, hr, e.id, s["id"], d0).status_code == 201
    assert _assign(client, hr, e.id, s["id"], d0 + timedelta(days=1)).status_code == 201
    # 第 3 个连续日 → 409 带违规明细
    r = _assign(client, hr, e.id, s["id"], d0 + timedelta(days=2))
    assert r.status_code == 409, r.text
    vs = r.json()["detail"]["violations"]
    assert any(v["rule"] == "consecutive_days" for v in vs)
    # force 放行
    r = _assign(client, hr, e.id, s["id"], d0 + timedelta(days=2), force=True)
    assert r.status_code == 201


def test_compliance_window_and_availability(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu, "员工乙")
    s = _shift(client, hr)
    # 员工把某天登记为不可排
    d = date.today() + timedelta(days=45)
    r = client.put("/api/me/shift-availability", json={
        "items": [{"weekday": d.weekday(), "preference": "unavailable"}],
    }, headers=_hdr(eu))
    assert r.status_code == 200
    # 该日排班 → 409 availability
    r = _assign(client, hr, e.id, s["id"], d)
    assert r.status_code == 409
    assert any(v["rule"] == "availability" for v in r.json()["detail"]["violations"])
    # force 后合规体检可见
    _assign(client, hr, e.id, s["id"], d, force=True)
    r = client.get(
        f"/api/shifts/compliance?month={d.isoformat()[:7]}", headers=_hdr(hr))
    assert r.status_code == 200
    mine = [v for v in r.json()["violations"] if v["employee_id"] == e.id]
    assert any(v["rule"] == "availability" for v in mine)


# ---------- 员工侧:我的班次 / 可用性 / 换班 ----------


def test_me_shifts_and_availability(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)
    s = _shift(client, hr)
    d = date.today() + timedelta(days=60)
    _assign(client, hr, e.id, s["id"], d)

    r = client.get(f"/api/me/shifts?month={d.isoformat()[:7]}",
                   headers=_hdr(eu))
    assert r.status_code == 200
    mine = [x for x in r.json() if x["day"] == d.isoformat()]
    assert len(mine) == 1 and mine[0]["shift_name"] == s["name"]

    r = client.get("/api/me/shift-availability", headers=_hdr(eu))
    assert len(r.json()["items"]) == 7
    assert all(x["preference"] == "available" for x in r.json()["items"])

    r = client.put("/api/me/shift-availability", json={
        "items": [{"weekday": 6, "preference": "unavailable"},
                  {"weekday": 0, "preference": "preferred"}],
    }, headers=_hdr(eu))
    assert r.status_code == 200
    items = {x["weekday"]: x["preference"] for x in r.json()["items"]}
    assert items[6] == "unavailable" and items[0] == "preferred"


def test_swap_flow_exchange(db, client):
    """A(早班)↔B(晚班)同一天:发起→对方同意→HR批准→shift_id 互换。"""
    t = _tenant(db)
    hr = _user(db, t)
    ua = _user(db, t, "employee")
    ub = _user(db, t, "employee")
    ea = _employee(db, t, ua, "员工A")
    eb = _employee(db, t, ub, "员工B")
    sa = _shift(client, hr, "早班", 540, 1020)
    sb = _shift(client, hr, "晚班", 840, 1320)
    d = date.today() + timedelta(days=75)
    _assign(client, hr, ea.id, sa["id"], d)
    _assign(client, hr, eb.id, sb["id"], d)

    # 发起(双方有班 → exchange)
    r = client.post("/api/me/shift-swaps", json={
        "target_employee_id": eb.id, "day": d.isoformat(),
        "note": "家里有事",
    }, headers=_hdr(ua))
    assert r.status_code == 201, r.text
    swap = r.json()
    assert swap["mode"] == "exchange" and swap["status"] == "pending_peer"
    # 对方收到通知
    n = db.scalars(select(Notification).where(
        Notification.user_id == ub.id,
        Notification.kind == "shift.swap")).all()
    assert len(n) == 1

    # 同天重复发起 → 409
    r = client.post("/api/me/shift-swaps", json={
        "target_employee_id": eb.id, "day": d.isoformat(),
    }, headers=_hdr(ua))
    assert r.status_code == 409

    # 非目标员工不能确认
    r = client.post(f"/api/me/shift-swaps/{swap['id']}/peer",
                    json={"accept": True}, headers=_hdr(ua))
    assert r.status_code == 404

    # 对方同意 → pending_approval
    r = client.post(f"/api/me/shift-swaps/{swap['id']}/peer",
                    json={"accept": True}, headers=_hdr(ub))
    assert r.status_code == 200
    assert r.json()["status"] == "pending_approval"

    # HR 待办可见 + 批准
    r = client.get("/api/shifts/swaps?status=pending_approval",
                   headers=_hdr(hr))
    assert len(r.json()) == 1
    r = client.post(f"/api/shifts/swaps/{swap['id']}/decide",
                    json={"approve": True}, headers=_hdr(hr))
    assert r.status_code == 200, r.text
    assert r.json()["swap"]["status"] == "approved"

    # 班次已互换
    a1 = db.scalars(select(ShiftAssignment).where(
        ShiftAssignment.employee_id == ea.id,
        ShiftAssignment.day == d)).one()
    a2 = db.scalars(select(ShiftAssignment).where(
        ShiftAssignment.employee_id == eb.id,
        ShiftAssignment.day == d)).one()
    assert a1.shift_id == sb["id"] and a2.shift_id == sa["id"]
    assert a1.source == "swap"


def test_swap_transfer_decline_cancel(db, client):
    """对方当天无班 → transfer;拒绝/撤回路径。"""
    t = _tenant(db)
    hr = _user(db, t)
    ua = _user(db, t, "employee")
    ub = _user(db, t, "employee")
    ea = _employee(db, t, ua, "员工C")
    eb = _employee(db, t, ub, "员工D")
    s = _shift(client, hr)
    d = date.today() + timedelta(days=90)
    _assign(client, hr, ea.id, s["id"], d)

    r = client.post("/api/me/shift-swaps", json={
        "target_employee_id": eb.id, "day": d.isoformat(),
    }, headers=_hdr(ua))
    assert r.status_code == 201
    assert r.json()["mode"] == "transfer"
    sw = r.json()

    # 对方拒绝 → peer_declined,发起方收到通知
    r = client.post(f"/api/me/shift-swaps/{sw['id']}/peer",
                    json={"accept": False, "note": "不好意思"},
                    headers=_hdr(ub))
    assert r.json()["status"] == "peer_declined"

    # 再发起 → 撤回
    r = client.post("/api/me/shift-swaps", json={
        "target_employee_id": eb.id, "day": d.isoformat(),
    }, headers=_hdr(ua))
    sw2 = r.json()
    r = client.post(f"/api/me/shift-swaps/{sw2['id']}/cancel",
                    headers=_hdr(ua))
    assert r.json()["status"] == "cancelled"

    # 第三次:同意 + 批准 → 班归 B
    r = client.post("/api/me/shift-swaps", json={
        "target_employee_id": eb.id, "day": d.isoformat(),
    }, headers=_hdr(ua))
    sw3 = r.json()
    client.post(f"/api/me/shift-swaps/{sw3['id']}/peer",
                json={"accept": True}, headers=_hdr(ub))
    r = client.post(f"/api/shifts/swaps/{sw3['id']}/decide",
                    json={"approve": True}, headers=_hdr(hr))
    assert r.status_code == 200
    a = db.scalars(select(ShiftAssignment).where(
        ShiftAssignment.employee_id == eb.id,
        ShiftAssignment.day == d)).one()
    assert a.source == "swap"


# ---------- 模板 / 循环 ----------


def test_template_apply_cycle(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu)
    sa = _shift(client, hr, "早班")
    sb = _shift(client, hr, "晚班")
    start = date.today() + timedelta(days=120)
    # 4 天循环:早 早 晚 休
    r = client.post("/api/shifts/templates", json={
        "name": "早早晚休", "period_days": 4,
        "pattern": [
            {"offset": 0, "shift_id": sa["id"]},
            {"offset": 1, "shift_id": sa["id"]},
            {"offset": 2, "shift_id": sb["id"]},
            {"offset": 3, "shift_id": None},
        ],
    }, headers=_hdr(hr))
    assert r.status_code == 201, r.text
    tid = r.json()["id"]

    # dry_run 预览:2 个周期 = 8 天,6 条排班(2 休)
    r = client.post(f"/api/shifts/templates/{tid}/apply", json={
        "employee_ids": [e.id], "start": start.isoformat(),
        "cycles": 2, "dry_run": True,
    }, headers=_hdr(hr))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["planned"] == 6
    days = sorted(p["day"] for p in out["plan"])
    assert days == [d.isoformat() for d in
                    [start, start + timedelta(1), start + timedelta(2),
                     start + timedelta(4), start + timedelta(5),
                     start + timedelta(6)]]
    assert db.scalars(select(ShiftAssignment).where(
        ShiftAssignment.employee_id == e.id,
        ShiftAssignment.day >= start)).all() == []  # dry_run 不落库

    # 落库 + 重跑跳过已有
    r = client.post(f"/api/shifts/templates/{tid}/apply", json={
        "employee_ids": [e.id], "start": start.isoformat(),
        "cycles": 2, "dry_run": False,
    }, headers=_hdr(hr))
    assert r.json()["planned"] == 6
    r = client.post(f"/api/shifts/templates/{tid}/apply", json={
        "employee_ids": [e.id], "start": start.isoformat(),
        "cycles": 2, "dry_run": False,
    }, headers=_hdr(hr))
    assert r.json()["planned"] == 0
    assert len(r.json()["skipped"]) == 6
    rows = db.scalars(select(ShiftAssignment).where(
        ShiftAssignment.employee_id == e.id,
        ShiftAssignment.day >= start)).all()
    assert len(rows) == 6 and all(x.source == "template" for x in rows)


# ---------- AI 排班助手(盖雅时间智能体):看板 / 解释 / 方案对比 ----------


def test_day_board(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu, "看板员工")
    s = _shift(client, hr, "看板早班")
    d = date.today() + timedelta(days=75)
    _assign(client, hr, e.id, s["id"], d)

    r = client.get(f"/api/shifts/board?day={d.isoformat()}", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    slot = next(x for x in r.json()["slots"] if x["shift_id"] == s["id"])
    assert slot["required"] == 1 and slot["covered"] == 1 and slot["missing"] == 0
    assert slot["assigned"][0]["employee_id"] == e.id
    assert slot["assigned"][0]["on_leave"] is False


def test_explain_schedule(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    eu = _user(db, t, "employee")
    e = _employee(db, t, eu, "解释员工")
    s = _shift(client, hr, "解释班")
    d = date.today() + timedelta(days=80)

    # 空闲日:reasons 说明可排,假设排班合规预判 ok
    r = client.get(
        f"/api/shifts/explain?employee_id={e.id}&day={d.isoformat()}"
        f"&shift_id={s['id']}", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["assignment"] is None and j["on_leave"] is False
    assert j["hypothetical"]["ok"] is True
    assert any("空闲" in x for x in j["reasons"])

    # 登记不可排 → 原因 + hypothetical 违规
    client.put("/api/me/shift-availability", json={
        "items": [{"weekday": d.weekday(), "preference": "unavailable"}],
    }, headers=_hdr(eu))
    r = client.get(
        f"/api/shifts/explain?employee_id={e.id}&day={d.isoformat()}"
        f"&shift_id={s['id']}", headers=_hdr(hr))
    j = r.json()
    assert any("不可排" in x for x in j["reasons"])
    assert j["hypothetical"]["ok"] is False
    assert any(v["rule"] == "availability"
               for v in j["hypothetical"]["violations"])

    # 已排班 → assignment 呈现
    _assign(client, hr, e.id, s["id"], d, force=True)
    r = client.get(
        f"/api/shifts/explain?employee_id={e.id}&day={d.isoformat()}",
        headers=_hdr(hr))
    j = r.json()
    assert j["assignment"]["shift_id"] == s["id"]
    assert j["consecutive_days_to_date"] == 1
    assert any("已排" in x for x in j["reasons"])


def test_plan_compare(db, client):
    t = _tenant(db)
    hr = _user(db, t)
    e_free = _employee(db, t, name="空闲员工")
    e_pref = _employee(db, t, name="偏好员工")
    e_busy = _employee(db, t, name="已排员工")
    s = _shift(client, hr, "对比班")
    d = date.today() + timedelta(days=85)
    # e_busy 当天已有别的班 → ineligible
    s2 = _shift(client, hr, "占用班")
    _assign(client, hr, e_busy.id, s2["id"], d)
    # e_pref 登记偏好日
    from app.domain.models import EmployeeAvailability
    db.add(EmployeeAvailability(
        tenant_id=t.id, employee_id=e_pref.id,
        weekday=d.weekday(), preference="preferred"))
    db.commit()

    r = client.get(
        f"/api/shifts/plan-compare?day={d.isoformat()}&shift_id={s['id']}",
        headers=_hdr(hr))
    assert r.status_code == 200, r.text
    j = r.json()
    ids = [p["employee_id"] for p in j["plans"]]
    assert e_free.id in ids and e_pref.id in ids
    assert e_busy.id not in ids
    assert any(x["employee_id"] == e_busy.id for x in j["ineligible"])
    # 偏好日员工排在最前(preferred 加权)
    assert j["plans"][0]["employee_id"] == e_pref.id
    assert any("偏好" in p for p in j["plans"][0]["pros"])

    # 连班收紧后 e_free 引入违规 → eligible=False + cons 有违规项
    client.put("/api/shifts/rules", json={
        "max_consecutive_days": 1, "max_weekly_hours": 40,
        "min_rest_minutes": 720, "respect_availability": True,
    }, headers=_hdr(hr))
    _assign(client, hr, e_free.id, s2["id"], d - timedelta(days=1), force=True)
    r = client.get(
        f"/api/shifts/plan-compare?day={d.isoformat()}&shift_id={s['id']}",
        headers=_hdr(hr))
    j = r.json()
    p_free = next(p for p in j["plans"] if p["employee_id"] == e_free.id)
    assert p_free["eligible"] is False
    assert any("违规" in c for c in p_free["cons"])
