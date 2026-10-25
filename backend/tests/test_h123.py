# -*- coding: utf-8 -*-
"""H1-H3 人事助手场景测试。

覆盖:
- H1 bulk excuse:批量免计(正常/跳过已免/越权 denied)/day 过滤
- H1 hr_movements:入离调升聚合、月份口径、双算去重、租户隔离
- H2 org changes:preview 规范化+错误、apply 事务+LifecycleEvent+reports_to
- H3 scheduling:生成器跳过请假/填缺口/gaps 虚覆盖/auto_cover 换班
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-h123")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-h123"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Attendance, AttendanceFlag, Base, Employee, LeaveApplication,
    LeaveType, LifecycleEvent, OrgUnit, Shift, ShiftAssignment, Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.hr_movements import build_movements_report  # noqa: E402
from app.services.scheduling import (  # noqa: E402
    auto_cover, find_gaps, generate_schedule, month_range,
)


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
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
    t = Tenant(name=f"h-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, role="admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _emp(db, t, name, org=None, no=None, hire=None, status="active") -> Employee:
    e = Employee(
        tenant_id=t.id, name=name,
        employee_no=no or uuid.uuid4().hex[:8],
        org_unit_id=org.id if org else None,
        hire_date=hire, status=status,
    )
    db.add(e)
    db.commit()
    return e


def _org(db, t, name, parent=None) -> OrgUnit:
    o = OrgUnit(tenant_id=t.id, name=name,
                parent_id=parent.id if parent else None)
    db.add(o)
    db.commit()
    return o


# ---------------------------------------------------------------------------
# H1:考勤异常批量免计 + day 过滤
# ---------------------------------------------------------------------------

def _flag(db, t, emp, day, kind="late_in", minutes=15, excused=False):
    att = db.scalars(
        __import__("sqlalchemy").select(Attendance).where(
            Attendance.employee_id == emp.id, Attendance.day == day)).first()
    if att is None:
        att = Attendance(tenant_id=t.id, employee_id=emp.id, day=day,
                         status="present")
        db.add(att)
        db.commit()
    f = AttendanceFlag(tenant_id=t.id, attendance_id=att.id,
                       employee_id=emp.id, day=day, kind=kind,
                       minutes=minutes, excused=excused)
    db.add(f)
    db.commit()
    return f


def test_flags_day_filter_and_bulk_excuse(db, client):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t, "迟到大王")
    d1, d2 = date(2026, 2, 10), date(2026, 2, 11)
    f1 = _flag(db, t, e, d1)
    f2 = _flag(db, t, e, d1, kind="early_out")
    f3 = _flag(db, t, e, d2)
    f4 = _flag(db, t, e, d2, kind="early_out", excused=True)

    r = client.get(f"/api/attendance/flags?day={d1}", headers=_hdr(u))
    assert r.status_code == 200, r.text
    assert {i["id"] for i in r.json()["items"]} == {f1.id, f2.id}

    r = client.post("/api/attendance/flags/bulk-excuse", headers=_hdr(u),
                    json={"flag_ids": [f1.id, f2.id, f3.id, f4.id, "missing"],
                          "note": "核对过"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["excused"] == 3 and body["skipped"] == 1 and body["denied"] == 0
    db.refresh(f1)
    assert f1.excused and f1.excused_note == "核对过"
    # 幂等重放:全部已免 → excused=0 skipped=4
    r = client.post("/api/attendance/flags/bulk-excuse", headers=_hdr(u),
                    json={"flag_ids": [f1.id, f2.id, f3.id, f4.id]})
    assert r.json()["excused"] == 0 and r.json()["skipped"] == 4


def test_bulk_excuse_denied_for_out_of_scope(db, client):
    """manager 角色无考勤权限:can_decide 不覆盖的员工计入 denied。"""
    t = _tenant(db)
    mgr = _user(db, t, role="manager")
    e = _emp(db, t, "普通员工")
    f = _flag(db, t, e, date(2026, 2, 12))
    # manager 无任何考勤权限位 → 端点 403(require_permission 先拦)
    r = client.post("/api/attendance/flags/bulk-excuse", headers=_hdr(mgr),
                    json={"flag_ids": [f.id]})
    assert r.status_code in (403, 401)


# ---------------------------------------------------------------------------
# H1:hr_movements 人事异动月报
# ---------------------------------------------------------------------------

def test_hr_movements_report(db):
    t = _tenant(db)
    u = _user(db, t)
    org_a = _org(db, t, "研发部")
    org_b = _org(db, t, "市场部")
    m_start, _ = month_range(None)  # 当月
    # 上月(默认口径)
    first = m_start.replace(day=1)
    prev_end = first - timedelta(days=1)
    prev_start = prev_end.replace(day=1)
    mid = prev_start + timedelta(days=10)

    e1 = _emp(db, t, "新入职", org_a, hire=mid)
    e2 = _emp(db, t, "离职者", org_a)
    e3 = _emp(db, t, "调动者", org_b)
    # 离职/调走 lifecycle 事件落上月
    db.add_all([
        LifecycleEvent(tenant_id=t.id, employee_id=e2.id, kind="leave",
                       effective_date=mid, status="completed"),
        LifecycleEvent(tenant_id=t.id, employee_id=e3.id, kind="transfer",
                       effective_date=mid, status="completed",
                       payload={"from_org_unit_id": org_b.id,
                                "to_org_unit_id": org_a.id}),
        # join 事件与 hire_date 同一天 → 不双算
        LifecycleEvent(tenant_id=t.id, employee_id=e1.id, kind="join",
                       effective_date=mid, status="completed"),
        # 取消的事件不计
        LifecycleEvent(tenant_id=t.id, employee_id=e3.id, kind="leave",
                       effective_date=mid, status="cancelled"),
    ])
    db.commit()

    r = build_movements_report(db, u)  # 默认上月
    assert r["month"] == prev_start.strftime("%Y-%m")
    assert r["hires"] == 1 and r["departures"] == 1
    assert r["transfers"] == 1 and r["promotions"] == 0
    assert r["net_change"] == 0
    kinds = [i["kind"] for i in r["items"]]
    assert kinds.count("join") == 1  # join 事件 + hire_date 不双算
    by_org = {b["org"]: b for b in r["by_org"]}
    assert by_org["研发部"]["join"] == 1
    assert by_org["研发部"]["leave"] == 1
    assert by_org["市场部"]["transfer"] == 1


def test_hr_movements_api_and_isolation(db, client):
    t, t2 = _tenant(db), _tenant(db)
    u, u2 = _user(db, t), _user(db, t2)
    first = date.today().replace(day=1)
    prev_end = first - timedelta(days=1)
    _emp(db, t, "上月入职", hire=prev_end - timedelta(days=5))
    _emp(db, t2, "别租户员工", hire=prev_end - timedelta(days=5))

    r = client.post("/api/report-defs/hr_movements/run", headers=_hdr(u),
                    json={"params": {}})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["hires"] == 1
    r2 = client.post("/api/report-defs/hr_movements/run", headers=_hdr(u2),
                     json={"params": {}})
    assert r2.json()["data"]["hires"] == 1  # 只看到自己人
    defs = client.get("/api/report-defs", headers=_hdr(u)).json()
    assert any(d["kind"] == "hr_movements" for d in defs)


# ---------------------------------------------------------------------------
# H2:组织调整 preview + apply
# ---------------------------------------------------------------------------

def test_org_changes_preview_and_apply(db, client):
    t = _tenant(db)
    u = _user(db, t)
    tech = _org(db, t, "技术部")
    e1 = _emp(db, t, "张三", tech)
    e2 = _emp(db, t, "李四", tech)

    changes = [
        {"type": "create_org_unit", "name": "算法组", "parent": "技术部"},
        {"type": "transfer_employee", "employee": "张三",
         "to_org": "算法组", "new_manager": "李四"},
        {"type": "set_manager", "employee": "李四", "manager": None},
        {"type": "rename_org_unit", "org": "技术部", "new_name": "技术平台部"},
    ]
    r = client.post("/api/org/preview-changes", headers=_hdr(u),
                    json={"changes": changes})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body["errors"]
    assert len(body["changes"]) == 4
    assert "新建组织 算法组" in body["changes"][0]["describe"]

    r = client.post("/api/org/apply-changes", headers=_hdr(u),
                    json={"changes": changes})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == 4
    db.refresh(e1)
    db.refresh(tech)
    new_org = db.scalars(
        __import__("sqlalchemy").select(OrgUnit).where(
            OrgUnit.tenant_id == t.id, OrgUnit.name == "算法组")).first()
    assert new_org is not None and new_org.parent_id == tech.id
    assert e1.org_unit_id == new_org.id
    assert e1.reports_to_id == e2.id
    assert tech.name == "技术平台部"
    ev = db.scalars(
        __import__("sqlalchemy").select(LifecycleEvent).where(
            LifecycleEvent.employee_id == e1.id,
            LifecycleEvent.kind == "transfer")).first()
    assert ev is not None and ev.payload["to_org_unit_id"] == new_org.id


def test_org_changes_errors_block_apply(db, client):
    t = _tenant(db)
    u = _user(db, t)
    changes = [
        {"type": "transfer_employee", "employee": "不存在的人",
         "to_org": "技术部"},
        {"type": "move_org_unit", "org": "不存在部门", "new_parent": "技术部"},
        {"type": "bogus_type"},
    ]
    r = client.post("/api/org/preview-changes", headers=_hdr(u),
                    json={"changes": changes})
    body = r.json()
    assert body["ok"] is False and len(body["errors"]) >= 3
    r = client.post("/api/org/apply-changes", headers=_hdr(u),
                    json={"changes": changes})
    assert r.status_code == 422


def test_move_org_cycle_blocked(db, client):
    t = _tenant(db)
    u = _user(db, t)
    a = _org(db, t, "A")
    b = _org(db, t, "B", parent=a)
    r = client.post("/api/org/apply-changes", headers=_hdr(u),
                    json={"changes": [
                        {"type": "move_org_unit", "org": "A", "new_parent": "B"}]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# H3:排班生成 / 缺口 / 替班
# ---------------------------------------------------------------------------

def _shift(db, t, name, required=1, start=540, end=1020) -> Shift:
    s = Shift(tenant_id=t.id, name=name, required=required,
              start_min=start, end_min=end)
    db.add(s)
    db.commit()
    return s


def test_generate_schedule_skips_leave_and_fills(db):
    t = _tenant(db)
    s = _shift(db, t, "早班", required=2)
    org = _org(db, t, "门店")
    emps = [_emp(db, t, f"店员{i}", org) for i in range(4)]
    lt = LeaveType(tenant_id=t.id, name="年假", annual_days=10)
    db.add(lt)
    db.commit()
    # 店员0 全月请假 → 永远不被排
    d_start = date(2026, 3, 1)
    d_end = date(2026, 3, 31)
    db.add(LeaveApplication(
        tenant_id=t.id, employee_id=emps[0].id, leave_type_id=lt.id,
        start_date=d_start, end_date=d_end, days=23, status="approved"))
    db.commit()

    out = generate_schedule(db, t.id, start=d_start, end=d_end, dry_run=True)
    assert out["planned"] > 0
    assert all(p["employee_id"] != emps[0].id for p in out["plan"])
    # 每个工作日早班排满 2 人(3 个可用人手轮着来)
    by_day = {}
    for p in out["plan"]:
        by_day[p["day"]] = by_day.get(p["day"], 0) + 1
    assert all(v == 2 for v in by_day.values())
    # dry_run 不落库
    assert db.scalars(
        __import__("sqlalchemy").select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == t.id)).first() is None

    out2 = generate_schedule(db, t.id, start=d_start, end=d_end, dry_run=False)
    assert out2["planned"] == out["planned"]
    # 重跑生成器:槽位已满 → planned=0(幂等不重复排)
    out3 = generate_schedule(db, t.id, start=d_start, end=d_end, dry_run=True)
    assert out3["planned"] == 0


def test_find_gaps_and_auto_cover(db):
    t = _tenant(db)
    s = _shift(db, t, "晚班", required=1)
    e1 = _emp(db, t, "请假人")
    e2 = _emp(db, t, "顶班人")
    day = date(2026, 3, 5)  # 周四工作日
    db.add(ShiftAssignment(tenant_id=t.id, employee_id=e1.id, shift_id=s.id,
                           day=day, source="auto"))
    lt = LeaveType(tenant_id=t.id, name="事假", annual_days=5)
    db.add(lt)
    db.commit()
    db.add(LeaveApplication(
        tenant_id=t.id, employee_id=e1.id, leave_type_id=lt.id,
        start_date=day, end_date=day, days=1, status="approved"))
    db.commit()

    gaps = find_gaps(db, t.id, start=day, end=day)
    assert len(gaps) == 1
    assert gaps[0]["on_leave"][0]["employee_name"] == "请假人"
    assert gaps[0]["covered"] == 0

    out = auto_cover(db, t.id, start=day, end=day)
    assert len(out["covered"]) == 1
    assert out["covered"][0]["to"] == "顶班人"
    a = db.scalars(
        __import__("sqlalchemy").select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == t.id)).first()
    assert a.employee_id == e2.id and a.source == "cover"
    # 换班后再查缺口 → 已覆盖
    assert find_gaps(db, t.id, start=day, end=day) == []


def test_shifts_api_flow(db, client):
    t = _tenant(db)
    u = _user(db, t)
    _emp(db, t, "员工甲")
    _emp(db, t, "员工乙")
    r = client.post("/api/shifts", headers=_hdr(u),
                    json={"name": "早班", "start_min": 540, "end_min": 1020,
                          "required": 1})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    r = client.post("/api/shifts", headers=_hdr(u), json={"name": "早班"})
    assert r.status_code == 409  # 同名冲突
    r = client.post("/api/shifts/generate", headers=_hdr(u),
                    json={"month": "2026-03", "dry_run": False})
    assert r.status_code == 200
    assert r.json()["planned"] > 0
    r = client.get("/api/shifts/assignments?month=2026-03", headers=_hdr(u))
    assert r.json()["total"] > 0
    # 跨租户隔离
    t2 = _tenant(db)
    u2 = _user(db, t2)
    r = client.get("/api/shifts/assignments?month=2026-03", headers=_hdr(u2))
    assert r.json()["total"] == 0
