# -*- coding: utf-8 -*-
"""E3 审批中心测试。

覆盖:
- 策略 CRUD + 静态校验(未知字段/空 steps/坏 resolver/非 admin 禁止)
- route_request:命中策略建实例+步骤,未命中走旧一级审批(向后兼容)
- 多级链:manager → role(hr) 逐步推进,末步批准才落域副作用(台账)
- 中途驳回 → 整单 rejected + 域状态同步 + 申请人通知
- 自批禁止、步骤越权(非当前步解析人 403)
- 代理人(delegate_user_id)并入可批集合
- 统一收件箱 pending_for:step 项只给解析人,legacy 项给权限角色
- 员工撤回 pending 申请 → 实例级联 cancelled
- 实例唯一(重复 route_request 幂等)
"""
from __future__ import annotations

import os
import uuid
from datetime import date

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-e3")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-e3"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    ApprovalInstance, ApprovalPolicy, ApprovalStep, Base, Employee,
    EmployeeRequest, LeaveApplication, LeaveLedgerEntry, LeaveType,
    Notification, Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.approvals import instance_for, route_request  # noqa: E402


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
    t = Tenant(name=f"e3-{uuid.uuid4().hex[:8]}")
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


def _emp(db, t, name, user=None, mgr=None) -> Employee:
    e = Employee(
        tenant_id=t.id, name=name, employee_no=uuid.uuid4().hex[:8],
        user_id=user.id if user else None,
        reports_to_id=mgr.id if mgr else None,
        hire_date=date(2025, 1, 1), status="active",
    )
    db.add(e)
    db.commit()
    return e


def _leave_type(db, t) -> LeaveType:
    lt = LeaveType(tenant_id=t.id, name="年假", annual_days=10)
    db.add(lt)
    db.commit()
    return lt


def _mk_policy(client, admin, **over):
    body = {
        "name": "两级审批", "request_type": "leave", "priority": 0,
        "conditions": [],
        "steps": [
            {"resolver": "manager", "label": "直属上级"},
            {"resolver": "role", "key": "hr", "label": "HR 复核"},
        ],
    }
    body.update(over)
    return client.post("/api/approvals/policies", headers=_hdr(admin), json=body)


def _mk_leave(client, emp_user, lt, start="2026-04-13", end="2026-04-14"):
    return client.post(
        "/api/me/leave-applications", headers=_hdr(emp_user),
        json={"leave_type_id": lt.id, "start_date": start, "end_date": end})


# ---------------------------------------------------------------------------
# 策略 CRUD + 校验
# ---------------------------------------------------------------------------

def test_policy_crud_and_validation(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    hr = _user(db, t, "hr")

    # 非法:未知条件字段 / 空 steps / 坏 resolver
    r = _mk_policy(client, admin,
                   conditions=[{"field": "salary", "op": "eq", "value": 1}])
    assert r.status_code == 422
    r = _mk_policy(client, admin, steps=[])
    assert r.status_code == 422
    r = _mk_policy(client, admin, steps=[{"resolver": "boss"}])
    assert r.status_code == 422
    # role/user 缺 key
    r = _mk_policy(client, admin, steps=[{"resolver": "role"}])
    assert r.status_code == 422
    # 非 admin 不能建
    r = _mk_policy(client, hr)
    assert r.status_code == 403

    r = _mk_policy(client, admin)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert r.json()["steps"][0]["resolver"] == "manager"

    lst = client.get("/api/approvals/policies", headers=_hdr(admin)).json()
    assert any(p["id"] == pid for p in lst["items"])
    assert "leave" in lst["request_types"]
    assert "days" in lst["condition_fields"]["leave"]

    r = client.put(f"/api/approvals/policies/{pid}", headers=_hdr(admin),
                   json={"priority": 5, "active": False})
    assert r.status_code == 200
    assert r.json()["priority"] == 5 and r.json()["active"] is False

    r = client.delete(f"/api/approvals/policies/{pid}", headers=_hdr(admin))
    assert r.status_code == 204
    assert not any(p["id"] == pid for p in
                   client.get("/api/approvals/policies",
                              headers=_hdr(admin)).json()["items"])


def test_policy_condition_routing(db, client):
    """conditions 命中才接管:days>3 走策略,<=3 走旧一级。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    e = _emp(db, t, "张三", user=emp_user)
    lt = _leave_type(db, t)
    _mk_policy(client, admin,
               conditions=[{"field": "days", "op": "gt", "value": 3}])

    # 2 天 → 不命中,无实例
    r = _mk_leave(client, emp_user, lt, "2026-04-13", "2026-04-14")
    assert r.status_code == 201, r.text
    assert instance_for(db, t.id, "leave", r.json()["id"]) is None
    # 5 天(周一~周五)→ 命中
    r = _mk_leave(client, emp_user, lt, "2026-04-20", "2026-04-24")
    assert r.status_code == 201, r.text
    inst = instance_for(db, t.id, "leave", r.json()["id"])
    assert inst is not None and inst.status == "pending"
    steps = db.scalars(select(ApprovalStep).where(
        ApprovalStep.instance_id == inst.id).order_by(ApprovalStep.idx)).all()
    assert [s.resolver for s in steps] == ["manager", "role"]
    # 幂等:再 route 一次不会建第二个实例
    obj = db.get(LeaveApplication, inst.request_id)
    assert route_request(db, t.id, "leave", obj) is None


# ---------------------------------------------------------------------------
# 多级审批链
# ---------------------------------------------------------------------------

def _setup_chain(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    mgr_user = _user(db, t, "hiring_manager")
    hr = _user(db, t, "hr")
    mgr_emp = _emp(db, t, "王经理", user=mgr_user)
    e = _emp(db, t, "张三", user=emp_user, mgr=mgr_emp)
    lt = _leave_type(db, t)
    _mk_policy(client, admin)
    return t, admin, emp_user, mgr_user, hr, e, lt


def test_two_step_chain_end_to_end(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    r = _mk_leave(client, emp_user, lt)
    assert r.status_code == 201, r.text
    app_id = r.json()["id"]

    # 首步通知发给经理
    n = db.scalars(select(Notification).where(
        Notification.user_id == mgr_user.id,
        Notification.kind == "approval")).first()
    assert n is not None and "请假" in (n.body or "")

    # 收件箱:经理可见 step 项;申请人本人不可见;hr 此时看不到(第 2 步)
    box = client.get("/api/approvals/inbox", headers=_hdr(mgr_user)).json()
    assert any(i["request_id"] == app_id and i["mode"] == "step"
               and i["step_idx"] == 0 and i["step_count"] == 2
               for i in box["items"])
    assert not client.get(
        "/api/approvals/inbox", headers=_hdr(emp_user)).json()["items"]
    assert not any(i["request_id"] == app_id for i in client.get(
        "/api/approvals/inbox", headers=_hdr(hr)).json()["items"])

    # hr 越步审批 → 403;申请人自批 → 403
    r = client.post("/api/approvals/decide", headers=_hdr(hr),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 403
    r = client.post("/api/approvals/decide", headers=_hdr(emp_user),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 403

    # 经理批第 1 步 → 实例仍 pending,推进到第 2 步,单据本身还是 pending
    r = client.post("/api/approvals/decide", headers=_hdr(mgr_user),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved", "note": "ok"})
    assert r.status_code == 200, r.text
    assert r.json()["instance_status"] == "pending"
    db.refresh(a := db.get(LeaveApplication, app_id))
    assert a.status == "pending"
    inst = instance_for(db, t.id, "leave", app_id)
    assert inst.current_step_idx == 1

    # 第 2 步轮到 hr;经理已不可再批
    box = client.get("/api/approvals/inbox", headers=_hdr(hr)).json()
    assert any(i["request_id"] == app_id and i["step_idx"] == 1
               for i in box["items"])
    r = client.post("/api/approvals/decide", headers=_hdr(mgr_user),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 403

    # hr 批末步 → 实例 approved + 域副作用:单据 approved + 台账扣减
    r = client.post("/api/approvals/decide", headers=_hdr(hr),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 200
    assert r.json()["instance_status"] == "approved"
    db.refresh(a)
    assert a.status == "approved" and a.approver_id == hr.id
    led = db.scalars(select(LeaveLedgerEntry).where(
        LeaveLedgerEntry.ref_id == app_id,
        LeaveLedgerEntry.ref_type == "application")).first()
    assert led is not None and led.delta_days == -a.days
    # 申请人收到通过通知
    n = db.scalars(select(Notification).where(
        Notification.user_id == emp_user.id,
        Notification.title == "审批已通过")).first()
    assert n is not None
    # 历史记录
    hist = client.get("/api/approvals/history", headers=_hdr(hr)).json()
    assert any(h["request_id"] == app_id and h["status"] == "approved"
               for h in hist["items"])
    # 再批 → 400 已结束
    r = client.post("/api/approvals/decide", headers=_hdr(admin),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 400


def test_mid_chain_reject(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    r = client.post("/api/approvals/decide", headers=_hdr(mgr_user),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "rejected", "note": "人手紧张"})
    assert r.status_code == 200
    inst = instance_for(db, t.id, "leave", app_id)
    assert inst.status == "rejected"
    a = db.get(LeaveApplication, app_id)
    assert a.status == "rejected" and a.approver_id == mgr_user.id
    # 驳回不写台账
    assert db.scalars(select(LeaveLedgerEntry).where(
        LeaveLedgerEntry.ref_id == app_id)).first() is None
    n = db.scalars(select(Notification).where(
        Notification.user_id == emp_user.id,
        Notification.title == "审批被驳回")).first()
    assert n is not None and "人手紧张" in (n.body or "")
    # 剩余步骤 pending 保留(不再可批)
    r = client.post("/api/approvals/decide", headers=_hdr(hr),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code == 400


def test_delegate_can_decide(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    delegate = _user(db, t, "viewer")  # 无审批权限角色也能被指定为代理人
    # 经理设置代理人
    r = client.patch("/api/approvals/delegate", headers=_hdr(mgr_user),
                     json={"delegate_user_id": delegate.id})
    assert r.status_code == 200, r.text
    # 不能代理给自己 / 跨租户
    r = client.patch("/api/approvals/delegate", headers=_hdr(mgr_user),
                     json={"delegate_user_id": mgr_user.id})
    assert r.status_code == 400

    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    # 代理人收件箱可见
    box = client.get("/api/approvals/inbox", headers=_hdr(delegate)).json()
    assert any(i["request_id"] == app_id for i in box["items"])
    # 代理人可批
    r = client.post("/api/approvals/decide", headers=_hdr(delegate),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved", "note": "代批"})
    assert r.status_code == 200, r.text
    step = db.scalars(select(ApprovalStep).where(
        ApprovalStep.instance_id == instance_for(db, t.id, "leave", app_id).id,
        ApprovalStep.idx == 0)).first()
    assert step.decided_by_id == delegate.id


def test_self_approval_blocked_via_manager_resolver(db, client):
    """申请人自己就是经理(自己管自己)时,manager 解析出本人 → 被剔除,
    只剩兜底逻辑外无人能批。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    # 张三 reports_to 指向自己
    e = _emp(db, t, "张三", user=emp_user)
    e.reports_to_id = e.id
    db.commit()
    lt = _leave_type(db, t)
    _mk_policy(client, admin)  # steps: manager → role hr

    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    inst = instance_for(db, t.id, "leave", app_id)
    step = db.scalars(select(ApprovalStep).where(
        ApprovalStep.instance_id == inst.id, ApprovalStep.idx == 0)).first()
    from app.services.approvals import resolve_step_users
    assert emp_user.id not in resolve_step_users(db, inst, step)


# ---------------------------------------------------------------------------
# 旧路径兼容 + 撤回
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_without_policy(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    _emp(db, t, "张三", user=emp_user)
    lt = _leave_type(db, t)
    # 无策略 → 不建实例,旧 approve 端点照旧
    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    assert instance_for(db, t.id, "leave", app_id) is None
    # 旧端点可批
    r = client.post(f"/api/leave/applications/{app_id}/approve",
                    headers=_hdr(admin), json={})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "approved"


def test_unified_decide_legacy_and_inbox(db, client):
    """无策略的单据走统一收件箱 legacy 模式 + 统一 decide。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    e = _emp(db, t, "张三", user=emp_user)
    lt = _leave_type(db, t)
    app_id = _mk_leave(client, emp_user, lt).json()["id"]

    box = client.get("/api/approvals/inbox", headers=_hdr(admin)).json()
    assert any(i["request_id"] == app_id and i["mode"] == "legacy"
               for i in box["items"])
    # 员工本人收件箱为空(自批排除)
    assert not client.get(
        "/api/approvals/inbox", headers=_hdr(emp_user)).json()["items"]

    r = client.post("/api/approvals/decide", headers=_hdr(admin),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved", "note": "统一入口"})
    assert r.status_code == 200 and r.json()["mode"] == "legacy"
    assert db.get(LeaveApplication, app_id).status == "approved"
    # 未知类型 → 422;不存在单据 → 404
    r = client.post("/api/approvals/decide", headers=_hdr(admin),
                    json={"request_type": "bogus", "request_id": "x",
                          "action": "approved"})
    assert r.status_code == 422


def test_cancel_pending_closes_instance(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    inst = instance_for(db, t.id, "leave", app_id)
    assert inst.status == "pending"
    r = client.post(f"/api/me/leave-applications/{app_id}/cancel",
                    headers=_hdr(emp_user))
    assert r.status_code == 200, r.text
    db.refresh(inst)
    assert inst.status == "cancelled"
    steps = db.scalars(select(ApprovalStep).where(
        ApprovalStep.instance_id == inst.id)).all()
    assert all(s.status == "cancelled" for s in steps)
    # 收件箱不再出现
    box = client.get("/api/approvals/inbox", headers=_hdr(mgr_user)).json()
    assert not any(i["request_id"] == app_id for i in box["items"])


def test_employee_request_routed_and_decided(db, client):
    """employee_request 类型也走引擎(me/requests 创建 → 策略 → decide)。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    hr = _user(db, t, "hr")
    _emp(db, t, "张三", user=emp_user)
    r = client.post("/api/approvals/policies", headers=_hdr(admin), json={
        "name": "申请单 HR 批", "request_type": "employee_request",
        "conditions": [], "steps": [{"resolver": "role", "key": "hr"}]})
    assert r.status_code == 201, r.text

    r = client.post("/api/me/requests", headers=_hdr(emp_user),
                    json={"kind": "general", "subject": "开具在职证明"})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    inst = instance_for(db, t.id, "employee_request", rid)
    assert inst is not None

    box = client.get("/api/approvals/inbox", headers=_hdr(hr)).json()
    assert any(i["request_id"] == rid for i in box["items"])
    r = client.post("/api/approvals/decide", headers=_hdr(hr),
                    json={"request_type": "employee_request",
                          "request_id": rid, "action": "approved"})
    assert r.status_code == 200
    assert db.get(EmployeeRequest, rid).status == "approved"


def test_instance_detail_endpoint(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    r = client.get(f"/api/approvals/instances/leave/{app_id}",
                   headers=_hdr(admin))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending" and body["current_step_idx"] == 0
    assert len(body["steps"]) == 2
    assert body["steps"][0]["resolver"] == "manager"
    # 查一个不存在的单据 → 404
    r = client.get("/api/approvals/instances/leave/nonexistent",
                   headers=_hdr(admin))
    assert r.status_code == 404


def test_tenant_isolation(db, client):
    t, admin, emp_user, mgr_user, hr, e, lt = _setup_chain(db, client)
    app_id = _mk_leave(client, emp_user, lt).json()["id"]
    t2 = _tenant(db)
    admin2 = _user(db, t2, "admin")
    # 跨租户 decide → instance_for 按 tenant 过滤,落到 legacy 404
    r = client.post("/api/approvals/decide", headers=_hdr(admin2),
                    json={"request_type": "leave", "request_id": app_id,
                          "action": "approved"})
    assert r.status_code in (403, 404)
    # 跨租户收件箱不含该单
    box = client.get("/api/approvals/inbox", headers=_hdr(admin2)).json()
    assert not any(i["request_id"] == app_id for i in box["items"])
