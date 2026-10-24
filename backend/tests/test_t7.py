"""T7 事件驱动自动化 + 考勤深化测试。

覆盖:条件求值/收件人解析/watch_fields/规则 CRUD/run 留痕、
ip_allowed/apply_flags/update_flag 重算、policy API、flags API 三档 scope、
考勤规则页数据、late_repeat 评估器、租户与权限隔离。
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, time, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t7")

from zoneinfo import ZoneInfo  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t7"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.attendance_policy import (  # noqa: E402
    evaluate_flags,
    get_policy,
    ip_allowed,
)
from app.services.automation import (  # noqa: E402
    dispatch,
    eval_conditions,
    render_tpl,
    suspend_automation,
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


def _tenant(db) -> Tenant:
    t = Tenant(name=f"t7-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
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


def _emp(db, t: Tenant, **kw) -> m.Employee:
    e = m.Employee(
        tenant_id=t.id,
        employee_no=kw.pop("employee_no", f"E{uuid.uuid4().hex[:6]}"),
        name=kw.pop("name", f"员工{uuid.uuid4().hex[:4]}"),
        **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


@pytest.fixture
def client():
    return TestClient(app)


def _calendar(db, t: Tenant) -> m.WorkCalendar:
    cal = m.WorkCalendar(
        tenant_id=t.id,
        workdays=[1, 2, 3, 4, 5],
        workday_hours=8.0,
        work_start=time(9, 0),
        work_end=time(18, 0),
    )
    db.add(cal)
    db.commit()
    return cal


def _policy(db, t: Tenant, **kw) -> m.AttendancePolicy:
    p = get_policy(db, t.id)
    for k, v in kw.items():
        setattr(p, k, v)
    db.commit()
    return p


def _attendance(db, t: Tenant, e: m.Employee, day: date, **kw) -> m.Attendance:
    a = m.Attendance(
        tenant_id=t.id,
        employee_id=e.id,
        day=day,
        status=kw.pop("status", "present"),
        **kw,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ---------- 自动化:条件求值 ----------


def test_eval_conditions_ops():
    snap = {"status": "approved", "days": 3, "name": "张三", "reason": "年假"}
    assert eval_conditions([], snap) is True
    assert eval_conditions([{"field": "status", "op": "eq", "value": "approved"}], snap)
    assert not eval_conditions(
        [{"field": "status", "op": "eq", "value": "pending"}], snap
    )
    assert eval_conditions([{"field": "status", "op": "ne", "value": "pending"}], snap)
    assert eval_conditions([{"field": "days", "op": "gt", "value": "2"}], snap)
    assert eval_conditions([{"field": "days", "op": "ge", "value": "3"}], snap)
    assert not eval_conditions([{"field": "days", "op": "lt", "value": "3"}], snap)
    # 数字比较遇非数字 → false(不炸)
    assert not eval_conditions([{"field": "name", "op": "gt", "value": "1"}], snap)
    assert eval_conditions([{"field": "name", "op": "contains", "value": "张"}], snap)
    assert eval_conditions(
        [{"field": "status", "op": "in", "value": ["approved", "pending"]}], snap
    )
    # in 的逗号字符串形式
    assert eval_conditions(
        [{"field": "status", "op": "in", "value": "pending, approved"}], snap
    )
    assert eval_conditions([{"field": "reason", "op": "exists"}], snap)
    assert not eval_conditions([{"field": "missing", "op": "exists"}], snap)
    # eq 宽松:数字 "3" == 3
    assert eval_conditions([{"field": "days", "op": "eq", "value": "3"}], snap)
    # 多条件 AND
    assert not eval_conditions(
        [
            {"field": "status", "op": "eq", "value": "approved"},
            {"field": "days", "op": "gt", "value": "9"},
        ],
        snap,
    )


def test_render_tpl():
    u = m.User(email="hr@x.com")
    snap = {"name": "张三", "status": "approved", "days": 2}
    assert render_tpl("{{name}}的请假已{{status}}", snap, u) == "张三的请假已approved"
    # 未知键保留原占位(便于发现模板笔误);{{actor}} 替换为操作者邮箱
    assert render_tpl("{{missing}}|{{actor}}", snap, u) == "{{missing}}|hr@x.com"
    # 值为 None 的已知键 → 空串
    assert render_tpl("[{{status}}]", {"status": None}, u) == "[]"
    assert render_tpl(None, snap, u) == ""


# ---------- 自动化:dispatch 端到端 ----------


def _rule(db, t: Tenant, **kw) -> m.AutomationRule:
    r = m.AutomationRule(
        tenant_id=t.id,
        name=kw.pop("name", f"rule-{uuid.uuid4().hex[:4]}"),
        entity_type=kw.pop("entity_type", "employee"),
        trigger=kw.pop("trigger", "update"),
        action=kw.pop("action", "notify"),
        recipients=kw.pop("recipients", [{"kind": "role", "value": "hr"}]),
        title_tpl=kw.pop("title_tpl", "员工{{name}}已更新"),
        **kw,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _log(db, t: Tenant, u: User, entity_id: str, action: str = "update",
         detail: dict | None = None, result: str = "success") -> m.AuditLog:
    log = m.AuditLog(
        tenant_id=t.id, actor_id=u.id, actor_email=u.email,
        entity_type="employee", entity_id=entity_id, action=action,
        detail=detail or {}, result=result, via="web",
    )
    db.add(log)
    db.flush()  # event_key 需要 log.id
    return log


def _disp(db, log, actor) -> int:
    """dispatch + flush — SessionLocal 全局 autoflush=False,先落盘再断言查询。"""
    n = dispatch(db, log=log, actor=actor)
    db.flush()
    return n


def test_dispatch_notify_e2e(db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    actor = _user(db, t, role="admin")
    e = _emp(db, t, name="张三", status="active")
    r = _rule(
        db, t,
        conditions=[{"field": "status", "op": "eq", "value": "active"}],
        title_tpl="员工{{name}}已{{status}}",
        body_tpl="工号 {{employee_no}}",
    )
    log = _log(db, t, actor, e.id)
    assert _disp(db, log=log, actor=actor) == 1
    run = db.scalar(
        select(m.AutomationRun).where(m.AutomationRun.rule_id == r.id)
    )
    assert run.status == "ok" and run.event_key == log.id
    assert run.recipients == [hr.id]
    assert run.entity_id == e.id
    notif = db.scalar(
        select(m.Notification).where(
            m.Notification.user_id == hr.id, m.Notification.kind == "automation"
        )
    )
    assert notif.title == "员工张三已active"
    assert notif.body.startswith("工号")
    assert notif.link == f"/employees/{e.id}"
    db.refresh(r)
    assert r.last_fired_at is not None
    # 幂等:同一 audit 事件重放 → run 已存在,不再发
    assert _disp(db, log=log, actor=actor) == 0


def test_dispatch_condition_fail_and_no_recipients(db):
    t = _tenant(db)
    actor = _user(db, t)
    e = _emp(db, t, name="李四", status="active")
    _rule(
        db, t,
        conditions=[{"field": "status", "op": "eq", "value": "terminated"}],
    )
    # 条件不满足 → 0 且不留 run
    assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 0
    assert db.scalars(
        select(m.AutomationRun).where(m.AutomationRun.tenant_id == t.id)
    ).all() == []

    # 收件人解析为空 → run 记 skipped_no_recipients,但不计入触发数
    _rule(db, t, name="r2", trigger="update",
          conditions=[], recipients=[{"kind": "role", "value": "nonexistent_role"}])
    assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 0
    run = db.scalar(
        select(m.AutomationRun).where(
            m.AutomationRun.tenant_id == t.id,
            m.AutomationRun.status == "skipped_no_recipients",
        )
    )
    assert run is not None


def test_dispatch_record_manager_and_actor(db):
    t = _tenant(db)
    mgr_u = _user(db, t, role="employee")
    mgr = _emp(db, t, name="主管", user_id=mgr_u.id)
    e = _emp(db, t, name="下属", reports_to_id=mgr.id, status="active")
    actor = _user(db, t)
    _rule(db, t, recipients=[{"kind": "record_manager"}])
    assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 1
    assert db.scalar(
        select(m.Notification).where(m.Notification.user_id == mgr_u.id)
    ) is not None
    # actor 收件人
    _rule(db, t, name="r2", trigger="delete", recipients=[{"kind": "actor"}])
    assert _disp(db, log=_log(db, t, actor, e.id, "delete"), actor=actor) == 1
    assert db.scalar(
        select(m.Notification).where(
            m.Notification.user_id == actor.id, m.Notification.kind == "automation"
        )
    ) is not None


def test_dispatch_watch_fields_suspend_and_inactive(db):
    t = _tenant(db)
    _user(db, t, role="hr")
    actor = _user(db, t)
    e = _emp(db, t, name="王五", status="active")
    _rule(db, t, watch_fields=["status"])
    # detail.fields 不含 status → 跳过
    log = _log(db, t, actor, e.id, detail={"fields": {"note": {"old": "a"}}})
    assert _disp(db, log=log, actor=actor) == 0
    # 含 status → 触发
    log = _log(db, t, actor, e.id, detail={"fields": {"status": {"old": "x"}}})
    assert _disp(db, log=log, actor=actor) == 1
    # fields 非 dict(老调用方没传)→ 按全变更处理,不跳过
    log = _log(db, t, actor, e.id, detail={})
    assert _disp(db, log=log, actor=actor) == 1
    # suspend 上下文内不触发
    with suspend_automation(db):
        assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 0
    # result != success 不触发
    assert _disp(
        db, log=_log(db, t, actor, e.id, result="failure"), actor=actor
    ) == 0
    # 停用后不触发
    rule = db.scalar(select(m.AutomationRule).where(m.AutomationRule.tenant_id == t.id))
    rule.is_active = False
    db.commit()
    assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 0


def test_dispatch_email_action_queues(db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    actor = _user(db, t)
    e = _emp(db, t, name="邮件员工", status="active")
    _rule(db, t, name="mail-rule", action="email")
    assert _disp(db, log=_log(db, t, actor, e.id), actor=actor) == 1
    run = db.scalar(
        select(m.AutomationRun).where(m.AutomationRun.tenant_id == t.id)
    )
    assert run.status == "queued_email"
    assert run.payload["to"] == [hr.email]
    # run.id 排进 after_commit 队列(不落 celery,只验证挂勾登记)
    assert run.id in db.info.get("automation_email_runs", [])
    db.rollback()  # after_commit 钩子会在真 commit 时派发 celery


def test_dispatch_trigger_alias_and_unregistered(db):
    t = _tenant(db)
    _user(db, t, role="hr")
    actor = _user(db, t)
    e = _emp(db, t, name="别名", status="active")
    _rule(db, t, trigger="create")
    # import 动作命中 create 别名组
    assert _disp(db, log=_log(db, t, actor, e.id, "import"), actor=actor) == 1
    # 未注册实体不触发
    log = _log(db, t, actor, e.id, "create")
    log.entity_type = "audit_log"
    assert _disp(db, log=log, actor=actor) == 0


# ---------- 自动化 API ----------


def test_automation_api_crud_and_perms(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    viewer = _user(db, t, role="viewer")

    r = client.get("/api/automations/schema", headers=_hdr(admin))
    assert r.status_code == 200
    assert "leave_application" in r.json()["entities"]

    r = client.post(
        "/api/automations",
        headers=_hdr(admin),
        json={
            "name": "批准通知",
            "entity_type": "leave_application",
            "trigger": "approve",
            "action": "notify",
            "recipients": [{"kind": "role", "value": "hr"}],
            "title_tpl": "已{{status}}",
        },
    )
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert r.json()["entity_label"] == "请假申请"

    # 非法 entity → 400
    r = client.post(
        "/api/automations",
        headers=_hdr(admin),
        json={
            "name": "x",
            "entity_type": "passwords",
            "trigger": "create",
            "action": "notify",
            "recipients": [],
            "title_tpl": "x",
        },
    )
    assert r.status_code == 400

    r = client.patch(
        f"/api/automations/{rid}", headers=_hdr(admin), json={"is_active": False}
    )
    assert r.status_code == 200 and r.json()["is_active"] is False

    # 直写一条 run → runs 端点能取回
    db.add(
        m.AutomationRun(
            tenant_id=t.id,
            rule_id=rid,
            event_key="ev1",
            entity_type="leave_application",
            entity_id="x",
            trigger="approve",
            action="notify",
            status="ok",
        )
    )
    db.commit()
    r = client.get(f"/api/automations/{rid}/runs", headers=_hdr(admin))
    assert r.status_code == 200 and r.json()["total"] == 1

    # viewer 全被拒
    assert client.get("/api/automations", headers=_hdr(viewer)).status_code == 403
    assert client.get(
        f"/api/automations/{rid}/runs", headers=_hdr(viewer)
    ).status_code == 403
    # 跨租户 404
    t2 = _tenant(db)
    u2 = _user(db, t2)
    assert client.patch(
        f"/api/automations/{rid}", headers=_hdr(u2), json={"is_active": True}
    ).status_code == 404
    # 删除
    r = client.delete(f"/api/automations/{rid}", headers=_hdr(admin))
    assert r.status_code == 204


# ---------- 考勤:IP 白名单 ----------


def _pol(*, enabled: bool = True, ips: list | None = None) -> m.AttendancePolicy:
    return m.AttendancePolicy(
        tenant_id="x", ip_check_enabled=enabled, allowed_ips=ips or []
    )


def test_ip_allowed():
    assert ip_allowed(_pol(enabled=False, ips=["10.0.0.1"]), "1.2.3.4") is True
    assert ip_allowed(_pol(ips=[]), "1.2.3.4") is True
    assert ip_allowed(_pol(ips=["10.0.0.1", "192.168.1.0/24"]), "10.0.0.1") is True
    assert ip_allowed(_pol(ips=["192.168.1.0/24"]), "192.168.1.77") is True
    assert ip_allowed(_pol(ips=["192.168.1.0/24"]), "192.168.2.1") is False
    assert ip_allowed(_pol(ips=["bad-entry"]), "1.2.3.4") is False  # 解析失败不通过
    assert ip_allowed(_pol(ips=["10.0.0.1"]), "garbage-ip") is False
    assert ip_allowed(_pol(ips=["10.0.0.1"]), None) is False


def test_checkin_ip_enforcement(client, db):
    t = _tenant(db)
    eu = _user(db, t, role="employee")
    _emp(db, t, user_id=eu.id)
    _policy(db, t, ip_check_enabled=True, allowed_ips=["10.0.0.1"])
    # TestClient 的 client ip 是 testclient → 不在白名单 → 403
    r = client.post("/api/me/checkins", headers=_hdr(eu), json={"kind": "in"})
    assert r.status_code == 403
    # 关闭后恢复
    _policy(db, t, ip_check_enabled=False)
    r = client.post("/api/me/checkins", headers=_hdr(eu), json={"kind": "in"})
    assert r.status_code == 201


# ---------- 考勤:迟到/早退标记 ----------


def test_evaluate_flags_grace_and_recalc(db):
    t = _tenant(db)
    e = _emp(db, t)
    cal = _calendar(db, t)
    policy = _policy(db, t, grace_in_minutes=10, grace_out_minutes=5)
    day = date(2026, 3, 2)  # 周一

    def L(h: int, mi: int = 0) -> datetime:
        """本地墙钟(h:mi)→ 存库的 UTC-naive 值(app_timezone=Asia/Shanghai)。"""
        return (
            datetime(day.year, day.month, day.day, h, mi, tzinfo=ZoneInfo("Asia/Shanghai"))
            .astimezone(ZoneInfo("UTC"))
            .replace(tzinfo=None)
        )

    def _eval(a: m.Attendance, wd: bool = True):
        evaluate_flags(db, att=a, is_workday=wd, policy=policy, calendar=cal)
        db.flush()

    def _flags(a: m.Attendance):
        return db.scalars(
            select(m.AttendanceFlag).where(m.AttendanceFlag.attendance_id == a.id)
        ).all()

    # 宽限内 → 无 flag
    a = _attendance(
        db, t, e, day,
        first_in=L(9, 8),
        last_out=L(18, 30),
    )
    _eval(a)
    assert _flags(a) == []

    # 超宽限:迟到 20min + 早退 30min
    a.first_in = L(9, 20)
    a.last_out = L(17, 30)
    _eval(a)
    kinds = {f.kind: f.minutes for f in _flags(a)}
    assert kinds == {"late_in": 20, "early_out": 30}

    # 已免计的 flag:事实仍存在时重算 → 行保留 + 分钟数刷新
    late = next(f for f in _flags(a) if f.kind == "late_in")
    late.excused = True
    db.commit()
    a.first_in = L(9, 15)  # 仍迟到 15 > 10
    _eval(a)
    late2 = next(f for f in _flags(a) if f.kind == "late_in")
    assert late2.excused and late2.minutes == 15

    # 事实消失(改回宽限内)→ flag 连同免计态一起清掉,台账与事实一致
    a.first_in = L(9, 5)
    a.last_out = L(18, 0)
    _eval(a)
    assert _flags(a) == []

    # 非工作日不评;关闭追踪后 evaluate 早退(不动既有台账)
    a.first_in = L(10, 0)
    _eval(a, wd=False)
    assert _flags(a) == []
    a.first_in = L(9, 0)   # 上班准点 → 只出早退
    a.last_out = L(16, 0)
    _eval(a)
    assert {f.kind for f in _flags(a)} == {"early_out"}
    _policy(db, t, track_late_early=False)
    _eval(a)
    assert {f.kind for f in _flags(a)} == {"early_out"}  # 关开关不清存量


def test_flag_excuse_api_and_scopes(client, db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    mgr_u = _user(db, t, role="hiring_manager")
    mgr = _emp(db, t, user_id=mgr_u.id)
    eu = _user(db, t, role="employee")
    e = _emp(db, t, user_id=eu.id, reports_to_id=mgr.id)
    other_eu = _user(db, t, role="employee")
    _emp(db, t, user_id=other_eu.id)  # 不属于 mgr 团队
    viewer = _user(db, t, role="viewer")

    a = _attendance(db, t, e, date(2026, 3, 2),
                   first_in=datetime(2026, 3, 2, 9, 30))
    flag = m.AttendanceFlag(
        tenant_id=t.id, attendance_id=a.id, employee_id=e.id,
        day=a.day, kind="late_in", minutes=30,
    )
    db.add(flag)
    db.commit()
    db.refresh(flag)

    # employee 角色无任何权限位 → 管理端点 403;自己的异常走 /api/me
    assert client.get("/api/attendance/flags", headers=_hdr(eu)).status_code == 403
    r = client.get("/api/me/attendance-flags", headers=_hdr(eu))
    assert r.status_code == 200 and len(r.json()) == 1
    r = client.get("/api/me/attendance-flags", headers=_hdr(other_eu))
    assert r.status_code == 200 and len(r.json()) == 0
    # hiring_manager scope=team 看直属下属
    r = client.get("/api/attendance/flags?scope=team", headers=_hdr(mgr_u))
    assert r.json()["total"] == 1
    # scope=mine(主管自己不是异常员工)→ 0
    r = client.get("/api/attendance/flags?scope=mine", headers=_hdr(mgr_u))
    assert r.json()["total"] == 0
    # hr scope=all
    r = client.get("/api/attendance/flags", headers=_hdr(hr))
    assert r.json()["total"] == 1
    # viewer 403(只有 view_reports)
    assert client.get("/api/attendance/flags", headers=_hdr(viewer)).status_code == 403

    # 员工不能免计(连权限门都过不了)
    assert client.post(
        f"/api/attendance/flags/{flag.id}/excuse", headers=_hdr(eu), json={}
    ).status_code == 403
    # 直属主管可免计下属
    r = client.post(
        f"/api/attendance/flags/{flag.id}/excuse",
        headers=_hdr(mgr_u),
        json={"note": "地铁故障"},
    )
    assert r.status_code == 200
    db.refresh(flag)
    assert flag.excused and flag.excused_note == "地铁故障"
    # HR 撤销
    r = client.post(f"/api/attendance/flags/{flag.id}/unexcuse", headers=_hdr(hr))
    assert r.status_code == 200
    db.refresh(flag)
    assert not flag.excused


def test_policy_put_and_calendar_times(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hr = _user(db, t, role="hr")
    viewer = _user(db, t, role="viewer")

    r = client.get("/api/attendance/policy", headers=_hdr(admin))
    assert r.status_code == 200 and r.json()["grace_in_minutes"] == 0

    r = client.put(
        "/api/attendance/policy",
        headers=_hdr(admin),
        json={
            "grace_in_minutes": 15,
            "grace_out_minutes": 10,
            "track_late_early": True,
            "ip_check_enabled": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["grace_in_minutes"] == 15

    # hr 有 manage_attendance 权限也能改
    r = client.put(
        "/api/attendance/policy", headers=_hdr(hr), json={"grace_in_minutes": 5}
    )
    assert r.status_code == 200
    assert client.put(
        "/api/attendance/policy", headers=_hdr(viewer), json={}
    ).status_code == 403

    # policy 更新写了审计
    audit = db.scalar(
        select(m.AuditLog).where(
            m.AuditLog.tenant_id == t.id,
            m.AuditLog.entity_type == "attendance_policy",
            m.AuditLog.action == "update",
        )
    )
    assert audit is not None

    # 日历上下班车次字段持久化
    r = client.put(
        "/api/work-calendar",
        headers=_hdr(admin),
        json={
            "workdays": [1, 2, 3, 4, 5],
            "workday_hours": 8,
            "work_start": "09:30:00",
            "work_end": "18:30:00",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["work_start"] == "09:30:00"
    assert r.json()["work_end"] == "18:30:00"


def test_monthly_summary(client, db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    e = _emp(db, t, name="月度员工")
    d = date(2026, 3, 3)
    _attendance(db, t, e, d, status="present", work_hours=8.0)
    _attendance(db, t, e, d + timedelta(days=1), status="absent", work_hours=0)
    _attendance(db, t, e, d + timedelta(days=2), status="leave", work_hours=0)
    db.add(
        m.AttendanceFlag(
            tenant_id=t.id,
            attendance_id=db.scalar(
                select(m.Attendance.id).where(
                    m.Attendance.employee_id == e.id, m.Attendance.day == d
                )
            ),
            employee_id=e.id,
            day=d,
            kind="late_in",
            minutes=15,
        )
    )
    db.commit()

    r = client.get("/api/attendance/monthly-summary?month=2026-03", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    row = next(x for x in r.json() if x["employee_id"] == e.id)
    assert row["present"] == 1 and row["absent"] == 1 and row["leave"] == 1
    assert row["work_hours"] == 8.0
    assert row["late_count"] == 1 and row["late_minutes"] == 15
    # 月份格式非法 → 400
    assert client.get(
        "/api/attendance/monthly-summary?month=bad", headers=_hdr(hr)
    ).status_code == 400
    # viewer 有 view_reports 可访问,但数据域按行级收敛(未绑员工 → 空)
    v = _user(db, t, role="viewer")
    r = client.get(
        "/api/attendance/monthly-summary?month=2026-03", headers=_hdr(v)
    )
    assert r.status_code == 200 and r.json() == []


def test_late_repeat_evaluator(db):
    from app.services.watchers import _eval_late_repeat

    t = _tenant(db)
    hr = _user(db, t, role="hr")
    eu = _user(db, t, role="employee")
    e = _emp(db, t, user_id=eu.id, name="迟到大王")
    other = _emp(db, t, name="全勤员工")
    today = date.today()

    def _flag(emp: m.Employee, d: date, **kw):
        a = _attendance(db, t, emp, d, status="present")
        db.add(m.AttendanceFlag(
            tenant_id=t.id, attendance_id=a.id, employee_id=emp.id,
            day=d, kind="late_in", **kw,
        ))

    for i in range(3):
        _flag(e, today - timedelta(days=i), minutes=20)
    # 别的员工:2 条未免计 + 1 条已免计 → 不够 3
    for i in range(2):
        _flag(other, today - timedelta(days=i), minutes=10)
    _flag(other, today - timedelta(days=3), minutes=10, excused=True)
    db.flush()  # autoflush=False,先落盘再评估
    w = m.AiWatcher(
        tenant_id=t.id, user_id=hr.id, kind="late_repeat",
        params={"days": 7, "count": 3},
    )
    hits = _eval_late_repeat(db, w)
    assert len(hits) == 1
    assert "迟到大王" in hits[0]["body"] and "3 次" in hits[0]["body"]
    assert hits[0]["link"] == "/attendance/flags"
    assert e.id in hits[0]["dedupe_key"]
    # 阈值抬高 → 不命中
    w.params = {"days": 7, "count": 5}
    assert _eval_late_repeat(db, w) == []
    # 时间窗收窄到昨天前 → 3 条里有今天的,仍 ≥3;改 count=4 → 空
    w.params = {"days": 0, "count": 4}
    assert _eval_late_repeat(db, w) == []
    # owner 数据域收敛:无关 hiring_manager 看不到该员工
    mgr_u = _user(db, t, role="hiring_manager")
    w2 = m.AiWatcher(
        tenant_id=t.id, user_id=mgr_u.id, kind="late_repeat",
        params={"days": 7, "count": 1},
    )
    assert _eval_late_repeat(db, w2) == []
