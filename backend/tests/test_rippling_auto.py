"""Rippling 自动化:date_field 定时触发 + create_case 动作 + 审批自动裁决。

覆盖:
- run_scheduled:日期到达触发、offset/recurring、幂等去重、terminated 跳过
- create_case 动作:建 HR 工单(case_no/assignee/收件人通知)、run 记账
- API 校验:trigger_kind/schedule/action_config 白名单
- auto_approve:命中提交即批准(实例 auto_approved、步骤 approved、
  领域副作用同人工)、不命中走人工链、条件白名单校验
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-ra")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-ra"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.approvals import instance_for  # noqa: E402
from app.services.automation import run_scheduled  # noqa: E402


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
    t = Tenant(name=f"ra-{uuid.uuid4().hex[:8]}")
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


def _emp(db, t, name, user=None, **kw) -> m.Employee:
    e = m.Employee(
        tenant_id=t.id, name=name, employee_no=uuid.uuid4().hex[:8],
        user_id=user.id if user else None,
        hire_date=kw.pop("hire_date", date(2025, 1, 1)),
        status=kw.pop("status", "active"),
        **kw,
    )
    db.add(e)
    db.commit()
    return e


def _leave_type(db, t) -> m.LeaveType:
    lt = m.LeaveType(tenant_id=t.id, name="年假", annual_days=10)
    db.add(lt)
    db.commit()
    return lt


def _rule(db, t: Tenant, **kw) -> m.AutomationRule:
    r = m.AutomationRule(
        tenant_id=t.id,
        name=kw.pop("name", f"规则{uuid.uuid4().hex[:4]}"),
        entity_type=kw.pop("entity_type", "employee"),
        trigger=kw.pop("trigger", "date_field"),
        title_tpl=kw.pop("title_tpl", "提醒 {{name}}"),
        is_active=True,
        **kw,
    )
    db.add(r)
    db.commit()
    return r


def _runs(db, rule: m.AutomationRule) -> list[m.AutomationRun]:
    """本规则的执行记录 —— 共享测试库可能有其他租户的孤儿规则,
    一律按 rule_id 过滤断言。"""
    db.flush()
    return db.scalars(
        select(m.AutomationRun).where(m.AutomationRun.rule_id == rule.id)
    ).all()


# ---------------------------------------------------------------------------
# date_field 定时触发
# ---------------------------------------------------------------------------


def test_scheduled_fires_on_offset_and_dedupes(db):
    """probation_end -3 天命中:通知 + run 记账;二次调用幂等;terminated 不触发。"""
    t = _tenant(db)
    hr = _user(db, t, "hr")
    today = date(2026, 6, 15)
    hit = _emp(db, t, "转正将到", personal={
        "probation_end": (today + timedelta(days=3)).isoformat()})
    miss = _emp(db, t, "还很远", personal={
        "probation_end": (today + timedelta(days=30)).isoformat()})
    gone = _emp(db, t, "已离职", status="terminated", personal={
        "probation_end": (today + timedelta(days=3)).isoformat()})
    rule = _rule(
        db, t, trigger_kind="date_field",
        schedule={"field": "probation_end", "offset_days": -3},
        action="notify",
        recipients=[{"kind": "role", "value": "hr"}],
        title_tpl="转正提醒:{{name}} 3 天后转正",
    )
    run_scheduled(db, today=today)
    runs = _runs(db, rule)  # SessionLocal autoflush=False
    assert len(runs) == 1
    assert runs[0].status == "ok" and runs[0].entity_id == hit.id
    titles = [n.title for n in db.scalars(select(m.Notification).where(
        m.Notification.user_id == hr.id,
        m.Notification.kind == "automation"))]
    assert any(hit.name in t_ for t_ in titles)
    assert not any(miss.name in t_ for t_ in titles)
    assert not any(gone.name in t_ for t_ in titles)
    # 幂等:同天再扫不再触发(本规则)
    run_scheduled(db, today=today)
    assert len(_runs(db, rule)) == 1
    # 明天也不在窗口
    run_scheduled(db, today=today + timedelta(days=1))
    assert len(_runs(db, rule)) == 1


def test_scheduled_recurring_birthday(db):
    """recurring 生日规则:按今年纪念日匹配,明年同日仍触发。"""
    t = _tenant(db)
    hr = _user(db, t, "hr")
    _emp(db, t, "寿星", personal={"birth_date": "1990-08-20"})
    rule = _rule(
        db, t, trigger_kind="date_field",
        schedule={"field": "birth_date", "offset_days": 0, "recurring": True},
        action="notify", recipients=[{"kind": "role", "value": "hr"}],
        title_tpl="今天是 {{name}} 的生日 🎂",
    )
    run_scheduled(db, today=date(2026, 8, 20))
    assert len(_runs(db, rule)) == 1
    run_scheduled(db, today=date(2026, 8, 21))
    assert len(_runs(db, rule)) == 1
    # 不同年份同一天 → 不同 event_key,仍触发
    run_scheduled(db, today=date(2027, 8, 20))
    assert len(_runs(db, rule)) == 2


def test_scheduled_conditions_filter(db):
    """条件过滤:只有命中条件的员工触发。"""
    t = _tenant(db)
    hr = _user(db, t, "hr")
    ou = m.OrgUnit(tenant_id=t.id, name="研发部")
    db.add(ou)
    db.commit()
    _emp(db, t, "研发转正", org_unit_id=ou.id, personal={
        "probation_end": "2026-06-18"})
    _emp(db, t, "他部转正", personal={"probation_end": "2026-06-18"})
    rule = _rule(
        db, t, trigger_kind="date_field",
        schedule={"field": "probation_end", "offset_days": 0},
        conditions=[{"field": "org_unit_id", "op": "eq", "value": ou.id}],
        action="notify", recipients=[{"kind": "role", "value": "hr"}],
        title_tpl="转正:{{name}}",
    )
    run_scheduled(db, today=date(2026, 6, 18))
    runs = _runs(db, rule)
    assert len(runs) == 1
    n = db.scalar(select(m.Notification).where(
        m.Notification.user_id == hr.id,
        m.Notification.kind == "automation"))
    assert "研发转正" in n.title


# ---------------------------------------------------------------------------
# create_case 动作
# ---------------------------------------------------------------------------


def test_create_case_action(db):
    """合同到期自动建工单:case_no 递增、requester=员工、assignee=hr、收件人获通知。"""
    t = _tenant(db)
    hr = _user(db, t, "hr")
    emp = _emp(db, t, "合同将满", personal={"contract_end": "2026-07-01"})
    rule = _rule(
        db, t, trigger_kind="date_field",
        schedule={"field": "contract_end", "offset_days": -30},
        action="create_case",
        action_config={
            "kind": "admin", "priority": "high",
            "subject_tpl": "合同续签:{{name}}",
            "description_tpl": "{{name}}({{employee_no}})合同 30 天内到期",
            "assignee_kind": "role", "assignee_value": "hr",
        },
        recipients=[{"kind": "role", "value": "hr"}],
        title_tpl="合同到期工单:{{name}}",
    )
    run_scheduled(db, today=date(2026, 6, 1))
    assert len(_runs(db, rule)) == 1
    case = db.scalar(select(m.HrCase).where(m.HrCase.tenant_id == t.id))
    assert case is not None
    assert case.case_no == 1 and case.kind == "admin"
    assert case.priority == "high" and case.status == "assigned"
    assert case.requester_employee_id == emp.id
    assert case.assignee_user_id == hr.id
    assert emp.name in case.subject and "30 天内到期" in case.description
    run = _runs(db, rule)[0]
    assert run.status == "ok" and run.payload["case_id"] == case.id
    # 收件人(hr)+ 受理人(同为 hr,去重)→ 至少一条通知
    notes = db.scalars(select(m.Notification).where(
        m.Notification.user_id == hr.id)).all()
    assert len(notes) >= 1


# ---------------------------------------------------------------------------
# API 校验
# ---------------------------------------------------------------------------


def test_rule_api_validation_and_schema(client, db):
    t = _tenant(db)
    admin = _user(db, t, "admin")

    schema = client.get("/api/automations/schema", headers=_hdr(admin)).json()
    assert {x["value"] for x in schema["trigger_kinds"]} == {"audit", "date_field"}
    assert "probation_end" in [x["value"] for x in schema["date_fields"]]
    assert "create_case" in schema["actions"]

    # date_field 非员工实体 → 400
    r = client.post("/api/automations", headers=_hdr(admin), json={
        "name": "x", "entity_type": "leave_application",
        "trigger_kind": "date_field",
        "schedule": {"field": "probation_end", "offset_days": 0},
        "action": "notify", "recipients": [{"kind": "role", "value": "hr"}],
        "title_tpl": "x"})
    assert r.status_code == 400

    # 日期字段不在白名单 → 400
    r = client.post("/api/automations", headers=_hdr(admin), json={
        "name": "x", "entity_type": "employee",
        "trigger_kind": "date_field",
        "schedule": {"field": "salary", "offset_days": 0},
        "action": "notify", "recipients": [{"kind": "role", "value": "hr"}],
        "title_tpl": "x"})
    assert r.status_code == 400

    # create_case 坏工单类型 → 400;合法创建 → 201
    r = client.post("/api/automations", headers=_hdr(admin), json={
        "name": "x", "entity_type": "employee",
        "trigger_kind": "date_field",
        "schedule": {"field": "contract_end", "offset_days": -30},
        "action": "create_case",
        "action_config": {"kind": "bogus"},
        "title_tpl": "x"})
    assert r.status_code == 400
    r = client.post("/api/automations", headers=_hdr(admin), json={
        "name": "合同到期建单", "entity_type": "employee",
        "trigger_kind": "date_field",
        "schedule": {"field": "contract_end", "offset_days": -30},
        "action": "create_case",
        "action_config": {"kind": "admin", "priority": "high"},
        "title_tpl": "合同到期:{{name}}"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["trigger_kind"] == "date_field"
    assert body["schedule"]["field"] == "contract_end"
    assert body["action_config"]["kind"] == "admin"


# ---------------------------------------------------------------------------
# 审批自动裁决
# ---------------------------------------------------------------------------


def _mk_policy(client, admin, **over):
    body = {
        "name": "自动审批策略", "request_type": "leave", "priority": 0,
        "conditions": [],
        "steps": [{"resolver": "role", "key": "hr", "label": "HR"}],
    }
    body.update(over)
    return client.post("/api/approvals/policies", headers=_hdr(admin), json=body)


def test_auto_approve_leave(db, client):
    """命中 auto_approve 条件的请假:提交即批准 + auto_approved 标记 + 通知。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    _emp(db, t, "小李", user=emp_user)
    lt = _leave_type(db, t)
    r = _mk_policy(client, admin, name="2天内自动批",
                   conditions=[{"field": "days", "op": "le", "value": 5}],
                   auto_approve=[{"field": "days", "op": "le", "value": 2}])
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert r.json()["auto_approve"][0]["field"] == "days"

    # 2 天 → 命中自动裁决
    r = client.post("/api/me/leave-applications", headers=_hdr(emp_user),
                    json={"leave_type_id": lt.id,
                          "start_date": "2027-03-01",
                          "end_date": "2027-03-02"})
    assert r.status_code in (200, 201), r.text
    app = db.get(m.LeaveApplication, r.json()["id"])
    inst = instance_for(db, t.id, "leave", app.id)
    assert inst is not None
    assert inst.status == "approved" and inst.auto_approved is True
    steps = db.scalars(select(m.ApprovalStep).where(
        m.ApprovalStep.instance_id == inst.id)).all()
    assert all(s.status == "approved" for s in steps)
    # 领域副作用:申请单 approved + approver = 策略创建者
    assert app.status == "approved" and app.approver_id == admin.id
    # 申请人收到"已自动批准"通知
    n = db.scalar(select(m.Notification).where(
        m.Notification.user_id == emp_user.id,
        m.Notification.title.like("%自动批准%")))
    assert n is not None
    # 实例详情透出 auto_approved
    det = client.get(f"/api/approvals/instances/leave/{app.id}",
                     headers=_hdr(emp_user)).json()
    assert det["auto_approved"] is True and det["status"] == "approved"

    # 5 个工作日(周一~周五)→ 命中策略但不满足 auto 条件 → 正常待批
    r = client.post("/api/me/leave-applications", headers=_hdr(emp_user),
                    json={"leave_type_id": lt.id,
                          "start_date": "2027-04-05",
                          "end_date": "2027-04-09"})
    assert r.status_code in (200, 201), r.text
    app2 = db.get(m.LeaveApplication, r.json()["id"])
    inst2 = instance_for(db, t.id, "leave", app2.id)
    assert inst2.status == "pending" and inst2.auto_approved is False
    assert app2.status == "pending"

    # 更新策略清掉 auto_approve
    r = client.put(f"/api/approvals/policies/{pid}", headers=_hdr(admin),
                   json={"auto_approve": []})
    assert r.status_code == 200 and r.json()["auto_approve"] is None


def test_auto_approve_validation_and_unmatched(db, client):
    """auto_approve 字段白名单校验;条件不命中退回人工链。"""
    t = _tenant(db)
    admin = _user(db, t, "admin")
    emp_user = _user(db, t, "employee")
    _emp(db, t, "小王", user=emp_user)
    lt = _leave_type(db, t)

    r = _mk_policy(client, admin,
                   auto_approve=[{"field": "salary", "op": "eq", "value": 1}])
    assert r.status_code == 422

    r = _mk_policy(client, admin, name="永不自动",
                   auto_approve=[{"field": "days", "op": "gt", "value": 999}])
    assert r.status_code == 201
    r = client.post("/api/me/leave-applications", headers=_hdr(emp_user),
                    json={"leave_type_id": lt.id,
                          "start_date": "2027-05-06",
                          "end_date": "2027-05-06"})
    assert r.status_code in (200, 201), r.text
    app = db.get(m.LeaveApplication, r.json()["id"])
    inst = instance_for(db, t.id, "leave", app.id)
    assert inst is not None and inst.status == "pending"
    assert inst.auto_approved is False
