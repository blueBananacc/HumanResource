"""HRIS 员工核心全链路测试(license feature: hr.suite,试用版默认开启)。

覆盖:
1. 员工 CRUD + 工号唯一 + 组织图 + 防环
2. 员工账号绑定(ESS):role=employee、登录、/me/* 行级过滤
3. 假期:类型 → 额度(台账+) → 员工申请 → HR 审批(台账-,考勤联动) → 撤销返还
4. 考勤:打卡 → 单日汇总
5. 生命周期:发起 → 任务勾选 → 完成副作用(transfer 回写部门 / separation 停账号)
6. 招聘需求:HM 提交 → admin 审批 → 转岗位
7. Offer:创建 → 发出 → 接受 → 自动建员工 + onboarding 事件 + 候选人转 hired
8. 绩效:周期 open → 铺表 → 员工自评 → HR 评定
9. 报销:员工建单 → HR 审批 → 标记付款
10. 工资条:创建 → 发放 → 员工端可见 → 回收
11. 权限边界:employee 角色访问管理端被拒;HM 只能批直属下属
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-hris")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-hris"

from app.domain.models import (  # noqa: E402
    Attendance,
    AuditLog,
    Base,
    Employee,
    LeaveApplication,
    OrgUnit,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from sqlalchemy import select  # noqa: E402


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


def _tenant(db) -> Tenant:
    t = Tenant(name=f"hris-{uuid.uuid4().hex[:8]}")
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
    from app.domain import models as m

    hris = (
        m.SalarySlip, m.ExpenseItem, m.ExpenseClaim, m.ExpenseType,
        m.Appraisal, m.AppraisalCycle, m.Goal,
        m.JobOffer, m.JobRequisition,
        m.LifecycleTask, m.LifecycleEvent, m.LifecycleTemplate,
        m.Attendance, m.EmployeeCheckin,
        m.LeaveLedgerEntry, m.LeaveApplication, m.LeaveAllocation, m.LeaveType,
        m.Employee,
    )
    for t in tenants:
        for model in hris:
            db.query(model).filter(model.tenant_id == t.id).delete()
        db.query(User).filter(User.tenant_id == t.id).delete()
        db.query(OrgUnit).filter(OrgUnit.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def _mk_emp(client, hdr, **kw) -> dict:
    body = {"name": kw.pop("name", f"员工{uuid.uuid4().hex[:4]}")}
    body.update(kw)
    res = client.post("/api/employees", headers=hdr, json=body)
    assert res.status_code == 201, res.text
    return res.json()


# ============ P0: 员工 + ESS ============


def test_employee_crud_no_and_org_chart(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        dept = OrgUnit(tenant_id=t.id, name="技术部", kind="department")
        db_session.add(dept)
        db_session.commit()

        mgr = _mk_emp(client, _hdr(admin), name="王经理", org_unit_id=dept.id,
                    designation="技术总监")
        e1 = _mk_emp(client, _hdr(admin), name="李工", org_unit_id=dept.id,
                   designation="工程师", reports_to_id=mgr["id"])
        e2 = _mk_emp(client, _hdr(admin), name="张工", reports_to_id=mgr["id"])

        # 工号自动递增且唯一
        assert mgr["employee_no"] != e1["employee_no"]

        # 防环:经理不能挂到下属下面
        res = client.put(
            f"/api/employees/{mgr['id']}", headers=_hdr(admin),
            json={"name": "王经理", "reports_to_id": e1["id"]},
        )
        assert res.status_code == 400

        # 组织图
        res = client.get("/api/employees/org-chart", headers=_hdr(admin))
        assert res.status_code == 200
        roots = res.json()
        mgr_node = next(r for r in roots if r["id"] == mgr["id"])
        assert {c["name"] for c in mgr_node["children"]} == {"李工", "张工"}

        # 列表 + 搜索
        res = client.get("/api/employees?q=李工", headers=_hdr(admin))
        assert res.json()["total"] == 1

        # 删除有下属的会保留(只清 reports_to),直接删干净员工 OK
        res = client.delete(f"/api/employees/{e2['id']}", headers=_hdr(admin))
        assert res.status_code == 204
        res = client.get(f"/api/employees/{e2['id']}", headers=_hdr(admin))
        assert res.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_employee_account_ess_flow(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp = _mk_emp(client, _hdr(admin), name="员工甲")
        email = f"emp-{uuid.uuid4().hex[:6]}@example.com"

        # 绑定账号 → 返回一次性密码
        res = client.post(
            f"/api/employees/{emp['id']}/account", headers=_hdr(admin),
            json={"email": email},
        )
        assert res.status_code == 200
        acct = res.json()
        assert acct["password"]

        # 用返回的密码登录
        res = client.post(
            "/api/auth/login", json={"email": email, "password": acct["password"]}
        )
        assert res.status_code == 200, res.text
        token = res.json()["access_token"]
        ess_hdr = {"Authorization": f"Bearer {token}"}

        # ESS profile
        res = client.get("/api/me/profile", headers=ess_hdr)
        assert res.status_code == 200
        assert res.json()["name"] == "员工甲"

        # employee 角色碰管理端 → 403
        res = client.get("/api/employees", headers=ess_hdr)
        assert res.status_code == 403
        res = client.get("/api/leave/applications", headers=ess_hdr)
        assert res.status_code == 403

        # 未绑定账号的普通 user 访问 /me → 403
        hr = _user(db_session, t, role="hr")
        res = client.get("/api/me/profile", headers=_hdr(hr))
        assert res.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_employee_no_duplicate_conflict(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        _mk_emp(client, _hdr(admin), employee_no="E100")
        res = client.post(
            "/api/employees", headers=_hdr(admin),
            json={"name": "另一个人", "employee_no": "E100"},
        )
        assert res.status_code == 409
    finally:
        _cleanup(db_session, t)


# ============ P1: 假期 + 考勤 ============


def _mk_leave_type(client, hdr, name="年假", days=10) -> dict:
    res = client.post(
        "/api/leave/types", headers=hdr,
        json={"name": name, "annual_days": days, "is_paid": True},
    )
    assert res.status_code == 201, res.text
    return res.json()


def _mk_ess_employee(client, hdr, name="员工乙"):
    """建员工 + 绑账号 + 登录,返回 (emp, ess_hdr)。"""
    emp = _mk_emp(client, hdr, name=name)
    email = f"ess-{uuid.uuid4().hex[:6]}@example.com"
    acct = client.post(
        f"/api/employees/{emp['id']}/account", headers=hdr, json={"email": email}
    ).json()
    token = client.post(
        "/api/auth/login", json={"email": email, "password": acct["password"]}
    ).json()["access_token"]
    return emp, {"Authorization": f"Bearer {token}"}


def test_leave_full_chain(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        emp, ess = _mk_ess_employee(client, _hdr(admin))

        # 额度 10 天
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 10},
        )
        assert res.status_code == 201

        # 员工申请 2 天
        start = date.today() + timedelta(days=7)
        end = start + timedelta(days=1)
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(start),
                  "end_date": str(end), "days": 2, "reason": "家里有"},
        )
        assert res.status_code == 201
        app_id = res.json()["id"]

        # 员工余额:pending 2
        res = client.get("/api/me/leave-balances", headers=ess)
        b = res.json()[0]
        assert b["allocated"] == 10 and b["pending"] == 2 and b["balance"] == 8

        # HR 端能看到并批准
        res = client.get(
            "/api/leave/applications?status=pending", headers=_hdr(admin)
        )
        assert res.json()["total"] == 1
        res = client.post(
            f"/api/leave/applications/{app_id}/approve",
            headers=_hdr(admin), json={"note": "ok"},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "approved"

        # 台账: alloc +10, application -2
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        entries = res.json()["items"]
        assert len(entries) == 2
        assert {e["delta_days"] for e in entries} == {10.0, -2.0}

        # 考勤联动:请假日 → leave
        att = db_session.scalars(
            select(Attendance).where(
                Attendance.employee_id == emp["id"],
                Attendance.day == start,
            )
        ).first()
        assert att and att.status == "leave"

        # 员工撤销 → 台账 +2 返还,考勤恢复
        res = client.post(
            f"/api/me/leave-applications/{app_id}/cancel", headers=ess
        )
        assert res.status_code == 200
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        assert res.json()["total"] == 3
        db_session.expire_all()
        att = db_session.scalars(
            select(Attendance).where(
                Attendance.employee_id == emp["id"], Attendance.day == start
            )
        ).first()
        assert att is None or att.status != "leave"

        # 余额恢复
        res = client.get("/api/me/leave-balances", headers=ess)
        assert res.json()[0]["balance"] == 10
    finally:
        _cleanup(db_session, t)


def test_leave_manager_team_scope(db_session):
    """hiring_manager 绑定员工档案后:scope=team 只见下属,只能批下属的单。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        mgr_emp = _mk_emp(client, _hdr(admin), name="经理")
        sub_emp, sub_ess = _mk_ess_employee(
            client, _hdr(admin), name="下属"
        )
        other_emp, other_ess = _mk_ess_employee(
            client, _hdr(admin), name="路人"
        )
        # 下属挂在经理下面;经理档案绑定 HM 账号(角色保留 hiring_manager)
        client.put(
            f"/api/employees/{sub_emp['id']}", headers=_hdr(admin),
            json={"name": "下属", "reports_to_id": mgr_emp["id"]},
        )
        res = client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        assert res.status_code == 200
        db_session.refresh(hm)
        assert hm.role == "hiring_manager"  # 保留角色不降级
        hm_hdr = _hdr(hm)

        # 下属、路人各发一单
        for hdr_ess in (sub_ess, other_ess):
            s = date.today() + timedelta(days=10)
            client.post(
                "/api/me/leave-applications", headers=hdr_ess,
                json={"leave_type_id": lt["id"], "start_date": str(s),
                      "end_date": str(s), "days": 1, "reason": "事"},
            )

        # HM scope=team 只见下属(+自己)的单
        res = client.get(
            "/api/leave/applications?scope=team&status=pending",
            headers=hm_hdr,
        )
        items = res.json()["items"]
        names = {i["employee_name"] for i in items}
        assert "下属" in names and "路人" not in names

        # HM 批下属 OK,批路人 403
        sub_app = next(i for i in items if i["employee_name"] == "下属")
        res = client.post(
            f"/api/leave/applications/{sub_app['id']}/approve",
            headers=hm_hdr, json={},
        )
        assert res.status_code == 200
        other_app = db_session.scalars(
            select(LeaveApplication.id).where(
                LeaveApplication.employee_id == other_emp["id"]
            )
        ).first()
        res = client.post(
            f"/api/leave/applications/{other_app}/approve",
            headers=hm_hdr, json={},
        )
        assert res.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_leave_allocation_adjust_and_revoke(db_session):
    """额度调整写台账 delta + 审计;重复发放 409;收回写负台账且余额行消失。"""
    t = _tenant(db_session)
    t2 = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    admin2 = _user(db_session, t2, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        emp = _mk_emp(client, _hdr(admin), name="调额")
        year = date.today().year
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": year, "days": 10},
        )
        assert res.status_code == 201
        alloc_id = res.json()["id"]

        # 同年同类型重复发放 → 409
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": year, "days": 5},
        )
        assert res.status_code == 409

        # 调整 10→6:台账记 allocation_adjust -4,余额/响应实时重算
        res = client.put(
            f"/api/leave/allocations/{alloc_id}", headers=_hdr(admin),
            json={"days": 6, "note": "减半"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["days"] == 6 and res.json()["note"] == "减半"
        b = client.get(
            f"/api/leave/balances?employee_id={emp['id']}&year={year}",
            headers=_hdr(admin),
        ).json()
        assert b[0]["allocation_id"] == alloc_id
        assert b[0]["allocated"] == 6 and b[0]["balance"] == 6
        by_type = {
            e["ref_type"]: e["delta_days"]
            for e in client.get("/api/leave/ledger", headers=_hdr(admin)).json()["items"]
        }
        assert by_type == {"allocation": 10.0, "allocation_adjust": -4.0}

        # 天数不变仅改备注 → 不再写台账
        res = client.put(
            f"/api/leave/allocations/{alloc_id}", headers=_hdr(admin),
            json={"days": 6, "note": "仅备注"},
        )
        assert res.status_code == 200 and res.json()["note"] == "仅备注"
        assert client.get("/api/leave/ledger", headers=_hdr(admin)).json()["total"] == 2

        # 跨租户不可见
        res = client.put(
            f"/api/leave/allocations/{alloc_id}", headers=_hdr(admin2),
            json={"days": 1},
        )
        assert res.status_code == 404
        res = client.delete(
            f"/api/leave/allocations/{alloc_id}", headers=_hdr(admin2)
        )
        assert res.status_code == 404

        # 收回 → 台账 allocation_revoke -6,行从余额列表消失
        res = client.delete(
            f"/api/leave/allocations/{alloc_id}", headers=_hdr(admin)
        )
        assert res.status_code == 204
        assert client.get(
            f"/api/leave/balances?employee_id={emp['id']}&year={year}",
            headers=_hdr(admin),
        ).json() == []
        entries = client.get("/api/leave/ledger", headers=_hdr(admin)).json()["items"]
        by_type = {e["ref_type"]: e["delta_days"] for e in entries}
        assert by_type["allocation_revoke"] == -6.0
        assert sum(e["delta_days"] for e in entries) == 0
    finally:
        _cleanup(db_session, t, t2)


def test_checkin_and_summarize(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        # 打卡 in/out
        res = client.post("/api/me/checkins", headers=ess, json={"kind": "in"})
        assert res.status_code == 201
        res = client.post("/api/me/checkins", headers=ess, json={"kind": "out"})
        assert res.status_code == 201

        # 今日 attendance 已自动汇总
        res = client.get("/api/me/attendance", headers=ess)
        rows = res.json()
        today = [r for r in rows if r["day"] == str(date.today())]
        assert today and today[0]["status"] == "present"
        assert today[0]["first_in"] and today[0]["last_out"]

        # HR 端考勤列表
        res = client.get(
            f"/api/attendance?employee_id={emp['id']}", headers=_hdr(admin)
        )
        assert res.json()["total"] >= 1
        # 手工修正
        att_id = res.json()["items"][0]["id"]
        res = client.put(
            f"/api/attendance/{att_id}", headers=_hdr(admin),
            json={"status": "remote"},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "remote"
        assert res.json()["source"] == "manual"

        # 手工修正写审计日志
        db_session.expire_all()
        log = db_session.scalars(
            select(AuditLog).where(
                AuditLog.entity_type == "attendance",
                AuditLog.entity_id == att_id,
                AuditLog.action == "manual_correct",
            )
        ).first()
        assert log and log.detail["status"] == "remote"
    finally:
        _cleanup(db_session, t)


# ============ P2: 生命周期 + 需求/Offer ============


def test_lifecycle_transfer_and_separation(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        dept_a = OrgUnit(tenant_id=t.id, name="技术部", kind="department")
        dept_b = OrgUnit(tenant_id=t.id, name="产品部", kind="department")
        db_session.add_all([dept_a, dept_b])
        db_session.commit()
        emp = _mk_emp(client, _hdr(admin), name="调岗人", org_unit_id=dept_a.id)

        # 发起调动
        res = client.post(
            "/api/lifecycle/events", headers=_hdr(admin),
            json={"employee_id": emp["id"], "kind": "transfer",
                  "payload": {"to_org_unit_id": dept_b.id},
                  "effective_date": str(date.today())},
        )
        assert res.status_code == 201, res.text
        ev = res.json()
        assert ev["payload"]["from_org_unit_id"] == dept_a.id
        assert ev["status"] == "in_progress"

        # 勾掉所有任务 → 事件完成 → 员工部门已换
        for task in ev["tasks"]:
            res = client.post(
                f"/api/lifecycle/tasks/{task['id']}/toggle", headers=_hdr(admin)
            )
            assert res.status_code == 200
        res = client.get(f"/api/lifecycle/events/{ev['id']}", headers=_hdr(admin))
        assert res.json()["status"] == "completed"
        res = client.get(f"/api/employees/{emp['id']}", headers=_hdr(admin))
        assert res.json()["org_unit_id"] == dept_b.id

        # 离职:全部完成 → status=terminated + 员工账号禁用
        email = f"sep-{uuid.uuid4().hex[:6]}@example.com"
        client.post(
            f"/api/employees/{emp['id']}/account", headers=_hdr(admin),
            json={"email": email},
        )
        res = client.post(
            "/api/lifecycle/events", headers=_hdr(admin),
            json={"employee_id": emp["id"], "kind": "separation"},
        )
        ev = res.json()
        for task in ev["tasks"]:
            client.post(
                f"/api/lifecycle/tasks/{task['id']}/toggle", headers=_hdr(admin)
            )
        res = client.get(f"/api/employees/{emp['id']}", headers=_hdr(admin))
        assert res.json()["status"] == "terminated"
        u = db_session.scalars(
            select(User).where(User.email == email)
        ).first()
        assert u.status == "disabled"
    finally:
        _cleanup(db_session, t)


def test_requisition_to_job(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    client = TestClient(app)
    try:
        # HM 提需求
        res = client.post(
            "/api/requisitions", headers=_hdr(hm),
            json={"title": "后端工程师", "headcount": 2, "reason": "业务扩张"},
        )
        assert res.status_code == 201
        rid = res.json()["id"]

        # HM 不能自己批
        res = client.post(
            f"/api/requisitions/{rid}/approve", headers=_hdr(hm)
        )
        assert res.status_code == 403

        # admin 批 → 转岗位(JSON body 里的审批意见要落库)
        res = client.post(
            f"/api/requisitions/{rid}/approve", headers=_hdr(admin),
            json={"note": "同意"},
        )
        assert res.json()["status"] == "approved"
        assert res.json()["decision_note"] == "同意"
        res = client.post(
            f"/api/requisitions/{rid}/to-job", headers=_hdr(admin)
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]
        assert job_id
        res = client.get(f"/api/jobs/{job_id}", headers=_hdr(admin))
        assert res.status_code == 200
        assert res.json()["title"] == "后端工程师"

        # 驳回同样记录审批意见
        res = client.post(
            "/api/requisitions", headers=_hdr(hm),
            json={"title": "前端工程师", "headcount": 1},
        )
        rid2 = res.json()["id"]
        res = client.post(
            f"/api/requisitions/{rid2}/reject", headers=_hdr(admin),
            json={"note": "编制冻结"},
        )
        assert res.json()["status"] == "rejected"
        assert res.json()["decision_note"] == "编制冻结"
    finally:
        _cleanup(db_session, t)


def test_offer_accept_creates_employee_and_onboarding(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        # 造候选人
        from app.domain.models import Candidate

        cand = Candidate(
            tenant_id=t.id, name="候选人丙",
            display_email=f"cand-{uuid.uuid4().hex[:6]}@example.com",
        )
        db_session.add(cand)
        db_session.commit()

        res = client.post(
            "/api/offers", headers=_hdr(admin),
            json={"candidate_id": cand.id, "position": "高级后端",
                  "start_date": str(date.today() + timedelta(days=14)),
                  "salary": {"base": 25000, "currency": "CNY", "period": "month"}},
        )
        assert res.status_code == 201
        oid = res.json()["id"]
        assert res.json()["letter"]

        res = client.post(f"/api/offers/{oid}/send", headers=_hdr(admin))
        assert res.json()["status"] == "sent"
        res = client.post(
            f"/api/offers/{oid}/respond", headers=_hdr(admin),
            json={"accepted": True},
        )
        body = res.json()
        assert body["status"] == "accepted"
        emp_id = body["employee_id"]
        assert emp_id

        # 员工档案已建,来源候选人对上
        res = client.get(f"/api/employees/{emp_id}", headers=_hdr(admin))
        emp = res.json()
        assert emp["name"] == "候选人丙"
        assert emp["source_candidate_id"] == cand.id
        assert emp["designation"] == "高级后端"

        # onboarding 事件已发起
        res = client.get(
            f"/api/lifecycle/events?employee_id={emp_id}&kind=onboarding",
            headers=_hdr(admin),
        )
        assert res.json()["total"] == 1
        assert len(res.json()["items"][0]["tasks"]) >= 3

        # 候选人转 hired
        db_session.expire_all()
        db_session.refresh(cand)
        assert cand.status == "hired"
    finally:
        _cleanup(db_session, t)


# ============ P4: 绩效 / 费用 / 薪酬 ============


def test_performance_cycle_flow(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        # 目标
        res = client.post(
            "/api/performance/goals", headers=_hdr(admin),
            json={"employee_id": emp["id"], "title": "完成招聘系统上线",
                  "due_date": str(date.today() + timedelta(days=90))},
        )
        assert res.status_code == 201
        gid = res.json()["id"]

        # 员工看到并更新进度
        res = client.get("/api/me/goals", headers=ess)
        assert len(res.json()) == 1
        res = client.patch(
            f"/api/me/goals/{gid}", headers=ess, json={"progress": 60}
        )
        assert res.json()["progress"] == 60

        # 周期
        res = client.post(
            "/api/performance/cycles", headers=_hdr(admin),
            json={"name": "2026 下半年", "period_start": "2026-07-01",
                  "period_end": "2026-12-31"},
        )
        cid = res.json()["id"]
        client.post(f"/api/performance/cycles/{cid}/open", headers=_hdr(admin))

        # 铺表 → 员工自评 → HR 评
        res = client.post(
            "/api/performance/appraisals", headers=_hdr(admin),
            json={"cycle_id": cid, "employee_ids": [emp["id"]]},
        )
        assert res.status_code == 201
        aid = res.json()[0]["id"]

        res = client.post(
            f"/api/me/appraisals/{aid}/submit-self", headers=ess,
            json={"self_review": "本季度完成招聘模块", "self_rating": 4},
        )
        assert res.json()["status"] == "self_submitted"
        res = client.post(
            f"/api/performance/appraisals/{aid}/review", headers=_hdr(admin),
            json={"manager_review": "达标", "manager_rating": 4},
        )
        assert res.json()["status"] == "reviewed"

        # 员工端现在能看到上级评定
        res = client.get("/api/me/appraisals", headers=ess)
        assert res.json()[0]["manager_rating"] == 4
    finally:
        _cleanup(db_session, t)


def test_expense_claim_flow(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        res = client.post(
            "/api/expenses/types", headers=_hdr(admin), json={"name": "差旅"},
        )
        tid = res.json()["id"]

        res = client.post(
            "/api/me/expense-claims", headers=ess,
            json={"title": "上海出差", "items": [
                {"type_id": tid, "occurred_on": str(date.today()),
                 "amount": 800, "note": "高铁往返"},
                {"occurred_on": str(date.today()), "amount": 320,
                 "note": "住宿一晚"},
            ]},
        )
        assert res.status_code == 201
        claim = res.json()
        assert claim["total"] == 1120
        assert len(claim["items"]) == 2

        res = client.post(
            f"/api/expenses/claims/{claim['id']}/approve",
            headers=_hdr(admin), json={"note": "票据齐全"},
        )
        assert res.json()["status"] == "approved"
        res = client.post(
            f"/api/expenses/claims/{claim['id']}/pay", headers=_hdr(admin)
        )
        assert res.json()["status"] == "paid"

        res = client.get("/api/me/expense-claims", headers=ess)
        assert res.json()[0]["status"] == "paid"
    finally:
        _cleanup(db_session, t)


def test_salary_slip_issue_and_ess_visibility(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        res = client.post(
            "/api/payroll/slips", headers=_hdr(admin),
            json={"employee_id": emp["id"], "period": "2026-10",
                  "items": [
                      {"label": "基本工资", "kind": "earning", "amount": 20000},
                      {"label": "绩效", "kind": "earning", "amount": 5000},
                      {"label": "社保", "kind": "deduction", "amount": 2200},
                      {"label": "个税", "kind": "deduction", "amount": 980},
                  ]},
        )
        assert res.status_code == 201
        s = res.json()
        assert s["gross"] == 25000 and s["deductions"] == 3180
        assert s["net"] == 25000 - 3180
        assert s["status"] == "draft"

        # draft 员工端看不到
        res = client.get("/api/me/salary-slips", headers=ess)
        assert res.json() == []

        # 发放 → 可见;再回收 → 不可见
        client.post(f"/api/payroll/slips/{s['id']}/issue", headers=_hdr(admin))
        res = client.get("/api/me/salary-slips", headers=ess)
        assert len(res.json()) == 1
        assert res.json()[0]["net"] == 21820
        client.post(f"/api/payroll/slips/{s['id']}/retract", headers=_hdr(admin))
        res = client.get("/api/me/salary-slips", headers=ess)
        assert res.json() == []

        # 同员工同期间唯一
        res = client.post(
            "/api/payroll/slips", headers=_hdr(admin),
            json={"employee_id": emp["id"], "period": "2026-10", "items": []},
        )
        assert res.status_code == 409
    finally:
        _cleanup(db_session, t)


def test_hr_suite_feature_gate(db_session):
    """license 关掉 hr.suite → 全部新端点 402。"""
    from app.api import license as license_api

    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        orig = license_api.is_feature_enabled
        license_api.is_feature_enabled = lambda db, f: False
        try:
            res = client.get("/api/employees", headers=_hdr(admin))
            assert res.status_code == 402
            res = client.get("/api/me/profile", headers=_hdr(admin))
            assert res.status_code == 402
        finally:
            license_api.is_feature_enabled = orig
    finally:
        _cleanup(db_session, t)
