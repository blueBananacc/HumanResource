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
from datetime import date, datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-hris")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-hris"

from app.domain.models import (  # noqa: E402
    Attendance,
    AuditLog,
    Base,
    EmployeeCheckin,
    LeaveAllocation,
    LeaveApplication,
    LeaveType,
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
        m.AttendanceRequest, m.OvertimeRequest,
        m.LeavePolicy, m.Holiday, m.WorkCalendar,
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


def _wk_span(n_workdays: int, *, weeks_ahead: int = 2) -> tuple[date, date]:
    """默认周一~周五工作制下,返回 n 个连续工作日的 (start, end) 区间。

    起点取 ``weeks_ahead`` 周后的下周一,n<=5 时整段落在同一周内,
    保证区间内工作日数恰为 n(不含节假日覆盖的租户用)。
    """
    assert 1 <= n_workdays <= 5, "辅助函数只支持同一周内的区间"
    s = date.today() + timedelta(weeks=weeks_ahead)
    s += timedelta(days=(7 - s.weekday()) % 7)  # 对齐到周一
    return s, s + timedelta(days=n_workdays - 1)


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

        # 员工申请 2 天(周一~周二,默认工作制下都是工作日)
        start, end = _wk_span(2)
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
        # 撤销后考勤就地重算:工作日、无打卡、无在途请假 → absent,而不是删行留空洞
        assert att and att.status == "absent"

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
            s, _ = _wk_span(1, weeks_ahead=1)
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


def test_closed_cycle_blocks_self_submit_and_review(db_session):
    """周期关闭后:员工不能再自评,主管不能再评定。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        res = client.post(
            "/api/performance/cycles", headers=_hdr(admin),
            json={"name": "2026 Q3", "period_start": "2026-07-01",
                  "period_end": "2026-09-30"},
        )
        cid = res.json()["id"]
        client.post(f"/api/performance/cycles/{cid}/open", headers=_hdr(admin))
        res = client.post(
            "/api/performance/appraisals", headers=_hdr(admin),
            json={"cycle_id": cid, "employee_ids": [emp["id"]]},
        )
        aid = res.json()[0]["id"]

        # 关闭周期 → 自评被拒
        client.post(f"/api/performance/cycles/{cid}/close", headers=_hdr(admin))
        res = client.post(
            f"/api/me/appraisals/{aid}/submit-self", headers=ess,
            json={"self_review": "补交", "self_rating": 5},
        )
        assert res.status_code == 400

        # 已提交自评的考核,关闭后主管也不能再评
        res = client.post(
            "/api/performance/cycles", headers=_hdr(admin),
            json={"name": "2026 Q4", "period_start": "2026-10-01",
                  "period_end": "2026-12-31"},
        )
        cid2 = res.json()["id"]
        client.post(f"/api/performance/cycles/{cid2}/open", headers=_hdr(admin))
        res = client.post(
            "/api/performance/appraisals", headers=_hdr(admin),
            json={"cycle_id": cid2, "employee_ids": [emp["id"]]},
        )
        aid2 = res.json()[0]["id"]
        client.post(
            f"/api/me/appraisals/{aid2}/submit-self", headers=ess,
            json={"self_review": "Q4 自评", "self_rating": 3},
        )
        client.post(f"/api/performance/cycles/{cid2}/close", headers=_hdr(admin))
        res = client.post(
            f"/api/performance/appraisals/{aid2}/review", headers=_hdr(admin),
            json={"manager_review": "达标", "manager_rating": 4},
        )
        assert res.status_code == 400
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
        # 回归:多明细报销单在 ESS 列表里必须保留全部明细
        assert len(res.json()[0]["items"]) == 2
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


# ============ 回归:行级可见性 / 事务 / 字段保护 ============


def test_leave_apps_all_scope_still_row_filtered(db_session):
    """scope=all(默认)对非 admin/hr 也不能看见非下属的单。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        mgr_emp = _mk_emp(client, _hdr(admin), name="经理2")
        sub_emp, sub_ess = _mk_ess_employee(client, _hdr(admin), name="下属2")
        _, other_ess = _mk_ess_employee(client, _hdr(admin), name="外人2")
        client.put(
            f"/api/employees/{sub_emp['id']}", headers=_hdr(admin),
            json={"name": "下属2", "reports_to_id": mgr_emp["id"]},
        )
        client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        for e in (sub_ess, other_ess):
            s, _ = _wk_span(1, weeks_ahead=2)
            client.post(
                "/api/me/leave-applications", headers=e,
                json={"leave_type_id": lt["id"], "start_date": str(s),
                      "end_date": str(s), "days": 1},
            )

        # 不传 scope(默认 all)→ 仍只见下属+自己
        res = client.get("/api/leave/applications", headers=_hdr(hm))
        names = {i["employee_name"] for i in res.json()["items"]}
        assert "下属2" in names and "外人2" not in names
        # 显式 all 也一样
        res = client.get("/api/leave/applications?scope=all", headers=_hdr(hm))
        names = {i["employee_name"] for i in res.json()["items"]}
        assert "外人2" not in names
        # admin 不受限
        res = client.get("/api/leave/applications", headers=_hdr(admin))
        names = {i["employee_name"] for i in res.json()["items"]}
        assert {"下属2", "外人2"} <= names
    finally:
        _cleanup(db_session, t)


def test_expense_claim_detail_row_scope(db_session):
    """详情接口与列表同口径:非下属/无关人 404,不暴露存在性。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    viewer = _user(db_session, t, role="viewer")
    client = TestClient(app)
    try:
        res = client.post(
            "/api/expenses/types", headers=_hdr(admin), json={"name": "餐费"}
        )
        tid = res.json()["id"]
        mgr_emp = _mk_emp(client, _hdr(admin), name="报销经理")
        sub_emp, sub_ess = _mk_ess_employee(client, _hdr(admin), name="报销下属")
        _, other_ess = _mk_ess_employee(client, _hdr(admin), name="报销外人")
        client.put(
            f"/api/employees/{sub_emp['id']}", headers=_hdr(admin),
            json={"name": "报销下属", "reports_to_id": mgr_emp["id"]},
        )
        client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        c1 = client.post(
            "/api/me/expense-claims", headers=sub_ess,
            json={"title": "下属单", "items": [
                {"type_id": tid, "occurred_on": str(date.today()), "amount": 50}
            ]},
        ).json()
        c2 = client.post(
            "/api/me/expense-claims", headers=other_ess,
            json={"title": "外人单", "items": [
                {"occurred_on": str(date.today()), "amount": 60}
            ]},
        ).json()

        # HM:下属单可读,外人单 404
        assert client.get(
            f"/api/expenses/claims/{c1['id']}", headers=_hdr(hm)
        ).status_code == 200
        assert client.get(
            f"/api/expenses/claims/{c2['id']}", headers=_hdr(hm)
        ).status_code == 404
        # viewer:两个都 404
        assert client.get(
            f"/api/expenses/claims/{c1['id']}", headers=_hdr(viewer)
        ).status_code == 404
        # admin 不受限
        assert client.get(
            f"/api/expenses/claims/{c2['id']}", headers=_hdr(admin)
        ).status_code == 200
    finally:
        _cleanup(db_session, t)


def test_spawn_appraisals_skips_duplicates_atomically(db_session):
    """重复铺表:已存在的跳过,新插入的不应被 rollback 吞掉。"""
    from app.domain.models import Appraisal

    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        e1 = _mk_emp(client, _hdr(admin), name="考核1")
        e2 = _mk_emp(client, _hdr(admin), name="考核2")
        res = client.post(
            "/api/performance/cycles", headers=_hdr(admin),
            json={"name": "周期X", "period_start": "2026-01-01",
                  "period_end": "2026-06-30"},
        )
        cid = res.json()["id"]
        client.post(f"/api/performance/cycles/{cid}/open", headers=_hdr(admin))

        res = client.post(
            "/api/performance/appraisals", headers=_hdr(admin),
            json={"cycle_id": cid, "employee_ids": [e1["id"], e2["id"]]},
        )
        assert res.status_code == 201 and len(res.json()) == 2

        # 第二次:e1 重复 + e2 重复 + 一个新员工 e3 → 只应新增 e3
        e3 = _mk_emp(client, _hdr(admin), name="考核3")
        res = client.post(
            "/api/performance/appraisals", headers=_hdr(admin),
            json={"cycle_id": cid,
                  "employee_ids": [e1["id"], e2["id"], e3["id"], e3["id"]]},
        )
        assert res.status_code == 201
        got = res.json()
        assert len(got) == 1 and got[0]["employee_id"] == e3["id"]
        # 库内总数 = 3,响应里的行真实存在
        db_session.expire_all()
        n = db_session.scalars(
            select(Appraisal.id).where(Appraisal.cycle_id == cid)
        ).all()
        assert len(n) == 3
        assert db_session.get(Appraisal, got[0]["id"]) is not None
    finally:
        _cleanup(db_session, t)


def test_update_employee_preserves_personal(db_session):
    """PUT 不带 personal → 员工自助端填的数据不能被清空。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp = _mk_emp(client, _hdr(admin), name="个人信息")
        client.put(
            f"/api/employees/{emp['id']}", headers=_hdr(admin),
            json={"name": "个人信息",
                  "personal": {"住址": "北京", "紧急联系人": "张三 138xx"}},
        )
        # 第二次编辑不带 personal
        res = client.put(
            f"/api/employees/{emp['id']}", headers=_hdr(admin),
            json={"name": "个人信息", "designation": "新职位"},
        )
        assert res.status_code == 200
        assert res.json()["personal"] == {"住址": "北京", "紧急联系人": "张三 138xx"}
        # 显式传 null 才清空
        res = client.put(
            f"/api/employees/{emp['id']}", headers=_hdr(admin),
            json={"name": "个人信息", "personal": None},
        )
        assert res.json()["personal"] is None
    finally:
        _cleanup(db_session, t)


def test_lifecycle_task_locked_after_complete(db_session):
    """事件 completed/cancelled 后任务不可再勾选。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp = _mk_emp(client, _hdr(admin), name="锁定人")
        res = client.post(
            "/api/lifecycle/events", headers=_hdr(admin),
            json={"employee_id": emp["id"], "kind": "onboarding"},
        )
        ev = res.json()
        for task in ev["tasks"]:
            client.post(
                f"/api/lifecycle/tasks/{task['id']}/toggle", headers=_hdr(admin)
            )
        assert client.get(
            f"/api/lifecycle/events/{ev['id']}", headers=_hdr(admin)
        ).json()["status"] == "completed"
        res = client.post(
            f"/api/lifecycle/tasks/{ev['tasks'][0]['id']}/toggle",
            headers=_hdr(admin),
        )
        assert res.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_expense_type_delete_guarded_by_usage(db_session):
    """被报销明细引用的类型不能删;空类型可删。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        t1 = client.post(
            "/api/expenses/types", headers=_hdr(admin), json={"name": "打车"}
        ).json()
        t2 = client.post(
            "/api/expenses/types", headers=_hdr(admin), json={"name": "快递"}
        ).json()
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        client.post(
            "/api/me/expense-claims", headers=ess,
            json={"title": "用类型", "items": [
                {"type_id": t1["id"], "occurred_on": str(date.today()),
                 "amount": 30}
            ]},
        )
        assert client.delete(
            f"/api/expenses/types/{t1['id']}", headers=_hdr(admin)
        ).status_code == 400
        assert client.delete(
            f"/api/expenses/types/{t2['id']}", headers=_hdr(admin)
        ).status_code == 204
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


# ============ 审查回归:读边界 / 自批 / 账号解绑 ============


def test_hris_read_boundaries(db_session):
    """viewer/interviewer 只有 view_reports:员工档案、Offer、额度、生命周期 → 403;
    组织架构图(目录级)仍可读。"""
    from app.domain.models import Candidate

    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    viewer = _user(db_session, t, role="viewer")
    interviewer = _user(db_session, t, role="interviewer")
    client = TestClient(app)
    try:
        emp = _mk_emp(client, _hdr(admin))
        lt = _mk_leave_type(client, _hdr(admin))
        client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 5},
        )
        ev = client.post(
            "/api/lifecycle/events", headers=_hdr(admin),
            json={"employee_id": emp["id"], "kind": "onboarding"},
        ).json()
        cand = Candidate(
            tenant_id=t.id, name="候选人丁",
            display_email=f"cand-{uuid.uuid4().hex[:6]}@example.com",
        )
        db_session.add(cand)
        db_session.commit()
        offer = client.post(
            "/api/offers", headers=_hdr(admin), json={"candidate_id": cand.id},
        ).json()

        for u in (viewer, interviewer):
            hdr = _hdr(u)
            assert client.get("/api/employees", headers=hdr).status_code == 403
            assert client.get(
                f"/api/employees/{emp['id']}", headers=hdr
            ).status_code == 403
            assert client.get(
                "/api/leave/allocations", headers=hdr
            ).status_code == 403
            assert client.get("/api/offers", headers=hdr).status_code == 403
            assert client.get(
                f"/api/offers/{offer['id']}", headers=hdr
            ).status_code == 403
            assert client.get(
                "/api/lifecycle/events", headers=hdr
            ).status_code == 403
            assert client.get(
                f"/api/lifecycle/events/{ev['id']}", headers=hdr
            ).status_code == 403
            # 目录级组织图对内部角色仍开放(只含姓名/职位/部门)
            assert client.get(
                "/api/employees/org-chart", headers=hdr
            ).status_code == 200
    finally:
        _cleanup(db_session, t)


def test_self_approval_blocked(db_session):
    """职责分离:绑定了员工档案的 admin 不能批自己的请假/报销/需求。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        self_emp = _mk_emp(client, _hdr(admin), name="管理员本人")
        client.post(
            f"/api/employees/{self_emp['id']}/account", headers=_hdr(admin),
            json={"email": admin.email},
        )

        # 请假:admin 自提 → 自批 403;hr 代批 200
        s, _ = _wk_span(1, weeks_ahead=3)
        app_id = client.post(
            "/api/me/leave-applications", headers=_hdr(admin),
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(s), "days": 1},
        ).json()["id"]
        assert client.post(
            f"/api/leave/applications/{app_id}/approve",
            headers=_hdr(admin), json={},
        ).status_code == 403
        assert client.post(
            f"/api/leave/applications/{app_id}/approve",
            headers=_hdr(hr), json={},
        ).status_code == 200

        # 报销:admin 自提 → 自批 403;hr 代批 200
        cid = client.post(
            "/api/me/expense-claims", headers=_hdr(admin),
            json={"title": "自己的单", "items": [
                {"occurred_on": str(date.today()), "amount": 10}]},
        ).json()["id"]
        assert client.post(
            f"/api/expenses/claims/{cid}/approve",
            headers=_hdr(admin), json={},
        ).status_code == 403
        assert client.post(
            f"/api/expenses/claims/{cid}/approve",
            headers=_hdr(hr), json={},
        ).status_code == 200

        # 招聘需求:admin 自提 → 自批 403;hr 代批 200
        rid = client.post(
            "/api/requisitions", headers=_hdr(admin),
            json={"title": "自己的需求", "headcount": 1},
        ).json()["id"]
        assert client.post(
            f"/api/requisitions/{rid}/approve",
            headers=_hdr(admin), json={},
        ).status_code == 403
        assert client.post(
            f"/api/requisitions/{rid}/approve",
            headers=_hdr(hr), json={},
        ).status_code == 200
    finally:
        _cleanup(db_session, t)


def test_expense_approve_perm_and_hm_scope(db_session):
    """approve_expenses 权限:viewer/interviewer 403;HM 只批直属下属。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    viewer = _user(db_session, t, role="viewer")
    interviewer = _user(db_session, t, role="interviewer")
    client = TestClient(app)
    try:
        mgr_emp = _mk_emp(client, _hdr(admin), name="报销主管")
        sub_emp, sub_ess = _mk_ess_employee(client, _hdr(admin), name="报销下属2")
        _, other_ess = _mk_ess_employee(client, _hdr(admin), name="报销外人2")
        client.put(
            f"/api/employees/{sub_emp['id']}", headers=_hdr(admin),
            json={"name": "报销下属2", "reports_to_id": mgr_emp["id"]},
        )
        client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        c_sub = client.post(
            "/api/me/expense-claims", headers=sub_ess,
            json={"title": "下属的", "items": [
                {"occurred_on": str(date.today()), "amount": 20}]},
        ).json()
        c_other = client.post(
            "/api/me/expense-claims", headers=other_ess,
            json={"title": "外人的", "items": [
                {"occurred_on": str(date.today()), "amount": 30}]},
        ).json()

        # viewer/interviewer 无 approve_expenses → 403
        for u in (viewer, interviewer):
            assert client.post(
                f"/api/expenses/claims/{c_sub['id']}/approve",
                headers=_hdr(u), json={},
            ).status_code == 403
        # HM 批下属 OK;批外人 403
        assert client.post(
            f"/api/expenses/claims/{c_sub['id']}/approve",
            headers=_hdr(hm), json={},
        ).status_code == 200
        assert client.post(
            f"/api/expenses/claims/{c_other['id']}/approve",
            headers=_hdr(hm), json={},
        ).status_code == 403
    finally:
        _cleanup(db_session, t)


def test_unbind_disables_ess_account(db_session):
    """解绑 ESS 账号 → 账号停用不能再登录;重新绑定同邮箱恢复。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp = _mk_emp(client, _hdr(admin))
        email = f"ub-{uuid.uuid4().hex[:6]}@example.com"
        acct = client.post(
            f"/api/employees/{emp['id']}/account", headers=_hdr(admin),
            json={"email": email},
        ).json()

        res = client.delete(
            f"/api/employees/{emp['id']}/account", headers=_hdr(admin)
        )
        assert res.status_code == 204
        u = db_session.scalars(select(User).where(User.email == email)).first()
        assert u.status == "disabled"
        # 停用后旧密码登录被拒
        res = client.post(
            "/api/auth/login",
            json={"email": email, "password": acct["password"]},
        )
        assert res.status_code in (401, 403)
        # 重新绑定同邮箱 → 恢复 active
        res = client.post(
            f"/api/employees/{emp['id']}/account", headers=_hdr(admin),
            json={"email": email},
        )
        assert res.status_code == 200
        db_session.expire_all()
        u = db_session.scalars(select(User).where(User.email == email)).first()
        assert u.status == "active"
    finally:
        _cleanup(db_session, t)


def test_half_day_validation(db_session):
    """半天粒度:0.3 天申请/额度 422,0.5 合法。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        emp, ess = _mk_ess_employee(client, _hdr(admin))

        # 额度:0.3 → 422;7.5 → 201
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 0.3},
        )
        assert res.status_code == 422
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 7.5},
        )
        assert res.status_code == 201

        # ESS 申请:0.3 → 422;0.5 → 201
        s, _ = _wk_span(1, weeks_ahead=8)
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(s), "days": 0.3},
        )
        assert res.status_code == 422
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(s), "days": 0.5},
        )
        assert res.status_code == 201, res.text
    finally:
        _cleanup(db_session, t)

# ============ 回归修复:ESS 绑定账号 / 请假天数区间 / 台账收回 ============


def test_bound_non_employee_ess_access(db_session):
    """绑定员工档案的非 employee 角色(用人经理等)也能用 /me 自助区。

    回归点:/auth/me 必须返回 employee_id,否则前端无法放行 ESS 路由;
    绑定后该账号应能走 /api/me/profile、提交自己的请假申请。
    """
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        mgr_emp = _mk_emp(client, _hdr(admin), name="经理")
        res = client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        assert res.status_code == 200

        # /auth/me 返回 employee_id
        res = client.get("/api/auth/me", headers=_hdr(hm))
        assert res.status_code == 200
        assert res.json()["employee_id"] == mgr_emp["id"]

        # /api/me/* 可用:读档案 + 自己提交请假
        res = client.get("/api/me/profile", headers=_hdr(hm))
        assert res.status_code == 200
        assert res.json()["name"] == "经理"

        # 提交自己的请假 → 出现在管理端 pending 列表(scope=mine)
        client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": mgr_emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 10},
        )
        s, _ = _wk_span(1, weeks_ahead=12)
        res = client.post(
            "/api/me/leave-applications", headers=_hdr(hm),
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(s), "days": 1, "reason": "经理自己请假"},
        )
        assert res.status_code == 201, res.text
    finally:
        _cleanup(db_session, t)


def test_leave_days_must_match_span(db_session):
    """请假天数只允许 = 区间工作日数 或 少一个半天,防止少扣额度换整段休假考勤。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        s, e = _wk_span(3)  # 周一~周三,3 个工作日

        # days=1 对 3 个工作日 → 400
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(e), "days": 1, "reason": "钻空子"},
        )
        assert res.status_code == 400

        # days=2.5(少一个半天)→ 201
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(s),
                  "end_date": str(e), "days": 2.5, "reason": "带半天"},
        )
        assert res.status_code == 201, res.text
    finally:
        _cleanup(db_session, t)


def test_leave_span_skips_weekends(db_session):
    """跨周末的请假:days 按工作日算 — Fri~Mon 是 2 个工作日不是 4 天。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin))
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        mon, _ = _wk_span(1)
        fri, next_mon = mon + timedelta(days=4), mon + timedelta(days=7)

        # Fri~Mon = 4 个日历天但只有 2 个工作日:days=4 超过工作日数 → 400
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(fri),
                  "end_date": str(next_mon), "days": 4},
        )
        assert res.status_code == 400
        # days=2 等于工作日数 → 201;不传 days 服务端也算 2
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt["id"], "start_date": str(fri),
                  "end_date": str(next_mon), "days": 2},
        )
        assert res.status_code == 201, res.text

        # 批准后考勤只标记工作日(Fri/Mon),周末两天无 leave 考勤行
        app_id = res.json()["id"]
        res = client.post(
            f"/api/leave/applications/{app_id}/approve",
            headers=_hdr(admin), json={},
        )
        assert res.status_code == 200
        for d, want in ((fri, "leave"), (fri + timedelta(days=1), None),
                        (fri + timedelta(days=2), None), (next_mon, "leave")):
            att = db_session.scalars(
                select(Attendance).where(
                    Attendance.employee_id == emp["id"], Attendance.day == d
                )
            ).first()
            if want is None:
                assert not att or att.status != "leave"
            else:
                assert att and att.status == "leave"

        # 整段重算考勤仍按同一口径:周末不能补出 leave 行(标 holiday),
        # 否则台账扣 2 天而考勤标 4 天,两边对不上
        res = client.post(
            "/api/attendance/summarize", headers=_hdr(admin),
            json={"start": str(fri), "end": str(next_mon)},
        )
        assert res.status_code == 200
        for d, want in ((fri, "leave"), (fri + timedelta(days=1), "holiday"),
                        (fri + timedelta(days=2), "holiday"),
                        (next_mon, "leave")):
            att = db_session.scalars(
                select(Attendance).where(
                    Attendance.employee_id == emp["id"], Attendance.day == d
                )
            ).first()
            assert att and att.status == want, (d, att.status if att else None)

        # 撤销后 leave 行就地重算回考勤口径(工作日 → absent,周末 → holiday),
        # 不再出现"行被删光、整天无考勤"的空洞
        res = client.post(
            f"/api/me/leave-applications/{app_id}/cancel", headers=ess
        )
        assert res.status_code == 200
        for d, want in ((fri, "absent"), (fri + timedelta(days=1), "holiday"),
                        (fri + timedelta(days=2), "holiday"),
                        (next_mon, "absent")):
            att = db_session.scalars(
                select(Attendance).where(
                    Attendance.employee_id == emp["id"], Attendance.day == d
                )
            ).first()
            assert att and att.status == want, (d, att.status if att else None)
    finally:
        _cleanup(db_session, t)


def test_delete_leave_type_leaves_no_orphans(db_session):
    """删除假期类型:其额度与子台账随 FK CASCADE 清掉,不留孤儿行。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin), name="调休假", days=3)
        lt2 = _mk_leave_type(client, _hdr(admin), name="年假", days=10)
        emp = _mk_emp(client, _hdr(admin))
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt["id"],
                  "year": date.today().year, "days": 3},
        )
        assert res.status_code == 201
        res = client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt2["id"],
                  "year": date.today().year, "days": 10},
        )
        assert res.status_code == 201

        res = client.delete(f"/api/leave/types/{lt['id']}", headers=_hdr(admin))
        assert res.status_code == 204

        # 台账只剩年假类型的 +10,调休假的 +3 随级联清除(无孤儿行)
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        entries = res.json()["items"]
        assert len(entries) == 1
        assert entries[0]["delta_days"] == 10.0

        # 其额度也一并删除
        res = client.get("/api/leave/allocations", headers=_hdr(admin))
        allocs = res.json()["items"]
        assert len(allocs) == 1 and allocs[0]["leave_type_name"] == "年假"
    finally:
        _cleanup(db_session, t)


def test_delete_leave_type_blocked_by_refs(db_session):
    """删除被引用的假期类型:挂着加班申请单 / 被假期政策引用的类型都要拒绝。

    overtime_requests.leave_type_id 也是 ON DELETE CASCADE —— 不拦的话
    删类型会把审批历史和已入账额度一起抹掉。
    """
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        comp = client.post(
            "/api/leave/types", headers=_hdr(admin),
            json={"name": "调休假", "annual_days": 0, "is_paid": True,
                  "is_comp": True},
        )
        assert comp.status_code == 201, comp.text
        comp = comp.json()
        emp, ess = _mk_ess_employee(client, _hdr(admin))

        # 有加班申请(任意状态)→ 400
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"],
                  "work_date": str(date.today() - timedelta(days=1)),
                  "hours": 4},
        )
        assert res.status_code == 201, res.text
        res = client.delete(
            f"/api/leave/types/{comp['id']}", headers=_hdr(admin)
        )
        assert res.status_code == 400

        # 被假期政策引用 → 400
        plain = _mk_leave_type(client, _hdr(admin), name="婚假")
        res = client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "含婚假的包",
                  "items": [{"leave_type_id": plain["id"], "days": 3}]},
        )
        assert res.status_code == 201, res.text
        res = client.delete(
            f"/api/leave/types/{plain['id']}", headers=_hdr(admin)
        )
        assert res.status_code == 400

        # 没有任何引用的普通类型仍可删
        free = _mk_leave_type(client, _hdr(admin), name="公益假")
        res = client.delete(
            f"/api/leave/types/{free['id']}", headers=_hdr(admin)
        )
        assert res.status_code == 204
    finally:
        _cleanup(db_session, t)


# ============ 工作日历 / 补卡 / 加班调休 / 假期政策(0031 workforce)============


def test_work_calendar_holidays_and_attendance(db_session):
    """工作日历配置 + 节假日覆盖 + workday-count + 考勤 holiday/absent 口径。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        # 默认日历:周一~周五 8h
        res = client.get("/api/work-calendar", headers=_hdr(admin))
        assert res.status_code == 200
        assert res.json() == {"workdays": [1, 2, 3, 4, 5], "workday_hours": 8.0}

        # 改成周一~周六、7.5h
        res = client.put(
            "/api/work-calendar", headers=_hdr(admin),
            json={"workdays": [1, 2, 3, 4, 5, 6], "workday_hours": 7.5},
        )
        assert res.status_code == 200
        assert res.json()["workday_hours"] == 7.5

        mon, _ = _wk_span(1)
        # 周一放假 → 那一周工作日数 = 5(周六补上了)
        res = client.post(
            "/api/work-calendar/holidays", headers=_hdr(admin),
            json={"day": str(mon), "name": "司庆日", "kind": "holiday"},
        )
        assert res.status_code == 201
        hid = res.json()["id"]
        # 同一天重复 → 409
        res = client.post(
            "/api/work-calendar/holidays", headers=_hdr(admin),
            json={"day": str(mon), "name": "重复", "kind": "holiday"},
        )
        assert res.status_code == 409
        res = client.get(
            f"/api/work-calendar/count?start={mon}&end={mon + timedelta(days=5)}",
            headers=_hdr(admin),
        )
        assert res.json()["workdays"] == 5  # 6 个工作日制 - 1 天假

        # ESS 端 workday-count 同口径
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        res = client.get(
            f"/api/me/workday-count?start={mon}&end={mon + timedelta(days=5)}",
            headers=ess,
        )
        assert res.status_code == 200
        assert res.json()["workdays"] == 5

        # 汇总考勤:放假的周一(无打卡)→ holiday;正常工作日 → absent
        res = client.post(
            "/api/attendance/summarize", headers=_hdr(admin),
            json={"start": str(mon), "end": str(mon + timedelta(days=2))},
        )
        assert res.status_code == 200
        for d, want in ((mon, "holiday"), (mon + timedelta(days=1), "absent")):
            att = db_session.scalars(
                select(Attendance).where(
                    Attendance.employee_id == emp["id"], Attendance.day == d
                )
            ).first()
            assert att and att.status == want, (d, att.status if att else None)

        # 删除节假日后重算:周一变回工作日 → absent
        res = client.delete(
            f"/api/work-calendar/holidays/{hid}", headers=_hdr(admin)
        )
        assert res.status_code == 204
        res = client.post(
            "/api/attendance/summarize", headers=_hdr(admin),
            json={"start": str(mon), "end": str(mon)},
        )
        att = db_session.scalars(
            select(Attendance).where(
                Attendance.employee_id == emp["id"], Attendance.day == mon
            )
        ).first()
        assert att.status == "absent"
    finally:
        _cleanup(db_session, t)


def test_attendance_request_chain(db_session):
    """补卡申请:ESS 发起 → 审批补写 checkin 重算当日;驳回/撤销路径。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess = _mk_ess_employee(client, _hdr(admin))
        day = date.today() - timedelta(days=3)
        ts = datetime(day.year, day.month, day.day, 9, 5)

        # 未来时刻 → 400
        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": (datetime.now() + timedelta(days=1)).isoformat(),
                  "kind": "in"},
        )
        assert res.status_code == 400

        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": ts.isoformat(), "kind": "in", "reason": "早会忘打卡"},
        )
        assert res.status_code == 201, res.text
        req_id = res.json()["id"]

        # 同日同向 pending 去重 → 400
        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": ts.replace(hour=10).isoformat(), "kind": "in"},
        )
        assert res.status_code == 400

        # admin 列表可见并批准 → 补写 checkin(method=request)+ 当日 present
        res = client.get(
            "/api/attendance/requests?status=pending", headers=_hdr(admin)
        )
        assert any(i["id"] == req_id for i in res.json()["items"])
        res = client.post(
            f"/api/attendance/requests/{req_id}/approve",
            headers=_hdr(admin), json={},
        )
        assert res.status_code == 200
        res = client.get(f"/api/me/checkins?day={day}", headers=ess)
        ck = [c for c in res.json() if c["kind"] == "in"]
        assert ck and ck[0]["method"] == "request"
        att = db_session.scalars(
            select(Attendance).where(
                Attendance.employee_id == emp["id"], Attendance.day == day
            )
        ).first()
        assert att and att.status == "present" and att.first_in == ts

        # 重复批准 → 400
        assert client.post(
            f"/api/attendance/requests/{req_id}/approve",
            headers=_hdr(admin), json={},
        ).status_code == 400

        # 撤销已批准的 → 补写的 checkin 删除,考勤重算为非 present,
        # 且 first_in/last_out/work_hours 一并清掉(不留已撤销的时刻)
        res = client.post(
            f"/api/me/attendance-requests/{req_id}/cancel", headers=ess
        )
        assert res.status_code == 200
        db_session.refresh(att)
        assert att.status in ("absent", "holiday")
        assert att.first_in is None and att.last_out is None
        assert att.work_hours is None

        # 驳回路径:不发 checkin
        ts2 = ts.replace(hour=18)
        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": ts2.isoformat(), "kind": "out", "reason": "忘签退"},
        )
        assert res.status_code == 201
        rid = res.json()["id"]
        res = client.post(
            f"/api/attendance/requests/{rid}/reject",
            headers=_hdr(admin), json={"note": "时间不对"},
        )
        assert res.status_code == 200
        res = client.get(f"/api/me/checkins?day={day}", headers=ess)
        assert not any(c["kind"] == "out" for c in res.json())

        # 撤销 pending → cancelled
        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": ts2.isoformat(), "kind": "out"},
        )
        rid2 = res.json()["id"]
        res = client.post(
            f"/api/me/attendance-requests/{rid2}/cancel", headers=ess
        )
        assert res.status_code == 200
        assert res.json()["status"] == "cancelled"

        # 超长事由:批准时补写的 checkin.note 是 VARCHAR(256),截断防 DataError
        ts3 = ts - timedelta(days=1)
        res = client.post(
            "/api/me/attendance-requests", headers=ess,
            json={"ts": ts3.isoformat(), "kind": "in", "reason": "长" * 300},
        )
        assert res.status_code == 201, res.text
        rid3 = res.json()["id"]
        res = client.post(
            f"/api/attendance/requests/{rid3}/approve",
            headers=_hdr(admin), json={},
        )
        assert res.status_code == 200, res.text
        ck3 = db_session.scalars(
            select(EmployeeCheckin).where(
                EmployeeCheckin.employee_id == emp["id"],
                EmployeeCheckin.ts == ts3,
            )
        ).first()
        assert ck3 and len(ck3.note) == 256
    finally:
        _cleanup(db_session, t)


def test_attendance_request_manager_scope(db_session):
    """用人经理只能批直属下属的补卡申请。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hm = _user(db_session, t, role="hiring_manager")
    client = TestClient(app)
    try:
        mgr_emp = _mk_emp(client, _hdr(admin), name="经理")
        res = client.post(
            f"/api/employees/{mgr_emp['id']}/account", headers=_hdr(admin),
            json={"email": hm.email},
        )
        assert res.status_code == 200
        sub_emp = _mk_emp(client, _hdr(admin), name="下属",
                          reports_to_id=mgr_emp["id"])
        other_emp = _mk_emp(client, _hdr(admin), name="外人")

        ts = (datetime.now() - timedelta(days=2)).replace(
            hour=9, minute=0, second=0, microsecond=0)

        # 给两个员工各绑 ESS 账号并提交补卡申请
        for emp in (sub_emp, other_emp):
            email = f"ess-{uuid.uuid4().hex[:6]}@example.com"
            acct = client.post(
                f"/api/employees/{emp['id']}/account", headers=_hdr(admin),
                json={"email": email},
            ).json()
            tok = client.post(
                "/api/auth/login",
                json={"email": email, "password": acct["password"]},
            ).json()["access_token"]
            res = client.post(
                "/api/me/attendance-requests",
                headers={"Authorization": f"Bearer {tok}"},
                json={"ts": ts.isoformat(), "kind": "in"},
            )
            assert res.status_code == 201, res.text
            emp["req_id"] = res.json()["id"]

        hm_hdr = _hdr(hm)
        # scope=team 只见直属下属
        res = client.get(
            "/api/attendance/requests?scope=team", headers=hm_hdr
        )
        names = {i["employee_name"] for i in res.json()["items"]}
        assert "下属" in names and "外人" not in names

        # 批外人 → 403;批下属 → 200
        assert client.post(
            f"/api/attendance/requests/{other_emp['req_id']}/approve",
            headers=hm_hdr, json={},
        ).status_code == 403
        assert client.post(
            f"/api/attendance/requests/{sub_emp['req_id']}/approve",
            headers=hm_hdr, json={},
        ).status_code == 200
    finally:
        _cleanup(db_session, t)


def test_overtime_request_chain(db_session):
    """加班 → 调休:审批后 upsert 当年额度 + 台账 +;员工端可花调休。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        comp = client.post(
            "/api/leave/types", headers=_hdr(admin),
            json={"name": "调休假", "annual_days": 0, "is_paid": True,
                  "is_comp": True},
        )
        assert comp.status_code == 201, comp.text
        comp = comp.json()
        normal = _mk_leave_type(client, _hdr(admin), name="年假")

        emp, ess = _mk_ess_employee(client, _hdr(admin))
        work_date = date.today() - timedelta(days=1)

        # 非调休类型 → 400
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": normal["id"], "work_date": str(work_date),
                  "hours": 4},
        )
        assert res.status_code == 400

        # 未来加班日期 → 400(事后申报,不能预提)
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"],
                  "work_date": str(date.today() + timedelta(days=3)),
                  "hours": 4},
        )
        assert res.status_code == 400

        # 4h / 8h = 0.5 天
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"], "work_date": str(work_date),
                  "hours": 4, "reason": "版本发布"},
        )
        assert res.status_code == 201, res.text
        assert res.json()["comp_days"] == 0.5
        rid = res.json()["id"]

        # 同一加班日在途申请去重 → 400;换个日期可以,测完撤回
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"], "work_date": str(work_date),
                  "hours": 2},
        )
        assert res.status_code == 400
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"],
                  "work_date": str(work_date - timedelta(days=1)),
                  "hours": 2},
        )
        assert res.status_code == 201
        client.post(
            f"/api/me/overtime-requests/{res.json()['id']}/cancel",
            headers=ess,
        )

        res = client.post(
            f"/api/overtime/requests/{rid}/approve",
            headers=_hdr(admin), json={},
        )
        assert res.status_code == 200

        # 额度:当年 allocation 0.5 天;台账 +0.5 ref_type=overtime
        res = client.get(
            f"/api/leave/allocations?employee_id={emp['id']}",
            headers=_hdr(admin),
        )
        allocs = [a for a in res.json()["items"]
                  if a["leave_type_id"] == comp["id"]]
        assert len(allocs) == 1 and allocs[0]["days"] == 0.5
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        ot = [e for e in res.json()["items"] if e["ref_type"] == "overtime"]
        assert len(ot) == 1 and ot[0]["delta_days"] == 0.5

        # 再批一单 16h → 2 天,累计 2.5
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"], "work_date": str(work_date),
                  "hours": 16},
        )
        rid2 = res.json()["id"]
        assert res.json()["comp_days"] == 2.0
        client.post(
            f"/api/overtime/requests/{rid2}/approve",
            headers=_hdr(admin), json={},
        )
        res = client.get("/api/me/leave-balances", headers=ess)
        bal = next(b for b in res.json() if b["leave_type_id"] == comp["id"])
        assert bal["allocated"] == 2.5 and bal["balance"] == 2.5

        # 员工可用调休假:请 1 个工作日并由 admin 代批(非自批)
        s, _ = _wk_span(1)
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": comp["id"], "start_date": str(s),
                  "end_date": str(s), "days": 1},
        )
        assert res.status_code == 201, res.text
        leave_id = res.json()["id"]
        res = client.post(
            f"/api/leave/applications/{leave_id}/approve",
            headers=_hdr(admin), json={},
        )
        assert res.status_code == 200

        # 调休已花 1 天时撤销 2 天加班 → 拒绝(回冲后余额会变负)
        res = client.post(
            f"/api/me/overtime-requests/{rid2}/cancel", headers=ess
        )
        assert res.status_code == 400

        # 把请假撤掉(用量吐回)→ 再撤加班放行,额度回冲 2.0,台账 -delta
        res = client.post(
            f"/api/me/leave-applications/{leave_id}/cancel", headers=ess
        )
        assert res.status_code == 200
        res = client.post(
            f"/api/me/overtime-requests/{rid2}/cancel", headers=ess
        )
        assert res.status_code == 200
        res = client.get("/api/me/leave-balances", headers=ess)
        bal = next(b for b in res.json() if b["leave_type_id"] == comp["id"])
        assert bal["allocated"] == 0.5
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        oc = [e for e in res.json()["items"]
              if e["ref_type"] == "overtime_cancel"]
        assert len(oc) == 1 and oc[0]["delta_days"] == -2.0

        # 驳回路径:不产额度
        res = client.post(
            "/api/me/overtime-requests", headers=ess,
            json={"leave_type_id": comp["id"], "work_date": str(work_date),
                  "hours": 2},
        )
        rid3 = res.json()["id"]
        client.post(
            f"/api/overtime/requests/{rid3}/reject",
            headers=_hdr(admin), json={},
        )
        res = client.get("/api/me/leave-balances", headers=ess)
        bal = next(b for b in res.json() if b["leave_type_id"] == comp["id"])
        assert bal["allocated"] == 0.5
    finally:
        _cleanup(db_session, t)


def test_leave_policy_chain(db_session):
    """假期政策:CRUD + assign 即发额度 + run 年度结转 + 幂等。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt_a = _mk_leave_type(client, _hdr(admin), name="年假", days=10)
        lt_b = _mk_leave_type(client, _hdr(admin), name="病假", days=5)
        emp, ess = _mk_ess_employee(client, _hdr(admin), name="员工甲")
        year = date.today().year

        res = client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "正式员工包",
                  "items": [{"leave_type_id": lt_a["id"], "days": 10},
                            {"leave_type_id": lt_b["id"], "days": 5}],
                  "carry_forward": 5},
        )
        assert res.status_code == 201, res.text
        pid = res.json()["id"]
        # 重名 → 409;坏类型 → 400
        assert client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "正式员工包",
                  "items": [{"leave_type_id": lt_a["id"], "days": 1}]},
        ).status_code == 409
        assert client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "坏包",
                  "items": [{"leave_type_id": "no-such", "days": 1}]},
        ).status_code == 400
        # 调休类型不能进政策包 — run 的 set-total 会把加班攒的额度清零
        comp = client.post(
            "/api/leave/types", headers=_hdr(admin),
            json={"name": "调休假", "is_comp": True},
        )
        assert comp.status_code == 201, comp.text
        assert client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "含调休的包",
                  "items": [{"leave_type_id": comp.json()["id"], "days": 0}]},
        ).status_code == 400
        # 已在政策包里的类型也不能改成调休(否则包里出现死配置)
        assert client.put(
            f"/api/leave/types/{lt_a['id']}", headers=_hdr(admin),
            json={"name": "年假", "annual_days": 10, "is_paid": True,
                  "is_comp": True},
        ).status_code == 400

        # assign → 立即按当年发放
        res = client.post(
            f"/api/leave/policies/{pid}/assign", headers=_hdr(admin),
            json={"employee_ids": [emp["id"]]},
        )
        assert res.status_code == 200 and res.json()["assigned"] == 1
        res = client.get(
            f"/api/leave/allocations?employee_id={emp['id']}",
            headers=_hdr(admin),
        )
        alloc = {a["leave_type_id"]: a["days"] for a in res.json()["items"]}
        assert alloc[lt_a["id"]] == 10 and alloc[lt_b["id"]] == 5
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        pr = [e for e in res.json()["items"] if e["ref_type"] == "policy_run"]
        assert len(pr) == 2

        # 幂等:同年再 run,天数与台账不变
        res = client.post(
            f"/api/leave/policies/{pid}/run", headers=_hdr(admin),
            json={"year": year},
        )
        assert res.json()["applied"] == 1
        res = client.get("/api/leave/ledger", headers=_hdr(admin))
        pr = [e for e in res.json()["items"] if e["ref_type"] == "policy_run"]
        assert len(pr) == 2

        # ---- 结转:补一条上年 allocation(10 天)+ 上年已批 3 天 ----
        client.post(
            "/api/leave/allocations", headers=_hdr(admin),
            json={"employee_id": emp["id"], "leave_type_id": lt_a["id"],
                  "year": year - 1, "days": 10},
        )
        # 上年第一个周一请 3 天(ESS 端发起,admin 代批 — 非自批)
        prev_mon = date(year - 1, 1, 1)
        prev_mon += timedelta(days=(7 - prev_mon.weekday()) % 7)
        res = client.post(
            "/api/me/leave-applications", headers=ess,
            json={"leave_type_id": lt_a["id"], "start_date": str(prev_mon),
                  "end_date": str(prev_mon + timedelta(days=2)),
                  "days": 3, "reason": "上年已休"},
        )
        assert res.status_code == 201, res.text
        client.post(
            f"/api/leave/applications/{res.json()['id']}/approve",
            headers=_hdr(admin), json={},
        )

        # 再 run 当年:carry = min(10-3, cap=5)=5 → target 15,补 +5
        res = client.post(
            f"/api/leave/policies/{pid}/run", headers=_hdr(admin),
            json={"year": year},
        )
        assert res.json()["applied"] == 1
        res = client.get(
            f"/api/leave/allocations?employee_id={emp['id']}",
            headers=_hdr(admin),
        )
        alloc = {a["leave_type_id"]: a["days"] for a in res.json()["items"]
                 if a["year"] == year}
        assert alloc[lt_a["id"]] == 15

        # 有绑定时不能删;unassign 后可删
        assert client.delete(
            f"/api/leave/policies/{pid}", headers=_hdr(admin)
        ).status_code == 400
        client.post(
            f"/api/leave/policies/{pid}/unassign", headers=_hdr(admin),
            json={"employee_ids": [emp["id"]]},
        )
        assert client.delete(
            f"/api/leave/policies/{pid}", headers=_hdr(admin)
        ).status_code == 204
    finally:
        _cleanup(db_session, t)


def test_policy_run_skips_comp_type(db_session):
    """防御路径:政策项的类型被直改成调休后,run 跳过该类型。

    API 层面已双保险(政策写时拒收调休类型 + 政策引用的类型禁止改成调休),
    这里是数据层兜底 — 即使出现"调休类型在政策包里"的存量状态,
    run 也不能把加班攒的额度按政策目标清零。
    """
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        lt = _mk_leave_type(client, _hdr(admin), name="弹性假")
        emp, _ess = _mk_ess_employee(client, _hdr(admin))
        year = date.today().year
        res = client.post(
            "/api/leave/policies", headers=_hdr(admin),
            json={"name": "弹性包",
                  "items": [{"leave_type_id": lt["id"], "days": 3}]},
        )
        assert res.status_code == 201, res.text
        pid = res.json()["id"]
        client.post(
            f"/api/leave/policies/{pid}/assign", headers=_hdr(admin),
            json={"employee_ids": [emp["id"]]},
        )
        # 模拟加班入账过的存量:alloc 7 ≠ 政策目标 3;类型直改成调休
        alloc = db_session.scalars(
            select(LeaveAllocation).where(
                LeaveAllocation.employee_id == emp["id"],
                LeaveAllocation.leave_type_id == lt["id"],
            )
        ).first()
        alloc.days = 7
        db_session.get(LeaveType, lt["id"]).is_comp = True
        db_session.commit()

        res = client.post(
            f"/api/leave/policies/{pid}/run", headers=_hdr(admin),
            json={"year": year},
        )
        assert res.status_code == 200
        db_session.refresh(alloc)
        assert alloc.days == 7
    finally:
        _cleanup(db_session, t)
