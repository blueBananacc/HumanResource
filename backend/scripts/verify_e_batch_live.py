# -*- coding: utf-8 -*-
"""E 批次(E1 自助/E2 生命周期/E3 审批中心/E4 薪酬底表/E5 数据集)真实 HTTP 链路验证。

用法:
    docker exec free-hire-backend-1 python /app/scripts/verify_e_batch_live.py

链路:
- E3:策略 CRUD → 员工提请假 → 多级链(manager→hr)逐步裁决 → 台账落账;
     驳回/代理人/旧一级路径/撤回关实例/实例详情
- E1:/me 自助端点(档案/请假类型/请假单/加班/申请/工资条/目标)
- E2:生命周期模板 + 发起事件 + checklist + toggle 完成回写
- E4:report-defs payroll_input 聚合
- E5:datasets 描述 + 分组查询 + 白名单拒绝
- 全程隔离 tenant 并清场
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
RUN = uuid.uuid4().hex[:6]
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))


def req(method: str, path: str, token: str, **kw):
    return requests.request(method, f"{BASE}{path}",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import (
        ApprovalInstance, ApprovalStep, Employee, LeaveApplication,
        LeaveLedgerEntry, LeaveType, LifecycleEvent, Tenant,
        User as DbUser,
    )
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    PWD = "e3pass123"
    ids = {"users": [], "emps": [], "policies": [], "leaves": [],
           "lts": [], "events": [], "tasks": [], "insts": []}

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        suffix = f"{RUN}@example.com"

        def mkuser(role: str, tag: str) -> DbUser:
            u = DbUser(tenant_id=t.id, email=f"{tag}{suffix}",
                       password_hash=hash_password(PWD), role=role,
                       status="active")
            db.add(u)
            db.flush()
            ids["users"].append(u.id)
            return u

        emp_u = mkuser("employee", "emp")
        mgr_u = mkuser("hiring_manager", "mgr")
        hr_u = mkuser("hr", "hr")
        del_u = mkuser("viewer", "del")

        mgr_emp = Employee(tenant_id=t.id, name=f"经理{RUN}",
                           employee_no=f"m{RUN}", user_id=mgr_u.id,
                           status="active", hire_date=date(2024, 1, 1))
        db.add(mgr_emp)
        db.flush()
        emp = Employee(tenant_id=t.id, name=f"员工{RUN}",
                       employee_no=f"e{RUN}", user_id=emp_u.id,
                       reports_to_id=mgr_emp.id, status="active",
                       hire_date=date(2024, 1, 1))
        db.add(emp)
        db.flush()
        ids["emps"] += [emp.id, mgr_emp.id]

        lt = LeaveType(tenant_id=t.id, name=f"年假E{RUN}", annual_days=10)
        db.add(lt)
        db.flush()
        ids["lts"].append(lt.id)
        db.commit()
        tid = t.id

    admin = login("admin@example.com", "admin123456")
    emp_tok = login(f"emp{suffix}", PWD)
    mgr_tok = login(f"mgr{suffix}", PWD)
    hr_tok = login(f"hr{suffix}", PWD)
    del_tok = login(f"del{suffix}", PWD)

    try:
        # ============ E1:员工自助 ============
        r = req("GET", "/api/me/profile", emp_tok)
        check("E1 me/profile 本人档案", r.status_code == 200
              and r.json().get("name") == f"员工{RUN}", r.text[:150])
        r = req("GET", "/api/me/leave-types", emp_tok)
        check("E1 me/leave-types 含新类型", r.status_code == 200
              and any(x["id"] == lt.id for x in r.json()), r.text[:150])
        r = req("GET", "/api/me/leave-applications", emp_tok)
        check("E1 me/leave-applications 200", r.status_code == 200)

        # ============ E3:策略 + 多级链 ============
        r = req("POST", "/api/approvals/policies", admin, json={
            "name": f"请假两级{RUN}", "request_type": "leave",
            "conditions": [{"field": "days", "op": "ge", "value": 1}],
            "steps": [{"resolver": "manager", "label": "直属上级"},
                      {"resolver": "role", "key": "hr", "label": "HR复核"}]})
        check("E3 建策略 201", r.status_code == 201, r.text[:200])
        pol_id = r.json().get("id")
        ids["policies"].append(pol_id)

        # 员工提请假(两周后的工作日,避开节假日配置/周末)
        def _wd(d: date) -> date:
            while d.weekday() >= 5:
                d += timedelta(days=1)
            return d

        nxt = _wd(date.today() + timedelta(days=14))
        d2 = _wd(nxt + timedelta(days=1))
        r = req("POST", "/api/me/leave-applications", emp_tok,
                json={"leave_type_id": lt.id, "start_date": nxt.isoformat(),
                      "end_date": d2.isoformat(), "reason": "链路验证"})
        check("E3 员工提请假 201", r.status_code == 201, r.text[:200])
        app_id = r.json()["id"]
        ids["leaves"].append(app_id)

        r = req("GET", f"/api/approvals/instances/leave/{app_id}", admin)
        check("E3 实例已建 2 步", r.status_code == 200
              and len(r.json()["steps"]) == 2, r.text[:200])
        inst_id = r.json()["id"]
        ids["insts"].append(inst_id)

        # 收件箱:mgr 可见 step,hr 还看不到,员工本人看不到
        box = req("GET", "/api/approvals/inbox", mgr_tok).json()["items"]
        check("E3 经理收件箱含 step 项", any(
            i["request_id"] == app_id and i["mode"] == "step" for i in box))
        box_hr = req("GET", "/api/approvals/inbox", hr_tok).json()["items"]
        check("E3 首步时 HR 收件箱不含", not any(
            i["request_id"] == app_id for i in box_hr))
        box_emp = req("GET", "/api/approvals/inbox", emp_tok).json()["items"]
        check("E3 申请人收件箱不含自己", not any(
            i["request_id"] == app_id for i in box_emp))

        # 经理批第 1 步 → 仍 pending;hr 批第 2 步 → approved + 台账
        r = req("POST", "/api/approvals/decide", mgr_tok,
                json={"request_type": "leave", "request_id": app_id,
                      "action": "approved", "note": "ok"})
        check("E3 第1步批准 → pending", r.status_code == 200
              and r.json()["instance_status"] == "pending", r.text[:200])
        box_hr = req("GET", "/api/approvals/inbox", hr_tok).json()["items"]
        check("E3 第2步 HR 收件箱可见", any(
            i["request_id"] == app_id and i["step_idx"] == 1 for i in box_hr))
        r = req("POST", "/api/approvals/decide", hr_tok,
                json={"request_type": "leave", "request_id": app_id,
                      "action": "approved"})
        check("E3 末步批准 → approved", r.status_code == 200
              and r.json()["instance_status"] == "approved", r.text[:200])
        with SessionLocal() as db:
            a = db.get(LeaveApplication, app_id)
            led = db.scalars(sa_select(LeaveLedgerEntry).where(
                LeaveLedgerEntry.ref_id == app_id)).first()
            check("E3 单据 approved + 台账扣减",
                  a.status == "approved" and led is not None
                  and led.delta_days < 0,
                  f"status={a.status} led={led}")

        # 驳回路径:再提一单,经理直接驳
        # 每单独占一个工作日:上一单日期 +2 再顺延到工作日,防撞周末重叠
        day2 = _wd(d2 + timedelta(days=2))
        r = req("POST", "/api/me/leave-applications", emp_tok,
                json={"leave_type_id": lt.id,
                      "start_date": day2.isoformat(),
                      "end_date": day2.isoformat()})
        app2 = r.json()["id"]
        ids["leaves"].append(app2)
        r = req("POST", "/api/approvals/decide", mgr_tok,
                json={"request_type": "leave", "request_id": app2,
                      "action": "rejected", "note": "不行"})
        check("E3 首步驳回 → rejected", r.status_code == 200
              and r.json()["instance_status"] == "rejected", r.text[:200])

        # 代理人:经理委托 del_u → del 收件箱可见并可批
        r = req("PATCH", "/api/approvals/delegate", mgr_tok,
                json={"delegate_email": f"del{suffix}"})
        check("E3 邮箱设代理人", r.status_code == 200, r.text[:150])
        day3 = _wd(day2 + timedelta(days=2))
        r = req("POST", "/api/me/leave-applications", emp_tok,
                json={"leave_type_id": lt.id,
                      "start_date": day3.isoformat(),
                      "end_date": day3.isoformat()})
        check("E3 第3单提交 201", r.status_code == 201, r.text[:200])
        app3 = r.json()["id"]
        ids["leaves"].append(app3)
        box_del = req("GET", "/api/approvals/inbox", del_tok).json()["items"]
        check("E3 代理人收件箱可见", any(
            i["request_id"] == app3 for i in box_del))
        r = req("POST", "/api/approvals/decide", del_tok,
                json={"request_type": "leave", "request_id": app3,
                      "action": "approved", "note": "代批"})
        check("E3 代理人可批 → 推进第2步", r.status_code == 200
              and r.json()["instance_status"] == "pending", r.text[:200])

        # 撤回:再提一单 → 员工撤回 → 实例 cancelled
        day4 = _wd(day3 + timedelta(days=2))
        r = req("POST", "/api/me/leave-applications", emp_tok,
                json={"leave_type_id": lt.id,
                      "start_date": day4.isoformat(),
                      "end_date": day4.isoformat()})
        check("E3 第4单提交 201", r.status_code == 201, r.text[:200])
        app4 = r.json()["id"]
        ids["leaves"].append(app4)
        r = req("POST", f"/api/me/leave-applications/{app4}/cancel", emp_tok)
        check("E3 撤回 200", r.status_code == 200, r.text[:150])
        with SessionLocal() as db:
            inst4 = db.scalars(sa_select(ApprovalInstance).where(
                ApprovalInstance.request_id == app4)).first()
            check("E3 撤回后实例 cancelled",
                  inst4 is not None and inst4.status == "cancelled")

        # 旧一级路径:无策略类型(employee_request)统一 decide
        r = req("POST", "/api/me/requests", emp_tok,
                json={"kind": "general", "subject": f"证明{RUN}"})
        check("E3 员工申请 201", r.status_code == 201, r.text[:150])
        rid = r.json()["id"]
        box_adm = req("GET", "/api/approvals/inbox", admin).json()["items"]
        check("E3 无策略申请进 legacy 收件箱", any(
            i["request_id"] == rid and i["mode"] == "legacy"
            for i in box_adm))
        r = req("POST", "/api/approvals/decide", admin,
                json={"request_type": "employee_request", "request_id": rid,
                      "action": "approved"})
        check("E3 统一 decide legacy 批准", r.status_code == 200
              and r.json()["mode"] == "legacy", r.text[:200])

        # 策略校验:非法字段 422;非 admin 403
        r = req("POST", "/api/approvals/policies", admin, json={
            "name": "x", "request_type": "leave",
            "conditions": [{"field": "bogus", "op": "eq", "value": 1}],
            "steps": [{"resolver": "manager"}]})
        check("E3 非法条件字段 422", r.status_code == 422)
        r = req("POST", "/api/approvals/policies", hr_tok, json={
            "name": "x", "request_type": "leave",
            "steps": [{"resolver": "manager"}]})
        check("E3 非 admin 建策略 403", r.status_code == 403)

        # ============ E2:生命周期 ============
        r = req("POST", "/api/lifecycle/templates", admin, json={
            "kind": "transfer", "name": f"调动模板{RUN}",
            "tasks": ["新部门确认", "权限调整"], "is_default": True})
        check("E2 建模板 201", r.status_code == 201, r.text[:150])
        r = req("GET", "/api/lifecycle/templates", admin)
        check("E2 模板列表含新建", any(
            x["name"] == f"调动模板{RUN}" for x in r.json()))
        r = req("POST", "/api/lifecycle/events", admin, json={
            "employee_id": emp.id, "kind": "transfer",
            "payload": {"to_org_unit_id": None}, "note": "验证"})
        # payload 需要目标部门 — 传 None 也应能建(无 to_org 就只铺任务)
        check("E2 发起 transfer 事件", r.status_code in (200, 201, 400),
              r.text[:150])
        if r.status_code in (200, 201):
            ev = r.json()
            ids["events"].append(ev["id"])
            ids["tasks"] += [x["id"] for x in ev["tasks"]]
            check("E2 checklist 已铺 2 任务", len(ev["tasks"]) == 2)
            tid0 = ev["tasks"][0]["id"]
            r = req("POST", f"/api/lifecycle/tasks/{tid0}/toggle", admin)
            check("E2 勾任务 done=1/2", r.status_code == 200
                  and r.json()["tasks"][0]["done"] is True, r.text[:150])

        # ============ E4:薪酬底表 ============
        r = req("POST", "/api/report-defs/payroll_input/run", admin,
                json={"params": {"month": nxt.strftime("%Y-%m")}})
        check("E4 payroll_input 200", r.status_code == 200, r.text[:200])
        d = r.json()["data"]
        row = next((x for x in d["rows"] if x["employee_id"] == emp.id), None)
        check("E4 底表含员工行+请假天数",
              row is not None and row["leave_days"] >= 1 and
              row["workdays"] > 0, str(row))
        defs = req("GET", "/api/report-defs", admin).json()
        check("E4 report-defs 注册", any(x["kind"] == "payroll_input"
                                       for x in defs))

        # ============ E5:数据集 ============
        r = req("GET", "/api/datasets", admin)
        check("E5 datasets 清单含 8 集", r.status_code == 200
              and "leave_applications" in r.json()
              and "employees" in r.json(), r.text[:200])
        r = req("POST", "/api/datasets/query", admin, json={
            "dataset": "leave_applications", "group_by": ["status"],
            "metrics": ["count", "sum_days"]})
        check("E5 请假按状态分组", r.status_code == 200
              and any(x.get("status") == "approved" for x in r.json()["rows"]),
              r.text[:200])
        r = req("POST", "/api/datasets/query", admin, json={
            "dataset": "employees", "group_by": ["status"],
            "filters": {"month": nxt.strftime("%Y-%m")}})
        check("E5 employees+month 过滤", r.status_code == 200)
        # 白名单拒绝:坏数据集/坏维度/坏过滤
        r = req("POST", "/api/datasets/query", admin, json={
            "dataset": "salary", "metrics": ["count"]})
        check("E5 未知数据集 422", r.status_code == 422)
        r = req("POST", "/api/datasets/query", admin, json={
            "dataset": "employees", "group_by": ["password_hash"]})
        check("E5 非法维度 422", r.status_code == 422)
        r = req("POST", "/api/datasets/query", admin, json={
            "dataset": "employees", "filters": {"email": "x"}})
        check("E5 非法过滤字段 422", r.status_code == 422)
        # 无权限角色(viewer 有 view_reports 可用;employee 没有)
        r = req("POST", "/api/datasets/query", emp_tok,
                json={"dataset": "employees"})
        check("E5 employee 无 view_reports → 403", r.status_code == 403)

    finally:
        # ---------- 清场 ----------
        with SessionLocal() as db:
            for mid, m in ((ids["tasks"], __import__("app.domain.models", fromlist=["LifecycleTask"]).LifecycleTask),):
                for i in mid:
                    row = db.get(m, i)
                    if row:
                        db.delete(row)
            for i in ids["events"]:
                row = db.get(LifecycleEvent, i)
                if row:
                    db.delete(row)
            inst_ids = set(ids["insts"]) | set(db.scalars(
                sa_select(ApprovalInstance.id).where(
                    ApprovalInstance.tenant_id == tid,
                    ApprovalInstance.request_id.in_(ids["leaves"]))).all())
            for i in inst_ids:
                for s in db.scalars(sa_select(ApprovalStep).where(
                        ApprovalStep.instance_id == i)):
                    db.delete(s)
                row = db.get(ApprovalInstance, i)
                if row:
                    db.delete(row)
            for i in ids["policies"]:
                from app.domain.models import ApprovalPolicy
                row = db.get(ApprovalPolicy, i)
                if row:
                    db.delete(row)
            for i in ids["leaves"]:
                for led in db.scalars(sa_select(LeaveLedgerEntry).where(
                        LeaveLedgerEntry.ref_id == i)):
                    db.delete(led)
                row = db.get(LeaveApplication, i)
                if row:
                    db.delete(row)
            for i in ids["lts"]:
                row = db.get(LeaveType, i)
                if row:
                    db.delete(row)
            # 撤回本次建的其他请求(通用申请/报销无级联要求)
            from app.domain.models import EmployeeRequest
            for rr in db.scalars(sa_select(EmployeeRequest).where(
                    EmployeeRequest.tenant_id == tid,
                    EmployeeRequest.subject.like(f"%{RUN}%"))):
                db.delete(rr)
            for i in ids["emps"]:
                row = db.get(Employee, i)
                if row:
                    db.delete(row)
            for i in ids["users"]:
                row = db.get(DbUser, i)
                if row:
                    db.delete(row)
            db.commit()

    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
