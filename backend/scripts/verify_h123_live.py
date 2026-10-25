# -*- coding: utf-8 -*-
"""H1-H3 人事助手场景 —— 真实 HTTP 链路验证(容器内运行)。

用法:
    docker exec free-hire-backend-1 python /app/scripts/verify_h123_live.py

链路:
- H1:bulk-excuse 考勤异常 / day 过滤 / hr_movements 报表 / 假期余额
- H2:preview-org-changes → apply-org-changes 组织调整事务
- H3:shifts 建班 → generate → assignments → 请假缺口 → auto-cover 替班
- 全部种子数据收尾清理
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
        Attendance, AttendanceFlag, Employee, LeaveApplication, LeaveType,
        LifecycleEvent, OrgUnit, Shift, ShiftAssignment, Tenant,
        User as DbUser,
    )
    from app.infra.db import SessionLocal

    admin = login("admin@example.com", "admin123456")
    today = date.today()
    d1 = today - timedelta(days=1)
    ids = {"emps": [], "orgs": [], "flags": [], "atts": [], "leaves": [],
           "lts": [], "events": [], "shift": None, "assigns": []}

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()

        # ---------- 种子:组织 + 员工 ----------
        org_old = OrgUnit(tenant_id=t.id, name=f"旧部门{RUN}")
        db.add(org_old)
        db.flush()
        ids["orgs"].append(org_old.id)
        e1 = Employee(tenant_id=t.id, name=f"员工甲{RUN}",
                      employee_no=f"h1{RUN}1", org_unit_id=org_old.id,
                      status="active", hire_date=today - timedelta(days=40))
        e2 = Employee(tenant_id=t.id, name=f"员工乙{RUN}",
                      employee_no=f"h1{RUN}2", org_unit_id=org_old.id,
                      status="active", hire_date=today - timedelta(days=40))
        e3 = Employee(tenant_id=t.id, name=f"请假人{RUN}",
                      employee_no=f"h1{RUN}3", org_unit_id=org_old.id,
                      status="active", hire_date=today - timedelta(days=40))
        db.add_all([e1, e2, e3])
        db.flush()
        ids["emps"] += [e1.id, e2.id, e3.id]

        # ---------- H1:考勤异常 + 假期余额 ----------
        for emp, kd in ((e1, "late_in"), (e2, "early_out")):
            att = Attendance(tenant_id=t.id, employee_id=emp.id, day=d1,
                             status="present")
            db.add(att)
            db.flush()
            ids["atts"].append(att.id)
            f = AttendanceFlag(tenant_id=t.id, attendance_id=att.id,
                               employee_id=emp.id, day=d1, kind=kd, minutes=12)
            db.add(f)
            db.flush()
            ids["flags"].append(f.id)
        lt = LeaveType(tenant_id=t.id, name=f"年假{RUN}", annual_days=10)
        db.add(lt)
        db.flush()
        ids["lts"].append(lt.id)
        from app.domain.models import LeaveAllocation
        db.add(LeaveAllocation(tenant_id=t.id, employee_id=e1.id,
                               leave_type_id=lt.id, year=today.year, days=10))
        la = LeaveApplication(tenant_id=t.id, employee_id=e3.id,
                              leave_type_id=lt.id, start_date=d1, end_date=today,
                              days=2, status="approved")
        db.add(la)
        db.flush()
        ids["leaves"].append(la.id)
        db.commit()

        r = req("GET", f"/api/attendance/flags?day={d1}&excused=false", admin)
        got = {i["id"] for i in r.json()["items"]}
        check("H1 flags day 过滤命中 2 条未核", ids["flags"] and
              set(ids["flags"]) <= got, str(got))

        r = req("POST", "/api/attendance/flags/bulk-excuse", admin,
                json={"flag_ids": ids["flags"], "note": "链路核对"})
        check("H1 bulk-excuse excused=2", r.status_code == 200
              and r.json()["excused"] == 2, r.text[:200])
        r = req("POST", "/api/attendance/flags/bulk-excuse", admin,
                json={"flag_ids": ids["flags"]})
        check("H1 bulk-excuse 幂等 skipped=2", r.json()["skipped"] == 2)

        r = req("GET", f"/api/leave/balances?employee_id={e1.id}", admin)
        bal = r.json()
        check("H1 假期余额 allocated=10 used=0 balance=10",
              r.status_code == 200 and len(bal) == 1
              and bal[0]["balance"] == 10, str(bal))

        # ---------- H1:hr_movements 报表 ----------
        leav = LifecycleEvent(tenant_id=t.id, employee_id=e2.id, kind="leave",
                              effective_date=today, status="completed")
        db.add(leav)
        db.flush()
        ids["events"].append(leav.id)
        db.commit()
        r = req("POST", "/api/report-defs/hr_movements/run", admin,
                json={"params": {"month": today.strftime("%Y-%m")}})
        d = r.json()["data"]
        check("H1 hr_movements departures>=1 且含离职事件",
              d["departures"] >= 1 and any(i["kind"] == "leave" for i in d["items"]),
              str(d)[:200])
        check("H1 hr_movements KPI 齐(net/headcount/rate)",
              "net_change" in d and "departure_rate" in d and "by_org" in d)

        # ---------- H2:组织调整 ----------
        changes = [
            {"type": "create_org_unit", "name": f"新组{RUN}", "parent": f"旧部门{RUN}"},
            {"type": "transfer_employee", "employee": f"员工甲{RUN}",
             "to_org": f"新组{RUN}", "new_manager": f"员工乙{RUN}"},
        ]
        r = req("POST", "/api/org/preview-changes", admin,
                json={"changes": changes})
        body = r.json()
        check("H2 preview ok + 2 变更 + 上级警告", body["ok"]
              and len(body["changes"]) == 2, str(body)[:300])
        bad = req("POST", "/api/org/preview-changes", admin,
                  json={"changes": [{"type": "move_org_unit",
                                     "org": f"旧部门{RUN}",
                                     "new_parent": f"新组{RUN}"}]})
        check("H2 preview 环检测(父挂子)", bad.json()["ok"] is False,
              str(bad.json())[:200])

        r = req("POST", "/api/org/apply-changes", admin,
                json={"changes": changes})
        check("H2 apply applied=2", r.status_code == 200
              and r.json()["applied"] == 2, r.text[:200])
        db.refresh(e1)
        new_org = db.scalars(sa_select(OrgUnit).where(
            OrgUnit.tenant_id == t.id, OrgUnit.name == f"新组{RUN}")).first()
        ids["orgs"].append(new_org.id)
        check("H2 员工已调入新组+上级已设", e1.org_unit_id == new_org.id
              and e1.reports_to_id == e2.id)
        ev = db.scalars(sa_select(LifecycleEvent).where(
            LifecycleEvent.employee_id == e1.id,
            LifecycleEvent.kind == "transfer")).first()
        ids["events"].append(ev.id)
        check("H2 transfer lifecycle 事件已写", ev is not None)

        # ---------- H3:排班 ----------
        r = req("POST", "/api/shifts", admin,
                json={"name": f"早班{RUN}", "start_min": 540, "end_min": 1020,
                      "required": 1})
        check("H3 建班次 201", r.status_code == 201, r.text[:200])
        sid = r.json()["id"]
        ids["shift"] = sid

        month = today.strftime("%Y-%m")
        r = req("POST", "/api/shifts/generate", admin,
                json={"month": month, "dry_run": True})
        check("H3 generate dry_run 预览不落库", r.status_code == 200
              and r.json()["planned"] > 0, r.text[:200])
        n_plan = r.json()["planned"]
        r = req("POST", "/api/shifts/generate", admin,
                json={"month": month, "dry_run": False})
        check("H3 generate 落库 planned>0", r.json()["planned"] == n_plan,
              r.text[:200])
        db.commit()
        ids["assigns"] = [a.id for a in db.scalars(
            sa_select(ShiftAssignment).where(
                ShiftAssignment.tenant_id == t.id,
                ShiftAssignment.shift_id == sid)).all()]
        check("H3 assignments 已入库", len(ids["assigns"]) == n_plan)

        r = req("GET", f"/api/shifts/assignments?month={month}&limit=200", admin)
        check("H3 assignments 列表返回", r.status_code == 200
              and r.json()["total"] >= n_plan, r.text[:200])

        # 请假人 e3 在 d1-today 请假 → 若有其排班则为待替班
        r = req("GET", f"/api/shifts/gaps?month={month}", admin)
        gaps = r.json()["gaps"]
        check("H3 gaps 接口返回结构", r.status_code == 200
              and isinstance(gaps, list))
        r = req("POST", "/api/shifts/auto-cover", admin,
                json={"month": month, "dry_run": True})
        check("H3 auto-cover dry_run 返回三段", r.status_code == 200
              and all(k in r.json() for k in ("covered", "filled", "remaining")),
              r.text[:200])
        r = req("POST", "/api/shifts/auto-cover", admin,
                json={"month": month})
        check("H3 auto-cover 执行成功", r.status_code == 200, r.text[:200])
        # 替班把 assignment 换人了 → 重新收集 id(行可能被重用)
        db.commit()
        ids["assigns"] = [a.id for a in db.scalars(
            sa_select(ShiftAssignment).where(
                ShiftAssignment.tenant_id == t.id,
                ShiftAssignment.shift_id == sid)).all()]

        # ---------- 清理 ----------
        db.query(ShiftAssignment).filter(
            ShiftAssignment.tenant_id == t.id).delete(synchronize_session=False)
        db.query(Shift).filter(Shift.id == sid).delete(synchronize_session=False)
        db.query(AttendanceFlag).filter(
            AttendanceFlag.id.in_(ids["flags"])).delete(synchronize_session=False)
        db.query(Attendance).filter(
            Attendance.id.in_(ids["atts"])).delete(synchronize_session=False)
        db.query(LeaveApplication).filter(
            LeaveApplication.id.in_(ids["leaves"])).delete(synchronize_session=False)
        db.query(LeaveAllocation).filter(
            LeaveAllocation.employee_id.in_(ids["emps"])).delete(
            synchronize_session=False)
        db.query(LeaveType).filter(
            LeaveType.id.in_(ids["lts"])).delete(synchronize_session=False)
        db.query(LifecycleEvent).filter(
            LifecycleEvent.id.in_(ids["events"])).delete(synchronize_session=False)
        db.query(Employee).filter(
            Employee.id.in_(ids["emps"])).delete(synchronize_session=False)
        db.query(OrgUnit).filter(
            OrgUnit.id.in_(ids["orgs"])).delete(synchronize_session=False)
        db.commit()
        check("清理完成", True)

    print(f"\n=> {len(PASS)} PASS / {len(FAIL)} FAIL")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
