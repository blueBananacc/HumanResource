"""T7 事件自动化 + 考勤深化 — 真实 HTTP 链路验证(容器内运行)。

用法:
    docker compose exec backend python scripts/verify_t7_live.py

覆盖:
- automations schema / CRUD / 白名单校验(未知实体、字段、算子、动作 400)
- 触发链:POST /api/jobs → 审计 → dispatch → AutomationRun + 站内通知(模板渲染)
- 条件不满足不触发;is_active=false 不触发;run 幂等(一事件一 run)
- PATCH 启停、DELETE、列表过滤、执行记录分页
- 权限:employee 角色访问 /api/automations → 403
- 租户隔离:他租户规则不可见/不可改(404)
- 考勤:policy PUT roundtrip、work-calendar work_start/end roundtrip
- DB 造打卡(迟到/早退)→ POST /api/attendance/summarize → flags
- flags month/kind/excused 过滤;excuse/unexcuse + 审计
- monthly-summary 计数;excused 的 flag 不计入
- ESS /api/me/attendance-flags 可见自己
- IP 白名单:开启+CIDR 不匹配 → checkin 403;匹配 → 201;审计留 denied

退出码 0 = 全过。
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
RUN = uuid.uuid4().hex[:6]

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))


def req(method: str, path: str, token: str, **kw):
    return requests.request(
        method, f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    admin = login("admin@example.com", "admin123456")

    # ============================================================ 自动化
    # ---------- 1. schema ----------
    r = req("GET", "/api/automations/schema", admin)
    sch = r.json() if r.status_code == 200 else {}
    ents = set(sch.get("entities", {}).keys()) if isinstance(sch.get("entities"), dict) else {
        e.get("key") for e in sch.get("entities", [])}
    check("schema 含注册实体", r.status_code == 200 and
          {"job", "employee", "leave_application"} <= ents, r.text[:200])

    # ---------- 2. 校验拒绝 ----------
    bad = [
        ({"entity_type": "nope", "trigger": "create", "recipients": [{"kind": "actor"}],
          "title_tpl": "x"}, "未知实体 400"),
        ({"entity_type": "job", "trigger": "Warp Up", "recipients": [{"kind": "actor"}],
          "title_tpl": "x"}, "非法 trigger 形态 400"),
        ({"entity_type": "job", "trigger": "status;published", "recipients": [{"kind": "actor"}],
          "title_tpl": "x"}, "trigger 非法字符 400"),
        ({"entity_type": "job", "trigger": "create",
          "conditions": [{"field": "secret_col", "op": "eq", "value": 1}],
          "recipients": [{"kind": "actor"}], "title_tpl": "x"}, "未知条件字段 400"),
        ({"entity_type": "job", "trigger": "create",
          "conditions": [{"field": "status", "op": "regex", "value": ".*"}],
          "recipients": [{"kind": "actor"}], "title_tpl": "x"}, "未知算子 400"),
        ({"entity_type": "job", "trigger": "create", "recipients": [],
          "title_tpl": "x"}, "空收件人 400"),
    ]
    for payload, name in bad:
        p = {"name": f"bad-{RUN}", "action": "notify", "is_active": True, **payload}
        r = req("POST", "/api/automations", admin, json=p)
        check(name, r.status_code in (400, 422), f"status={r.status_code}")

    # 非三件套但合法的审计动作名(开放词表,精确匹配)
    r = req("POST", "/api/automations", admin, json={
        "name": f"t7-approve-{RUN}", "entity_type": "leave_application",
        "trigger": "approve", "action": "notify",
        "recipients": [{"kind": "actor"}], "title_tpl": "x",
    })
    approve_rid = r.json()["id"] if r.status_code == 201 else None
    check("自定义动作 trigger(approve) 201", r.status_code == 201, r.text[:200])
    if approve_rid:
        req("DELETE", f"/api/automations/{approve_rid}", admin)

    # ---------- 3. 建规则:job create → notify actor ----------
    rule_body = {
        "name": f"T7岗位通知{RUN}", "entity_type": "job", "trigger": "create",
        "action": "notify", "recipients": [{"kind": "actor"}],
        "conditions": [{"field": "title", "op": "contains", "value": RUN}],
        "title_tpl": f"岗位创建 {{{{title}}}}", "body_tpl": "操作人 {{actor}}",
        "is_active": True,
    }
    r = req("POST", "/api/automations", admin, json=rule_body)
    check("创建规则 201", r.status_code == 201, r.text[:200])
    rid = r.json()["id"] if r.status_code == 201 else None

    # ---------- 4. 触发:建岗位(标题含 RUN 命中条件)----------
    job_title = f"T7自动岗{RUN}"
    r = req("POST", "/api/jobs/", admin, json={"title": job_title, "level": "intermediate"})
    check("触发用岗位创建", r.status_code == 201, r.text[:200])
    job_id = r.json()["id"] if r.status_code == 201 else None

    # ---------- 5. run 记录 ----------
    r = req("GET", f"/api/automations/{rid}/runs", admin)
    runs = r.json().get("items", []) if r.status_code == 200 else []
    ok_run = next((x for x in runs if x["entity_id"] == job_id), None)
    check("run 记录生成且 status=ok", ok_run is not None and ok_run["status"] == "ok",
          r.text[:200])
    check("run payload 模板渲染", ok_run is not None and
          job_title in str(ok_run.get("payload", {}).get("title", "")),
          str(ok_run)[:200] if ok_run else "")
    check("run 幂等(同一事件仅一条)", len([x for x in runs if x["entity_id"] == job_id]) == 1)

    # ---------- 6. 站内通知送达 ----------
    r = req("GET", "/api/me/notifications", admin)
    notes = r.json() if r.status_code == 200 else []
    if isinstance(notes, dict):
        notes = notes.get("items", [])
    check("actor 收到自动化通知", any(
        n.get("kind") == "automation" and job_title in (n.get("title") or "")
        for n in notes), str(notes)[:300])

    # ---------- 7. 条件不满足不触发 ----------
    r = req("POST", "/api/jobs/", admin,
            json={"title": f"不含标记的岗位-{uuid.uuid4().hex[:4]}", "level": "junior"})
    miss_job = r.json()["id"] if r.status_code == 201 else None
    r = req("GET", f"/api/automations/{rid}/runs", admin)
    runs = r.json().get("items", []) if r.status_code == 200 else []
    check("条件不满足不触发", not any(x["entity_id"] == miss_job for x in runs))

    # ---------- 8. 停用后不触发 ----------
    r = req("PATCH", f"/api/automations/{rid}", admin, json={"is_active": False})
    check("PATCH 停用", r.status_code == 200 and r.json()["is_active"] is False)
    r = req("POST", "/api/jobs/", admin,
            json={"title": f"停用期岗位{RUN}", "level": "senior"})
    off_job = r.json()["id"] if r.status_code == 201 else None
    r = req("GET", f"/api/automations/{rid}/runs", admin)
    runs = r.json().get("items", []) if r.status_code == 200 else []
    check("停用规则不触发", not any(x["entity_id"] == off_job for x in runs))
    req("PATCH", f"/api/automations/{rid}", admin, json={"is_active": True})

    # ---------- 9. 权限:employee 无权管理 ----------
    ess_email = f"t7-ess-{RUN}@example.com"
    ess_pwd = "T7ess!pass"
    tid, ess_uid, emp_id = None, None, None
    from sqlalchemy import select as sa_select
    from app.domain.models import Employee, Tenant, User as DbUser
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        tid = t.id
        eu = DbUser(tenant_id=tid, email=ess_email,
                    password_hash=hash_password(ess_pwd), role="employee")
        db.add(eu)
        db.flush()
        e = Employee(tenant_id=tid, user_id=eu.id, name=f"T7员工{RUN}",
                     employee_no=f"T7{RUN}", status="active")
        db.add(e)
        db.commit()
        ess_uid, emp_id = eu.id, e.id
    ess = login(ess_email, ess_pwd)
    r = req("GET", "/api/automations", ess)
    check("employee 访问规则列表 403", r.status_code == 403)
    r = req("POST", "/api/automations", ess, json=rule_body)
    check("employee 创建规则 403", r.status_code == 403)

    # ---------- 10. 租户隔离 ----------
    t2_uid = None
    with SessionLocal() as db:
        t2 = Tenant(name=f"t7-iso-{RUN}")
        db.add(t2)
        db.flush()
        u2 = DbUser(tenant_id=t2.id, email=f"t7-iso-{RUN}@example.com",
                    password_hash=hash_password("iso123456"), role="admin")
        db.add(u2)
        db.commit()
        t2_uid = u2.id
    iso = login(f"t7-iso-{RUN}@example.com", "iso123456")
    r = req("GET", "/api/automations", iso)
    check("他租户看不到本租户规则",
          r.status_code == 200 and all(x["id"] != rid for x in r.json().get("items", [])))
    r = req("PATCH", f"/api/automations/{rid}", iso, json={"is_active": False})
    check("他租户 PATCH 本租户规则 404", r.status_code == 404)
    r = req("GET", f"/api/automations/{rid}/runs", iso)
    check("他租户读 runs 404", r.status_code == 404)

    # ============================================================ 考勤
    # ---------- 11. policy roundtrip ----------
    r = req("GET", "/api/attendance/policy", admin)
    check("读默认 policy", r.status_code == 200 and "grace_in_minutes" in r.json())
    pol = {"grace_in_minutes": 15, "grace_out_minutes": 15,
           "track_late_early": True, "ip_check_enabled": False,
           "allowed_ips": [], "notify_on_flag": True}
    r = req("PUT", "/api/attendance/policy", admin, json=pol)
    check("PUT policy 生效", r.status_code == 200 and
          r.json()["grace_in_minutes"] == 15 and r.json()["track_late_early"])
    r = req("PUT", "/api/attendance/policy", admin,
            json={**pol, "ip_check_enabled": True, "allowed_ips": ["bad-cidr"]})
    check("非法 CIDR 400", r.status_code == 400)
    r = req("PUT", "/api/attendance/policy", admin, json={**pol, "grace_in_minutes": -1})
    check("负宽限 400", r.status_code == 400)
    r = req("GET", "/api/attendance/policy", ess)
    check("employee 读 policy 403", r.status_code == 403)

    # ---------- 12. work-calendar 时间往返 ----------
    r = req("PUT", "/api/work-calendar", admin, json={
        "workdays": [1, 2, 3, 4, 5], "workday_hours": 8,
        "work_start": "09:00:00", "work_end": "18:00:00"})
    check("PUT 日历上下班时间", r.status_code == 200 and
          r.json()["work_start"].startswith("09:00") and
          r.json()["work_end"].startswith("18:00"), r.text[:200])
    r = req("PUT", "/api/work-calendar", admin, json={
        "workdays": [0, 1, 2], "workday_hours": 8,
        "work_start": "09:00:00", "work_end": "18:00:00"})
    check("非法 workday 422", r.status_code == 422)

    # ---------- 13. 造迟到/早退打卡 → 汇总 → flags ----------
    # 最近一个工作日(周一~周五);时区 Asia/Shanghai:本地 09:40 入 = UTC 01:40
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    month = d.strftime("%Y-%m")
    with SessionLocal() as db:
        from app.domain.models import EmployeeCheckin
        db.add(EmployeeCheckin(tenant_id=tid, employee_id=emp_id,
                               ts=datetime(d.year, d.month, d.day, 1, 40),
                               kind="in", method="web"))
        db.add(EmployeeCheckin(tenant_id=tid, employee_id=emp_id,
                               ts=datetime(d.year, d.month, d.day, 9, 20),
                               kind="out", method="web"))
        db.commit()
    r = req("POST", "/api/attendance/summarize", admin,
            json={"start": str(d), "end": str(d)})
    check("日结汇总", r.status_code == 200, r.text[:200])

    r = req("GET", f"/api/attendance/flags?month={month}&employee_id={emp_id}", admin)
    flags = r.json().get("items", []) if r.status_code == 200 else []
    late = next((f for f in flags if f["kind"] == "late_in" and f["day"] == str(d)), None)
    early = next((f for f in flags if f["kind"] == "early_out" and f["day"] == str(d)), None)
    check("迟到 flag(40 分钟)", late is not None and late["minutes"] == 40,
          r.text[:300])
    check("早退 flag(40 分钟)", early is not None and early["minutes"] == 40,
          r.text[:300])
    r = req("GET", f"/api/attendance/flags?month={month}&kind=early_out&employee_id={emp_id}", admin)
    check("kind 过滤", r.status_code == 200 and all(
        f["kind"] == "early_out" for f in r.json().get("items", [])))

    # ---------- 14. excuse/unexcuse ----------
    fid = late["id"] if late else None
    if fid:
        r = req("POST", f"/api/attendance/flags/{fid}/excuse", admin,
                json={"note": "客户拜访"})
        check("免计", r.status_code == 200 and r.json()["excused"] is True
              and r.json()["excused_note"] == "客户拜访", r.text[:200])
        r = req("GET", f"/api/attendance/flags?month={month}&excused=true&employee_id={emp_id}", admin)
        check("excused 过滤", r.status_code == 200 and any(
            f["id"] == fid for f in r.json().get("items", [])))
        # 重汇总后免计态保留
        req("POST", "/api/attendance/summarize", admin,
            json={"start": str(d), "end": str(d)})
        r = req("GET", f"/api/attendance/flags?month={month}&employee_id={emp_id}", admin)
        still = next((f for f in r.json().get("items", []) if f["id"] == fid), None)
        check("重汇总后免计态保留", still is not None and still["excused"] is True)
        r = req("POST", f"/api/attendance/flags/{fid}/unexcuse", admin)
        check("取消免计", r.status_code == 200 and r.json()["excused"] is False)
        # excuse 审计
        r = req("GET", "/api/audit/events?entity_type=attendance_flag&size=10", admin)
        acts = {x.get("action") for x in r.json().get("items", [])} if r.status_code == 200 else set()
        check("excuse/unexcuse 审计留痕", {"excuse", "unexcuse"} <= acts, str(acts))

    # ---------- 15. monthly-summary ----------
    r = req("GET", f"/api/attendance/monthly-summary?month={month}", admin)
    rows = r.json() if r.status_code == 200 else []
    row = next((x for x in rows if x["employee_id"] == emp_id), None)
    check("月汇总行含迟到/早退计数", row is not None and
          row["late_count"] >= 1 and row["early_count"] >= 1 and
          row["late_minutes"] >= 40, str(row))

    # ---------- 16. ESS 自助查看 ----------
    r = req("GET", f"/api/me/attendance-flags?month={month}", ess)
    check("ESS 查看自己 flags", r.status_code == 200 and any(
        f["day"] == str(d) for f in r.json()), r.text[:200])

    # ---------- 17. IP 白名单打卡 ----------
    r = req("PUT", "/api/attendance/policy", admin, json={
        **pol, "ip_check_enabled": True, "allowed_ips": ["10.9.8.0/24"]})
    check("开启 IP 白名单", r.status_code == 200 and r.json()["ip_check_enabled"])
    r = requests.post(f"{BASE}/api/me/checkins",
                      headers={"Authorization": f"Bearer {ess}",
                               "X-Forwarded-For": "1.2.3.4"},
                      json={"kind": "in"}, timeout=30)
    check("白名单外 IP 打卡 403", r.status_code == 403, r.text[:200])
    r = requests.post(f"{BASE}/api/me/checkins",
                      headers={"Authorization": f"Bearer {ess}",
                               "X-Forwarded-For": "10.9.8.7"},
                      json={"kind": "in"}, timeout=30)
    check("白名单内 IP 打卡 201", r.status_code == 201, r.text[:200])
    r = req("GET", "/api/audit/events?entity_type=employee_checkin&size=10", admin)
    denied = any(x.get("result") == "denied" for x in r.json().get("items", [])) \
        if r.status_code == 200 else False
    check("拒绝打卡审计 result=denied", denied)
    req("PUT", "/api/attendance/policy", admin, json=pol)  # 还原

    # ---------- 18. 汇总重复执行幂等(flag 不翻倍)----------
    r = req("POST", "/api/attendance/summarize", admin,
            json={"start": str(d), "end": str(d)})
    r = req("GET", f"/api/attendance/flags?month={month}&employee_id={emp_id}", admin)
    day_flags = [f for f in r.json().get("items", []) if f["day"] == str(d)]
    check("重复汇总 flag 不重复", len(day_flags) <= 2, str(len(day_flags)))

    # ---------- 汇总 ----------
    print(f"\n===== {len(PASS)} PASS / {len(FAIL)} FAIL =====")
    for n in FAIL:
        print(f"  FAIL: {n}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
