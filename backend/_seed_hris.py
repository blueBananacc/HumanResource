"""HRIS 全链路种子数据 — 通过真实 API 走完整链(兼作后端链审查)。

用法(宿主机):
    python backend/_seed_hris.py

默认打 http://localhost(nginx 网关 → backend)。需要 admin 登录。
幂等:按名称/工号查重,重复运行不会翻倍造数据。
产出:
  - 组织树 / 6 名员工 / 2 个 ESS 账号(emp1@hronfire.dev / emp2@hronfire.dev,密码 Emp@123456)
  - 假期类型+额度+申请(待批/已批/已驳) + 今日打卡 + 考勤汇总
  - 生命周期模板+事件 / 招聘需求(已批+待批) / Offer(已接受→自动建档)
  - 目标+考核(含一条待上级评) / 报销(待批+已支付) / 工资条(已发放)
  - _seed_checkins.sql:过去 5 个工作日的历史打卡,psql 执行后考勤更饱满
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import date, timedelta

BASE = os.environ.get("SEED_BASE", "http://localhost:8000/api")
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123456"
EMP_PASSWORD = "Emp@123456"


def req(method: str, path: str, token: str | None = None, body=None, quiet=False):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if quiet:
            return {"_status": e.code, "_detail": detail}
        raise SystemExit(f"FAIL {method} {path} → {e.code}: {detail}")


def login(email: str, password: str) -> str:
    r = req("POST", "/auth/login", body={"email": email, "password": password})
    return r["access_token"]


def find_employee(tok, name_or_no: str):
    items = req("GET", "/employees?limit=200", tok).get("items", [])
    for e in items:
        if e["name"] == name_or_no or e["employee_no"] == name_or_no:
            return e
    return None


def main():
    print("== 登录 admin ==")
    tok = login(ADMIN_EMAIL, ADMIN_PASSWORD)

    # ---------- 组织树 ----------
    print("== 组织节点 ==")
    tree = req("GET", "/org/tree", tok)
    def node_by_name(name):
        stack = list(tree)
        while stack:
            n = stack.pop()
            if n["name"] == name:
                return n
            stack += n.get("children", [])
        return None

    if not node_by_name("HROnfire 科技"):
        company = req("POST", "/org/nodes", tok, {"name": "HROnfire 科技", "kind": "company"})
    else:
        company = node_by_name("HROnfire 科技")
    cid = company["id"]
    for name in ("研发部", "产品部", "人事部"):
        if not node_by_name(name):
            req("POST", "/org/nodes", tok, {"name": name, "kind": "department", "parent_id": cid})
    tree = req("GET", "/org/tree", tok)
    dev = node_by_name("研发部")
    prod = node_by_name("产品部")
    hr = node_by_name("人事部")
    print(f"  company={cid[:8]} dev={dev['id'][:8]} prod={prod['id'][:8]} hr={hr['id'][:8]}")

    # ---------- 员工 ----------
    print("== 员工 ==")
    roster = [
        ("EMP001", "陈志远", "CTO", dev["id"], None, "active"),
        ("EMP002", "李晓芸", "资深前端工程师", dev["id"], "EMP001", "active"),
        ("EMP003", "王浩", "后端工程师", dev["id"], "EMP001", "active"),
        ("EMP004", "赵雨晴", "产品经理", prod["id"], "EMP001", "active"),
        ("EMP005", "孙倩", "HR 专员", hr["id"], None, "active"),
        ("EMP006", "周凯", "测试工程师(试用)", dev["id"], "EMP002", "leave"),
    ]
    emp_by_no = {}
    for no, name, title, org, mgr_no, status_ in roster:
        e = find_employee(tok, no) or find_employee(tok, name)
        if not e:
            e = req("POST", "/employees", tok, {
                "employee_no": no, "name": name, "designation": title,
                "org_unit_id": org, "employment_type": "full_time",
                "hire_date": "2024-03-01", "status": status_,
                "work_email": f"{no.lower()}@hronfire.dev",
            })
        elif mgr_no and not e.get("reports_to_id"):
            mgr = emp_by_no.get(mgr_no) or find_employee(tok, mgr_no)
            if mgr:
                e = req("PUT", f"/employees/{e['id']}", tok, {
                    "name": e["name"], "reports_to_id": mgr["id"],
                })
        emp_by_no[no] = e
        print(f"  {no} {name} → {e['id'][:8]}")

    # ---------- ESS 账号 ----------
    print("== ESS 账号 ==")
    for no, email in (("EMP002", "emp1@hronfire.dev"), ("EMP003", "emp2@hronfire.dev")):
        e = emp_by_no[no]
        if not e.get("has_account"):
            r = req("POST", f"/employees/{e['id']}/account", tok,
                    {"email": email, "password": EMP_PASSWORD})
            print(f"  {email} 绑定(密码 {EMP_PASSWORD}) user={r['user_id'][:8]}")
        else:
            print(f"  {email} 已绑定")

    emp1 = login("emp1@hronfire.dev", EMP_PASSWORD)
    emp2 = login("emp2@hronfire.dev", EMP_PASSWORD)

    # ---------- 假期 ----------
    print("== 假期类型/额度/申请 ==")
    types = {t["name"]: t for t in req("GET", "/leave/types", tok)}
    for name, days, paid in (("年假", 10, True), ("病假", 5, True), ("事假", 0, False)):
        if name not in types:
            types[name] = req("POST", "/leave/types", tok,
                              {"name": name, "annual_days": days, "is_paid": paid})
    annual = types["年假"]
    year = date.today().year
    allocs = req("GET", f"/leave/allocations?limit=200", tok).get("items", [])
    for no in ("EMP002", "EMP003", "EMP004"):
        e = emp_by_no[no]
        if not any(a["employee_id"] == e["id"] and a["leave_type_id"] == annual["id"] and a["year"] == year for a in allocs):
            req("POST", "/leave/allocations", tok, {
                "employee_id": e["id"], "leave_type_id": annual["id"],
                "year": year, "days": 10, "note": "年度额度",
            })
            print(f"  年假 10 天 → {e['name']}")

    apps = req("GET", "/leave/applications?limit=200", tok).get("items", [])
    if not any(a["status"] == "pending" for a in apps):
        r = req("POST", "/me/leave-applications", emp1, {
            "leave_type_id": annual["id"],
            "start_date": str(date.today() + timedelta(days=7)),
            "end_date": str(date.today() + timedelta(days=8)),
            "days": 2, "reason": "家中有事",
        })
        print(f"  待审批申请 → {r['id'][:8]}")
    if not any(a["status"] == "approved" for a in apps):
        r = req("POST", "/me/leave-applications", emp2, {
            "leave_type_id": types["病假"]["id"],
            "start_date": str(date.today() - timedelta(days=10)),
            "end_date": str(date.today() - timedelta(days=9)),
            "days": 2, "reason": "感冒发烧",
        })
        req("POST", f"/leave/applications/{r['id']}/approve", tok, {"note": "注意休息"})
        print(f"  已批准申请 → {r['id'][:8]}")
    if not any(a["status"] == "rejected" for a in apps):
        r = req("POST", "/me/leave-applications", emp2, {
            "leave_type_id": types["事假"]["id"],
            "start_date": str(date.today() + timedelta(days=20)),
            "end_date": str(date.today() + timedelta(days=25)),
            "days": 5, "reason": "长途旅行",
        })
        req("POST", f"/leave/applications/{r['id']}/reject", tok, {"note": "事假天数超限,请拆分"})
        print(f"  已驳回申请 → {r['id'][:8]}")

    # ---------- 打卡 + 考勤 ----------
    print("== 打卡/考勤 ==")
    my_today = req("GET", "/me/checkins", emp1)
    if not any(c["kind"] == "in" for c in my_today):
        req("POST", "/me/checkins", emp1, {"kind": "in"})
        req("POST", "/me/checkins", emp1, {"kind": "out"})
        print("  emp1 今日打卡 in/out")
    my2_today = req("GET", "/me/checkins", emp2)
    if not any(c["kind"] == "in" for c in my2_today):
        req("POST", "/me/checkins", emp2, {"kind": "in"})
        print("  emp2 今日打卡 in")
    first = date.today().replace(day=1)
    req("POST", "/attendance/summarize", tok, {"start": str(first), "end": str(date.today())})
    print("  本月考勤已汇总")

    # ---------- 生命周期 ----------
    print("== 生命周期 ==")
    tpls = req("GET", "/lifecycle/templates", tok)
    if not tpls:
        req("POST", "/lifecycle/templates", tok, {
            "kind": "onboarding", "name": "标准入职清单", "is_default": True,
            "tasks": ["发放offer并确认入职日期", "创建系统账号", "开通门禁/邮箱",
                      "设备领取", "入职培训", "试用期目标确认"],
        })
        req("POST", "/lifecycle/templates", tok, {
            "kind": "separation", "name": "标准离职清单", "is_default": True,
            "tasks": ["工作交接", "收回设备", "关闭账号权限", "离职面谈", "结算工资"],
        })
        print("  模板 ×2")
    evs = req("GET", "/lifecycle/events?limit=200", tok).get("items", [])
    if not any(e["kind"] == "transfer" for e in evs):
        r = req("POST", "/lifecycle/events", tok, {
            "employee_id": emp_by_no["EMP003"]["id"], "kind": "transfer",
            "effective_date": str(date.today() + timedelta(days=14)),
            "note": "后端 → 产品部轮岗",
        })
        print(f"  transfer 事件 → {r['id'][:8]}({len(r['tasks'])} 项任务)")

    # ---------- 招聘需求 ----------
    print("== 招聘需求 ==")
    reqs = req("GET", "/requisitions?limit=200", tok).get("items", [])
    if not any(r["status"] == "approved" or r["status"] == "converted" for r in reqs):
        r = req("POST", "/requisitions", tok, {
            "title": "研发部-高级后端工程师", "org_unit_id": dev["id"],
            "headcount": 2, "level": "senior", "reason": "新业务线扩张",
        })
        req("POST", f"/requisitions/{r['id']}/approve", tok, {"note": "Q4 编制内"})
        print(f"  已批准需求 → {r['title']}")
    if not any(r["status"] == "pending" for r in reqs):
        r = req("POST", "/requisitions", tok, {
            "title": "产品部-产品设计师", "org_unit_id": prod["id"],
            "headcount": 1, "level": "intermediate", "reason": "移动端改版",
        })
        print(f"  待审批需求 → {r['title']}")

    # ---------- Offer(需要候选人,由调用方先 SQL 建好或直接跳过) ----------
    print("== Offer ==")
    offers = req("GET", "/offers?limit=200", tok).get("items", [])
    if not offers:
        cand_id = sys.argv[sys.argv.index("--candidate") + 1] if "--candidate" in sys.argv else None
        if not cand_id:
            resumes = req("GET", "/resumes?limit=50", tok).get("items", [])
            cand_id = resumes[0]["candidate"]["id"] if resumes else None
        if cand_id:
            o = req("POST", "/offers", tok, {
                "candidate_id": cand_id, "position": "高级后端工程师",
                "salary": {"amount": 32000, "currency": "CNY", "period": "monthly"},
                "start_date": str(date.today() + timedelta(days=30)),
                "letter": "欢迎加入 HROnfire!",
            })
            req("POST", f"/offers/{o['id']}/send", tok)
            req("POST", f"/offers/{o['id']}/respond", tok, {"accepted": True})
            print(f"  Offer 已接受 → 自动建员工 + onboarding")
        else:
            print("  !! 无候选人,跳过 offer — 传 --candidate <id> 或先上传简历")

    # ---------- 绩效 ----------
    print("== 绩效 ==")
    goals = req("GET", "/performance/goals?limit=200", tok).get("items", [])
    if not goals:
        req("POST", "/performance/goals", tok, {
            "employee_id": emp_by_no["EMP002"]["id"],
            "title": "完成新版前端重构", "description": "React 19 + 设计系统迁移",
            "due_date": str(date.today() + timedelta(days=60)),
        })
        req("POST", "/performance/goals", tok, {
            "employee_id": emp_by_no["EMP003"]["id"],
            "title": "API 性能优化 P95<200ms", "description": "慢查询治理",
            "due_date": str(date.today() + timedelta(days=45)),
        })
        print("  目标 ×2")
    cycles = req("GET", "/performance/cycles", tok)
    if not cycles:
        cy = req("POST", "/performance/cycles", tok, {
            "name": f"{year} 年度考核",
            "period_start": f"{year}-01-01", "period_end": f"{year}-12-31",
        })
        req("POST", f"/performance/cycles/{cy['id']}/open", tok)
        print(f"  周期 {cy['name']} 已开启")
        apps2 = req("POST", "/performance/appraisals", tok, {
            "cycle_id": cy["id"],
            "employee_ids": [emp_by_no["EMP002"]["id"], emp_by_no["EMP003"]["id"]],
        })
        req("POST", f"/me/appraisals/{apps2[0]['id']}/submit-self", emp1, {
            "self_review": "本季度完成题库批量管理与导航改造,主动补了 3 个线上缺陷。",
            "self_rating": 4,
        })
        print("  考核 ×2(emp1 已自评待上级评,emp2 待自评)")

    # ---------- 报销 ----------
    print("== 报销 ==")
    etypes = req("GET", "/expenses/types", tok)
    if not etypes:
        for n in ("差旅", "餐费", "交通"):
            req("POST", "/expenses/types", tok, {"name": n})
        etypes = req("GET", "/expenses/types", tok)
    claims = req("GET", "/expenses/claims?limit=200", tok).get("items", [])
    et = {t["name"]: t["id"] for t in etypes}
    if not any(c["status"] == "pending" for c in claims):
        req("POST", "/me/expense-claims", emp1, {
            "title": "上海出差(11月)",
            "items": [
                {"type_id": et.get("差旅"), "occurred_on": str(date.today() - timedelta(days=5)),
                 "amount": 1280.0, "note": "高铁往返"},
                {"type_id": et.get("餐费"), "occurred_on": str(date.today() - timedelta(days=4)),
                 "amount": 320.5, "note": "客户宴请"},
            ],
        })
        print("  待批报销单 ×1")
    if not any(c["status"] in ("approved", "paid") for c in claims):
        r = req("POST", "/me/expense-claims", emp2, {
            "title": "团队加班餐费",
            "items": [{"type_id": et.get("餐费"), "occurred_on": str(date.today() - timedelta(days=2)),
                       "amount": 486.0, "note": "版本发布夜"}],
        })
        req("POST", f"/expenses/claims/{r['id']}/approve", tok, {"note": "OK"})
        req("POST", f"/expenses/claims/{r['id']}/pay", tok)
        print("  已支付报销单 ×1")

    # ---------- 工资条 ----------
    print("== 工资条 ==")
    slips = req("GET", "/payroll/slips?limit=200", tok).get("items", [])
    period = date.today().replace(day=1)
    period_s = str(period)[:7]
    for no, base, bonus in (("EMP002", 25000, 3000), ("EMP003", 22000, 1500)):
        e = emp_by_no[no]
        if not any(s["employee_id"] == e["id"] and s["period"] == period_s for s in slips):
            s = req("POST", "/payroll/slips", tok, {
                "employee_id": e["id"], "period": period_s,
                "items": [
                    {"label": "基本工资", "kind": "earning", "amount": base},
                    {"label": "绩效奖金", "kind": "earning", "amount": bonus},
                    {"label": "社保公积金", "kind": "deduction", "amount": round(base * 0.175, 2)},
                    {"label": "个税", "kind": "deduction", "amount": round(base * 0.08, 2)},
                ],
            })
            req("POST", f"/payroll/slips/{s['id']}/issue", tok)
            print(f"  {e['name']} {period_s} 已发放")

    # ---------- 历史打卡 SQL ----------
    emps_for_sql = [emp_by_no["EMP002"], emp_by_no["EMP003"], emp_by_no["EMP004"]]
    lines = ["-- 过去 5 个工作日历史打卡(psql -f _seed_checkins.sql 执行)",
             f"-- tenant/employee id 见下方;跑完后可再调 POST /api/attendance/summarize"]
    weekdays = []
    d = date.today()
    while len(weekdays) < 5:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            weekdays.append(d)
    for e in emps_for_sql:
        for wd in weekdays:
            lines.append(
                "INSERT INTO employee_checkins (id, tenant_id, employee_id, ts, kind, method) VALUES "
                f"(gen_random_uuid()::text, (SELECT tenant_id FROM employees WHERE id='{e['id']}'), "
                f"'{e['id']}', '{wd} 09:0{len(lines)%6}:00', 'in', 'web'),"
            )
            lines[-1] = lines[-1][:-1] + ";"
            lines.append(
                "INSERT INTO employee_checkins (id, tenant_id, employee_id, ts, kind, method) VALUES "
                f"(gen_random_uuid()::text, (SELECT tenant_id FROM employees WHERE id='{e['id']}'), "
                f"'{e['id']}', '{wd} 18:1{len(lines)%6}:00', 'out', 'web');"
            )
    with open("_seed_checkins.sql", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("=" * 50)
    print("完成。账号:")
    print(f"  管理员   admin@example.com / {ADMIN_PASSWORD}")
    print(f"  员工 ESS emp1@hronfire.dev / {EMP_PASSWORD} (李晓芸)")
    print(f"  员工 ESS emp2@hronfire.dev / {EMP_PASSWORD} (王浩)")
    print("历史打卡: docker exec -i free-hire-postgres-1 psql -U freehire -d freehire < backend/_seed_checkins.sql")
    print("然后 POST /api/attendance/summarize 重算本月。")


if __name__ == "__main__":
    main()
