"""HRIS 全链路种子数据 — 通过真实 API 走完整链(兼作后端链审查)。

用法(宿主机):
    python backend/_seed_hris.py                          # 本地
    SEED_BASE=http://<host>/api SEED_ADMIN_PASSWORD=xxx \
        python backend/_seed_hris.py                      # 云端/任意部署

默认打 http://localhost(nginx 网关 → backend)。需要 admin 登录。
幂等:按名称/工号查重,重复运行不会翻倍造数据。
产出:
  - 组织树 / 6 名员工 / 2 个 ESS 账号(emp1@hronfire.dev / emp2@hronfire.dev,密码 Emp@123456)
  - admin 绑定 EMP001(管理员也走 /me 自助端演示"绑档案=也是员工")
  - 假期类型+额度+申请(待批/已批/已驳) + 今日打卡 + 考勤汇总
  - 生命周期模板+事件 / 招聘需求(已批+待批) / Offer(已接受→自动建档)
  - 目标+考核(含一条待上级评) / 报销(待批+已支付) / 工资条(已发放)
  - 工作日历+节假日 / 假期政策(结转+发放) / 补卡+加班→调休(待批+已批)
  - ATS:技能字典 / 岗位(已发布×2+草稿×1) / 简历上传×3 / 人才分组 /
    管道申请(applied→interview) / 岗位技能模型 / 题库 / AI 面试+面试安排 /
    评分规则 / 问卷(已发布+链接) / 会议室+预约 / 公告×3
  - 员工服务:请求×2(emp1/emp2) / HR 工单×2(1 处理中+1 新建) / 知识库×3
  - 其他:职位编制+在岗 / 项目×2 / 班次+排班 / 薪酬区间+调薪评审 /
    自动化规则×2 / 收藏筛选×2 / 外联活动(草稿) / 审批策略 / AI 匹配触发
  - 近 3 个工作日历史考勤:纯 API 补卡审批回填(云端无 psql 也能跑);
    另有 _seed_checkins.sql 供本地 psql 造更长的历史
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from datetime import date, timedelta

# 默认打 nginx 网关(:80);backend 容器不对外发布端口,8000 是宿主机上别的服务
BASE = os.environ.get("SEED_BASE", "http://localhost/api")
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = os.environ.get("SEED_ADMIN_PASSWORD", "admin123456")
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


def req_upload(path: str, token: str, filename: str, data: bytes, content_type="text/plain"):
    """multipart 文件上传(简历用);失败返回 {_status,_detail} 不抛异常。"""
    boundary = "seedb" + uuid.uuid4().hex[:16]
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    r = urllib.request.Request(f"{BASE}{path}", data=body, method="POST")
    r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return {"_status": e.code, "_detail": e.read().decode(errors="replace")}


def login(email: str, password: str) -> str:
    r = req("POST", "/auth/login", body={"email": email, "password": password})
    return r["access_token"]


def ensure_hr_token(tok) -> str | None:
    """幂等创建 hr@example.com(role=hr)并返回其 token;受限时返回 None。

    职责分离修复后,admin 自提的需求不能再自批 — 审批改由这个 hr 账号执行。
    已存在时走 reset 拿一次性密码,保证重复运行也能登录。
    """
    email = "hr@example.com"
    r = req("POST", "/users/", tok, {"email": email, "role": "hr"}, quiet=True)
    if r.get("_status"):
        if r["_status"] != 409:
            # 配额(license)或权限受限 — 没有第二审批人,需求留 pending
            print(f"  (hr 账号不可用:{r['_status']},需求保持待审批)")
            return None
        users = req("GET", "/users/", tok)
        uid = next((u["id"] for u in users if u["email"] == email), None)
        if uid is None:
            return None
        pw = req("POST", f"/users/{uid}/reset", tok)["new_password"]
        print(f"  {email} 密码已重置 → {pw}")
    else:
        pw = r["initial_password"]
        print(f"  {email} 已创建 → 初始密码 {pw}")
    return login(email, pw)


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
        ("EMP001", "卢品超", "CTO", dev["id"], None, "active"),
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
        # 修复被历史编辑清掉的部门/职位(PUT 要带上现有值,避免其他字段被重置)
        if e and (e.get("org_unit_id") != org or e.get("designation") != title):
            mgr = emp_by_no.get(mgr_no) if mgr_no else None
            e = req("PUT", f"/employees/{e['id']}", tok, {
                "name": e["name"], "designation": title,
                "org_unit_id": org,
                "reports_to_id": e.get("reports_to_id") or (mgr["id"] if mgr else None),
                "work_email": e.get("work_email"), "phone": e.get("phone"),
                "hire_date": e.get("hire_date"), "status": e.get("status") or status_,
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
    # admin 也挂到 EMP001 — 管理员既能管全端,也能走 /me 自助端
    e1 = emp_by_no["EMP001"]
    if not e1.get("has_account"):
        req("POST", f"/employees/{e1['id']}/account", tok, {"email": ADMIN_EMAIL})
        print(f"  {ADMIN_EMAIL} 绑定到 EMP001({e1['name']})")

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

    def _overlaps(emp_id: str, s: date, e: date) -> bool:
        return any(
            a["employee_id"] == emp_id and a["status"] != "cancelled"
            and a["start_date"] <= str(e) and a["end_date"] >= str(s)
            for a in apps
        )

    def _apply(emp_tok, emp_id, lt_id, offset: int, span: int, reason: str):
        """幂等建申请:窗口与该员工现有申请重叠就往后挪一周再找。

        days 不传,由服务端按工作日历算(周末/节假日不算请假);
        全周末窗口(0 个工作日)直接跳过。
        """
        for shift in range(0, 366, 7):
            s = date.today() + timedelta(days=offset + shift)
            e = s + timedelta(days=span - 1)
            wd = sum(1 for i in range((e - s).days + 1)
                     if (s + timedelta(days=i)).weekday() < 5)
            if wd == 0 or _overlaps(emp_id, s, e):
                continue
            r = req("POST", "/me/leave-applications", emp_tok, {
                "leave_type_id": lt_id, "start_date": str(s),
                "end_date": str(e), "reason": reason,
            })
            apps.append({"employee_id": emp_id, "status": "pending",
                         "start_date": str(s), "end_date": str(e)})
            return r
        raise SystemExit("找不到可用的请假窗口(未来一年内全部重叠)")

    if not any(a["status"] == "pending" for a in apps):
        r = _apply(emp1, emp_by_no["EMP002"]["id"], annual["id"], 7, 2, "家中有事")
        print(f"  待审批申请 → {r['id'][:8]}")
    if not any(a["status"] == "approved" for a in apps):
        r = _apply(emp2, emp_by_no["EMP003"]["id"], types["病假"]["id"], -10, 2, "感冒发烧")
        req("POST", f"/leave/applications/{r['id']}/approve", tok, {"note": "注意休息"})
        apps[-1]["status"] = "approved"
        print(f"  已批准申请 → {r['id'][:8]}")
    if not any(a["status"] == "rejected" for a in apps):
        r = _apply(emp2, emp_by_no["EMP003"]["id"], types["事假"]["id"], 20, 5, "长途旅行")
        req("POST", f"/leave/applications/{r['id']}/reject", tok, {"note": "事假天数超限,请拆分"})
        apps[-1]["status"] = "rejected"
        print(f"  已驳回申请 → {r['id'][:8]}")

    # ---------- 工作日历 / 节假日 ----------
    print("== 工作日历/节假日 ==")
    cal = req("GET", "/work-calendar", tok)
    if cal["workdays"] != [1, 2, 3, 4, 5] or cal["workday_hours"] != 8:
        req("PUT", "/work-calendar", tok,
            {"workdays": [1, 2, 3, 4, 5], "workday_hours": 8})
        print("  日历已重置为周一~周五 8h")
    hols = {h["day"]: h for h in req("GET", "/work-calendar/holidays", tok)}
    today = date.today()
    y, m = today.year, today.month
    # 当月最后一个周六作为调休补班示例
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    month_end = (date(ny, nm, 1) - timedelta(days=1)).day
    last_sat = max(d for d in range(22, month_end + 1)
                   if date(y, m, d).weekday() == 5)
    seed_hols = [
        (date(y, m, 1), "月初司庆假", "holiday"),
        (date(y, m, 15), "季度团建(放假)", "holiday"),
        (date(y, m, last_sat), "调休补班", "workday"),
    ]
    for d, name, kind in seed_hols:
        if str(d) not in hols:
            req("POST", "/work-calendar/holidays", tok,
                {"day": str(d), "name": name, "kind": kind}, quiet=True)
            print(f"  {d} {name}({kind})")

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
        # admin 提交 + hr 审批 — 同一人不能自批,走双账号演示职责分离
        r = req("POST", "/requisitions", tok, {
            "title": "研发部-高级后端工程师", "org_unit_id": dev["id"],
            "headcount": 2, "level": "senior", "reason": "新业务线扩张",
        })
        hr_tok = ensure_hr_token(tok)
        if hr_tok:
            req("POST", f"/requisitions/{r['id']}/approve", hr_tok, {"note": "Q4 编制内"})
            print(f"  已批准需求 → {r['title']}(hr 账号审批)")
        else:
            print(f"  待审批需求 → {r['title']}")
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

    # ---------- 调休/加班/补卡/假期政策 ----------
    print("== 调休/加班/补卡/假期政策 ==")
    types = {t["name"]: t for t in req("GET", "/leave/types", tok)}
    if "调休假" not in types:
        types["调休假"] = req("POST", "/leave/types", tok,
                              {"name": "调休假", "annual_days": 0,
                               "is_paid": True, "is_comp": True})
        print("  调休假(is_comp)已建")
    comp = types["调休假"]

    policies = req("GET", "/leave/policies", tok)
    pol = next((p for p in policies if p["name"] == "正式员工假勤包"), None)
    if not pol:
        pol = req("POST", "/leave/policies", tok, {
            "name": "正式员工假勤包",
            "items": [{"leave_type_id": annual["id"], "days": 10},
                      {"leave_type_id": types["病假"]["id"], "days": 5}],
            "carry_forward": 5,
        })
        print(f"  政策 正式员工假勤包 → {pol['id'][:8]}")
    if pol["assigned_count"] == 0:
        r = req("POST", f"/leave/policies/{pol['id']}/assign", tok, {
            "employee_ids": [emp_by_no["EMP004"]["id"]],
        })
        print(f"  政策已绑定 EMP004,发放 {r['year']} 年额度")

    att_reqs = req("GET", "/attendance/requests?limit=50", tok).get("items", [])
    if not any(i["status"] == "pending" for i in att_reqs):
        yesterday = today - timedelta(days=1)
        r = req("POST", "/me/attendance-requests", emp1, {
            "ts": f"{yesterday}T09:02:00Z", "kind": "in",
            "reason": "早会直接进会议室,忘打卡"}, quiet=True)
        if not r.get("_status"):
            print(f"  待审批补卡 → {r['id'][:8]}")
    ot_reqs = req("GET", "/overtime/requests?limit=50", tok).get("items", [])
    if not any(i["status"] == "pending" for i in ot_reqs):
        r = req("POST", "/me/overtime-requests", emp2, {
            "leave_type_id": comp["id"],
            "work_date": str(today - timedelta(days=2)),
            "hours": 6, "reason": "版本发布加班"}, quiet=True)
        if not r.get("_status"):
            print(f"  待审批加班 → {r['id'][:8]}(折算 {r['comp_days']} 天)")
    if not any(i["status"] == "approved" for i in ot_reqs):
        r = req("POST", "/me/overtime-requests", emp1, {
            "leave_type_id": comp["id"],
            "work_date": str(today - timedelta(days=5)),
            "hours": 8, "reason": "周末值班"}, quiet=True)
        if not r.get("_status"):
            req("POST", f"/overtime/requests/{r['id']}/approve", tok,
                {"note": "记录属实"})
            print(f"  已批加班 → 调休入账 {r['comp_days']} 天")

    # ---------- 历史补卡回填考勤(纯 API,云端无 psql 也能出饱满考勤) ----------
    print("== 历史考勤回填 ==")
    hols = {h["day"] for h in req("GET", "/work-calendar/holidays", tok)}
    past_wds: list[date] = []
    d = today
    while len(past_wds) < 3:
        d -= timedelta(days=1)
        if d.weekday() < 5 and str(d) not in hols:
            past_wds.append(d)
    month_start = today.replace(day=1)
    for emp_tok, emp in (
        (emp1, emp_by_no["EMP002"]),
        (emp2, emp_by_no["EMP003"]),
    ):
        att = req(
            "GET",
            f"/attendance?month={str(month_start)[:7]}&employee_id={emp['id']}&limit=100",
            tok,
        ).get("items", [])
        done_days = {a["day"] for a in att
                     if a["status"] in ("present", "leave", "remote")}
        for wd in past_wds:
            if str(wd) in done_days:
                continue
            r = req("POST", "/me/attendance-requests", emp_tok, {
                "ts": f"{wd}T09:0{wd.day % 6}:00Z", "kind": "in",
                "reason": "外勤见客户,忘打卡"}, quiet=True)
            if r.get("_status"):
                continue  # 同日已有待批同向申请 → 整天跳过,别补半个 in/out
            req("POST", f"/attendance/requests/{r['id']}/approve", tok,
                {"note": "属实"})
            r = req("POST", "/me/attendance-requests", emp_tok, {
                "ts": f"{wd}T18:1{wd.day % 6}:00Z", "kind": "out",
                "reason": "下班赶车忘打"}, quiet=True)
            if not r.get("_status"):
                req("POST", f"/attendance/requests/{r['id']}/approve", tok,
                    {"note": "属实"})
            print(f"  {emp['name']} {wd} 补卡 in/out 已批 → present")
    req("POST", "/attendance/summarize", tok,
        {"start": str(month_start), "end": str(today)})
    print("  本月考勤重算完成")

    # =====================================================================
    # ATS / 员工服务 / 其余模块 —— 每个模块 2~3 条,全部按名称查重幂等
    # =====================================================================
    admin_uid = req("GET", "/auth/me", tok)["id"]

    # ---------- 技能字典 ----------
    print("== 技能 ==")
    skills_map = {s["name"]: s for s in req("GET", "/skills?limit=200", tok)}
    for nm, cat in (("React", "tech"), ("Python", "language"), ("沟通能力", "soft")):
        if nm not in skills_map:
            skills_map[nm] = req("POST", "/skills", tok, {"name": nm, "category": cat})
    print(f"  技能 {len(skills_map)} 项")

    # ---------- 岗位 ----------
    print("== 岗位 ==")
    jobs = {j["title"]: j for j in req("GET", "/jobs/?limit=200", tok).get("items", [])}
    job_specs = [
        ("高级前端工程师", "advanced", ["React", "TypeScript", "Node.js"],
         "负责 C 端产品前端架构与组件库建设,推动性能与体验优化。"),
        ("Python 后端工程师", "intermediate", ["Python", "FastAPI", "PostgreSQL"],
         "负责业务 API、异步任务与数据管线,支撑招聘与人事流程。"),
        ("产品设计师", "intermediate", ["Figma", "用户研究"],
         "负责移动端改版,从用户调研到高保真交付全流程。"),
    ]
    for title, level, sks, desc in job_specs:
        if title not in jobs:
            jobs[title] = req("POST", "/jobs/", tok, {
                "title": title, "level": level, "skills": sks, "description": desc,
            })
            print(f"  岗位 {title} → {jobs[title]['id'][:8]}")
    # 治理流:前两个岗位提交审批并发布,产品设计师留 draft 演示待审态
    for title in ("高级前端工程师", "Python 后端工程师"):
        j = jobs[title]
        if j.get("publish_status") == "draft":
            req("POST", f"/jobs/{j['id']}/submit-approval", tok)
            j["publish_status"] = "pending_approval"
        if j.get("publish_status") == "pending_approval":
            req("POST", f"/jobs/{j['id']}/approve", tok, {"note": "编制内,同意发布"})
            j["publish_status"] = "published"
        if j.get("status") != "open":
            req("PATCH", f"/jobs/{j['id']}/status", tok, {"status": "open"})
    print(f"  岗位 {len(jobs)} 个({jobs['高级前端工程师'].get('publish_status')}/"
          f"{jobs['Python 后端工程师'].get('publish_status')}/{jobs['产品设计师'].get('publish_status')})")

    # ---------- 简历上传(自动生成候选人) ----------
    print("== 简历/候选人 ==")
    resume_specs = [
        ("chenmo.txt", "陈默\nchen.mo@example.com / 13800000001\n"
         "5 年前端开发,React + TypeScript + Node.js,主导 3 个中后台项目重构。\n期望:高级前端工程师"),
        ("qianqi.txt", "钱七\nqian.qi@example.com / 13800000002\n"
         "4 年后端,Python + FastAPI + PostgreSQL,熟悉异步任务与高并发接口优化。\n期望:Python 后端工程师"),
        ("sunyue.txt", "孙悦\nsun.yue@example.com / 13800000003\n"
         "3 年产品设计,Figma + 用户研究,独立完成从调研到高保真的全流程。\n期望:产品设计师"),
    ]
    resume_items = req("GET", "/resumes/?limit=200", tok).get("items", [])
    have_files = {r["file_name"] for r in resume_items}
    for fn, text in resume_specs:
        if fn not in have_files:
            r = req_upload("/resumes/upload", tok, fn, text.encode("utf-8"))
            if r and not r.get("_status"):
                print(f"  {fn} → resume {r['id'][:8]}")
            else:
                print(f"  !! {fn} 上传失败 {r}")
    resume_items = req("GET", "/resumes/?limit=200", tok).get("items", [])
    FILE_NAMES = {"chenmo": "陈默", "qianqi": "钱七", "sunyue": "孙悦"}
    cand_ids = {}
    for r in resume_items:
        c = r.get("candidate") or {}
        if not c.get("id"):
            continue
        stem = (r.get("file_name") or "").rsplit(".", 1)[0]
        want = FILE_NAMES.get(stem)
        if want and c.get("name") != want:
            req("PATCH", f"/resumes/{r['id']}", tok, {"name": want}, quiet=True)
            c["name"] = want
        if c.get("name"):
            cand_ids[c["name"]] = c["id"]
    print(f"  候选人 {len(cand_ids)} 位:{','.join(cand_ids)}")

    # ---------- 人才分组 ----------
    print("== 人才分组 ==")
    groups = req("GET", "/talent-groups", tok)
    grp = next((g for g in groups if g["name"] == "重点跟进"), None)
    if grp is None:
        grp = req("POST", "/talent-groups", tok,
                  {"name": "重点跟进", "description": "高分或用人经理点名的候选人"})
        print(f"  分组 重点跟进 → {grp['id'][:8]}")
    member_cids = [cid for cname, cid in cand_ids.items() if cname in ("陈默", "钱七")]
    if member_cids:
        req("POST", f"/talent-groups/{grp['id']}/members", tok,
            {"candidate_ids": member_cids}, quiet=True)

    # ---------- 招聘管道 ----------
    print("== 招聘管道 ==")
    STAGES = ["applied", "screening", "interview", "offer", "hired"]
    apps = req("GET", "/applications/?limit=200", tok).get("items", [])
    for cname, jtitle, stage in (
        ("陈默", "高级前端工程师", "screening"),
        ("钱七", "Python 后端工程师", "interview"),
        ("孙悦", "产品设计师", "applied"),
    ):
        cid, jid = cand_ids.get(cname), jobs.get(jtitle, {}).get("id")
        if not cid or not jid:
            continue
        cur = next((a for a in apps
                    if a["candidate_id"] == cid and a["job_id"] == jid), None)
        if cur is None:
            cur = req("POST", "/applications", tok,
                      {"candidate_id": cid, "job_id": jid, "note": "种子导入"})
            apps.append(cur)
            print(f"  {cname} → {jtitle}")
        while (cur["stage"] in STAGES and stage in STAGES
               and STAGES.index(cur["stage"]) < STAGES.index(stage)):
            cur = req("POST", f"/applications/{cur['id']}/transition", tok,
                      {"to_stage": STAGES[STAGES.index(cur["stage"]) + 1]})

    # ---------- 岗位技能模型 + 候选人技能档案 ----------
    print("== 技能档案 ==")
    fe_job, py_job = jobs["高级前端工程师"], jobs["Python 后端工程师"]
    req("PUT", f"/skills/jobs/{fe_job['id']}", tok, {"items": [
        {"name": "React", "importance": "must"},
        {"name": "TypeScript", "importance": "must"},
        {"name": "沟通能力", "importance": "nice"},
    ]}, quiet=True)
    req("PUT", f"/skills/jobs/{py_job['id']}", tok, {"items": [
        {"name": "Python", "importance": "must"},
        {"name": "沟通能力", "importance": "nice"},
    ]}, quiet=True)
    for cname, items in (
        ("陈默", [{"name": "React", "level": 5}, {"name": "沟通能力", "level": 4}]),
        ("钱七", [{"name": "Python", "level": 4}, {"name": "沟通能力", "level": 3}]),
    ):
        cid = cand_ids.get(cname)
        if cid:
            req("PUT", f"/skills/candidates/{cid}", tok, {"items": items}, quiet=True)

    # ---------- 题库 ----------
    print("== 题库 ==")
    qtexts = {q["question"] for q in
              req("GET", "/question-library/?limit=200", tok).get("items", [])}
    for q, kind, diff in (
        ("描述 React 并发渲染的工作原理与适用场景", "tech", "advanced"),
        ("讲一个你主导过的跨团队项目,如何推进?", "project", "intermediate"),
        ("线上出现严重故障时,你的应急处理流程?", "scenario", "intermediate"),
    ):
        if q not in qtexts:
            req("POST", "/question-library/", tok, {
                "question": q, "kind": kind, "difficulty": diff,
                "category": "通用", "answer_points": ["要点一", "要点二"],
            })
    print("  题目就绪")

    # ---------- AI 面试 + 面试安排(评分卡) ----------
    print("== 面试 ==")
    ivs = req("GET", "/interviews/?limit=100", tok).get("items", [])
    iv_keys = {(i.get("candidate_name") or (i.get("candidate") or {}).get("name"),
                i.get("job_title") or (i.get("job") or {}).get("title")) for i in ivs}
    for cname, jtitle in (("陈默", "高级前端工程师"), ("钱七", "Python 后端工程师")):
        cid, jid = cand_ids.get(cname), jobs.get(jtitle, {}).get("id")
        if cid and jid and (cname, jtitle) not in iv_keys:
            r = req("POST", "/interviews/start", tok, {
                "job_id": jid, "candidate_id": cid, "question_count": 3,
                "kinds": ["tech", "project"], "delivery": "link",
                "expires_in_hours": 72,
            }, quiet=True)
            if not r.get("_status"):
                print(f"  AI 面试 {cname} × {jtitle} → {r.get('invite_path', '')}")
    scs = req("GET", "/scorecards/?limit=100", tok)
    sc_list = scs.get("items", scs) if isinstance(scs, dict) else scs
    tomorrow = today + timedelta(days=1)
    for cname, jtitle, tm in (
        ("陈默", "高级前端工程师", "10:00"),
        ("孙悦", "产品设计师", "14:00"),
    ):
        cid, jid = cand_ids.get(cname), jobs.get(jtitle, {}).get("id")
        exists = any(s.get("candidate_id") == cid for s in sc_list) if cid else True
        if cid and not exists:
            req("POST", "/scorecards", tok, {
                "candidate_id": cid, "job_id": jid,
                "title": f"{jtitle} · 一面",
                "scheduled_at": f"{tomorrow}T{tm}:00",
                "interviewer_ids": [admin_uid],
                "criteria": [{"name": "专业能力", "dimension": "tech", "weight": 3},
                             {"name": "沟通表达", "dimension": "general", "weight": 2}],
            }, quiet=True)
            print(f"  面试安排 {cname} {tomorrow} {tm}")

    # ---------- 评分规则 ----------
    print("== 评分规则 ==")
    rules = {r["name"]: r for r in req("GET", "/scoring-rules", tok)}
    if "技术岗通用五维" not in rules:
        req("POST", "/scoring-rules", tok, {
            "name": "技术岗通用五维", "description": "技术岗位默认评分规则",
            "dimensions": [
                {"name": "专业能力", "weight": 30},
                {"name": "项目经验", "weight": 25},
                {"name": "沟通协作", "weight": 20},
                {"name": "学习能力", "weight": 15},
                {"name": "文化匹配", "weight": 10},
            ],
        }, quiet=True)
        print("  评分规则 技术岗通用五维")

    # ---------- 问卷调查 ----------
    print("== 问卷调查 ==")
    surveys = {s["title"]: s for s in req("GET", "/surveys", tok)}
    if "面试体验反馈" not in surveys:
        s = req("POST", "/surveys", tok, {
            "title": "面试体验反馈",
            "description": "候选人面试流程满意度调查",
            "questions": [
                {"type": "rating", "text": "整体面试体验评分", "order": 0},
                {"type": "single_choice", "text": "面试官专业度如何",
                 "options": ["很好", "一般", "较差"], "order": 1},
                {"type": "long_text", "text": "其他建议", "required": False, "order": 2},
            ],
        })
        req("POST", f"/surveys/{s['id']}/status", tok, {"status": "published"})
        surveys["面试体验反馈"] = s
        print("  问卷 面试体验反馈 已发布")
    s1 = req("GET", f"/surveys/{surveys['面试体验反馈']['id']}", tok)
    if s1.get("status") == "draft":
        req("POST", f"/surveys/{surveys['面试体验反馈']['id']}/status", tok,
            {"status": "published"})
    links = req("GET", f"/surveys/{surveys['面试体验反馈']['id']}/links", tok)
    if not links and cand_ids.get("陈默"):
        req("POST", f"/surveys/{surveys['面试体验反馈']['id']}/links", tok,
            {"candidate_id": cand_ids["陈默"], "expires_in_days": 14})
        print("  链接已生成(绑定陈默)")

    # ---------- 会议室 + 预约 ----------
    print("== 会议室 ==")
    rooms = {r["name"]: r for r in req("GET", "/rooms", tok)}
    for nm, cap, eq in (
        ("水星厅", 6, ["投屏", "白板"]),
        ("火星厅", 10, ["视频会议", "投屏"]),
    ):
        if nm not in rooms:
            rooms[nm] = req("POST", "/rooms", tok,
                            {"name": nm, "capacity": cap, "equipment": eq})
    resources = {r["name"]: r for r in req("GET", "/resources", tok)}
    if "便携投影仪" not in resources:
        req("POST", "/resources", tok,
            {"name": "便携投影仪", "type": "equipment"})
    res_list = req("GET", f"/reservations?start={today}T00:00:00&end={today + timedelta(days=30)}T00:00:00", tok)
    if not any(r["title"] == "前端一面·陈默" for r in res_list):
        req("POST", "/reservations", tok, {
            "kind": "room", "room_id": rooms["水星厅"]["id"],
            "title": "前端一面·陈默",
            "starts_at": f"{tomorrow}T10:00:00", "ends_at": f"{tomorrow}T11:00:00",
            "interviewer_ids": [admin_uid],
        }, quiet=True)
        print(f"  预约 水星厅 {tomorrow} 10:00")

    # ---------- 公告 ----------
    print("== 公告 ==")
    anns = {a["title"] for a in req("GET", "/announcements?limit=100", tok).get("items", [])}
    for t_, c_ in (
        ("Q4 全员大会通知", "本周五 15:00 三层大会议室,同步年度目标进展。"),
        ("新版报销流程上线", "报销单现在支持明细子表与在线支付标记,详见知识库。"),
        ("周五弹性办公提醒", "本月最后一个周五可申请远程办公。"),
    ):
        if t_ not in anns:
            req("POST", "/announcements", tok, {"title": t_, "content": c_}, quiet=True)
    print("  公告就绪")

    # ---------- 员工服务请求 ----------
    print("== 员工请求 ==")
    my_reqs = req("GET", "/me/requests", emp1) or []
    my_subjects = {r["subject"] for r in my_reqs}
    if "申请新款笔记本" not in my_subjects:
        req("POST", "/me/requests", emp1, {
            "kind": "equipment", "subject": "申请新款笔记本",
            "detail": "现用 Mac 已 4 年,编译明显变慢,申请换新。"}, quiet=True)
    my_reqs2 = req("GET", "/me/requests", emp2) or []
    if "Python 进阶培训报名" not in {r["subject"] for r in my_reqs2}:
        req("POST", "/me/requests", emp2, {
            "kind": "training", "subject": "Python 进阶培训报名",
            "detail": "申请参加下月外部培训课程,预算约 3000 元。"}, quiet=True)
    print("  请求就绪")

    # ---------- HR 工单 ----------
    print("== HR 工单 ==")
    my_cases = req("GET", "/me/cases", emp2) or []
    if "VPN 无法连接" not in {c["subject"] for c in my_cases}:
        c = req("POST", "/me/cases", emp2, {
            "kind": "it", "priority": "high", "subject": "VPN 无法连接",
            "description": "居家办公连不上公司 VPN,影响代码拉取,请尽快处理。"}, quiet=True)
        if not c.get("_status"):
            req("POST", f"/cases/{c['id']}/assign", tok,
                {"assignee_user_id": admin_uid}, quiet=True)
            req("POST", f"/cases/{c['id']}/status", tok,
                {"status": "in_progress"}, quiet=True)
            print(f"  工单 VPN 无法连接 → 已指派处理中")
    my_cases1 = req("GET", "/me/cases", emp1) or []
    if "工资条金额疑问" not in {c["subject"] for c in my_cases1}:
        req("POST", "/me/cases", emp1, {
            "kind": "payroll", "subject": "工资条金额疑问",
            "description": "本月绩效奖金与考核结果不符,请 HR 核对。"}, quiet=True)
        print("  工单 工资条金额疑问")

    # ---------- 知识库 ----------
    print("== 知识库 ==")
    arts = {a["title"] for a in req("GET", "/knowledge/articles?limit=200", tok).get("items", [])}
    for t_, cat_, c_ in (
        ("报销制度说明", "policy",
         "报销需在发生后 30 天内提交;单笔超 500 元需附发票照片;差旅报销按城市等级定额。"),
        ("年假与调休 FAQ", "faq",
         "年假按自然年发放,入职满一年后可全额使用;加班审批通过后按 8h=1 天折算调休额度。"),
        ("新人 IT 指引", "it",
         "入职首日领取笔记本并开通企业邮箱;VPN 账号由 IT 在 24h 内下发;WiFi 使用 HRON-Staff。"),
    ):
        if t_ not in arts:
            req("POST", "/knowledge/articles", tok,
                {"title": t_, "category": cat_, "content": c_, "tags": ["seed"]},
                quiet=True)
    print("  知识库就绪")

    # ---------- 职位编制 ----------
    print("== 职位编制 ==")
    positions = req("GET", "/positions", tok).get("items", [])
    pos_titles = {p["title"] for p in positions}
    for t_, ou in (("资深前端工程师", dev), ("后端工程师", dev), ("产品设计师", prod)):
        if t_ not in pos_titles:
            req("POST", "/positions", tok,
                {"title": t_, "org_unit_id": ou["id"],
                 "note": "年度编制内"}, quiet=True)
    positions = req("GET", "/positions", tok).get("items", [])
    pos = next((p for p in positions
                if p["title"] == "资深前端工程师" and not p.get("incumbent_id")), None)
    if pos:
        req("POST", f"/positions/{pos['id']}/fill", tok,
            {"employee_id": emp_by_no["EMP002"]["id"]}, quiet=True)
    print(f"  编制 {len(positions)} 个")

    # ---------- 项目 ----------
    print("== 项目 ==")
    projs = req("GET", "/projects", tok).get("items", [])
    proj_names = {p["name"] for p in projs}
    for nm, desc in (
        ("新版候选人门户", "候选人侧门户改版,目标提升面试完成率"),
        ("薪酬体系年度评审", "对齐市场分位,完成全员薪酬区间评审"),
    ):
        if nm not in proj_names:
            req("POST", "/projects", tok, {
                "name": nm, "description": desc, "status": "active",
                "owner_employee_id": emp_by_no["EMP002"]["id"],
                "start_date": str(today - timedelta(days=10)),
            }, quiet=True)
    print("  项目就绪")

    # ---------- 排班 ----------
    print("== 排班 ==")
    shift_map = {s["name"]: s for s in req("GET", "/shifts", tok)}
    for nm, st, en, col in (("早班", 540, 1020, "#22c55e"), ("晚班", 840, 1380, "#f59e0b")):
        if nm not in shift_map:
            shift_map[nm] = req("POST", "/shifts", tok,
                                {"name": nm, "start_min": st, "end_min": en,
                                 "required": 2, "color": col})
    assigns = req("GET", "/shifts/assignments?limit=200", tok).get("items", [])
    next_mon = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    for no, sname, off in (("EMP002", "早班", 0), ("EMP003", "晚班", 0), ("EMP002", "早班", 1)):
        e = emp_by_no[no]
        day = next_mon + timedelta(days=off)
        if not any(a["employee_id"] == e["id"] and a["day"] == str(day) for a in assigns):
            req("POST", "/shifts/assignments", tok, {
                "employee_id": e["id"], "shift_id": shift_map[sname]["id"],
                "day": str(day)}, quiet=True)
    print(f"  班次 {[s['name'] for s in shift_map.values()]}")

    # ---------- 薪酬区间 + 调薪评审 ----------
    print("== 薪酬 ==")
    bands = {b["name"] for b in req("GET", "/salary-bands?limit=100", tok).get("items", [])}
    for nm, lo, hi in (("P5 高级工程师", 20000, 35000), ("P3 中级工程师", 12000, 20000)):
        if nm not in bands:
            req("POST", "/salary-bands", tok, {
                "name": nm, "gross_min": lo, "gross_max": hi,
                "org_unit_id": dev["id"]}, quiet=True)
    matrix = req("GET", "/comp-review/matrix?cycle=default", tok, quiet=True)
    if not matrix.get("_status"):
        decided = {d["employee_id"] for d in matrix.get("decisions", [])}
        for no, pct in (("EMP002", 8.0), ("EMP003", 5.0)):
            e = emp_by_no[no]
            if e["id"] not in decided:
                req("POST", "/comp-review/decisions", tok, {
                    "employee_id": e["id"], "cycle": "default",
                    "final_pct": pct, "note": "年度调薪"}, quiet=True)

    # ---------- 自动化规则 ----------
    print("== 自动化 ==")
    rules_auto = req("GET", "/automations?limit=100", tok).get("items", [])
    rule_names = {r["name"] for r in rules_auto}
    if "新员工入职提醒" not in rule_names:
        req("POST", "/automations", tok, {
            "name": "新员工入职提醒", "entity_type": "employee",
            "trigger_kind": "audit", "trigger": "create",
            "action": "notify", "title_tpl": "新员工已建档:{{name}}",
            "body_tpl": "请 HR 完成后续入职任务安排。",
        }, quiet=True)
    if "报销审批通过提醒" not in rule_names:
        req("POST", "/automations", tok, {
            "name": "报销审批通过提醒", "entity_type": "expense_claim",
            "trigger_kind": "audit", "trigger": "approve",
            "action": "notify", "title_tpl": "报销单已通过:{{title}}",
        }, quiet=True)
    print("  自动化就绪")

    # ---------- 收藏筛选 ----------
    filters = req("GET", "/saved-filters", tok)
    filter_names = {f["name"] for f in filters}
    for nm, page, params in (
        ("高分候选人", "talents", {"min_score": 80}),
        ("在途面试", "interviews", {"status": "in_progress"}),
    ):
        if nm not in filter_names:
            req("POST", "/saved-filters", tok,
                {"page": page, "name": nm, "params": params}, quiet=True)

    # ---------- 外联活动 ----------
    camps = req("GET", "/campaigns?limit=100", tok).get("items", [])
    if not any(c["name"] == "银牌候选人召回" for c in camps):
        req("POST", "/campaigns", tok, {
            "name": "银牌候选人召回",
            "subject": "新岗位机会邀请",
            "body_text": "您好,我们有与您经历高度匹配的新岗位,欢迎回访。",
            "filter": {"query": ""},
        }, quiet=True)
        print("  外联活动 银牌候选人召回(草稿)")

    # ---------- 审批策略 ----------
    pols = req("GET", "/approvals/policies?limit=100", tok).get("items", [])
    if not any(p["request_type"] == "employee_request" for p in pols):
        req("POST", "/approvals/policies", tok, {
            "name": "员工请求上级审批", "request_type": "employee_request",
            "steps": [{"resolver": "manager", "label": "直属上级"}],
        }, quiet=True)
        print("  审批策略 员工请求→上级")

    # ---------- 早报/记忆中枢:今日面试 + 待跟进 Offer + 决策数据 ----------
    print("== 早报/记忆中枢 ==")
    # 1) 今日面试:钱七排一张今天的卡(明天的两张不变,两栏都有数据)
    scs2 = req("GET", "/scorecards/?limit=100", tok)
    sc_list2 = scs2.get("items", scs2) if isinstance(scs2, dict) else scs2
    if cand_ids.get("钱七") and not any(
            s.get("candidate_id") == cand_ids["钱七"] for s in sc_list2):
        req("POST", "/scorecards", tok, {
            "candidate_id": cand_ids["钱七"], "job_id": py_job["id"],
            "title": "Python 后端工程师 · 一面",
            "scheduled_at": f"{today}T15:00:00",
            "interviewer_ids": [admin_uid],
            "criteria": [{"name": "专业能力", "dimension": "tech", "weight": 3},
                         {"name": "沟通表达", "dimension": "general", "weight": 2}],
        }, quiet=True)
        print(f"  面试安排 钱七 {today} 15:00")

    # 2) 待跟进 Offer:draft + sent 各一
    offer_items = req("GET", "/offers?limit=100", tok).get("items", [])
    have_status = {o["status"] for o in offer_items}
    if "draft" not in have_status and cand_ids.get("孙悦"):
        req("POST", "/offers", tok, {
            "candidate_id": cand_ids["孙悦"], "job_id": jobs["产品设计师"]["id"],
            "position": "产品设计师", "letter": "期待你的加入。",
            "salary": {"base": 18000, "months": 13},
            "start_date": str(today + timedelta(days=30)),
        }, quiet=True)
        print("  Offer(draft) 孙悦·产品设计师")
    if "sent" not in have_status and cand_ids.get("陈默"):
        o = req("POST", "/offers", tok, {
            "candidate_id": cand_ids["陈默"], "job_id": fe_job["id"],
            "position": "高级前端工程师",
            "salary": {"base": 28000, "months": 14},
            "start_date": str(today + timedelta(days=21)),
        }, quiet=True)
        if not o.get("_status"):
            req("POST", f"/offers/{o['id']}/send", tok, quiet=True)
            print("  Offer(sent) 陈默·高级前端工程师")

    # 3) 决策数据:新简历 + 跨岗位申请,等 match done 再流转,
    #    DecisionFeedback 攒到 >=20 出「高置信度」;含 1 条 rejected 负样本
    resume_items = req("GET", "/resumes/?limit=200", tok).get("items", [])
    have_files = {r["file_name"] for r in resume_items}
    for fn, text in (
        ("zhoukai.txt", "姓名:周凯\nzhou.kai@example.com / 13800000004\n"
         "6 年 DevOps/后端,Python + Go + K8s,主导 CI/CD 与可观测体系。\n期望:Python 后端工程师"),
        ("wuqian.txt", "姓名:吴倩\nwu.qian@example.com / 13800000005\n"
         "4 年前端,Vue/React + 工程化,负责过设计系统落地。\n期望:高级前端工程师"),
    ):
        if fn not in have_files:
            req_upload("/resumes/upload", tok, fn, text.encode("utf-8"))
    FILE_NAMES.update({"zhoukai": "周凯", "wuqian": "吴倩"})
    for r in req("GET", "/resumes/?limit=200", tok).get("items", []):
        c = r.get("candidate") or {}
        stem = (r.get("file_name") or "").rsplit(".", 1)[0]
        want = FILE_NAMES.get(stem)
        if c.get("id") and want and c.get("name") != want:
            req("PATCH", f"/resumes/{r['id']}", tok, {"name": want}, quiet=True)
            c["name"] = want
        if c.get("id") and c.get("name"):
            cand_ids[c["name"]] = c["id"]

    DECISION_PLAN = [
        ("周凯", "Python 后端工程师", ["screening", "interview"]),
        ("吴倩", "高级前端工程师", ["screening", "interview"]),
        ("陈默", "Python 后端工程师", ["screening", "interview"]),
        ("孙悦", "高级前端工程师", ["rejected"]),
        ("钱七", "高级前端工程师", ["screening"]),
        ("周凯", "高级前端工程师", ["screening"]),
    ]
    for jid in (fe_job["id"], py_job["id"]):
        req("POST", f"/matches/job/{jid}/evaluate-all", tok, quiet=True)
    needed = {(cand_ids.get(c), jobs[j]["id"])
              for c, j, _ in DECISION_PLAN if cand_ids.get(c)}
    done_pairs = set()
    for _ in range(24):  # 最多等 ~120s 让 match 评估完
        for jid in {j for _, j in needed}:
            for m in req("GET", f"/matches/job/{jid}?status=done", tok):
                if m.get("candidate_id"):
                    done_pairs.add((m["candidate_id"], jid))
        if needed <= done_pairs:
            break
        time.sleep(5)
    print(f"  match done {len(done_pairs & needed)}/{len(needed)} 对")

    apps = req("GET", "/applications/?limit=200", tok).get("items", [])
    made = 0
    for cname, jtitle, stages in DECISION_PLAN:
        cid, jid = cand_ids.get(cname), jobs.get(jtitle, {}).get("id")
        if not cid or not jid or (cid, jid) not in done_pairs:
            continue
        cur = next((a for a in apps
                    if a["candidate_id"] == cid and a["job_id"] == jid), None)
        if cur is None:
            cur = req("POST", "/applications", tok,
                      {"candidate_id": cid, "job_id": jid}, quiet=True)
            if cur.get("_status"):
                continue
        for st in stages:
            if cur.get("stage") == st:
                continue
            cur = req("POST", f"/applications/{cur['id']}/transition", tok,
                      {"to_stage": st}, quiet=True)
            if cur.get("_status"):
                break
            made += 1
    print(f"  决策流转 {made} 次")

    # ---------- 匹配评估(异步,尽力触发不等待) ----------
    print("== AI 匹配触发 ==")
    for jtitle in ("高级前端工程师", "Python 后端工程师"):
        jid = jobs.get(jtitle, {}).get("id")
        if jid:
            r = req("POST", f"/matches/job/{jid}/evaluate-all", tok, quiet=True)
            if not r.get("_status"):
                print(f"  {jtitle} 匹配评估已排队")

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
    sql_path = os.path.join(os.path.dirname(__file__), "_seed_checkins.sql")
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("=" * 50)
    print("完成。账号:")
    print(f"  管理员   admin@example.com / {ADMIN_PASSWORD}")
    print("  HR      hr@example.com / 上方「已创建/已重置」行的一次性密码")
    print(f"  员工 ESS emp1@hronfire.dev / {EMP_PASSWORD} (李晓芸)")
    print(f"  员工 ESS emp2@hronfire.dev / {EMP_PASSWORD} (王浩)")
    print("历史打卡: docker exec -i free-hire-postgres-1 psql -U freehire -d freehire < backend/_seed_checkins.sql")
    print("然后 POST /api/attendance/summarize 重算本月。")


if __name__ == "__main__":
    main()
