"""HROnfire 演示租户全功能 mock 数据 —— 幂等,可重复执行。

用法(容器内):  python /app/scripts/seed_demo.py
覆盖:组织/员工、假勤(类型/额度/申请/考勤/异常/补卡/加班/日历)、
排班、绩效(目标/考核)、费用、薪酬(工资条/带宽)、审批、公告、
申请、工单、知识库已有、项目与人才画像、生命周期、题库、
招聘需求/Offer/评分卡/外联、自动化、筛选器/订阅/通知。
"""
import os, sys, uuid
sys.path.insert(0, "/app")
from datetime import date, datetime, time, timedelta
from sqlalchemy import select, text

from app.infra.db import SessionLocal
from app.infra.security import hash_password
from app.domain.models import (
    Announcement, Appraisal, AppraisalCycle, ApprovalPolicy, Attendance,
    AttendanceFlag, AttendanceRequest, AutomationRule, Employee,
    EmployeeRequest, EmployeeSkill, ExpenseClaim, ExpenseItem, ExpenseType,
    Goal, Holiday, HrCase, InsightNote, Interview, InterviewScorecard,
    JobApplication, JobOffer, JobRequisition, LeaveAllocation,
    LeaveApplication, LeavePolicy, LeaveType, LifecycleEvent, LifecycleTask,
    LifecycleTemplate, Notification, OrgUnit, OvertimeRequest, Project,
    ProjectMember, QuestionLibraryItem, ReportSubscription, SalaryBand,
    SalarySlip, SavedFilter, Shift, ShiftAssignment, Skill, TalentCampaign,
    User, Job, Candidate, DecisionFeedback, KnowledgeArticle, RecruitRun,
    RecruitRunStep, ScoringRule,
)
from app.services.approvals import route_request

ADMIN_EMAIL = "admin@example.com"
db = SessionLocal()
made = {"n": 0}

def q(model, **kw):
    return db.scalars(select(model).where(
        model.tenant_id == T, *[getattr(model, k) == v for k, v in kw.items()]
    )).first()

def add(obj):
    db.add(obj); made["n"] += 1; return obj

def ensure(model, defaults=None, **kw):
    obj = q(model, **kw)
    if obj:
        return obj
    return add(model(tenant_id=T, **kw, **(defaults or {})))

def emp(name):
    return db.scalars(select(Employee).where(
        Employee.tenant_id == T, Employee.name == name)).first()

admin = db.scalars(select(User).where(User.email == ADMIN_EMAIL)).first()
assert admin, "admin 账号不存在"
T = str(admin.tenant_id)          # 跟随 admin 所在租户,本地/线上通用
ADMIN_UID = admin.id

# ============ 0. 演示账号引导(幂等,跨环境) ============
def ensure_user(email, password, role):
    u = db.scalars(select(User).where(User.email == email)).first()
    if not u:
        u = add(User(tenant_id=T, email=email, password_hash=hash_password(password),
                     role=role, status="active"))
        db.flush()
    return u

def ensure_emp(name, designation, user=None):
    e = emp(name)
    if not e:
        no = "D" + uuid.uuid4().hex[:6].upper()
        e = add(Employee(tenant_id=T, name=name, designation=designation,
                         hire_date=date(2022, 1, 10), employee_no=no,
                         work_email=f"{no.lower()}@hronfire.demo",
                         employment_type="full_time", status="active"))
        db.flush()
    return e

mgr_user = ensure_user("mgr-uie2e@example.com", "mgr123456", "hiring_manager")
emp_user = ensure_user("emp-uie2e@example.com", "emp123456", "employee")
_boss = ensure_emp("卢品超", "HR 负责人")
ensure_emp("UI经理", "工程经理")
ensure_emp("UI员工", "前端工程师")
# 账号↔档案绑定(只在未绑定时写,不动既有绑定)
if not db.scalars(select(Employee).where(Employee.tenant_id == T, Employee.user_id == admin.id)).first():
    _boss.user_id = admin.id
if not db.scalars(select(Employee).where(Employee.tenant_id == T, Employee.user_id == mgr_user.id)).first():
    emp("UI经理").user_id = mgr_user.id
if not db.scalars(select(Employee).where(Employee.tenant_id == T, Employee.user_id == emp_user.id)).first():
    emp("UI员工").user_id = emp_user.id
db.flush()

# ============ 1. 组织与员工 ============
org_eng = ensure(OrgUnit, name="平台工程组", kind="department")
org_pd = ensure(OrgUnit, name="产品部", kind="department")
org_dsn = ensure(OrgUnit, name="设计组", kind="department")
org_hr = ensure(OrgUnit, name="人力资源部", kind="department")
# 清掉测试残留组织
for junk in db.scalars(select(OrgUnit).where(
        OrgUnit.tenant_id == T, OrgUnit.name.in_(["回归测试部", "带宽回归部"]))):
    db.delete(junk)
db.flush()  # org.id 供下面员工引用

boss = emp("卢品超"); boss.org_unit_id = org_hr.id
uimgr = emp("UI经理"); uimgr.org_unit_id = org_eng.id
uiemp = emp("UI员工"); uiemp.org_unit_id = org_eng.id
uimgr.reports_to_id = boss.id          # 经理也挂在卢品超下,树更深
boss.designation = "HR 负责人"

PEOPLE = [
    # name, designation, org, reports_to, hire_date, emp_no
    ("张伟", "高级前端工程师", org_eng, uimgr, date(2023, 3, 15), "E0101"),
    ("王芳", "后端工程师", org_eng, uimgr, date(2024, 6, 1), "E0102"),
    ("李强", "测试工程师", org_eng, uimgr, date(2025, 2, 10), "E0103"),
    ("赵敏", "产品经理", org_pd, boss, date(2022, 11, 20), "E0104"),
    ("陈晨", "UI设计师", org_dsn, None, date(2024, 9, 5), "E0105"),
    ("周杰", "前端工程师", org_eng, uimgr, date(2025, 8, 18), "E0106"),
]
emps = {}
for name, desig, org, mgr, hd, no in PEOPLE:
    e = emp(name)
    if not e:
        e = add(Employee(tenant_id=T, name=name, designation=desig,
                         org_unit_id=org.id, reports_to_id=mgr.id if mgr else None,
                         hire_date=hd, employee_no=no,
                         work_email=f"{no.lower()}@hronfire.demo",
                         employment_type="full_time", status="active"))
        db.flush()
    emps[name] = e
zhaomin = emps["赵敏"]
emps["陈晨"].reports_to_id = zhaomin.id   # 设计师汇报给产品
db.flush()
ALL = [boss, uimgr, uiemp] + [emps[n] for n, *_ in PEOPLE]
print("1. org/employees ok:", len(ALL))

# ============ 2. 假期类型 / 政策 / 额度 ============
lt_nj = ensure(LeaveType, name="年假", defaults=dict(annual_days=12, is_paid=True))
lt_sick = ensure(LeaveType, name="病假", defaults=dict(annual_days=10, is_paid=True))
lt_pers = ensure(LeaveType, name="事假", defaults=dict(annual_days=5, is_paid=False))
lt_comp = ensure(LeaveType, name="调休", defaults=dict(annual_days=0, is_paid=True, is_comp=True))
db.flush()  # lt_*.id 是列默认,先落库再引用
pol = ensure(LeavePolicy, name="标准假期包",
             defaults=dict(items=[{"leave_type_id": lt_nj.id, "days": 12},
                                  {"leave_type_id": lt_sick.id, "days": 10}],
                           carry_forward=5))
db.flush()  # pol.id 供下面引用
for e in ALL:
    e.leave_policy_id = pol.id
    for lt, d in ((lt_nj, 12), (lt_sick, 10), (lt_pers, 5)):
        if not q(LeaveAllocation, employee_id=e.id, leave_type_id=lt.id, year=2026):
            add(LeaveAllocation(tenant_id=T, employee_id=e.id,
                                leave_type_id=lt.id, year=2026, days=d,
                                note="2026 年度额度"))
db.flush()
print("2. leave types/allocations ok")

# ============ 2.5 审批策略(须在 route_request 之前建好) ============
# 两级请假链:第二级从 role=hr(本租户无)改为 role=admin
two_lv = db.scalars(select(ApprovalPolicy).where(
    ApprovalPolicy.tenant_id == T,
    ApprovalPolicy.name == "UI创建-请假3天以上两级")).first()
if two_lv and two_lv.steps and two_lv.steps[-1].get("resolver") == "role" \
        and two_lv.steps[-1].get("key") == "hr":
    steps = list(two_lv.steps)
    steps[-1] = {"resolver": "role", "key": "admin", "label": "HR 复核"}
    two_lv.steps = steps
if two_lv:
    two_lv.priority = 10   # 高于单级策略,days>=3 稳定命中两级链

def policy(name, rtype, steps, priority=5, conditions=None):
    ensure(ApprovalPolicy, name=name,
           defaults=dict(request_type=rtype, priority=priority,
                         conditions=conditions or [], steps=steps,
                         active=True, created_by=ADMIN_UID))

USER_STEP = [{"resolver": "user", "key": ADMIN_UID, "label": "HR 审批"}]
policy("费用报销-HR审批", "expense_claim", USER_STEP)
policy("员工申请-HR受理", "employee_request", USER_STEP)
policy("补卡申请-考勤管理员", "attendance_request", USER_STEP)
db.flush()
print("2.5 approval policies ok")

# ============ 3. 请假申请(已批+驳回+待批进收件箱) ============
def leave_app(e, lt, s, days, status, reason, note=None):
    a = q(LeaveApplication, employee_id=e.id, start_date=s, leave_type_id=lt.id)
    if a:
        return a
    a = add(LeaveApplication(tenant_id=T, employee_id=e.id, leave_type_id=lt.id,
                             start_date=s, end_date=s + timedelta(days=int(days) - 1),
                             days=days, reason=reason, status=status,
                             approver_id=ADMIN_UID if status != "pending" else None,
                             decided_at=datetime(2026, 10, 20) if status != "pending" else None,
                             decision_note=note))
    db.flush()
    if status == "pending":
        route_request(db, T, "leave", a)
    return a

leave_app(emps["张伟"], lt_nj, date(2026, 10, 12), 2, "approved", "家里有事回老家", "同意,注意交接")
leave_app(emps["王芳"], lt_sick, date(2026, 10, 8), 1, "approved", "感冒发烧就医")
leave_app(emps["赵敏"], lt_nj, date(2026, 11, 16), 3, "pending", "孩子学校秋游,陪同请假")
leave_app(emps["李强"], lt_pers, date(2026, 10, 27), 1, "pending", "办理房产过户手续")
leave_app(emps["陈晨"], lt_nj, date(2026, 10, 19), 1, "rejected", "个人事务", "发版周,建议改期")
leave_app(emps["周杰"], lt_comp, date(2026, 10, 30), 0.5, "pending", "上周末加班调休")
leave_app(uimgr, lt_nj, date(2026, 12, 24), 2, "pending", "年底休假")
print("3. leave applications ok")

# ============ 4. 考勤记录 + 异常 + 补卡 + 加班 ============
db.execute(text("UPDATE attendances SET status='present' WHERE status='normal'"))
OCT_DAYS = [d for d in (date(2026, 10, x) for x in range(1, 24))
            if d.weekday() < 5]
leave_days = {(a.employee_id, a.start_date): a for a in db.scalars(
    select(LeaveApplication).where(LeaveApplication.tenant_id == T,
                                   LeaveApplication.status == "approved"))}
for e in ALL:
    for i, d in enumerate(OCT_DAYS):
        if q(Attendance, employee_id=e.id, day=d):
            continue
        if e is emps["李强"] and d == date(2026, 10, 22):
            continue   # 留给后面的 absent 记录
        if (e.id, d) in leave_days:
            st, fi, lo, wh = "leave", None, None, None
        else:
            st = "present"
            fi = datetime(d.year, d.month, d.day, 9, 0) + timedelta(minutes=(i * 7 + hash(e.name)) % 20)
            lo = datetime(d.year, d.month, d.day, 18, 10) + timedelta(minutes=(i * 11) % 40)
            wh = round((lo - fi).seconds / 3600 - 1.0, 1)
        add(Attendance(tenant_id=T, employee_id=e.id, day=d, status=st,
                       first_in=fi, last_out=lo, work_hours=wh, source="seed"))
# 一天缺勤
if not q(Attendance, employee_id=emps["李强"].id, day=date(2026, 10, 22)):
    add(Attendance(tenant_id=T, employee_id=emps["李强"].id,
                   day=date(2026, 10, 22), status="absent", source="seed"))
db.flush()

def flag(e, d, kind, minutes, excused=False):
    if q(AttendanceFlag, employee_id=e.id, day=d, kind=kind):
        return
    att = q(Attendance, employee_id=e.id, day=d)
    add(AttendanceFlag(tenant_id=T, employee_id=e.id, attendance_id=att.id if att else None,
                       day=d, kind=kind, minutes=minutes, excused=excused,
                       detail={"source": "seed"},
                       excused_by_id=ADMIN_UID if excused else None,
                       excused_note="暴雨导致延误" if excused else None,
                       excused_at=datetime(2026, 10, 21) if excused else None))

flag(emps["张伟"], date(2026, 10, 21), "late_in", 28)
flag(emps["李强"], date(2026, 10, 23), "late_in", 47)
flag(emps["王芳"], date(2026, 10, 20), "late_in", 15, excused=True)
flag(emps["周杰"], date(2026, 10, 22), "early_out", 35)

def att_req(e, ts, kind, reason):
    if not q(AttendanceRequest, employee_id=e.id, ts=ts, kind=kind):
        r = add(AttendanceRequest(tenant_id=T, employee_id=e.id, ts=ts, kind=kind,
                                  reason=reason, status="pending"))
        db.flush(); route_request(db, T, "attendance_request", r)

att_req(emps["王芳"], datetime(2026, 10, 23, 9, 5), "in", "早上在客户现场开会,忘记打卡")
att_req(emps["李强"], datetime(2026, 10, 22, 18, 40), "out", "加班离开时漏打下班卡")

def overtime(e, d, hours, status, reason):
    r = q(OvertimeRequest, employee_id=e.id, work_date=d)
    if r:
        return
    r = add(OvertimeRequest(tenant_id=T, employee_id=e.id, work_date=d, hours=hours,
                            leave_type_id=lt_comp.id, reason=reason, status=status,
                            approver_id=ADMIN_UID if status != "pending" else None,
                            decided_at=datetime(2026, 10, 19) if status != "pending" else None,
                            comp_days=hours / 8 if status == "approved" else None))
    db.flush()
    if status == "pending":
        route_request(db, T, "overtime", r)

overtime(emps["张伟"], date(2026, 10, 17), 4, "approved", "周末发版支持")
overtime(emps["周杰"], date(2026, 10, 24), 3, "pending", "赶 Q4 评审材料")
print("4. attendance/flags/requests/overtime ok")

# ============ 5. 节假日历 + 排班 ============
for d, name, kind in [(date(2027, 1, 1), "元旦", "holiday"),
                      (date(2027, 1, 4), "元旦调休上班", "workday"),
                      (date(2027, 2, 17), "春节", "holiday")]:
    ensure(Holiday, day=d, defaults=dict(name=name, kind=kind))
s_am = db.scalars(select(Shift).where(Shift.tenant_id == T, Shift.name == "早班")).first()
s_pm = db.scalars(select(Shift).where(Shift.tenant_id == T, Shift.name == "晚班")).first()
week = [date(2026, 10, 26) + timedelta(days=i) for i in range(5)]
for e, shift, days in [(emps["张伟"], s_am, week), (emps["王芳"], s_am, week[:3]),
                       (emps["李强"], s_pm, week[2:]), (emps["周杰"], s_am, week[1:4])]:
    if not shift:
        continue
    for d in days:
        if not q(ShiftAssignment, employee_id=e.id, day=d):
            add(ShiftAssignment(tenant_id=T, employee_id=e.id, shift_id=shift.id,
                                day=d, source="manual", created_by=ADMIN_UID))
print("5. calendar/shifts ok")

# ============ 6. 绩效:目标 + 考核周期 ============
for e, title, desc, prog in [
    (uimgr, "带领团队完成审批中心推广", "覆盖请假/报销/加班全部场景", 70),
    (emps["张伟"], "组件库 v2 架构落地", "完成 20 个核心组件迁移", 45),
    (emps["王芳"], "审批引擎性能优化", "P99 延迟 < 300ms", 60),
    (emps["赵敏"], "输出 2027 产品路线图", "含 Q1 排期与人力评估", 30),
    (emps["李强"], "自动化测试覆盖率 80%", "核心链路全量回归", 55)]:
    ensure(Goal, employee_id=e.id, title=title,
           defaults=dict(description=desc, due_date=date(2026, 12, 31),
                         progress=prog, status="active"))

cycle = ensure(AppraisalCycle, name="2026 Q4 绩效考核",
               defaults=dict(period_start=date(2026, 10, 1),
                             period_end=date(2026, 12, 31), status="open"))
db.flush()  # cycle.id
for e, st, self_r, mgr_r, self_tx, mgr_tx in [
    (emps["张伟"], "reviewed", 4, 5, "本季度完成组件库架构升级,主导 3 次技术分享", "架构贡献突出,注意文档沉淀"),
    (emps["王芳"], "self_submitted", 4, None, "完成审批引擎优化,线上事故 0 起", None),
    (emps["李强"], "self_submitted", 3, None, "回归覆盖率提升到 78%", None),
    (emps["赵敏"], "draft", None, None, None, None)]:
    a = q(Appraisal, cycle_id=cycle.id, employee_id=e.id)
    if not a:
        add(Appraisal(tenant_id=T, cycle_id=cycle.id, employee_id=e.id, status=st,
                      self_review=self_tx, self_rating=self_r,
                      manager_review=mgr_tx, manager_rating=mgr_r,
                      reviewer_id=ADMIN_UID if st == "reviewed" else None))
print("6. performance ok")

# ============ 7. 费用类型 + 报销单 ============
et_travel = ensure(ExpenseType, name="差旅费")
et_meal = ensure(ExpenseType, name="餐饮招待费")
et_office = ensure(ExpenseType, name="办公用品")
db.flush()  # et_*.id

def claim(e, title, status, items, note=None):
    c = q(ExpenseClaim, employee_id=e.id, title=title)
    if c:
        return c
    total = sum(i[2] for i in items)
    c = add(ExpenseClaim(tenant_id=T, employee_id=e.id, title=title, total=total,
                         status=status,
                         approver_id=ADMIN_UID if status != "pending" else None,
                         decided_at=datetime(2026, 10, 21) if status != "pending" else None,
                         decision_note=note))
    db.flush()
    for tid, on, amt, note_i in items:
        add(ExpenseItem(tenant_id=T, claim_id=c.id, type_id=tid,
                        occurred_on=on, amount=amt, note=note_i))
    if status == "pending":
        db.flush(); route_request(db, T, "expense_claim", c)
    return c

claim(emps["张伟"], "10月北京客户现场差旅", "pending",
      [(et_travel.id, date(2026, 10, 14), 1156, "高铁往返"),
       (et_travel.id, date(2026, 10, 15), 528, "酒店两晚"),
       (et_meal.id, date(2026, 10, 15), 320, "客户工作餐")])
claim(emps["王芳"], "9月服务器备件采购", "approved",
      [(et_office.id, date(2026, 9, 28), 890, "SSD 硬盘×2")], "按预算执行")
claim(emps["李强"], "10月团队加班餐费", "pending",
      [(et_meal.id, date(2026, 10, 17), 268, "发版加班晚餐"),
       (et_meal.id, date(2026, 10, 21), 189, "加班夜宵")])
print("7. expenses ok")

# ============ 8. 工资条(全员 8/9/10 月)+ 薪酬带宽 ============
PAY = {"卢品超": (26000, 4000, 4800, 2900), "UI经理": (23000, 3000, 4200, 2300),
       "UI员工": (18000, 2500, 3200, 1450), "张伟": (24000, 3500, 4500, 2600),
       "王芳": (20000, 2800, 3600, 1800), "李强": (15000, 1800, 2700, 900),
       "赵敏": (25000, 3800, 4600, 2800), "陈晨": (17000, 2200, 3000, 1200),
       "周杰": (14000, 1500, 2500, 750)}
for e in ALL:
    base, bonus, ins, tax = PAY[e.name]
    for p in ("2026-08", "2026-09", "2026-10"):
        if q(SalarySlip, employee_id=e.id, period=p):
            continue
        items = [{"label": "基本工资", "kind": "earning", "amount": base},
                 {"label": "绩效奖金", "kind": "earning", "amount": bonus},
                 {"label": "社保公积金", "kind": "deduction", "amount": ins},
                 {"label": "个税", "kind": "deduction", "amount": tax}]
        add(SalarySlip(tenant_id=T, employee_id=e.id, period=p, items=items,
                       gross=base + bonus, deductions=ins + tax,
                       net=base + bonus - ins - tax, currency="CNY",
                       status="issued", issued_at=datetime(2026, 10, 25)))
for name, desig, org, lo, hi in [
    ("高级工程师 P6", "高级前端工程师", org_eng, 18000, 32000),
    ("工程师 P5", "后端工程师", org_eng, 14000, 24000),
    ("产品经理 P7", "产品经理", org_pd, 22000, 38000),
    ("设计师 P5", "UI设计师", org_dsn, 13000, 22000)]:
    ensure(SalaryBand, name=name,
           defaults=dict(designation=desig, org_unit_id=org.id, currency="CNY",
                         gross_min=lo, gross_max=hi,
                         employer_cost_min=lo * 1.3, employer_cost_max=hi * 1.35,
                         effective_from=date(2026, 1, 1), is_active=True))
print("8. payroll/bands ok")

# ============ 9. 公告 / 申请 / 工单 ============
for title, body in [
    ("2026 Q4 绩效考核启动", "请各位同学在 11 月 15 日前完成自评提交,主管复核截止 11 月 30 日。\n\n考核入口:我的 → 我的目标。"),
    ("11 月团建活动报名", "本月团建定为 11 月 21 日(周六)郊外徒步+烧烤,往返大巴接送。\n请本周五前在群里接龙报名。"),
    ("关于规范报销票据的通知", "即日起差旅报销请附行程单与发票抬头一致,否则财务将退回。")]:
    ensure(Announcement, title=title,
           defaults=dict(content=body, status="published",
                         published_at=datetime(2026, 10, 20, 9, 0),
                         created_by=ADMIN_UID))

def request(e, kind, subject, detail, status, note=None):
    r = q(EmployeeRequest, employee_id=e.id, subject=subject)
    if r:
        return
    r = add(EmployeeRequest(tenant_id=T, employee_id=e.id, kind=kind, subject=subject,
                            detail=detail, status=status,
                            decided_by=ADMIN_UID if status != "pending" else None,
                            decided_at=datetime(2026, 10, 22) if status != "pending" else None,
                            decision_note=note))
    db.flush()
    if status == "pending":
        route_request(db, T, "employee_request", r)

request(emps["赵敏"], "training", "申请参加产品经理年度峰会", "11月底上海PM峰会,预算3800元,含门票与差旅。", "pending")
request(emps["陈晨"], "equipment", "申请更换 4K 显示器", "设计稿对色准要求高,现显示器色域不足。", "pending")
request(uimgr, "general", "申请开通测试环境管理员权限", "团队联调需要,按流程申请。", "approved", "已开通")
request(emps["王芳"], "benefit", "补充商业保险家属信息", "为配偶添加补充医疗险。", "pending")

def case(e, subject, kind, prio, status, desc, resolution=None):
    if q(HrCase, subject=subject):
        return
    add(HrCase(tenant_id=T, subject=subject, kind=kind, priority=prio, status=status,
               description=desc, requester_employee_id=e.id,
               assignee_user_id=ADMIN_UID if status != "new" else None,
               resolution=resolution,
               resolved_at=datetime(2026, 10, 19) if status == "resolved" else None,
               case_no=abs(hash(subject)) % 90000 + 10000))

case(emps["张伟"], "社保缴费基数与实际工资不符", "payroll", "high", "in_progress",
     "9月工资条显示社保基数按最低标准,请核实调整。")
case(emps["王芳"], "申请开具收入证明", "admin", "normal", "resolved",
     "办理签证需要近6个月收入证明。", resolution="已开具并发送至个人邮箱")
case(emps["陈晨"], "请假记录显示有误", "leave_dispute", "normal", "new",
     "10月19日的事假被记成了年假,请协助更正。")
print("9. announcements/requests/cases ok")

# ============ 10. 项目 + 人才画像(技能/洞察) ============
def skill(name, cat=""):
    s = db.scalars(select(Skill).where(Skill.tenant_id == T, Skill.name == name)).first()
    if not s:
        s = add(Skill(tenant_id=T, name=name, category=cat)); db.flush()
    return s

def add_skill(e, sname, level, source="manual", status="confirmed", cat=""):
    s = skill(sname, cat)
    if q(EmployeeSkill, employee_id=e.id, skill_id=s.id, status=status):
        return
    add(EmployeeSkill(tenant_id=T, employee_id=e.id, skill_id=s.id, level=level,
                      source=source, status=status, created_by=ADMIN_UID))

for e, skills in [
    (emps["张伟"], [("TypeScript", 5), ("系统设计", 4), ("前端架构", 4)]),
    (emps["王芳"], [("Python", 4), ("PostgreSQL", 3), ("系统设计", 3)]),
    (emps["李强"], [("自动化测试", 4), ("Python", 3)]),
    (emps["赵敏"], [("需求分析", 5), ("用户研究", 3)]),
    (emps["陈晨"], [("视觉设计", 4), ("交互设计", 3)]),
    (emps["周杰"], [("TypeScript", 3), ("React", 3)])]:
    for sn, lv in skills:
        add_skill(e, sn, lv)
add_skill(emps["周杰"], "React Native", 3, source="ai", status="suggested")
add_skill(emps["周杰"], "工程管理", 2, source="ai", status="suggested")
add_skill(emps["李强"], "性能测试", 3, source="ai", status="suggested")
db.flush()

def note(e, kind, title, body, vis="managers"):
    if q(InsightNote, employee_id=e.id, title=title):
        return
    add(InsightNote(tenant_id=T, employee_id=e.id, kind=kind, title=title, body=body,
                    author_user_id=ADMIN_UID, visibility=vis))

note(emps["张伟"], "meeting", "Q4 技术评审纪要",
     "张伟主导组件库 v2 架构方案评审,提出分层依赖方案获一致通过;建议补文档沉淀。")
note(emps["王芳"], "manager_note", "三季度表现小结",
     "稳定性负责到位,审批引擎优化按期交付;可尝试带新人。")
note(emps["赵敏"], "milestone", "审批中心需求评审完成",
     "完成多级审批+代理人方案定稿,预计 Q4 全量上线。")

# 项目:保留现有,补一个 done 一个 active
pj_done = ensure(Project, name="官网 Q3 改版",
                 defaults=dict(description="市场官网视觉与内容改版", status="done",
                               owner_employee_id=zhaomin.id,
                               start_date=date(2026, 7, 1), end_date=date(2026, 9, 30),
                               required_skills=["视觉设计", "交互设计"], created_by=ADMIN_UID))
pj_active = ensure(Project, name="移动端 App 重构",
                   defaults=dict(description="React Native 重构混合应用", status="active",
                                 owner_employee_id=emps["张伟"].id,
                                 start_date=date(2026, 10, 12),
                                 required_skills=["React", "TypeScript", "React Native"],
                                 created_by=ADMIN_UID))
db.flush()

def member(pj, e, role, via="manual", pct=50, contrib=None):
    if not q(ProjectMember, project_id=pj.id, employee_id=e.id):
        add(ProjectMember(tenant_id=T, project_id=pj.id, employee_id=e.id, role=role,
                          joined_via=via, allocation_pct=pct,
                          contribution_summary=contrib))

member(pj_done, emps["陈晨"], "设计负责人", pct=60,
       contrib="主导改版视觉方案,输出 40+ 页面稿")
member(pj_done, emps["赵敏"], "产品负责人", pct=40, contrib="需求梳理与验收")
member(pj_done, uimgr, "技术评审", pct=20, contrib="技术方案把关")
member(pj_active, emps["张伟"], "技术负责人", via="ai_match", pct=70)
member(pj_active, emps["周杰"], "前端开发", via="ai_match", pct=80)
member(pj_active, uiemp, "前端开发", pct=50)
print("10. projects/profile ok")

# ============ 11. 生命周期模板 + 进行中入职 ============
tpl_join = ensure(LifecycleTemplate, name="标准入职流程", kind="join",
                  defaults=dict(tasks=["开通账号与权限", "准备工位与设备", "签署劳动合同",
                                       "入职培训安排", "建立员工档案"], is_default=True))
tpl_leave = ensure(LifecycleTemplate, name="标准离职流程", kind="leave",
                   defaults=dict(tasks=["工作交接清单", "资产归还确认", "账号注销",
                                        "离职面谈", "开具离职证明"], is_default=True))
ev = q(LifecycleEvent, employee_id=emps["周杰"].id, kind="join")
if not ev:
    ev = add(LifecycleEvent(tenant_id=T, employee_id=emps["周杰"].id, kind="join",
                            status="in_progress", effective_date=date(2025, 8, 18),
                            created_by=ADMIN_UID))
    db.flush()
    for i, t in enumerate(["开通账号与权限", "准备工位与设备", "签署劳动合同",
                           "入职培训安排", "建立员工档案"]):
        add(LifecycleTask(tenant_id=T, event_id=ev.id, title=t, sort=i,
                          done=i < 3,
                          done_at=datetime(2025, 8, 18) if i < 3 else None))
print("11. lifecycle ok")

# ============ 12. 题库 ============
QLIB = [
    ("请讲一次你主导的系统重构经历,遇到过什么坑?", "tech", "advanced", "系统设计", "系统设计",
     "考察真实主导经验:动机/方案/取舍/结果"),
    ("React 18 的并发特性是什么?useTransition 解决什么问题?", "tech", "intermediate", "前端", "React",
     "并发渲染概念、useTransition 使用场景"),
    ("设计一个多级审批流引擎,核心数据模型怎么建?", "design", "advanced", "系统设计", "系统设计",
     "策略/实例/步骤模型、条件路由、代理与回退"),
    ("如果一个项目deadline临近但质量不达标,你怎么办?", "scenario", "intermediate", "通用", "工程管理",
     "风险上报、范围取舍、质量底线"),
    ("写一段 SQL 求各部门近三月人均请假天数。", "tech", "intermediate", "数据库", "PostgreSQL",
     "JOIN+GROUP BY+窗口函数,边界处理"),
    ("如何向非技术背景的产品经理解释技术债务?", "soft", "initial", "通用", "沟通协作",
     "比喻与业务语言表达"),
]
for qt, kind, diff, cat, sk, ans in QLIB:
    if not q(QuestionLibraryItem, question=qt):
        add(QuestionLibraryItem(tenant_id=T, question=qt, kind=kind, difficulty=diff,
                                category=cat, skill=sk, answer_points=ans,
                                source="manual", created_by=ADMIN_UID))
print("12. question library ok")

# ============ 13. 招聘补齐:需求 / Offer / 评分卡 / 管道 / 外联 ============
job_any = db.scalars(select(Job).where(Job.tenant_id == T)).first()
cands = db.scalars(select(Candidate).where(Candidate.tenant_id == T).limit(6)).all()
req = ensure(JobRequisition, title="平台工程组 · 高级后端工程师",
             defaults=dict(org_unit_id=org_eng.id, headcount=2, level="P6",
                           reason="审批引擎二期人力缺口", status="pending",
                           requested_by=mgr_user.id if mgr_user else ADMIN_UID))
req2 = ensure(JobRequisition, title="设计组 · 资深 UI 设计师",
              defaults=dict(org_unit_id=org_dsn.id, headcount=1, level="P6",
                            reason="移动端重构配套设计人力", status="approved",
                            requested_by=ADMIN_UID, approver_id=ADMIN_UID,
                            decided_at=datetime(2026, 10, 15),
                            decision_note="同意,纳入 Q4 编制", job_id=job_any.id if job_any else None))
if job_any and cands:
    ensure(JobOffer, candidate_id=cands[0].id,
           defaults=dict(job_id=job_any.id, position="高级前端工程师",
                         salary="24-32K·14薪", start_date=date(2026, 11, 16),
                         status="sent", sent_at=datetime(2026, 10, 23),
                         created_by=ADMIN_UID))
    # 管道补几个阶段
    STAGES = ["applied", "screening", "interview", "offer"]
    for i, c in enumerate(cands[:4]):
        if not q(JobApplication, candidate_id=c.id, job_id=job_any.id):
            add(JobApplication(tenant_id=T, candidate_id=c.id, job_id=job_any.id,
                               stage=STAGES[i % 4], source="talent_pool",
                               owner_user_id=ADMIN_UID,
                               stage_changed_at=datetime(2026, 10, 20)))
# 面试评分卡(挂在已有面试/候选人上)
ivs = db.scalars(select(Interview).where(Interview.tenant_id == T)).all()
if ivs:
    iv = ivs[0]
    ensure(InterviewScorecard, interview_id=iv.id,
           defaults=dict(candidate_id=iv.candidate_id, job_id=iv.job_id,
                         title="技术面试评分卡",
                         criteria=[{"key": "tech", "label": "技术深度", "weight": 40},
                                   {"key": "comm", "label": "沟通表达", "weight": 30},
                                   {"key": "fit", "label": "团队匹配", "weight": 30}],
                         interviewer_ids=[ADMIN_UID], status="submitted",
                         ai_summary="技术扎实,架构思路清晰;建议二面考察协作场景。",
                         created_by=ADMIN_UID))
camp = ensure(TalentCampaign, name="Q4 前端人才再激活",
              defaults=dict(subject="我们有个新机会想和你聊聊",
                            body_text="基于你之前的面试记录,目前有一个更匹配的高级前端岗位开放中…",
                            status="done", total=8, sent_count=7, failed_count=1,
                            created_by=ADMIN_UID,
                            started_at=datetime(2026, 10, 20, 10, 0),
                            done_at=datetime(2026, 10, 20, 10, 12)))
print("13. recruiting extras ok")

# ============ 14. 自动化 / 筛选器 / 订阅 / 通知 ============
ensure(AutomationRule, name="请假批准自动通知",
       defaults=dict(entity_type="leave_application", trigger="updated",
                     watch_fields=["status"],
                     conditions=[{"field": "status", "op": "in", "value": ["approved"]}],
                     action="notify",
                     recipients=[{"kind": "record_owner"}],
                     title_tpl="你的请假申请已批准",
                     body_tpl="{employee_name} 的 {leave_type} {days} 天请假已通过",
                     is_active=True, created_by=ADMIN_UID))
ensure(SavedFilter, name="近30天活跃候选人", page="talents",
       defaults=dict(user_id=ADMIN_UID,
                     params={"activity": "30d", "status": "active"}))
ensure(ReportSubscription, kind="weekly_summary", user_id=ADMIN_UID,
       defaults=dict(weekday=1, hour=9, is_active=True))

def notif(uid, kind, title, body, link=None):
    if not db.scalars(select(Notification).where(
            Notification.tenant_id == T, Notification.user_id == uid,
            Notification.title == title)).first():
        add(Notification(tenant_id=T, user_id=uid, kind=kind, title=title,
                         body=body, link=link))

notif(ADMIN_UID, "approval", "有 3 条申请待你审批", "请假×2、报销×1,请在审批中心处理。", "/approvals")
notif(ADMIN_UID, "system", "2026 Q4 考核已开启", "自评提交截止 11 月 15 日。", "/performance")
emp_user = db.scalars(select(User).where(User.email == "emp-uie2e@example.com")).first()
if emp_user:
    notif(emp_user.id, "announcement", "11 月团建活动报名", "本周五前群内接龙报名。", "/me")
print("14. automation/filters/subs/notifs ok")

# ============ 15. P0-P3 演示数据:知识库 / 评分规则 / 决策反馈 / 任务链 ============
# 注意:依赖 0067-0069 迁移后的表,seed 须在新后端就绪后执行(工作流有健康检查兜底)
# 知识文章(发布后供 AI 助手 RAG 引用,P2 溯源演示)
KB = [
    ("请假制度", "policy", "# 年假\n正式员工每年 10 天年假,入职满一年后生效,可分次使用。\n\n# 病假\n每年 15 天带薪病假,连续 3 天以上需附医院证明。\n\n# 事假\n无薪,单次不超过 5 天,需提前 3 个工作日申请。", ["请假", "假期"]),
    ("报销制度", "policy", "# 差旅\n高铁二等座/经济舱实报实销,住宿一线城市上限 500 元/晚。\n\n# 餐饮\n出差餐补 100 元/天,市内招待需事前审批。\n\n# 发票\n所有报销需附增值税发票,抬头为公司全称。", ["报销", "费用"]),
    ("考勤与弹性工作", "policy", "# 工时\n标准工时 9:00-18:00,弹性带宽 ±1 小时。\n\n# 远程\n每周可申请 1 天远程办公,需提前一天在系统登记。\n\n# 补卡\n每月可补卡 3 次,漏打卡需 48 小时内申请。", ["考勤", "远程"]),
    ("新人入职 FAQ", "onboarding", "# 入职首日\n携带身份证原件,9:30 到前台领取工卡与设备。\n\n# 账号\n企业邮箱与 VPN 账号入职当天自动开通。\n\n# 试用期\n试用期 3 个月,第 2 个月末进行中期面谈。", ["入职", "FAQ"]),
    ("AI 招聘官 SOP", "other", "# 标准招聘任务链\n1. 明确 JD 与评分规则\n2. 解析人才库候选人技能\n3. 按技能+语义检索扩源\n4. 评分规则批量打分\n5. 高分候选人发起面试\n6. 汇总报告并输出招聘汇报\n\n每步均由对应工具落地,状态在任务链卡片实时可见。",
     ["SOP", "招聘"], True),
]
for title, cat, content, tags, *pb in KB:
    ensure(KnowledgeArticle, title=title,
           defaults=dict(category=cat, content=content, tags=tags,
                         status="published", is_playbook=bool(pb and pb[0]),
                         updated_by=ADMIN_UID))
print("15a. knowledge articles ok")

# 评分规则(P3「学到了什么」需要一条带权重的规则)
rule_seed = ensure(ScoringRule, name="通用技术岗评分",
                   defaults=dict(
                       description="技术岗通用评分维度,权重来自历史校准",
                       dimensions=[
                           {"id": "tech", "name": "技术深度", "weight": 40,
                            "evaluator": {"method": "ai"}},
                           {"id": "proj", "name": "项目经验", "weight": 30,
                            "evaluator": {"method": "ai"}},
                           {"id": "comm", "name": "沟通协作", "weight": 30,
                            "evaluator": {"method": "ai"}},
                       ],
                       version=2, created_by=ADMIN_UID))
print("15b. scoring rule ok")

# 决策反馈:人为构造「沟通强的人被推进、项目经验高的反而被淘汰」的信号,
# 让「学到了什么」给出 提权沟通协作 / 降权项目经验 的可解释建议。
def snap(tech, proj, comm):
    return [
        {"key": "tech", "name": "技术深度", "weight": 40, "level": tech},
        {"key": "proj", "name": "项目经验", "weight": 30, "level": proj},
        {"key": "comm", "name": "沟通协作", "weight": 30, "level": comm},
    ]

FEEDBACK = [  # (decision, to_stage, tech, proj, comm, score)
    ("advanced", "interview", 4, 2, 4, 82),
    ("advanced", "interview", 3, 2, 4, 76),
    ("offered", "offer", 4, 3, 4, 88),
    ("advanced", "screening", 3, 2, 3, 71),
    ("offered", "offer", 4, 2, 4, 85),
    ("hired", "hired", 4, 3, 4, 90),
    ("rejected", "rejected", 3, 4, 1, 62),
    ("rejected", "rejected", 4, 4, 2, 68),
    ("rejected", "rejected", 3, 3, 1, 55),
    ("rejected", "rejected", 3, 4, 2, 64),
]
if job_any:
    fb_cands = db.scalars(
        select(Candidate).where(Candidate.tenant_id == T).limit(12)).all()
    fb_made = 0
    for i, (dec, to_st, t_lv, p_lv, c_lv, sc) in enumerate(FEEDBACK):
        if i >= len(fb_cands):
            break
        c = fb_cands[i]
        app_row = q(JobApplication, candidate_id=c.id, job_id=job_any.id)
        if not app_row:
            app_row = add(JobApplication(
                tenant_id=T, candidate_id=c.id, job_id=job_any.id,
                stage=("rejected" if dec == "rejected" else "interview"),
                source="talent_pool", owner_user_id=ADMIN_UID,
                stage_changed_at=datetime(2026, 10, 24)))
            db.flush()
        if not q(DecisionFeedback, application_id=app_row.id, to_stage=to_st):
            add(DecisionFeedback(
                tenant_id=T, job_id=job_any.id, candidate_id=c.id,
                application_id=app_row.id, decision=dec,
                from_stage="applied", to_stage=to_st, match_score=sc,
                criteria_snapshot=snap(t_lv, p_lv, c_lv),
                decided_by=ADMIN_UID,
                created_at=datetime(2026, 10, 24, 14, i)))
            fb_made += 1
    print(f"15c. decision feedback ok (+{fb_made})")

# 一条已完成的招聘任务链,作为 P1 历史示例
if not q(RecruitRun, title="高级前端 · 全流程招聘"):
    run = add(RecruitRun(tenant_id=T, user_id=ADMIN_UID, kind="full_pipeline",
                         title="高级前端 · 全流程招聘", status="done",
                         context={"job": job_any.title if job_any else ""},
                         created_at=datetime(2026, 10, 22, 9, 30),
                         finished_at=datetime(2026, 10, 22, 9, 41)))
    db.flush()
    DEMO_STEPS = [
        ("jd", "确认 JD 与评分规则", "Planner", "/jobs"),
        ("parse", "解析人才库技能标签", "Screener", "/talents"),
        ("source", "内外部渠道扩源", "Sourcer", "/talents"),
        ("score", "批量匹配打分", "Evaluator", "/scoring-rules"),
        ("shortlist", "产出高分候选人清单", "Screener", "/pipeline"),
        ("interview", "发起首轮面试", "Coordinator", "/interviews"),
        ("report", "生成招聘汇报", "Reporter", "/reports"),
    ]
    for seq, (k, lb, ag, lk) in enumerate(DEMO_STEPS, start=1):
        add(RecruitRunStep(
            run_id=run.id, seq=seq, key=k, label=lb, agent_label=ag,
            status="done", link=lk,
            detail={"note": "演示数据"},
            started_at=datetime(2026, 10, 22, 9, 30 + seq),
            finished_at=datetime(2026, 10, 22, 9, 31 + seq)))
print("15d. recruit run demo ok")

db.commit()
print(f"\nDONE — 新增 {made['n']} 行演示数据")
