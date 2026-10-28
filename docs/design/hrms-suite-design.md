# HROnfire 员工核心(HRIS)设计

> 以 Employee 为核心实体的领域模型,
> 招聘/假勤/生命周期/绩效/费用/轻量薪酬全部挂在 Employee 上。
> License 功能位:`hr.suite`(Professional+ 档)。

## 数据模型(全部 tenant 隔离)

### Employee — 员工主数据
`employee_no`(租户内唯一) / name / org_unit_id(部门,复用 OrgUnit) /
designation(职位) / reports_to_id(直属上级,自引用) /
user_id(→users,唯一可空,**员工账号绑定**) /
work_email / phone / hire_date / employment_type(full_time|part_time|intern|contractor) /
status(active|leave|suspended|terminated) /
personal(JSON:address/emergency_contact/…) /
source_candidate_id(→candidates,录用转化来源)

### 员工账号(ESS,Employee Self Service)
- `users.role = "employee"`,`employee.user_id` 双向绑定
- 员工账号**无业务权限项**,全部走 `/api/me/*` 自助端点,行级过滤 `employee_id == me`
- HR 端 `/api/employees/*` 等管理端点走权限项,employee 角色天然拿不到
- 直属上级(hiring_manager 且绑定了 employee)可审批自己下属的请假/报销:
  `team_employee_ids(u) = employees where reports_to_id == u.employee_id`

### 假期(P1)
- `leave_types`: name / annual_days / is_paid
- `leave_allocations`: employee × type × year × days(额度发放)
- `leave_applications`: employee/type/起止/days(支持 0.5 半天)/reason/
  status(pending→approved|rejected|cancelled)/approver/decided_at/note
- `leave_ledger_entries`: 台账,allocation(+)/application(-) 各写一条,
  余额 = alloc(+) + ledger(-) 汇总;审批通过时才写 -delta
- `employee_checkins`: ts / kind(in|out) / method(web)
- `attendances`: employee × date 唯一,status(present|absent|leave|remote|holiday) /
  first_in / last_out / work_hours;由 checkin 汇总 + 请假单联动 + HR 手改

### 生命周期(P2)
- `lifecycle_templates`: kind(onboarding|separation|transfer|promotion) × tasks JSON
- `lifecycle_events`: employee/kind/status(in_progress|completed|cancelled)/
  effective_date/payload JSON(调动的 from/to 部门等)
- `lifecycle_tasks`: event × title × done × done_at × sort;
  全部勾选 → 事件自动 completed

### 招聘借力(P2)
- `job_requisitions`: 编制需求审批。org_unit/title/headcount/reason/level,
  hiring_manager 提,admin|hr 批;审批通过可一键转 Job(status=draft→published)
- `job_offers`: candidate_id/job_id/position/salary JSON/start_date/letter(md)/
  status(draft→sent→accepted|declined|withdrawn);
  **accepted → 创建 Employee(source_candidate_id)+ onboarding 事件**,
  这是招聘→人事的转化口

### 绩效 / 费用 / 轻量薪酬(P4)
- `goals`: employee/title/desc/due_date/progress(0-100)/status
- `appraisal_cycles`: name/period/status(draft|open|closed)
- `appraisals`: cycle×employee 唯一,self_review+self_rating → manager_review+
  manager_rating,status(draft|self_submitted|reviewed)
- `expense_types` + `expense_claims`(items 子表,total 聚合,审批流同请假)
- `salary_slips`: employee × period('2026-10'),gross/deductions/net + items JSON
  明细。**只登记不核算**(不接外部会计系统,明确不做),
  HR 录入/编辑,员工只读明细 + **电子签收**(confirmed_at,收回重发需重签,
  HR 可催签收,发放/催办走 Notification)

### 人事提醒 + 人事日历(慧人力/2号人事部同款)
- 数据源实时聚合不落库:`employees.personal` JSON 键 `birth_date`/`probation_end`/
  `contract_end`/`cert_expire` + `hire_date`(周年)+ LifecycleEvent(待入职/离职)
- `GET /api/hr/reminders?days=` 分组(逾期/今天/本周/30天内);`GET /api/hr/calendar?month=`
  月视图(周期事件按年展开,2/29→平年2/28)
- `run_people_reminders` 挂 `_watch_loop`:deadline 类进 7 天窗口推 HR;
  生日/周年当天推 HR + **员工本人祝福**;同日同事件 link fragment 去重
- `POST /api/hr/reminders/greet` HR 一键送祝福给员工(同日幂等)

### 明确不做
- 自动薪资核算、税务申报、排班 Roster 拖拽、工时 Timesheet、印度本地税

## 权限项新增
```
manage_employees   admin,hr          员工档案管理
manage_leave       admin,hr          假期类型/额度
approve_leave      admin,hr,hiring_manager  请假审批(HM 仅下属)
manage_attendance  admin,hr          考勤修正
manage_lifecycle   admin,hr          入职/调动/离职
write_requisitions admin,hr,hiring_manager  提需求
approve_requisitions admin,hr        批需求
manage_offers      admin,hr          Offer
manage_performance admin,hr          考核周期/评分
manage_expenses    admin,hr          报销类型/标记支付
approve_expenses   admin,hr,hiring_manager  报销审批(HM 仅下属)
manage_payroll     admin,hr          工资条
```
employee 角色:无权限项,仅 `/api/me/*`。

### 读边界(审查收口)
- 敏感读按 manage 权限而非 view_reports:员工档案 `/api/employees*`、
  Offer `/api/offers*`、额度流水 `/api/leave/allocations`、生命周期事件
  `/api/lifecycle/events*` — viewer/interviewer 一律 403。
- `/api/employees/org-chart` 是目录级信息(姓名/职位/部门),内部角色可读。
- 职责分离:审批人不能批自己提交的请假/报销/招聘需求
  (申请人绑定的 user_id == 当前用户 → 403)。

## API 结构
- 管理端:`/api/employees` `/api/leave/*` `/api/attendance` `/api/checkins`
  `/api/lifecycle/*` `/api/requisitions` `/api/offers` `/api/performance/*`
  `/api/expenses/*` `/api/payroll/*`
- 员工端:`/api/me/profile` `/api/me/leave-*` `/api/me/checkins` `/api/me/attendance`
  `/api/me/salary-slips` `/api/me/expense-claims` `/api/me/goals` `/api/me/appraisals`
- 下属可见:`/api/leave/applications?scope=team` 等,manager 自动过滤 reports_to

## 导航(两级)
一级新增:**组织**(员工列表/组织架构图/生命周期) **假勤**(请假审批/假期额度/
假期类型/考勤记录) **绩效**(目标/考核) **费用**(报销审批) **薪酬**(工资条)。
招聘模块下「招聘需求审批」由规划项变实装。
**employee 角色导航整组替换**:我的工作台(资料/请假/考勤/工资条/报销/目标),
通过 `useVisibleNav` 按 role 分流——role=="employee" 只返回 ESS 模块。

## 路由(前端)
`/employees` `/employees/:id` `/org-chart` `/lifecycle` `/leave` `/attendance`
`/performance` `/expenses` `/payroll` `/requisitions` `/offers`
ESS:`/me` `/me/leave` `/me/attendance` `/me/payslips` `/me/expenses` `/me/goals` `/me/profile`
