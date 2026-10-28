# 事件驱动自动化 + 考勤深化设计

> 对应实现批次:**T7**(automation + attendance deepening)
> 状态:设计定稿 → 实现进行中

本文档定义两块功能的完整方案:

- **A. 事件驱动自动化**:业务写操作发生瞬间按租户规则触发通知/邮件,补齐
  `ai_watchers`(周期性)之外的"即时反应"层。
- **B. 考勤深化**:期望作息、宽限时间、迟到/早退异常记录、打卡 IP 白名单、
  月度工时汇总、免计(excuse)工作流。

两者都遵循既有架构约束:租户隔离、集中权限、组织数据域、审计留痕、
AI 工具只打 backend API。

---

## 0. 现状基线

| 已有 | 缺口 |
|---|---|
| `AiWatcher`:用户级、周期(5min)评估固定条件 → Notification | 无即时事件触发;条件硬编码 5 种;无自定义动作 |
| `write_audit`:所有业务写统一漏斗(实体+动作+detail+via) | 只落日志,不驱动行为 |
| `Notification`:站内统一投递 | — |
| SMTP 账号 + celery 邮件任务(invite/done 模式) | 无通用业务邮件动作 |
| `EmployeeCheckin` 原始打卡 → `summarize_day` → `Attendance` 日结 | 无期望作息,无法判迟到早退 |
| `WorkCalendar`:workdays + workday_hours(每租户一行) | 无 work_start/work_end |
| 补卡申请审批流(method=request 回补 checkin 重算) | — |
| 无 | 打卡 IP 白名单、迟到早退台账、月度工时汇总、免计工作流 |

---

## A. 事件驱动自动化

### A.1 概念模型

```
业务写操作 → write_audit(已有漏斗,result=success)
              └→ automation.dispatch()
                    ├ 匹配 automation_rules(tenant, entity_type, trigger/action)
                    ├ on_update:watch_fields ∩ detail.fields 非空才继续
                    ├ 条件评估(snapshot 白名单字段,AND 语义)
                    ├ 收件人解析(actor / role / user / record_owner / record_manager)
                    ├ 动作:
                    │   notify → Notification 行(同事务落库)
                    │   email  → automation_runs 记 payload → after_commit → celery
                    └ automation_runs 记账 + audit(automation_fire)
```

**关键取舍:不从 Django 信号照搬,而是挂在 `write_audit` 上**。理由:

1. `write_audit` 已经是全部业务写的统一入口(60+ API 文件零改动获得全域覆盖,
   包括 `entities.py` 泛型写、`leave.approve`、`talents.merge` 等具名动作)。
2. 审计 `action` 词汇天然就是事件词汇:`create/update/delete` 之外还有
   `approve`、`reject`、`transition`、`merge`、`manual_correct` 等具名事件,
   比只有 CUD 三态的信号模型更强。
3. 同事务语义:通知行与业务写同生共死 —— 业务回滚则通知回滚,无孤儿通知。
4. 天然防递归:Notification 写入不经过 write_audit;规则表本身不在实体注册表里,
   "自动化触发自动化"无路径。

### A.2 数据模型(迁移 0052)

`automation_rules`(每租户):

| 列 | 说明 |
|---|---|
| `name` | 规则名,如"请假批准通知员工" |
| `entity_type` | 目标实体,白名单注册表键(见 A.4) |
| `trigger` | `create`/`update`/`delete` 或 `custom` |
| `action_name` | trigger=custom 时匹配的审计动作名(如 `approve`) |
| `watch_fields` | JSON 数组;trigger=update 时仅当 `detail.fields` 变更键 ∩ 此集非空才评估;空数组=任意变更 |
| `conditions` | JSON 数组 `[{field, op, value}]`,AND 语义 |
| `action` | `notify` / `email` |
| `recipients` | JSON 数组收件人规格(见 A.5) |
| `title_tpl` / `body_tpl` | `{{field}}`/`{{actor}}` 占位模板 |
| `is_active` | 开关 |
| `created_by` / `last_fired_at` / `created_at` / `updated_at` | 元数据 |

`automation_runs`(记账/幂等/可观测):

| 列 | 说明 |
|---|---|
| `rule_id` + `event_key` | `UniqueConstraint` — event_key = 触发它的 audit_log.id,天然幂等(同一审计行对同一规则只会执行一次,重试/双发被唯一约束拦截) |
| `entity_type` / `entity_id` / `trigger` / `action` | 事件上下文 |
| `status` | `ok` / `skipped_no_recipients` / `queued_email` / `sent` / `email_failed` / `error` |
| `recipients` | JSON,实际解析到的 user_id/email 列表 |
| `payload` | JSON,渲染后的 title/body(+ 邮件收件地址) |
| `error` | 失败原因 |
| `created_at` | — |

### A.3 条件与快照

实体注册表 `ENTITY_REGISTRY[entity_type]` = `{model, snapshot(db,obj)->dict,
owner_user_id(db,obj), employee_id(db,obj), link(obj)}`。快照返回扁平
`{field: scalar}` 白名单 —— **规则创建时校验 field 必须在快照键集合内,防任意
ORM 穿透**。v1 注册实体:`candidates, jobs, interviews, employees,
leave_applications, employee_requests, hr_cases, attendance, attendance_requests,
overtime_requests, offers, job_applications, announcements`。

条件算子:`eq ne gt lt ge le contains in exists`(`exists` 判非空/非 None);
数值自动 float 化比较,字符串 contains/in 按 str 处理。

模板渲染:`title_tpl`/`body_tpl` 中 `{{field}}` → snapshot 值,`{{actor}}` →
操作者 email;`link` 由注册表生成(如 `/talents/{id}`),通知可点跳转。

### A.4 触发匹配

```
audit.action ∈ {"create","import","bulk"}            → trigger="create"
               ∈ {"update","transition",...非具名写} → trigger="update"
               == "delete"                            → trigger="delete"
其余具名动作(approve/reject/merge/manual_correct/…) → trigger="custom" + action_name 精确匹配
```

实现上不做白名单映射 —— **直接用 audit action 字符串做 trigger 字段**:规则
`trigger` 存的就是要匹配的 action 名(UI 给 create/update/delete 快捷选项 +
custom 自由输入)。透明、无映射表维护成本。

`on_update` 的 watch_fields:entities.py PATCH 的 `detail={"fields": changed}`,
取 `detail["fields"]` 键集;其他端点 detail 无 `fields` 键时按"全字段变更"处理
(保守触发)。

`on_delete` 快照:对象已删,快照尽力而为(已加载属性可读则读,失败降级为
`{"id": entity_id}` + detail 内容)。

### A.5 收件人解析

`recipients` JSON 规格:

| spec | 解析 |
|---|---|
| `{"kind":"actor"}` | 触发操作的用户 |
| `{"kind":"role","value":"hr"}` | 租户内该角色全部 active 用户 |
| `{"kind":"user","value":"<uid>"}` | 指定用户 |
| `{"kind":"record_owner"}` | 注册表 `owner_user_id`(如 cases.assignee_user_id) |
| `{"kind":"record_manager"}` | 记录员工的 `reports_to` 对应用户 |

`email` 动作额外需要用户 email —— 上表解析出的用户都有 email。解析为空 → run
记 `skipped_no_recipients`。

### A.6 动作执行

- **notify**:逐收件人 `Notification(kind="automation", title, body, link)` 同事务
  落库。同 (rule,event_key,user) 天然去重(run 唯一约束保证整个分发只跑一次)。
- **email**:run 行 `status=queued_email`,payload 存 `{to[], subject, text}`。
  `write_audit` 内注册一次性 `after_commit` 回调(`db.info` 收集 run ids,
  `event.listen(db,"after_commit")` 防重)→ commit 成功后
  `send_automation_email.delay(run_id)`(新增 celery 任务,复用
  `get_active_account`+`send_email` 既有 SMTP 基建,无 SMTP 配置时记
  `email_failed`/`skipped`)。业务回滚 → 无 commit → 不发信,杜绝幽灵邮件。

### A.7 挂起与安全

- `suspend_automation(db)` contextmanager(`db.info["automation_suspended"]=True`),
  CSV 批量导入路径包裹使用,防通知风暴。
- 失败隔离:单条规则评估/分发异常 → run 记 error + logger.exception,绝不影响
  业务事务本身(dispatch 整体 try/except,业务写优先)。
- 权限:规则 CRUD = `manage_settings`;收件人解析出的用户不需要对记录有读权限
  (规则是租户管理员显式配置的通知推送,与 watcher 的 owner 数据视角不同 ——
  但 `record_owner/record_manager` 解析天然落在记录相关人上,无越权广播面)。

### A.8 API + AI 落点

```
GET    /api/automations                  列表(manage_settings)
POST   /api/automations                  创建(校验 entity/trigger/字段白名单)
PATCH  /api/automations/{id}             编辑/启停
DELETE /api/automations/{id}
GET    /api/automations/{id}/runs        执行记录
GET    /api/automations/schema           注册表元数据(entity/字段/算子/收件人 kind)
                                         → 前端表单动态渲染 + sidecar 工具描述共用
```

Sidecar 工具(写操作全走确认门):`list_automations` / `create_automation` /
`update_automation` / `delete_automation` —— "新员工入职后通知 HR"类自然语言
指令由 LLM 映射为规则 JSON,schema 端点喂给工具描述。

---

## B. 考勤深化

### B.1 数据模型(迁移 0052,同一迁移)

`attendance_policies`(每租户一行):

| 列 | 默认 | 说明 |
|---|---|---|
| `grace_in_minutes` | 0 | 上班宽限(分钟内打卡不算迟到) |
| `grace_out_minutes` | 0 | 下班宽限(分钟内早走不算早退) |
| `track_late_early` | true | 总开关 |
| `ip_check_enabled` | false | 打卡 IP 校验开关 |
| `allowed_ips` | `[]` | JSON,CIDR/IP 列表 |
| `notify_on_flag` | true | 产生异常时通知员工本人 |

`work_calendars` 加列:`work_start` Time 默认 `09:00`,`work_end` Time 默认
`18:00`(存量行迁移回填)—— 期望作息基准。排班/轮班不在本期(无 shift 实体,
工作日历单班制是当前模型事实)。

`attendance_flags`(异常台账):

| 列 | 说明 |
|---|---|
| `attendance_id` | FK;`UniqueConstraint(attendance_id, kind)` |
| `employee_id` / `day` | 冗余便于直查 |
| `kind` | `late_in` / `early_out` |
| `minutes` | 超出期望时刻的分钟数(明细口径,非"超宽限"口径) |
| `detail` | JSON `{expected:"09:00", actual:"09:32"}` |
| `excused` / `excused_by_id` / `excused_note` / `excused_at` | 免计工作流 |
| `created_at` / `updated_at` | — |

`employee_checkins` 加列:`ip` String(45),打卡来源 IP 留痕。

### B.2 判定逻辑(进 `summarize_day`)

时区口径:checkin `ts` 是 UTC naive;`work_start`/`work_end` 是部署时区墙钟。
判定前 `first_in/last_out` 先 UTC→`app_timezone` 换算再取 time-of-day 比较
(`attendance.day` 分桶口径维持现状不变,本特性只判定时刻)。

仅当 `status=="present"` 且当日为工作日且 `track_late_early`:

```
expected_in  = day + work_start            (本地墙钟)
actual_in    = first_in → 本地
late_minutes = (actual_in - expected_in) / 60
若 late_minutes > grace_in_minutes → upsert flag(late_in, minutes=late_minutes)

expected_out = day + work_end
early_minutes = (expected_out - last_out→本地) / 60
若 early_minutes > grace_out_minutes → upsert flag(early_out)
```

- upsert 保留 `excused*`(已免计的不丢豁免);不再符合条件 → 删除 flag
  (免计的也删 —— 事实变化后台账与事实一致优先)。
- 新 flag 产生且 `notify_on_flag` → 员工关联用户收 Notification
  (`kind="attendance_flag"`,link `/me/attendance`);flag 唯一键保证不重复推。
- `source=="manual"` 的考勤行同样评估(HR 手改后判定一致),但只有
  `first_in/last_out` 齐才评。

### B.3 打卡 IP 校验

`POST /me/checkins`:`ip_check_enabled` 且 `allowed_ips` 非空 →

1. 取 IP:`X-Forwarded-For` 首跳(部署在 nginx 后)→ 回退 `request.client.host`;
2. `ipaddress.ip_address(ip)` 落在任一 `ip_network(cidr, strict=False)` 内放行,
   否则 **403 "当前网络环境不允许打卡"** + `write_audit_denied`;
3. `checkin.ip` 无论开关都记录(留痕证据);
4. 开关开但列表为空 → 按未启用处理(防管理员把自己锁死;UI 侧加提示)。

### B.4 API

```
GET/PUT /api/attendance/policy                    策略(manage_attendance)
GET     /api/attendance/flags                     异常列表(scope=all/team/mine,
                                                  month/employee/kind/excused 过滤)
POST    /api/attendance/flags/{id}/excuse         免计(manage_attendance 或
                                                  can_decide_for_employee 主管)
POST    /api/attendance/flags/{id}/unexcuse       撤销免计
GET     /api/attendance/monthly-summary           月度汇总:per-employee
                                                  {present/absent/leave/holiday天数,
                                                  work_hours 合计, late/early 次数,
                                                  late_minutes, 批准加班时长}
GET     /api/me/attendance-flags                  ESS 本人异常
GET     /api/me/attendance-summary?month=         ESS 本人月度汇总
PUT     /api/work-calendar                        已存在 —— 加 work_start/work_end 字段
```

### B.5 watcher + AI 落点

- 新 watcher kind `late_repeat` `{days:30, count:3}`:范围内员工近 N 天未免计
  late_in ≥ count 次 → 通知 watcher owner;`WATCHER_PERMS = PERM_VIEW_REPORTS`。
- `entities.py` 注册表补 `attendance_flags`(只读)、`automation_rules`,
  assistant 可"查本月谁迟到最多""创建自动化规则"。
- ai_summary 员工摘要 context 加近 30 天 late/early 计数(小改)。

### B.6 前端

- `/attendance/flags` 新页(考勤组 nav,`manage_attendance`):tabs
  **异常记录 / 月度汇总 / 考勤规则**;异常表支持按人/月/类型过滤 + 免计/撤销;
  规则 tab 编辑 policy + work_start/work_end。
- `/automations` 新页(平台/设置组,`manage_settings`):规则表 + 创建/编辑
  抽屉(实体/触发/条件行编辑器/收件人编辑器/模板)+ 执行记录抽屉。
- `/me` 考勤区:本人 flag 徽标 + 月度汇总条。

### B.7 明确不做(本期)

- 排班/轮班(`EmployeeShift*`):无 shift 概念,单班制日历已覆盖目标客户。
- `AttendanceConflictResolution`(请假/节假日重叠裁决):现有 leave 覆盖逻辑
  已自洽,复杂裁决器性价比低。
- `PenaltyAccounts`(迟到罚款):劳动合规敏感,不做扣款联动,只保留台账。
- `AttendanceSummaryOverride`(月度工时人工覆盖):月度汇总只读计算 v1 够用。
- 生物识别/地理围栏打卡:硬件场景,IP 白名单已覆盖"办公网打卡"诉求。

---

## C. 实施批次

| 步 | 内容 |
|---|---|
| C1 | 迁移 0052 + models(attendance_policies/flags/checkins.ip/work_start/end/automation_rules/runs) |
| C2 | `services/automation.py`(注册表+条件评估+收件人+分发+挂起)+ write_audit 钩子 + celery 邮件任务 |
| C3 | `/api/automations` CRUD+schema+runs;entities.py 注册 attendance_flags/automation_rules |
| C4 | `services/attendance_policy.py` + summarize_day 挂 flag 逻辑 + /me/checkins IP 校验 |
| C5 | `/api/attendance` 加 policy/flags/excuse/monthly-summary;/api/hr_me 加 me/flags+summary |
| C6 | 前端 `/automations` + `/attendance/flags` + /me 徽标 + nav |
| C7 | sidecar 4 个 automation 工具 + watcher `late_repeat` + ai_summary context |
| C8 | `test_t7.py` 单测 + 全量回归 |
| C9 | `verify_t7_live.py` 链路脚本 |
| C10 | ego-browser 全流程 |

## D. 测试要点

- 条件算子全表;watch_fields 交并;custom action 匹配;收件人 5 种解析;
  幂等(同 audit_id 二次分发被唯一约束拦);suspend;email after_commit 不发于回滚。
- 宽限边界(恰等于 grace 不算迟);早退对称;upsert 保留 excused;事实转好删 flag;
  UTC→本地换算;非工作日/请假/缺打卡不产生 flag。
- IP:CIDR 命中/未命中 403、XFF 首跳、空列表放行、checkin.ip 留痕。
- 权限:viewer 403 规则 CRUD;主管只能免计下属;tenant 隔离;ESS 只见本人。
- watcher late_repeat;automation_rules/flags 进 entities 只读检索。

---

## E. Rippling 批次扩展(迁移 0079)

对标 Rippling Workflow Studio 的两项能力,在 T7 引擎上做加法式扩展:

### E.1 定时触发 `trigger_kind=date_field`

- 规则新增 `trigger_kind`(`audit` 默认 | `date_field`)+ `schedule`
  `{field, offset_days, recurring?}`。date_field 规则只挂员工实体:
  `field` ∈ `hire_date|birth_date|probation_end|contract_end|cert_expire`
  (后四个读 `employee.personal` JSON,已并入 employee 快照白名单)。
- `_watch_loop` 每日 `run_scheduled(db)`:`日期值 + offset_days == today`
  → 条件评估 → 执行;`recurring=true` 按今年纪念日算(2/29 平年 2/28)。
  幂等键 `event_key = s:{emp_id}:{today}`(与 rule_id 联合唯一),
  terminated 员工跳过;规则执行失败单条记日志不影响其他。
- 与 T7 差异:audit 规则走 write_audit 漏斗即时触发;date_field 规则
  走日扫,`dispatch` 查询加 `trigger_kind='audit'` 过滤互不相扰。

### E.2 `action=create_case`

- `action_config` `{kind, priority, subject_tpl?, description_tpl?,
  assignee_kind?: role|user, assignee_value?}`;subject/description 走同一
  `{{field}}` 模板;`requester_employee_id` 取实体员工关联(员工规则=本人);
  case_no 复用工单模块的租户内 max+1;assignee=role 取该角色首个活跃用户。
- 收件人(recipients)若配置照常收通知;受理人未在收件人中则单独收一条。
- notify/email 仍要求收件人非空,create_case 可选。

### E.3 审批自动裁决 `ApprovalPolicy.auto_approve`

- `auto_approve` JSON 条件组(与 `conditions` 同一白名单字段+算子)。
  `route_request` 命中策略后:非空且全部命中 → `_auto_approve` 提交即批准;
  否则走原人工链。执行人 = 策略创建者(active)→ 首个 active admin → hr。
- **与人工批准同一份副作用**:先跑 `apply_domain_decision`(`_apply_*` 域
  函数,余额/考勤护栏不打折),成功才标记 `inst.auto_approved=True` +
  全部步骤 approved;副作用抛错 → 保持 pending 走人工链(自动裁决永远
  可退化为人工)。通知申请人「已自动批准」;实例详情透出标记。
- 校验复用 `validate_policy(auto_approve=…)`,字段越白名单 → 422。
- API:`POST/PUT /approvals/policies` 接受 `auto_approve`;
  `GET /approvals/instances/{type}/{id}` 返回 `auto_approved`。
- sidecar:`save_approval_policy` 增 `auto_approve` 参数;新增
  `list_automation_rules`/`list_automation_runs`/`create_automation_rule`/
  `set_automation_rule_active`(后两个走确认门)。
- 前端:`/automations` 表单加触发方式切换(事件/日期)+ 日期配置 +
  建工单配置;`/approvals` 策略表单加自动批准条件行 + 列表徽标。


## F. Teamtailor 阶段触发器(Stage Triggers)

参照 Teamtailor「触发器长在招聘流程 stage 上」的交互,把自动化挂到
岗位详情的阶段视图上;底层仍是同一份 AutomationRule 引擎,零新引擎。

### F.1 规则形态

- 每条阶段触发器 = `entity_type=job_application` +
  `conditions=[{stage eq X},{job_id eq 本岗}]`;`applied` 阶段存
  `trigger=create`(建卡即进入),其余阶段存 `transition`。
- transition 审计 detail `{from,to}` 的标量键并入快照(`_snapshot_for`),
  模板可用 `{{from}}`/`{{to}}`;`job_application` 注册表加
  `extras` → `{{candidate_name}}`/`{{candidate_email}}`/`{{job_title}}`,
  加 `candidate_attr=candidate_id` 供候选人动作解析(candidate/job_offer/
  interview/interview_scorecard 同样注册)。

### F.2 候选人动作(CANDIDATE_ACTIONS)

- `email_candidate`:收件人 = 候选人 `display_email`,复用
  queued_email → after_commit → celery 通道;无邮箱记
  `skipped_no_recipients`。
- `add_to_group`:`action_config.group_id` → 幂等插入
  CandidateGroupMember(source=auto);已入组记 `already_member`;
  分组不存在/跨租户记 run.error。
- `send_survey`:`action_config.survey_id`(须 published)→ 自动建
  一次性 SurveyLink(默认 14 天,绑候选人)→ 链接附在邮件正文尾部发给
  候选人;无邮箱仍 ok(链接已建可转交)。
- 这三个动作不走 recipients;`validate_rule_payload` 要求实体带
  `candidate_attr` 且按动作校验必填 config(group_id/survey_id)。

### F.3 挂载面

- 前端 `JobStageTriggersCard`(岗位详情,manage_settings 门控):
  六阶段行(applied→rejected),每行列已有触发器(动作图标+名称+启停
  开关+编辑/删除)+「添加触发器」;模态只收最少参数(动作+模板/分组/
  问卷/收件人/工单优先级)。
- 回读 = `GET /automations?entity_type=job_application` 前端按
  conditions 过滤本岗本阶段;规则同时出现在全局自动化列表。
- schema_meta 透出 `candidate_actions` 与实体 `has_candidate`;
  全局表单对候选人动作隐藏收件人区并提示到岗位页配置。
