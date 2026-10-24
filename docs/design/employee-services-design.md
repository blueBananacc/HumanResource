# 员工服务与 AI 基建设计(通知/公告/申请/薪酬带宽/工单/知识库/订阅/外联)

第二批功能建设:通知中心 + AI watcher 基建先行,其上接 4 个员工服务模块
(公告/申请/薪酬带宽/HR 工单)+ 知识库 + 定时报表订阅 + 人才再激活外联 +
ICS 导出 + 活动流 + assistant 通用实体工具。

约定:`domain/models.py` 单文件、`String(36)` UUID、`tenant_id` CASCADE、
`api/<module>.py` + pydantic、`hr.suite` 门 HRIS、`require_permission` 复用现有 PERM、
迁移从 `0039` 顺序排。

## 基建事实(调研确认)

- **无 celery beat**:周期任务沿用 `app/main.py::_email_sync_loop` 模式——
  lifespan 起 asyncio loop + `redis_lock` 全局锁,fail-close。
- **无 Notification 表**:前端通知铃是占位。本批新增 `notifications` 表作为
  watcher/工单/申请的统一投递渠道(顺带让通知铃落地)。
- **ESS**:`/api/me/*` 按 `current.user_id → Employee` 行级过滤(`hr_me.py`)。
- **assistant 工具**:sidecar `tools.ts` 每个工具 = 一次 backend API 调用,
  权限/租户/license 全部由 backend 判定。通用工具以"实体注册表"白名单实现,
  **不暴露通用 ORM**。
- **通用实体元工具**:`search_records/count_records/list_entities` 三个工具走
  后端 `GET /api/entities` 注册表驱动的搜索端点,实体/字段全部白名单。
- **不做重型工作流引擎**:触发式自动化用 AI-native 的 `ai_watchers`(条件任务)。
- **不做拖拽报表器**:用 `report_subscriptions`(订阅制推送)+ 既有
  metrics/reports 端点覆盖报表诉求。

## 模块清单

### T3-0 notifications + ai_watchers(基建先行,迁移 `0039`)

- `notifications`:user_id(FK CASCADE)、kind(watch/case/request/system)、
  title(128)、body(256)、link(256 NULL)、read_at NULL、created_at。
  `(user_id, read_at)` 复合索引便于未读计数。
- `ai_watchers`:user_id、kind(枚举字符串)、params JSON、
  last_checked_at、last_fired_at、is_active、created_at。
- API `api/notifications.py`(仅 auth,行级=本人):
  - `GET /me/notifications?unread_only=` 列表(unread 优先,50 条)
  - `GET /me/notifications/unread-count`
  - `POST /me/notifications/{id}/read` / `POST /me/notifications/read-all`
  - `GET/POST /me/watchers`、`POST /me/watchers/{id}/toggle`、
    `DELETE /me/watchers/{id}`(watcher 属本人)
- watcher 条件评估器 `services/watchers.py`,注册表式:
  - `candidate_stalled`:params {days} → JobApplication 在 interview/screening
    停留超 N 天 → 通知(owner 或创建者)
  - `interview_pending_report`:已 ended 且 report 未生成的面试超 1 天
  - `case_open_hours`:params {hours} → 本人 assignee 的 case 超时长未解决
  - `request_pending_days`:params {days} → 本人提交的 request 超期未决
  - `reservation_upcoming`:params {minutes} → 本人预约开始前提醒
- 调度:`app/main.py` 增加 `_watch_loop()`(模式同 `_email_sync_loop`,
  间隔 `WATCH_INTERVAL_SECONDS` 默认 300),内部跑 watcher 评估 +
  报表订阅发送(T3-6)。
- 去重:watcher 每次触发写 notification 时按 (user_id,kind,link) 当日
  查重,避免每 5 分钟重复轰炸。
- assistant:`create_watch`/`list_watches`/`delete_watch` 工具 → 调用方
  "候选人 3 天没动提醒我"→ create_watch(candidate_stalled, {days:3})。
- 前端:通知铃真实化(未读徽标 + 下拉 + /me/notifications 页);
  watchers 管理放 /me。

### T3-1 announcements 公告(迁移 `0040`)

- `announcements`:title(128)、content(Text)、audience_org_unit_id NULL
  (=全公司)、status(draft/published/archived)、published_at、expires_at、
  created_by、时间戳。
- `announcement_reads`:announcement_id+user_id UNIQUE、read_at。
- API `api/announcements.py`(`hr.suite`):
  - 管理 `GET/POST /announcements`、`GET/PUT/DELETE /announcements/{id}`
    写 `PERM_MANAGE_EMPLOYEES`;`GET /announcements/{id}/reads` 已读名单。
    scope:audience 非空时写操作 ensure_can_see(org)。
  - ESS `GET /me/announcements`(published 未过期 + audience∈{null,我的组织}+
    已读标记)、`POST /me/announcements/{id}/read`。
- 前端 `pages/Announcements.tsx`(组织组);/me 工作台加"公司公告"卡。
- assistant:`list_announcements`/`publish_announcement` 工具(写走确认卡片)。

### T3-2 employee_requests 员工申请(迁移 `0041`)

- `employee_requests`:employee_id(FK)、kind(training/benefit/equipment/
  wellness/general)、subject(128)、detail(Text)、status(pending/approved/rejected/
  cancelled)、decided_by、decided_at、decision_note(500)、时间戳。
- `request_comments`:request_id、author_user_id、body(Text)、created_at。
- API `api/requests.py`(`hr.suite`):
  - ESS `GET/POST /me/requests`、`POST /me/requests/{id}/cancel`、
    `POST /me/requests/{id}/comments`。
  - HR `GET /requests?status&kind&page`、`POST /requests/{id}/decide`
    (approve/reject + note,写 notification 给提交人)、
    `POST /requests/{id}/comments` — `PERM_MANAGE_EMPLOYEES`,scope 过滤。
- assistant:`create_request`(代当前用户提交)/`list_requests`。
- 前端:HR 队列页 `pages/Requests.tsx` + /me "我的申请"。

### T3-3 salary_bands 薪酬带宽(迁移 `0042`)

- `salary_bands`:name(64)、designation(128 NULL=通用带宽)、
  org_unit_id NULL=全公司、currency(8,CNY 默认)、gross_min(int 元)、
  gross_max、employer_cost_min/max NULL、effective_from/to NULL、
  is_active、时间戳。`UniqueConstraint(tenant_id,name)`。
- API `api/salary_bands.py`(`hr.suite`):
  - CRUD `GET/POST/PUT/DELETE /salary-bands` — `PERM_MANAGE_PAYROLL` 写;
    读 `PERM_VIEW_REPORTS`;DELETE 无引用约束(纯参考表)。
  - `GET /salary-bands/match?designation=&org_unit_id=` → 命中带宽列表
    (org 特定 > designation 命中 > 通用,按优先级排序)。
- offer 联动 `hiring.py`:`GET /offers/{id}` 响应附
  `salary_band_check:{band_name,within,min,max}`,不阻塞,仅提示。
- assistant:`get_salary_band(designation)` 工具。
- 前端 `pages/SalaryBands.tsx`(薪酬组)。

### T3-4 hr_cases 工单(迁移 `0043`)

- `hr_cases`:case_no(int,租户内序列——insert 时锁 tenant 行取 max+1)、
  subject(128)、kind(it/admin/payroll/facility/leave_dispute/other)、priority
  (low/normal/high/urgent)、status(new/assigned/in_progress/resolved/closed)、
  description(Text)、requester_employee_id、assignee_user_id NULL、
  resolution(Text NULL)、resolved_at、时间戳。
- `hr_case_updates`:case_id、author_user_id、body、is_internal(bool,
  内部备注员工侧不可见)、created_at。
- API `api/cases.py`(`hr.suite`):
  - ESS `GET/POST /me/cases`、`POST /me/cases/{id}/updates`(公开留言)。
  - HR `GET /cases?status&priority&mine=&page`、`POST /cases/{id}/assign`、
    `POST /cases/{id}/status`(resolved 必填 resolution)、
    `POST /cases/{id}/updates`(internal 可标)— `PERM_MANAGE_EMPLOYEES`。
  - 通知:assign/status 变更 → requester 有账号则写 notification。
- assistant:`create_case`/`list_cases`/`update_case_status`。
- 前端 `pages/Cases.tsx` HR 队列(过滤 + 详情抽屉 + 留言)+ /me "我的工单"。

### T3-5 knowledge_articles 知识库(迁移 `0044`)

- `knowledge_articles`:title(128)、category(32:policy/benefit/it/faq/
  onboarding/other)、content(Text markdown)、tags JSON、status
  (draft/published/archived)、updated_by、时间戳。
- API `api/knowledge.py`(`hr.suite`):
  - 管理 `GET/POST/PUT/DELETE /knowledge/articles` — `PERM_MANAGE_EMPLOYEES` 写。
  - 读:`PERM_VIEW_REPORTS` 管端列表 + `GET /me/knowledge?q=` ESS 读 published
    + `GET /knowledge/articles/{id}`。
  - 检索:`POST /knowledge/search {q, limit}` → ILIKE title/tags/category
    评分排序(title 命中 > tag > content),返回 title+category+snippet。
    assistant `search_knowledge` 工具走此端点 → RAG 语料。
- 前端 `pages/Knowledge.tsx` 管理列表 + 编辑抽屉;/me 知识库搜索卡。

### T3-6 report_subscriptions 定时报表(迁移 `0045`)

- `report_subscriptions`:user_id、kind(pipeline_funnel/weekly_summary/
  pending_items)、weekday(0-6)、hour(0-23)、last_sent_at、is_active。
  `(user_id,kind)` UNIQUE。
- API:`GET/POST /me/report-subscriptions`、`POST /me/report-subscriptions/{id}/toggle`、
  `DELETE`(本人行级)。
- 发送:`_watch_loop` 每 tick 找 `weekday=今天 & hour=当前 & last_sent_at<今日` →
  `workers/tasks/email.py` 新增 `send_report_digest` celery task → 用既有
  metrics/reports 服务函数取数 → `render_email`("report.digest" 模板,
  seed 进 migration)→ SMTP 发送。
- 报表口径复用:pipeline_funnel=各 stage 计数(应用 events);
  weekly_summary=本周新面试/新候选人/待发 offer;pending_items=未读通知数
  +待办 case/request 计数。
- assistant:`subscribe_report`/`list_subscriptions` 工具。
- 前端:Reports 页头加订阅按钮 + /me 订阅管理。

### T3-7 talent_campaigns 人才再激活(迁移 `0046`)

- `talent_campaigns`:name(128)、filter JSON({group_id?/skill_ids?/query?})、
  subject(128)、body_text(Text)、status(draft/sending/done/failed)、
  total/sent/failed 计数、created_by、started_at/done_at、时间戳。
- `talent_campaign_sends`:campaign_id、candidate_id、to_email、status
  (queued/sent/failed/skipped_no_email)、error(256 NULL)、sent_at、
  UNIQUE(campaign_id,candidate_id)。
- API `api/campaigns.py`(`hr.suite` + `PERM_WRITE_RESUMES` 写/`VIEW_REPORTS` 读):
  - `GET/POST /campaigns`(创建时快照 resolve filter→sends 行,跳过无邮箱);
    `GET /campaigns/{id}`(含 sends 分页);`POST /campaigns/{id}/send`
    → celery `send_talent_campaign` 逐条 SMTP(`{{candidate_name}}` 变量替换);
    `POST /campaigns/{id}/cancel`。
  - 发送走 `SMTPAccount` 既有配置;逐条取 queued 更新保证幂等。
- 前端:TalentPool 页"发起触达"入口 + `pages/Campaigns.tsx` 列表/详情。
- 不做 assistant 写工具(批量副作用,留 UI 确认路径)。

### T3-8 ICS 导出(无迁移)

- `api/ics.py`:
  - `GET /interviews/{id}/ics` → 单个 .ics(auth,scope 检查)。VEVENT:
    DTSTART/DTEND UTC、SUMMARY=面试标题、LOCATION=room/meeting_url、
    DESCRIPTION=候选人×岗位。
  - `GET /reservations/{id}/ics` 同法。
  - 手写 ICS builder(≤30 行,RFC5545 escaping CRLF/;/,),返回
    `text/calendar` + `Content-Disposition: attachment`。
- 前端:interview 详情 + Schedule 预约卡加"下载 ICS"。

### T3-9 活动流(无迁移)

- `GET /api/activity?days=7&limit=50`(auth + view_reports):
  聚合 AuditLog(实体创建/变更)+ JobApplicationEvent(stage 变迁)+
  Reservation(新预约)+ SurveyResponse(新答卷)→ 统一
  `{at, actor_name, verb, entity_kind, entity_label, link}`。
- Dashboard 页右侧加"近期动态"卡。

### T3-10 assistant 工具层(无迁移)

- backend `api/entities.py`:`GET /api/entities`(白名单+可读字段说明)、
  `GET /api/entities/{entity}/search?q=&limit=`、`GET /api/entities/{entity}/count?q=`
  —— 注册表 dict:`{entity: {perm, query(db, tenant, scope, q)}}`,tenant 过滤 +
  按实体 org scope。白名单:talents/jobs/interviews/applications/employees/
  rooms/reservations/announcements/cases/requests/knowledge_articles/skills。
- assistant `tools.ts` 新增:`list_entities`、`search_records`、`count_records`、
  `search_knowledge`、`create_watch`、`list_watches`、`create_request`、
  `create_case`、`list_cases`、`get_salary_band`、`list_announcements`、
  `publish_announcement`(写确认)、`subscribe_report`、`get_activity`。
  每个照旧 = 一次 backend API 调用。

## 迁移编号

`0039` notifications+ai_watchers;`0040` announcements+reads;
`0041` employee_requests+comments;`0042` salary_bands;`0043` hr_cases+updates;
`0044` knowledge_articles;`0045` report_subscriptions(+report.digest 模板 seed);
`0046` talent_campaigns+sends。

## 实施顺序

1. `0039` notifications+watchers 基建(其他模块的通知依赖它)
2. announcements(最轻,立标杆)
3. requests → 4. cases → 5. salary_bands(offer 钩子)→ 6. knowledge
7. report_subscriptions(watch_loop 合并发送)→ 8. campaigns
9. ICS → 10. activity → 11. assistant 工具层 + entities 端点
12. 前端全部页面 + nav → 13. 全量测试 → 14. ego-browser 验证

## 测试策略

`freehire_test` 库。每模块 ≥5 用例覆盖:CRUD/状态机/权限位/
org scope/ESS 行级过滤/通知投递。watcher 用直接调评估函数的确定性测试
(不依赖 sleep)。campaigns 发信用 SMTP 桩。ICS 断言 DTSTART/折叠转义。
验收:新用例全绿 + 既有套件不回归 + ego-browser 全链路。
