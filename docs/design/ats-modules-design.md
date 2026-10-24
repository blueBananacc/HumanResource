# 招聘侧模块设计(管道/提示词/邮件模板/技能/预约/问卷)

第一批功能建设的设计与实施记录:申请管道看板、AI 提示词模板、邮件模板、
技能库、会议室/资源预约、问卷。落地遵循本项目既有约定:

- `domain/models.py` 单文件模型,`String(36)` UUID 主键,所有表带 `tenant_id` FK CASCADE
- `api/<module>.py` 单文件 router + pydantic In/Out,`_pagination` 分页,
  `require_feature("hr.suite")` 门 HRIS,`require_permission(PERM_*)` 鉴权
- Alembic 顺序号迁移,容器启动 `alembic upgrade head`
- 前端 React Query + `fetchJSON`,`pages/` + `nav.ts` 三级导航,UI 用 `components/ui/*`
- 权限尽量复用现有 PERM,不新增角色映射

## 测试策略(重要)

`backend/tests/conftest.py` 已把 `DATABASE_URL` 重写为 `freehire_test`。
**测试只在 `freehire_test` 库跑**,绝不指向开发库;首次使用前先
`CREATE DATABASE freehire_test` + `alembic upgrade head`。
API 链路测试直接对运行中的 docker stack 发真实 HTTP;
UI 测试用 ego-browser。

---

## T1-1 申请管道 job_applications + Kanban

申请管道记录候选人 × 岗位的推进关系,看板按状态分列。
状态机收敛为 7 态,贴合现有实体:

```
applied → screening → interview → offer → hired
                 ↘            rejected(可回退)      withdrawn ↗
```

- 表 `job_applications`:candidate_id + job_id(tenant 内唯一)、
  `stage`、`source`(manual/match/email)、`owner_user_id`、`note`、
  `rejected_reason`、`stage_changed_at`、`created_at`
- 表 `job_application_events`:application_id、from_stage、to_stage、
  actor_user_id(可空=系统)、note、created_at —— 全量变迁历史
- 联动钩子(内联、不走 worker):
  - `hiring.py` offer send → `offer`;respond accepted → `hired`;
    declined → `rejected`;withdraw → 回退 `interview`
  - `interviews.py` 创建 remote 面试 → 已有 application 时推进到 `interview`
  - 统一走 `app/services/applications.py` 的 `advance_application(...)`
- API `api/applications.py`(`hr.suite` + `write_jobs` 写 / `view_reports` 读):
  - `GET /applications?job_id&stage&page` 列表
  - `POST /applications` 建卡(冲突 409)
  - `GET /applications/{id}` 详情 + events
  - `POST /applications/{id}/transition` 换阶段(记 event + audit)
  - `GET /jobs/{id}/pipeline` 看板数据(每列 count + 卡片:候选人名、
    match 分、owner、停留天数)
  - `DELETE /applications/{id}`
- 前端 `pages/Pipeline.tsx`:岗位选择器 + 7 列看板,卡片可拖(HTML5 DnD,
  无依赖)到目标列 → transition;卡片点击 → 候选人详情;
  JobMatches 行加"加入管道"动作;nav `招聘 → 岗位` 组加"招聘管道"。

## T1-2 AI 提示词模板 prompt_templates

所有 AI 调用的 prompt 收进注册表化管理,支持租户覆盖与版本回滚。

- 表 `prompt_templates`:tenant+`key` 唯一(`match.evaluate` /
  `interview.question` / `interview.report` / `question.generate` /
  `resume.parse` / `assistant.system`)、name、description、content
  (`{{var}}` 占位)、is_active、version(每次保存 +1)、updated_by
- 表 `prompt_template_versions`:template_id、version、content、
  created_by、created_at —— 供回滚
- API `api/prompt_templates.py`(`manage_settings`):
  `GET` 列表 / `PUT /{key}` 保存(写版本)/ `POST /{key}/preview` 样例渲染 /
  `POST /{key}/rollback` 回滚到指定版本 / `POST /{key}/reset` 恢复内置默认
- 渲染:`app/services/prompting.py` `render_prompt(db, tenant, key, **vars)`
  —— 查库 → 模板 → `{{var}}` 替换(未知变量原样保留便于发现笔误);
  查无模板 → 返回 None,调用方回落硬编码常量。
- 接入点(每处 3~5 行):`match_evaluator`、`question_generator`、
  `interviewer`(出题/报告两处)、`question_bank`。
- 前端 `pages/PromptSettings.tsx`(设置 → AI 与语音 组):模板列表 +
  抽屉编辑(content textarea + 变量说明 + 实时预览 + 版本回滚)。

## T1-3 邮件模板 email_templates

事务性邮件文案同 prompt 模板一样注册表化:subject + body_text + body_html。

- 表 `email_templates`:tenant+`key` 唯一(`interview.invite` /
  `interview.done` / `offer.letter`)、name、subject、body_text、
  body_html(可空 → 只发纯文本)、variables JSON(文档化变量)、
  is_active、updated_by、updated_at
- API `api/email_templates.py`(`manage_settings`):同 prompt 模板的
  列表/PUT/preview/reset,preview 返回渲染后的 subject/text/html
- 渲染:`app/services/mail_templates.py` `render_email(db, tenant, key, vars)`
  → `(subject, text, html)`;`{{var}}` 替换,html 侧值做 `html.escape`。
- 接入:`workers/tasks/email.py` 的 `_format_invite_*` /
  `_format_done_*` 改为「查模板,无模板回落现有函数」;migration 里
  把现有默认文案 seed 成模板内容(各租户首次保存前的 fallback 仍是代码)。
- 前端 `pages/EmailTemplates.tsx`(设置 → 邮件 组):列表 + 编辑器
  (subject/body_text/body_html 三栏 + 变量 chips + 预览)。

## T2-1 技能库 skills

统一技能字典 + 岗位技能要求 + 候选人技能评分。

- 表 `skills`:tenant+`name` 唯一、category(tech/language/soft/cert/
  other)、description
- 表 `job_skills`:job_id+skill_id 唯一、`importance`(must/nice)
- 表 `candidate_skills`:candidate_id+skill_id 唯一、`level`(1-5)、
  `source`(manual/resume)
- API `api/skills.py`:
  - 字典 `GET/POST/PUT/DELETE /skills`(`manage_settings` 写,任何人可读)
  - `PUT /jobs/{id}/skills` 整体替换(`write_jobs`)
  - `PUT /candidates/{id}/skills` 整体替换(`write_resumes`)
  - `POST /candidates/{id}/skills/import-resume` 从最新 done 简历
    `parsed_data.skills` 合并导入
  - `GET /jobs/{id}/skill-coverage` 每个岗位要求 ↔ 候选池覆盖数 +
    已建管道卡覆盖,给匹配页补"技能缺口"解释
- 前端:JobDetail 加"技能要求"区块(搜索式 combobox 增删,标 must/nice);
  TalentDetail 加"技能"区块(level 1-5 + 从简历导入按钮)。

## T2-2 会议室/资源预约 rooms + reservations

会议室与共享资源的时段预约,带冲突检查。

- 表 `meeting_rooms`:name、location、capacity、equipment JSON、is_active
- 表 `room_reservations`:room_id、title、start_at、end_at、
  booked_by(user_id)、interview_id(预留,NULL)、status
  (confirmed/cancelled)、note
- API `api/rooms.py`:
  - 会议室 `GET/POST/PUT/DELETE /rooms`(`manage_settings`)
  - 预约 `GET /reservations?room_id&from&to`(读 `view_reports`)、
    `POST`(`write_interview`,重叠校验:`s < e && e0 < e && s0 > s` → 409)、
    `POST /{id}/cancel`、`PUT /{id}` 改期
  - `GET /rooms/availability?date=` 各会议室当天占用段
- 前端 `pages/Schedule.tsx`:左侧会议室列表,右侧选中日期的时段占用 +
  预约弹窗(会议室/标题/起止/备注) + 取消。

## T2-3 问卷 surveys(候选人反馈)

管理员建卷 → 生成一次性公开链接 → 候选人无登录提交 → 统计。

- 表 `surveys`:title、description、status(draft/published/closed)
- 表 `survey_questions`:survey_id、sort_order、type
  (text/textarea/single/multi/rating)、title、required、options JSON
- 表 `survey_links`:survey_id、token_hash(复用 `invite_tokens` 模式)、
  candidate_id 可空、expires_at、used_at(一次性)、created_at
- 表 `survey_responses`:survey_id、link_id、answers JSON、submitted_at
- 管理 API `api/surveys.py`(`write_interview` 写 / `view_reports` 读):
  `GET/POST /surveys`、`GET/PUT /{id}`(含 questions 全量保存)、
  `POST /{id}/publish` / `POST /{id}/close`、
  `POST /{id}/links`(发链接,可绑候选人)、`GET /{id}/responses`、
  `GET /{id}/stats`(每题聚合:选择计数/评分均值/文本列表)
- 公开 API `api/public_surveys.py`(无鉴权,hash 校验 + 未用 + 未过期):
  `GET /public/surveys/{token}` 返回卷面、`POST .../submit` 存卷标记 used
- 前端 `pages/Surveys.tsx` + `pages/SurveyDetail.tsx`:
  问卷列表 → 编辑器(题目增删改、类型、选项、必填) → 发布 →
  链接列表(复制/绑定候选人) → 答卷与统计;
  公开页 `pages/SurveyFill.tsx` 路由 `/s/{token}`,免登录。

## 实施顺序与验收

按依赖独立度并行推进,每模块交付 = migration + models + api + service
钩子 + 前端页 + nav 注册 + 单测:

1. `0032` job_applications + pipeline(T1-1)
2. `0033` prompt_templates + versions(T1-2)
3. `0034` email_templates(T1-3,与 T1-2 共用渲染器思路)
4. `0035` skills 三表(T2-1)
5. `0036` meeting_rooms + reservations(T2-2)
6. `0037` surveys 四表(T2-3)

验收基线:新单测全绿 + 既有套件不回归(`freehire_test` 库)+
API 链路手工 curl 走通 + ego-browser 每模块至少一条 UI 全流程。

## 实施状态(2026-10-23,全部完成)

| 模块 | 迁移 | 关键文件 | 单测 |
|---|---|---|---|
| T1-1 管道 | `0032` | `api/applications.py` + `services/applications.py` + `pages/Pipeline.tsx` + `JobMatches.tsx`(加入管道) | `test_applications.py` 8 |
| T1-2 提示词 | `0033` | `services/prompting.py`(注册表+渲染)+ `api/prompt_templates.py`(列表/保存/预览/reset/versions/rollback)+ `pages/PromptSettings.tsx`;6 处调用点全部接入(match/question.generate/interview.question/interview.score/interview.report/question.regen) | `test_prompt_templates.py` 5 |
| T1-3 邮件模板 | `0034` | `services/mail_templates.py`(html 侧变量 escape)+ `api/email_templates.py` + `pages/EmailTemplates.tsx`;`workers/tasks/email.py` 邀请+完成两个任务接入 | `test_email_templates.py` 4 |
| T2-1 技能 | `0035` | `api/skills.py`(字典+岗位 PUT+候选人 PUT+简历导入+skill-coverage)+ `components/skills/SkillWidgets.tsx` 挂 JobDetail/TalentDetail | `test_skills.py` 5 |
| T2-2 预约 | `0036` | `api/rooms.py`(rooms/resources/reservations/interviewers/availability)+ `pages/Schedule.tsx` 周历;同对象重叠 → 409 | `test_reservations.py` 5 |
| T2-3 问卷 | `0037`+`0038` | `api/surveys.py`(管理端 + `survey_links` 一次性 token 链接 + `/public/surveys/{token}` 免认证,respondent_key UNIQUE 防重复)+ `pages/Surveys.tsx` + `pages/SurveyFill.tsx` | `test_surveys.py` 4 |

与原方案的偏差:
- **重复提交防重**改为先查重再校验必填(原顺序会把重复提交报成 400)。
- 邮件模板未建版本表(提示词有);如需可后加。
- `reservations` 合并了计划的 `meeting_rooms + reservations` 两表为 `rooms + resources + reservations` 三表。
- 候选覆盖接口 `GET /jobs/{id}/skill-coverage` 为新增(方案未列):每技能给全池/管道内人数。

测试:30 个新测试 + 364 全量绿(`freehire_test`)。

## 审查修复(a2a996f5,2026-10-23)

- rooms:`scope.is_all` 误用 → 绑 org 用户列表必 500,改 `apply_org_filter`;
  `_overlapping` OR 语义 + interview 无对象不误报 + FOR UPDATE 防并发双订;
  面试官/会议室 org/资源 is_active 校验补齐,availability 按范围过滤。
- applications/skills:单资源端点补 `ensure_can_see`(原只查租户,org 范围可越
  读写);list/pipeline/coverage 加候选范围过滤;skill 重名 409。
- surveys:closed 禁编辑(原会级联清空答卷);`order` 缺省按下标(原默认 0
  乱序);prompt 版本列表返回 content。
- 前端:canWrite 一律 `hasPermission` 对齐后端(原按角色名硬判会错配
  interviewer/hiring_manager);SurveyFill crypto.randomUUID 兜底。
- 测试:+8 回归用例 → 全量 372 绿;API 真链路 6 模块 + ego-browser
  UI 全链路复核通过。

## 复审修复(2026-10-24,alembic 0038)

- **survey_links 一次性链接**(对齐 invite_tokens 惯例):
  `POST /surveys/{id}/links` 发明文一次(sha256 入库,可绑 candidate,
  默认 30 天,1-365 可调);公开端改 `/public/surveys/{token}`,
  原子 `UPDATE ... WHERE used_at IS NULL` 核销防并发双填;
  `GET /links` 状态(有效/已用/过期)、`DELETE` 吊销即失效。
  results 增加 respondents 带出绑定归属。前端 `/s/:token` + 链接抽屉
  (发链接/复制/吊销/候选人搜索)。原 `/s/:id` 直访已废除。
- **会议室/资源管理 UI**:排期页「会议室管理」抽屉(manage_settings),
  增删改 + 停用 + org 绑定 + equipment;有预约记录的对象删除 → 409 引导停用。
- **busy_titles 打码**:纯面试官预约对数据范围受限用户隐藏标题(防候选人名泄露)。
- **技能字典收口**:按 name 新建词条需 manage_settings(import-resume 例外),
  前端无权限时隐藏「新建」;技能被引用时删除 → 409;rooms 补 equipment 字段。
- **提示词版本预览**:版本列表可展开 content 再回滚。
- 测试:+5 用例(link 生命周期/删除守卫/标题打码/name 门槛/技能引用)→ 全量 378 绿。
