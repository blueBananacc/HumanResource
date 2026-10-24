# MintHCM 功能移植 · 第四批(T5)实施方案

来源:`C:\Users\666\Desktop\minthcm`。本批两个主题:**AI 工具面收口**(通用写工具)
+ **平台甜点**(saved_filters/favorites/merge/import/dictionaries/schedulers/history/optimistic-lock)。

约定同前三批:`domain/models.py` 单文件、`String(36)` UUID、`tenant_id` CASCADE、
`api/<module>.py` + pydantic、`hr.suite` 门 HRIS、`require_permission` 复用现有 PERM、
迁移从 `0050` 起。前端遵循 `docs/design/hibob-style-reference.md`。

## T5-0 AI 工具收口(entities.py 扩展,无迁移)

MintHCM 18 工具中剩余 6 个,全部以**白名单注册表**落地(不暴露通用 ORM):

- `GET /entities/{entity}/schema` → `{writable, linkable}` — 复用 WRITABLE/LINKS
  注册表生成字段+校验说明(类型/choices/required),供 LLM 事前发现。
- `POST /entities/{entity}` → `CREATABLE` 注册表:`{perm, create_fn, fields{attr:(required,validator)}, after_change}`。
  收窄白名单:`notes`(candidate_notes)、`knowledge`(articles draft)、`announcements`(draft)。
  有重副作用的实体(employees/jobs/cases…)不进白名单。
- `DELETE /entities/{entity}/{rid}` → `DELETABLE` 注册表:`{perm, get, delete_fn, audit}`。
  收窄:`knowledge`、`announcements`、`reservations`(cancel 语义)、`skills`(未被引用)、`notes`。
  candidates/jobs/employees/interviews/cases/offers 一律不放。
- `POST /entities/{entity}/{rid}/links {rel_name, related_id, extra?}` +
  `DELETE /entities/{entity}/{rid}/links/{rel_name}/{related_id}` → `LINKS` 注册表:
  - `jobs.skills` → `JobSkill(job_id,skill_id,importance)`(write_jobs)
  - `candidates.skills` → `CandidateSkill(candidate_id,skill_id,level,source=manual)`(write_resumes)
  - `candidates.groups` → `CandidateGroupMember`(write_resumes)
  join 字段(importance/level)走 validators;已存在→409;删除不存在→404。
- `list_report_kinds` 输出加 `params` schema(5e)。
- sidecar `mcp.ts` 加 `resources/list` + `resources/read`:
  `hr://playbooks/{id}`、`hr://entities/{entity}`(list/search 只读)(5f)。
- assistant tools:`create_record`/`delete_record`/`link_records`/`unlink_records`/`get_record_fields`
  全部进 WRITE_TOOLS(前两个写确认)。

## T5-1 saved_filters(迁移 `0050`)

- 表:`id, tenant_id, user_id, page(64), name(128), params JSON, created_at`,UNIQUE(user_id,page,name)。
- API `api/saved_filters.py`(仅 auth,行级=本人):`GET ?page=` 列表 / `POST` 建 / `DELETE /{id}`。
- 前端:`SavedFilterSelect` 组件(当前筛选 JSON.stringify 存 params)→ 接入 Talents/Employees/Cases 列表页头部。

## T5-2 favorites(迁移 `0050`)

- 表:`id, tenant_id, user_id, entity(32), entity_id(36), created_at`,UNIQUE(user_id,entity,entity_id)。
- API `api/favorites.py`(仅 auth,行级=本人):`GET ?entity=` / `POST {entity,entity_id}` / `DELETE /{id}`。
  entity 白名单 talents/jobs/employees/cases;POST 时校验记录存在+租户+scope。
- 前端:列表行/详情页星标 toggle;Talents 页加"只看收藏"筛选。

## T5-3 candidate_merge(无迁移)

- `POST /talents/{keep_id}/merge {drop_id, field_choices?}`(write_resumes + org scope 双方都可见):
  事务内把 drop 的 resumes/applications/interviews/offers/skills/notes/groups/matches/via Resume
  全 re-FK 到 keep;`field_choices {field: "keep"|"drop"}` 覆盖 name/display_email/display_phone/tags/status;
  删 drop(保留 email_hash 防重复建)。audit entity_type=candidate action=merge detail={drop_id}。
- `GET /talents/{id}/duplicates` → 同 email_hash 或同名的其他候选人(辅助找重)。
- 前端:TalentDetail「查找重复」→ 对照弹窗(逐字段 keep/drop 单选)→ 确认合并。

## T5-4 bulk_import(无迁移)

- `POST /talents/import`(multipart CSV,write_resumes):列约定 name,email,phone,tags(分号分隔),org_name(可选按名匹配 org)。
  逐行:必填校验 → email_hash 撞重 → skip 记原因;成功建 Candidate + audit(bulk,单条 detail={row})。
  返回 `{created, skipped:[{row,reason}], total}`。
- `POST /employees/import`(manage_employees):name,employee_no,work_email,designation,org_name 同构。
- 前端:Talents/Employees 页「批量导入」抽屉:文件选择 → 结果表(成功/跳过行+原因)。

## T5-5 dictionaries(迁移 `0050`)

- 表:`id, tenant_id, dict_key(64), value(64), label(128), sort(int), is_active, created_at`,UNIQUE(tenant_id,dict_key,value)。
- API `api/dictionaries.py`:
  - 读:`GET /dictionaries/{key}` → 激活项列表(任何登录用户;下拉用)
  - 写:`GET/POST/PUT/DELETE /dictionaries?key=` — `PERM_MANAGE_SETTINGS`
- 内置 key 注册表 `DICT_KEYS = {"request_kind","case_kind","skill_category"}`:
  request/case kind 校验改为「字典非空 → 字典值;空 → 内置默认」(后向兼容)。
  删除被引用值 → 仍允许(is_active=false 优于删除,前端禁用删除被引用的值…简化:删除即删,引用字段原样保留字符串)。
- 前端 `pages/Dictionaries.tsx`(设置→HR 组):key 选择器 + 值 CRUD。
- migration seed 内置默认三 key。

## T5-6 job_runs + scheduler 运维页(迁移 `0050`)

- 表 `job_runs`:`id, tenant_id NULL(全局任务), name(64), started_at, duration_ms, status(success/failure), message(256 NULL)`。索引 (name, started_at)。
- 打点:`app/main.py::_email_sync_loop` 一轮一条(sync 异常→failure);`_watch_loop::_tick` 一条;
  celery:`resume parse`(`workers/tasks/resume.py`)、`send_talent_campaign`、`evaluate_matches_for_job`、`send_report_subscription` 各记一条。
  helper `app/services/jobruns.py::record_run(db, name, started, status, message)` — 失败不抛。
- API `api/scheduler.py`(admin):`GET /scheduler/jobs` → 注册表 [{name,label,source}] JOIN 最近一次 run;
  `GET /scheduler/jobs/{name}/runs?limit=` 历史。
- 前端 `pages/Schedulers.tsx`(设置→系统组):任务表(名称/来源/上次运行/耗时/状态/消息)。

## T5-7 记录历史抽屉(无迁移)

- `GET /api/audit/events?entity_id=` 已支持;前端详情页(TalentDetail/EmployeeDetail/CaseDetail)加
  「变更记录」抽屉 —— 复用 AuditCenter 行渲染。

## T5-8 optimistic lock(无迁移)

- `PATCH /entities/{entity}/{rid}` body 加可选 `base_updated_at`;行有 `updated_at` 且不匹配 → 409
  `{"detail":"记录已被他人修改"}`。`PUT /employees/{id}`、`PUT /jobs/{id}` 同加。

## 迁移编号

`0050` saved_filters + favorites + dictionaries + job_runs(一次迁移四表,同批强相关)。

## 前端页面清单(hibob 规范)

- `SavedFilterSelect`(组件,接入 Talents/Employees/Cases)
- TalentDetail:星标 + 「查找重复」+ 变更记录抽屉;Talents 列表:收藏筛选 + 「批量导入」抽屉
- Employees 列表:星标 + 保存筛选 + 「批量导入」;EmployeeDetail:变更记录
- CaseDetail:变更记录;Cases:保存筛选
- `pages/Dictionaries.tsx`、`pages/Schedulers.tsx` + nav 注册(设置组,manage_settings)
- assistant 新写工具确认卡走既有 flow(无需 UI 新组件)

## 测试策略

`backend/tests/test_t5.py`:`freehire_test` 库。
- entities:schema 输出/create(白名单+校验+403)/delete(404/被引用 skill 409)/link 双向+409+404
- saved_filters/favorites CRUD+行级隔离+entity 白名单
- merge:关联迁移+field_choices+drop 删除+audit;duplicates 命中
- import:CSV 建/跳过/错误行;employees 同构
- dictionaries:CRUD+key 注册表+kind 校验后向兼容(空字典回退内置)
- scheduler:record_run 不落异常+jobs 列表
- optimistic lock:base_updated_at 匹配/失配 409
- 链路:ego-browser 全 UI 流程 + curl API 真链路
