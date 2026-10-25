# T9 结构化面试评分卡 + 声明式全局搜索设计

来源：Gauzy(`candidate-interview`/`candidate-feedbacks`/`candidate-criterions-rating` 与
`search_index_definition`/`search_document`/`plugin-search`)的**概念迁移**——
不抄代码(NestJS/TypeORM → FastAPI/SQLAlchemy),复用我们已有的审计漏斗、
实体注册表、权限/scope 体系与 PromptTemplate 覆盖模式。

## 一、结构化面试评分卡

### 业务定位

现有 `Interview` 是 **AI 面试会话**(LLM 出题/逐轮评分/转写)。评分卡补的是
**人类面试官**这一层：一场(线下/线上)面试的考察维度、多个面试官的逐维评分、
录用表态、以及 AI 对全部反馈的聚合分析。

### 数据模型(迁移 0054)

**interview_scorecards**

| 列 | 说明 |
|---|---|
| id / tenant_id | 常规 |
| candidate_id → candidates | 必填,scope 锚点(随候选人 org) |
| job_id → jobs | 可空(关联岗位,供 AI 出题纲) |
| interview_id → interviews | 可空(挂在某场 AI 轮次旁,或独立人类轮次) |
| title | 轮次名("一面""终面") |
| criteria | JSONB `[{key,name,dimension,weight}]`;dimension ∈ tech/soft/general;key 为创建时生成的稳定标识 |
| interviewer_ids | JSONB `[employees.id]` 被指派的面试官集合 |
| status | `open` / `closed`(closed 后反馈与维度冻结) |
| scheduled_at / location | 可空,排期信息 |
| note | 可空备注 |
| ai_summary | JSONB:`{summary,themes[],disagreements[],recommendation,generated_at,feedback_count}` |
| created_by → users | 创建人 |
| created_at / updated_at | updated_at 兼乐观锁 |

**interview_feedbacks**

| 列 | 说明 |
|---|---|
| id / tenant_id | 常规 |
| scorecard_id → interview_scorecards CASCADE | 必填 |
| employee_id → employees | 面试官本人(必须 ∈ scorecard.interviewer_ids) |
| rating | 总评 1-5,必填 |
| verdict | 表态:`recommend`/`hold`/`reject` |
| scores | JSONB `[{criterion,score(1-5),comment}]`,criterion 必须命中评分卡声明的 key |
| notes | 自由评语 |
| submitted_at / created_at / updated_at | 重复提交=更新(upsert) |

`UniqueConstraint(scorecard_id, employee_id)` — 一个面试官一张卡。

### API `/api/scorecards`

| 端点 | 权限 | 说明 |
|---|---|---|
| POST / | write_interview | 创建;interviewer_ids 校验为本租户员工 |
| GET / | view_reports | 列表,`candidate_id`/`job_id`/`status` 过滤;候选 org scope |
| GET /{id} | view_reports + scope | 详情 + 聚合(每维度均分/总均分/表态计数/分歧标记) |
| PATCH /{id} | write_interview | 改 title/criteria/interviewers/排期/状态;**已有反馈后 criteria 冻结**;乐观锁 |
| DELETE /{id} | write_interview | 删除(确认在前端) |
| POST /{id}/feedback | 本人 ∈ interviewer_ids | 提交/更新自己的反馈(status=open 才允许) |
| GET /mine | 登录即可 | 我被指派的评分卡(按 linked employee 过滤) |
| POST /{id}/synthesize | write_interview + `assistant.agent` | LLM 聚合全部反馈 → ai_summary 落库 + `ai_call` 审计 |
| POST /ai-criteria | write_interview + `assistant.agent` | 按 job_skills+JD 生成维度草稿(**只回草稿不落库**) |

与 Gauzy 的差异:criteria 不建独立实体,存 JSONB(与 `Interview.kinds`/
`Interview.summary` 同款风格);面试官引用 `employees.id` 而非单独
interviewer 表,反馈提交按 `get_linked_employee_id` 解析本人。

### 事件与 AI

- 审计:`entity_type="interview_scorecard"`,action ∈ create/update/delete/
  feedback/close/synthesize;**feedback 提交也记到 scorecard 上**,自动化规则
  可监听("全部面试官已提交 → 通知 HR")。
- automation `ENTITY_REGISTRY` 增补 `interview_scorecard`(fields: status/
  candidate_id/job_id/title,link → `/talents/{candidate_id}`)。
- 提示词:`scorecard.criteria`(维度建议)、`scorecard.synthesize`(聚合分析,
  输出 JSON:summary/themes/disagreements/recommendation)。mock 降级:
  synthesize 返回机械聚合文本,criteria 返回空列表→前端回落"手动添加"。

## 二、声明式全局搜索

### 三条架构性质(照抄 Gauzy)

1. **声明是数据不是代码**:`SEARCH_SOURCES` 代码注册表是默认声明;
   `search_index_definitions` 表存租户覆盖(与 PromptTemplate 同款两级回落:
   租户行 > 注册表)。租户可关实体、调权重、改模板,不发版。
2. **索引是可丢弃投影**:`search_documents.entity_id` **不带外键**;
   命中只回 entity+id,读路径经 `ENTITIES[entity].build()` 重读真实行——
   权限/org scope 永远在域模型侧生效,索引泄漏不可能。
3. **引擎内置**:本版只实现 database provider(Postgres ILIKE+权重打分);
   `engineKey`/外部引擎是将来接缝,不进本期。

### 数据模型(迁移 0054)

**search_index_definitions** — 租户覆盖行(可空即注册表默认):

| 列 | 说明 |
|---|---|
| tenant_id + entity | UniqueConstraint(tenant,entity) |
| label / is_active | 展示名、启停 |
| fields | JSONB `[{name,weight}]` 覆盖注册表权重(可只覆盖部分字段) |
| title_template / body_template | `{{field}}` 模板覆盖 |
| keyword_fields | JSONB 可过滤 token 字段集覆盖 |
| version | 覆盖内容变更时 +1;文档带 `definition_version`,不一致者重建 |

**search_documents** — 投影:

| 列 | 说明 |
|---|---|
| tenant_id / entity / entity_id | `UniqueConstraint(tenant,entity,entity_id)` —— 重复索引=upsert |
| title / body | 模板渲染产物 |
| keywords | JSONB `["status:open","org:研发部"]` 可过滤 token |
| source_updated_at | 源行 updated_at/created_at,增量跳过依据 |
| definition_version / indexed_at | 版本水印 |

### 写路径(挂 write_audit 漏斗)

`write_audit` 在 `automation.dispatch` 后追加 `search_index.on_audit(db, log)`:

- `SEARCH_SOURCES[entity]` 声明 `audit_types`(如 talents ← `candidate`);
  action ∈ delete 别名 → 删文档;create/update/import/transition… → 重建文档
- 重建 = `db.get(model, entity_id)` → 按**生效定义**(租户覆盖>注册表)
  渲染 title/body/keywords → upsert。对象已删 → 删文档兜底
- 整体 try/except:索引失败绝不影响业务事务(与 dispatch 同约束)
- `suspend_automation` 期间同样跳过索引?(批量导入 2000 行各一次 upsert 可
  接受,不跳——索引天然幂等)

### 读路径 `/api/search`

- `GET /api/search?q=&entity=&limit=` — 命中 `title ILIKE`(权重3)/
  `body ILIKE`(权重1)/`keywords` 文本命中(权重2),按 entity 分组排序;
  **每个命中 entity 先过 `ENTITIES[key].perm`+feature,再用 `build()` 子查询
  按 id 集过滤**——行级 scope 与列表 API 完全同口径
- `GET /api/search/facets?q=` — 各实体命中数(权限过滤后)
- `GET /api/search/definitions` + `PATCH /api/search/definitions/{entity}`
  — manage_settings;查看生效定义、覆盖权重/模板/启停
- `GET /api/search/index-status` — 每实体文档数/最旧索引时间
- `POST /api/search/reindex` — manage_settings;全量/单实体分批重建
  (声明 version 不一致者必重建;源行硬删的孤儿文档顺带清扫)

### 索引实体集(v1)

talents(candidate) / jobs / employees / interviews / cases(hr_case) /
requests(employee_request) / knowledge(knowledge_article) /
announcements / expenses(expense_claim) — 与 ENTITIES 注册表交集,
单条 `SEARCH_SOURCES` 声明同时声明 audit_types。

### 前端与助手

- `GlobalSearch` 升级:命中组从"功能+候选人"变"功能+N 实体分组";
  "查看全部结果"→ `/search?q=` 页(facet chip 过滤实体)
- `/search` 路由加入 AppShell 子路由,不进侧边导航(Ctrl+K 入口)
- sidecar 新增 `search_all` 工具 → `GET /api/search`,替代逐实体查询

## 三、测试计划

- `test_t9.py`:评分卡 CRUD/面试官校验/反馈 upsert/维度冻结/聚合/AI 合成
  (mock)/criteria 建议/权限与租户隔离;搜索 on_audit 索引/删除清扫/
  定义覆盖/权重排序/权限与行级 scope/reindex/孤儿清扫
- 功能链路 `verify_t9_live.py`:HTTP 全链路 + 跨租户/越权
- ego-browser:评分卡创建→面试官指派→反馈提交→AI 合成;全局搜索 Ctrl+K
  →跨实体命中→跳转;搜索定义页调权重→重建
