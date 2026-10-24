# AI 原生工作流设计(工具富卡片/模块级提示词/产品帮助/字段级 AI/记录摘要/文档起草)

第五批功能建设,主题:**让 AI 长在业务界面上**。不再扩充工具面,而是把 AI
能力嵌进日常操作路径:

| 项 | 实现 |
|---|---|
| T6-A | `prompt_templates.scope_module`:提示词模板支持按模块覆盖 |
| T6-B | sidecar `tool_result` SSE → 助手面板把工具结果渲染成可操作的富卡片 |
| T6-C | 产品使用文档切片灌知识库(`category=product_help`,只读)→ 助手可答"系统怎么用" |
| T6-1 | 字段级 AI 助手:表单字段旁 ✨ 起草/润色 |
| T6-2 | 记录级 AI 摘要卡(候选人360°/员工/工单) |
| T6-3 | NL→规则:`save_filter` 工具 + watcher 话术引导 |
| T6-4 | `doc_templates` + LLM 起草 offer 函/在职证明等文档 |

约定:`domain/models.py` 单文件、`String(36)` UUID、`tenant_id` CASCADE、
`api/<module>.py` + pydantic、`require_permission` 复用现有 PERM、AI 端点统一挂
`require_feature("assistant.agent")`、AI 调用/写操作 `write_audit` 留痕、迁移 `0051`、
前端遵循 `docs/design/hibob-style-reference.md`。

## T6-A prompt 模板按模块隔离(迁移 `0051`)

key 已编码用途(`match.evaluate`),模块归属是 key 的天然属性 —— 但同一用途
可能想按模块覆盖(如 polish 在 jobs 与 announcements 风格不同),故加列:

- 模型:`PromptTemplate.scope_module String(48) NOT NULL server_default ''`
  ("" = 全局)。唯一约束改 `(tenant_id, key, scope_module)` —— 迁移 drop 旧
  `uq_prompt_templates_tenant_key` 重建。
- `PROMPT_REGISTRY` 的 `PromptSpec` 加 `module: str` 元数据(管理页分组用):
  现有 key 归为 talents/jobs/interview/global;新增 field/summarize/doc key 归对应模块。
- `render_prompt(db, tenant, key, module=None, **vars)`:`module` 非空时先查
  `(key, scope_module=module)` 再回落 `(key, "")`,两级都查 `is_active`。
- API:`GET /prompt-templates` 输出加 `module`(spec 元数据)+ `scoped_count`
  (该 key 的模块级覆盖数);`PUT` body 加可选 `scope_module`。
- PromptSettings 页按 spec.module 分组渲染;抽屉内"生效范围"编辑器
  (空=全局,填模块名=模块覆盖),预览/版本/回滚/重置全走 scope。

## T6-B 助手面板工具结果富卡片(无迁移,改动集中在 sidecar+前端)

**我们自己的面板把工具结果渲染成可操作的卡片**——查询结果不再只是 LLM 复述
的文字,而是可点击进详情页的富 UI。

链路:pi `tool_execution_end` 事件已带完整 `result`(`{content, details, isError}`)
→ sidecar 把 raw data 截断后(数组 ≤20 项、字符串 ≤800 字、总载荷 ≤8KB)随
`tool_result {call_id, name, result}` 事件 emit → backend SSE 逐行透传(零改动)
→ 前端 `Msg.cards` 追加,`ToolResultCard` 按工具名渲染。

- `tool_result` 事件在 `tool_execution_end` 之后发出;历史会话恢复时 cards
  不回放(消息只存 content;v1 接受)。
- 前端 `tool_start` 时把 `args` 存进 ToolStep(渲染 search_records 卡需要 entity 参数)。
- 卡片映射(白卡片 + border-subtle + 12-13px,复用设计变量):
  - `search_talents` → 候选人表(name/email/status/top_match_score)→ `/talents/:id`
  - `list_jobs` → 岗位行 → `/jobs/:id`;`list_interviews` → `/interviews`
  - `get_job_matches` → 分数榜 → `/jobs/:id/matches`
  - `search_records` → 通用 {id,label,secondary} 表,entity→路由映射
  - `search_knowledge`/`search_product_help` → 文章卡(title+snippet+category)
  - `count_records`/`aggregate_records` → 大数字 stat 卡
  - `run_report` → 报表数据卡(键值 + 数组行成表)
  - `list_relationships` → 分组列表卡
  - 写工具(create_interview/create_case/create_record/publish_announcement/save_filter…)
    → action 成功卡:✓ 文案 + 实体链接
  - 未映射工具 → 折叠 `<details>` 通用 JSON(默认收起,调试用)

## T6-C 产品文档进知识库(无迁移)

产品帮助内容与租户知识分离:`category="product_help"` 是**只读内置类目**——
校验器 `CATEGORIES` 不含它,API 建/改/删全部拒绝,只能由灌库脚本写库。

- `backend/scripts/product_help/*.md`:产品手册源(customer-guide /
  quickstart / deploy-guide / offline-install;含环境敏感信息的运维文档不灌)。
- `backend/scripts/seed_product_help.py`:按 `##`/`###` 标题切片、长段按
  4000 字再切 → `status=published` 写库;每次运行先清租户旧 product_help
  行再灌,幂等;尽力算 embedding(无 provider 不阻塞)。
- `knowledge/search` 加 `category` 过滤;sidecar `search_product_help` 工具
  固定 `category=product_help` 召回。
- SYSTEM_PROMPT:问"系统怎么用/在哪设置"优先 `search_product_help`。

## T6-1 字段级 AI 助手(无迁移)

后端 `app/api/ai.py`(router `/ai`,整组 `require_feature("assistant.agent")`):

- `POST /ai/field-assist {target, intent, context{...}}`
  - `FIELD_TARGETS` 注册表:`{target: {perm, prompt_keys{intent→key}}}`
    - `jobs.description`(write_jobs):draft/polish
    - `announcements.content`(manage_employees):draft/polish
    - `cases.resolution`(manage_employees):draft
    - `campaigns.subject`/`campaigns.body`(write_resumes):draft
      (保留 {{candidate_name}} 占位)
  - intent ∈ draft/polish/expand/shorten;target 无专属 key → `field.generic.<intent>`;
    都没有 → 400。
  - context 白名单化:≤12 键、每值 ≤2000 字符;渲染时按 PromptSpec.variables 取子集。
  - LLM:`chat(msgs, db, tenant, temperature=0.7)` 非 JSON;失败 → 502
    (交互功能不静默兜底)。
  - audit `entity_type="ai_field_assist" action=<intent> detail={target}`。
- PROMPT_REGISTRY 新增:field.jobs.description.draft/polish、
  field.announcements.content.draft/polish、field.cases.resolution.draft、
  field.campaigns.subject.draft、field.campaigns.body.draft、
  field.generic.draft/polish/expand/shorten(模块各归其位)。

前端 `components/assistant/AiFieldButton.tsx`:
- props `{target, getContext()→dict, onApply(text), intents?, disabled?}`
- ✨ 小按钮 → 浮出 panel:intent chips + instruction 输入(可选)→ 生成 →
  预览区 + [应用][重新生成][关闭]。loading/error 内联。
- 埋点:Jobs.tsx 岗位描述;Announcements.tsx content;HrCases.tsx resolution;
  Campaigns.tsx subject+body(表单相应转受控以便回填)。

## T6-2 记录级 AI 摘要(无迁移)

- `GET /ai/summarize/{entity}/{id}` → `{text, entity, id, generated_at}`
  - `SUMMARIZABLE` 注册表:`{entity: {perm, context_fn, prompt_key}}`
    - `candidates`(view_reports):基础字段+tags+skills+top matches+applications
      阶段+interviews(状态/均分,按 started_at 排)+recent_notes(≤5)+resumes 数
    - `employees`(manage_employees):档案+在途 requests/cases+leave 余额+goals
    - `cases`(manage_employees):工单全字段+requester+updates 列表(≤10)
  - context_fn 在 `app/services/ai_summary.py` 实现,走 org scope。
  - prompt key `summarize.<entity>` → LLM 非 JSON → markdown 文本。
  - audit `entity_type=<entity> action="ai_summarize"`。
- `_mock_chat` 支持摘要/起草分支(demo/UI 验证不再只回 "mock response")。
- 前端 `components/assistant/SummaryCard.tsx`:"AI 摘要"卡 → [生成] loading →
  MarkdownContent + [重新生成]。埋点:TalentDetail、EmployeeDetail 顶部。

## T6-3 NL→watcher/saved_filter(无迁移)

sidecar tools.ts:
- 新工具 `save_filter`(WRITE_TOOLS,确认卡):`{page, name, params{}}` →
  `POST /api/saved-filters`。description 列出各页可用筛选键:
  talents{q,status,tags[]}, jobs{q,status}, cases{q,status,kind},
  employees{q,status,org_unit_id}, interviews{q}, knowledge{q,category}。
- `create_watcher` 描述补 NL 映射:"候选人X天没动→candidate_stalled{days};
  面试完没出报告→interview_pending_report;工单超时→case_open_hours{hours};
  申请积压→request_pending_days{days};预约开始前→reservation_upcoming{minutes}"。
- SYSTEM_PROMPT 追加行为准则:NL 提醒→create_watcher;NL 筛选收藏→save_filter;
  "系统怎么用"→search_product_help。

## T6-4 AI 文档起草(迁移 `0051`)

表 `doc_templates`:`id, tenant_id FK, key(64), name(128), entity(32),
body Text, is_active, is_builtin, updated_by, created_at, updated_at`,
UNIQUE(tenant_id,key)。

`app/services/doc_templates.py`:`DOC_SPECS` 注册表(内置模板)
- `offer_letter` 录用通知书(entity=candidates)
- `employment_certificate` 在职证明(entity=employees)
- `probation_notice` 转正通知书(entity=employees)
- `build_doc_vars(db,u,scope,entity,rid)` → 扁平 {{var}} 值(candidate_name/
  job_title/employee_name/employee_no/designation/org_name/hire_date/company 等)

`app/api/doc_templates.py`(merged view,同 prompt_templates 模式):
- `GET /doc-templates?entity=` → 内置(default|已覆盖)+ 自定义行
- `GET /doc-templates/options` → 轻量选项列表(详情页起草抽屉用)
- `POST /doc-templates {name, entity, body}` → 自定义(key=`custom.<uuid8>`)
- `PUT /doc-templates/{key} {body}` → 内置 key 建覆盖行;自定义更新
- `DELETE /doc-templates/{key}` → 自定义删行;内置 key 禁删(is_builtin 保护)
- 全部 manage_employees + audit。

`POST /ai/doc-draft {template_key, entity, record_id, instruction?}`:
- 解析模板(自定义行 > 覆盖行 > spec 默认);entity 不符 → 400
- `build_doc_vars` 得变量 → 预填 → 残余 {{var}} 由 LLM 补全
- system = prompt `doc.draft`(registry,可租户覆盖);user = 模板 + record 摘要
  context(复用 ai_summary 的 context_fn)+ instruction → 非 JSON 输出正文
- 返回 `{title, body, template_key}`;audit `entity_type="doc_draft" action="draft"`。

前端 `components/assistant/DocDraftDrawer.tsx`:
- 详情页 "AI 起草" 按钮(TalentDetail/EmployeeDetail)→ drawer:模板 select
  (按 entity 过滤)+ 附加要求 input → 生成 → 可编辑 textarea →
  [复制][下载 .md][重新生成]。

## 文件清单

迁移:`alembic/versions/0051_t6_ai_native.py`
后端:`api/ai.py`(新)、`api/doc_templates.py`(新)、`services/ai_summary.py`(新)、
`services/doc_templates.py`(新)、`services/prompting.py`(PromptSpec.module +
render_prompt module)、`api/prompt_templates.py`(module 输出 + scope_module 写入)、
`api/knowledge.py`(product_help + category 过滤)、`domain/models.py`
(scope_module + DocTemplate)、`main.py`(2 路由)
sidecar:`tools.ts`(save_filter/search_product_help + 描述)、`agent.ts`(tool_result emit)
前端:`components/assistant/ToolResultCard.tsx`(新)、`AiFieldButton.tsx`、
`SummaryCard.tsx`、`DocDraftDrawer.tsx`、`AiAssistant.tsx`(cards 管线)、
`PromptSettings.tsx`(module 分组+生效范围)、`Jobs/Announcements/HrCases/
Campaigns/TalentDetail/EmployeeDetail` 埋点
脚本:`scripts/product_help/*.md`、`scripts/seed_product_help.py`、
`scripts/verify_t6_live.py`、`scripts/stub_llm_server.py`
测试:`tests/test_t6.py`

## 测试点

单测(test_t6.py):
- scope_module:同 key 全局/模块覆盖共存;render_prompt module 两级回落
- field-assist:draft 返回文本(mock)/无权限 403/未知 target 400
- summarize:candidates 返回 text;他租户 id → 404;无 perm → 403
- doc-templates:merged list/PUT 覆盖内置/POST custom/内置禁删;audit 写入
- doc-draft:返回 title+body;entity 不匹配 → 400;变量预填验证
- product_help:灌库、幂等二次跑、search 限定 category 命中、API 拒写
- ai 端点全部经 require_feature + audit(ai_* entity_type)

链路(verify_t6_live.py):登录 → scope_module 写入/预览 → field-assist draft →
summarize candidate → doc-templates CRUD → doc-draft → 灌库 →
knowledge/search 命中 product_help → 真实 SSE 链路 tool_result 事件 +
confirm_required→confirm→saved_filter 落库 + `via=assistant` 审计。
`stub_llm_server.py` 是 OpenAI 兼容桩(带 tools 判定走工具调用),无外网 key
也能驱动完整工具链路。

ego-browser:富卡片(候选人表/产品帮助)、写工具确认卡全流程、摘要卡生成、
起草抽屉(实体过滤+编辑+导出)、岗位表单 ✨→生成→回填 JD、
PromptSettings 模块分组+生效范围。

## 实施状态(2026-10-24,已完成)

- `test_t6.py` 10/10;全量回归 471 通过 0 失败;frontend+assistant tsc 干净。
- `verify_t6_live.py` 36/36 PASS;ego-browser 全流程验证通过。
- 实施中修正:Interview 无 `created_at`(摘要改按 `started_at` 排序);
  backend litellm 需 `openai/` 前缀的模型名;product_help 类目不进
  `CATEGORIES` 校验器,从 API 层面天然只读。
