# MintHCM 借鉴 · 第五批(T6)实施方案 — AI 产品化

来源:`C:\Users\666\Desktop\minthcm`。本批主题:**让 AI 长在业务界面上**。

MintHCM 的 AI 本体(MCP server / 18 工具 / prompt 模板)T4/T5 已全量移植,本批不再抄
工具,而是借鉴其"AI 落点"做我们自己的 AI-native 工作流:

| 项 | MintHCM 参照 | 我们的实现 |
|---|---|---|
| T6-A | `AIPromptTemplates.root_module` | `prompt_templates.scope_module` + registry module 分组 |
| T6-B | `mcp/apps` 交互 widget | **反向落地**:sidecar `tool_result` SSE → 助手面板富卡片 |
| T6-C | `BrowseDocumentationTool` | 产品手册切片灌知识库(`category=help`)→ search_knowledge 命中 |
| T6-1 | (AI 落点,无对应实现) | 字段级 AI 助手:表单字段旁 ✨ 起草/润色 |
| T6-2 | (AI 落点) | 记录级 AI 摘要卡(候选人360°/员工/工单) |
| T6-3 | (AI 落点) | NL→规则:`save_filter` 工具 + watcher/saved_filter 话术引导 |
| T6-4 | `MailMerge`(智能替代) | doc_templates + LLM 起草 offer letter/在职证明等 |

约定同前批:`domain/models.py` 单文件、`String(36)` UUID、`tenant_id` CASCADE、
`api/<module>.py` + pydantic、`require_permission` 复用现有 PERM、AI 端点统一挂
`require_feature("assistant.agent")`、写操作/AI 调用 `write_audit` 留痕、迁移 `0051`、
前端遵循 `docs/design/hibob-style-reference.md`。

## T6-A prompt 模板按模块隔离(迁移 `0051`)

MintHCM `AIPromptTemplates` 每条记录带 `root_module`;我们的 key 已编码用途
(`match.evaluate`),模块归属是 key 的天然属性 —— 但同一用途可能想按模块覆盖
(如 polish 在 jobs 与 announcements 风格不同),故加列:

- 模型:`PromptTemplate.scope_module String(48) NOT NULL server_default ''`
  ("" = 全局)。唯一约束改 `(tenant_id, key, scope_module)` —— 迁移 drop 旧
  `uq_prompt_templates_tenant_key` 重建。
- `PROMPT_REGISTRY` 的 `PromptSpec` 加 `module: str` 元数据(管理页分组用):
  现有 key 归为 talents/jobs/interview/global;新增 field/summarize/doc key 归对应模块。
- `render_prompt(db, tenant, key, module=None, **vars)`:`module` 非空时先查
  `(key, scope_module=module)` 再回落 `(key, "")`,两级都查 `is_active`。
- API:`GET /prompt-templates` 输出加 `module`(spec 元数据)+ `scoped_count`
  (该 key 的模块级覆盖数);`PUT` body 加可选 `scope_module`(v1 UI 仍写全局)。
- PromptSettings 页按 spec.module 分组渲染(招聘/面试/工单/外联/通用)。

## T6-B 助手面板工具结果富卡片(无迁移,改动集中在 sidecar+前端)

MintHCM `mcp/apps/search-records` 是给**外部 MCP 宿主**渲染 widget —— 对我们 UI
无增益,反向落地:**我们自己的面板把工具结果渲染成可操作的卡片**。

链路:pi `tool_execution_end` 事件已带完整 `result`(`{content, details, isError}`)
→ sidecar `toolResult()` 把 raw data 经 `cardify()` 放 `details.card`
(限界:数组 ≤20 项、字符串 ≤800 字、深度 ≤4)→ agent.ts 在非错误结果时 emit
`tool_result {call_id, name, card}` → backend SSE 逐行透传(零改动)
→ 前端 `Msg.cards` 追加,`ToolResultCard` 按工具名渲染。

- `cardify()` 放 tools.ts;`details.card` 会随会话持久化(限界后 ≤ ~10KB,可接受),
  MCP 侧 details 原样透传无害。
- 前端:`tool_start` 时把 `args` 存进 ToolStep(渲染 search_records 卡需要 entity 参数)。
- 卡片映射(白卡片 + border-subtle + 12-13px,复用 hibob 变量):
  - `search_talents` → 候选人表(name/email/status/top_match_score)→ `/talents/:id`
  - `list_jobs` → 岗位行 → `/jobs/:id`;`list_interviews` → `/interviews`
  - `get_job_matches` → 分数榜 → `/jobs/:id/matches`
  - `search_records` → 通用 {id,label,secondary} 表,entity→路由映射
  - `search_knowledge` → 文章卡(title+snippet+category)→ `/knowledge`
  - `count_records`/`aggregate_records` → 大数字 stat 卡
  - `run_report` → 报表数据卡(键值 + 数组行成表)
  - `list_relationships` → 分组列表卡
  - 写工具(create_interview/create_case/create_record/publish_announcement/save_filter…)
    → action 成功卡:✓ 文案 + 实体链接
  - 未映射工具 → 折叠 `<details>` 通用 JSON(默认收起,调试用)
- 历史会话恢复时 cards 不回放(消息只存 content;v1 接受,标注)。

## T6-C 产品文档进知识库(无迁移)

`KnowledgeArticle.CATEGORIES` 加 `"help"`(代码枚举,非 DB 约束;entities.py
CREATABLE 的 category choices 同步加)。

- `backend/scripts/gen_product_help.py`(dev 生成器):解析
  `docs/ops/customer-guide.md` 按 `##`/`###` 切片 → 输出
  `backend/app/services/product_help.py`(`HELP_ARTICLES: list[dict]`,提交入库,
  随版本发布刷新)。content 保留 markdown 原文。
- `POST /api/knowledge/help/seed`(manage_employees):按 (category=help,title)
  upsert → status=published + `_refresh_embedding` 尽力算向量 → `{created,updated,skipped}`。
- `backend/scripts/seed_product_help.py`:容器内 CLI 薄封装(调同 service 函数)。
- sidecar:`search_knowledge` 描述加"含产品使用帮助(category=help)";
  SYSTEM_PROMPT 加"问'系统怎么用/在哪设置'优先 search_knowledge,命中 help 类即产品手册"。
- Knowledge.tsx 分类选项补 `help 产品帮助`。

## T6-1 字段级 AI 助手(无迁移)

后端 `app/api/ai.py`(router `/ai`,整组 `require_feature("assistant.agent")`):

- `POST /ai/field-assist {target, intent, context{...}}`
  - `FIELD_TARGETS` 注册表:`{target: {perm, prompt_keys{intent→key}}}`
    - `jobs.description`(write_jobs):draft=field.jobs.description.draft / polish=field.jobs.description.polish
    - `announcements.content`(manage_employees):draft/polish
    - `cases.resolution`(manage_employees):draft
    - `campaigns.subject`/`campaigns.body`(manage_employees):draft(保留 {{candidate_name}} 占位)
  - intent ∈ draft/polish/expand/shorten;target 无专属 key → `field.generic.<intent>`;
    都没有 → 400。
  - context 白名单化:≤12 键、每值 ≤2000 字符;渲染时按 PromptSpec.variables 取子集。
  - LLM:`chat(msgs, db, tenant, temperature=0.7)` 非 JSON;失败 → 502(交互功能不静默兜底)。
  - audit `entity_type="ai_field_assist" action=<intent> detail={target}`。
- PROMPT_REGISTRY 新增:field.jobs.description.draft/polish、field.announcements.content.draft/
  polish、field.cases.resolution.draft、field.campaigns.subject.draft、field.campaigns.body.draft、
  field.generic.draft/polish/expand/shorten(模块各归其位)。

前端 `components/ai/AiFieldButton.tsx`:
- props `{target, getContext()→dict, onApply(text), intents?, disabled?}`
- ✨ 小按钮 → 浮出 panel:intent chips + instruction 输入(可选)→ 生成 → 预览区
  (可滚动) + [插入替换][重新生成][关闭]。loading/error 内联。
- 埋点(表单改受控):Jobs.tsx 岗位描述(draft+polish,context=title/level/skills);
  Announcements.tsx content(draft+polish,context=title);HrCases.tsx resolution
  (draft,context=subject/kind/description);Campaigns.tsx subject+body(draft,
  context=campaign name/skills,提示保留 {{candidate_name}})。

## T6-2 记录级 AI 摘要(无迁移)

- `POST /ai/summarize {entity, id}` → `{text, entity, id, generated_at}`
  - `SUMMARIZABLE` 注册表:`{entity: {perm, context_fn, prompt_key}}`
    - `candidates`(view_reports):基础字段+tags+skills+top matches+applications
      阶段+interviews(状态/均分)+recent_notes(≤5,各200字)+resumes 数
    - `employees`(manage_employees):档案+在途 requests/cases+leave 余额+goals
    - `cases`(manage_employees):工单全字段+requester+updates 列表(≤10)
  - context_fn 走 org scope(复用 entities.py 的 `_get_*_row` / `_rel_*` 思路,
    在 `app/services/ai_summary.py` 实现,不复用 API 层私有函数)。
  - prompt key `summarize.<entity>`(registry 新增三条)→ LLM 非 JSON → markdown 文本。
  - audit `entity_type=<entity> action="ai_summarize"`。
- `_mock_chat` 微调:system 含"摘要"→ 固定示例摘要文本;含"起草/撰写"→ 示例草稿
  (demo/UI 验证不再只回 "mock response")。
- 前端 `components/ai/AiSummaryCard.tsx`:折叠卡 "✨ AI 摘要" → [生成] loading →
  MarkdownContent + [重新生成]。埋点:TalentDetail 头部操作区下、EmployeeDetail 顶部、
  HrCases CaseDetail 顶部。

## T6-3 NL→watcher/saved_filter(无迁移)

sidecar tools.ts:
- 新工具 `save_filter`(WRITE_TOOLS,确认卡):`{page, name, params{}}` →
  `POST /api/saved-filters`。description 列出各页可用筛选键:
  talents{q,status,tags[]}, jobs{q,status}, cases{q,status,kind},
  employees{q,status,org_unit_id}, interviews{q}, knowledge{q,category}。
- `create_watcher` 描述补 NL 映射:"候选人X天没动→candidate_stalled{days};
  面试完没出报告→interview_pending_report;工单超时→case_open_hours{hours};
  申请积压→request_pending_days{days};预约开始前→reservation_upcoming{minutes}"。
- SYSTEM_PROMPT 追加两条行为准则:NL 提醒→create_watcher;NL 筛选收藏→save_filter;
  以及"系统怎么用"→search_knowledge(help 类)。

## T6-4 AI 文档起草(迁移 `0051`,MailMerge 智能替代)

表 `doc_templates`:`id, tenant_id FK, key(64), name(128), entity(32),
body Text, is_active, updated_by, created_at, updated_at`,UNIQUE(tenant_id,key)。

`app/services/doc_templates.py`:`DOC_SPECS` 注册表(内置模板)
- `offer_letter` 录用通知书(entity=candidates)
- `employment_certificate` 在职证明(entity=employees)
- `probation_notice` 转正通知书(entity=employees)
- `build_doc_vars(db,u,scope,entity,rid)` → 扁平 {{var}} 值(candidate_name/
  job_title/employee_name/employee_no/designation/org_name/hire_date/company 等)

`app/api/doc_templates.py`(merged view,同 prompt_templates 模式):
- `GET /doc-templates?entity=` → 内置(default|已覆盖)+ 自定义行
- `POST /doc-templates {name, entity, body}` → 自定义(key=`custom.<uuid8>`)
- `PUT /doc-templates/{key} {body}` → 内置 key 建覆盖行;自定义更新
- `DELETE /doc-templates/{key}` → 自定义删行;内置删覆盖(回默认)
- 全部 manage_employees + audit。

`POST /ai/draft-document {template_key, entity, record_id, instruction?}`:
- 解析模板(自定义行 > 覆盖行 > spec 默认);entity 不符 → 400
- `build_doc_vars` 得变量 → `render_vars(body, vars)` 预填 → 残余 {{var}} 由 LLM 补
- system = prompt `doc.draft`(registry,可租户覆盖);user = 模板 + record 摘要 context
  (复用 ai_summary 的 context_fn)+ instruction → 非 JSON 输出正文
- 返回 `{title, body, template_key}`;audit `entity_type="doc_draft" action="draft"`。

前端 `components/ai/AiDocDraftModal.tsx` + 设置页:
- 详情页 "✨AI 文档" 按钮(TalentDetail/EmployeeDetail)→ modal:模板 select
  (按 entity 过滤)+ 附加要求 input → 生成 → 可编辑 textarea 预览 →
  [复制][下载 .md][重新生成]。
- `settings/doc-templates` 页(复用 PromptSettings 结构:列表+编辑抽屉;
  自定义可删;nav.ts 加入口;router 注册)。

## 文件清单

迁移:`alembic/versions/0051_t6_ai_native.py`
后端:`api/ai.py`(新)、`api/doc_templates.py`(新)、`services/ai_summary.py`(新)、
`services/doc_templates.py`(新)、`services/product_help.py`(生成)、
`services/prompting.py`(PromptSpec.module + render_prompt module)、
`api/prompt_templates.py`(module 输出 + scope_module 写入)、
`api/knowledge.py`(CATEGORIES + help/seed)、`api/entities.py`(CREATABLE choices+help)、
`domain/models.py`(scope_module + DocTemplate)、`main.py`(2 路由)、
`integrations/llm/provider.py`(mock 摘要/起草分支)
sidecar:`tools.ts`(cardify+details.card+save_filter+描述)、`agent.ts`(tool_result emit)
前端:`components/assistant/ToolResultCard.tsx`(新)、`components/ai/AiFieldButton.tsx`、
`components/ai/AiSummaryCard.tsx`、`components/ai/AiDocDraftModal.tsx`(新×4)、
`AiAssistant.tsx`(cards 管线)、`PromptSettings.tsx`(module 分组)、
`pages/DocTemplates.tsx`(新)、`Jobs/Announcements/HrCases/Campaigns/TalentDetail/
EmployeeDetail` 埋点、`nav.ts`+`router.tsx`、`Knowledge.tsx`(help 分类)
脚本:`scripts/gen_product_help.py`、`scripts/seed_product_help.py`、
`scripts/verify_t6_live.py`
测试:`tests/test_t6.py`

## 测试点

单测(test_t6.py):
- scope_module:同 key 全局/模块覆盖共存;render_prompt module 两级回落
- field-assist:draft 返回文本(mock)/无权限 403/未知 target 400/超上下文 422
- summarize:candidates 返回 text;他租户 id → 404;无 perm → 403
- doc-templates:merged list/PUT 覆盖内置/POST custom/DELETE 还原;audit 写入
- draft-document:返回 title+body;entity 不匹配 → 400;变量预填验证
- knowledge/help/seed:建 help 文章、幂等二次跑、search 命中
- ai 端点全部经 require_feature + audit(ai_* entity_type)

链路(verify_t6_live.py):登录 → prompt-templates 看 module → field-assist draft →
summarize candidate → doc-templates CRUD → draft-document → help/seed →
knowledge/search 命中 help → saved-filters API(save_filter 工具同路径)

ego-browser:登录 → Jobs 新建弹窗 ✨起草 JD → 公告 ✨起草 → 候选人详情 AI 摘要卡 +
AI 文档 modal(生成/复制) → 员工详情 AI 摘要 → 工单 resolution ✨起草 →
campaign 邮件 ✨起草 → prompt 设置页 module 分组 → doc-templates 设置页编辑 →
知识库 help 文章可见 → 助手面板发消息看富卡片(有 LLM 时;mock 模式降级核验)
