# MintHCM 功能移植 · 第三批(T4-A · AI 产品层)实施方案

来源:`C:\Users\666\Desktop\minthcm` 的 `legacy/include/AI/Tools`(18 工具已移一半)+
`mcp/`(完整 MCP 服务端 + OAuth + UI widget)+ `MCPDocumentation`(agent 操作手册库)。
本批主题:**把 AI 助手从"页内聊天工具"升级为"平台级 AI 产品"**。

约定同前两批:迁移从 `0047` 起;工具照旧 = 一次 backend API 调用,权限/租户/org/
license 全由 backend 判定;写工具照旧过 `beforeToolCall` 确认门。

## 调研结论(已核实)

- MCP 协议:JSON-RPC 2.0 over Streamable HTTP,`initialize`/`tools/list`/`tools/call`/`ping`。
  MintHCM 用独立 OAuth2 授权服务器;我们用**个人访问令牌**(PAT 模式)等价替代 ——
  用户自助签发的长期凭证,sha256 入库,与 invite_tokens/survey_links 同一惯例。
- sidecar 只对 docker 网内监听,**不对外发布端口** → MCP 由 backend `POST /api/mcp`
  代理进 sidecar `/mcp`,保持"sidecar 不出网"约束。
- `pg_trgm` 已确认可用(PG16,freehire 有 CREATE EXTENSION 权限),CJK 模糊匹配有效;
  **pgvector 不可用** → embedding 存 JSONB,Python 侧 cosine(知识库量级几百条足够)。
- `create_access_token` 已支持 `**claims` → tool_token 可带 `via` 声明,AI 留痕零侵入。
- `write_audit` 是统一入口,actor 上加 `token_via` 属性即可全链路传导,不用改几十处调用点。
- MintHCM MCP 对外不挂确认门(token=用户授权本身)→ 我们的 MCP 同样放行写工具;
  页内聊天的确认卡行为不变。

## T4-1 MCP Server(迁移 `0047`)

### 后端

- 表 `mcp_tokens`:id / tenant_id / user_id / name(64)/ token_hash(64,sha256,unique)/
  created_at / last_used_at / revoked_at NULL。
- `api/mcp.py`:
  - `GET /me/mcp-tokens` 列表(脱敏,name/last_used_at)
  - `POST /me/mcp-tokens {name}` → 明文 `hrm_<48hex>` 仅此一次
  - `DELETE /me/mcp-tokens/{id}` 吊销
  - `POST /mcp/token-exchange {token}` —— **仅 sidecar 内部可调**,要求
    `X-Internal-Token`(HMAC-SHA256(JWT_SECRET, 固定消息串));公开兑换会让
    PAT 越出 MCP 工具面。校验 sha256+revoked+用户状态 →
    返回 `{tool_token(via="mcp", 15min 短寿命), user:{tenant_id,id,role,name}}`;
    限 `assistant.agent` feature(用户无该 feature → 403);`last_used_at` 节流更新
  - `POST /api/mcp` 代理:透传 JSON-RPC body + Authorization 到 sidecar `/mcp`
- 前端 `pages/settings/McpTokens.tsx`(设置→AI 组):签发/吊销/复制一次明文 +
  MCP 客户端接入说明(地址 `{host}/api/mcp` + Bearer)。

### Sidecar `src/mcp.ts`

- `POST /mcp`:`Authorization: Bearer hrm_*` → 调 `POST {BACKEND}/api/mcp/token-exchange`
  换 tool_token+身份,缓存 60s(免每请求打 DB)
- 无状态 JSON-RPC:`initialize` → serverInfo+capabilities;
  `tools/list` → 复用 `hrTools()` 的 name/description/parameters;
  `tools/call` → 直接 execute(无确认门);`ping` → `{}`;未知 method → -32601
- 响应统一 JSON(不上 SSE)。

## T4-2 实体工具补齐(entities.py 扩展,无迁移)

- `GET /entities/{entity}/aggregate?agg=sum|avg|min|max|count&field=&q=` —
  registry 加 `agg_fields: {field: column}` 白名单:
  expenses(total)/payslips(gross,net)/matches(score)/campaigns(total,sent,failed)。
  新注册实体:expenses、 payslips、 matches、 campaigns(同样带 search/count builder)。
- `PATCH /entities/{entity}/{id} {fields:{k:v}}` — `write_fields` 白名单 +
  per-entity write perm + org scope + `write_audit`:
  talents(tags,display_email,display_phone,status)/
  jobs(level,description,status)/employees(phone,work_email,designation)/
  cases(priority)/skills(category,description)/rooms(name,location,capacity,equipment,is_active)。
  值校验(str 长度/enum/int 范围),未知字段 400。
- `GET /entities/{entity}/{id}/relations` — `relations` 注册表,每项
  `{key,label,builder→items|count}`:
  talents→applications/interviews/offers/resumes 数;jobs→applications 按阶段/
  interviews/matches/skills;employees→requests/cases/leave/goals/appraisals;
  applications→candidate/job;cases→requester/updates 数;rooms→reservations;
  campaigns→sends 计数。
- `api/report_defs.py` `GET /report-defs` + `POST /report-defs/{kind}/run`:
  kinds=`pipeline_funnel`/`weekly_summary`/`pending_items`/`headcount`,
  结构化 dict 返回(不复用 notify.py 的文本版),perms+feature 逐 kind 注册。
- assistant tools.ts 新增:`aggregate_records`/`update_record`(WRITE_TOOLS)/
  `list_relationships`/`run_report`/`list_report_kinds`。

## T4-3 技能剧本(迁移 `0048` 含 is_playbook)

- `knowledge_articles.is_playbook` bool;`ArticleIn.is_playbook`;
  `GET /knowledge/articles?is_playbook=true&status_filter=published`(已有过滤模式加参数);
  `GET /knowledge/articles/{id}` 已有,供 get_playbook。
- assistant tools:`list_playbooks`(标题+category)、`get_playbook(id)`(全文)。
- agent.ts SYSTEM_PROMPT 加规则:多步业务流程(发 offer/审批/面试/工单/报表等)
  先 `list_playbooks` 查 SOP,命中则 `get_playbook` 严格按步骤执行;说明剧本是租户自定义的。
- 前端 Knowledge.tsx:编辑器加"AI 剧本"勾选 + 列表徽标。

## T4-4 知识库混合检索(迁移 `0048` 含 pg_trgm + embedding)

- `llm_providers.embedding_model` 可空列;registry ResolvedProvider 透传;
  `services/embedding.py embed(db, tenant, texts)`:provider 有 base_url+key+
  embedding_model 才 POST `{base_url}/embeddings`(OpenAI 格式,10s 超时),任何失败 → None。
- `knowledge_articles.embedding` JSONB NULL;写路径(增/改/发布)尽力算 embedding
  (失败不阻塞);`POST /knowledge/embeddings/backfill`(manage_employees)→ celery
  `rebuild_knowledge_embeddings` 全量补齐。
- `_search_stmt` 改混合评分:
  - `lex = 2.0*similarity(title) + 1.0*similarity(content) + 0.6*similarity(tags)`
    (pg_trgm,`similarity()>0.05` 过滤保证召回质量)
  - 候选 = lex 命中 ∪ 旧 ILIKE 命中(回归保底),取 top-50
  - Python 侧重排:`score = lex + 0.8*cosine(q_emb, doc_emb)`(无 embedding → 纯 lex)
  - `SearchHit` 加 `score` 字段(可空,向后兼容)
- `GET /me/knowledge` 同走混合检索(参数 published_only 已分)。

## T4-5 AI 操作留痕(迁移 `0049`)

- `audit_logs.via` String(32) NULL;`AuditLog.via`。
- `assistant.py` tool_token 带 `via="assistant"`;mcp token-exchange 带 `via="mcp"`。
- `get_current_user`:`user.token_via = payload.get("via")`。
- `write_audit`:`via = getattr(actor, "token_via", None)` 自动落列。
- `GET /api/audit` AuditOut 加 via;前端 AuditCenter 表格 actor 旁显示
  `AI`/`MCP` 小徽标;`GET /api/activity` audit 源条目带 via。

## 迁移编号

`0047` mcp_tokens;`0048` knowledge(is_playbook+embedding)+ pg_trgm extension
+ llm_providers.embedding_model;`0049` audit_logs.via。

## 测试策略

`backend/tests/test_t4_ai.py`(freehire_test):
- mcp_tokens CRUD/一次性明文/revoke→exchange 401/租户隔离/feature 403
- token-exchange 成功链:tool_token 调真实 API 生效 + audit.via="mcp"
- entities:aggregate 各 agg/字段白名单/perm+feature 门;update 白名单外字段 400/
  越权 403/org 404;relations 内容+scope;report-defs 每 kind run
- knowledge:trgm 命中非全字命中、playbook 过滤、embedding stub 混合排序、
  无 embedding 降级不炸、backfill 幂等
- audit via:assistant tool_token 创建资源 → audit.via="assistant"
- sidecar /mcp:JSON-RPC initialize/tools.list/tools.call(链式测试实跑,stub 化 exchange)

ego-browser:MCP tokens 设置页签发/吊销;Knowledge 剧本勾选+徽标;
AuditCenter AI 徽标;助手调用新工具的真实面板链路(list_relationships/
run_report/list_playbooks)。

## 实施状态(2026-10-24,已完成)

- 全部 5 项功能已实现:MCP Server(令牌+exchange+sidecar /mcp)、工具缺口
  (aggregate/update/relations/run_report + 7 个新工具,共 37 个)、技能剧本
  (is_playbook + list/get_playbook + system prompt 准则 6)、混合检索
  (pg_trgm 词面 + JSONB embedding 语义召回)、AI 留痕(audit.via assistant/mcp + 前端徽标)。
- 测试:	ests/test_t4_ai.py 12 用例全绿;全量 433 用例无回归。
- 功能链路:真实 docker 环境 MCP 端到端脚本验证(initialize/tools.list/tools.call/
  error/revoke 401);list_playbooks/get_playbook/aggregate/run_report/list_relationships
  经 /api/mcp 实际调用通过。
- ego-browser:MCP 令牌页(签发→明文→吊销)、知识库剧本勾选+徽标、
  审计中心 MCP 徽标均已实测通过。
- 实施中发现:pg_trgm similarity 对长 CJK 字符串不稳定 → 词面分改为
  title/tags trgm + content ILIKE 加成;纯语义查询(零词面重叠)靠
  embedding 池召回,无 provider 时自动退化纯词面。
