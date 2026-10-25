# T11 设计:逐次 LLM 调用留痕 + 岗位级知识 chunks

> 来源:recruit-ai-platform `audit/writeModelCallLog` + `ai-gateway`(schema 校验/严格重试/版本号/input digest)+ `jobs` 模块 `job_knowledge_chunks`(段落切块→索引→喂给面试计划/咨询)。按我方多租户 FastAPI/SQLAlchemy 架构适配。

## 1. 逐次 LLM 调用留痕(model_call_logs)

### 1.1 表结构(迁移 0056)

`model_call_logs`:

| 列 | 类型 | 说明 |
|---|---|---|
| id | String(36) PK | uuid |
| tenant_id | String(36) FK tenants cascade | 租户隔离(NULL=系统级调用,如 env provider 测试) |
| user_id | String(36) NULL | 发起用户(可空,后台任务/助手会话无 HTTP 用户时为空) |
| call_type | String(48) | 业务调用类型,见 §1.4 |
| provider | String(64) | resolved provider source: `db`/`env`/`mock` |
| model | String(128) | LiteLLM 模型标识 |
| prompt_key | String(64) NULL | 租户 prompt 模板键(如 `match.evaluate`),无模板为空 |
| prompt_version | String(32) | prompt 内容 sha256[:12](模板渲染后) |
| schema_version | String(16) | 结构化输出约定版本,默认 `v1` |
| input_digest | String(16) | messages canonical sha256[:16] |
| output | JSONB | 归一化后的结构化输出(JSON 调用) |
| raw_output | Text | 原始模型文本,截断 20k |
| error_kind | String(32) | ``=成功;llm_error/parse_error/schema_error/timeout |
| error_message | Text | 错误摘要,截断 500 |
| latency_ms | Integer | chat() 往返耗时(含重试的总耗时分列累计) |
| created_at | DateTime | |

索引:`(tenant_id, created_at)`、`(tenant_id, call_type)`。

与 `audit_logs` 的分工:audit 记"谁做了什么业务动作",model_call_logs 记"每次模型调用的输入指纹/输出/耗时/错误"——AI 可观测性层,回答"模型当时看到什么、返回什么、为什么错"。

### 1.2 统一网关 `call_llm_json` / `call_llm`

在 `app/integrations/llm/provider.py` 新增(不改动 `chat()` 低层签名):

```python
def call_llm_json(
    call_type: str,
    messages: list[Message],
    *,
    db: Session | None = None,
    tenant_id: str | None = None,
    provider: ResolvedProvider | None = None,
    normalize=None,            # (parsed:dict)->dict 业务侧归一化
    required: tuple[str,...]=(),# schema 校验:必需顶层键
    temperature: float = 0.2,
    prompt_key: str | None = None,
    user_id: str | None = None,
    retry_strict: bool = True,
) -> dict
```

管线(对齐 recruit `callOpenAiCompatible`):
1. `chat(response_json=True)` → `parse_json`(已含 ```json 抢救)→ `normalize()` → `required` 键校验
2. 任一环节失败且 `retry_strict`:追加 system 消息"上次输出无法解析为合法 JSON,仅输出 JSON 对象",temperature=0 重试一次
3. 仍败 → 抛 `LLMCallError(error_kind, parsed/raw 摘要)`;成功 → 返回归一化 dict
4. **无论成败都写 model_call_logs**(error_kind 区分 llm_error=传输/parse_error=JSON/schema_error=缺字段);`raw_output` 恒记

`call_llm(call_type, messages, ...)` 同管线去 JSON 段(纯文本调用,如面试开场白)。

### 1.3 留痕写入服务 `services/model_calls.py`

`write_model_call(...)` 用**独立 SessionLocal** 提交——业务事务回滚不丢日志(recruit 语义:即时写)。写入失败仅 warning,绝不反噬业务链路。`raw_output`/`error_message` 入库前截断。

### 1.4 调用点接入(call_type 命名)

| call_type | 位置 |
|---|---|
| `match.evaluate` | services/match_evaluator.py |
| `interview.score` | services/interviewer.py 单轮评分 |
| `interview.opening` | interviewer.py 开场白(文本) |
| `interview.report` | interviewer.py 总结/report |
| `import.column_map` | services/imports.py 列映射 |
| `import.explain` | imports.py 行解释 |
| `scorecard.suggest` | scorecards.py 维度建议 |
| `scorecard.synthesize` | scorecards.py 聚合 |
| `question.generate` | question_generator.py |
| `question_bank.generate` | question_bank.py |
| `question_library.generate` | api/question_library.py |
| `job_governance.*` | job_governance.py ×2(按函数名细分) |
| `ai.field_assist` / `ai.doc_draft` | api/ai.py |
| `llm.test` | api/llm.py 连通性测试 |
| `assistant.turn` | sidecar 每轮回报(见 §1.6) |

各点保留原有 fallback 语义——网关只负责"调用+校验+留痕",业务兜底不变。

### 1.5 运维 API + UI

- `GET /api/model-call-logs`(`manage_settings`):分页+过滤(call_type/error_kind/时间窗/has_error)
- `GET /api/model-call-logs/stats`:总量/错误率/按 call_type 分组/平均·P95 延迟/近 24h 趋势
- 前端 `/settings/model-calls`:统计卡+过滤器+表格+详情抽屉(output/raw/error 全文)
- 侧边栏挂在「幂等键」旁

### 1.6 助手侧留痕

sidecar 在 `agent_end`(成功)与 `message_end stopReason=error`(失败)时,用已有 toolToken 调 `POST /api/assistant/model-call-log`(backend 校验 token→tenant/user 归属):

```json
{ "call_type":"assistant.turn", "model": "...", "latency_ms": 1234,
  "input_digest": "sha256[:16](last user message)",
  "output": {"tools_called": ["hr_get_candidate"], "reply_chars": 512},
  "error_kind": "", "error_message": "" }
```

backend 该端点只校验 token+租户,不写 audit(这是观测数据不是业务动作)。粒度=pi-agent 的"一轮"(可能内含多次 provider 往返,对齐 recruit 每 callType 一行的口径)。

## 2. 岗位级知识 chunks(job_knowledge)

### 2.1 表结构(同一迁移 0056)

`job_knowledge_sources`:每个岗位挂载的资料源

| 列 | 类型 | 说明 |
|---|---|---|
| id / tenant_id / job_id | uuid/FK | job cascade 删 |
| name | String(128) | 资料名 |
| kind | String(8) | `text`(内联文本)/ `file`(上传文件) |
| content | Text NULL | kind=text 时的原文 |
| file_name / mime / storage_key | NULL | kind=file 时 ObjectStore 引用 |
| status | String(16) | `indexing`→`ready`/`failed` |
| chunk_count | Integer | |
| extraction_status | String(16) | ok/partial/failed/none(纯文本) |
| extraction_method | String(32) | pymupdf/pdfplumber/docx/txt |
| error_message | Text | |
| created_by / created_at / updated_at | | |

`job_knowledge_chunks`:

| 列 | 类型 |
|---|---|
| id / tenant_id / job_id / source_id | uuid(FK 仅到 sources 逻辑层,job/tenant 为冗余列便于过滤) |
| chunk_index | Integer |
| content | Text |
| meta | JSONB:{source_name, extraction_status, extraction_method} |
| embedding | JSONB NULL(复用 embed_texts,best-effort) |

索引:`(job_id, source_id, chunk_index)`。

### 2.2 服务 `services/job_knowledge.py`

- `split_chunks(text, size=900, cap=40)`——移植 recruit `splitKnowledgeChunks`:规范化空白→按空行分段→贪心装箱→超长硬切
- `index_source(db, source)`:kind=text 直接用 content;kind=file 从 ObjectStore 取字节→`resume_parser._extract_text_by_format`→切块→删旧插新→best-effort embedding→状态机+`write_audit(action="job_knowledge.indexed"/"index_failed")`(入 T7 自动化漏斗)
- `job_knowledge_context(db, job, *, query=None, limit=6)`:ready sources 的 chunks;有 query 且 chunk 有 embedding→cosine top-N;否则按 (source,idx) 顺序取前 N。返回 `[(source_name, content)]`

### 2.3 API `api/job_knowledge.py` 挂 `/api/jobs/{id}/knowledge-sources`

- GET 列表+`GET /chunks` 预览(需 `PERM_VIEW_JOBS` 或复用 jobs 读权限)
- POST text 源 / POST file 源(multipart 上传→ObjectStore)/ DELETE / POST `/{sid}/reindex`(需 `PERM_WRITE_JOBS`)
- 文件上限 5MB,类型 pdf/docx/txt/md

### 2.4 消费点(让 AI 输出有据可依)

- `match_evaluator._user_message`:加"岗位知识摘要"段(query=job.title+skills)
- `question_generator`:出题 prompt 注入(query=job.title+技能)
- `interviewer` 开场/出题(query=候选人技能+岗位名)
- 无 ready 源时行为与现状完全一致(零回归)

### 2.5 前端

- `JobDetail` 新增「岗位知识」区:源列表(状态/chunk 数/抽取方式/错误)+「添加文本资料」/「上传文件」+「重新索引」+ chunks 预览抽屉
- 评分卡/匹配评估页无需改动(消费在后端 prompt 内)

## 3. 测试矩阵

- test_t11:切块边界(空文/超长/40 上限)、text/file 索引、失败态、cosine 召回、隔离;网关成功/parse 失败重试/schema 失败/strict retry 生效/独立会话留痕;assistant 回报端点;ops API 过滤/权限
- live:全链路(加源→索引→匹配评估 prompt 含知识→model_call_logs 落行→ops API)
- ego:岗位知识 UI 全流程 + 模型调用运维页
