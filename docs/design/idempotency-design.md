# T10 — 生产级幂等键(Idempotency Keys)

> 参考实现:Gauzy `packages/core/src/lib/idempotency/`(entity + policy + service + interceptor + decorator + cleanup scheduler)。
> 目标:可重试写操作 = 一个 key 一次副作用。AI 链路(确认门后的工具执行、MCP 客户端重试、前端双击/断网重试)不再重复落库。

## 1. 语义移植清单(Gauzy → 本项目)

| 语义 | Gauzy | 本项目适配 |
|---|---|---|
| 唯一锁元组 | `(tenant_id, organization_id, scope, key)` — nullable org 需 zero-uuid 折叠 | `(tenant_id, scope, key)` — 我们所有写路径都带认证 → tenant 恒非空,**不需要折叠** |
| 请求指纹 | `sha256(method, canonical path/query/body)` — key 同 body 不同 = REUSED 409 | 同。canonical:JSON key 排序、query 参数排序、path 去 query |
| 决策表 | SKIP / REQUIRE_KEY / INVALID_KEY(8–255) / EXECUTE / REPLAY / IN_FLIGHT(409+Retry-After) / REUSED_KEY(409+hash 前8位) | 全移植 |
| 状态机 | `in_progress → completed \| failed` | 同。`settle` 对已终态行幂等(崩溃安全) |
| 响应存储 | status + JSON body,1MB 上限,超限只存 status | 同(1MB) |
| 失败回放 | HTTPException 的 detail 信封也存,重试看到同样的 4xx | 同:<400 直返 body + `Idempotency-Replayed: true`;>=400 重建 HTTPException |
| 锁接管 | `locked_at` + staleLock(120s) + `FOR UPDATE` 内重校验 + 续约 expires_at | 同(`with_for_update`) |
| 惰性过期 | claim 时遇到 expired 且 lease 死 → 删行再抢 | 同 |
| 清扫 | 每小时有界 sweep;终态过期删 + 锁死过期删,谓词双写 | 挂 `_watch_loop`,每小时一次(模块级时间戳),record_run `idempotency_purge` |
| 操作员视图 | list/get/release,view **不含** response_body(回放体是调用方的数据) | `GET/DELETE /api/idempotency-keys`,perm=manage_settings |
| 声明式接入 | `@Idempotent(scope, required, ttl, resourceType)` + 全局 interceptor | `@idempotent(...)` 装饰器直包 FastAPI 端点(`__signature__` 注入 request/response) |

## 2. 数据模型(迁移 0055)

`idempotency_keys`:
- `id` uuid pk;`tenant_id` FK cascade,非空,index
- `user_id` FK setnull —— 谁提交的(运维视图/审计)
- `scope` varchar(64):操作命名空间,`entities.create` / `assistant.confirm`…
- `key` varchar(255):客户端键
- `request_hash` char(64):canonical 请求 SHA-256
- `status` varchar(16):in_progress / completed / failed
- `response_status` int;`response_body` JSONB
- `resource_type` varchar(64) / `resource_id` varchar(36):产物指针
- `expires_at` DateTime index(默认创建+24h,可 per-route ttl 60s–7d clamp)
- `locked_at` DateTime(claim 时刻;>120s 视为死锁可接管)
- `created_at` / `updated_at`
- `UniqueConstraint(tenant_id, scope, key)`

## 3. FastAPI 接入形态

Gauzy 是「全局 interceptor + `@Idempotent` 元数据」;FastAPI 等价物是**直接装饰端点函数**:

```python
@router.post("/scorecards", status_code=201)
@idempotent(scope="scorecards.create", resource_type="interview_scorecard")
def create_scorecard(body: ScorecardIn, ...): ...
```

装饰器内:
1. safe method / 无 key → 直通(required=True 且无 key → 400 `IDEMPOTENCY_KEY_REQUIRED`)
2. canonical hash → `services.idempotency.claim()`(**独立 SessionLocal**,claim 行立即 commit —— 它是给别的并发事务看的锁)
3. REPLAY → `response.status_code` + replay headers + 返回存 body(>=400 时 raise HTTPException 重建 detail 信封)
4. IN_FLIGHT → 409 + Retry-After;REUSED → 409 + hash 前缀对
5. EXECUTE → `run_in_threadpool` 调原函数 → 序列化结果存行(BaseModel→dict;超限只存 status;提取 `result.id` 作 resource_id)
6. 异常 → `fail()` 存 status+detail 信封后原样抛出;settle 失败只记日志(工作已做,不能让"存不上"诱发重试)

### 采用端点

| 端点 | scope | 说明 |
|---|---|---|
| `POST /api/entities/{entity}` | `entities.create` | 通用创建(助手 create_record 走这里) |
| `PATCH /api/entities/{entity}/{rid}` | `entities.update` | 通用更新(助手 update_record) |
| `POST /api/assistant/confirm` | `assistant.confirm` | **header 缺失时以 `call_id` 兜底**——call_id 天然唯一,确认卡双击/重试零成本防重 |
| `POST /api/scorecards` | `scorecards.create` | 新建评分卡 |
| `POST /api/scorecards/{id}/feedback` | `scorecards.feedback` | 反馈提交(天然 upsert,仍罩一层) |
| `POST /api/scorecards/{id}/synthesize` | `scorecards.synthesize` | AI 合成(贵) |
| `POST /api/imports/{id}/run` | `imports.run` | 导入执行(最重写操作) |

## 4. 助手 sidecar:工具级幂等

`tools.ts`:
- `callApi` opts 加 `key?: string` → 非 GET 时写 `Idempotency-Key` header。
- `AsyncLocalStorage` 包裹 `WRITE_TOOLS` 的 execute:`Idempotency-Key: tool:{call_id}`。同一工具调用无论重试/重放几次,落到后端都是同一把锁 —— `entities.*`/`scorecards.*` 装饰端点全部自动受保护,**40 个工具零改动**(包装发生在 `hrTools()` 返回前)。

## 5. 前端

- `fetchJSON` init 加 `idempotencyKey?: string` → `Idempotency-Key` header。
- `AiAssistant` confirm:header 缺省时由 `key_field` 兜底:`call_id:{callId}`(命名空间前缀同时补长度校验)(与服务端 call_id 兜底同值,双保险)。
- `ScorecardSection` 建卡/反馈 modal 每次打开生成 `crypto.randomUUID()` 作 key —— 双击/重复提交合并。
- 新设置页 `/settings/idempotency`:统计卡(in_progress/completed/failed/24h)+ 列表(scope/key 前缀/状态/资源/用户/到期)+ 释放按钮(活锁 409 提示)+ 自动刷新。导航挂在「系统」组。

## 6. 运维 API

- `GET /api/idempotency-keys?scope=&status=&q=&limit=` — 分页列表(剥 response_body)
- `GET /api/idempotency-keys/{id}` — 单条(同剥)
- `DELETE /api/idempotency-keys/{id}` — 释放;活锁 → 409 + retry_after_ms
- `GET /api/idempotency-keys/stats` — 状态计数 + 最近 24h 新建数
- 权限 `PERM_MANAGE_SETTINGS`;租户隔离。

## 7. 不搬的部分

- GraphQL 多 mutation 共请求形态(`idempotencyKey` input member)—— 我们没有 GraphQL。
- 双方言 ORM 分支 —— PostgreSQL 单方言,`insert ... on conflict do nothing` / IntegrityError 回读即可。
- `@Idempotent` class 级声明 —— 用不上。

## 8. 验证计划

- 单测 `test_t10.py`:canonical hash(key 重排/query 乱序同 hash,body 不同异 hash)、key 长度边界、决策表全路径、claim→complete→replay、reused 409、in-flight、stale 接管、过期重抢、purge 保活锁、release 拒活锁、租户隔离、entities PATCH 端点级回放(审计不产生第二条)、confirm call_id 兜底。
- live `verify_t10_live.py`:真实 HTTP 全链路 + 租户隔离 + stats/release API + 前端 key 生效。
- ego-browser:设置页渲染 + 评分卡建卡双击防重 + 释放按钮。
