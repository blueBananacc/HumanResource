# HROnfire 项目规则

- 每次内容更新后,必要的情况下调用 **ego-browser** 技能做自动化验证。

## docs 目录

需要项目信息(部署、设计规范、使用手册等)时,优先查看 [docs/DOCS-MAP.md](docs/DOCS-MAP.md) 文档地图找对应文档。

## AI 助手(pi-agent sidecar)

- `assistant/`(Node 22 + tsx)跑 `@earendil-works/pi-agent-core`,只接 docker 网内请求。
- 链路:前端 `/api/assistant/chat`(SSE)→ backend 认证/license/解析租户 LLM → sidecar → pi agent → 工具回打 backend 公共 API(带 tool_token,权限/租户全由 backend 判定)。
- 写工具(create_interview/generate_questions)在 sidecar `beforeToolCall` 停放,前端确认卡 → `POST /api/assistant/confirm` 放行。
- 会话存 `assistant/data/`(已 gitignore);sidecar 改代码后 `docker compose restart assistant`(override 里是 tsx watch,自动热更)。
- 构建拉不动 docker.io 时:`docker compose build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/node:22-alpine --build-arg NPM_REGISTRY=https://registry.npmmirror.com assistant`。
