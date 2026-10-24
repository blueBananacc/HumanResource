# HROnfire - AI 原生的人力资源管理系统(HCM)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.146+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-blue" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5.x-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Redis-7-red" alt="Redis">
  <img src="https://img.shields.io/badge/Node-22-green" alt="Node">
  <img src="https://img.shields.io/badge/License-AGPL-yellow" alt="License">
</p>

## 📖 HROnfire 简介

HROnfire 是一款 **AI 原生**的企业级人力资源管理系统,支持私有化部署。从招聘 ATS 到核心人力(HRIS),AI 不是外挂功能,而是长在每个工作流里的默认能力:侧边 AI 助手可以查数、办单、出草稿,表单上有字段级 AI 写作,详情页有 AI 摘要,外部 AI 客户端还能通过 MCP Server 直接操作系统数据。

系统基于 FastAPI + React + 独立 AI 助手 sidecar(Node 22 + pi-agent)构建,内置 PostgreSQL 与 Redis,通过 Docker Compose 一键启动即可获得完整体验;LLM 默认使用 mock,无需外部 API key 也能跑通简历匹配、AI 出题、自动评分、字段起草等核心流程,也可以随时接入 OpenAI 兼容的真实大模型。

## 项目特性

### 招聘 ATS

- 简历上传/邮箱采集、解析、候选人去重与原件管理
- 人才库沉淀:简历版本、面试历史、标签、分组、黑名单、备注与时间线
- 人才查重合并(关联整体迁移)、CSV 批量导入(文件内+库内双去重)
- 岗位管理、岗位治理(能力模型/JD 优化/审批)、招聘管道看板
- 招聘需求审批、Offer 管理、人才外联活动(campaign)
- 基于简历或岗位生成题集,题库复用与 AI 批量出题
- 候选人远程文本/语音面试,自动评分并生成报告;面试安排与会议室预约

### AI 原生能力

- **AI 助手面板**:自然语言查数/办单,工具结果渲染成可操作的富卡片(候选人表、面试、匹配分、知识命中),写操作先出确认卡再放行
- **字段级 AI 写作**:JD、公告、工单解决方案、外联邮件等表单旁的 ✨ 起草/润色/扩写/精简
- **AI 记录摘要**:候选人 360°/员工档案/工单详情页一键生成概况卡
- **AI 文档起草**:offer 函/在职证明等模板 + `{{变量}}` 自动预填 + LLM 补全,可编辑导出
- **自然语言建规则**:"每周一提醒停滞候选人"→ watcher;"面试阶段的候选人"→ 已存筛选
- **混合知识检索**:PostgreSQL 全文/三元组 + embedding 语义检索,租户知识库 + 内置产品帮助文档
- **提示词管理**:注册表化 prompt 模板,租户级 + 模块级两级覆盖,版本/回滚/预览
- **AI 操作留痕**:assistant/MCP 发起的调用按 `via` 区分,统一进审计流

### 开放集成

- **MCP Server**:外部 AI 客户端(Claude Desktop 等)经 PAT 直连,tools/resources 读写白名单实体,租户隔离与权限由后端统一判定
- **SOP 技能剧本**:租户可自定义助手 playbook,让 AI 按既定流程办事
- 通用实体 API(schema/create/update/delete/关系链接)+ 乐观锁
- 报表定义 + 定时邮件订阅 + Webhook 式的触发提醒(watcher)

### 核心人力(HRIS)

- 员工档案、组织架构图、入转调离生命周期
- 请假(类型/政策/余额/审批)、考勤(记录/打卡流水/补签/加班/日历)
- 绩效目标与考核周期、费用报销、薪酬结构与薪资带宽
- 员工服务:公告、申请、HR 工单、知识库;员工自助(ESS)门户
- 多角色权限 × 组织数据域、租户自定义字典、后台任务与 job-run 可观测、全量审计

### 平台与安全

- 多租户隔离、License 版本与配额(community / professional / enterprise)
- 多角色权限、团队协作、SSO、审计追踪、PIPL 联系人脱敏
- 60+ REST API 模块、40+ 测试文件、全链路冒烟脚本

## 角色模型

系统默认包含以下角色:

- `admin`:系统管理、团队、审计、License 等全权限操作
- `hr`:岗位、简历、面试、员工、假勤等日常 HR 操作
- `interviewer`:参与面试和查看结果
- `hiring_manager`:查看职责范围内的岗位与报告,并可编辑岗位
- `viewer`:只读查看
- `employee`:员工自助账号,仅访问 `/api/me/*` 行级数据

接口会结合角色权限和组织范围做访问控制。

## 快速启动

前置要求:

- Docker Desktop 或 Docker Engine 24+
- 已启用 BuildKit

启动开发体验环境:

```bash
cd deploy
docker compose up -d --build
```

首次构建通常需要 5-10 分钟。服务启动后访问 <http://localhost>,默认账号如下:

```text
邮箱: admin@example.com
密码: admin123456
```

默认体验包含:

- 自动 bootstrap 管理员账号
- 自动注入示例岗位
- 默认进入 30 天试用期,能力等同 `professional`
- LLM 默认使用 mock,无需外部 API key 也可跑通主要流程

完整体验步骤见 [`docs/ops/quickstart.md`](docs/ops/quickstart.md)。

## 配置真实模型

若需要接入真实 LLM,可在 `backend/.env` 中配置兼容 OpenAI 的接口:

```bash
cat > backend/.env <<'EOF'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_DEFAULT_MODEL=openai/gpt-4o-mini
EOF

cd deploy
docker compose down
docker compose up -d
```

支持 OpenAI 兼容 API,包括 OpenAI、阿里通义、自建 vLLM / Ollama 或其他网关;也可以登录后在 **设置 → LLM 配置** 页面直接改(按租户生效,支持 embedding 模型单独配置)。

## 仓库结构

```text
backend/        FastAPI 服务、领域模型、任务、测试与验证脚本
frontend/       React 前端
assistant/      AI 助手 sidecar(Node 22 + pi-agent,工具调用/确认门/SSE)
deploy/         Docker Compose、Dockerfile、Nginx 配置
docs/           设计规范、运维与部署文档(入口见 DOCS-MAP)
```

## 文档

- 本地体验:[`docs/ops/quickstart.md`](docs/ops/quickstart.md)
- 部署说明:[`docs/ops/customer-guide.md`](docs/ops/customer-guide.md)
- 离线安装:[`docs/ops/offline-install.md`](docs/ops/offline-install.md)
- 文档地图:[`docs/DOCS-MAP.md`](docs/DOCS-MAP.md)
- 前端设计规范:[`docs/design/hibob-style-reference.md`](docs/design/hibob-style-reference.md)
