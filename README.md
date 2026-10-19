# HROnfire - AI时代下的人力资源系统
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.146+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-blue" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5.x-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Redis-7-red" alt="Redis">
  <img src="https://img.shields.io/badge/License-AGPL-yellow" alt="License">
</p>
 
## 📖 HROnfire简介

HROnfire 是一款 AI 驱动的企业级智能招聘管理系统，支持私有化部署，覆盖 **简历采集解析 → 人才库沉淀 → 岗位管理 → AI 匹配 → 远程面试 → 评估报告** 的完整招聘流程，帮助团队高效完成人才筛选与面试评估。

系统基于 FastAPI + React 构建，内置 PostgreSQL 与 Redis，通过 Docker Compose 一键启动即可获得完整体验；LLM 默认使用 mock，无需外部 API key 也能跑通简历匹配、AI 出题、自动评分等核心流程，也可以随时接入 OpenAI 兼容的真实大模型。

## 项目特性

- 简历上传、解析、候选人去重与原件管理
- 人才库沉淀，聚合简历版本、面试历史、标签、备注与时间线
- 岗位创建、岗位详情、岗位匹配与岗位治理
- 基于简历或岗位生成题集，支持题库复用与 AI 批量生成
- 候选人远程文本 / 语音面试，自动评分并生成报告
- KPI、漏斗、趋势、评分分布等分析能力
- 多角色权限、团队协作、SSO、审计与系统配置

## 角色模型

系统默认包含以下角色：

- `admin`：系统管理、团队、审计、License 等全权限操作
- `hr`：岗位、简历、面试、报告等日常招聘操作
- `interviewer`：参与面试和查看结果
- `hiring_manager`：查看职责范围内的岗位与报告，并可编辑岗位
- `viewer`：只读查看

接口会结合角色权限和组织范围做访问控制。

## 快速启动

前置要求：

- Docker Desktop 或 Docker Engine 24+
- 已启用 BuildKit

启动开发体验环境：

```bash
cd deploy
docker compose up -d --build
```

首次构建通常需要 5-10 分钟。服务启动后访问 <http://localhost>，默认账号如下：

```text
邮箱: admin@example.com
密码: admin123456
```

默认体验包含：

- 自动 bootstrap 管理员账号
- 自动注入示例岗位
- 默认进入 30 天试用期，能力等同 `professional`
- LLM 默认使用 mock，无需外部 API key 也可跑通主要流程

完整体验步骤见 [`docs/ops/quickstart.md`](docs/ops/quickstart.md)。

## 配置真实模型

若需要接入真实 LLM，可在 `backend/.env` 中配置兼容 OpenAI 的接口：

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

支持 OpenAI 兼容 API，包括 OpenAI、阿里通义、自建 vLLM / Ollama 或其他网关。

## 仓库结构

```text
backend/        FastAPI 服务、领域模型、任务与测试
frontend/       React 前端与 E2E 用例
deploy/         Docker Compose、Dockerfile、Nginx 配置
docs/           公开运维与部署文档
```

## 文档

- 本地体验：[`docs/ops/quickstart.md`](docs/ops/quickstart.md)
- 部署说明：[`docs/ops/customer-guide.md`](docs/ops/customer-guide.md)
- 离线安装：[`docs/ops/offline-install.md`](docs/ops/offline-install.md)