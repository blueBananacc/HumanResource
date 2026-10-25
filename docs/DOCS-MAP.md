# 文档地图

本目录是项目的文档中心。每个 md 一行说明：

## ops/ — 运维与部署

| 文件 | 说明 |
|---|---|
| [quickstart.md](ops/quickstart.md) | 本地 5~10 分钟跑起来的最快速指南 |
| [customer-guide.md](ops/customer-guide.md) | 面向非 IT 客户/HR 的使用手册（初始化、日常操作） |
| [offline-install.md](ops/offline-install.md) | 离线安装包部署手册（无网络环境，面向 IT/系统管理员） |
| [deploy-guide.md](ops/deploy-guide.md) | 从零买服务器到自动部署的逐步教程（尚未有服务器时的搭建原理） |
| [environment-status.md](ops/environment-status.md) | **当前环境现状速查**——线上/本地地址、启动命令、secrets、已踩的坑（新会话先读这篇） |

## design/ — 设计规范

| 文件 | 说明 |
|---|---|
| [hibob-style-reference.md](design/hibob-style-reference.md) | 前端设计规范（配色/字体/形状/交互规则，前端改动先读） |
| [hrms-suite-design.md](design/hrms-suite-design.md) | 员工核心（HRIS）领域模型与权限设计（Employee/假勤/生命周期/绩效/费用/薪酬） |
| [ats-modules-design.md](design/ats-modules-design.md) | 招聘侧模块设计（管道看板/提示词/邮件模板/技能/会议室/问卷，表设计与验收基线） |
| [employee-services-design.md](design/employee-services-design.md) | 员工服务与 AI 基建设计（公告/申请/薪酬带宽/HR工单 + 通知/watcher/知识库/报表订阅/外联/ICS/活动流/实体检索） |
| [ai-platform-design.md](design/ai-platform-design.md) | AI 平台层设计（MCP Server/通用实体工具/SOP 剧本/混合检索/AI 留痕） |
| [platform-capabilities-design.md](design/platform-capabilities-design.md) | 平台能力层设计（实体写 API/已存筛选/收藏/候选人合并/批量导入/字典/调度运维/变更历史/乐观锁） |
| [ai-native-design.md](design/ai-native-design.md) | AI 原生工作流设计（工具富卡片/模块级提示词/产品帮助/字段级 AI/记录摘要/文档起草） |
| [automation-attendance-design.md](design/automation-attendance-design.md) | 事件驱动自动化 + 考勤深化设计（审计漏斗规则引擎/迟到早退/IP 白名单/月度汇总） |
| [ai-import-wizard-design.md](design/ai-import-wizard-design.md) | AI 导入向导设计（四步向导/LLM 列映射/校验归因/错误文件） |

## 维护约定

- 环境状态变化（服务器、密钥、端口）→ 更新 `ops/environment-status.md`
- 前端视觉/交互规则变化 → 更新 `design/hibob-style-reference.md`
- 新增 md → 在本文件补一行
