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
| [hibob-style-reference.md](design/hibob-style-reference.md) | HiBob 风格前端设计参考（配色/字体/形状/交互规则，前端改动先读） |

## 维护约定

- 环境状态变化（服务器、密钥、端口）→ 更新 `ops/environment-status.md`
- 前端视觉/交互规则变化 → 更新 `design/hibob-style-reference.md`
- 新增 md → 在本文件补一行
