# HROnfire 项目规则

- 每次内容更新后,必要的情况下调用 **ego-browser** 技能做自动化验证。

## docs 目录的作用

`docs/` 是项目的文档中心,索引见 [docs/README.md](docs/README.md)。约定:

- **新会话/新协作者先读** `docs/ops/environment-status.md`——它记录了当前线上部署与本地环境的真实状态、启动命令和已踩的坑,是接手上下文的最快入口。
- **做前端视觉/交互改动前**读 `docs/design/hibob-style-reference.md`,遵循其中的 HiBob 风格规范,不要凭印象改配色/字体/组件样式。
- 环境状态变化(服务器、密钥、启动方式)后同步更新 `environment-status.md`;新增任何 md 文档后在 `docs/README.md` 地图里补一行。
