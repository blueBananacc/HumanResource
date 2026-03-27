# MCP 模块 Todo List

## Phase 14 - MCP Email 集成

- [x] config.py: 添加 MCP_SERVERS 配置
- [x] mcp/client.py: 实现 MCP Client（start/stop/register_sync）
- [x] mcp/__init__.py: 更新模块文档
- [x] agents/graph.py: compile_graph 中注册 MCP 工具
- [x] main.py: _on_exit 中停止 MCP Client
- [x] tests/unit/test_mcp_client.py: 编写单元测试（10 tests）
- [x] 全量测试验证通过（194 passed）
