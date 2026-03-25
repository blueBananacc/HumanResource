# Tool Calling 模块改造 — TODO

## 目标
将 Tool Agent 从"意图实体 → 参数映射"模式改为"LLM + JSON Output → 工具选择 + 参数生成"模式。
遵循 system_arch.md 3.2 节两阶段工具选择架构：
1. Intent Classifier → requires_tools[] 粗筛
2. Tool Agent → LLM 读取候选工具 JSON Schema → 精细选择 + 参数生成

## 改动清单

- [x] 1. `tools/registry.py` — 新增 `get_tools_summary()` 和 `get_tools_with_schemas()` 方法，支持动态获取工具信息用于 prompt 注入
- [x] 2. `tools/selector.py` — 新建 ToolSelector 类，使用 DeepSeek JSON Output 模式调用 LLM 选择工具并生成参数
- [x] 3. `config.py` — `tool_selection` 模型改为 `deepseek-chat`（支持 JSON Output）
- [x] 4. `agents/orchestrator.py` — 重写 `tool_node`，使用 ToolSelector 替代实体参数映射；移除 `_build_tool_params`
- [x] 5. `intent/classifier.py` — 可用工具列表改为从 registry 动态获取，不再硬编码；明确 entities 仅用于意图理解
- [x] 6. 新增 `tests/unit/test_tool_selector.py` — ToolSelector 单测
- [x] 7. 更新 `tests/unit/test_orchestrator.py` — 适配新 tool_node（mock ToolSelector）
- [x] 8. 更新 `tests/unit/test_tool_registry.py` — 新增 registry 方法测试
- [x] 9. 全量测试验证通过

## 状态: ✅ 全部完成 (193 tests passing)
- ToolSelector 使用 `response_format: {"type": "json_object"}` 调用 DeepSeek API
- 候选工具来自 intent 的 `requires_tools[]`（粗筛），若为空则传所有注册工具
- 上下文包含已执行的工具结果（支持多步推理）和会话上下文
- 兼容未来 MCP 工具：任何注册到 Registry 的工具（含 MCP）自动获得 schema 注入能力
- param_mapper 保留在 registry 中（向后兼容），但 tool_node 不再使用它们
