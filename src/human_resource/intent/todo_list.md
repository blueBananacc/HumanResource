# 意图识别四场景完善 — TODO

## 目标
完善意图识别逻辑，正确处理四种场景：
1. 单意图 + 单Agent（如 policy_qa → rag_agent）
2. 单意图 + 多Agent（如 process_inquiry → rag_agent + tool_agent 串行）
3. 多意图 + 每个意图单Agent（如 [employee_lookup, policy_qa] → [tool_agent, rag_agent]）
4. 多意图 + 每个意图多Agent（如 [employee_lookup, process_inquiry] → [tool_agent, rag_agent, tool_agent]）

## 核心问题
- requires_tools 在 IntentResult 顶层，无法关联到具体 intent
- tool_node 只用 primary_intent.entities 给所有工具，多意图下参数错误
- 缺少 intent → agent 映射追踪，agent 不知道为哪个 intent 服务

## 任务列表

- [x] 1. models.py: IntentItem 增加 requires_tools 字段
- [x] 2. state.py: AgentState 增加 agent_intent_map 字段
- [x] 3. classifier.py: 更新 prompt 输出 per-intent requires_tools + 四场景 few-shot
- [x] 4. classifier.py: _parse_response 解析 per-intent requires_tools
- [x] 5. router.py: resolve_route 返回 (agents, agent_intent_map)
- [x] 6. orchestrator.py: route_agents_node 存储 agent_intent_map
- [x] 7. orchestrator.py: tool_node 按 intent 匹配 entities
- [x] 8. 测试: 4 种场景全覆盖
- [x] 9. 运行测试验证 — 136 passed, 0 failed
