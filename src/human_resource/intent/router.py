"""意图 → Agent 路由映射。

根据 IntentResult 决定路由到哪些 Specialist Agent。
供 LangGraph conditional_edges 使用。
"""

from __future__ import annotations

from human_resource.schemas.models import IntentLabel, IntentResult

# 路由映射表：intent_label → (主 Agent, 辅助 Agent 列表, 执行模式)
ROUTING_TABLE: dict[IntentLabel, dict] = {
    IntentLabel.POLICY_QA: {
        "primary": "rag_agent",
        "secondary": [],
        "mode": "single",
    },
    IntentLabel.DOCUMENT_SEARCH: {
        "primary": "rag_agent",
        "secondary": [],
        "mode": "single",
    },
    IntentLabel.EMPLOYEE_LOOKUP: {
        "primary": "tool_agent",
        "secondary": [],
        "mode": "single",
    },
    IntentLabel.TOOL_ACTION: {
        "primary": "tool_agent",
        "secondary": [],
        "mode": "single",
    },
    IntentLabel.PROCESS_INQUIRY: {
        "primary": "rag_agent",
        "secondary": ["tool_agent"],
        "mode": "serial",
    },
    IntentLabel.MEMORY_RECALL: {
        "primary": "memory_agent",
        "secondary": [],
        "mode": "single",
    },
    IntentLabel.CHITCHAT: {
        "primary": "orchestrator",
        "secondary": [],
        "mode": "direct",
    },
    IntentLabel.UNKNOWN: {
        "primary": "rag_agent",
        "secondary": [],
        "mode": "fallback",
    },
}


def resolve_route(
    intent_result: IntentResult,
) -> tuple[list[str], list[dict]]:
    """根据意图识别结果解析执行计划（不去重）。

    按意图的逻辑顺序，依次展开每个意图的 Agent 链（primary + secondary），
    生成扁平的执行计划。同一 Agent 可出现多次（服务不同意图），以保证
    存在顺序依赖的意图能按正确顺序分步执行。

    支持四种场景：
    1. 单意图 + 单Agent → 直接路由
    2. 单意图 + 多Agent → primary + secondary 串行
    3. 多意图 + 各意图单Agent → 按逻辑顺序依次路由
    4. 多意图 + 各意图多Agent → 按意图展开，不去重

    Args:
        intent_result: 意图识别结果。

    Returns:
        (target_agents, execution_plan) 二元组。
        - target_agents: 有序的目标 Agent 名称列表（可含重复）。
        - execution_plan: 与 target_agents 等长的执行步骤列表，
          每项 {"agent": str, "intent_index": int} 标识该步骤
          服务的单个意图下标。
    """
    execution_plan: list[dict] = []

    if intent_result is None or not intent_result.intents:
        return ["rag_agent"], [{"agent": "rag_agent", "intent_index": 0}]

    for intent_idx, intent_item in enumerate(intent_result.intents):
        route = ROUTING_TABLE.get(
            intent_item.label, ROUTING_TABLE[IntentLabel.UNKNOWN]
        )
        for agent in [route["primary"]] + route["secondary"]:
            execution_plan.append({
                "agent": agent,
                "intent_index": intent_idx,
            })

    if not execution_plan:
        execution_plan = [{"agent": "rag_agent", "intent_index": 0}]

    target_agents = [step["agent"] for step in execution_plan]
    return target_agents, execution_plan
