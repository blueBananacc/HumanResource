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
    """根据意图识别结果解析路由目标 Agent 列表和意图映射。

    支持四种场景：
    1. 单意图 + 单Agent → 直接路由
    2. 单意图 + 多Agent → primary + secondary 串行
    3. 多意图 + 各意图单Agent → 按逻辑顺序依次路由
    4. 多意图 + 各意图多Agent → 合并去重，追踪每个Agent服务的意图

    Args:
        intent_result: 意图识别结果。

    Returns:
        (target_agents, agent_intent_map) 二元组。
        - target_agents: 有序的目标 Agent 名称列表（去重）。
        - agent_intent_map: 与 target_agents 等长的映射列表，
          每项 {"agent": str, "intent_indices": [int]} 标识该 Agent
          服务的意图下标。
    """
    agents: list[str] = []
    agent_map: list[dict] = []
    seen: dict[str, int] = {}  # agent_name → index in agents/agent_map

    for intent_idx, intent_item in enumerate(intent_result.intents):
        route = ROUTING_TABLE.get(
            intent_item.label, ROUTING_TABLE[IntentLabel.UNKNOWN]
        )
        for agent in [route["primary"]] + route["secondary"]:
            if agent not in seen:
                seen[agent] = len(agents)
                agents.append(agent)
                agent_map.append({
                    "agent": agent,
                    "intent_indices": [intent_idx],
                })
            else:
                # Agent 已存在，追加意图下标
                map_entry = agent_map[seen[agent]]
                if intent_idx not in map_entry["intent_indices"]:
                    map_entry["intent_indices"].append(intent_idx)

    if not agents:
        return ["rag_agent"], [{"agent": "rag_agent", "intent_indices": [0]}]

    return agents, agent_map
