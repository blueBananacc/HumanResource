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
        "secondary": ["memory_agent"],
        "mode": "serial",
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


def resolve_route(intent_result: IntentResult) -> list[str]:
    """根据意图识别结果解析路由目标 Agent 列表。

    Args:
        intent_result: 意图识别结果。

    Returns:
        有序的目标 Agent 名称列表。
    """
    primary = intent_result.primary_intent
    if primary is None:
        return ["rag_agent"]  # fallback

    route = ROUTING_TABLE.get(primary.label, ROUTING_TABLE[IntentLabel.UNKNOWN])
    agents = [route["primary"]]
    agents.extend(route["secondary"])
    return agents
