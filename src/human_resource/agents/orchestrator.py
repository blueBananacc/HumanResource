"""Orchestrator Agent — 各 Node 函数实现。

每个函数对应 LangGraph StateGraph 中的一个 Node。
函数签名：接收 AgentState，返回需要更新的 state 字段字典。

Node 执行流程：
  load_context → classify_intent → route_agents → dispatch_agent
  → (rag_node / tool_node / memory_node) → generate_response → post_process
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from human_resource.config import INTENT_CONFIDENCE_THRESHOLD
from human_resource.context.prompt_builder import PromptBuilder
from human_resource.intent.classifier import IntentClassifier
from human_resource.intent.router import resolve_route
from human_resource.memory.session import SessionMemory
from human_resource.schemas.models import (
    IntentItem,
    IntentLabel,
    IntentResult,
    RetrievalResult,
    ToolResult,
)
from human_resource.schemas.state import AgentState
from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

# ── 模块级单例（按需初始化） ─────────────────────────────────
_classifier: IntentClassifier | None = None
_session_memory: SessionMemory | None = None
_prompt_builder: PromptBuilder | None = None


def _get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def _get_session_memory() -> SessionMemory:
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory


def _get_prompt_builder() -> PromptBuilder:
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


# ── Node 函数 ────────────────────────────────────────────────
def load_context_node(state: AgentState) -> dict[str, Any]:
    """Node: 加载上下文。

    从 Session Memory 获取会话历史
    """
    session_id = state.get("session_id", "default")
    sm = _get_session_memory()
    history = sm.get_history(session_id)

    # 将 session history 格式化为 memory_context
    memory_snippets: list[str] = []
    if history:
        recent = history[-6:]  # 最近 3 轮
        for msg in recent:
            memory_snippets.append(f"{msg.role}: {msg.content}")

    return {
        "memory_context": memory_snippets,
        "reflection_count": 0,
        "current_agent_index": 0,
    }


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    """Node: 意图识别。

    使用 IntentClassifier 对用户消息进行分类。
    """
    messages = state.get("messages", [])
    if not messages:
        return {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.0)]
            )
        }

    # 获取最后一条用户消息
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.0)]
            )
        }

    # 获取会话摘要作为上下文
    session_summary = ""
    memory_ctx = state.get("memory_context", [])
    if memory_ctx:
        session_summary = "\n".join(memory_ctx[-3:])

    classifier = _get_classifier()
    intent_result = classifier.classify(user_message, session_summary)

    logger.info(
        "意图识别完成: %s",
        [(i.label.value, i.confidence) for i in intent_result.intents],
    )

    return {"intent": intent_result}


def route_agents_node(state: AgentState) -> dict[str, Any]:
    """Node: 路由决策。

    根据意图分类结果确定目标 Agent 列表。
    """
    intent = state.get("intent")
    if intent is None:
        return {"target_agents": ["rag_agent"]}

    target_agents = resolve_route(intent)

    # 低置信度时追加提示
    primary = intent.primary_intent
    if primary and primary.confidence < INTENT_CONFIDENCE_THRESHOLD:
        logger.info("低置信度意图，将使用 fallback 策略")

    logger.info("路由目标: %s", target_agents)
    return {"target_agents": target_agents, "current_agent_index": 0}


def rag_node(state: AgentState) -> dict[str, Any]:
    """Node: RAG Agent 执行。

    从 HR 文档库检索相关信息。
    MVP 阶段返回空结果占位，待 RAG Pipeline 完整实现后接入。
    """
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    logger.info("RAG Agent 执行查询: %s", user_message[:50])

    # TODO: 接入完整 RAG Pipeline (retriever → reranker)
    # 当前返回空结果，Orchestrator 会在 response 中说明未找到文档
    return {"rag_results": RetrievalResult(chunks=[])}


def tool_node(state: AgentState) -> dict[str, Any]:
    """Node: Tool Agent 执行。

    执行工具调用。
    MVP 阶段通过 intent.requires_tools 和 entities 调用工具。
    """
    from human_resource.tools.executor import execute_tool

    intent = state.get("intent")
    results: list[ToolResult] = list(state.get("tool_results", []))

    if not intent or not intent.requires_tools:
        logger.info("Tool Agent: 无工具调用需求")
        return {"tool_results": results}

    # 从意图实体中提取参数
    primary = intent.primary_intent
    entities = primary.entities if primary else {}

    for tool_name in intent.requires_tools:
        # 根据工具名构建参数
        params = _build_tool_params(tool_name, entities)
        logger.info("执行工具: %s, 参数: %s", tool_name, params)

        result = execute_tool(tool_name, params)
        results.append(result)

        if not result.success:
            logger.warning("工具 %s 执行失败: %s", tool_name, result.error)

    return {"tool_results": results}


def _build_tool_params(tool_name: str, entities: dict[str, Any]) -> dict[str, Any]:
    """根据工具名和意图实体构建工具参数。"""
    if tool_name == "lookup_employee":
        query = entities.get("name", entities.get("employee_id", ""))
        return {"query": str(query)}
    elif tool_name == "get_leave_balance":
        return {"employee_id": str(entities.get("employee_id", ""))}
    elif tool_name == "list_hr_processes":
        return {}
    elif tool_name == "get_process_steps":
        return {"process_id": str(entities.get("process", ""))}
    return {}


def memory_node(state: AgentState) -> dict[str, Any]:
    """Node: Memory Agent 执行。

    检索长期记忆。MVP 阶段返回已有的 memory_context。
    """
    logger.info("Memory Agent 执行检索")
    # TODO: 接入 mem0 Cloud 检索
    return {"memory_context": state.get("memory_context", [])}


def generate_response_node(state: AgentState) -> dict[str, Any]:
    """Node: 生成最终回复。

    基于所有 Agent 中间结果，组装 prompt 并调用 LLM 生成用户可见的最终回复。
    仅此节点生成用户可见回复，保证风格一致。
    """
    messages = state.get("messages", [])
    intent = state.get("intent")

    # 获取用户消息
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # ── chitchat 直接响应 ──
    if intent and intent.primary_intent:
        label = intent.primary_intent.label
        if label == IntentLabel.CHITCHAT:
            return _handle_chitchat(user_message)

    # ── 组装上下文 ──
    builder = _get_prompt_builder()

    # RAG 结果
    rag_text = ""
    rag_results = state.get("rag_results")
    if rag_results and rag_results.chunks:
        rag_text = "\n\n".join(
            f"[来源: {c.metadata.get('source', '未知')}] {c.text}"
            for c in rag_results.chunks
        )

    # 工具结果
    tool_text = ""
    tool_results = state.get("tool_results", [])
    if tool_results:
        parts = []
        for tr in tool_results:
            if tr.success:
                parts.append(tr.formatted)
            else:
                parts.append(f"工具调用失败: {tr.error}")
        tool_text = "\n".join(parts)

    # 记忆上下文
    memory_text = "\n".join(state.get("memory_context", []))

    # 对话历史
    history_parts = []
    for msg in messages[:-1]:  # 排除当前消息
        if isinstance(msg, HumanMessage):
            history_parts.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_parts.append(f"助手: {msg.content}")
    conversation_history = "\n".join(history_parts[-6:])  # 最近 3 轮

    # 构建 prompt
    prompt = builder.build(
        user_message=user_message,
        retrieved_context=rag_text,
        tool_results=tool_text,
        relevant_memories=memory_text,
        conversation_history=conversation_history,
    )

    # 选择模型：复杂推理用 reasoner，简单问题用 chat
    purpose = "response_simple"
    if intent and intent.primary_intent:
        label = intent.primary_intent.label
        if label in (IntentLabel.POLICY_QA, IntentLabel.PROCESS_INQUIRY):
            # 如果有检索结果需要综合推理，使用更强模型
            if rag_text or tool_text:
                purpose = "response_complex"

    try:
        llm = get_llm(purpose)
        response = llm.invoke(prompt)
        final_text = response.content.strip()
    except Exception:
        logger.exception("LLM 生成回复失败")
        final_text = "抱歉，系统暂时处理出错，请稍后再试或联系 HR 部门。"

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


def _handle_chitchat(user_message: str) -> dict[str, Any]:
    """处理闲聊意图，轻量直接响应。"""
    try:
        llm = get_llm("response_simple")
        prompt = (
            "你是一个友好的 HR 助手。请简短回复以下闲聊，"
            "并可以引导用户使用 HR 相关服务。\n\n"
            f"用户: {user_message}"
        )
        response = llm.invoke(prompt)
        text = response.content.strip()
    except Exception:
        text = "你好！我是 HR 智能助手，有什么可以帮你的吗？"

    return {
        "final_response": text,
        "messages": [AIMessage(content=text)],
    }


def post_process_node(state: AgentState) -> dict[str, Any]:
    """Node: 后处理。

    1. 将当轮对话写入 Session Memory 并持久化。
    2. 判断是否需要写入长期记忆（MVP 阶段简化）。
    """
    session_id = state.get("session_id", "default")
    sm = _get_session_memory()

    # 获取本轮的用户消息和助手回复
    messages = state.get("messages", [])
    user_msg = ""
    assistant_msg = state.get("final_response", "")

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_msg = msg.content

    # 追加到会话并持久化
    if user_msg:
        sm.append(session_id, "user", user_msg)
    if assistant_msg:
        sm.append(session_id, "assistant", assistant_msg)
    sm.save(session_id)

    logger.info("会话已持久化: session=%s", session_id)

    # TODO: 触发长期记忆写入（判断是否包含值得记忆的信息）
    return {}


# ── 工具注册初始化 ───────────────────────────────────────────
def register_default_tools() -> None:
    """将内置 HR 工具注册到全局 Tool Registry。"""
    from human_resource.tools.hr_tools.employee_lookup import (
        get_leave_balance,
        lookup_employee,
    )
    from human_resource.tools.hr_tools.process_tools import (
        get_process_steps,
        list_hr_processes,
    )
    from human_resource.tools.registry import registry

    registry.register(lookup_employee, category="employee", source="internal")
    registry.register(get_leave_balance, category="employee", source="internal")
    registry.register(list_hr_processes, category="process", source="internal")
    registry.register(get_process_steps, category="process", source="internal")
