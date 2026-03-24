"""Orchestrator Agent — 各 Node 函数实现。

每个函数对应 LangGraph StateGraph 中的一个 Node。
函数签名：接收 AgentState，返回需要更新的 state 字段字典。

Node 执行流程：
  load_context → memory_retrieval → classify_intent → route_agents
  → dispatch_agent → (rag_node / tool_node / memory_node)
  → generate_response → post_process（含长期记忆写入）
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from human_resource.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    MEMORY_IMPORTANCE_THRESHOLD,
    MEMORY_SEARCH_TOP_K,
    MEMORY_WRITE_INTERVAL,
    MEMORY_WRITE_KEYWORDS,
)
from human_resource.context.compressor import ContextCompressor
from human_resource.context.prompt_builder import PromptBuilder
from human_resource.intent.classifier import IntentClassifier
from human_resource.intent.router import resolve_route
from human_resource.memory.longterm import LongTermMemory
from human_resource.memory.profile import UserProfileStore
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
_compressor: ContextCompressor | None = None
_longterm_memory: LongTermMemory | None = None


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


def _get_compressor() -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor()
    return _compressor


def _get_longterm_memory() -> LongTermMemory:
    global _longterm_memory
    if _longterm_memory is None:
        _longterm_memory = LongTermMemory()
    return _longterm_memory


def _extract_user_message(messages: list) -> str:
    """从消息列表中提取最后一条用户消息文本。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _is_explicit_memory_command(user_message: str) -> bool:
    """检查用户消息是否包含显式记忆写入指令。"""
    lower = user_message.lower().strip()
    return any(kw in lower for kw in MEMORY_WRITE_KEYWORDS)


def _is_turn_interval_trigger(session_id: str) -> bool:
    """检查当前轮次是否达到固定写入间隔。"""
    sm = _get_session_memory()
    history = sm.get_history(session_id)
    # 加上本轮（当前还未写入 session 时调用）
    turn_count = (len(history) // 2) + 1
    return turn_count > 0 and turn_count % MEMORY_WRITE_INTERVAL == 0


def _assess_memory_worthiness(user_msg: str, assistant_msg: str) -> bool:
    """使用 LLM 判断当前对话是否包含值得记忆的信息。"""
    prompt = (
        "请判断以下对话是否包含值得长期记忆的信息。\n"
        "值得记忆的信息包括：用户个人信息（姓名、部门、职位）、"
        "用户偏好、重要决策、关键事实等。\n"
        "闲聊、问候、简单确认等不值得记忆。\n\n"
        f"用户: {user_msg}\n"
        f"助手: {assistant_msg}\n\n"
        "请仅回答 'yes' 或 'no'，不要输出其他内容。"
    )
    try:
        llm = get_llm("memory_extraction")
        response = llm.invoke(prompt)
        answer = str(response.content).strip().lower()
        return answer.startswith("yes")
    except Exception:
        logger.exception("记忆价值评估失败，默认跳过写入")
        return False


def _should_write_memory(
    user_msg: str,
    assistant_msg: str,
    session_id: str,
) -> str | None:
    """三级触发判断，返回触发原因或 None（不写入）。

    优先级：
    1. 用户显式命令 → "explicit_command"
    2. 固定轮次触发 → "turn_interval"
    3. LLM 语义评估 → "llm_assessment"
    """
    # 策略 1: 用户显式命令
    if _is_explicit_memory_command(user_msg):
        logger.info("长期记忆触发: 用户显式命令")
        return "explicit_command"

    # 策略 2: 固定轮次间隔
    if _is_turn_interval_trigger(session_id):
        logger.info("长期记忆触发: 固定轮次间隔 (每 %d 轮)", MEMORY_WRITE_INTERVAL)
        return "turn_interval"

    # 策略 3: LLM 语义评估
    if _assess_memory_worthiness(user_msg, assistant_msg):
        logger.info("长期记忆触发: LLM 评估对话有记忆价值")
        return "llm_assessment"

    logger.info("长期记忆: 未触发写入")
    return None


# ── Node 函数 ────────────────────────────────────────────────
def load_context_node(state: AgentState) -> dict[str, Any]:
    """Node: 加载上下文。

    从 Session Memory 获取会话历史。
    读取存储的摘要 + 当前消息列表。
    """
    session_id = state.get("session_id", "default")
    sm = _get_session_memory()
    history = sm.get_history(session_id)

    memory_snippets: list[str] = []
    if not history:
        return {
            "memory_context": memory_snippets,
            "reflection_count": 0,
            "current_agent_index": 0,
        }

    # 读取 write-time trimming 产生的增量摘要
    stored_summary = sm.get_summary(session_id)
    if stored_summary:
        memory_snippets.append(f"[历史摘要] {stored_summary}")

    # 当前保留的消息（已被 trim 过，数量不会无限增长）
    for msg in history:
        memory_snippets.append(f"{msg.role}: {msg.content}")

    logger.info(
        "加载上下文: session_id=%s, 原始消息数=%d, 摘要数=%d",
        session_id, len(history), len(memory_snippets),
    )

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
    user_message = _extract_user_message(messages)

    if not user_message:
        return {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.0)]
            )
        }

    # 从 memory_context 中分离会话上下文和长期记忆
    session_summary = ""
    long_term_memory = ""
    memory_ctx = state.get("memory_context", [])
    if memory_ctx:
        session_parts = []
        ltm_parts = []
        for item in memory_ctx:
            if item.startswith("[长期记忆]"):
                ltm_parts.append(item.removeprefix("[长期记忆] "))
            else:
                session_parts.append(item)
        if session_parts:
            session_summary = "\n".join(session_parts)
        if ltm_parts:
            long_term_memory = "\n".join(ltm_parts)

    # 提取用户画像
    user_profile_text = ""
    user_profile = state.get("user_profile")
    if user_profile:
        user_profile_text = "\n".join(f"{k}: {v}" for k, v in user_profile.items())

    classifier = _get_classifier()
    intent_result = classifier.classify(
        user_message,
        session_summary=session_summary,
        long_term_memory=long_term_memory,
        user_profile=user_profile_text,
    )

    logger.info(
        "意图识别完成: %s",
        [(i.label.value, i.confidence) for i in intent_result.intents],
    )

    # 低置信度回退：当所有意图的置信度都低于阈值时标记需要澄清
    needs_clarification = False
    if intent_result.intents:
        max_confidence = max(i.confidence for i in intent_result.intents)
        if max_confidence < INTENT_CONFIDENCE_THRESHOLD:
            needs_clarification = True
            logger.info(
                "所有意图置信度 (最高=%.2f) 低于阈值 %.2f，标记需要澄清",
                max_confidence,
                INTENT_CONFIDENCE_THRESHOLD,
            )

    return {"intent": intent_result, "needs_clarification": needs_clarification}


def route_agents_node(state: AgentState) -> dict[str, Any]:
    """Node: 路由决策。

    根据意图分类结果确定目标 Agent 列表。
    低置信度三级回退策略：
    1. 追问澄清 — needs_clarification=True 时直接进入 generate_response
       生成澄清问题，不执行任何 Agent
    2. 用户重试后若 UNKNOWN 意图，由 ROUTING_TABLE 路由到 RAG 做宽泛检索
    3. RAG 也无结果 → generate_response 生成友好的"无法回答"提示
    """
    intent = state.get("intent")
    if intent is None:
        return {
            "target_agents": ["rag_agent"],
            "current_agent_index": 0,
            "agent_intent_map": [{"agent": "rag_agent", "intent_indices": [0]}],
        }

    needs_clarification = state.get("needs_clarification", False)

    # 三级回退 Level 1：低置信度，直接追问澄清，不执行 Agent
    if needs_clarification:
        primary = intent.primary_intent
        logger.info(
            "低置信度回退 Level-1: 主意图=%s, 置信度=%.2f, 追问澄清",
            primary.label.value if primary else "None",
            primary.confidence if primary else 0.0,
        )
        # target_agents 为空 → _dispatch_router 直接到 generate_response
        return {"target_agents": [], "current_agent_index": 0, "agent_intent_map": []}

    target_agents, agent_intent_map = resolve_route(intent)

    logger.info("路由目标: %s", target_agents)
    logger.info("意图映射: %s", agent_intent_map)
    return {
        "target_agents": target_agents,
        "current_agent_index": 0,
        "agent_intent_map": agent_intent_map,
    }


def rag_node(state: AgentState) -> dict[str, Any]:
    """Node: RAG Agent 执行。

    从 HR 文档库检索相关信息。
    MVP 阶段返回空结果占位，待 RAG Pipeline 完整实现后接入。
    """
    messages = state.get("messages", [])
    user_message = _extract_user_message(messages)

    logger.info("RAG Agent 执行查询: %s", user_message[:50])

    # TODO: 接入完整 RAG Pipeline (retriever → reranker)
    # 当前返回空结果，Orchestrator 会在 response 中说明未找到文档
    return {"rag_results": RetrievalResult(chunks=[])}


def tool_node(state: AgentState) -> dict[str, Any]:
    """Node: Tool Agent 执行。

    执行工具调用。根据 agent_intent_map 确定当前 tool_agent 服务的意图，
    仅执行这些意图声明的工具，并使用对应意图的 entities 构建参数。

    支持四种场景：
    - 单意图单工具：直接执行
    - 单意图多工具：依次执行同一意图的所有工具
    - 多意图各有工具：按意图映射执行各自的工具
    - 无映射信息时降级为遍历所有意图的工具（向后兼容）
    """
    from human_resource.tools.executor import execute_tool

    intent = state.get("intent")
    results: list[ToolResult] = list(state.get("tool_results", []))

    if not intent or not intent.requires_tools:
        logger.info("Tool Agent: 无工具调用需求")
        return {"tool_results": results}

    # 确定当前 tool_agent 服务哪些意图
    agent_intent_map = state.get("agent_intent_map", [])
    current_idx = state.get("current_agent_index", 0)

    serving_indices: list[int] | None = None
    if current_idx < len(agent_intent_map):
        entry = agent_intent_map[current_idx]
        if entry.get("agent") == "tool_agent":
            serving_indices = entry.get("intent_indices", [])

    # 收集需要执行的 (tool_name, entities) 对
    tool_tasks: list[tuple[str, dict[str, Any]]] = []
    if serving_indices is not None:
        # 有映射：精确匹配每个意图声明的工具 + 对应 entities
        for idx in serving_indices:
            if idx < len(intent.intents):
                intent_item = intent.intents[idx]
                for tool_name in intent_item.requires_tools:
                    tool_tasks.append((tool_name, intent_item.entities))
    else:
        # 无映射降级：遍历所有意图的工具
        for intent_item in intent.intents:
            for tool_name in intent_item.requires_tools:
                tool_tasks.append((tool_name, intent_item.entities))

    for tool_name, entities in tool_tasks:
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


def memory_retrieval_node(state: AgentState) -> dict[str, Any]:
    """Node: 对话前长期记忆检索。

    在意图识别之前：
    1. 用用户消息检索 mem0 长期记忆（factual + episodic）
    2. 加载用户画像（profile 类型记忆）至 state.user_profile
    """
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    memory_ctx: list[str] = list(state.get("memory_context", []))

    # 提取用户消息
    user_message = _extract_user_message(messages)

    updates: dict[str, Any] = {"memory_context": memory_ctx}

    # ── 加载用户画像 ──
    try:
        ltm = _get_longterm_memory()
        profile_store = UserProfileStore(ltm)
        profile = profile_store.get_profile(user_id)
        if profile:
            updates["user_profile"] = profile
            logger.info("用户画像已加载: user=%s, fields=%d", user_id, len(profile))
    except Exception:
        logger.exception("用户画像加载失败，跳过")

    if not user_message:
        return updates

    # ── 语义检索相关长期记忆 ──
    try:
        ltm = _get_longterm_memory()
        results = ltm.search(user_message, user_id=user_id, top_k=MEMORY_SEARCH_TOP_K)
        for item in results:
            memory_text = item.get("memory", "")
            if memory_text:
                memory_ctx.append(f"[长期记忆] {memory_text}")
        logger.info("长期记忆检索完成: user=%s, 结果数=%d", user_id, len(results))
    except Exception:
        logger.exception("长期记忆检索失败，跳过")

    return updates


def memory_node(state: AgentState) -> dict[str, Any]:
    """Node: Memory Agent 执行。

    当路由决策将 memory_recall 意图分发至此节点时，
    根据用户 query 从 mem0 检索相关长期记忆。
    """
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    memory_ctx: list[str] = list(state.get("memory_context", []))

    user_message = _extract_user_message(messages)

    if not user_message:
        return {"memory_context": memory_ctx}

    try:
        ltm = _get_longterm_memory()
        results = ltm.search(user_message, user_id=user_id, top_k=MEMORY_SEARCH_TOP_K)
        for item in results:
            memory_text = item.get("memory", "")
            if memory_text and f"[长期记忆] {memory_text}" not in memory_ctx:
                memory_ctx.append(f"[长期记忆] {memory_text}")
        logger.info("Memory Agent 检索完成: 结果数=%d", len(results))
    except Exception:
        logger.exception("Memory Agent 检索失败")

    return {"memory_context": memory_ctx}


def generate_response_node(state: AgentState) -> dict[str, Any]:
    """Node: 生成最终回复。

    基于所有 Agent 中间结果，组装 prompt 并调用 LLM 生成用户可见的最终回复。
    仅此节点生成用户可见回复，保证风格一致。
    """
    messages = state.get("messages", [])
    intent = state.get("intent")

    # 获取用户消息
    user_message = _extract_user_message(messages)

    # ── 低置信度回退 Level 1：生成澄清问题 ──
    needs_clarification = state.get("needs_clarification", False)
    if needs_clarification:
        return _generate_clarification(user_message, intent)

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

    # 用户画像
    user_profile = state.get("user_profile")
    profile_text = ""
    if user_profile:
        profile_text = "\n".join(f"{k}: {v}" for k, v in user_profile.items())

    # 对话历史
    history_parts = []
    for msg in messages[:-1]:  # 排除当前消息
        if isinstance(msg, HumanMessage):
            history_parts.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_parts.append(f"助手: {msg.content}")
    conversation_history = "\n".join(history_parts[-6:])  # 最近 3 轮

    # ── 三级回退 Level 3：UNKNOWN 意图 + RAG 无结果 → 友好提示 ──
    if (intent and intent.primary_intent
            and intent.primary_intent.label == IntentLabel.UNKNOWN
            and not rag_text and not tool_text):
        fallback_text = (
            "抱歉，我暂时无法理解您的问题，也未找到相关信息。\n"
            "您可以尝试换个方式描述，或直接联系 HR 部门获取帮助。"
        )
        return {
            "final_response": fallback_text,
            "messages": [AIMessage(content=fallback_text)],
        }

    # 构建 prompt
    prompt = builder.build(
        user_message=user_message,
        user_profile=profile_text,
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
        final_text = str(response.content).strip()
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
        text = str(response.content).strip()
    except Exception:
        text = "你好！我是 HR 智能助手，有什么可以帮你的吗？"

    return {
        "final_response": text,
        "messages": [AIMessage(content=text)],
    }


def _generate_clarification(
    user_message: str, intent: IntentResult | None
) -> dict[str, Any]:
    """低置信度回退 Level-1：生成澄清问题返回给用户。"""
    try:
        llm = get_llm("response_simple")
        intent_hint = ""
        if intent and intent.primary_intent:
            intent_hint = f"（系统猜测可能与 {intent.primary_intent.label.value} 相关，但不确定）"

        prompt = (
            "你是一个 HR 智能助手。用户发送了一条消息，但你无法确定其意图。"
            "请根据用户消息生成一个友好的澄清问题，帮助用户明确需求。\n"
            "要求：\n"
            "1. 简洁友好，不超过两句话\n"
            "2. 给出 2-3 个可能的选项供用户选择\n"
            "3. 不要编造信息\n\n"
            f"用户消息: {user_message}\n"
            f"{intent_hint}"
        )
        response = llm.invoke(prompt)
        text = str(response.content).strip()
    except Exception:
        text = (
            "抱歉，我不太确定您的具体需求。您可以尝试：\n"
            "1. 咨询 HR 政策（如年假、薪资等）\n"
            "2. 查询员工信息\n"
            "3. 了解 HR 流程（如请假、入职等）\n"
            "请更详细地描述您的问题，我会尽力帮助您。"
        )

    return {
        "final_response": text,
        "messages": [AIMessage(content=text)],
    }


def post_process_node(state: AgentState) -> dict[str, Any]:
    """Node: 后处理。

    1. 将当轮对话写入 Session Memory（含 intent/tools 元数据）并持久化。
    2. 超过阈值时执行 trim_and_summarize（write-time 裁剪 + 增量摘要）。
    3. 三级触发机制判断是否需要写入长期记忆。
    """
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "default_user")
    sm = _get_session_memory()

    # 获取本轮的用户消息（取最后一条 HumanMessage）
    messages = state.get("messages", [])
    user_msg = _extract_user_message(messages)

    assistant_msg = state.get("final_response") or ""

    # ── 收集元数据 ──
    intent = state.get("intent")
    intent_label = ""
    if intent and intent.primary_intent:
        intent_label = intent.primary_intent.label.value

    tools_used: list[str] = []
    tool_results = state.get("tool_results", [])
    if tool_results:
        tools_used = [tr.tool_name for tr in tool_results if hasattr(tr, "tool_name")]

    target_agents = state.get("target_agents", [])

    metadata = {
        "intent_label": intent_label,
        "tools_used": tools_used,
        "target_agents": target_agents,
    }

    # ── 三级触发判断（在写入 session 之前，以便 turn count 计算正确）
    trigger_reason: str | None = None
    if user_msg and assistant_msg:
        trigger_reason = _should_write_memory(user_msg, assistant_msg, session_id)

    # 追加到会话（含元数据）并持久化
    if user_msg:
        sm.append(session_id, "user", user_msg, metadata=metadata)
    if assistant_msg:
        sm.append(session_id, "assistant", assistant_msg)
    sm.save(session_id)

    # ── Write-time trim + summarize ──
    compressor = _get_compressor()
    trimmed = sm.trim_and_summarize(
        session_id, summarize_fn=compressor.summarize_text,
    )
    if trimmed:
        sm.save(session_id)
        logger.info("会话裁剪完成: session=%s", session_id)

    logger.info("会话已持久化: session=%s", session_id)

    # ── 触发长期记忆写入 ──
    if trigger_reason is not None:
        _write_longterm_memory(user_id, session_id, user_msg, assistant_msg)

    return {}


def _write_longterm_memory(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """使用 LLM 提取值得记住的信息，按类型分类写入 mem0。

    提取流程：
    1. LLM 分析对话，输出 [{type, content, importance}]
    2. 过滤 importance > 阈值的记忆
    3. 按 type (profile/episodic/factual) 分类写入 mem0
    """
    extracted = _extract_memorable_info(user_msg, assistant_msg)

    if not extracted:
        logger.info("LLM 未提取到值得记忆的信息，跳过写入")
        return

    ltm = _get_longterm_memory()
    for item in extracted:
        importance = item.get("importance", 0.0)
        if importance <= MEMORY_IMPORTANCE_THRESHOLD:
            continue

        memory_type = item.get("type", "factual")
        content = item.get("content", "")
        if not content:
            continue

        try:
            conversation = [
                {"role": "user", "content": content},
            ]
            metadata = {"session_id": session_id}
            ltm.add(
                conversation,
                user_id=user_id,
                memory_type=memory_type,
                metadata=metadata,
            )
            logger.info(
                "长期记忆写入: type=%s, importance=%.2f, user=%s",
                memory_type, importance, user_id,
            )
        except Exception:
            logger.exception("长期记忆写入失败 (type=%s)", memory_type)


def _extract_memorable_info(
    user_msg: str,
    assistant_msg: str,
) -> list[dict[str, Any]]:
    """使用 LLM 从对话中提取值得长期记忆的信息。

    Returns:
        提取结果列表，每项包含 type/content/importance。
    """
    import json

    prompt = (
        "请分析以下对话，提取值得长期记忆的信息。\n\n"
        "记忆类型说明：\n"
        '- "profile": 用户个人信息（姓名、部门、职位、角色、偏好等）\n'
        '- "episodic": 会话中的关键事件或决策（如确认了某个流程、做出了某个选择）\n'
        '- "factual": 用户明确告知的事实（如直属上级、常用的系统等）\n\n'
        "请严格以 JSON 数组格式输出，不要包含其他文字：\n"
        '[{"type": "profile|episodic|factual", "content": "提取的信息", "importance": 0.0-1.0}]\n\n'
        "如果对话中没有值得记忆的信息（如闲聊、问候），请返回空数组 []\n\n"
        f"用户: {user_msg}\n"
        f"助手: {assistant_msg}"
    )

    try:
        llm = get_llm("memory_extraction")
        response = llm.invoke(prompt)
        text = str(response.content).strip()

        # 处理 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]).strip()

        result = json.loads(text)
        if not isinstance(result, list):
            return []
        return result
    except Exception:
        logger.exception("LLM 记忆提取失败")
        return []


def finalize_session(session_id: str, user_id: str) -> None:
    """会话结束时生成 episodic memory 并写入长期记忆。

    将整个会话历史摘要为 episodic 记忆，包含关键问题和决策。
    """
    sm = _get_session_memory()
    session = sm.get_or_create(session_id)

    # 构建完整对话文本（存储摘要 + 当前消息）
    parts: list[str] = []
    if session.summary:
        parts.append(f"早期对话摘要: {session.summary}")
    for msg in session.messages:
        parts.append(f"{msg.role}: {msg.content}")

    full_text = "\n".join(parts)
    if not full_text.strip():
        logger.info("会话为空，跳过 episodic 记忆写入: session=%s", session_id)
        return

    # 生成会话级 episodic 摘要
    compressor = _get_compressor()
    episodic_summary = compressor.summarize_text(full_text)

    if not episodic_summary:
        return

    # 写入 mem0 作为 episodic memory
    try:
        ltm = _get_longterm_memory()
        ltm.add(
            [{"role": "user", "content": episodic_summary}],
            user_id=user_id,
            memory_type="episodic",
            metadata={"session_id": session_id, "type": "session_summary"},
        )
        logger.info(
            "会话结束 episodic 记忆已写入: session=%s, user=%s",
            session_id, user_id,
        )
    except Exception:
        logger.exception("会话结束 episodic 记忆写入失败")


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
