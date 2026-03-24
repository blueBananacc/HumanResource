"""Orchestrator 模块单元测试。

使用 mock 替代 LLM 调用，验证：
- 意图分类解析逻辑
- 路由决策
- 图编译与执行流程
- Node 函数行为
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from human_resource.agents.orchestrator import (
    _build_tool_params,
    classify_intent_node,
    generate_response_node,
    load_context_node,
    post_process_node,
    rag_node,
    register_default_tools,
    route_agents_node,
    tool_node,
)
from human_resource.intent.classifier import IntentClassifier
from human_resource.schemas.models import (
    IntentItem,
    IntentLabel,
    IntentResult,
    RetrievalResult,
    ToolResult,
)


# ── IntentClassifier 解析测试 ────────────────────────────────


class TestIntentClassifierParsing:
    """测试 IntentClassifier._parse_response 解析逻辑。"""

    def setup_method(self):
        with patch("human_resource.intent.classifier.get_llm"):
            self.classifier = IntentClassifier()

    def test_parse_single_intent(self):
        raw = '{"intents": [{"label": "policy_qa", "confidence": 0.95, "entities": {"topic": "\u5e74\u5047"}, "requires_tools": []}]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 1
        assert result.intents[0].label == IntentLabel.POLICY_QA
        assert result.intents[0].confidence == 0.95
        assert result.intents[0].entities["topic"] == "\u5e74\u5047"
        assert result.intents[0].requires_tools == []

    def test_parse_multi_intent(self):
        raw = '{"intents": [{"label": "tool_action", "confidence": 0.9, "entities": {}, "requires_tools": ["get_leave_balance"]}, {"label": "process_inquiry", "confidence": 0.85, "entities": {"process": "\u8bf7\u5047"}, "requires_tools": []}]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 2
        assert result.intents[0].requires_tools == ["get_leave_balance"]
        assert result.requires_tools == ["get_leave_balance"]

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"intents": [{"label": "chitchat", "confidence": 0.9, "entities": {}, "requires_tools": []}]}\n```'
        result = self.classifier._parse_response(raw)
        assert result.intents[0].label == IntentLabel.CHITCHAT

    def test_parse_unknown_label_fallback(self):
        raw = '{"intents": [{"label": "nonexistent", "confidence": 0.5, "entities": {}, "requires_tools": []}]}'
        result = self.classifier._parse_response(raw)
        assert result.intents[0].label == IntentLabel.UNKNOWN

    def test_parse_empty_intents_fallback(self):
        raw = '{"intents": []}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 1
        assert result.intents[0].label == IntentLabel.UNKNOWN

    def test_primary_intent_is_first_not_max_confidence(self):
        """primary_intent 应返回第一个意图（逻辑顺序），非最高置信度。"""
        result = IntentResult(
            intents=[
                IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.85),
                IntentItem(label=IntentLabel.POLICY_QA, confidence=0.95),
            ]
        )
        assert result.primary_intent.label == IntentLabel.EMPLOYEE_LOOKUP

    def test_multi_intent_preserves_order(self):
        """_parse_response 应保留 LLM 返回的意图顺序。"""
        raw = '{"intents": [{"label": "employee_lookup", "confidence": 0.95, "entities": {"name": "张三"}}, {"label": "policy_qa", "confidence": 0.9, "entities": {"topic": "考勤"}}], "requires_tools": ["lookup_employee"]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 2
        assert result.intents[0].label == IntentLabel.EMPLOYEE_LOOKUP
        assert result.intents[1].label == IntentLabel.POLICY_QA
    def test_per_intent_requires_tools(self):
        """每个意图独立声明 requires_tools。"""
        raw = '{"intents": [{"label": "employee_lookup", "confidence": 0.95, "entities": {"name": "张三"}, "requires_tools": ["lookup_employee"]}, {"label": "tool_action", "confidence": 0.9, "entities": {"employee_id": "E001"}, "requires_tools": ["get_leave_balance"]}]}'
        result = self.classifier._parse_response(raw)
        assert result.intents[0].requires_tools == ["lookup_employee"]
        assert result.intents[1].requires_tools == ["get_leave_balance"]
        # 顶层 requires_tools 为并集
        assert result.requires_tools == ["lookup_employee", "get_leave_balance"]

    def test_scenario_single_intent_multi_agent(self):
        """场景二: 单意图涉及多工具。"""
        raw = '{"intents": [{"label": "process_inquiry", "confidence": 0.9, "entities": {"process": "入职"}, "requires_tools": ["list_hr_processes", "get_process_steps"]}]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 1
        assert result.intents[0].label == IntentLabel.PROCESS_INQUIRY
        assert result.intents[0].requires_tools == ["list_hr_processes", "get_process_steps"]
        assert result.requires_tools == ["list_hr_processes", "get_process_steps"]

    def test_scenario_multi_intent_multi_agent(self):
        """场景四: 多意图 + 每个意图多Agent。"""
        raw = '{"intents": [{"label": "tool_action", "confidence": 0.9, "entities": {"name": "李四"}, "requires_tools": ["get_leave_balance"]}, {"label": "process_inquiry", "confidence": 0.85, "entities": {"process": "请假"}, "requires_tools": ["get_process_steps"]}]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 2
        assert result.intents[0].label == IntentLabel.TOOL_ACTION
        assert result.intents[1].label == IntentLabel.PROCESS_INQUIRY
        assert result.requires_tools == ["get_leave_balance", "get_process_steps"]

# ── Node 函数测试 ────────────────────────────────────────────


class TestLoadContextNode:
    def test_returns_default_context(self):
        state = {"session_id": "test123", "messages": []}
        result = load_context_node(state)
        assert "memory_context" in result
        assert result["reflection_count"] == 0
        assert result["current_agent_index"] == 0


class TestClassifyIntentNode:
    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_classifies_message(self, mock_get_cls):
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95,
                               requires_tools=["lookup_employee"])],
        )
        mock_get_cls.return_value = mock_cls

        state = {"messages": [HumanMessage(content="查询张三的部门")]}
        result = classify_intent_node(state)

        assert result["intent"].primary_intent.label == IntentLabel.EMPLOYEE_LOOKUP
        assert result["needs_clarification"] is False
        mock_cls.classify.assert_called_once()

    def test_empty_messages_returns_unknown(self):
        state = {"messages": []}
        result = classify_intent_node(state)
        assert result["intent"].primary_intent.label == IntentLabel.UNKNOWN

    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_separates_session_and_longterm_memory(self, mock_get_cls):
        """验证 classify_intent_node 将会话上下文和长期记忆分离传递。"""
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)],
        )
        mock_get_cls.return_value = mock_cls

        state = {
            "messages": [HumanMessage(content="年假政策")],
            "memory_context": [
                "[历史摘要] 用户之前询问了入职流程",
                "user: 请假怎么办",
                "assistant: 请假需要提交申请",
                "[长期记忆] 用户偏好简洁回答",
                "[长期记忆] 用户是技术部门员工",
            ],
            "user_profile": {"department": "研发部", "role": "工程师"},
        }
        result = classify_intent_node(state)

        call_kwargs = mock_cls.classify.call_args
        # session_summary 不应包含长期记忆
        session_arg = call_kwargs.kwargs.get("session_summary", "")
        assert "[长期记忆]" not in session_arg
        assert "[历史摘要]" in session_arg
        assert "user: 请假怎么办" in session_arg
        # long_term_memory 应包含长期记忆内容
        ltm_arg = call_kwargs.kwargs.get("long_term_memory", "")
        assert "用户偏好简洁回答" in ltm_arg
        assert "用户是技术部门员工" in ltm_arg
        # user_profile 应传入
        profile_arg = call_kwargs.kwargs.get("user_profile", "")
        assert "研发部" in profile_arg
        assert "工程师" in profile_arg

    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_low_confidence_sets_needs_clarification(self, mock_get_cls):
        """低置信度时 needs_clarification 应为 True。"""
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.3)],
        )
        mock_get_cls.return_value = mock_cls

        state = {"messages": [HumanMessage(content="嗯这个那个")]}
        result = classify_intent_node(state)

        assert result["needs_clarification"] is True
        assert result["intent"].primary_intent.label == IntentLabel.UNKNOWN

    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_empty_memory_context(self, mock_get_cls):
        """memory_context 为空时 session_summary 为空字符串。"""
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.CHITCHAT, confidence=0.95)],
        )
        mock_get_cls.return_value = mock_cls

        state = {"messages": [HumanMessage(content="你好")], "memory_context": []}
        result = classify_intent_node(state)

        mock_cls.classify.assert_called_once_with(
            "你好",
            session_summary="",
            long_term_memory="",
            user_profile="",
        )
        assert result["needs_clarification"] is False

    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_only_longterm_memory_no_session(self, mock_get_cls):
        """仅有长期记忆、无会话历史时 session_summary 为空。"""
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.9)],
        )
        mock_get_cls.return_value = mock_cls

        state = {
            "messages": [HumanMessage(content="查张三")],
            "memory_context": [
                "[长期记忆] 张三是研发部的",
            ],
        }
        classify_intent_node(state)

        call_kwargs = mock_cls.classify.call_args.kwargs
        assert call_kwargs["session_summary"] == ""
        assert "张三是研发部的" in call_kwargs["long_term_memory"]

    @patch("human_resource.agents.orchestrator._get_classifier")
    def test_user_profile_without_memory(self, mock_get_cls):
        """仅有 user_profile 时也能传递给分类器。"""
        mock_cls = MagicMock()
        mock_cls.classify.return_value = IntentResult(
            intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)],
        )
        mock_get_cls.return_value = mock_cls

        state = {
            "messages": [HumanMessage(content="年假多少天")],
            "memory_context": [],
            "user_profile": {"department": "财务部"},
        }
        classify_intent_node(state)

        call_kwargs = mock_cls.classify.call_args.kwargs
        assert "财务部" in call_kwargs["user_profile"]
        assert call_kwargs["session_summary"] == ""
        assert call_kwargs["long_term_memory"] == ""


class TestRouteAgentsNode:
    def test_routes_employee_lookup(self):
        """场景一: 单意图 + 单Agent。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["tool_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "tool_agent", "intent_indices": [0]},
        ]

    def test_routes_policy_qa(self):
        """场景一: 单意图 + 单Agent (policy_qa → rag_agent)。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "rag_agent", "intent_indices": [0]},
        ]

    def test_routes_chitchat_to_orchestrator(self):
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.CHITCHAT, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["orchestrator"]

    def test_routes_process_inquiry_serial(self):
        """场景二: 单意图 + 多Agent (process_inquiry → rag_agent + tool_agent 串行)。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.PROCESS_INQUIRY, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent", "tool_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "rag_agent", "intent_indices": [0]},
            {"agent": "tool_agent", "intent_indices": [0]},
        ]

    def test_none_intent_fallback(self):
        state = {}
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "rag_agent", "intent_indices": [0]},
        ]

    def test_low_confidence_fallback_asks_clarification(self):
        """低置信度 + needs_clarification 时 target_agents 为空（直接到 generate_response 追问澄清）。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.3)]
            ),
            "needs_clarification": True,
        }
        result = route_agents_node(state)
        assert result["target_agents"] == []
        assert result["agent_intent_map"] == []

    def test_unknown_intent_normal_confidence_routes_to_rag(self):
        """UNKNOWN 意图但高置信度时正常路由到 rag_agent（Level 2）。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.8)]
            ),
            "needs_clarification": False,
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent"]

    def test_multi_intent_single_agent_each(self):
        """场景三: 多意图 + 每个意图单Agent。
        employee_lookup→tool_agent, policy_qa→rag_agent，各自独立。
        """
        state = {
            "intent": IntentResult(
                intents=[
                    IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95),
                    IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9),
                ]
            ),
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["tool_agent", "rag_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "tool_agent", "intent_indices": [0]},
            {"agent": "rag_agent", "intent_indices": [1]},
        ]

    def test_multi_intent_multi_agent_dedup(self):
        """场景四: 多意图 + 每个意图多Agent。
        employee_lookup→tool_agent, process_inquiry→rag_agent+tool_agent。
        tool_agent 被去重，但 intent_indices 合并为 [0, 1]。
        """
        state = {
            "intent": IntentResult(
                intents=[
                    IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95),
                    IntentItem(label=IntentLabel.PROCESS_INQUIRY, confidence=0.85),
                ]
            ),
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["tool_agent", "rag_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "tool_agent", "intent_indices": [0, 1]},
            {"agent": "rag_agent", "intent_indices": [1]},
        ]

    def test_multi_intent_deduplicates_same_agent(self):
        """多意图涉及完全相同 Agent 时去重（intent_indices 合并）。"""
        state = {
            "intent": IntentResult(
                intents=[
                    IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9),
                    IntentItem(label=IntentLabel.DOCUMENT_SEARCH, confidence=0.85),
                ]
            ),
        }
        result = route_agents_node(state)
        # 两个意图都路由到 rag_agent，去重后单个 agent 但 intent_indices=[0,1]
        assert result["target_agents"] == ["rag_agent"]
        assert result["agent_intent_map"] == [
            {"agent": "rag_agent", "intent_indices": [0, 1]},
        ]

    def test_high_confidence_no_fallback(self):
        """高置信度时不触发 fallback，正常路由。"""
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.TOOL_ACTION, confidence=0.9)]
            ),
            "needs_clarification": False,
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["tool_agent"]


class TestRagNode:
    def test_returns_empty_results(self):
        state = {"messages": [HumanMessage(content="年假政策")]}
        result = rag_node(state)
        assert isinstance(result["rag_results"], RetrievalResult)
        assert result["rag_results"].chunks == []


class TestToolNode:
    def test_no_tools_required(self):
        state = {
            "intent": IntentResult(intents=[]),
            "tool_results": [],
        }
        result = tool_node(state)
        assert result["tool_results"] == []

    def test_executes_lookup_employee(self):
        """场景一: 单意图单工具，使用该意图的 entities。"""
        register_default_tools()
        state = {
            "intent": IntentResult(
                intents=[IntentItem(
                    label=IntentLabel.EMPLOYEE_LOOKUP,
                    confidence=0.95,
                    entities={"name": "张三"},
                    requires_tools=["lookup_employee"],
                )],
            ),
            "tool_results": [],
            "agent_intent_map": [{"agent": "tool_agent", "intent_indices": [0]}],
            "current_agent_index": 0,
        }
        result = tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success is True
        assert result["tool_results"][0].data["name"] == "张三"

    def test_multi_intent_uses_correct_entities(self):
        """场景三: 多意图各有工具，tool_node 根据 agent_intent_map 使用正确的 entities。"""
        register_default_tools()
        state = {
            "intent": IntentResult(
                intents=[
                    IntentItem(
                        label=IntentLabel.EMPLOYEE_LOOKUP,
                        confidence=0.95,
                        entities={"name": "张三"},
                        requires_tools=["lookup_employee"],
                    ),
                    IntentItem(
                        label=IntentLabel.TOOL_ACTION,
                        confidence=0.9,
                        entities={"employee_id": "E001"},
                        requires_tools=["get_leave_balance"],
                    ),
                ],
            ),
            "tool_results": [],
            # tool_agent 服务两个意图
            "agent_intent_map": [{"agent": "tool_agent", "intent_indices": [0, 1]}],
            "current_agent_index": 0,
        }
        result = tool_node(state)
        assert len(result["tool_results"]) == 2
        # 第一个工具使用 intent[0] 的 entities
        assert result["tool_results"][0].success is True
        assert result["tool_results"][0].data["name"] == "张三"
        # 第二个工具使用 intent[1] 的 entities
        assert result["tool_results"][1].success is True

    def test_no_agent_intent_map_fallback(self):
        """无 agent_intent_map 时降级为遍历所有意图的工具。"""
        register_default_tools()
        state = {
            "intent": IntentResult(
                intents=[IntentItem(
                    label=IntentLabel.EMPLOYEE_LOOKUP,
                    confidence=0.95,
                    entities={"name": "张三"},
                    requires_tools=["lookup_employee"],
                )],
            ),
            "tool_results": [],
            # 无 agent_intent_map
        }
        result = tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success is True


class TestBuildToolParams:
    def test_lookup_employee_params(self):
        params = _build_tool_params("lookup_employee", {"name": "张三"})
        assert params == {"query": "张三"}

    def test_get_leave_balance_params(self):
        params = _build_tool_params("get_leave_balance", {"employee_id": "E001"})
        assert params == {"employee_id": "E001"}

    def test_unknown_tool_returns_empty(self):
        params = _build_tool_params("unknown_tool", {})
        assert params == {}


class TestGenerateResponseNode:
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_generates_response(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="这是回复")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="你好")],
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.CHITCHAT, confidence=0.9)]
            ),
            "rag_results": None,
            "tool_results": [],
            "memory_context": [],
        }
        result = generate_response_node(state)
        assert result["final_response"] == "这是回复"
        assert isinstance(result["messages"][0], AIMessage)

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_tool_results_in_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="张三在研发部")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="查询张三的部门")],
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.9)]
            ),
            "rag_results": None,
            "tool_results": [
                ToolResult(success=True, data={"name": "张三", "department": "研发部"}, formatted="{'name': '张三', 'department': '研发部'}")
            ],
            "memory_context": [],
        }
        result = generate_response_node(state)
        assert "张三在研发部" in result["final_response"]

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_needs_clarification_generates_question(self, mock_get_llm):
        """Level 1: needs_clarification=True 时生成澄清问题。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="请问您想了解年假政策还是查询员工信息？")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="那个东西")],
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.3)]
            ),
            "needs_clarification": True,
        }
        result = generate_response_node(state)
        assert result["final_response"] is not None
        assert isinstance(result["messages"][0], AIMessage)

    def test_unknown_no_rag_results_returns_fallback(self):
        """Level 3: UNKNOWN 意图 + RAG 无结果时返回友好提示，不调用 LLM。"""
        state = {
            "messages": [HumanMessage(content="xyz随机内容")],
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.8)]
            ),
            "rag_results": RetrievalResult(chunks=[]),
            "tool_results": [],
            "memory_context": [],
        }
        result = generate_response_node(state)
        assert "无法理解" in result["final_response"]
        assert "联系 HR" in result["final_response"]


class TestPostProcessNode:
    def test_saves_session(self, tmp_path):
        with patch("human_resource.agents.orchestrator._get_session_memory") as mock_sm, \
             patch("human_resource.agents.orchestrator._get_compressor") as mock_comp:
            from human_resource.memory.session import SessionMemory
            sm = SessionMemory(persist_dir=tmp_path)
            mock_sm.return_value = sm
            mock_comp.return_value = MagicMock()

            state = {
                "session_id": "test_session",
                "messages": [HumanMessage(content="你好")],
                "final_response": "你好！",
            }
            post_process_node(state)

            history = sm.get_history("test_session")
            assert len(history) == 2
            assert history[0].role == "user"
            assert history[1].role == "assistant"


# ── 图编译测试 ────────────────────────────────────────────────


class TestGraphCompilation:
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_graph_compiles(self, mock_get_llm):
        """验证图能正常编译，不报错。"""
        from human_resource.agents.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_graph_full_execution(self, mock_get_llm):
        """验证图完整执行流程（mock LLM）。"""
        mock_llm = MagicMock()

        # 意图分类返回 chitchat
        intent_response = MagicMock()
        intent_response.content = '{"intents": [{"label": "chitchat", "confidence": 0.95, "entities": {}, "requires_tools": []}]}'

        # 回复生成
        reply_response = MagicMock()
        reply_response.content = "你好！有什么可以帮助你的吗？"

        mock_llm.invoke.side_effect = [intent_response, reply_response]
        mock_get_llm.return_value = mock_llm

        from human_resource.agents.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()

        result = compiled.invoke({
            "messages": [HumanMessage(content="你好")],
            "session_id": "test",
            "user_id": "u1",
        })

        assert result.get("final_response") is not None
        assert len(result.get("final_response", "")) > 0
