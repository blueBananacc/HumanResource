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
        raw = '{"intents": [{"label": "policy_qa", "confidence": 0.95, "entities": {"topic": "年假"}}], "requires_tools": []}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 1
        assert result.intents[0].label == IntentLabel.POLICY_QA
        assert result.intents[0].confidence == 0.95
        assert result.intents[0].entities["topic"] == "年假"

    def test_parse_multi_intent(self):
        raw = '{"intents": [{"label": "tool_action", "confidence": 0.9, "entities": {}}, {"label": "process_inquiry", "confidence": 0.85, "entities": {"process": "请假"}}], "requires_tools": ["get_leave_balance"]}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 2
        assert result.requires_tools == ["get_leave_balance"]

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"intents": [{"label": "chitchat", "confidence": 0.9, "entities": {}}], "requires_tools": []}\n```'
        result = self.classifier._parse_response(raw)
        assert result.intents[0].label == IntentLabel.CHITCHAT

    def test_parse_unknown_label_fallback(self):
        raw = '{"intents": [{"label": "nonexistent", "confidence": 0.5, "entities": {}}], "requires_tools": []}'
        result = self.classifier._parse_response(raw)
        assert result.intents[0].label == IntentLabel.UNKNOWN

    def test_parse_empty_intents_fallback(self):
        raw = '{"intents": [], "requires_tools": []}'
        result = self.classifier._parse_response(raw)
        assert len(result.intents) == 1
        assert result.intents[0].label == IntentLabel.UNKNOWN


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
            intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95)],
            requires_tools=["lookup_employee"],
        )
        mock_get_cls.return_value = mock_cls

        state = {"messages": [HumanMessage(content="查询张三的部门")]}
        result = classify_intent_node(state)

        assert result["intent"].primary_intent.label == IntentLabel.EMPLOYEE_LOOKUP
        mock_cls.classify.assert_called_once()

    def test_empty_messages_returns_unknown(self):
        state = {"messages": []}
        result = classify_intent_node(state)
        assert result["intent"].primary_intent.label == IntentLabel.UNKNOWN


class TestRouteAgentsNode:
    def test_routes_employee_lookup(self):
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["tool_agent"]

    def test_routes_policy_qa(self):
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent", "memory_agent"]

    def test_routes_chitchat_to_orchestrator(self):
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.CHITCHAT, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["orchestrator"]

    def test_routes_process_inquiry_serial(self):
        state = {
            "intent": IntentResult(
                intents=[IntentItem(label=IntentLabel.PROCESS_INQUIRY, confidence=0.9)]
            )
        }
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent", "tool_agent"]

    def test_none_intent_fallback(self):
        state = {}
        result = route_agents_node(state)
        assert result["target_agents"] == ["rag_agent"]


class TestRagNode:
    def test_returns_empty_results(self):
        state = {"messages": [HumanMessage(content="年假政策")]}
        result = rag_node(state)
        assert isinstance(result["rag_results"], RetrievalResult)
        assert result["rag_results"].chunks == []


class TestToolNode:
    def test_no_tools_required(self):
        state = {
            "intent": IntentResult(intents=[], requires_tools=[]),
            "tool_results": [],
        }
        result = tool_node(state)
        assert result["tool_results"] == []

    def test_executes_lookup_employee(self):
        register_default_tools()
        state = {
            "intent": IntentResult(
                intents=[IntentItem(
                    label=IntentLabel.EMPLOYEE_LOOKUP,
                    confidence=0.95,
                    entities={"name": "张三"},
                )],
                requires_tools=["lookup_employee"],
            ),
            "tool_results": [],
        }
        result = tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success is True
        assert result["tool_results"][0].data["name"] == "张三"


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
        intent_response.content = '{"intents": [{"label": "chitchat", "confidence": 0.95, "entities": {}}], "requires_tools": []}'

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
