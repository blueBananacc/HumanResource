"""Orchestrator 模块单元测试。

使用 mock 替代 LLM 调用，验证：
- IntentAnalyzer 意图提示生成
- Orchestrator 决策中心
- 图编译与执行流程
- Node 函数行为（新架构：Orchestrator 驱动决策循环）
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from human_resource.agents.orchestrator import (
    _parse_decision,
    generate_response_node,
    intent_hints_node,
    load_context_node,
    orchestrator_decision_node,
    post_process_node,
    rag_node,
    register_default_tools,
    tool_node,
)
from human_resource.schemas.models import (
    RetrievalResult,
    ToolResult,
)
from human_resource.tools.selector import ToolCallRequest


# ── IntentAnalyzer 测试 ──────────────────────────────────────


class TestIntentHintsNode:
    """测试 intent_hints_node 意图提示生成逻辑。"""

    @patch("human_resource.agents.orchestrator._get_intent_analyzer")
    def test_generates_hints(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "意图为：policy_qa。理由：用户想了解年假政策。"
        mock_get_analyzer.return_value = mock_analyzer

        state = {
            "messages": [HumanMessage(content="年假政策是什么")],
            "session_context": [],
            "memory_context": [],
        }
        result = intent_hints_node(state)
        assert result["intent_hints"] is not None
        assert "policy_qa" in result["intent_hints"]
        mock_analyzer.analyze.assert_called_once()

    def test_empty_messages_returns_none(self):
        state = {"messages": []}
        result = intent_hints_node(state)
        assert result["intent_hints"] is None

    @patch("human_resource.agents.orchestrator._get_intent_analyzer")
    def test_passes_context_to_analyzer(self, mock_get_analyzer):
        """验证 intent_hints_node 传递会话上下文和记忆给分析器。"""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "意图为：employee_lookup"
        mock_get_analyzer.return_value = mock_analyzer

        state = {
            "messages": [HumanMessage(content="查张三")],
            "session_context": ["user: 上次查了李四"],
            "memory_context": ["张三是研发部的"],
            "user_profile": {"department": "研发部"},
        }
        intent_hints_node(state)

        call_args = mock_analyzer.analyze.call_args
        # 第一个位置参数是 user_message
        assert "查张三" in call_args.args[0]
        # 关键字参数包含上下文
        assert "上次查了李四" in call_args.kwargs.get("session_summary", "")
        assert "张三是研发部的" in call_args.kwargs.get("long_term_memory", "")
        assert "研发部" in str(call_args.kwargs.get("user_profile", ""))


# ── Orchestrator 决策解析测试 ────────────────────────────────


class TestParseDecision:
    """测试 _parse_decision JSON 解析逻辑。"""

    def test_parse_valid_json(self):
        raw = '{"reasoning": "用户询问年假", "action": "rag", "action_input": {"query": "年假政策"}}'
        result = _parse_decision(raw)
        assert result["action"] == "rag"
        assert result["action_input"]["query"] == "年假政策"

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"reasoning": "闲聊", "action": "answer", "action_input": {}}\n```'
        result = _parse_decision(raw)
        assert result["action"] == "answer"

    def test_parse_unknown_action_fallback(self):
        raw = '{"reasoning": "test", "action": "unknown_action", "action_input": {}}'
        result = _parse_decision(raw)
        assert result["action"] == "answer"

    def test_parse_invalid_json_fallback(self):
        raw = "这不是JSON"
        result = _parse_decision(raw)
        assert result["action"] == "answer"

    def test_parse_all_valid_actions(self):
        for action in ("rag", "tool", "memory", "answer", "clarify"):
            raw = f'{{"reasoning": "test", "action": "{action}", "action_input": {{}}}}'
            result = _parse_decision(raw)
            assert result["action"] == action


# ── Orchestrator 决策节点测试 ────────────────────────────────


class TestOrchestratorDecisionNode:
    """测试 orchestrator_decision_node 决策循环逻辑。"""

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_decides_rag_action(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "用户问年假政策，需要检索文档", "action": "rag", "action_input": {"query": "年假政策"}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="年假政策是什么")],
            "intent_hints": "意图为：policy_qa",
            "loop_count": 0,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "rag"
        assert result["loop_count"] == 1

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_decides_tool_action(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "需要查员工信息", "action": "tool", "action_input": {"query": "查询张三的部门"}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="查询张三的部门")],
            "intent_hints": "意图为：employee_lookup",
            "loop_count": 0,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "tool"

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_decides_answer_with_existing_results(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "工具结果已充足", "action": "answer", "action_input": {}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="查张三")],
            "intent_hints": "employee_lookup",
            "tool_results": [
                ToolResult(success=True, data={"name": "张三"}, formatted="name=张三")
            ],
            "loop_count": 1,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "answer"

    def test_max_loops_forces_answer(self):
        """达到最大循环次数时强制生成回答。"""
        state = {
            "messages": [HumanMessage(content="测试")],
            "loop_count": 5,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "answer"
        assert "最大循环" in result["orchestrator_reasoning"]

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_llm_failure_defaults_to_answer(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="测试")],
            "loop_count": 0,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "answer"

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_decides_clarify(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "信息不足", "action": "clarify", "action_input": {}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="那个东西")],
            "intent_hints": "意图不明确",
            "loop_count": 0,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "clarify"

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_decides_memory(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "用户想回忆之前聊过的", "action": "memory", "action_input": {"query": "之前的对话"}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="我们之前聊了什么")],
            "intent_hints": "memory_recall",
            "loop_count": 0,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["orchestrator_action"] == "memory"

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_loop_count_increments(self, mock_get_llm):
        """每次决策后 loop_count 应递增。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"reasoning": "test", "action": "rag", "action_input": {"query": "test"}}'
        )
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="测试")],
            "loop_count": 2,
            "max_loops": 5,
        }
        result = orchestrator_decision_node(state)
        assert result["loop_count"] == 3


# ── Node 函数测试 ────────────────────────────────────────────


class TestLoadContextNode:
    def test_returns_default_context(self):
        state = {"session_id": "test123", "messages": []}
        result = load_context_node(state)
        assert "session_context" in result
        assert "memory_context" in result
        assert result["memory_context"] == []
        assert result["loop_count"] == 0
        assert result["max_loops"] == 5


class TestRagNode:
    @patch("human_resource.rag.retriever.hybrid_search")
    def test_returns_empty_results(self, mock_hs):
        mock_hs.return_value = RetrievalResult(chunks=[])
        state = {
            "messages": [HumanMessage(content="年假政策")],
            "orchestrator_action_input": {"query": "年假政策"},
        }
        result = rag_node(state)
        assert isinstance(result["rag_results"], RetrievalResult)
        assert result["rag_results"].chunks == []

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_uses_orchestrator_query(self, mock_hs):
        """RAG Node 应使用 Orchestrator 提供的 query。"""
        mock_hs.return_value = RetrievalResult(chunks=[])
        state = {
            "messages": [HumanMessage(content="原始消息")],
            "orchestrator_action_input": {"query": "年假政策文档"},
        }
        rag_node(state)
        call_args = mock_hs.call_args
        assert call_args[0][0] == "年假政策文档"

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_process_inquiry_uses_sop_collection(self, mock_hs):
        """intent_hints 包含 process_inquiry 时使用 sop_collection。"""
        mock_hs.return_value = RetrievalResult(chunks=[])
        state = {
            "messages": [HumanMessage(content="请假流程")],
            "orchestrator_action_input": {"query": "请假流程"},
            "intent_hints": "意图为：process_inquiry。理由：用户想了解请假流程。",
        }
        rag_node(state)
        call_kwargs = mock_hs.call_args.kwargs
        assert call_kwargs.get("collection_name") == "sop_collection"


class TestToolNode:
    def test_no_query_returns_empty(self):
        state = {
            "messages": [],
            "tool_results": [],
        }
        result = tool_node(state)
        assert result["tool_results"] == []

    @patch("human_resource.agents.orchestrator._get_tool_selector")
    def test_executes_lookup_employee(self, mock_get_selector):
        """ToolSelector 选择工具并生成参数。"""
        register_default_tools()
        mock_selector = MagicMock()
        mock_selector.select.return_value = [
            ToolCallRequest(tool_name="lookup_employee", parameters={"query": "张三"}),
        ]
        mock_get_selector.return_value = mock_selector

        state = {
            "messages": [HumanMessage(content="查询张三的部门")],
            "orchestrator_action_input": {"query": "查询张三的部门"},
            "tool_results": [],
        }
        result = tool_node(state)
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].success is True
        assert result["tool_results"][0].data["name"] == "张三"

    @patch("human_resource.agents.orchestrator._get_tool_selector")
    def test_selector_returns_empty(self, mock_get_selector):
        """ToolSelector 返回空列表时，不执行任何工具。"""
        register_default_tools()
        mock_selector = MagicMock()
        mock_selector.select.return_value = []
        mock_get_selector.return_value = mock_selector

        state = {
            "messages": [HumanMessage(content="你好")],
            "orchestrator_action_input": {"query": "你好"},
            "tool_results": [],
        }
        result = tool_node(state)
        assert result["tool_results"] == []


class TestGenerateResponseNode:
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_generates_response(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="这是回复")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="你好")],
            "orchestrator_action": "answer",
            "rag_results": None,
            "tool_results": [],
            "session_context": [],
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
            "orchestrator_action": "answer",
            "rag_results": None,
            "tool_results": [
                ToolResult(success=True, data={"name": "张三", "department": "研发部"}, formatted="{'name': '张三', 'department': '研发部'}")
            ],
            "session_context": [],
            "memory_context": [],
        }
        result = generate_response_node(state)
        assert "张三在研发部" in result["final_response"]

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_clarify_action_generates_question(self, mock_get_llm):
        """action=clarify 时生成澄清问题。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="请问您想了解年假政策还是查询员工信息？")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="那个东西")],
            "orchestrator_action": "clarify",
        }
        result = generate_response_node(state)
        assert result["final_response"] is not None
        assert isinstance(result["messages"][0], AIMessage)

    def test_unknown_no_context_returns_fallback(self):
        """无任何上下文 + unknown 意图提示时返回友好提示。"""
        state = {
            "messages": [HumanMessage(content="xyz随机内容")],
            "orchestrator_action": "answer",
            "intent_hints": "意图为：unknown。无法识别为HR相关类别。",
            "rag_results": RetrievalResult(chunks=[]),
            "tool_results": [],
            "session_context": [],
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
    def test_graph_has_decision_loop(self, mock_get_llm):
        """验证图包含决策循环结构。"""
        from human_resource.agents.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()
        node_names = list(compiled.get_graph().nodes.keys())
        assert "orchestrator_decision" in node_names
        assert "intent_hints" in node_names
        assert "rag_node" in node_names
        assert "tool_node" in node_names
        assert "memory_node" in node_names

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_graph_edges_form_loop(self, mock_get_llm):
        """验证 rag/tool/memory 执行后会回到 orchestrator_decision。"""
        from human_resource.agents.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()
        edges = [(e.source, e.target) for e in compiled.get_graph().edges]
        assert ("rag_node", "orchestrator_decision") in edges
        assert ("tool_node", "orchestrator_decision") in edges
        assert ("memory_node", "orchestrator_decision") in edges

    @patch("human_resource.agents.orchestrator._get_intent_analyzer")
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_graph_full_execution(self, mock_get_llm, mock_get_analyzer):
        """验证图完整执行流程（mock LLM）。"""
        # 设置 IntentAnalyzer mock
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "意图为：chitchat。理由：用户在打招呼。"
        mock_get_analyzer.return_value = mock_analyzer

        mock_llm = MagicMock()

        # Orchestrator 决策：直接 answer
        decision_response = MagicMock()
        decision_response.content = '{"reasoning": "闲聊直接回答", "action": "answer", "action_input": {}}'

        # 回复生成
        reply_response = MagicMock()
        reply_response.content = "你好！有什么可以帮助你的吗？"

        mock_llm.invoke.side_effect = [decision_response, reply_response]
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

    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_intent_analyzer")
    @patch("human_resource.agents.orchestrator.get_llm")
    @patch("human_resource.rag.retriever.hybrid_search")
    def test_graph_rag_loop_execution(self, mock_hs, mock_get_llm, mock_get_analyzer, mock_should):
        """验证 Orchestrator → RAG → Orchestrator → answer 循环。"""
        from human_resource.schemas.models import RetrievedChunk

        mock_should.return_value = None  # 不触发记忆写入

        # 设置 IntentAnalyzer mock
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "意图为：policy_qa。理由：用户想了解年假。"
        mock_get_analyzer.return_value = mock_analyzer

        mock_llm = MagicMock()

        # 第1轮决策：需要 RAG
        decision1 = MagicMock()
        decision1.content = '{"reasoning": "需要检索年假政策", "action": "rag", "action_input": {"query": "年假政策"}}'

        # RAG query rewrite (rag_node 内部调用)
        rewrite_response = MagicMock()
        rewrite_response.content = "年假政策"

        # 第2轮决策：已有信息，answer
        decision2 = MagicMock()
        decision2.content = '{"reasoning": "RAG 返回了年假信息，可以回答", "action": "answer", "action_input": {}}'

        # 回复生成
        reply_response = MagicMock()
        reply_response.content = "年假政策是每年15天。"

        mock_llm.invoke.side_effect = [decision1, rewrite_response, decision2, reply_response]
        mock_get_llm.return_value = mock_llm

        mock_hs.return_value = RetrievalResult(chunks=[
            RetrievedChunk(text="年假规定：每年15天", score=0.9, metadata={"source": "handbook"}),
        ])

        from human_resource.agents.graph import build_graph
        graph = build_graph()
        compiled = graph.compile()

        result = compiled.invoke({
            "messages": [HumanMessage(content="年假政策是什么")],
            "session_id": "test_rag",
            "user_id": "u1",
        })

        assert result.get("final_response") == "年假政策是每年15天。"
