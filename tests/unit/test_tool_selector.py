"""ToolSelector 单元测试。

使用 mock 替代 LLM 调用，验证：
- Native Function Calling 工具选择与参数生成
- 无工具调用返回（LLM 不选择工具）
- 未注册工具过滤
- 无候选工具处理
- 空消息处理
- LLM 异常容错
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import tool

from human_resource.tools.selector import ToolCallRequest, ToolSelector


# ── 辅助工具注册 ─────────────────────────────────────────────

def _register_test_tools():
    """注册测试用工具到 registry。"""
    from human_resource.tools.registry import registry

    @tool
    def lookup_employee(query: str) -> dict:
        """根据姓名或工号查询员工信息。"""
        return {"name": query}

    @tool
    def get_leave_balance(employee_id: str) -> dict:
        """查询员工假期余额。"""
        return {"annual": 10}

    if not registry.has("lookup_employee"):
        registry.register(lookup_employee, category="employee")
    if not registry.has("get_leave_balance"):
        registry.register(get_leave_balance, category="employee")


# ── ToolSelector 测试 ────────────────────────────────────────


class TestToolSelector:
    @patch("human_resource.tools.selector.get_llm")
    def test_select_single_tool(self, mock_get_llm):
        """LLM 通过 Native FC 选择单个工具。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            tool_calls=[{"name": "lookup_employee", "args": {"query": "张三"}, "id": "call_1"}],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查询张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].tool_name == "lookup_employee"
        assert result[0].parameters == {"query": "张三"}

    @patch("human_resource.tools.selector.get_llm")
    def test_select_multiple_tools(self, mock_get_llm):
        """LLM 通过 Native FC 同时选择多个工具。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            tool_calls=[
                {"name": "lookup_employee", "args": {"query": "张三"}, "id": "call_1"},
                {"name": "get_leave_balance", "args": {"employee_id": "E001"}, "id": "call_2"},
            ],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三假期", ["lookup_employee", "get_leave_balance"])

        assert len(result) == 2
        assert result[0].tool_name == "lookup_employee"
        assert result[1].tool_name == "get_leave_balance"
        assert result[1].parameters == {"employee_id": "E001"}

    @patch("human_resource.tools.selector.get_llm")
    def test_no_tool_calls_returns_empty(self, mock_get_llm):
        """LLM 未选择任何工具（返回无 tool_calls 的响应）。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(tool_calls=[], content="你好")
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("你好", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_no_tool_calls_attr_returns_empty(self, mock_get_llm):
        """LLM 响应没有 tool_calls 属性时，返回空列表。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        response = MagicMock(spec=[])  # no tool_calls attr
        mock_bound.invoke.return_value = response
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_unregistered_tool_filtered(self, mock_get_llm):
        """LLM 返回未注册的工具名，应被过滤。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            tool_calls=[
                {"name": "ghost_tool", "args": {}, "id": "call_1"},
                {"name": "lookup_employee", "args": {"query": "张三"}, "id": "call_2"},
            ],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].tool_name == "lookup_employee"

    @patch("human_resource.tools.selector.get_llm")
    def test_llm_exception_returns_empty(self, mock_get_llm):
        """LLM 调用异常，返回空列表。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.side_effect = Exception("API 超时")
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_empty_message_returns_empty(self, mock_get_llm):
        """空消息直接返回空列表，不调用 LLM。"""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("", ["lookup_employee"])

        assert result == []
        mock_bound.invoke.assert_not_called()

    @patch("human_resource.tools.selector.get_llm")
    def test_context_passed_to_system_message(self, mock_get_llm):
        """上下文信息应注入到系统消息中。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(tool_calls=[])
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        selector.select("查假期", ["get_leave_balance"], context="之前查到张三的工号是E001")

        # 验证上下文出现在系统消息中
        call_args = mock_bound.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "之前查到张三的工号是E001" in system_msg.content

    @patch("human_resource.tools.selector.get_llm")
    def test_invalid_args_type_defaults_to_empty(self, mock_get_llm):
        """LLM 返回非 dict 的 args，应默认为空 dict。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            tool_calls=[{"name": "lookup_employee", "args": "invalid", "id": "call_1"}],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].parameters == {}

    @patch("human_resource.tools.selector.get_llm")
    def test_bind_tools_called_with_candidate_tools(self, mock_get_llm):
        """验证 bind_tools 使用候选工具列表调用（Native FC）。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(tool_calls=[])
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        selector.select("查张三", ["lookup_employee"])

        # 验证 bind_tools 被调用，且参数是 BaseTool 列表
        mock_llm.bind_tools.assert_called_once()
        bound_tools = mock_llm.bind_tools.call_args[0][0]
        assert len(bound_tools) == 1
        assert bound_tools[0].name == "lookup_employee"

    @patch("human_resource.tools.selector.get_llm")
    def test_no_candidate_tools_returns_empty(self, mock_get_llm):
        """候选工具名列表中无有效工具时，直接返回空。"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["nonexistent_tool"])

        assert result == []
        mock_llm.bind_tools.assert_not_called()

    @patch("human_resource.tools.selector.get_llm")
    def test_tool_call_with_empty_name_skipped(self, mock_get_llm):
        """tool_calls 中 name 为空的条目应被跳过。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            tool_calls=[
                {"name": "", "args": {}, "id": "call_1"},
                {"name": "lookup_employee", "args": {"query": "张三"}, "id": "call_2"},
            ],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].tool_name == "lookup_employee"


# ── ToolCallRequest 测试 ─────────────────────────────────────


class TestToolCallRequest:
    def test_default_values(self):
        tc = ToolCallRequest(tool_name="test")
        assert tc.tool_name == "test"
        assert tc.parameters == {}
        assert tc.reason == ""

    def test_with_values(self):
        tc = ToolCallRequest(
            tool_name="lookup_employee",
            parameters={"query": "张三"},
            reason="查询员工",
        )
        assert tc.tool_name == "lookup_employee"
        assert tc.parameters == {"query": "张三"}
        assert tc.reason == "查询员工"
