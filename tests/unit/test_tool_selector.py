"""ToolSelector 单元测试。

使用 mock 替代 LLM 调用，验证：
- 正常工具选择与参数生成
- 空 content 重试
- 非法 JSON 容错
- 未注册工具过滤
- 无候选工具处理
- 空消息处理
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
        """LLM 正常返回单个工具调用。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            content='{"tool_calls": [{"tool_name": "lookup_employee", "parameters": {"query": "张三"}, "reason": "查询员工"}]}'
        )
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查询张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].tool_name == "lookup_employee"
        assert result[0].parameters == {"query": "张三"}
        assert result[0].reason == "查询员工"

    @patch("human_resource.tools.selector.get_llm")
    def test_select_multiple_tools(self, mock_get_llm):
        """LLM 返回多个工具调用。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            content='{"tool_calls": [{"tool_name": "lookup_employee", "parameters": {"query": "张三"}}, {"tool_name": "get_leave_balance", "parameters": {"employee_id": "E001"}}]}'
        )
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三假期", ["lookup_employee", "get_leave_balance"])

        assert len(result) == 2
        assert result[0].tool_name == "lookup_employee"
        assert result[1].tool_name == "get_leave_balance"

    @patch("human_resource.tools.selector.get_llm")
    def test_empty_tool_calls(self, mock_get_llm):
        """LLM 返回空 tool_calls。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(content='{"tool_calls": []}')
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("你好", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_empty_content_retry(self, mock_get_llm):
        """LLM 第一次返回空 content，第二次正常返回。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        # 第一次空，第二次正常
        mock_bound.invoke.side_effect = [
            MagicMock(content=""),
            MagicMock(
                content='{"tool_calls": [{"tool_name": "lookup_employee", "parameters": {"query": "张三"}}]}'
            ),
        ]
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].tool_name == "lookup_employee"
        assert mock_bound.invoke.call_count == 2

    @patch("human_resource.tools.selector.get_llm")
    def test_both_empty_content_returns_empty(self, mock_get_llm):
        """LLM 两次都返回空 content，返回空列表。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(content="")
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_invalid_json_returns_empty(self, mock_get_llm):
        """LLM 返回非法 JSON，返回空列表。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(content="这不是JSON")
        mock_llm.bind.return_value = mock_bound
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
            content='{"tool_calls": [{"tool_name": "ghost_tool", "parameters": {}}, {"tool_name": "lookup_employee", "parameters": {"query": "张三"}}]}'
        )
        mock_llm.bind.return_value = mock_bound
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
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert result == []

    @patch("human_resource.tools.selector.get_llm")
    def test_empty_message_returns_empty(self, mock_get_llm):
        """空消息直接返回空列表，不调用 LLM。"""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("", ["lookup_employee"])

        assert result == []
        mock_bound.invoke.assert_not_called()

    @patch("human_resource.tools.selector.get_llm")
    def test_context_passed_to_prompt(self, mock_get_llm):
        """上下文信息应注入到 prompt 中。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(content='{"tool_calls": []}')
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        selector.select("查假期", ["get_leave_balance"], context="之前查到张三的工号是E001")

        # 验证上下文出现在 prompt 中
        call_args = mock_bound.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "之前查到张三的工号是E001" in system_msg.content

    @patch("human_resource.tools.selector.get_llm")
    def test_invalid_parameters_type_defaults_to_empty(self, mock_get_llm):
        """LLM 返回非 dict 的 parameters，应默认为空 dict。"""
        _register_test_tools()

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = MagicMock(
            content='{"tool_calls": [{"tool_name": "lookup_employee", "parameters": "invalid"}]}'
        )
        mock_llm.bind.return_value = mock_bound
        mock_get_llm.return_value = mock_llm

        selector = ToolSelector()
        result = selector.select("查张三", ["lookup_employee"])

        assert len(result) == 1
        assert result[0].parameters == {}

    @patch("human_resource.tools.selector.get_llm")
    def test_json_output_format_bound(self, mock_get_llm):
        """验证 ToolSelector 使用 response_format=json_object 绑定 LLM。"""
        mock_llm = MagicMock()
        mock_llm.bind.return_value = MagicMock()
        mock_get_llm.return_value = mock_llm

        ToolSelector()

        mock_llm.bind.assert_called_once_with(
            response_format={"type": "json_object"},
        )


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
