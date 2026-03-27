"""MCP Client 模块单元测试。

覆盖：
- start_mcp_client: 正常注册、名称冲突、空配置、异常处理
- stop_mcp_client: 清理单例
- register_mcp_tools_sync: 同步包装行为
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, tool

from human_resource.mcp import client as mcp_client_module
from human_resource.mcp.client import (
    register_mcp_tools_sync,
    start_mcp_client,
    stop_mcp_client,
)


# ── helpers ──────────────────────────────────────────────────

def _make_tool(name: str) -> BaseTool:
    """创建一个用于测试的简单工具。"""
    @tool
    def dummy(x: str) -> str:
        """Dummy tool."""
        return x

    dummy.name = name
    return dummy


def _run(coro):
    """同步运行异步协程。"""
    return asyncio.run(coro)


# ── start_mcp_client ─────────────────────────────────────────


class TestStartMcpClient:
    """start_mcp_client 异步函数测试。"""

    def setup_method(self):
        # 重置模块级单例
        mcp_client_module._mcp_client = None

    @patch("human_resource.mcp.client.MCP_SERVERS", {
        "test-server": {
            "command": "node",
            "args": ["test.js"],
            "transport": "stdio",
        },
    })
    @patch("human_resource.mcp.client.registry")
    @patch("human_resource.mcp.client.MultiServerMCPClient")
    def test_normal_registration(self, MockClient, mock_registry):
        """正常情况：MCP Server 返回工具，全部注册。"""
        tool_a = _make_tool("send_email")
        tool_b = _make_tool("test_connection")

        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(return_value=[tool_a, tool_b])
        MockClient.return_value = mock_instance
        mock_registry.has.return_value = False

        count = _run(start_mcp_client())

        assert count == 2
        assert mock_registry.register.call_count == 2
        mock_registry.register.assert_any_call(
            tool_a, category="mcp", source="mcp"
        )
        mock_registry.register.assert_any_call(
            tool_b, category="mcp", source="mcp"
        )

    @patch("human_resource.mcp.client.MCP_SERVERS", {
        "test-server": {
            "command": "node",
            "args": ["test.js"],
            "transport": "stdio",
        },
    })
    @patch("human_resource.mcp.client.registry")
    @patch("human_resource.mcp.client.MultiServerMCPClient")
    def test_name_conflict_skips_duplicate(self, MockClient, mock_registry):
        """名称冲突：已注册的工具名应跳过。"""
        tool_a = _make_tool("send_email")
        tool_b = _make_tool("existing_tool")

        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(return_value=[tool_a, tool_b])
        MockClient.return_value = mock_instance
        mock_registry.has.side_effect = lambda name: name == "existing_tool"

        count = _run(start_mcp_client())

        assert count == 1
        assert mock_registry.register.call_count == 1
        mock_registry.register.assert_called_once_with(
            tool_a, category="mcp", source="mcp"
        )

    @patch("human_resource.mcp.client.MCP_SERVERS", {})
    def test_empty_config_returns_zero(self):
        """空配置：无 MCP Server 时返回 0。"""
        count = _run(start_mcp_client())
        assert count == 0

    @patch("human_resource.mcp.client.MCP_SERVERS", {
        "test-server": {
            "command": "node",
            "args": ["test.js"],
            "transport": "stdio",
        },
    })
    @patch("human_resource.mcp.client.registry")
    @patch("human_resource.mcp.client.MultiServerMCPClient")
    def test_get_tools_exception_returns_zero(self, MockClient, mock_registry):
        """get_tools 异常时返回 0，不崩溃。"""
        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(side_effect=ConnectionError("refused"))
        MockClient.return_value = mock_instance

        count = _run(start_mcp_client())

        assert count == 0
        mock_registry.register.assert_not_called()

    @patch("human_resource.mcp.client.MCP_SERVERS", {
        "test-server": {
            "command": "node",
            "args": ["test.js"],
            "transport": "stdio",
        },
    })
    @patch("human_resource.mcp.client.registry")
    @patch("human_resource.mcp.client.MultiServerMCPClient")
    def test_no_tools_returned(self, MockClient, mock_registry):
        """MCP Server 返回空工具列表。"""
        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(return_value=[])
        MockClient.return_value = mock_instance

        count = _run(start_mcp_client())

        assert count == 0
        mock_registry.register.assert_not_called()

    @patch("human_resource.mcp.client.MCP_SERVERS", {
        "test-server": {
            "command": "node",
            "args": ["test.js"],
            "transport": "stdio",
        },
    })
    @patch("human_resource.mcp.client.registry")
    @patch("human_resource.mcp.client.MultiServerMCPClient")
    def test_sets_module_singleton(self, MockClient, mock_registry):
        """启动后设置模块级单例 _mcp_client。"""
        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(return_value=[])
        MockClient.return_value = mock_instance

        _run(start_mcp_client())

        assert mcp_client_module._mcp_client is mock_instance


# ── stop_mcp_client ──────────────────────────────────────────


class TestStopMcpClient:
    """stop_mcp_client 异步函数测试。"""

    def test_stop_clears_singleton(self):
        """停止后单例应为 None。"""
        mcp_client_module._mcp_client = MagicMock()

        _run(stop_mcp_client())

        assert mcp_client_module._mcp_client is None

    def test_stop_when_already_none(self):
        """单例已为 None 时不抛异常。"""
        mcp_client_module._mcp_client = None

        _run(stop_mcp_client())  # should not raise

        assert mcp_client_module._mcp_client is None


# ── register_mcp_tools_sync ──────────────────────────────────


class TestRegisterMcpToolsSync:
    """register_mcp_tools_sync 同步包装测试。"""

    @patch("human_resource.mcp.client.start_mcp_client")
    def test_sync_wrapper_calls_start(self, mock_start):
        """同步包装应调用 start_mcp_client 并返回结果。"""
        async def fake_start():
            return 3

        mock_start.side_effect = fake_start

        mcp_client_module._mcp_client = None
        result = register_mcp_tools_sync()

        assert result == 3

    @patch("human_resource.mcp.client.start_mcp_client")
    def test_sync_wrapper_returns_zero_on_empty(self, mock_start):
        """无 MCP Server 时同步包装返回 0。"""
        async def fake_start():
            return 0

        mock_start.side_effect = fake_start

        mcp_client_module._mcp_client = None
        result = register_mcp_tools_sync()

        assert result == 0
