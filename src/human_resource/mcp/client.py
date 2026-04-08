"""MCP Client。

通过 langchain-mcp-adapters 连接本地运行的 MCP Server，
发现并调用 MCP 工具，适配为系统内部 Tool Registry 统一格式。

使用 MultiServerMCPClient 管理 MCP Server 生命周期：
  - CLI 启动时同步启动本地 MCP Server 进程（stdio transport）
  - 自动发现 MCP 工具并注册到 Tool Registry（source="mcp"）
  - CLI 退出时关闭 MCP Server 进程
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from human_resource.config import MCP_ALLOWED_TOOLS, MCP_SERVERS
from human_resource.tools.registry import registry

logger = logging.getLogger(__name__)

# 模块级 MCP Client 实例（进程内单例）
_mcp_client: MultiServerMCPClient | None = None


def _is_allowed_mcp_tool(tool_name: str) -> bool:
    """判断 MCP 工具是否在允许注册白名单内。

    兼容两类命名：
    - 原始名：send_email
    - 带服务器前缀名：mcp_firecrawl_fir_firecrawl_search
    """
    return any(
        tool_name == allowed or tool_name.endswith(f"_{allowed}")
        for allowed in MCP_ALLOWED_TOOLS
    )


async def start_mcp_client() -> int:
    """启动 MCP Client，连接所有配置的 MCP Server 并注册工具。

    MCP Server 进程随 MultiServerMCPClient 在后台启动（stdio transport），
    工具通过 get_tools() 自动发现；日志列出全部发现结果，
    实际仅注册白名单中的工具到 Tool Registry。

    Returns:
        注册的 MCP 工具数量。
    """
    global _mcp_client

    if not MCP_SERVERS:
        logger.info("MCP: 无配置的 MCP Server，跳过")
        return 0

    _mcp_client = MultiServerMCPClient(MCP_SERVERS)  # type: ignore[arg-type]

    tool_count = 0
    try:
        tools = await _mcp_client.get_tools()
        discovered_tool_names = [tool.name for tool in tools]
        if discovered_tool_names:
            logger.info(
                "MCP 可发现工具(%d): %s",
                len(discovered_tool_names),
                ", ".join(discovered_tool_names),
            )
        else:
            logger.info("MCP 未发现可用工具")

        for tool in tools:
            if not _is_allowed_mcp_tool(tool.name):
                logger.info("MCP 工具未在白名单中，跳过注册: %s", tool.name)
                continue
            if not registry.has(tool.name):
                registry.register(tool, category="mcp", source="mcp")
                logger.info("MCP 工具已注册: %s", tool.name)
                tool_count += 1
            else:
                logger.warning("MCP 工具名冲突，跳过: %s", tool.name)
        logger.info(
            "MCP Client 启动完成: 发现 %d 个工具, 注册 %d 个白名单工具",
            len(discovered_tool_names),
            tool_count,
        )
    except Exception:
        logger.exception("MCP Client 启动失败")

    return tool_count


async def stop_mcp_client() -> None:
    """停止 MCP Client，关闭所有 MCP Server 连接。"""
    global _mcp_client
    if _mcp_client is None:
        return
    # MultiServerMCPClient 的 session 会自动关闭
    # 标记单例为空以支持重新初始化
    _mcp_client = None
    logger.info("MCP Client 已停止")


def register_mcp_tools_sync() -> int:
    """同步包装：启动 MCP Client 并注册工具。

    在非 async 上下文中调用（如 compile_graph）。

    Returns:
        注册的 MCP 工具数量。
    """
    return asyncio.run(start_mcp_client())
