"""MCP Client。

通过 langchain-mcp-adapters 连接外部 MCP Server，
发现并调用 MCP 工具，适配为系统内部 Tool Registry 统一格式。

当前为预留骨架，MCP Server 就绪后只需配置连接信息即可接入。
"""

from __future__ import annotations

# TODO: MCP Server 就绪后实现
# 1. 使用 langchain-mcp-adapters 建立 MCP Client 连接
# 2. 调用 tools/list 发现所有可用 MCP 工具
# 3. 将 MCP 工具适配为 LangChain Tool 并注册到 Tool Registry
# 4. 支持 stdio 和 SSE transport
