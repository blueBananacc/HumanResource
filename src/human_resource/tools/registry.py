"""工具注册表。

统一管理内置工具和 MCP 工具的注册、发现和查找。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


class ToolRegistry:
    """工具注册中心，统一管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool: BaseTool,
        category: str = "general",
        source: str = "internal",
    ) -> None:
        """注册一个工具。

        Args:
            tool: LangChain BaseTool 实例。
            category: 工具类别（employee, process, document 等）。
            source: 来源标记（"internal" | "mcp"）。
        """
        self._tools[tool.name] = tool
        self._metadata[tool.name] = {
            "category": category,
            "source": source,
        }

    def get(self, name: str) -> BaseTool | None:
        """根据名称获取工具。"""
        return self._tools.get(name)

    def list_tools(self, category: str | None = None) -> list[BaseTool]:
        """列出所有或指定类别的工具。"""
        if category is None:
            return list(self._tools.values())
        return [
            tool
            for name, tool in self._tools.items()
            if self._metadata[name]["category"] == category
        ]

    def get_all_tools(self) -> list[BaseTool]:
        """获取所有已注册工具列表（用于绑定到 LLM）。"""
        return list(self._tools.values())


# 全局注册表单例
registry = ToolRegistry()
