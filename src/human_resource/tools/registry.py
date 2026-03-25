"""工具注册表。

统一管理内置工具和 MCP 工具的注册、发现和查找。
支持参数映射：将意图实体（Intent entities）转换为工具参数。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 参数映射函数类型：接收 entities dict，返回 tool params dict
ParamMapper = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """工具注册中心，统一管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._param_mappers: dict[str, ParamMapper] = {}

    def register(
        self,
        tool: BaseTool,
        category: str = "general",
        source: str = "internal",
        param_mapper: ParamMapper | None = None,
    ) -> None:
        """注册一个工具。

        Args:
            tool: LangChain BaseTool 实例。
            category: 工具类别（employee, process 等）。
            source: 来源标记（"internal" | "mcp"）。
            param_mapper: 可选，将意图 entities 映射为工具参数的函数。
        """
        self._tools[tool.name] = tool
        self._metadata[tool.name] = {
            "category": category,
            "source": source,
        }
        if param_mapper is not None:
            self._param_mappers[tool.name] = param_mapper

    def get(self, name: str) -> BaseTool | None:
        """根据名称获取工具。"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    def list_tools(self, category: str | None = None) -> list[BaseTool]:
        """列出所有或指定类别的工具。"""
        if category is None:
            return list(self._tools.values())
        return [
            tool
            for name, tool in self._tools.items()
            if self._metadata[name]["category"] == category
        ]

    def list_by_names(self, names: list[str]) -> list[BaseTool]:
        """根据工具名称列表获取对应工具。"""
        return [self._tools[n] for n in names if n in self._tools]

    def get_all_tools(self) -> list[BaseTool]:
        """获取所有已注册工具列表（用于绑定到 LLM）。"""
        return list(self._tools.values())

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """获取工具的输入参数 JSON Schema。

        LangChain @tool 装饰器自动从函数签名生成 args_schema。
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            return None
        return schema.model_json_schema()

    def get_metadata(self, name: str) -> dict[str, Any] | None:
        """获取工具的元信息（category, source）。"""
        return self._metadata.get(name)

    def build_params(
        self, tool_name: str, entities: dict[str, Any],
    ) -> dict[str, Any]:
        """使用注册的映射函数将意图实体转换为工具参数。

        如果工具未注册映射函数，则直接透传 entities。
        """
        mapper = self._param_mappers.get(tool_name)
        if mapper is not None:
            return mapper(entities)
        return dict(entities)


# 全局注册表单例
registry = ToolRegistry()
