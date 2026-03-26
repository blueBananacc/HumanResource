"""工具注册表。

统一管理内置工具和 MCP 工具的注册、发现和查找。
提供工具描述和 Schema 获取方法，用于 LLM prompt 注入（意图分类粗筛 + 工具选择精选）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


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
            category: 工具类别（employee, process 等）。
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

    # ── 工具描述方法（供 LLM prompt 注入） ──────────────────────

    def _get_target_tools(
        self, names: list[str] | None = None,
    ) -> list[BaseTool]:
        """按名称列表获取工具，为空时返回全部。"""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def get_tools_summary(
        self, names: list[str] | None = None,
    ) -> str:
        """获取工具名称和描述的摘要列表（用于意图分类粗筛）。

        格式：每行 "- 工具名: 描述首行"

        Args:
            names: 指定工具名列表，为 None 时返回全部。
        """
        tools = self._get_target_tools(names)
        if not tools:
            return "无可用工具"
        lines = []
        for t in tools:
            desc = (t.description or "").split("\n")[0].strip()
            lines.append(f"- {t.name}: {desc}")
        return "\n".join(lines)

    def get_tools_with_schemas(
        self, names: list[str] | None = None,
    ) -> str:
        """获取工具名称、描述和参数 Schema（用于 LLM 工具选择精选）。

        Args:
            names: 指定工具名列表，为 None 时返回全部。
        """
        tools = self._get_target_tools(names)
        if not tools:
            return "无可用工具"
        parts = []
        for t in tools:
            desc = (t.description or "").split("\n")[0].strip()
            schema = self.get_schema(t.name)
            params_desc = self._format_params(schema)
            parts.append(f"### {t.name}\n描述: {desc}\n参数:\n{params_desc}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_params(schema: dict[str, Any] | None) -> str:
        """将 JSON Schema 格式化为可读的参数描述。"""
        if not schema:
            return "  无参数"
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            return "  无参数"
        lines = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = "必填" if pname in required else "可选"
            lines.append(f"  - {pname} ({ptype}, {req}): {pdesc}")
        return "\n".join(lines)


# 全局注册表单例
registry = ToolRegistry()
