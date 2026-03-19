"""工具执行器。

处理参数校验、工具调用和结果格式化。
"""

from __future__ import annotations

from typing import Any

from human_resource.schemas.models import ToolResult
from human_resource.tools.registry import registry


def execute_tool(tool_name: str, parameters: dict[str, Any]) -> ToolResult:
    """执行指定工具。

    Args:
        tool_name: 工具名称。
        parameters: 工具参数。

    Returns:
        ToolResult 结构化结果。
    """
    tool = registry.get(tool_name)
    if tool is None:
        return ToolResult(
            success=False,
            error=f"未找到工具: {tool_name}",
        )

    try:
        result = tool.invoke(parameters)
        return ToolResult(
            success=True,
            data=result,
            formatted=str(result),
        )
    except Exception as exc:
        return ToolResult(
            success=False,
            error=f"工具执行失败: {exc}",
        )
