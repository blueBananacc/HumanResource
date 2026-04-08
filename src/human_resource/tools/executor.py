"""工具执行器。

处理参数校验、工具调用、超时保护和结果格式化。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from human_resource.schemas.models import ToolResult
from human_resource.tools.registry import registry

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 30


def validate_params(tool_name: str, params: dict[str, Any]) -> tuple[bool, str | None]:
    """使用工具的 args_schema 校验参数。

    Returns:
        (is_valid, error_message) — 校验通过时 error_message 为 None。
    """
    tool = registry.get(tool_name)
    if tool is None:
        return False, f"未找到工具: {tool_name}"

    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is None:
        return True, None

    # MCP 工具的 args_schema 可能是 dict（JSON Schema），跳过校验
    if isinstance(schema_cls, dict):
        return True, None

    try:
        schema_cls(**params)
        return True, None
    except Exception as exc:
        return False, f"参数校验失败: {exc}"


def format_result(tool_name: str, data: Any) -> str:
    """将工具原始返回值格式化为 LLM 上下文友好的文本。

    保留关键字段，结构化呈现。
    """
    if isinstance(data, dict):
        if "error" in data:
            return f"[{tool_name}] 错误: {data['error']}"
        lines = [f"[{tool_name} 结果]"]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    if isinstance(data, list):
        lines = [f"[{tool_name} 结果] 共 {len(data)} 项"]
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                summary = ", ".join(f"{k}={v}" for k, v in item.items())
                lines.append(f"  {i}. {summary}")
            else:
                lines.append(f"  {i}. {item}")
        return "\n".join(lines)

    return f"[{tool_name} 结果] {data}"


def execute_tool(tool_name: str, parameters: dict[str, Any]) -> ToolResult:
    """执行指定工具。

    流程：查找工具 → 参数校验 → 超时保护执行 → 结果格式化。

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
            tool_name=tool_name,
            error=f"未找到工具 '{tool_name}'",
        )

    # ── 参数校验 ──
    valid, err_msg = validate_params(tool_name, parameters)
    if not valid:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"工具 '{tool_name}' 参数错误: {err_msg}",
        )

    # ── 超时保护执行 ──
    # MCP 工具仅支持 async 调用，通过 registry 元信息判断来源
    metadata = registry.get_metadata(tool_name)
    is_mcp = metadata is not None and metadata.get("source") == "mcp"
    try:
        if is_mcp:
            # 异步分支：必须使用 asyncio.wait_for 来实现超时控制
            async def run_with_timeout():
                return await asyncio.wait_for(
                    tool.ainvoke(parameters), 
                    timeout=TOOL_TIMEOUT_SECONDS
                )
            
            # 注意：如果当前已经在事件循环中，这里可能需要 nest_asyncio
            result = asyncio.run(run_with_timeout())
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.invoke, parameters)
                result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
    except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
        logger.warning("工具 %s 执行超时（%d秒）", tool_name, TOOL_TIMEOUT_SECONDS)
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"工具 '{tool_name}' 执行超时（超过{TOOL_TIMEOUT_SECONDS}秒未响应）",
        )
    except Exception as exc:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"工具 '{tool_name}' 执行异常: {type(exc).__name__}: {exc}",
        )

    # ── 结果格式化 ──
    # 工具返回 {"error": ...} 表示业务级失败
    if isinstance(result, dict) and "error" in result:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=result,
            error=f"工具 '{tool_name}' 调用失败: {result['error']}",
        )

    formatted = format_result(tool_name, result)
    return ToolResult(
        success=True,
        tool_name=tool_name,
        data=result,
        formatted=formatted,
    )
