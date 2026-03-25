"""LLM-based 工具选择器。

使用 DeepSeek JSON Output 模式，根据用户请求和候选工具的 Schema，
让 LLM 决定调用哪些工具并生成正确的参数。

两阶段工具选择中的第二阶段（精选）：
  ① Intent Classifier → requires_tools[] 粗筛
  ② ToolSelector → LLM 读取候选工具 Schema → 精细选择 + 参数生成
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from human_resource.tools.registry import registry
from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

_TOOL_SELECTION_PROMPT = """\
你是一个工具调用决策助手。根据用户请求和可用工具的定义，决定需要调用哪些工具及其参数。

## 可用工具

{tools_description}

## 用户请求

{user_message}
{context_section}
## 输出要求

请严格以 json 格式输出，选择需要调用的工具并生成正确的参数。每个工具的参数值必须符合其定义中的类型要求。

输出格式示例：
{{
  "tool_calls": [
    {{
      "tool_name": "工具名称",
      "parameters": {{"参数名": "参数值"}},
      "reason": "简要说明选择理由"
    }}
  ]
}}

如果不需要调用任何工具，返回 {{"tool_calls": []}}。
仅输出 json，不要包含其他文字。"""


@dataclass
class ToolCallRequest:
    """LLM 选择的单个工具调用请求。"""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class ToolSelector:
    """基于 LLM 的工具选择器。

    使用 DeepSeek JSON Output 模式调用 LLM，根据用户请求和候选工具的
    JSON Schema 确定需要调用的工具及其参数。
    """

    def __init__(self) -> None:
        base_llm = get_llm("tool_selection")
        self._llm = base_llm.bind(
            response_format={"type": "json_object"},
        )

    def select(
        self,
        user_message: str,
        candidate_tool_names: list[str],
        context: str = "",
    ) -> list[ToolCallRequest]:
        """使用 LLM 选择工具并生成参数。

        Args:
            user_message: 用户原始请求。
            candidate_tool_names: 候选工具名列表（由 Intent 粗筛得出）。
            context: 上下文信息（已有工具结果、会话历史等）。

        Returns:
            LLM 选择的工具调用请求列表。LLM 失败时返回空列表。
        """
        if not user_message:
            return []

        # 获取候选工具的 Schema 描述
        tools_description = registry.get_tools_with_schemas(
            candidate_tool_names if candidate_tool_names else None,
        )

        # 构建 prompt
        context_section = ""
        if context:
            context_section = f"\n## 上下文信息\n\n{context}\n"

        system_content = _TOOL_SELECTION_PROMPT.format(
            tools_description=tools_description,
            user_message=user_message,
            context_section=context_section,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        # 调用 LLM
        try:
            response = self._llm.invoke(messages)
            content = str(response.content).strip()

            # DeepSeek JSON Output 有概率返回空 content，重试一次
            if not content:
                logger.warning("工具选择 LLM 返回空内容，重试一次")
                response = self._llm.invoke(messages)
                content = str(response.content).strip()
                if not content:
                    logger.warning("工具选择 LLM 两次返回空内容，跳过工具调用")
                    return []

            return self._parse_response(content)
        except Exception:
            logger.exception("工具选择 LLM 调用失败")
            return []

    def _parse_response(self, content: str) -> list[ToolCallRequest]:
        """解析 LLM JSON 输出为 ToolCallRequest 列表。"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("工具选择 LLM 返回非法 JSON: %s", content[:200])
            return []

        tool_calls: list[ToolCallRequest] = []
        for item in data.get("tool_calls", []):
            name = item.get("tool_name", "")
            if not name:
                continue
            # 校验工具是否在 registry 中
            if not registry.has(name):
                logger.warning("LLM 选择了未注册的工具: %s，跳过", name)
                continue
            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            reason = item.get("reason", "")
            tool_calls.append(
                ToolCallRequest(tool_name=name, parameters=params, reason=reason)
            )

        return tool_calls
