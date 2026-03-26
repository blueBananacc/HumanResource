"""LLM-based 工具选择器。

使用 DeepSeek Native Function Calling，通过 llm.bind_tools() 将候选工具
绑定到 LLM，利用模型后训练（Post-training）的工具选择能力进行精确调用。

两阶段工具选择中的第二阶段（精选）：
  ① Intent Classifier → requires_tools[] 粗筛
  ② ToolSelector → bind_tools(候选工具) → Native Function Calling 精选
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from human_resource.tools.registry import registry
from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

_TOOL_SELECTION_SYSTEM_PROMPT = """\
你是一个 HR 智能助手的工具调用代理。根据用户请求和上下文信息，选择合适的工具并生成正确的调用参数。

规则：
- 仅在用户请求明确需要工具时才调用
- 参数值必须从用户消息或上下文中提取，不要编造不存在的信息
- 可以同时调用多个工具"""


@dataclass
class ToolCallRequest:
    """LLM 选择的单个工具调用请求。"""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class ToolSelector:
    """基于 Native Function Calling 的工具选择器。

    通过 llm.bind_tools() 将候选工具绑定到 LLM，利用模型后训练
    （Post-training）的工具选择能力，而非纯文本 Prompt 引导。
    DeepSeek API 支持 OpenAI function calling 格式，与 LangChain
    tool binding 完全兼容。
    """

    def __init__(self) -> None:
        self._llm = get_llm("tool_selection")

    def select(
        self,
        user_message: str,
        candidate_tool_names: list[str],
        context: str = "",
    ) -> list[ToolCallRequest]:
        """使用 Native Function Calling 选择工具并生成参数。

        Args:
            user_message: 用户原始请求。
            candidate_tool_names: 候选工具名列表（由 Intent 粗筛得出）。
            context: 上下文信息（已有工具结果、会话历史等）。

        Returns:
            LLM 选择的工具调用请求列表。LLM 失败时返回空列表。
        """
        if not user_message:
            return []

        # 从 registry 获取候选 BaseTool 对象
        candidate_tools = registry.list_by_names(candidate_tool_names)
        if not candidate_tools:
            logger.info("ToolSelector: 无有效候选工具")
            return []

        # 通过 bind_tools 将候选工具绑定到 LLM（Native Function Calling）
        llm_with_tools = self._llm.bind_tools(candidate_tools)

        # 构建消息（工具 schema 由 bind_tools 自动注入 API，无需在 prompt 中描述）
        system_content = _TOOL_SELECTION_SYSTEM_PROMPT
        if context:
            system_content += f"\n\n## 上下文信息\n{context}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        try:
            response = llm_with_tools.invoke(messages)
            return self._parse_tool_calls(response)
        except Exception:
            logger.exception("工具选择 LLM 调用失败")
            return []

    def _parse_tool_calls(self, response: Any) -> list[ToolCallRequest]:
        """从 AIMessage.tool_calls 解析工具调用请求。

        LangChain 自动将 Native Function Calling 的返回解析为
        tool_calls 列表，每项包含 name、args、id。
        """
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return []

        results: list[ToolCallRequest] = []
        for tc in tool_calls:
            name = tc.get("name", "")
            if not name:
                continue
            if not registry.has(name):
                logger.warning("LLM 选择了未注册的工具: %s，跳过", name)
                continue
            args = tc.get("args", {})
            if not isinstance(args, dict):
                args = {}
            results.append(ToolCallRequest(tool_name=name, parameters=args))

        return results
