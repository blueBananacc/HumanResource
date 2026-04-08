"""Prompt 模板构建。

按固定结构组装最终 prompt，遵循 CRISPE 框架。
"""

from __future__ import annotations

from human_resource.context.manager import ContextManager

# Prompt 结构模板
SYSTEM_PROMPT = """你是一个专业的 HR 助手，为员工提供以下服务：
- HR 政策问答
- HR 流程咨询
- 员工信息查询
- HR 文档检索
- HR 工具操作
- 闲聊或查询总结服务

请基于提供的上下文信息准确回答问题。如果信息不足，请如实说明并建议联系 HR 部门。
回答要求专业、准确、友好。"""


class PromptBuilder:
    """Prompt 模板构建器。

    按固定结构组装各模块的输出为最终 prompt。
    """

    def __init__(self, context_manager: ContextManager | None = None) -> None:
        self._ctx = context_manager or ContextManager()

    def build(
        self,
        user_message: str,
        user_profile: str = "",
        relevant_memories: str = "",
        retrieved_context: str = "",
        tool_results: str = "",
        conversation_history: str = "",
    ) -> str:
        """组装最终 prompt。

        按优先级顺序组装各段落：
        System Prompt > Current Message > Tool Results >
        Retrieved Context > Conversation History > Memories

        Args:
            user_message: 当前用户输入。
            user_profile: 用户画像文本。
            relevant_memories: 相关长期记忆。
            retrieved_context: RAG 检索结果。
            tool_results: 工具调用结果。
            conversation_history: 会话历史。

        Returns:
            组装好的完整 prompt 文本。
        """
        sections: list[str] = [SYSTEM_PROMPT]

        if user_profile:
            profile = self._ctx.truncate_to_budget("user_profile", user_profile)
            sections.append(f"[用户信息]\n{profile}")

        if relevant_memories:
            memories = self._ctx.truncate_to_budget(
                "relevant_memories", relevant_memories
            )
            sections.append(f"[先前相关操作]\n{memories}")

        if retrieved_context:
            context = self._ctx.truncate_to_budget(
                "retrieved_context", retrieved_context
            )
            sections.append(f"[参考资料]\n{context}")

        if tool_results:
            tools = self._ctx.truncate_to_budget("tool_results", tool_results)
            sections.append(f"[工具结果]\n{tools}")

        if conversation_history:
            history = self._ctx.truncate_to_budget(
                "conversation_history", conversation_history
            )
            sections.append(f"[对话历史]\n{history}")

        msg = self._ctx.truncate_to_budget("current_message", user_message)
        sections.append(f"[当前问题]\n{msg}")

        return "\n\n".join(sections)
