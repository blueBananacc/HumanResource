"""意图提示生成器（Intent Analyzer）。

对用户输入进行轻量预分析，生成自然语言意图提示，
供 Orchestrator 决策中心参考。使用 DeepSeek-chat（低成本、快速）。

- 不输出结构化分类结果，而是输出自然语言提示
- 不做精确分类和置信度判断，仅提供方向性建议
- 最终决策由 Orchestrator（DeepSeek-Reasoner）自主完成
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

_ANALYZER_PROMPT = """\
你是一个意图分析助手。请对用户消息进行轻量预分析，生成简短的意图提示。

可参考的意图类别：
- policy_qa: HR 政策问答（年假政策、薪资规定、福利制度等）
- process_inquiry: HR 流程咨询（请假流程、入职流程、离职流程等）
- employee_lookup: 员工信息查询（查某人的部门、职位、联系方式等）
- memory_recall: 回忆之前的对话内容
- chitchat: 闲聊/问候
- unknown: 无法识别或与 HR 完全无关

规则：
1. 用一两句话描述用户的意图方向，包含意图类别和理由
2. 如果包含多个意图，全部列出
3. 若意图不明确，直接说明“不明确”，并简要说明原因

示例：
用户：公司年假是怎么计算的？
输出：policy_qa。理由：用户询问年假计算规则，属于HR政策问题。
---
消息：请问请假要走什么流程？
输出：process_inquiry。理由：用户咨询请假流程。
---
消息：帮我查一下张三在哪个部门？
输出：employee_lookup。理由：用户查询员工张三的部门信息。
---
消息：我刚刚问过的那个报销政策你还记得吗？
输出：memory_recall。理由：用户希望回忆之前对话中的报销政策内容。
---
消息：你好呀，今天心情不错～
输出：chitchat。理由：用户进行闲聊问候。
---
消息：我想知道请假几天会影响绩效，还有请假怎么申请？
输出：policy_qa + process_inquiry。理由：用户既询问请假对绩效的政策影响，也询问请假流程。
---
消息：这个东西怎么弄？
输出：unknown。理由：用户表达过于模糊，缺乏明确指代对象，无法判断具体意图。
---
消息：德国现在天气怎么样？
输出：unknown。理由：问题与HR领域无关，无法归类到已有意图类别。

上下文信息：{context_section}

用户消息: {user_message}

请直接输出意图提示，不要包含任何额外格式。"""


class IntentAnalyzer:
    """轻量意图提示生成器。"""

    def __init__(self) -> None:
        self._llm = get_llm("intent_hints")

    def analyze(
        self,
        message: str,
        session_summary: str = "",
        long_term_memory: str = "",
        user_profile: str = "",
    ) -> str:
        """对用户消息生成自然语言意图提示。

        Args:
            message: 用户输入消息。
            session_summary: 当前会话摘要。
            long_term_memory: 长期记忆片段。
            user_profile: 用户画像。

        Returns:
            自然语言意图提示字符串。
        """
        context_parts: list[str] = []
        if user_profile:
            context_parts.append(f"用户画像：{user_profile}")
        if long_term_memory:
            context_parts.append(f"用户相关记忆：{long_term_memory}")
        if session_summary:
            context_parts.append(f"当前会话摘要：{session_summary}")

        context_section = ""
        if context_parts:
            context_section = "上下文信息：\n" + "\n".join(context_parts)

        prompt = _ANALYZER_PROMPT.format(
            user_message=message,
            context_section=context_section,
        )

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            hints = str(response.content).strip()
            logger.info("意图提示生成: %s", hints[:100])
            return hints
        except Exception:
            logger.exception("意图提示生成失败")
            return ""
