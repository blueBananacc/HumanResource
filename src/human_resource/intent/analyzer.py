"""意图提示生成器（Intent Analyzer）。

对用户输入进行轻量预分析，生成自然语言意图提示，
供 Orchestrator 决策中心参考。使用 DeepSeek-chat（低成本、快速）。

- 不输出结构化分类结果，而是输出自然语言提示
- 不做精确分类和置信度判断，仅提供方向性建议
- 最终决策由 Orchestrator（DeepSeek-Reasoner）自主完成
- 支持 Skill 元数据注入：动态追加可用技能作为额外意图类别
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from human_resource.utils.llm_client import get_llm

if TYPE_CHECKING:
    from human_resource.skills.loader import SkillMetadata

logger = logging.getLogger(__name__)

_ANALYZER_PROMPT = """\
你是一个意图分析助手。请对用户消息进行轻量预分析，生成简短的意图提示。

可参考的意图类别：
- policy_qa: HR 政策问答（年假政策、薪资规定、福利制度等）
- process_inquiry: HR 流程咨询（请假流程、入职流程、离职流程等）
- employee_lookup: 员工信息或个人数据查询（查某人的部门、职位、员工号等）
- memory_recall: 回忆之前的对话内容
- chitchat: 闲聊/问候
- unknown: 无法识别或与 HR 完全无关
{skill_section}
规则：
1. 用一两句话描述用户的意图方向，包含意图类别和理由
2. 如果包含多个意图，全部列出
3. 若意图不明确，直接说明"不明确"，并简要说明原因
4. 若用户请求明确匹配某个可用技能，使用 skill:<技能名> 作为意图标签
5. 判断 Skill 状态时，参考会话上下文中是否已经提议过技能以及用户是否已确认：
   - 若上下文中有技能提议消息且用户本轮确认（如"好的""可以""启用"等肯定回复），输出 skill:<技能名>，并标注"用户已确认技能"
   - 若上下文中有技能提议消息且用户本轮拒绝（如"不用了""取消"等否定回复），不输出 skill 意图，按正常意图分析处理
   - 若上下文中技能正在执行（有技能相关的工具调用或中间结果），继续输出 skill:<技能名>，标注"技能执行中"
   - 若首次检测到匹配技能（上下文中无先前提议记录），输出 skill:<技能名>，标注"首次检测，需提议"

示例：
用户：公司年假是怎么计算的？
输出：理由：用户询问年假计算规则，属于HR政策问题。意图为：policy_qa。
---
消息：请问请假要走什么流程？
输出：理由：用户咨询请假流程。意图为：process_inquiry。
---
消息：我想查一下我的年假额度和今年的调休规则
输出：理由：用户查询个人年假额度（员工信息）并询问调休规则（政策信息）。意图为：employee_lookup + policy_qa。
---
消息：我刚刚问过的那个报销政策你还记得吗？
输出：理由：用户希望回忆之前对话中的报销政策内容。意图为：memory_recall。
---
消息：你好呀，今天心情不错～
输出：理由：用户进行闲聊问候。意图为：chitchat。
---
消息：我想知道请假几天会影响绩效，还有请假怎么申请？
输出：理由：用户既询问请假对绩效的政策影响，也询问请假流程。意图为：policy_qa + process_inquiry。
---
消息：这个东西怎么弄？
输出：理由：用户表达过于模糊，缺乏明确指代对象，无法判断具体意图。意图为：unknown。
---
消息：德国现在天气怎么样？
输出：理由：问题与HR领域无关，无法归类到已有意图类别。意图为：unknown。
---
消息：帮我在知乎上搜5篇关于绩效管理的文章
上下文信息：（无历史记录）
输出：理由：用户明确要求在知乎搜索文章并生成摘要，匹配 zhihu_crawl 技能，上下文中无先前提议记录。意图为：skill:zhihu_crawl。首次检测，需提议
---
消息：好的，启用吧
上下文信息：当前会话摘要：user: 帮我在知乎上搜5篇关于绩效管理的文章\nassistant: 检测到可以使用「根据用户的搜索输入在知乎上搜索 x 篇文章并生成摘要返回给用户」技能来完成此任务。是否启用？
输出：理由：上下文中助手已提议 zhihu_crawl 技能，用户本轮回复"好的，启用吧"表示肯定确认。意图为：skill:zhihu_crawl。用户已确认技能
---
消息：不用了
上下文信息：当前会话摘要：user: 帮我搜几篇知乎文章\nassistant: 检测到可以使用「根据用户的搜索输入在知乎上搜索 x 篇文章并生成摘要返回给用户」技能来完成此任务。是否启用？
输出：理由：上下文中助手已提议 zhihu_crawl 技能，但用户本轮回复"不用了"表示拒绝。意图为：chitchat。
---
消息：搜索结果里第3篇能再详细看看吗
上下文信息：当前会话摘要：user: 好的\nassistant: 已搜索到5篇文章，以下是摘要...\n[工具调用: firecrawl_search 成功]
输出：理由：上下文中 zhihu_crawl 技能正在执行（有搜索工具调用记录），用户要求查看更多细节，仍属于技能任务范围。意图为：skill:zhihu_crawl。技能执行中
---
消息：好了谢谢，我想问一下年假怎么请
上下文信息：当前会话摘要：assistant: 已为您搜索到5篇关于绩效管理的知乎文章并生成摘要...
输出：理由：虽然上下文中有技能执行记录，但用户已明显切换话题到请假流程，不再继续技能任务。意图为：process_inquiry。

上下文信息：{context_section}

用户消息: {user_message}

请直接输出意图提示，不要包含任何额外格式。"""


class IntentAnalyzer:
    """轻量意图提示生成器。"""

    def __init__(self) -> None:
        self._llm = get_llm("intent_hints")
        self._skill_metadata: list[SkillMetadata] = []

    def set_skill_metadata(self, metadata: list[SkillMetadata]) -> None:
        """设置可用 Skill 元数据列表，供 prompt 注入。"""
        self._skill_metadata = list(metadata)

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

        # 动态构建 Skill 元数据段落
        skill_section = ""
        if self._skill_metadata:
            skill_lines = [
                "\n可用技能（当用户请求明确匹配某个技能时，使用 skill:<技能名> 作为意图标签）：",
            ]
            for meta in self._skill_metadata:
                skill_lines.append(f"- {meta.name}: {meta.description}")
            skill_section = "\n".join(skill_lines)

        prompt = _ANALYZER_PROMPT.format(
            user_message=message,
            context_section=context_section,
            skill_section=skill_section,
        )

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            hints = str(response.content).strip()
            return hints
        except Exception:
            logger.exception("意图提示生成失败")
            return ""
