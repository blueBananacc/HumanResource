"""LLM-based 意图分类器。

使用 DeepSeek-reasoner 进行意图分类，输出结构化 IntentResult。
支持多意图识别和低置信度回退。
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from human_resource.config import INTENT_CONFIDENCE_THRESHOLD
from human_resource.schemas.models import IntentItem, IntentLabel, IntentResult
from human_resource.utils.llm_client import get_llm

logger = logging.getLogger(__name__)

_CLASSIFICATION_SYSTEM_PROMPT = """\
你是一个意图分类器。请根据用户消息判断其意图类别。

可选的意图类别：
- policy_qa: HR 政策问答（如年假政策、薪资规定、福利制度等）
- process_inquiry: HR 流程咨询（如请假流程、入职流程、离职流程等）
- employee_lookup: 员工信息查询（如查询某人的部门、职位、联系方式等）
- document_search: HR 文档检索（如查找员工手册、规章制度文档等）
- tool_action: HR 工具操作（如查询假期余额、提交申请等）
- memory_recall: 回忆之前的对话内容（如"我们之前聊了什么"）
- chitchat: 闲聊/问候（如"你好"、"谢谢"）
- unknown: 无法识别为上述任何类别

规则：
1. 一条消息可能包含多个意图，请全部列出。
2. 多意图时，按照解决问题的逻辑顺序排列（先查数据再做分析、先获取信息再回答问题），而非按置信度排序。
3. 为每个意图给出 0.0 到 1.0 之间的置信度。
4. 每个意图单独声明所需工具（requires_tools），仅列出该意图自身需要的工具。
   可用工具：lookup_employee, get_leave_balance, list_hr_processes, get_process_steps
5. 提取消息中的关键实体（人名、部门、日期等）放入 entities。

请严格以 JSON 格式输出，不要包含其他文字：
{
  "intents": [
    {"label": "意图类别", "confidence": 0.9, "entities": {"name": "张三"}, "requires_tools": ["工具名"]}
  ]
}
"""

_FEW_SHOT_EXAMPLES = """
示例：

场景一（单意图 + 单Agent）:
用户: "年假政策是什么"
输出: {"intents": [{"label": "policy_qa", "confidence": 0.95, "entities": {"topic": "年假"}, "requires_tools": []}]}

用户: "查询张三的部门"
输出: {"intents": [{"label": "employee_lookup", "confidence": 0.95, "entities": {"name": "张三"}, "requires_tools": ["lookup_employee"]}]}

用户: "你好"
输出: {"intents": [{"label": "chitchat", "confidence": 0.95, "entities": {}, "requires_tools": []}]}

场景二（单意图 + 多Agent 串行）:
用户: "入职流程是什么"
说明: 单一 process_inquiry 意图，需要 RAG 检索流程文档，并可能调用流程工具获取步骤
输出: {"intents": [{"label": "process_inquiry", "confidence": 0.9, "entities": {"process": "入职"}, "requires_tools": ["list_hr_processes", "get_process_steps"]}]}

场景三（多意图 + 每个意图单Agent）:
用户: "张三是哪个部门的？他们部门的考勤制度是什么"
说明: 先查员工信息获取部门（tool_agent），再检索该部门考勤政策（rag_agent），每个意图各需一个Agent
输出: {"intents": [{"label": "employee_lookup", "confidence": 0.95, "entities": {"name": "张三"}, "requires_tools": ["lookup_employee"]}, {"label": "policy_qa", "confidence": 0.9, "entities": {"topic": "考勤制度"}, "requires_tools": []}]}

场景四（多意图 + 每个意图多Agent）:
用户: "帮我查一下李四的假期余额，然后告诉我请假流程"
说明: 先查假期余额（tool_agent），再查请假流程（rag_agent + tool_agent 串行），多个意图且 process_inquiry 涉及多Agent
输出: {"intents": [{"label": "tool_action", "confidence": 0.9, "entities": {"name": "李四"}, "requires_tools": ["get_leave_balance"]}, {"label": "process_inquiry", "confidence": 0.85, "entities": {"process": "请假"}, "requires_tools": ["get_process_steps"]}]}
"""


class IntentClassifier:
    """基于 LLM 的意图分类器。"""

    def __init__(self) -> None:
        self._llm = get_llm("intent_classification")

    def classify(
        self,
        message: str,
        session_summary: str = "",
        long_term_memory: str = "",
        user_profile: str = "",
    ) -> IntentResult:
        """对用户消息进行意图分类。

        Args:
            message: 用户输入消息。
            session_summary: 当前会话摘要（可选，用于上下文消歧）。
            long_term_memory: 长期记忆片段（可选，提供用户历史偏好）。
            user_profile: 用户画像（可选，提供用户背景信息）。

        Returns:
            IntentResult，包含一个或多个意图分类结果。
        """
        system_content = _CLASSIFICATION_SYSTEM_PROMPT + _FEW_SHOT_EXAMPLES
        if user_profile:
            system_content += f"\n用户画像（用于理解用户背景）：\n{user_profile}"
        if long_term_memory:
            system_content += f"\n用户相关记忆（用于辅助消歧）：\n{long_term_memory}"
        if session_summary:
            system_content += f"\n当前会话摘要（用于辅助消歧）：\n{session_summary}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=message),
        ]

        try:
            response = self._llm.invoke(messages)
            return self._parse_response(response.content)
        except Exception:
            logger.exception("意图分类失败，使用 fallback")
            return IntentResult(
                intents=[IntentItem(label=IntentLabel.UNKNOWN, confidence=0.0)]
            )

    def _parse_response(self, content: str) -> IntentResult:
        """解析 LLM JSON 输出为 IntentResult。"""
        # 从返回内容中提取 JSON（处理可能的 markdown 代码块包裹）
        text = content.strip()
        if text.startswith("```"):
            # 去除 ```json ... ``` 包裹
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]).strip()

        data = json.loads(text)
        intents: list[IntentItem] = []
        for item in data.get("intents", []):
            try:
                label = IntentLabel(item["label"])
            except ValueError:
                label = IntentLabel.UNKNOWN
            intents.append(
                IntentItem(
                    label=label,
                    confidence=float(item.get("confidence", 0.0)),
                    entities=item.get("entities", {}),
                    requires_tools=item.get("requires_tools", []),
                )
            )

        if not intents:
            intents = [IntentItem(label=IntentLabel.UNKNOWN, confidence=0.0)]

        return IntentResult(intents=intents)
