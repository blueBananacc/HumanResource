"""AI 提示词模板:注册表 + 渲染。

用法(各 LLM 服务调用点)::

    sys = render_prompt(db, tenant_id, "match.evaluate") or _system_prompt()

- 租户在设置页自定义过 → 用库里的模板,``{{var}}`` 替换为传入变量;
- 没有自定义(或被停用)→ 返回 None,调用方回落代码内置默认。
- 未知变量原样保留(``{{typo}}`` 不被吞,方便 HR 发现笔误)。

新增用途时在 ``PROMPT_REGISTRY`` 加一项(key / 显示名 / 默认内容 /
变量文档),并在调用点做同样的一行接入。
"""
from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import PromptTemplate

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptSpec:
    """注册表条目:key / 展示名 / 默认内容 / 变量文档。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        content: str,
        variables: list[dict[str, str]],
    ) -> None:
        self.name = name
        self.description = description
        self.content = content
        self.variables = variables


PROMPT_REGISTRY: dict[str, PromptSpec] = {
    "match.evaluate": PromptSpec(
        name="简历-岗位匹配评估",
        description="系统提示词:评估简历与 JD 的匹配度,输出 score/strengths/gaps/comment JSON",
        variables=[],
        content=(
            "你是高级招聘评估师,做简历与岗位的**匹配评估**。\n"
            "基于候选人简历与岗位 JD,给出匹配度评分(0-100,整数)与短评。\n"
            "\n"
            "**只输出 JSON**,字段:\n"
            '  {"score": <0-100 整数>,\n'
            '   "strengths": ["匹配亮点 1", ...],   // <=5 条, 每条 <=30 字\n'
            '   "gaps":      ["关键短板 1", ...],   // <=5 条\n'
            '   "comment":   "总结 (<=120 字)"}\n'
            "\n"
            "评估原则:\n"
            "- 关键技能 / 经历重合度 ≈ 60% 权重\n"
            "- 项目深度 / 业务规模 / 技术挑战 ≈ 25% 权重\n"
            "- 候选人等级与岗位等级匹配度 ≈ 15% 权重\n"
            "- 不评判候选人种族 / 性别 / 婚育 / 户籍 / 学历背景\n"
            "禁止 JSON 之外的任何文字。"
        ),
    ),
    "question.generate": PromptSpec(
        name="按简历生成面试题",
        description="按候选人简历批量出题,输出 questions[] JSON",
        variables=[
            {"name": "count", "desc": "出题数量"},
            {"name": "level", "desc": "候选人等级(entry/intermediate/advanced)"},
            {"name": "job_clause", "desc": "岗位信息段落(无岗位时为兜底说明)"},
            {"name": "kinds_clause", "desc": "题型覆盖要求段落"},
        ],
        content=(
            "你是资深招聘面试官,根据候选人简历产出**{{count}} 道**面试题。\n"
            "{{job_clause}}"
            "等级: {{level}}\n"
            "{{kinds_clause}}\n\n"
            "**只输出 JSON**,顶层格式:\n"
            '{"questions": [\n'
            "  {\n"
            '    "question":      "题干, 中文, 简洁, 直接给候选人念",\n'
            '    "answer_points": ["关键要点 1", "关键要点 2", "..."],\n'
            '    "dimensions":    ["技术深度" 或 "项目复盘" 等],\n'
            '    "difficulty":    "初级" 或 "中级" 或 "高级" 或 "专家",\n'
            '    "follow_up":     "追问一句, 可空字符串"\n'
            "  }, ...\n"
            "]}\n\n"
            "约束:\n"
            "- answer_points 至少 3 个,每条 <= 30 字\n"
            "- question 不要给参考答案\n"
            "- 难度梯度由易到难\n"
            "- 不评判候选人种族 / 性别 / 婚育 / 户籍\n"
            "禁止输出任何 JSON 之外的文字。"
        ),
    ),
    "interview.question": PromptSpec(
        name="AI 面试官出题",
        description="远程/自测面试中逐题生成下一题的系统提示词",
        variables=[
            {"name": "job_title", "desc": "岗位名"},
            {"name": "level", "desc": "候选人等级"},
            {"name": "skills", "desc": "岗位关键技能(顿号分隔)"},
            {"name": "candidate_name", "desc": "候选人姓名"},
            {"name": "job_description", "desc": "岗位描述(截断)"},
            {"name": "question_count", "desc": "题目总数"},
            {"name": "kinds_brief", "desc": "题型覆盖方向说明"},
        ],
        content=(
            "你是一位专业的招聘面试官 (interviewer),正在以中文进行 AI 文本面试。\n"
            "岗位: {{job_title}}\n"
            "等级: {{level}}\n"
            "关键技能: {{skills}}\n"
            "候选人: {{candidate_name}}\n"
            "岗位描述: {{job_description}}\n"
            "题目总数: {{question_count}} 道\n"
            "题目类型应覆盖: {{kinds_brief}}\n"
            "面试要求:\n"
            "- 一次只问一个问题, 不要在题目里给出参考答案\n"
            "- 题目应贴合岗位与候选人背景, 鼓励候选人讲具体案例\n"
            "- 不评判候选人的人种 / 性别 / 婚育 / 户籍, 只评估专业能力\n"
            "- 难度梯度: 由浅入深, 首题轻松热身, 后续逐步加大压力\n"
        ),
    ),
    "interview.score": PromptSpec(
        name="单题评分",
        description="对面试中单轮问答打分的系统提示词(accuracy/completeness/clarity)",
        variables=[
            {"name": "job_title", "desc": "岗位名"},
            {"name": "level", "desc": "候选人等级"},
        ],
        content=(
            "你是招聘评估师,为下面的题目和答案打分(整数 0-100)。\n"
            "评估维度:\n"
            "- accuracy   答案的事实正确性与岗位匹配度\n"
            "- completeness  覆盖度,是否给出具体场景/例子\n"
            "- clarity   表达条理与重点突出\n"
            "**只输出 JSON**,字段:\n"
            '  {"scores": {"accuracy": int, "completeness": int, "clarity": int},\n'
            '   "evidence": "答案中支撑你打分的关键句子(<=80 字)"}\n'
            "禁止输出任何 JSON 之外的文字。\n"
            "岗位: {{job_title}} ({{level}} 级别)\n"
        ),
    ),
    "interview.report": PromptSpec(
        name="面试总结推荐",
        description="面试完成后给推荐结论的系统提示词",
        variables=[],
        content=(
            "你是招聘官,基于多维度评分对候选人这场面试给出推荐结论。\n"
            '只输出 JSON: {"recommendation": "推荐进入下一轮 / 保留 / 不推荐", "comment": "<=120 字"}'
        ),
    ),
    "question.regen": PromptSpec(
        name="题库单题重生成",
        description="按 HR 修改要求对既有题目产出替换题",
        variables=[
            {"name": "job_clause", "desc": "岗位/方向信息段落(可空)"},
            {"name": "kind_label", "desc": "题型中文标签"},
            {"name": "difficulty_label", "desc": "难度中文标签"},
            {"name": "instruction", "desc": "HR 修改要求"},
        ],
        content=(
            "你是资深招聘面试官,任务是重新生成单题:基于下面这道面试题,产出 1 道替换题。\n"
            "{{job_clause}}"
            "题型: {{kind_label}}\n"
            "难度: {{difficulty_label}}\n"
            "HR 的修改要求: {{instruction}}\n\n"
            "只输出 JSON,格式:\n"
            '{"questions": [{\n'
            '  "question":      "新题干,中文,简洁",\n'
            '  "answer_points": ["要点1","要点2","要点3"],\n'
            '  "difficulty":    "初级" 或 "中级" 或 "高级" 或 "专家",\n'
            '  "follow_up":     "追问(可空字符串)"\n'
            "}]}\n"
            "约束: answer_points 至少 3 个,每条<=30字;题干不给参考答案;"
            "除非修改要求涉及难度,否则保持原难度;禁止输出 JSON 之外的文字。"
        ),
    ),
}


def render_vars(template: str, vars: dict[str, object]) -> str:
    """``{{var}}`` → ``str(value)``;未提供的变量原样保留。"""

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in vars:
            return str(vars[name])
        return m.group(0)

    return _VAR_RE.sub(_sub, template)


def render_prompt(
    db: Session, tenant_id: str, key: str, **vars: object
) -> str | None:
    """查租户的自定义模板并渲染;无模板 / 已停用 / key 未注册 → None。

    调用方对 ``None`` 一律回落到代码内置默认 prompt。
    """
    if key not in PROMPT_REGISTRY:
        return None
    tpl = db.scalars(
        select(PromptTemplate).where(
            PromptTemplate.tenant_id == tenant_id,
            PromptTemplate.key == key,
            PromptTemplate.is_active.is_(True),
        )
    ).first()
    if tpl is None:
        return None
    return render_vars(tpl.content, vars)


def vars_of(key: str) -> list[dict[str, str]]:
    """给 API/前端返回该 key 声明的变量文档。"""
    spec = PROMPT_REGISTRY.get(key)
    return spec.variables if spec else []
