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
    """注册表条目:key / 展示名 / 默认内容 / 变量文档 / 归属模块。

    ``module`` 是展示分组 + ``render_prompt(module=...)`` 两级回落的依据:
    租户可在 ``prompt_templates.scope_module`` 里按模块覆盖同 key 的措辞
    (模块覆盖 → 全局两级回落)。
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        content: str,
        variables: list[dict[str, str]],
        module: str = "global",
    ) -> None:
        self.name = name
        self.description = description
        self.content = content
        self.variables = variables
        self.module = module


PROMPT_REGISTRY: dict[str, PromptSpec] = {
    "match.evaluate": PromptSpec(
        name="简历-岗位匹配评估",
        description="系统提示词:评估简历与 JD 的匹配度,输出 score/strengths/gaps/comment JSON",
        module="talents",
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
        module="interviews",
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
        module="interviews",
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
        module="interviews",
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
        module="interviews",
        variables=[],
        content=(
            "你是招聘官,基于多维度评分对候选人这场面试给出推荐结论。\n"
            '只输出 JSON: {"recommendation": "推荐进入下一轮 / 保留 / 不推荐", "comment": "<=120 字"}'
        ),
    ),
    "question.regen": PromptSpec(
        name="题库单题重生成",
        description="按 HR 修改要求对既有题目产出替换题",
        module="interviews",
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
    # ===== T6-1 字段级 AI 助手(field.<target>.<intent>,产出纯文本非 JSON)=====
    "field.jobs.description.draft": PromptSpec(
        name="JD 起草",
        description="按岗位标题/等级/技能起草岗位描述",
        module="jobs",
        variables=[
            {"name": "title", "desc": "岗位名称"},
            {"name": "level", "desc": "等级"},
            {"name": "skills", "desc": "技能要求"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是资深招聘 HR,为下面岗位起草一段**岗位描述(JD)**。\n"
            "岗位: {{title}}\n等级: {{level}}\n技能要求: {{skills}}\n"
            "补充要求: {{instruction}}\n\n"
            "输出要求:\n"
            "- 直接给 JD 正文,不要标题、不要解释\n"
            "- 结构:职责(3-5 条)、任职要求(3-5 条)、加分项(可选 2 条)\n"
            "- 语气专业、面向候选人、避免歧视性表述\n"
        ),
    ),
    "field.jobs.description.polish": PromptSpec(
        name="JD 润色",
        description="润色已有岗位描述,提升专业度与吸引力",
        module="jobs",
        variables=[
            {"name": "title", "desc": "岗位名称"},
            {"name": "current_text", "desc": "当前 JD 内容"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是资深招聘 HR,润色下面这段岗位描述,让它更专业、对候选人更有吸引力。\n"
            "岗位: {{title}}\n补充要求: {{instruction}}\n\n"
            "当前内容:\n{{current_text}}\n\n"
            "输出要求:只给润色后的正文,保留原有信息,不要加解释。\n"
        ),
    ),
    "field.announcements.content.draft": PromptSpec(
        name="公告起草",
        description="按标题与要求起草公司公告正文",
        module="announcements",
        variables=[
            {"name": "title", "desc": "公告标题"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是企业 HR,起草一条公司公告正文。\n"
            "标题: {{title}}\n要求: {{instruction}}\n\n"
            "输出要求:正式但亲和的公司内部公告口吻,只给正文,2-4 段,不加标题。\n"
        ),
    ),
    "field.announcements.content.polish": PromptSpec(
        name="公告润色",
        description="润色公司公告正文",
        module="announcements",
        variables=[
            {"name": "title", "desc": "公告标题"},
            {"name": "current_text", "desc": "当前公告内容"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是企业 HR,润色下面这条公司公告正文,使其更清晰得体。\n"
            "标题: {{title}}\n补充要求: {{instruction}}\n\n"
            "当前内容:\n{{current_text}}\n\n输出要求:只给润色后的正文,不加解释。\n"
        ),
    ),
    "field.cases.resolution.draft": PromptSpec(
        name="工单解决方案起草",
        description="按工单主题/类型/描述起草解决方案回复",
        module="cases",
        variables=[
            {"name": "subject", "desc": "工单主题"},
            {"name": "kind", "desc": "工单类型"},
            {"name": "description", "desc": "工单描述"},
        ],
        content=(
            "你是 HR 服务台专员,为下面这条员工工单起草**解决方案/答复**。\n"
            "主题: {{subject}}\n类型: {{kind}}\n描述: {{description}}\n\n"
            "输出要求:\n"
            "- 先共情一句,再给出清晰可执行的解决步骤或答复结论\n"
            "- 只给答复正文,语气专业、面向员工\n"
            "- 信息不足时给出合理通用答复并提示补充材料\n"
        ),
    ),
    "field.campaigns.subject.draft": PromptSpec(
        name="外联邮件主题起草",
        description="起草人才外联活动邮件主题行",
        module="campaigns",
        variables=[
            {"name": "name", "desc": "活动名称"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是招聘外联专家,起草人才再激活邮件的**主题行**。\n"
            "活动: {{name}}\n要求: {{instruction}}\n\n"
            "输出要求:只给一行主题,可用 {{candidate_name}} 占位符,20 字以内,有点击欲。\n"
        ),
    ),
    "field.campaigns.body.draft": PromptSpec(
        name="外联邮件正文起草",
        description="起草人才外联活动邮件正文",
        module="campaigns",
        variables=[
            {"name": "name", "desc": "活动名称"},
            {"name": "instruction", "desc": "HR 补充要求(可空)"},
        ],
        content=(
            "你是招聘外联专家,起草人才再激活邮件正文。\n"
            "活动: {{name}}\n要求: {{instruction}}\n\n"
            "输出要求:\n"
            "- 只给邮件正文,3-5 句,真诚不套路\n"
            "- 保留 {{candidate_name}} 占位符供发送时替换\n"
            "- 结尾给出明确行动号召(回复/更新简历/看机会)\n"
        ),
    ),
    "field.generic.draft": PromptSpec(
        name="通用起草",
        description="按上下文与要求起草一段文本",
        module="global",
        variables=[
            {"name": "instruction", "desc": "起草要求"},
            {"name": "context", "desc": "上下文信息"},
        ],
        content=(
            "你是企业 HR 写作助手,按要求起草文本。\n"
            "上下文: {{context}}\n要求: {{instruction}}\n\n"
            "输出要求:只给正文,不加解释、不加标题,中文。\n"
        ),
    ),
    "field.generic.polish": PromptSpec(
        name="通用润色",
        description="润色文本,保持原意提升表达",
        module="global",
        variables=[
            {"name": "current_text", "desc": "当前文本"},
            {"name": "instruction", "desc": "补充要求(可空)"},
        ],
        content=(
            "你是企业 HR 写作助手,润色下面这段文本,使其更专业清晰,保持原意。\n"
            "补充要求: {{instruction}}\n\n当前文本:\n{{current_text}}\n\n"
            "输出要求:只给润色后的文本。\n"
        ),
    ),
    "field.generic.expand": PromptSpec(
        name="通用扩写",
        description="扩写文本,补充细节",
        module="global",
        variables=[
            {"name": "current_text", "desc": "当前文本"},
            {"name": "instruction", "desc": "补充要求(可空)"},
        ],
        content=(
            "你是企业 HR 写作助手,扩写下面这段文本,补充合理细节,保持原意与风格。\n"
            "补充要求: {{instruction}}\n\n当前文本:\n{{current_text}}\n\n"
            "输出要求:只给扩写后的文本,篇幅约为原文 1.5-2 倍。\n"
        ),
    ),
    "field.generic.shorten": PromptSpec(
        name="通用精简",
        description="精简文本,保留要点",
        module="global",
        variables=[
            {"name": "current_text", "desc": "当前文本"},
            {"name": "instruction", "desc": "补充要求(可空)"},
        ],
        content=(
            "你是企业 HR 写作助手,精简下面这段文本,保留全部要点,去除冗余。\n"
            "补充要求: {{instruction}}\n\n当前文本:\n{{current_text}}\n\n"
            "输出要求:只给精简后的文本,篇幅约为原文一半。\n"
        ),
    ),
    # ===== T6-2 记录级 AI 摘要(非 JSON 输出,markdown 文本)=====
    "summarize.candidate": PromptSpec(
        name="候选人 360° 摘要",
        description="聚合简历/匹配/面试/备注,生成候选人一页纸摘要",
        module="talents",
        variables=[{"name": "context", "desc": "候选人聚合数据 JSON"}],
        content=(
            "你是资深招聘顾问,根据下面的候选人聚合数据生成**360° 摘要**。\n"
            "数据(JSON):\n{{context}}\n\n"
            "输出格式(markdown):\n"
            "- **一句话画像**:资历 + 方向 + 当前状态\n"
            "- **亮点**:3-5 条(技能/经历/匹配高分)\n"
            "- **风险/待确认**:0-3 条(短板/停滞/信息缺口)\n"
            "- **建议下一步**:1-2 条可执行动作\n"
            "要求:只基于给定数据,不编造;数据缺的部分明确写'暂无'。\n"
        ),
    ),
    "summarize.employee": PromptSpec(
        name="员工档案摘要",
        description="聚合档案/申请/工单/目标,生成员工概况摘要",
        module="hris",
        variables=[{"name": "context", "desc": "员工聚合数据 JSON"}],
        content=(
            "你是 HRBP,根据下面的员工聚合数据生成**员工概况摘要**。\n"
            "数据(JSON):\n{{context}}\n\n"
            "输出格式(markdown):\n"
            "- **概况**:岗位/司龄/状态一句话\n"
            "- **在途事项**:申请/工单/请假等待处理项\n"
            "- **目标进展**:goal 状态概览\n"
            "- **关注建议**:0-2 条\n"
            "要求:只基于给定数据,不编造;数据缺的部分明确写'暂无'。\n"
        ),
    ),
    "summarize.case": PromptSpec(
        name="工单摘要",
        description="聚合工单字段与留言,生成处理进展摘要",
        module="cases",
        variables=[{"name": "context", "desc": "工单聚合数据 JSON"}],
        content=(
            "你是 HR 服务台主管,根据下面的工单数据生成**处理进展摘要**。\n"
            "数据(JSON):\n{{context}}\n\n"
            "输出格式(markdown):\n"
            "- **诉求**:一句话概括员工要什么\n"
            "- **进展**:当前状态 + 已采取动作\n"
            "- **建议下一步**:1-2 条\n"
            "要求:只基于给定数据,不编造;留言为空时如实说明。\n"
        ),
    ),
    # ===== T6-4 文档起草 =====
    "doc.draft": PromptSpec(
        name="文档起草",
        description="按模板+记录上下文起草正式 HR 文档(offer/证明等)",
        module="global",
        variables=[
            {"name": "template", "desc": "已预填变量的模板正文"},
            {"name": "context", "desc": "记录上下文 JSON"},
            {"name": "instruction", "desc": "HR 附加要求(可空)"},
        ],
        content=(
            "你是企业 HR 文书专家,根据下面的**文档模板**完成一份正式文档。\n"
            "模板(已填入变量,残余 {{var}} 需你依据上下文补全):\n{{template}}\n\n"
            "记录上下文(JSON):\n{{context}}\n\n"
            "附加要求: {{instruction}}\n\n"
            "输出要求:\n"
            "- 只给文档正文,不加解释\n"
            "- 语气正式规范,符合中文 HR 文书习惯\n"
            "- 模板中残余的 {{var}} 必须全部替换为实际值;上下文没有的信息\n"
            "  用合理通用表述(如'公司规定日期'),不要留下 {{}} 占位符\n"
        ),
    ),
    "import.column_map": PromptSpec(
        name="导入列智能映射",
        description="AI 导入向导:把上传文件的列名映射到目标字段",
        module="imports",
        variables=[
            {"name": "dataset", "desc": "目标数据集名(如 员工)"},
            {"name": "fields", "desc": "目标字段 spec JSON(key/label/aliases/kind/sample)"},
            {"name": "columns", "desc": "文件列名+样本值 JSON"},
        ],
        content=(
            "你是数据导入映射助手。目标数据集「{{dataset}}」的字段 spec:\n"
            "{{fields}}\n\n"
            "上传文件的列与样本值:\n"
            "{{columns}}\n\n"
            "为每个目标字段选择最合适的文件列:看列名语义,也看样本值形态\n"
            "(邮箱样的是邮箱列、日期样的是日期列)。同一文件列最多映射一个\n"
            "字段;拿不准就留空字符串,不要硬猜。\n\n"
            "**只输出 JSON**:\n"
            '{"mapping": {"<field_key>": "<文件列名 或 \\"\\">"},\n'
            ' "confidence": {"<field_key>": <0-1>},\n'
            ' "reasons": {"<field_key>": "一句话理由"},\n'
            ' "warnings": ["表头疑似缺失/数据疑点,<=3 条"]}\n'
            "禁止输出 JSON 之外的任何文字。"
        ),
    ),
    "import.check_note": PromptSpec(
        name="导入错误归因摘要",
        description="AI 导入向导:对校验错误做一句中文归因与修复建议",
        module="imports",
        variables=[
            {"name": "dataset", "desc": "目标数据集名"},
            {"name": "errors", "desc": "按字段聚合的错误统计 JSON"},
        ],
        content=(
            "你是数据导入质检助手。「{{dataset}}」导入校验发现如下按字段\n"
            "聚合的错误:\n{{errors}}\n\n"
            "用一句中文(<=120 字)概括主要原因 + 一条最优先的修复建议。\n"
            "直接输出这句话,不要前缀、不要 JSON。"
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
    db: Session, tenant_id: str, key: str, module: str | None = None, **vars: object
) -> str | None:
    """查租户的自定义模板并渲染;无模板 / 已停用 / key 未注册 → None。

    ``module`` 非空时先查模块级覆盖 ``(key, scope_module=module)``,
    无则回落全局 ``(key, "")``。
    调用方对 ``None`` 一律回落到代码内置默认 prompt。
    """
    if key not in PROMPT_REGISTRY:
        return None
    scopes = [module, ""] if module else [""]
    for scope in scopes:
        tpl = db.scalars(
            select(PromptTemplate).where(
                PromptTemplate.tenant_id == tenant_id,
                PromptTemplate.key == key,
                PromptTemplate.scope_module == scope,
                PromptTemplate.is_active.is_(True),
            )
        ).first()
        if tpl is not None:
            return render_vars(tpl.content, vars)
    return None


def vars_of(key: str) -> list[dict[str, str]]:
    """给 API/前端返回该 key 声明的变量文档。"""
    spec = PROMPT_REGISTRY.get(key)
    return spec.variables if spec else []
