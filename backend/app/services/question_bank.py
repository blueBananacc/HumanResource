"""题库领域服务:题目是唯一实体,题集只是题目的集合。

题集有两种:
- 简历题集:从简历 AI 生成,对应 :class:`QuestionSet` 记录,题目通过 ``question_set_id`` 挂靠
- 分类题集:独立题目(``question_set_id`` 为空)按分类标签自动聚合,同分类 >= 2 道即成集,
  不落库,随题目增删 / 改分类自然出现或消失

- :func:`replace_set_questions` 出题 / 整集重新生成后,把 LLM 结果落为题集下的题目
- :func:`regenerate_item` 单题重新生成(可带 HR 自定义的修改要求)
"""
from __future__ import annotations

import logging
import zlib
from datetime import datetime
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.domain.models import Candidate, Job, QuestionLibraryItem, QuestionSet, Resume
from app.integrations.llm.provider import LLMError, Message, call_llm_json
from app.services.prompting import DATA_LABEL, guarded, render_prompt

logger = logging.getLogger(__name__)

#: 同一分类标签下独立题目达到该数量即自动构成分类题集
CATEGORY_SET_MIN = 2

KIND_LABELS = {
    "tech": "技术深度",
    "project": "项目复盘",
    "scenario": "场景排查",
    "design": "系统设计",
    "soft": "软技能",
}
DIFFICULTY_LABELS = {
    "initial": "初级",
    "intermediate": "中级",
    "advanced": "高级",
    "expert": "专家",
}
_DIFF_FROM_LABEL = {v: k for k, v in DIFFICULTY_LABELS.items()}
_KIND_FROM_LABEL = {v: k for k, v in KIND_LABELS.items()}


def kind_from_dimensions(dims: list[str]) -> str:
    """出题维度(中文标签) → 题型码;识别不了归为技术深度。"""
    for d in dims:
        if d in _KIND_FROM_LABEL:
            return _KIND_FROM_LABEL[d]
        low = d.lower()
        if "设计" in d or "design" in low:
            return "design"
        if "项目" in d or "project" in low:
            return "project"
        if "场景" in d or "排查" in d or "scenario" in low:
            return "scenario"
        if "软" in d or "沟通" in d or "soft" in low:
            return "soft"
        if "技术" in d or "tech" in low:
            return "tech"
    return "tech"


def difficulty_code(raw: str | None, default: str = "intermediate") -> str:
    """兼容中文标签("中级")与码值("intermediate")。"""
    if not raw:
        return default
    if raw in DIFFICULTY_LABELS:
        return raw
    return _DIFF_FROM_LABEL.get(raw, default)


def set_category(db: Session, qs: QuestionSet) -> str:
    """简历题集里题目的分类标签:"岗位名 · 候选人姓名";未指定岗位时只用姓名。"""
    job = db.get(Job, qs.job_id) if qs.job_id else None
    resume = db.get(Resume, qs.resume_id)
    cand = db.get(Candidate, resume.candidate_id) if resume else None
    parts = [p for p in (job.title if job else None, cand.name if cand else None) if p]
    return " · ".join(parts)[:128]


def replace_set_questions(
    db: Session, qs: QuestionSet, questions: list[dict[str, Any]]
) -> list[QuestionLibraryItem]:
    """清掉题集原有题目,按 LLM 结果重建(不 commit)。"""
    db.execute(delete(QuestionLibraryItem).where(QuestionLibraryItem.question_set_id == qs.id))
    category = set_category(db, qs)
    now = datetime.utcnow()
    items: list[QuestionLibraryItem] = []
    for pos, q in enumerate(questions):
        if not q.get("question"):
            continue
        dims = [str(d) for d in (q.get("dimensions") or [])]
        item = QuestionLibraryItem(
            tenant_id=qs.tenant_id,
            question=q["question"],
            answer_points=list(q.get("answer_points") or []),
            kind=kind_from_dimensions(dims),
            difficulty=difficulty_code(q.get("difficulty"), qs.level),
            category=category,
            follow_up=q.get("follow_up") or None,
            source="ai_generated",
            generated_from_job_id=qs.job_id,
            question_set_id=qs.id,
            position=pos,
            dimensions=dims,
            created_by=qs.created_by,
            updated_at=now,
        )
        db.add(item)
        items.append(item)
    return items


def _regen_prompt(item: QuestionLibraryItem, *, job: Job | None, instruction: str) -> str:
    # 简历题集的分类是"岗位 · 姓名",不能当岗位方向用
    direction = item.category if item.category and not item.question_set_id else ""
    job_clause = (
        f"目标岗位: {job.title} (技能要求: {', '.join(job.skills) if job.skills else '未填'})\n"
        if job
        else (f"岗位方向: {direction}\n" if direction else "")
    )
    return (
        "你是资深招聘面试官,任务是重新生成单题:基于下面这道面试题,产出 1 道替换题。\n"
        f"{job_clause}"
        f"题型: {KIND_LABELS.get(item.kind, item.kind)}\n"
        f"难度: {DIFFICULTY_LABELS.get(item.difficulty, item.difficulty)}\n"
        f"HR 的修改要求: {instruction or '换一个角度考察同一能力,不要与原题雷同'}\n\n"
        "只输出 JSON,格式:\n"
        '{"questions": [{\n'
        '  "question":      "新题干,中文,简洁",\n'
        '  "answer_points": ["要点1","要点2","要点3"],\n'
        '  "difficulty":    "初级" 或 "中级" 或 "高级" 或 "专家",\n'
        '  "follow_up":     "追问(可空字符串)"\n'
        "}]}\n"
        "约束: answer_points 至少 3 个,每条<=30字;题干不给参考答案;"
        "除非修改要求涉及难度,否则保持原难度;禁止输出 JSON 之外的文字。"
    )


def regenerate_item(db: Session, item: QuestionLibraryItem, instruction: str = "") -> None:
    """按修改要求就地替换题目内容(id / 所属题集 / 顺序不变,不 commit)。

    LLM 失败抛 :class:`LLMError`,由 API 层转 502。
    """
    job = db.get(Job, item.generated_from_job_id) if item.generated_from_job_id else None
    user_parts = [
        f"原题: {item.question}",
        f"原答题要点: {' / '.join(item.answer_points or [])}",
    ]
    if item.follow_up:
        user_parts.append(f"原追问: {item.follow_up}")
    # 题集里的题带上候选人简历,让新题仍贴合这个人
    if item.question_set_id:
        qs = db.get(QuestionSet, item.question_set_id)
        resume = db.get(Resume, qs.resume_id) if qs else None
        if resume and resume.parsed_text:
            user_parts.append(f"候选人简历摘录{DATA_LABEL}:\n{resume.parsed_text[:2000]}")
    user_parts.append(f"请求编号: {zlib.crc32((item.question + instruction).encode())}")

    regen_sys = render_prompt(
        db,
        item.tenant_id,
        "question.regen",
        job_clause=(
            f"目标岗位: {job.title} (技能要求: {', '.join(job.skills) if job.skills else '未填'})\n"
            if job
            else (
                f"岗位方向: {item.category}\n"
                if item.category and not item.question_set_id
                else ""
            )
        ),
        kind_label=KIND_LABELS.get(item.kind, item.kind),
        difficulty_label=DIFFICULTY_LABELS.get(item.difficulty, item.difficulty),
        instruction=instruction.strip() or "换一个角度考察同一能力,不要与原题雷同",
    ) or _regen_prompt(item, job=job, instruction=instruction.strip())
    messages: list[Message] = [
        {"role": "system", "content": guarded(regen_sys)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    try:
        parsed = call_llm_json(
            "question_bank.generate",
            messages,
            db=db,
            tenant_id=item.tenant_id,
            temperature=0.7,
            prompt_key="question_bank.regen",
        )
        q = (parsed.get("questions") or [parsed])[0]
        question = str(q.get("question") or "").strip()[:1000]
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"AI 返回格式异常: {e}") from e
    if not question:
        raise LLMError("AI 未返回题干")

    points = q.get("answer_points") or []
    if isinstance(points, str):
        points = [points]
    item.question = question
    item.answer_points = [str(p)[:200] for p in points][:8]
    item.follow_up = str(q.get("follow_up") or "")[:300] or None
    item.difficulty = difficulty_code(q.get("difficulty"), item.difficulty)
    item.source = "ai_generated"
    item.updated_at = datetime.utcnow()
