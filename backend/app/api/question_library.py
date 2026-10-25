"""题库 API — 租户级题目池。题目是唯一实体,可独立存在或隶属某个题集。

端点:
- GET    /api/question-library                  分页列表,支持 q/kind/difficulty/category/skill/question_set_id/standalone 过滤
- GET    /api/question-library/categories       分类题集(同分类独立题目 >= 2 道自动成集)
- POST   /api/question-library/categories/rename  分类题集改名(批量改分类标签)
- POST   /api/question-library/categories/delete  删除分类题集(删其中全部独立题目)
- POST   /api/question-library                  手动创建单道题
- PUT    /api/question-library/{id}             编辑题目
- DELETE /api/question-library/{id}             删除题目
- POST   /api/question-library/{id}/regenerate  AI 重新生成单题(可带自定义修改要求)
- POST   /api/question-library/generate         AI 批量生成并存入题库

权限:viewer 只读,hr/admin 可增删改。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.domain.models import Candidate, Job, QuestionLibraryItem, QuestionSet, Resume, User
from app.infra.db import get_db
from app.integrations.llm.provider import LLMError, Message, call_llm_json
from app.services.question_bank import (
    CATEGORY_SET_MIN,
    DIFFICULTY_LABELS,
    KIND_LABELS,
    regenerate_item,
)

router = APIRouter(prefix="/question-library", tags=["question-library"])
logger = logging.getLogger(__name__)

KINDS = tuple(KIND_LABELS)
Kind = Literal["tech", "project", "scenario", "design", "soft"]


# ---- schemas ----


class LibraryItemOut(BaseModel):
    id: str
    question: str
    answer_points: list[str]
    kind: str
    difficulty: str
    category: str
    skill: str | None
    follow_up: str | None
    source: str
    use_count: int
    avg_score: float | None
    generated_from_job_id: str | None
    question_set_id: str | None = None
    # 所属题集的展示名:"候选人 · 岗位"
    question_set_label: str | None = None
    position: int = 0
    dimensions: list[str] = Field(default_factory=list)
    # 独立题目且同分类独立题 >= 2 道 → 属于以 category 命名的分类题集
    in_category_set: bool = False
    created_by: str | None
    created_at: datetime
    updated_at: datetime


class CategorySetOut(BaseModel):
    """分类题集:同一分类标签下的独立题目自动聚合而成(不落库)。"""

    category: str
    question_count: int
    kinds: list[str]
    last_updated_at: datetime


class CategoryRenameIn(BaseModel):
    old: str = Field(..., min_length=1, max_length=128)
    new: str = Field(..., min_length=1, max_length=128)


class CategoryDeleteIn(BaseModel):
    category: str = Field(..., min_length=1, max_length=128)


class CreateIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    answer_points: list[str] = Field(default_factory=list)
    kind: Kind = "tech"
    difficulty: Literal["initial", "intermediate", "advanced", "expert"] = "intermediate"
    category: str = Field("", max_length=128)
    skill: str | None = Field(None, max_length=128)
    follow_up: str | None = Field(None, max_length=500)


class UpdateIn(BaseModel):
    question: str | None = Field(None, min_length=2, max_length=2000)
    answer_points: list[str] | None = None
    kind: Kind | None = None
    difficulty: Literal["initial", "intermediate", "advanced", "expert"] | None = None
    category: str | None = Field(None, max_length=128)
    skill: str | None = Field(None, max_length=128)
    follow_up: str | None = Field(None, max_length=500)


class GenerateIn(BaseModel):
    job_id: str | None = None
    category: str = Field("", max_length=128)
    kind: Kind = "tech"
    difficulty: Literal["initial", "intermediate", "advanced", "expert"] = "intermediate"
    count: int = Field(5, ge=1, le=20)


class RegenerateIn(BaseModel):
    instruction: str = Field("", max_length=500, description="HR 对新题的修改要求,留空=换个角度")


# ---- helpers ----


def _get_in_tenant(db: Session, *, item_id: str, tenant_id: str) -> QuestionLibraryItem:
    item = db.get(QuestionLibraryItem, item_id)
    if not item or item.tenant_id != tenant_id:
        raise HTTPException(404, "题库条目不存在")
    return item


def set_labels(db: Session, set_ids: set[str]) -> dict[str, str]:
    """题集 id → "候选人 · 岗位" 展示名(批量,避免 N+1)。"""
    if not set_ids:
        return {}
    sets = db.scalars(select(QuestionSet).where(QuestionSet.id.in_(set_ids))).all()
    resumes = {
        r.id: r
        for r in db.scalars(select(Resume).where(Resume.id.in_({s.resume_id for s in sets}))).all()
    }
    cand_ids = {r.candidate_id for r in resumes.values()}
    names = {
        c.id: c.name for c in db.scalars(select(Candidate).where(Candidate.id.in_(cand_ids))).all()
    }
    job_ids = {s.job_id for s in sets if s.job_id}
    titles = {j.id: j.title for j in db.scalars(select(Job).where(Job.id.in_(job_ids))).all()}
    out: dict[str, str] = {}
    for s in sets:
        r = resumes.get(s.resume_id)
        name = names.get(r.candidate_id, "(已删除)") if r else "(已删除)"
        out[s.id] = f"{name} · {titles.get(s.job_id, '未指定岗位') if s.job_id else '未指定岗位'}"
    return out


def _standalone_in_tenant(tenant_id: str):
    return (
        QuestionLibraryItem.tenant_id == tenant_id,
        QuestionLibraryItem.question_set_id.is_(None),
    )


def _category_counts(db: Session, tenant_id: str, categories: set[str]) -> dict[str, int]:
    """各分类下的独立题目数。"""
    cats = {c for c in categories if c}
    if not cats:
        return {}
    rows = db.execute(
        select(QuestionLibraryItem.category, func.count())
        .where(*_standalone_in_tenant(tenant_id), QuestionLibraryItem.category.in_(cats))
        .group_by(QuestionLibraryItem.category)
    ).all()
    return {c: n for c, n in rows}


def items_to_out(db: Session, items: list[QuestionLibraryItem]) -> list[LibraryItemOut]:
    labels = set_labels(db, {i.question_set_id for i in items if i.question_set_id})
    standalone = [i for i in items if not i.question_set_id]
    counts = (
        _category_counts(db, standalone[0].tenant_id, {i.category for i in standalone})
        if standalone
        else {}
    )
    return [
        _to_out(
            i,
            labels.get(i.question_set_id or ""),
            in_category_set=not i.question_set_id
            and counts.get(i.category, 0) >= CATEGORY_SET_MIN,
        )
        for i in items
    ]


def _to_out(
    item: QuestionLibraryItem, set_label: str | None = None, *, in_category_set: bool = False
) -> LibraryItemOut:
    return LibraryItemOut(
        id=item.id,
        question=item.question,
        answer_points=list(item.answer_points or []),
        kind=item.kind,
        difficulty=item.difficulty,
        category=item.category or "",
        skill=item.skill,
        follow_up=item.follow_up,
        source=item.source,
        use_count=item.use_count,
        avg_score=item.avg_score,
        generated_from_job_id=item.generated_from_job_id,
        question_set_id=item.question_set_id,
        question_set_label=set_label,
        position=item.position or 0,
        dimensions=list(item.dimensions or []),
        in_category_set=in_category_set,
        created_by=item.created_by,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _generate_for_library(
    db: Session,
    *,
    tenant_id: str,
    job: Job | None,
    category: str,
    kind: str,
    difficulty: str,
    count: int,
) -> list[dict[str, Any]]:
    """调 LLM 生成题目,不依赖简历,按岗位/分类/题型出题。"""
    kind_label = KIND_LABELS.get(kind, kind)
    diff_label = DIFFICULTY_LABELS.get(difficulty, difficulty)
    job_clause = (
        f"目标岗位: {job.title} (技能要求: {', '.join(job.skills) if job.skills else '未填'})\n"
        if job
        else (f"岗位方向: {category}\n" if category else "通用面试题。\n")
    )
    system = (
        f"你是资深招聘面试官,请生成 {count} 道【{kind_label}】类型的面试题。\n"
        f"{job_clause}"
        f"难度等级: {diff_label}\n\n"
        "只输出 JSON,格式:\n"
        '{"questions": [\n'
        "  {\n"
        '    "question":      "题干,中文,简洁",\n'
        '    "answer_points": ["要点1","要点2","要点3"],\n'
        '    "follow_up":     "追问(可空字符串)"\n'
        "  }, ...\n"
        "]}\n"
        "约束: answer_points 至少 2 个,每条<=30字;禁止输出 JSON 之外的文字。"
    )
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"请生成 {count} 道题。"},
    ]
    try:
        parsed = call_llm_json(
            "question_library.generate",
            messages,
            db=db,
            tenant_id=tenant_id,
            required=("questions",),
            temperature=0.6,
        )
        qs = parsed.get("questions") or []
        if not isinstance(qs, list) or not qs:
            raise ValueError("questions 为空")
        return [
            {
                "question": str(q.get("question") or "").strip()[:1000],
                "answer_points": [str(p)[:200] for p in (q.get("answer_points") or [])][:8],
                "follow_up": str(q.get("follow_up") or "")[:300],
            }
            for q in qs
            if isinstance(q, dict) and q.get("question")
        ][:count]
    except (LLMError, ValueError, Exception) as e:  # noqa: BLE001
        logger.warning("题库 AI 生成失败: %s", e)
        return []


# ---- endpoints ----


@router.get("/", response_model=PageOut[LibraryItemOut])
def list_items(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    kind: str | None = None,
    difficulty: str | None = None,
    category: str | None = None,
    skill: str | None = None,
    question_set_id: str | None = None,
    standalone: bool | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> PageOut[LibraryItemOut]:
    """``standalone=true`` 只看独立题目;配合 ``category`` 即分类题集里的题。"""
    limit, offset, q = p
    stmt = select(QuestionLibraryItem).where(
        QuestionLibraryItem.tenant_id == current.tenant_id
    )
    if question_set_id:
        stmt = stmt.where(QuestionLibraryItem.question_set_id == question_set_id)
    if standalone is True:
        stmt = stmt.where(QuestionLibraryItem.question_set_id.is_(None))
    elif standalone is False:
        stmt = stmt.where(QuestionLibraryItem.question_set_id.is_not(None))
    if kind:
        stmt = stmt.where(QuestionLibraryItem.kind == kind)
    if difficulty:
        stmt = stmt.where(QuestionLibraryItem.difficulty == difficulty)
    if category:
        stmt = stmt.where(QuestionLibraryItem.category == category)
    if skill:
        stmt = stmt.where(QuestionLibraryItem.skill == skill)
    stmt = apply_q_ilike(stmt, q, QuestionLibraryItem.question, QuestionLibraryItem.category)

    total = count_total(db, stmt)
    order = (
        (QuestionLibraryItem.position,)
        if question_set_id
        else (QuestionLibraryItem.created_at.desc(), QuestionLibraryItem.position)
    )
    rows = db.scalars(stmt.order_by(*order).limit(limit).offset(offset)).all()
    return PageOut[LibraryItemOut](
        items=items_to_out(db, list(rows)),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/", response_model=LibraryItemOut, status_code=status.HTTP_201_CREATED)
def create_item(
    body: CreateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> LibraryItemOut:
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能创建题目")
    item = QuestionLibraryItem(
        tenant_id=current.tenant_id,
        question=body.question,
        answer_points=body.answer_points,
        kind=body.kind,
        difficulty=body.difficulty,
        category=body.category.strip(),
        skill=body.skill,
        follow_up=body.follow_up,
        source="manual",
        created_by=current.id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return items_to_out(db, [item])[0]


# ---- 分类题集(同分类独立题目 >= 2 道自动成集) ----


@router.get("/categories", response_model=list[CategorySetOut])
def list_categories(
    q: str | None = None,
    min_count: int = CATEGORY_SET_MIN,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[CategorySetOut]:
    """按分类标签聚合独立题目。默认只返回成集的(>= 2 道);``min_count=1`` 可拿全部分类做输入提示。"""
    stmt = (
        select(
            QuestionLibraryItem.category,
            func.count(),
            func.max(QuestionLibraryItem.updated_at),
        )
        .where(*_standalone_in_tenant(current.tenant_id), QuestionLibraryItem.category != "")
        .group_by(QuestionLibraryItem.category)
        .having(func.count() >= max(1, min_count))
    )
    if q and q.strip():
        stmt = stmt.where(QuestionLibraryItem.category.ilike(f"%{q.strip()}%"))
    rows = db.execute(stmt.order_by(func.max(QuestionLibraryItem.updated_at).desc())).all()
    kinds: dict[str, set[str]] = {}
    if rows:
        for cat, kind in db.execute(
            select(QuestionLibraryItem.category, QuestionLibraryItem.kind)
            .where(
                *_standalone_in_tenant(current.tenant_id),
                QuestionLibraryItem.category.in_([r[0] for r in rows]),
            )
            .distinct()
        ).all():
            kinds.setdefault(cat, set()).add(kind)
    return [
        CategorySetOut(
            category=cat,
            question_count=n,
            kinds=sorted(kinds.get(cat, set()), key=list(KIND_LABELS).index),
            last_updated_at=last,
        )
        for cat, n, last in rows
    ]


@router.post("/categories/rename")
def rename_category(
    body: CategoryRenameIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, int]:
    """把分类题集改名 = 批量改其中独立题目的分类;新名已存在时两组自然合并。"""
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能修改分类")
    new = body.new.strip()
    if not new:
        raise HTTPException(422, "分类名不能为空")
    res = db.execute(
        update(QuestionLibraryItem)
        .where(*_standalone_in_tenant(current.tenant_id), QuestionLibraryItem.category == body.old)
        .values(category=new, updated_at=datetime.utcnow())
    )
    db.commit()
    return {"updated": res.rowcount or 0}


@router.post("/categories/delete")
def delete_category(
    body: CategoryDeleteIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, int]:
    """删除分类题集 = 删除该分类下的全部独立题目(简历题集里的题不受影响)。"""
    if current.role == "viewer":
        raise HTTPException(403, "viewer 不能删题目")
    res = db.execute(
        delete(QuestionLibraryItem).where(
            *_standalone_in_tenant(current.tenant_id),
            QuestionLibraryItem.category == body.category,
        )
    )
    db.commit()
    return {"deleted": res.rowcount or 0}


@router.put("/{item_id}", response_model=LibraryItemOut)
def update_item(
    item_id: str,
    body: UpdateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> LibraryItemOut:
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能编辑题目")
    item = _get_in_tenant(db, item_id=item_id, tenant_id=current.tenant_id)
    if body.question is not None:
        item.question = body.question
    if body.answer_points is not None:
        item.answer_points = body.answer_points
    if body.kind is not None:
        item.kind = body.kind
    if body.difficulty is not None:
        item.difficulty = body.difficulty
    if body.category is not None:
        item.category = body.category.strip()
    if body.skill is not None:
        item.skill = body.skill
    if body.follow_up is not None:
        item.follow_up = body.follow_up
    item.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(item)
    return items_to_out(db, [item])[0]


@router.post("/{item_id}/regenerate", response_model=LibraryItemOut)
def regenerate(
    item_id: str,
    body: RegenerateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> LibraryItemOut:
    """AI 重新生成单题,就地替换内容(所属题集与顺序不变)。"""
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能重新生成题目")
    item = _get_in_tenant(db, item_id=item_id, tenant_id=current.tenant_id)
    try:
        regenerate_item(db, item, body.instruction)
    except LLMError as e:
        db.rollback()
        logger.warning("单题重新生成失败 item=%s err=%s", item_id, e)
        raise HTTPException(502, "AI 重新生成失败,请稍后重试或检查 LLM 配置") from e
    db.commit()
    db.refresh(item)
    return items_to_out(db, [item])[0]


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(
    item_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    if current.role == "viewer":
        raise HTTPException(403, "viewer 不能删题目")
    item = _get_in_tenant(db, item_id=item_id, tenant_id=current.tenant_id)
    db.delete(item)
    db.commit()


@router.post("/generate", response_model=list[LibraryItemOut], status_code=status.HTTP_201_CREATED)
def generate_and_save(
    body: GenerateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[LibraryItemOut]:
    """AI 生成题目并直接存入题库(同步,不超过 20 题)。"""
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能生成题目")
    job: Job | None = None
    if body.job_id:
        job = db.get(Job, body.job_id)
        if not job or job.tenant_id != current.tenant_id:
            raise HTTPException(404, "岗位不存在")

    qs = _generate_for_library(
        db,
        tenant_id=current.tenant_id,
        job=job,
        category=body.category,
        kind=body.kind,
        difficulty=body.difficulty,
        count=body.count,
    )
    if not qs:
        raise HTTPException(502, "AI 生成失败,请稍后重试或检查 LLM 配置")

    now = datetime.utcnow()
    items: list[QuestionLibraryItem] = []
    for q in qs:
        item = QuestionLibraryItem(
            tenant_id=current.tenant_id,
            question=q["question"],
            answer_points=q["answer_points"],
            kind=body.kind,
            difficulty=body.difficulty,
            category=job.title if job else body.category.strip(),
            follow_up=q.get("follow_up") or None,
            source="ai_generated",
            generated_from_job_id=body.job_id,
            created_by=current.id,
            updated_at=now,
        )
        db.add(item)
        items.append(item)
    db.commit()
    for i in items:
        db.refresh(i)
    return [_to_out(i) for i in items]
