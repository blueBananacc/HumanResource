"""面试题集 API。题集只是题目的集合,题目本身是题库条目(question_set_id 指回题集)。

端点:
- POST   /api/question-sets             创建(异步入队)
- GET    /api/question-sets             列出当前租户所有题集(可按 resume_id 过滤)
- GET    /api/question-sets/{id}        详情(含题目列表,即该题集下的题库条目)
- DELETE /api/question-sets/{id}        删除(连同题集下的题目)
- POST   /api/question-sets/{id}/regen  重新生成(同 resume / job / level / count / kinds)

单题的编辑 / 删除 / 重新生成走题库接口 ``/api/question-library/{id}``。

权限:任何登录用户可创建 / 看自己租户的题集(viewer 只读,hr/admin 可创建)。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.question_library import LibraryItemOut, items_to_out
from app.domain.models import Candidate, Job, QuestionLibraryItem, QuestionSet, Resume, User
from app.infra.db import get_db
from app.services.question_generator import KINDS

router = APIRouter(prefix="/question-sets", tags=["question-sets"])
logger = logging.getLogger(__name__)


# ---- schemas ----


class QuestionSetSummary(BaseModel):
    """列表用,不返回 questions(节省传输)。"""

    id: str
    resume_id: str
    job_id: str | None
    level: str
    count: int
    kinds: list[str]
    status: str
    error: str | None
    created_at: datetime
    finished_at: datetime | None
    # 列表上常用的衍生字段:简历文件名 + 候选人名 + 岗位标题
    resume_file_name: str
    candidate_name: str
    job_title: str | None
    # 题集当前实际题数(单题删除后会少于 count;count 是出题时请求的题量)
    question_count: int = 0


class QuestionSetDetail(QuestionSetSummary):
    questions: list[LibraryItemOut] | None = None
    started_at: datetime | None = None


class CreateIn(BaseModel):
    resume_id: str
    job_id: str | None = None
    level: Literal["initial", "intermediate", "advanced", "expert"] = "intermediate"
    count: int = Field(5, ge=1, le=20)
    kinds: list[str] = Field(default_factory=list)


# ---- helpers ----


def _validate_kinds(kinds: list[str]) -> list[str]:
    """KINDS 里的元素才合法,顺序保留,去重。"""
    seen: set[str] = set()
    out: list[str] = []
    for k in kinds:
        if k in KINDS and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _get_in_tenant(db: Session, *, set_id: str, tenant_id: str) -> QuestionSet:
    qs = db.get(QuestionSet, set_id)
    if not qs or qs.tenant_id != tenant_id:
        raise HTTPException(404, "题集不存在")
    return qs


def _set_items(db: Session, set_id: str) -> list[QuestionLibraryItem]:
    return list(
        db.scalars(
            select(QuestionLibraryItem)
            .where(QuestionLibraryItem.question_set_id == set_id)
            .order_by(QuestionLibraryItem.position, QuestionLibraryItem.created_at)
        ).all()
    )


def _question_counts(db: Session, set_ids: list[str]) -> dict[str, int]:
    if not set_ids:
        return {}
    rows = db.execute(
        select(QuestionLibraryItem.question_set_id, func.count())
        .where(QuestionLibraryItem.question_set_id.in_(set_ids))
        .group_by(QuestionLibraryItem.question_set_id)
    ).all()
    return {sid: n for sid, n in rows}


def _summary(db: Session, qs: QuestionSet, question_count: int | None = None) -> QuestionSetSummary:
    resume = db.get(Resume, qs.resume_id)
    cand_name = "(已删除)"
    file_name = "(已删除)"
    if resume:
        file_name = resume.file_name
        c = db.get(Candidate, resume.candidate_id)
        if c:
            cand_name = c.name
    job_title: str | None = None
    if qs.job_id:
        j = db.get(Job, qs.job_id)
        if j:
            job_title = j.title

    return QuestionSetSummary(
        id=qs.id,
        resume_id=qs.resume_id,
        job_id=qs.job_id,
        level=qs.level,
        count=qs.count,
        kinds=list(qs.kinds or []),
        status=qs.status,
        error=qs.error,
        created_at=qs.created_at,
        finished_at=qs.finished_at,
        resume_file_name=file_name,
        candidate_name=cand_name,
        job_title=job_title,
        question_count=(
            question_count
            if question_count is not None
            else _question_counts(db, [qs.id]).get(qs.id, 0)
        ),
    )


def _detail(db: Session, qs: QuestionSet) -> QuestionSetDetail:
    items = _set_items(db, qs.id)
    s = _summary(db, qs, len(items))
    return QuestionSetDetail(
        **s.model_dump(),
        questions=items_to_out(db, items) if items else None,
        started_at=qs.started_at,
    )


def _enqueue_or_fallback(set_id: str) -> None:
    """入队,失败时同步降级。"""
    try:
        from app.workers.tasks.questions import generate_question_set_task

        generate_question_set_task.delay(set_id)
        logger.info("出题任务入队 set=%s", set_id)
        return
    except Exception as e:  # noqa: BLE001
        logger.warning("Celery broker 不可达, 降级同步出题 set=%s err=%s", set_id, e)
    # 同步路径:直接调任务函数
    from app.workers.tasks.questions import generate_question_set_task

    generate_question_set_task(set_id)  # type: ignore[call-arg]


# ---- endpoints ----


@router.post(
    "/", response_model=QuestionSetDetail, status_code=status.HTTP_201_CREATED
)
def create(
    body: CreateIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> QuestionSetDetail:
    if current.role == "viewer":
        raise HTTPException(403, "viewer 角色不能创建题集")
    resume = db.get(Resume, body.resume_id)
    if not resume or resume.tenant_id != current.tenant_id:
        raise HTTPException(404, "简历不存在")
    if body.job_id:
        job = db.get(Job, body.job_id)
        if not job or job.tenant_id != current.tenant_id:
            raise HTTPException(404, "岗位不存在")

    qs = QuestionSet(
        tenant_id=current.tenant_id,
        resume_id=resume.id,
        job_id=body.job_id,
        level=body.level,
        count=body.count,
        kinds=_validate_kinds(body.kinds),
        status="pending",
        created_by=current.id,
    )
    db.add(qs)
    db.commit()
    db.refresh(qs)

    _enqueue_or_fallback(qs.id)
    db.refresh(qs)
    return _detail(db, qs)


@router.get("/", response_model=PageOut[QuestionSetSummary])
def list_sets(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    resume_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> PageOut[QuestionSetSummary]:
    """列出本租户题集(分页)。

    保留 ``?resume_id=`` 过滤,简历详情抽屉里查"该简历的题集"会用到。
    ``q`` 命中候选人名 / 简历文件名 / 岗位标题(题集本身没有"题目正文"
    搜索的需求,题目内容在详情页里)。

    搜索时把 Resume / Candidate / Job 都 outerjoin —— 用 outer 是因为
    岗位允许为空(``job_id`` 可空),resume/candidate 理论上不应缺但
    数据上历史也可能有孤儿,outer 更安全。
    """
    limit, offset, q = p
    stmt = (
        select(QuestionSet)
        .outerjoin(Resume, Resume.id == QuestionSet.resume_id)
        .outerjoin(Candidate, Candidate.id == Resume.candidate_id)
        .outerjoin(Job, Job.id == QuestionSet.job_id)
        .where(QuestionSet.tenant_id == current.tenant_id)
    )
    if resume_id:
        stmt = stmt.where(QuestionSet.resume_id == resume_id)
    stmt = apply_q_ilike(stmt, q, Candidate.name, Resume.file_name, Job.title)

    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(QuestionSet.created_at.desc()).limit(limit).offset(offset)
    ).all()
    counts = _question_counts(db, [r.id for r in rows])
    return PageOut[QuestionSetSummary](
        items=[_summary(db, r, counts.get(r.id, 0)) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{set_id}", response_model=QuestionSetDetail)
def get_one(
    set_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> QuestionSetDetail:
    qs = _get_in_tenant(db, set_id=set_id, tenant_id=current.tenant_id)
    return _detail(db, qs)


@router.delete("/{set_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(
    set_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    if current.role == "viewer":
        raise HTTPException(403, "viewer 不能删题集")
    qs = _get_in_tenant(db, set_id=set_id, tenant_id=current.tenant_id)
    # 显式删题集下的题目(不依赖数据库层 FK 级联是否开启)
    db.execute(sa_delete(QuestionLibraryItem).where(QuestionLibraryItem.question_set_id == qs.id))
    db.delete(qs)
    db.commit()


@router.post("/{set_id}/regen", response_model=QuestionSetDetail)
def regenerate(
    set_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> QuestionSetDetail:
    """重置成 pending + 清空题集下的题目 + 重新入队跑同样配置。"""
    if current.role == "viewer":
        raise HTTPException(403, "viewer 不能重新生成")
    qs = _get_in_tenant(db, set_id=set_id, tenant_id=current.tenant_id)
    db.execute(sa_delete(QuestionLibraryItem).where(QuestionLibraryItem.question_set_id == qs.id))
    qs.status = "pending"
    qs.error = None
    qs.started_at = None
    qs.finished_at = None
    db.commit()
    _enqueue_or_fallback(qs.id)
    db.refresh(qs)
    return _detail(db, qs)
