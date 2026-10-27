"""JD / 岗位 CRUD。

按租户隔离;列表支持 ``?limit=&offset=&q=&status=`` 服务端分页与搜索
(详见 :mod:`app.api._pagination`)。

岗位创建 / 更新后(只要 status='open')自动入队匹配评估,与最近
50 份 done 简历跑 LLM 匹配。失败 silent log,不影响主链路。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_within_quota
from app.domain.models import Job, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.job_governance import snapshot_version
from app.services.permissions import (
    PERM_DELETE_JOBS,
    PERM_WRITE_JOBS,
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger(__name__)


_LEVELS = set(Job.LEVELS)
_STATUSES = set(Job.STATUSES)


class JobIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    level: str = Field("intermediate")
    description: str = Field("", max_length=20000)
    skills: list[str] = Field(default_factory=list)
    # T13:挂声明式评分规则(NULL → 走默认五维证据评分)
    scoring_rule_id: str | None = None
    # optimistic lock:带读取时的 updated_at,失配 → 409
    base_updated_at: datetime | None = None


class JobStatusIn(BaseModel):
    """单独切换岗位状态(open / paused / closed),不动 JD 字段。

    走独立 PATCH 端点的原因:
    - 状态是岗位生命周期事件(开/暂停/关闭),HR 通常单独触发,
      不该和 JD 编辑混在一个 PUT 里(避免误改 JD)
    - 关闭/暂停后能保留岗位记录与历史面试,不级联清数据
    """

    status: str = Field(..., min_length=1, max_length=16)


class JobOut(BaseModel):
    id: str
    title: str
    level: str
    status: str
    description: str
    skills: list[str]
    org_unit_id: str | None
    publish_status: str
    current_version: int
    scoring_rule_id: str | None = None
    created_at: datetime
    updated_at: datetime


def _job_out(j: Job) -> JobOut:
    return JobOut(
        id=j.id,
        title=j.title,
        level=j.level,
        status=j.status,
        description=j.description,
        skills=j.skills or [],
        org_unit_id=j.org_unit_id,
        scoring_rule_id=j.scoring_rule_id,
        publish_status=j.publish_status,
        current_version=j.current_version,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


def _validate_level(lvl: str) -> str:
    if lvl not in _LEVELS:
        raise HTTPException(400, f"非法 level: {lvl},允许 {sorted(_LEVELS)}")
    return lvl


def _validate_status(s: str) -> str:
    if s not in _STATUSES:
        raise HTTPException(400, f"非法 status: {s},允许 {sorted(_STATUSES)}")
    return s


def _validate_rule_id(db: Session, rid: str | None, current: User) -> str | None:
    """T13:校验挂的评分规则存在且属本租户(防跨租户引用)。"""
    if not rid:
        return None
    from app.domain.models import ScoringRule

    r = db.get(ScoringRule, rid)
    if not r or r.tenant_id != current.tenant_id:
        raise HTTPException(400, "评分规则不存在")
    return rid


def _enqueue_match_eval(job_id: str, user_id: str | None) -> None:
    """岗位创建/更新后入队匹配评估。失败 silent — broker 不可达不应阻塞 CRUD。"""
    try:
        from app.workers.tasks.match import evaluate_matches_for_job

        evaluate_matches_for_job.delay(job_id, 50, user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[jobs] enqueue match eval failed job=%s err=%s", job_id, e
        )


@router.get("/", response_model=PageOut[JobOut])
def list_jobs(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    job_status: str | None = Query(
        None, alias="status", description="按状态过滤 open/paused/closed"
    ),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> PageOut[JobOut]:
    """列出本租户岗位(分页)。

    ``q`` 命中 ``title``;``skills`` 是 JSON 数组,SQLAlchemy 这层不好
    优雅地做 ``ANY ILIKE``,所以暂不在 SQL 里搜技能(95% 用户搜的是岗位
    名,真要按技能找有匹配评估页可走)。
    """
    limit, offset, q = p
    stmt = select(Job).where(Job.tenant_id == current.tenant_id)
    stmt = apply_org_filter(
        stmt, org_column=Job.org_unit_id, scope=get_org_scope(db, current)
    )
    if job_status:
        if job_status not in _STATUSES:
            raise HTTPException(400, f"非法 status: {job_status},允许 {sorted(_STATUSES)}")
        stmt = stmt.where(Job.status == job_status)
    stmt = apply_q_ilike(stmt, q, Job.title)

    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Job.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut[JobOut](
        items=[_job_out(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.post(
    "/",
    response_model=JobOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[
        Depends(require_within_quota("max_jobs")),
        Depends(require_permission(PERM_WRITE_JOBS)),
    ],
)
def create_job(
    body: JobIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> JobOut:
    job = Job(
        tenant_id=current.tenant_id,
        # 新岗位默认归属创建者的组织节点。admin 创建时若没有 org 则留空(全租户共享)
        org_unit_id=current.org_unit_id,
        title=body.title.strip(),
        level=_validate_level(body.level),
        description=body.description,
        skills=body.skills,
        scoring_rule_id=_validate_rule_id(db, body.scoring_rule_id, current),
        created_by=current.id,
        # EPIC-05:新岗位走治理流,默认 draft;旧岗位由 migration 默认 published
        publish_status="draft",
        current_version=1,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    # 落首个版本快照
    snapshot_version(
        db, job=job, author=current, change_kind="create", bump_version=False
    )
    write_audit(
        db,
        actor=current,
        entity_type="job",
        entity_id=job.id,
        action="create",
        detail={"title": job.title, "level": job.level},
        request=request,
    )
    db.commit()
    # 新岗位默认 status='open',自动评估最近简历
    if job.status == "open":
        _enqueue_match_eval(job.id, current.id)
    return _job_out(job)


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> JobOut:
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    return _job_out(job)


@router.get("/{job_id}/ai-profile")
def job_ai_profile(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """动态职位画像(对标 Moka「动态校准画像」)。

    合成三源数据,让「岗位当前看重什么」可见:
    - 评分规则的声明式权重(挂在岗位上的当前标准)
    - 决策学习的权重建议(近 90 天 HR 真实决策教出来的偏好)
    - 匹配评估统计(已评人数/平均分,感知画像覆盖面)

    ``one_liner`` 是模板化的一句话画像——不依赖 LLM,任何环境可用。
    """
    from sqlalchemy import func

    from app.domain.models import ResumeJobMatch, ScoringRule
    from app.services.decision_learning import weight_suggestions

    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)

    # 评分规则当前权重
    rule_out = None
    if job.scoring_rule_id:
        rule = db.get(ScoringRule, job.scoring_rule_id)
        if rule:
            rule_out = {
                "name": rule.name,
                "version": rule.version,
                "dimensions": [
                    {"key": d.get("key"), "name": d.get("name"), "weight": d.get("weight")}
                    for d in (rule.dimensions or [])
                ],
            }

    # 岗位级决策学习(90 天)
    learned = weight_suggestions(db, current.tenant_id, job_id=job_id, days=90)
    n = learned["decisions_total"]
    confidence = "high" if n >= 20 else ("medium" if n >= 5 else "low")
    learned_out = {
        "decisions_total": n,
        "positive": learned["positive"],
        "negative": learned["negative"],
        "confidence": confidence,
        "dimensions": [
            {
                "name": d["name"],
                "current_weight": d["current_weight"],
                "suggested_weight": d["suggested_weight"],
                "gap": d["gap"],
                "pos_avg_level": d.get("pos_avg_level"),
                "neg_avg_level": d.get("neg_avg_level"),
            }
            for d in learned["dimensions"]
        ],
    }

    # 匹配评估统计
    stats = db.execute(
        select(func.count(ResumeJobMatch.id), func.avg(ResumeJobMatch.score), func.max(ResumeJobMatch.score)).where(
            ResumeJobMatch.tenant_id == current.tenant_id,
            ResumeJobMatch.job_id == job_id,
        )
    ).one()
    match_stats = {
        "evaluated": stats[0] or 0,
        "avg_score": round(float(stats[1]), 1) if stats[1] is not None else None,
        "top_score": stats[2],
    }

    # 模板化一句话画像:权重最高的学习维度 + 决策规模
    top = next(
        (d for d in learned_out["dimensions"] if d["suggested_weight"] is not None), None
    )
    parts = []
    if top and abs(top["gap"] or 0) >= 2:
        direction = "更看重" if (top["gap"] or 0) > 0 else "放宽"
        parts.append(f"实际决策显示该岗{direction}「{top['name']}」(建议权重 {top['suggested_weight']})")
    elif rule_out and rule_out["dimensions"]:
        w = max(rule_out["dimensions"], key=lambda d: d.get("weight") or 0)
        parts.append(f"当前标准以「{w['name']}」为重(权重 {w['weight']})")
    else:
        parts.append("使用默认五维评估标准")
    if n:
        parts.append(f"近 90 天 {n} 次流转决策持续校准画像")
    else:
        parts.append("尚无流转决策,画像保持初始标准")
    if match_stats["evaluated"]:
        parts.append(f"已评估 {match_stats['evaluated']} 人,平均 {match_stats['avg_score']} 分")

    return {
        "job_id": job.id,
        "title": job.title,
        "rule": rule_out,
        "learned": learned_out,
        "match_stats": match_stats,
        "one_liner": ";".join(parts) + "。",
    }


@router.put(
    "/{job_id}",
    response_model=JobOut,
    dependencies=[Depends(require_permission(PERM_WRITE_JOBS))],
)
def update_job(
    job_id: str,
    body: JobIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> JobOut:
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    if body.base_updated_at is not None and job.updated_at is not None:
        cur = job.updated_at
        if cur.tzinfo is None:
            cur = cur.replace(tzinfo=UTC)
        if cur != body.base_updated_at:
            raise HTTPException(409, "记录已被他人修改,请刷新后重试")
    before = {
        "title": job.title,
        "level": job.level,
        "skills": list(job.skills or []),
        "description_len": len(job.description or ""),
    }
    job.title = body.title.strip()
    job.level = _validate_level(body.level)
    job.description = body.description
    job.skills = body.skills
    job.scoring_rule_id = _validate_rule_id(db, body.scoring_rule_id, current)
    job.updated_at = datetime.now(UTC).replace(tzinfo=None)
    after = {
        "title": job.title,
        "level": job.level,
        "skills": list(job.skills or []),
        "description_len": len(job.description or ""),
    }
    content_changed = before != after
    if content_changed:
        # EPIC-05:内容变更落一条版本,version_no +1
        snapshot_version(
            db,
            job=job,
            author=current,
            change_kind="content_update",
        )
        # 已发布的岗位被改动 → 回到草稿,重新走审批
        if job.publish_status == "published":
            job.publish_status = "draft"
            job.submitted_by = None
            job.submitted_at = None
    write_audit(
        db,
        actor=current,
        entity_type="job",
        entity_id=job.id,
        action="update",
        detail={"before": before, "after": after, "version_no": job.current_version},
        request=request,
    )
    db.commit()
    db.refresh(job)
    # JD 改动后重投匹配评估(只对还没评的对生效;已 done 的不会重跑)
    if job.status == "open":
        _enqueue_match_eval(job.id, current.id)
    return _job_out(job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission(PERM_DELETE_JOBS))],
)
def delete_job(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    write_audit(
        db,
        actor=current,
        entity_type="job",
        entity_id=job.id,
        action="delete",
        detail={"title": job.title},
        request=request,
    )
    db.delete(job)
    db.commit()


@router.patch(
    "/{job_id}/status",
    response_model=JobOut,
    dependencies=[Depends(require_permission(PERM_WRITE_JOBS))],
)
def update_job_status(
    job_id: str,
    body: JobStatusIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> JobOut:
    """切换岗位状态。

    open → paused / closed:停止后续匹配评估的入队,但已生成的匹配分保留
    paused / closed → open:重新触发一次匹配评估(对最近 50 份 done 简历)
    """
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    new_status = _validate_status(body.status)
    old_status = job.status
    job.status = new_status
    job.updated_at = datetime.now(UTC).replace(tzinfo=None)
    write_audit(
        db,
        actor=current,
        entity_type="job",
        entity_id=job.id,
        action="status_change",
        detail={"before": old_status, "after": new_status},
        request=request,
    )
    db.commit()
    db.refresh(job)
    if new_status == "open" and old_status != "open":
        _enqueue_match_eval(job.id, current.id)
    return _job_out(job)
