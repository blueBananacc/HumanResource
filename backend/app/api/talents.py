"""人才库 API。EPIC-04 P0:以候选人为主视角的聚合视图 + 运营动作。

接口分组:
- ``GET  /talents``                  — 候选人列表(以 Candidate 为粒度)
- ``GET  /talents/{id}``             — 主档案聚合(简历 / 面试 / 匹配 / 标签 / 分组 / 最近备注)
- ``DELETE /talents/{id}``           — 删除候选人及全部依赖(简历 / 面试 / 备注 / 分组关系)
- ``GET  /talents/{id}/timeline``    — 统一时间线
- ``PUT  /talents/{id}/tags``        — 整体覆盖标签数组
- ``POST /talents/{id}/blacklist``   — 加入黑名单
- ``DELETE /talents/{id}/blacklist`` — 移出黑名单
- ``GET / POST /talents/{id}/notes`` — 备注
- ``GET / POST /talent-groups``      — 分组管理
- ``DELETE /talent-groups/{id}``     — 删除分组
- ``POST / DELETE /talent-groups/{id}/members`` — 分组成员

设计:
- 列表 / 详情走 ``view_reports`` 权限,viewer 即可查看
- 运营动作走 ``write_resumes`` 权限(HR / admin)
- 全部走 ``get_org_scope`` 数据范围过滤,接到 EPIC-01
- 所有写动作落审计,接到 EPIC-02
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.resumes import delete_candidate_with_dependents
from app.domain.models import (
    Candidate,
    CandidateGroup,
    CandidateGroupMember,
    CandidateNote,
    CandidateSkill,
    Interview,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    JobApplication,
    JobOffer,
    OrgUnit,
    Resume,
    ResumeJobMatch,
    Skill,
    TalentCampaignSend,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.candidates import _hash as _contact_hash
from app.services import dedup as dedup_svc
from app.services.permissions import (
    PERM_DELETE_RESUMES,
    PERM_VIEW_REPORTS,
    PERM_WRITE_RESUMES,
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/talents", tags=["talents"])
groups_router = APIRouter(prefix="/talent-groups", tags=["talents"])


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _touch_last_active(c: Candidate) -> None:
    c.last_active_at = _utcnow_naive()


def _user_label(db: Session, user_id: str | None) -> str | None:
    """持锁人展示名:优先员工档案名,兜底 email。"""
    return _user_labels(db, [user_id] if user_id else []).get(user_id or "")


def _user_labels(db: Session, user_ids: list[str]) -> dict[str, str]:
    """批量解析展示名(列表页防 N+1)。"""
    if not user_ids:
        return {}
    from app.domain.models import Employee

    emp_map = {
        e.user_id: e.name
        for e in db.scalars(
            select(Employee).where(Employee.user_id.in_(user_ids))
        )
    }
    user_map = {
        u.id: u.email
        for u in db.scalars(select(User).where(User.id.in_(user_ids)))
    }
    return {
        uid: emp_map.get(uid) or user_map.get(uid) or "?" for uid in user_ids
    }


# ============ schemas ============


class TagsIn(BaseModel):
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for raw in v:
            s = (raw or "").strip()
            if not s:
                continue
            if len(s) > 32:
                s = s[:32]
            if s not in out:
                out.append(s)
            if len(out) >= 20:
                break  # 上限 20 个标签,避免无限增长
        return out


class BlacklistIn(BaseModel):
    reason: str = Field(default="", max_length=512)


class NoteIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class GroupIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        return v.strip()


class GroupMemberIn(BaseModel):
    candidate_ids: list[str] = Field(min_length=1, max_length=200)


class TalentListItem(BaseModel):
    id: str
    name: str
    display_email: str | None
    display_phone: str | None
    org_unit_id: str | None
    tags: list[str]
    status: str
    is_blacklisted: bool
    blacklist_reason: str | None
    last_active_at: datetime | None
    created_at: datetime
    resume_count: int
    interview_count: int
    last_interview_status: str | None
    last_interview_at: datetime | None
    top_match_job_title: str | None
    top_match_score: int | None
    dedup_flag: str | None = None
    dedup_peer_name: str | None = None
    locked_by_id: str | None = None
    locked_by_name: str | None = None
    locked_until: datetime | None = None
    locked: bool = False


class TalentDetail(BaseModel):
    id: str
    name: str
    display_email: str | None
    display_phone: str | None
    org_unit_id: str | None
    tags: list[str]
    status: str
    is_blacklisted: bool
    blacklist_reason: str | None
    blacklisted_at: datetime | None
    blacklisted_by: str | None
    last_active_at: datetime | None
    created_at: datetime
    dedup_flag: str | None = None
    dedup_peer_id: str | None = None
    dedup_peer_name: str | None = None
    locked_by_id: str | None = None
    locked_by_name: str | None = None
    locked_until: datetime | None = None
    locked: bool = False
    resumes: list[dict[str, Any]]
    interviews: list[dict[str, Any]]
    matches: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    recent_notes: list[dict[str, Any]]


class TimelineEvent(BaseModel):
    """统一时间线事件。``kind`` 决定 UI 图标 / 颜色;``ref`` 可指向详情链接。"""

    at: datetime
    kind: str
    title: str
    detail: dict[str, Any] | None = None
    ref: dict[str, str] | None = None


class GroupOut(BaseModel):
    id: str
    name: str
    description: str | None
    member_count: int
    auto_rule: dict[str, Any] | None = None
    auto_count: int = 0
    created_at: datetime
    updated_at: datetime


class GroupRuleIn(BaseModel):
    """自动入库规则写入:``auto_rule=None`` 表示清除规则回到纯手动组。"""

    auto_rule: dict[str, Any] | None = None


class RulePreviewIn(BaseModel):
    auto_rule: dict[str, Any]


class NoteOut(BaseModel):
    id: str
    author_id: str
    author_email: str
    content: str
    created_at: datetime


# ============ helpers ============


def _get_candidate(
    db: Session, *, candidate_id: str, current: User
) -> Candidate:
    c = db.get(Candidate, candidate_id)
    if not c or c.tenant_id != current.tenant_id:
        raise HTTPException(404, "候选人不存在")
    ensure_can_see(get_org_scope(db, current), c.org_unit_id)
    return c


def _serialize_resume(r: Resume) -> dict[str, Any]:
    return {
        "id": r.id,
        "file_name": r.file_name,
        "file_size": r.file_size,
        "source": r.source,
        "parse_status": r.parse_status,
        "created_at": r.created_at,
        "skills": (r.parsed_data or {}).get("skills") or [],
    }


def _serialize_interview(i: Interview, job_title: str | None) -> dict[str, Any]:
    summary = i.summary or {}
    rec = summary.get("recommendation") if isinstance(summary, dict) else None
    return {
        "id": i.id,
        "job_id": i.job_id,
        "job_title": job_title,
        "mode": i.mode,
        "modality": i.modality,
        "status": i.status,
        "recommendation": rec,
        "started_at": i.started_at,
        "finished_at": i.finished_at,
    }


def _serialize_match(m: ResumeJobMatch, job_title: str | None) -> dict[str, Any]:
    return {
        "id": m.id,
        "resume_id": m.resume_id,
        "job_id": m.job_id,
        "job_title": job_title,
        "status": m.status,
        "score": m.score,
        "created_at": m.created_at,
    }


def _serialize_note(n: CandidateNote) -> dict[str, Any]:
    return {
        "id": n.id,
        "author_id": n.author_id,
        "author_email": n.author_email,
        "content": n.content,
        "created_at": n.created_at,
    }


def _bulk_job_titles(db: Session, job_ids: set[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    rows = db.execute(
        select(Job.id, Job.title).where(Job.id.in_(job_ids))
    ).all()
    return {jid: title for jid, title in rows}


# ============ 列表 ============


@router.get(
    "",
    response_model=PageOut[TalentListItem],
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def list_talents(
    p: tuple[int, int, str | None] = Depends(paginate_params),
    tag: str | None = Query(default=None, description="标签精确匹配"),
    group_id: str | None = Query(default=None, description="所属分组 id"),
    blacklisted: bool | None = Query(default=None, description="是否黑名单"),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> PageOut[TalentListItem]:
    """以候选人为粒度的人才库列表。

    聚合策略:简历数 / 面试数 / 最近面试 / 最高匹配分都在一条 SQL 之外用单独
    聚合查询,避免一条 join 把候选人行炸开。租户内候选人通常 ≤ 数万,这种
    "列表 ID 取出 → 聚合关联"方案的 latency 在 SQLite / PostgreSQL 上都可接受。
    """
    limit, offset, q = p
    scope = get_org_scope(db, current)

    stmt = select(Candidate).where(Candidate.tenant_id == current.tenant_id)
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = apply_q_ilike(
            stmt, q, Candidate.name, Candidate.display_email, Candidate.display_phone
        )
    if blacklisted is True:
        stmt = stmt.where(Candidate.is_blacklisted.is_(True))
    elif blacklisted is False:
        stmt = stmt.where(Candidate.is_blacklisted.is_(False))
    if group_id:
        # exists 子查询:候选人在指定分组里
        sub = select(CandidateGroupMember.candidate_id).where(
            CandidateGroupMember.group_id == group_id,
            CandidateGroupMember.candidate_id == Candidate.id,
        )
        stmt = stmt.where(sub.exists())

    total = count_total(db, stmt)
    # 排序:活跃时间倒序,空值排最后(用 created_at fallback 保证有顺序)
    stmt = stmt.order_by(
        Candidate.last_active_at.desc().nullslast(),
        Candidate.created_at.desc(),
    )
    rows = db.scalars(stmt.limit(limit).offset(offset)).all()
    # tag 过滤 — PG 上可以 SQL 端 jsonb 包含,跨方言通用做法是 Python 侧过滤(本页只有 limit 行)
    if tag:
        rows = [r for r in rows if tag in (r.tags or [])]

    cand_ids = [r.id for r in rows]
    if not cand_ids:
        return PageOut[TalentListItem](items=[], total=total, limit=limit, offset=offset)

    # 聚合:简历数 / 面试数 / 最近面试 / 最高匹配
    resume_counts = dict(
        db.execute(
            select(Resume.candidate_id, func.count(Resume.id))
            .where(Resume.candidate_id.in_(cand_ids))
            .group_by(Resume.candidate_id)
        ).all()
    )
    interview_rows = db.execute(
        select(
            Interview.candidate_id,
            func.count(Interview.id),
            func.max(Interview.started_at),
        )
        .where(
            Interview.candidate_id.in_(cand_ids),
            Interview.mode == "remote",
        )
        .group_by(Interview.candidate_id)
    ).all()
    iv_count_map = {cid: int(cnt) for cid, cnt, _ in interview_rows}
    iv_last_at_map = {cid: latest for cid, _, latest in interview_rows}

    # 最近面试的 status:再查一次"按候选人取最新一条面试的 status"
    iv_status_map: dict[str, str] = {}
    if iv_count_map:
        iv_status_rows = db.execute(
            select(Interview.candidate_id, Interview.status, Interview.started_at)
            .where(
                Interview.candidate_id.in_(iv_count_map.keys()),
                Interview.mode == "remote",
            )
            .order_by(Interview.candidate_id, Interview.started_at.desc())
        ).all()
        seen: set[str] = set()
        for cid, status_, _ in iv_status_rows:
            if cid in seen:
                continue
            iv_status_map[cid] = status_
            seen.add(cid)

    # 最高匹配 — 需要 candidate_id,通过 resume 关联;P0 用 join 取每个候选人的最高 score
    top_match_rows = db.execute(
        select(
            Resume.candidate_id,
            ResumeJobMatch.job_id,
            ResumeJobMatch.score,
        )
        .join(ResumeJobMatch, ResumeJobMatch.resume_id == Resume.id)
        .where(
            Resume.candidate_id.in_(cand_ids),
            ResumeJobMatch.status == "done",
            ResumeJobMatch.score.is_not(None),
        )
        .order_by(Resume.candidate_id, ResumeJobMatch.score.desc())
    ).all()
    top_match_map: dict[str, tuple[str, int]] = {}
    for cid, jid, score in top_match_rows:
        if cid in top_match_map:
            continue
        top_match_map[cid] = (jid, int(score))
    job_titles = _bulk_job_titles(db, {jid for jid, _ in top_match_map.values()})

    lock_names = _user_labels(
        db, [r.locked_by_id for r in rows if r.locked_by_id]
    )
    peer_names = {
        c.id: c.name
        for c in db.scalars(
            select(Candidate).where(
                Candidate.id.in_([r.dedup_peer_id for r in rows if r.dedup_peer_id])
            )
        )
    } if any(r.dedup_peer_id for r in rows) else {}

    items = [
        TalentListItem(
            id=r.id,
            name=r.name,
            display_email=r.display_email,
            display_phone=r.display_phone,
            org_unit_id=r.org_unit_id,
            tags=list(r.tags or []),
            status=r.status,
            is_blacklisted=r.is_blacklisted,
            blacklist_reason=r.blacklist_reason,
            last_active_at=r.last_active_at,
            created_at=r.created_at,
            resume_count=int(resume_counts.get(r.id, 0)),
            interview_count=int(iv_count_map.get(r.id, 0)),
            last_interview_status=iv_status_map.get(r.id),
            last_interview_at=iv_last_at_map.get(r.id),
            top_match_job_title=(
                job_titles.get(top_match_map[r.id][0])
                if r.id in top_match_map
                else None
            ),
            top_match_score=top_match_map[r.id][1] if r.id in top_match_map else None,
            dedup_flag=r.dedup_flag,
            dedup_peer_name=peer_names.get(r.dedup_peer_id or ""),
            locked_by_id=r.locked_by_id,
            locked_by_name=lock_names.get(r.locked_by_id or ""),
            locked_until=r.locked_until,
            locked=dedup_svc.is_locked(r),
        )
        for r in rows
    ]
    # tag 过滤导致的 total 修正:更准确的总数需要再算一遍 — 但 tag 是次要筛选,
    # 这里返回原始 total,前端 UI 已经显示"当前页 N 条"。
    return PageOut[TalentListItem](
        items=items, total=total, limit=limit, offset=offset
    )


# ---- 静态子路径必须在 /{candidate_id} 之前注册,否则被详情路由捕获 ----


@router.get(
    "/dedup-queue",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def dedup_queue(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """待审重复队列:精准对(两通道命中)+ 疑似对(单通道命中)。

    队列 = 当前 dedup_flag='suspect' 的候选人与对端的配对 + 存量扫描中
    的精准对(精准对不一定挂标,它们更接近"同一人",优先合并)。
    """
    result = dedup_svc.scan_tenant(db, current.tenant_id)
    db.commit()  # scan 挂标落库
    return {
        "scanned": result["scanned"],
        "exact": [
            {
                "candidate": _dup_person(p["candidate"]),
                "peer": _dup_person(p["peer"]),
                "channels": p["channels"],
            }
            for p in result["exact"]
        ],
        "suspect": [
            {
                "candidate": _dup_person(p["candidate"]),
                "peer": _dup_person(p["peer"]),
                "channels": p["channels"],
            }
            for p in result["suspect"]
        ],
    }


@router.post(
    "/dedup-scan",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def dedup_scan(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """手动全量重扫:给疑似候选挂标,返回各级计数。幂等。"""
    result = dedup_svc.scan_tenant(db, current.tenant_id)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id="bulk",
        action="dedup_scan",
        detail={
            "scanned": result["scanned"],
            "exact": len(result["exact"]),
            "suspect": len(result["suspect"]),
            "flagged": result["flagged"],
        },
        request=request,
    )
    db.commit()
    return {
        "scanned": result["scanned"],
        "exact": len(result["exact"]),
        "suspect": len(result["suspect"]),
        "flagged": result["flagged"],
    }


# ============ 详情 ============


@router.get(
    "/{candidate_id}",
    response_model=TalentDetail,
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def get_talent(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> TalentDetail:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)

    resumes = db.scalars(
        select(Resume)
        .where(Resume.candidate_id == c.id)
        .order_by(Resume.created_at.desc())
        .limit(20)
    ).all()
    interviews = db.scalars(
        select(Interview)
        .where(
            Interview.candidate_id == c.id,
            Interview.mode == "remote",
        )
        .order_by(Interview.started_at.desc().nullslast(), Interview.id.desc())
        .limit(20)
    ).all()
    job_ids = {i.job_id for i in interviews}
    matches = db.execute(
        select(ResumeJobMatch, Resume.candidate_id)
        .join(Resume, ResumeJobMatch.resume_id == Resume.id)
        .where(Resume.candidate_id == c.id)
        .order_by(ResumeJobMatch.score.desc().nullslast())
        .limit(20)
    ).all()
    job_ids.update(m.job_id for m, _ in matches)
    titles = _bulk_job_titles(db, job_ids)

    group_rows = db.scalars(
        select(CandidateGroup)
        .join(
            CandidateGroupMember,
            CandidateGroupMember.group_id == CandidateGroup.id,
        )
        .where(
            CandidateGroup.tenant_id == c.tenant_id,
            CandidateGroupMember.candidate_id == c.id,
        )
        .order_by(CandidateGroup.created_at.asc())
    ).all()
    notes = db.scalars(
        select(CandidateNote)
        .where(CandidateNote.candidate_id == c.id)
        .order_by(CandidateNote.created_at.desc())
        .limit(5)
    ).all()

    # 详情视为"访问"动作,落审计(EPIC-02 候选人访问日志)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="view",
        request=request,
    )
    db.commit()

    return TalentDetail(
        id=c.id,
        name=c.name,
        display_email=c.display_email,
        display_phone=c.display_phone,
        org_unit_id=c.org_unit_id,
        tags=list(c.tags or []),
        status=c.status,
        is_blacklisted=c.is_blacklisted,
        blacklist_reason=c.blacklist_reason,
        blacklisted_at=c.blacklisted_at,
        blacklisted_by=c.blacklisted_by,
        last_active_at=c.last_active_at,
        created_at=c.created_at,
        resumes=[_serialize_resume(r) for r in resumes],
        interviews=[
            _serialize_interview(i, titles.get(i.job_id)) for i in interviews
        ],
        matches=[_serialize_match(m, titles.get(m.job_id)) for m, _ in matches],
        groups=[
            {"id": g.id, "name": g.name, "description": g.description}
            for g in group_rows
        ],
        recent_notes=[_serialize_note(n) for n in notes],
        dedup_flag=c.dedup_flag,
        dedup_peer_id=c.dedup_peer_id,
        dedup_peer_name=(
            db.get(Candidate, c.dedup_peer_id).name
            if c.dedup_peer_id and db.get(Candidate, c.dedup_peer_id)
            else None
        ),
        locked_by_id=c.locked_by_id,
        locked_by_name=_user_label(db, c.locked_by_id),
        locked_until=c.locked_until,
        locked=dedup_svc.is_locked(c),
    )


# ============ 删除 ============


@router.delete(
    "/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission(PERM_DELETE_RESUMES))],
)
async def delete_talent(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    """删除候选人及其全部依赖数据。

    权限与简历删除同级(``delete_resumes``):候选人档案本质上由简历聚合而来。
    DB 层 ``ondelete=CASCADE`` 连带删 resumes / interviews / notes /
    group members;对象存储里的简历原文件与面试音频由
    ``app.api.resumes.delete_candidate_with_dependents`` 尽力清理。
    """
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    dedup_svc.assert_unlocked(c, current.id)
    await delete_candidate_with_dependents(
        db, c, actor=current, request=request, reason="manual_delete"
    )
    db.commit()


# ============ 时间线 ============


@router.get(
    "/{candidate_id}/timeline",
    response_model=list[TimelineEvent],
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def get_timeline(
    candidate_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[TimelineEvent]:
    """聚合所有事件并按时间倒序返回。

    数据源:
    - candidate.created_at(候选人创建)
    - resume.created_at(简历上传)
    - interview.started_at(面试发起)+ interview.finished_at(面试完成)
    - candidate_notes.created_at(备注)
    - blacklist.blacklisted_at(若已加入黑名单)
    - group_member.added_at(加入分组)
    """
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    events: list[TimelineEvent] = []

    events.append(
        TimelineEvent(
            at=c.created_at,
            kind="candidate_created",
            title="候选人入库",
        )
    )

    resumes = db.scalars(
        select(Resume).where(Resume.candidate_id == c.id)
    ).all()
    for r in resumes:
        events.append(
            TimelineEvent(
                at=r.created_at,
                kind="resume_upload",
                title=f"上传简历 {r.file_name}",
                detail={"resume_id": r.id, "source": r.source},
                ref={"resume_id": r.id},
            )
        )

    interviews = db.scalars(
        select(Interview).where(
            Interview.candidate_id == c.id,
            Interview.mode == "remote",
        )
    ).all()
    job_titles = _bulk_job_titles(db, {i.job_id for i in interviews})
    for i in interviews:
        title = job_titles.get(i.job_id, "未知岗位")
        if i.started_at:
            events.append(
                TimelineEvent(
                    at=i.started_at,
                    kind="interview_start",
                    title=f"发起面试 · {title}",
                    detail={"interview_id": i.id, "status": i.status},
                    ref={"interview_id": i.id},
                )
            )
        if i.finished_at and i.status == "done":
            summary = i.summary or {}
            rec = summary.get("recommendation") if isinstance(summary, dict) else None
            events.append(
                TimelineEvent(
                    at=i.finished_at,
                    kind="interview_done",
                    title=f"面试完成 · {title}",
                    detail={
                        "interview_id": i.id,
                        "recommendation": rec,
                    },
                    ref={"interview_id": i.id},
                )
            )

    notes = db.scalars(
        select(CandidateNote).where(CandidateNote.candidate_id == c.id)
    ).all()
    for n in notes:
        events.append(
            TimelineEvent(
                at=n.created_at,
                kind="note",
                title=f"{n.author_email} 添加备注",
                detail={"content": n.content[:120]},
            )
        )

    if c.is_blacklisted and c.blacklisted_at:
        events.append(
            TimelineEvent(
                at=c.blacklisted_at,
                kind="blacklisted",
                title="加入黑名单",
                detail={"reason": c.blacklist_reason or ""},
            )
        )

    group_links = db.execute(
        select(CandidateGroupMember, CandidateGroup)
        .join(
            CandidateGroup,
            CandidateGroup.id == CandidateGroupMember.group_id,
        )
        .where(
            CandidateGroupMember.candidate_id == c.id,
            CandidateGroup.tenant_id == c.tenant_id,
        )
    ).all()
    for member, grp in group_links:
        events.append(
            TimelineEvent(
                at=member.added_at,
                kind="group_join",
                title=f"加入分组 {grp.name}",
                detail={"group_id": grp.id},
            )
        )

    events.sort(key=lambda e: e.at, reverse=True)
    return events


# ============ tags ============


@router.put(
    "/{candidate_id}/tags",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def set_tags(
    candidate_id: str,
    body: TagsIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, list[str]]:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    dedup_svc.assert_unlocked(c, current.id)
    before = list(c.tags or [])
    c.tags = body.tags
    _touch_last_active(c)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="update_tags",
        detail={"before": before, "after": body.tags},
        request=request,
    )
    try:
        from app.services.group_rules import enroll_candidate

        enroll_candidate(db, c)
    except Exception:  # noqa: BLE001
        logger.exception("auto_enroll 失败 candidate=%s", c.id)
    db.commit()
    return {"tags": list(c.tags or [])}


# ============ blacklist ============


@router.post(
    "/{candidate_id}/blacklist",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def add_blacklist(
    candidate_id: str,
    body: BlacklistIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    dedup_svc.assert_unlocked(c, current.id)
    c.is_blacklisted = True
    c.blacklist_reason = body.reason or None
    c.blacklisted_at = _utcnow_naive()
    c.blacklisted_by = current.id
    _touch_last_active(c)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="blacklist",
        detail={"reason": body.reason or ""},
        request=request,
    )
    db.commit()
    return {
        "is_blacklisted": True,
        "blacklist_reason": c.blacklist_reason,
        "blacklisted_at": c.blacklisted_at,
    }


@router.delete(
    "/{candidate_id}/blacklist",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def remove_blacklist(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    dedup_svc.assert_unlocked(c, current.id)
    prev_reason = c.blacklist_reason
    c.is_blacklisted = False
    c.blacklist_reason = None
    c.blacklisted_at = None
    c.blacklisted_by = None
    _touch_last_active(c)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="unblacklist",
        detail={"prev_reason": prev_reason},
        request=request,
    )
    db.commit()
    return {"is_blacklisted": False}


# ============ notes ============


@router.get(
    "/{candidate_id}/notes",
    response_model=list[NoteOut],
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def list_notes(
    candidate_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[NoteOut]:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    rows = db.scalars(
        select(CandidateNote)
        .where(CandidateNote.candidate_id == c.id)
        .order_by(CandidateNote.created_at.desc())
        .limit(200)
    ).all()
    return [
        NoteOut(
            id=n.id,
            author_id=n.author_id,
            author_email=n.author_email,
            content=n.content,
            created_at=n.created_at,
        )
        for n in rows
    ]


@router.post(
    "/{candidate_id}/notes",
    response_model=NoteOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def add_note(
    candidate_id: str,
    body: NoteIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> NoteOut:
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    note = CandidateNote(
        tenant_id=c.tenant_id,
        candidate_id=c.id,
        author_id=current.id,
        author_email=current.email,
        content=body.content,
    )
    db.add(note)
    _touch_last_active(c)
    db.flush()
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="add_note",
        detail={"note_id": note.id, "len": len(body.content)},
        request=request,
    )
    db.commit()
    db.refresh(note)
    return NoteOut(
        id=note.id,
        author_id=note.author_id,
        author_email=note.author_email,
        content=note.content,
        created_at=note.created_at,
    )


# ============ groups ============


@groups_router.get(
    "",
    response_model=list[GroupOut],
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def list_groups(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[GroupOut]:
    rows = db.scalars(
        select(CandidateGroup)
        .where(CandidateGroup.tenant_id == current.tenant_id)
        .order_by(CandidateGroup.created_at.asc())
    ).all()
    if not rows:
        return []
    counts_rows = db.execute(
        select(
            CandidateGroupMember.group_id,
            CandidateGroupMember.source,
            func.count(CandidateGroupMember.id),
        )
        .where(CandidateGroupMember.group_id.in_([g.id for g in rows]))
        .group_by(CandidateGroupMember.group_id, CandidateGroupMember.source)
    ).all()
    counts: dict[str, int] = {}
    auto_counts: dict[str, int] = {}
    for gid, src, cnt in counts_rows:
        counts[gid] = counts.get(gid, 0) + int(cnt)
        if src == "auto":
            auto_counts[gid] = auto_counts.get(gid, 0) + int(cnt)
    return [
        GroupOut(
            id=g.id,
            name=g.name,
            description=g.description,
            member_count=counts.get(g.id, 0),
            auto_rule=g.auto_rule,
            auto_count=auto_counts.get(g.id, 0),
            created_at=g.created_at,
            updated_at=g.updated_at,
        )
        for g in rows
    ]


@groups_router.post(
    "",
    response_model=GroupOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def create_group(
    body: GroupIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> GroupOut:
    g = CandidateGroup(
        tenant_id=current.tenant_id,
        name=body.name,
        description=body.description,
        created_by=current.id,
    )
    db.add(g)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "同名分组已存在") from None
    write_audit(
        db,
        actor=current,
        entity_type="candidate_group",
        entity_id=g.id,
        action="create",
        detail={"name": g.name},
        request=request,
    )
    db.commit()
    db.refresh(g)
    return GroupOut(
        id=g.id,
        name=g.name,
        description=g.description,
        member_count=0,
        created_at=g.created_at,
        updated_at=g.updated_at,
    )


@groups_router.delete(
    "/{group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def delete_group(
    group_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    g = db.get(CandidateGroup, group_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(404, "分组不存在")
    write_audit(
        db,
        actor=current,
        entity_type="candidate_group",
        entity_id=g.id,
        action="delete",
        detail={"name": g.name},
        request=request,
    )
    db.delete(g)
    db.commit()


# ============ 自动入库规则 ============


def _get_group(db: Session, group_id: str, current: User) -> CandidateGroup:
    g = db.get(CandidateGroup, group_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(404, "分组不存在")
    return g


@groups_router.put(
    "/{group_id}/rule",
    response_model=GroupOut,
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def set_group_rule(
    group_id: str,
    body: GroupRuleIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> GroupOut:
    """设置/清除自动入库规则。设置后新候选人即时评估;存量用 rescan。"""
    from app.services import group_rules

    g = _get_group(db, group_id, current)
    rule = None
    if body.auto_rule is not None:
        try:
            rule = group_rules.validate_rule(body.auto_rule)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
    g.auto_rule = rule
    g.updated_at = _utcnow_naive()
    write_audit(
        db,
        actor=current,
        entity_type="candidate_group",
        entity_id=g.id,
        action="set_rule",
        detail={"group_name": g.name, "rule": rule},
        request=request,
    )
    db.commit()
    rows = db.execute(
        select(CandidateGroupMember.source, func.count(CandidateGroupMember.id))
        .where(CandidateGroupMember.group_id == g.id)
        .group_by(CandidateGroupMember.source)
    ).all()
    cnt = sum(int(c) for _s, c in rows)
    auto_cnt = sum(int(c) for s, c in rows if s == "auto")
    return GroupOut(
        id=g.id, name=g.name, description=g.description, member_count=cnt,
        auto_rule=g.auto_rule, auto_count=auto_cnt,
        created_at=g.created_at, updated_at=g.updated_at,
    )


@groups_router.post(
    "/{group_id}/rescan",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def rescan_group(
    group_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """手动补扫:对租户全部候选人重放规则,命中补入组。幂等。"""
    from app.services import group_rules

    g = _get_group(db, group_id, current)
    try:
        result = group_rules.rescan_group(db, g)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if result["added"]:
        write_audit(
            db,
            actor=current,
            entity_type="candidate_group",
            entity_id=g.id,
            action="rescan",
            detail={
                "group_name": g.name,
                "scanned": result["scanned"],
                "added": len(result["added"]),
            },
            request=request,
        )
    db.commit()
    return {"scanned": result["scanned"], "added": result["added"]}


@groups_router.post(
    "/rule-preview",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def preview_group_rule(
    body: RulePreviewIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """规则试运行:不落库,返回命中样本与总数(编辑器预览)。"""
    from app.services import group_rules

    try:
        rule = group_rules.validate_rule(body.auto_rule)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return group_rules.preview_rule(db, current.tenant_id, rule)


@groups_router.get(
    "/rule-fields",
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def rule_fields() -> dict[str, Any]:
    """规则编辑器字段词表 — 前端渲染下拉用,与后端校验同源。"""
    from app.services.group_rules import FIELDS, MODES

    return {
        "modes": [{"value": m, "label": "全部满足" if m == "all" else "任一满足"} for m in MODES],
        "fields": [
            {"field": f, "label": spec["label"], "ops": spec["ops"], "value": spec["value"]}
            for f, spec in FIELDS.items()
        ],
        "stages": list(JobApplication.STAGES),
    }


@groups_router.post(
    "/{group_id}/members",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def add_group_members(
    group_id: str,
    body: GroupMemberIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """批量加入分组。已存在的成员跳过(idempotent)。"""
    g = db.get(CandidateGroup, group_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(404, "分组不存在")
    scope = get_org_scope(db, current)
    cand_rows = db.scalars(
        select(Candidate).where(
            Candidate.tenant_id == current.tenant_id,
            Candidate.id.in_(body.candidate_ids),
        )
    ).all()
    visible = [c for c in cand_rows if _visible(scope, c.org_unit_id)]
    existing = set(
        db.scalars(
            select(CandidateGroupMember.candidate_id).where(
                CandidateGroupMember.group_id == g.id,
                CandidateGroupMember.candidate_id.in_(
                    [c.id for c in visible]
                ),
            )
        ).all()
    )
    added: list[str] = []
    now = _utcnow_naive()
    for c in visible:
        if c.id in existing:
            continue
        db.add(
            CandidateGroupMember(
                group_id=g.id,
                candidate_id=c.id,
                added_by=current.id,
            )
        )
        c.last_active_at = now
        added.append(c.id)
    if added:
        g.updated_at = now
        write_audit(
            db,
            actor=current,
            entity_type="candidate_group",
            entity_id=g.id,
            action="add_members",
            detail={"added": added, "group_name": g.name},
            request=request,
        )
    db.commit()
    return {"added": added, "skipped": len(body.candidate_ids) - len(added)}


@groups_router.delete(
    "/{group_id}/members/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def remove_group_member(
    group_id: str,
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> None:
    g = db.get(CandidateGroup, group_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(404, "分组不存在")
    m = db.scalars(
        select(CandidateGroupMember).where(
            CandidateGroupMember.group_id == g.id,
            CandidateGroupMember.candidate_id == candidate_id,
        )
    ).first()
    if m is None:
        raise HTTPException(404, "候选人不在该分组")
    write_audit(
        db,
        actor=current,
        entity_type="candidate_group",
        entity_id=g.id,
        action="remove_member",
        detail={"candidate_id": candidate_id, "group_name": g.name},
        request=request,
    )
    db.delete(m)
    db.commit()


def _visible(scope: list[str] | None, target_org_id: str | None) -> bool:
    if scope is None:
        return True
    if target_org_id is None:
        return True
    return target_org_id in scope


# ---------- T5-3 候选人合并 + T5-4 CSV 导入 ----------

_MERGEABLE_FIELDS = ("name", "display_email", "display_phone", "tags", "status")


class MergeIn(BaseModel):
    drop_id: str
    # 字段级取舍:{field: "keep"|"drop"},默认 keep。
    field_choices: dict[str, str] = Field(default_factory=dict)


@router.get(
    "/{candidate_id}/duplicates",
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def find_duplicates(
    candidate_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """疑似重复:同 email_hash / phone_hash / 同名 的其他候选人。"""
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    conds = []
    if c.email_hash:
        conds.append(Candidate.email_hash == c.email_hash)
    if c.phone_hash:
        conds.append(Candidate.phone_hash == c.phone_hash)
    conds.append(Candidate.name == c.name)
    stmt = (
        select(Candidate)
        .where(
            Candidate.tenant_id == current.tenant_id,
            Candidate.id != c.id,
            or_(*conds),
        )
        .limit(20)
    )
    scope = get_org_scope(db, current)
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    rows = db.scalars(stmt).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "display_email": r.display_email,
            "display_phone": r.display_phone,
            "status": r.status,
            "match": (
                "email"
                if c.email_hash and r.email_hash == c.email_hash
                else "phone"
                if c.phone_hash and r.phone_hash == c.phone_hash
                else "name"
            ),
        }
        for r in rows
    ]


@router.post(
    "/{candidate_id}/merge",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
async def merge_talent(
    candidate_id: str,
    body: MergeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """合并候选人:drop 的全部关联迁移到 keep,字段按 field_choices 取舍,删 drop。"""
    keep = _get_candidate(db, candidate_id=candidate_id, current=current)
    drop = _get_candidate(db, candidate_id=body.drop_id, current=current)
    if keep.id == drop.id:
        raise HTTPException(400, "不能与自己合并")
    dedup_svc.assert_unlocked(keep, current.id)
    dedup_svc.assert_unlocked(drop, current.id)
    unknown = set(body.field_choices) - set(_MERGEABLE_FIELDS)
    if unknown:
        raise HTTPException(
            400, f"不可取舍的字段:{','.join(sorted(unknown))};可用:{','.join(_MERGEABLE_FIELDS)}"
        )
    # 唯一约束对(skill/group/campaign_send):keep 已占的先删 drop 侧再迁移
    keep_skill_ids = set(
        db.scalars(select(CandidateSkill.skill_id).where(CandidateSkill.candidate_id == keep.id))
    )
    db.query(CandidateSkill).filter(
        CandidateSkill.candidate_id == drop.id,
        CandidateSkill.skill_id.in_(keep_skill_ids),
    ).delete(synchronize_session=False)
    keep_group_ids = set(
        db.scalars(
            select(CandidateGroupMember.group_id).where(
                CandidateGroupMember.candidate_id == keep.id
            )
        )
    )
    db.query(CandidateGroupMember).filter(
        CandidateGroupMember.candidate_id == drop.id,
        CandidateGroupMember.group_id.in_(keep_group_ids),
    ).delete(synchronize_session=False)
    keep_campaign_ids = set(
        db.scalars(
            select(TalentCampaignSend.campaign_id).where(
                TalentCampaignSend.candidate_id == keep.id
            )
        )
    )
    db.query(TalentCampaignSend).filter(
        TalentCampaignSend.candidate_id == drop.id,
        TalentCampaignSend.campaign_id.in_(keep_campaign_ids),
    ).delete(synchronize_session=False)

    # 关联迁移
    for model in (
        Resume,
        JobApplication,
        Interview,
        JobOffer,
        CandidateSkill,
        CandidateNote,
        CandidateGroupMember,
        TalentCampaignSend,
    ):
        db.query(model).filter(model.candidate_id == drop.id).update(
            {"candidate_id": keep.id}, synchronize_session=False
        )
    # 字段取舍
    picked: dict[str, str] = {}
    for field, choice in body.field_choices.items():
        if choice not in ("keep", "drop"):
            raise HTTPException(400, f"field_choices.{field} 必须是 keep/drop")
        if choice == "drop":
            setattr(keep, field, getattr(drop, field))
            picked[field] = "drop"
    _touch_last_active(keep)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=keep.id,
        action="merge",
        detail={"drop_id": drop.id, "drop_name": drop.name, "picked": picked},
        request=request,
    )
    # drop 的简历已迁走,delete_candidate_with_dependents 只会删行 + 留 delete 审计
    await delete_candidate_with_dependents(
        db, drop, actor=current, request=request, reason="merged"
    )
    db.commit()
    return {"ok": True, "keep_id": keep.id}


# ============ 两级查重队列 + 候选人锁定(Moka 同款) ============


def _dup_person(c: Candidate) -> dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "display_email": c.display_email,
        "display_phone": c.display_phone,
        "status": c.status,
        "created_at": c.created_at,
        "dedup_flag": c.dedup_flag,
    }


@router.post(
    "/{candidate_id}/dedup-dismiss",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def dedup_dismiss(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """确认非重复:摘除疑标(对端不受影响)。"""
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    dedup_svc.assert_unlocked(c, current.id)
    peer_id = c.dedup_peer_id
    dedup_svc.dismiss_suspect(db, c)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="dedup_dismiss",
        detail={"peer_id": peer_id},
        request=request,
    )
    db.commit()
    return {"ok": True}


class LockIn(BaseModel):
    days: int = Field(default=7, ge=1, le=90)


@router.post(
    "/{candidate_id}/lock",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def lock_talent(
    candidate_id: str,
    body: LockIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """锁定候选人:防止其他 HR 拉入流程/编辑,超期自动释放。"""
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    if dedup_svc.is_locked(c) and c.locked_by_id != current.id:
        raise HTTPException(
            409,
            f"候选人已被 {_user_label(db, c.locked_by_id) or '其他同事'} 锁定至 {c.locked_until:%Y-%m-%d}",
        )
    dedup_svc.lock_candidate(db, c, current.id, days=body.days)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="lock",
        detail={"days": body.days, "until": c.locked_until.isoformat()},
        request=request,
    )
    db.commit()
    return {"locked": True, "locked_by_id": c.locked_by_id, "locked_until": c.locked_until}


@router.post(
    "/{candidate_id}/unlock",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
def unlock_talent(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> dict[str, Any]:
    """释放锁:持锁人本人可解;过期锁任何人可接管(先 lock 再操作)。"""
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    if dedup_svc.is_locked(c) and c.locked_by_id != current.id:
        raise HTTPException(
            403, f"仅持锁人({_user_label(db, c.locked_by_id) or '?'})可释放"
        )
    dedup_svc.unlock_candidate(db, c)
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id=c.id,
        action="unlock",
        detail={},
        request=request,
    )
    db.commit()
    return {"locked": False}


_CSV_LIMIT = 5 * 1024 * 1024


@router.post(
    "/import",
    dependencies=[Depends(require_permission(PERM_WRITE_RESUMES))],
)
async def import_talents(
    file: UploadFile,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """CSV 批量导入候选人。列:name(必填),email,phone,tags(分号),org_name。"""
    import csv
    import io

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(400, "仅支持 CSV 文件")
    data = await file.read()
    if len(data) > _CSV_LIMIT:
        raise HTTPException(400, "文件过大(>5MB)")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件编码必须是 UTF-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise HTTPException(400, "CSV 为空")
    cols = {c.strip().lower() for c in (rows[0] or {})}
    if "name" not in cols:
        raise HTTPException(400, "缺少 name 列;可用列:name,email,phone,tags,org_name")

    created, skipped = 0, []
    seen_hashes: set[str] = set()  # 文件内去重(autoflush=False,DB 查询看不到待插入行)
    for i, row in enumerate(rows[:2000], start=2):  # 上限 2000 行,行号含表头
        rec = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        name = rec.get("name", "")
        if not name:
            skipped.append({"row": i, "reason": "name 为空"})
            continue
        email, phone = rec.get("email") or None, rec.get("phone") or None
        e_hash = _contact_hash(email, current.tenant_id)
        p_hash = _contact_hash(phone, current.tenant_id)
        row_hashes = {h for h in (e_hash, p_hash) if h}
        if row_hashes & seen_hashes:
            skipped.append({"row": i, "reason": f"文件内重复({name})"})
            continue
        dup_conds = [
            cond
            for cond, h in (
                (Candidate.email_hash == e_hash, e_hash),
                (Candidate.phone_hash == p_hash, p_hash),
            )
            if h
        ]
        if dup_conds and db.scalar(
            select(Candidate.id).where(
                Candidate.tenant_id == current.tenant_id, or_(*dup_conds)
            ).limit(1)
        ):
            skipped.append({"row": i, "reason": f"重复候选人({name})"})
            continue
        org_id = None
        if rec.get("org_name"):
            org = db.scalar(
                select(OrgUnit).where(
                    OrgUnit.tenant_id == current.tenant_id,
                    OrgUnit.name == rec["org_name"],
                ).limit(1)
            )
            if org is None:
                skipped.append({"row": i, "reason": f"组织不存在:{rec['org_name']}"})
                continue
            org_id = org.id
        tags = [t for t in (rec.get("tags") or "").split(";") if t][:20]
        c = Candidate(
            tenant_id=current.tenant_id,
            name=name[:128],
            email_hash=e_hash,
            phone_hash=p_hash,
            display_email=email,
            display_phone=phone,
            tags=tags or None,
            org_unit_id=org_id,
            last_active_at=_utcnow_naive(),
        )
        db.add(c)
        db.flush()
        try:
            dedup_svc.flag_suspects(db, c)
        except Exception:  # noqa: BLE001
            logger.exception("dedup flag 失败 candidate=%s", c.id)
        seen_hashes |= row_hashes
        created += 1
    write_audit(
        db,
        actor=current,
        entity_type="candidate",
        entity_id="bulk",
        action="import",
        detail={"created": created, "skipped": len(skipped), "file": file.filename},
        request=request,
    )
    db.commit()
    return {"created": created, "skipped": skipped, "total": len(rows[:2000])}


# ---------- 职业能力画像(Workday skill-level 模式:多源聚合+来源分解) ----------


class CapabilitySource(BaseModel):
    kind: str  # match | interview | declared
    label: str
    score5: float  # 归一化到 0-5
    detail: str | None = None  # evidence / comment
    at: str | None = None


class CapabilityDim(BaseModel):
    name: str
    avg: float
    min: float
    max: float
    count: int
    sources: list[CapabilitySource]


class CapabilityProfileOut(BaseModel):
    candidate_id: str
    evaluations_total: int
    sources_count: dict[str, int]
    dimensions: list[CapabilityDim]
    skills: list[str]
    strengths: list[str]
    gaps: list[str]


@router.get(
    "/{candidate_id}/capability-profile",
    response_model=CapabilityProfileOut,
    dependencies=[Depends(require_permission(PERM_VIEW_REPORTS))],
)
def capability_profile(
    candidate_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> CapabilityProfileOut:
    """人级能力画像:聚合该候选人全部评估数据为稳定维度画像。

    三源(对标 Workday skill level 多源算分):
    - match:     人岗匹配逐维 level(0-4 → 归一 0-5)+ 证据
    - interview: 评分卡逐维 score(1-5)+ 评语
    - declared:  候选人技能自评/简历提取 level(1-5)
    """
    c = _get_candidate(db, candidate_id=candidate_id, current=current)
    buckets: dict[str, list[CapabilitySource]] = {}
    counts = {"match": 0, "interview": 0, "declared": 0}

    def put(name: str, src: CapabilitySource) -> None:
        name = (name or "").strip()[:64]
        if name:
            buckets.setdefault(name, []).append(src)

    # 源一:人岗匹配逐维
    rows = db.execute(
        select(ResumeJobMatch, Job.title)
        .join(Resume, ResumeJobMatch.resume_id == Resume.id)
        .outerjoin(Job, ResumeJobMatch.job_id == Job.id)
        .where(Resume.candidate_id == c.id, ResumeJobMatch.tenant_id == current.tenant_id)
        .order_by(ResumeJobMatch.created_at.desc())
        .limit(50)
    ).all()
    for m, job_title in rows:
        crits = m.criteria if isinstance(m.criteria, list) else []
        for cr in crits:
            if not isinstance(cr, dict):
                continue
            lvl = cr.get("level")
            if not isinstance(lvl, (int, float)):
                continue
            counts["match"] += 1
            put(
                str(cr.get("name") or cr.get("key") or "未命名维度"),
                CapabilitySource(
                    kind="match",
                    label=f"人岗匹配 · {job_title or '岗位'}",
                    score5=round(float(lvl) / 4 * 5, 2),
                    detail=(str(cr.get("evidence"))[:200] or None),
                    at=m.created_at.isoformat() if m.created_at else None,
                ),
            )

    # 源二:评分卡逐维反馈
    fb_rows = db.execute(
        select(InterviewFeedback, InterviewScorecard)
        .join(InterviewScorecard, InterviewFeedback.scorecard_id == InterviewScorecard.id)
        .where(
            InterviewScorecard.candidate_id == c.id,
            InterviewFeedback.tenant_id == current.tenant_id,
        )
        .order_by(InterviewFeedback.submitted_at.desc())
        .limit(100)
    ).all()
    if fb_rows:
        from app.services import scorecards as sc_svc

        uids = list({f.interviewer_user_id for f, _ in fb_rows})
        names = sc_svc.user_display_names(db, current.tenant_id, uids)
        for f, sc in fb_rows:
            name_map = {
                cr.get("key"): cr.get("name")
                for cr in (sc.criteria or [])
                if isinstance(cr, dict)
            }
            for s in (f.scores or []):
                if not isinstance(s, dict) or not isinstance(s.get("score"), (int, float)):
                    continue
                counts["interview"] += 1
                put(
                    str(name_map.get(s.get("criterion")) or s.get("criterion") or "未命名维度"),
                    CapabilitySource(
                        kind="interview",
                        label=f"{sc.title} · {names.get(f.interviewer_user_id, '面试官')}",
                        score5=round(float(s["score"]), 2),
                        detail=(str(s.get("comment"))[:200] or None),
                        at=f.submitted_at.isoformat() if f.submitted_at else None,
                    ),
                )

    # 源三:声明技能(manual/resume 自评)
    skill_rows = db.execute(
        select(CandidateSkill, Skill.name)
        .join(Skill, CandidateSkill.skill_id == Skill.id)
        .where(CandidateSkill.candidate_id == c.id)
    ).all()
    for cs, sname in skill_rows:
        counts["declared"] += 1
        put(
            sname,
            CapabilitySource(
                kind="declared",
                label="技能档案" + ("(简历提取)" if cs.source == "resume" else "(人工标注)"),
                score5=float(cs.level or 3),
                at=cs.created_at.isoformat() if cs.created_at else None,
            ),
        )

    dims = [
        CapabilityDim(
            name=n,
            avg=round(sum(s.score5 for s in srcs) / len(srcs), 2),
            min=min(s.score5 for s in srcs),
            max=max(s.score5 for s in srcs),
            count=len(srcs),
            sources=sorted(srcs, key=lambda s: s.at or "", reverse=True),
        )
        for n, srcs in buckets.items()
    ]
    dims.sort(key=lambda d: (-d.count, -d.avg))

    skill_names = [n for _, n in skill_rows]
    tag_names = list(c.tags or [])
    latest_skills: list[str] = []
    res = db.scalars(
        select(Resume)
        .where(Resume.candidate_id == c.id)
        .order_by(Resume.created_at.desc())
        .limit(3)
    ).all()
    for r in res:
        latest_skills += [str(s) for s in (r.parsed_data or {}).get("skills") or []][:10]
    skills = list(dict.fromkeys(skill_names + tag_names + latest_skills))[:30]

    multi = [d for d in dims if d.count > 0]
    strengths = [d.name for d in sorted(multi, key=lambda d: -d.avg)[:3]]
    gaps = [d.name for d in sorted(multi, key=lambda d: d.avg)[:3] if d.avg < 3.5]

    return CapabilityProfileOut(
        candidate_id=c.id,
        evaluations_total=sum(counts.values()),
        sources_count=counts,
        dimensions=dims,
        skills=skills,
        strengths=strengths,
        gaps=gaps,
    )
