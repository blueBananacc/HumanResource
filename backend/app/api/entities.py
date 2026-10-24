"""通用实体检索与操作(MintHCM AI Tools 移植 + T4 扩展)。

白名单注册表驱动,**不暴露通用 ORM**——每个 entity 有独立 query builder,
tenant 过滤 + 按实体 org scope + 权限映射,统一返回 {id,label,secondary}。

- ``GET /entities`` → 可检索实体清单(assistant 工具发现)
- ``GET /entities/{entity}/search?q=&limit=``
- ``GET /entities/{entity}/count?q=``
- ``GET /entities/{entity}/aggregate?agg=&field=&q=`` → sum/avg/min/max/count
- ``PATCH /entities/{entity}/{id}`` → 白名单字段更新(写权限 + audit)
- ``GET /entities/{entity}/{id}/relations`` → 关联实体穿透
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Announcement,
    Candidate,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    Goal,
    HrCase,
    HrCaseUpdate,
    Interview,
    InterviewTurn,
    Job,
    JobApplication,
    JobOffer,
    KnowledgeArticle,
    LeaveApplication,
    Reservation,
    Resume,
    ResumeJobMatch,
    Room,
    SalarySlip,
    Skill,
    TalentCampaign,
    TalentCampaignSend,
    User,
)
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_MANAGE_OFFERS,
    PERM_MANAGE_PAYROLL,
    PERM_MANAGE_SETTINGS,
    PERM_VIEW_REPORTS,
    PERM_WRITE_INTERVIEW,
    PERM_WRITE_JOBS,
    PERM_WRITE_RESUMES,
    apply_org_filter,
    ensure_can_see,
    get_org_scope,
    has_permission,
    visible_employee_ids,
    visible_org_ids,
)

router = APIRouter(prefix="/entities", tags=["entity-search"])


class EntityHit(BaseModel):
    id: str
    label: str
    secondary: str | None = None


class EntityMeta(BaseModel):
    key: str
    label: str
    description: str
    searchable_fields: list[str]


def _like(col, q):
    return col.ilike(f"%{q}%")


# builder 统一签名 (db, u, scope, q, extra=None):extra 是追加到 SELECT 的
# 额外套件列(目前只有 aggregate 用到,把聚合列一起带进子查询)。
def _talents(db, u, scope, q, extra=None):
    stmt = select(Candidate.id, Candidate.name.label("label"), Candidate.display_email.label("secondary"), *(extra or [])).where(
        Candidate.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(Candidate.display_email, q)))
    return stmt


def _jobs(db, u, scope, q, extra=None):
    stmt = select(Job.id, Job.title.label("label"), Job.status.label("secondary"), *(extra or [])).where(
        Job.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Job.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Job.title, q))
    return stmt


def _employees(db, u, scope, q, extra=None):
    stmt = select(Employee.id, Employee.name.label("label"), Employee.employee_no.label("secondary"), *(extra or [])).where(
        Employee.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Employee.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Employee.name, q), _like(Employee.employee_no, q)))
    return stmt


def _interviews(db, u, scope, q, extra=None):
    stmt = (
        select(Interview.id, Candidate.name.label("label"), Job.title.label("secondary"), *(extra or []))
        .join(Candidate, Candidate.id == Interview.candidate_id)
        .join(Job, Job.id == Interview.job_id)
        .where(Interview.tenant_id == u.tenant_id)
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(Job.title, q)))
    return stmt


def _applications(db, u, scope, q, extra=None):
    stmt = (
        select(JobApplication.id, Candidate.name.label("label"), JobApplication.stage.label("secondary"), *(extra or []))
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(JobApplication.tenant_id == u.tenant_id)
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(JobApplication.stage, q)))
    return stmt


def _rooms(db, u, scope, q, extra=None):
    stmt = select(Room.id, Room.name.label("label"), Room.location.label("secondary"), *(extra or [])).where(
        Room.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Room.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Room.name, q))
    return stmt


def _reservations(db, u, scope, q, extra=None):
    # 预约无 org —— 跟随所约对象:room 看 Room.org_unit_id;
    # interview 预约看候选人 org(标题常含候选人名,与 ics.py 同一收敛口径);
    # resource 预约无 org 维度,按公共可见。两个 org 来源分别套
    # apply_org_filter:任一非空 org 出界即隐藏。
    stmt = (
        select(Reservation.id, Reservation.title.label("label"), Reservation.status.label("secondary"), *(extra or []))
        .where(Reservation.tenant_id == u.tenant_id)
        .outerjoin(Room, Room.id == Reservation.room_id)
        .outerjoin(Interview, Interview.id == Reservation.interview_id)
        .outerjoin(Candidate, Candidate.id == Interview.candidate_id)
    )
    stmt = apply_org_filter(stmt, org_column=Room.org_unit_id, scope=scope)
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Reservation.title, q))
    return stmt


def _announcements(db, u, scope, q, extra=None):
    stmt = select(Announcement.id, Announcement.title.label("label"), Announcement.status.label("secondary"), *(extra or [])).where(
        Announcement.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Announcement.audience_org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Announcement.title, q))
    return stmt


def _cases(db, u, scope, q, extra=None):
    stmt = (
        select(HrCase.id, HrCase.subject.label("label"), HrCase.status.label("secondary"), *(extra or []))
        .where(HrCase.tenant_id == u.tenant_id)
    )
    # 与 /api/cases 列表同口径:行级 scope 按汇报线收敛(admin/hr 不限)
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(HrCase.requester_employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(_like(HrCase.subject, q))
    return stmt


def _requests(db, u, scope, q, extra=None):
    stmt = (
        select(EmployeeRequest.id, EmployeeRequest.subject.label("label"), EmployeeRequest.status.label("secondary"), *(extra or []))
        .where(EmployeeRequest.tenant_id == u.tenant_id)
    )
    # 与 /api/requests 列表同口径
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(EmployeeRequest.employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(_like(EmployeeRequest.subject, q))
    return stmt


def _knowledge(db, u, scope, q, extra=None):
    stmt = (
        select(KnowledgeArticle.id, KnowledgeArticle.title.label("label"), KnowledgeArticle.category.label("secondary"), *(extra or []))
        .where(KnowledgeArticle.tenant_id == u.tenant_id, KnowledgeArticle.status == "published")
    )
    if q:
        stmt = stmt.where(_like(KnowledgeArticle.title, q))
    return stmt


def _skills(db, u, scope, q, extra=None):
    stmt = select(Skill.id, Skill.name.label("label"), Skill.category.label("secondary"), *(extra or [])).where(
        Skill.tenant_id == u.tenant_id
    )
    if q:
        stmt = stmt.where(_like(Skill.name, q))
    return stmt


def _expenses(db, u, scope, q, extra=None):
    stmt = (
        select(ExpenseClaim.id, ExpenseClaim.title.label("label"), ExpenseClaim.status.label("secondary"), *(extra or []))
        .where(ExpenseClaim.tenant_id == u.tenant_id)
    )
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(ExpenseClaim.employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(_like(ExpenseClaim.title, q))
    return stmt


def _payslips(db, u, scope, q, extra=None):
    stmt = (
        select(SalarySlip.id, Employee.name.label("label"), SalarySlip.period.label("secondary"), *(extra or []))
        .join(Employee, Employee.id == SalarySlip.employee_id)
        .where(SalarySlip.tenant_id == u.tenant_id)
    )
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(SalarySlip.employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(or_(_like(Employee.name, q), _like(SalarySlip.period, q)))
    return stmt


def _matches(db, u, scope, q, extra=None):
    stmt = (
        select(ResumeJobMatch.id, Candidate.name.label("label"), Job.title.label("secondary"), *(extra or []))
        .join(Resume, Resume.id == ResumeJobMatch.resume_id)
        .join(Candidate, Candidate.id == Resume.candidate_id)
        .join(Job, Job.id == ResumeJobMatch.job_id)
        .where(ResumeJobMatch.tenant_id == u.tenant_id)
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(Job.title, q)))
    return stmt


def _campaigns(db, u, scope, q, extra=None):
    stmt = (
        select(TalentCampaign.id, TalentCampaign.name.label("label"), TalentCampaign.status.label("secondary"), *(extra or []))
        .where(TalentCampaign.tenant_id == u.tenant_id)
    )
    if q:
        stmt = stmt.where(_like(TalentCampaign.name, q))
    return stmt


ENTITIES: dict[str, dict] = {
    "talents": {
        "label": "候选人",
        "description": "人才库候选人(姓名/邮箱搜索)",
        "fields": ["name", "email"],
        "perm": PERM_VIEW_REPORTS,
        "build": _talents,
        "feature": None,
    },
    "jobs": {
        "label": "岗位",
        "description": "招聘岗位(标题搜索)",
        "fields": ["title", "status"],
        "perm": PERM_VIEW_REPORTS,
        "build": _jobs,
        "feature": None,
    },
    "employees": {
        "label": "员工",
        "description": "员工档案(姓名/工号搜索)",
        "fields": ["name", "employee_no"],
        "perm": PERM_MANAGE_EMPLOYEES,
        "build": _employees,
        "feature": "hr.suite",
    },
    "interviews": {
        "label": "面试",
        "description": "AI 面试(候选人名/岗位名搜索)",
        "fields": ["candidate_name", "job_title"],
        "perm": PERM_VIEW_REPORTS,
        "build": _interviews,
        "feature": None,
    },
    "applications": {
        "label": "管道卡",
        "description": "招聘管道(候选人名/阶段搜索)",
        "fields": ["candidate_name", "stage"],
        "perm": PERM_VIEW_REPORTS,
        "build": _applications,
        "feature": "hr.suite",
    },
    "rooms": {
        "label": "会议室",
        "description": "会议室(名称搜索)",
        "fields": ["name", "location"],
        "perm": PERM_VIEW_REPORTS,
        "build": _rooms,
        "feature": "hr.suite",
    },
    "reservations": {
        "label": "预约",
        "description": "会议室/资源预约(标题搜索)",
        "fields": ["title", "status"],
        "perm": PERM_VIEW_REPORTS,
        "build": _reservations,
        "feature": "hr.suite",
    },
    "announcements": {
        "label": "公告",
        "description": "公司公告(标题搜索)",
        "fields": ["title", "status"],
        "perm": PERM_VIEW_REPORTS,
        "build": _announcements,
        "feature": "hr.suite",
    },
    "cases": {
        "label": "工单",
        "description": "HR 工单(主题搜索)",
        "fields": ["subject", "status"],
        "perm": PERM_MANAGE_EMPLOYEES,
        "build": _cases,
        "feature": "hr.suite",
    },
    "requests": {
        "label": "员工申请",
        "description": "员工申请(主题搜索)",
        "fields": ["subject", "status"],
        "perm": PERM_MANAGE_EMPLOYEES,
        "build": _requests,
        "feature": "hr.suite",
    },
    "knowledge": {
        "label": "知识文章",
        "description": "知识库已发布文章(标题搜索)",
        "fields": ["title", "category"],
        "perm": PERM_VIEW_REPORTS,
        "build": _knowledge,
        "feature": "hr.suite",
    },
    "skills": {
        "label": "技能",
        "description": "技能字典(名称搜索)",
        "fields": ["name", "category"],
        "perm": PERM_VIEW_REPORTS,
        "build": _skills,
        "feature": "hr.suite",
    },
    "expenses": {
        "label": "报销单",
        "description": "员工报销申请(标题/状态搜索)",
        "fields": ["title", "status"],
        # 与 /api/expenses/claims 同口径:view_reports 进门,行级 scope 走
        # visible_employee_ids(admin/hr 全量,其余仅直属下属+自己)。
        "perm": PERM_VIEW_REPORTS,
        "build": _expenses,
        "feature": "hr.suite",
        "agg_fields": {"total": ExpenseClaim.total},
    },
    "payslips": {
        "label": "工资条",
        "description": "工资条(员工名/期间搜索)",
        "fields": ["employee_name", "period"],
        "perm": PERM_MANAGE_PAYROLL,
        "build": _payslips,
        "feature": "hr.suite",
        "agg_fields": {"gross": SalarySlip.gross, "net": SalarySlip.net},
    },
    "matches": {
        "label": "匹配结果",
        "description": "简历↔岗位 AI 匹配(候选人名/岗位名搜索)",
        "fields": ["candidate_name", "job_title"],
        "perm": PERM_VIEW_REPORTS,
        "build": _matches,
        "feature": None,
        "agg_fields": {"score": ResumeJobMatch.score},
    },
    "campaigns": {
        "label": "外联活动",
        "description": "人才再激活外联活动(名称/状态搜索)",
        "fields": ["name", "status"],
        "perm": PERM_VIEW_REPORTS,
        "build": _campaigns,
        "feature": "hr.suite",
        "agg_fields": {
            "total": TalentCampaign.total,
            "sent_count": TalentCampaign.sent_count,
            "failed_count": TalentCampaign.failed_count,
        },
    },
}


def _spec(entity: str, current: User) -> dict:
    spec = ENTITIES.get(entity)
    if spec is None:
        raise HTTPException(404, f"未知实体,可用:{','.join(ENTITIES)}")
    if not has_permission(current, spec["perm"]):
        raise HTTPException(403, "权限不足")
    return spec


@router.get("", response_model=list[EntityMeta])
def list_entities(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return [
        EntityMeta(
            key=k,
            label=spec["label"],
            description=spec["description"],
            searchable_fields=spec["fields"],
        )
        for k, spec in ENTITIES.items()
        if has_permission(current, spec["perm"])
        and (not spec["feature"] or is_feature_enabled(db, spec["feature"]))
    ]


@router.get("/{entity}/search", response_model=list[EntityHit])
def search_entities(
    entity: str,
    q: str | None = Query(None, max_length=128),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    spec = _spec(entity, current)
    if spec["feature"]:
        require_feature(spec["feature"])()
    scope = get_org_scope(db, current)
    stmt = spec["build"](db, current, scope, q.strip() if q else None)
    rows = db.execute(stmt.limit(limit)).all()
    return [EntityHit(id=r.id, label=r.label, secondary=r.secondary) for r in rows]


@router.get("/{entity}/count")
def count_entities(
    entity: str,
    q: str | None = Query(None, max_length=128),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    spec = _spec(entity, current)
    if spec["feature"]:
        require_feature(spec["feature"])()
    scope = get_org_scope(db, current)
    stmt = spec["build"](db, current, scope, q.strip() if q else None)
    n = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    return {"count": int(n)}


# ============ T4:aggregate / update / relations ============

AGG_FNS = {"sum": func.sum, "avg": func.avg, "min": func.min, "max": func.max}


@router.get("/{entity}/aggregate")
def aggregate_entities(
    entity: str,
    agg: str = Query(..., pattern="^(sum|avg|min|max|count)$"),
    field: str | None = Query(None, max_length=64),
    q: str | None = Query(None, max_length=128),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """对过滤后的实体集做聚合 — 等价 MintHCM SumRecordsTool。"""
    spec = _spec(entity, current)
    if spec["feature"]:
        require_feature(spec["feature"])()
    agg_fields: dict = spec.get("agg_fields") or {}
    if agg == "count":
        # count 不需要字段 — 与 /count 同口径但聚合语义
        fn = func.count
        col = None
    else:
        if not field or field not in agg_fields:
            raise HTTPException(
                400, f"field 必填且须在白名单:{','.join(agg_fields) or '该实体不支持聚合'}"
            )
        col = agg_fields[field]
        fn = AGG_FNS[agg]
    scope = get_org_scope(db, current)
    extra = [col.label("agg_val")] if col is not None else []
    stmt = spec["build"](db, current, scope, q.strip() if q else None, extra=extra)
    sub = stmt.subquery()
    sel = func.count(sub.c.id) if col is None else fn(sub.c.agg_val)
    # matched 随行数一起返回 — 空集合时 LLM 能区分"没数据"与"聚合值为 0/空"
    value, matched = db.execute(
        select(sel, func.count(sub.c.id))
    ).one()
    return {
        "entity": entity,
        "agg": agg,
        "field": field,
        "value": value,
        "matched": int(matched),
    }


# ---------- update ----------

class UpdateIn(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)


def _v_str(v: Any, max_len: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise HTTPException(400, "值必须是非空字符串")
    return v.strip()[:max_len]


def _v_str_null(v: Any, max_len: int) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise HTTPException(400, "值必须是字符串或 null")
    return v.strip()[:max_len] or None


def _v_int(v: Any, lo: int, hi: int) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= hi):
        raise HTTPException(400, f"值必须是 {lo}-{hi} 的整数")
    return v


def _v_bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise HTTPException(400, "值必须是布尔")
    return v


def _v_strlist(v: Any, max_len: int, max_items: int) -> list[str]:
    if not isinstance(v, list) or len(v) > max_items or not all(isinstance(x, str) for x in v):
        raise HTTPException(400, f"值必须是不多于 {max_items} 项的字符串数组")
    return [x.strip()[:max_len] for x in v if x.strip()]


def _v_choice(v: Any, choices: tuple) -> str:
    if not isinstance(v, str) or v not in choices:
        raise HTTPException(400, f"值必须是:{','.join(choices)}")
    return v


# writable: {entity: {"perm":..., "get": fn(db,u,scope,id)->row, "fields": {name: (attr, validator)}}}
# validator 签名为 (value)->cleaned;raise HTTPException(400) 拒绝非法值。
def _get_candidate_row(db, u, scope, rid):
    row = db.get(Candidate, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "候选人不存在")
    ensure_can_see(scope, row.org_unit_id)
    return row


def _get_job_row(db, u, scope, rid):
    row = db.get(Job, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(scope, row.org_unit_id)
    return row


def _get_employee_row(db, u, scope, rid):
    row = db.get(Employee, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "员工不存在")
    ensure_can_see(scope, row.org_unit_id)
    return row


def _get_case_row(db, u, scope, rid):
    row = db.get(HrCase, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "工单不存在")
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None and (row.requester_employee_id or "__none__") not in emp_ids:
        raise HTTPException(404, "工单不存在")
    return row


def _get_skill_row(db, u, scope, rid):
    row = db.get(Skill, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "技能不存在")
    return row


def _get_room_row(db, u, scope, rid):
    row = db.get(Room, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "会议室不存在")
    ensure_can_see(scope, row.org_unit_id)
    return row


def _get_interview_row(db, u, scope, rid):
    row = db.get(Interview, rid)
    if row is None or row.tenant_id != u.tenant_id:
        raise HTTPException(404, "面试不存在")
    cand = db.get(Candidate, row.candidate_id)
    ensure_can_see(scope, cand.org_unit_id if cand else None)
    return row


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _job_after_change(db, row: Job, changed: dict[str, Any], current: User) -> None:
    """岗位内容变更治理 — 与 ``PUT /jobs/{id}`` 同口径(事务内)。

    level/description 任一变化即落 JobVersion 快照,已发布岗位回到 draft
    重新走审批 — 不允许 AI 静默改已发布 JD。
    """
    if "level" not in changed and "description" not in changed:
        return
    from app.services.job_governance import snapshot_version

    snapshot_version(db, job=row, author=current, change_kind="content_update")
    if row.publish_status == "published":
        row.publish_status = "draft"
        row.submitted_by = None
        row.submitted_at = None


def _job_after_commit(db, row: Job, changed: dict[str, Any], current: User) -> None:
    """岗位变更后的异步副作用(commit 后) — 与 ``update_job`` /
    ``update_job_status`` 同口径:

    - status → open(重开)→ 重投匹配评估
    - 岗位仍 open 时内容字段变更 → 重投匹配评估
    """
    content_changed = "level" in changed or "description" in changed
    if changed.get("status") == "open" or (content_changed and row.status == "open"):
        from app.api.jobs import _enqueue_match_eval

        _enqueue_match_eval(row.id, current.id)


def _candidate_after_change(db, row: Candidate, changed: dict[str, Any], current: User) -> None:
    """候选人写操作刷新 last_active_at — 与手写端点
    (``set_tags`` / ``add_blacklist`` 的 ``_touch_last_active``)同口径,
    保持"最近活跃"排序不失真。"""
    row.last_active_at = _now()


WRITABLE: dict[str, dict] = {
    "talents": {
        "perm": PERM_WRITE_RESUMES,
        "get": _get_candidate_row,
        "audit": "candidate",
        "after_change": _candidate_after_change,
        "fields": {
            "display_email": ("display_email", lambda v: _v_str_null(v, 256)),
            "display_phone": ("display_phone", lambda v: _v_str_null(v, 64)),
            "status": ("status", lambda v: _v_choice(v, Candidate.STATUSES)),
            "tags": ("tags", lambda v: _v_strlist(v, 32, 20)),
        },
    },
    "jobs": {
        "perm": PERM_WRITE_JOBS,
        "get": _get_job_row,
        "audit": "job",
        "after_change": _job_after_change,
        "after_commit": _job_after_commit,
        "fields": {
            "level": ("level", lambda v: _v_choice(v, Job.LEVELS)),
            "description": ("description", lambda v: _v_str(v, 20000)),
            "status": ("status", lambda v: _v_choice(v, Job.STATUSES)),
        },
    },
    "employees": {
        "perm": PERM_MANAGE_EMPLOYEES,
        "get": _get_employee_row,
        "audit": "employee",
        "fields": {
            "phone": ("phone", lambda v: _v_str_null(v, 64)),
            "work_email": ("work_email", lambda v: _v_str_null(v, 256)),
            "designation": ("designation", lambda v: _v_str_null(v, 128)),
        },
    },
    "cases": {
        "perm": PERM_MANAGE_EMPLOYEES,
        "get": _get_case_row,
        "audit": "hr_case",
        "fields": {
            "priority": ("priority", lambda v: _v_choice(v, ("low", "normal", "high", "urgent"))),
        },
    },
    "skills": {
        "perm": PERM_MANAGE_SETTINGS,
        "get": _get_skill_row,
        "audit": "skill",
        "fields": {
            "category": ("category", lambda v: _v_choice(v, Skill.CATEGORIES)),
            "description": ("description", lambda v: _v_str_null(v, 512)),
        },
    },
    "rooms": {
        "perm": PERM_MANAGE_SETTINGS,
        "get": _get_room_row,
        "audit": "room",
        "fields": {
            "name": ("name", lambda v: _v_str(v, 64)),
            "location": ("location", lambda v: _v_str_null(v, 128)),
            "capacity": ("capacity", lambda v: _v_int(v, 0, 1000)),
            "equipment": ("equipment", lambda v: _v_strlist(v, 64, 20)),
            "is_active": ("is_active", lambda v: _v_bool(v)),
        },
    },
    "interviews": {
        "perm": PERM_WRITE_INTERVIEW,
        "get": _get_interview_row,
        "audit": "interview",
        "fields": {
            "notify_email": ("notify_email", lambda v: _v_str_null(v, 256)),
        },
    },
}


@router.patch("/{entity}/{rid}")
def update_entity(
    entity: str,
    rid: str,
    body: UpdateIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """白名单字段更新 — 等价 MintHCM UpdateRecordTool(收窄到安全字段)。"""
    spec = _spec(entity, current)
    if spec["feature"]:
        require_feature(spec["feature"])()
    w = WRITABLE.get(entity)
    if w is None:
        raise HTTPException(400, "该实体不支持更新")
    if not has_permission(current, w["perm"]):
        raise HTTPException(403, "权限不足")
    if not body.fields:
        raise HTTPException(400, "fields 不能为空")
    unknown = set(body.fields) - set(w["fields"])
    if unknown:
        raise HTTPException(400, f"字段不可写:{','.join(sorted(unknown))};可写:{','.join(w['fields'])}")
    scope = get_org_scope(db, current)
    row = w["get"](db, current, scope, rid)
    changed: dict[str, Any] = {}
    for name, raw in body.fields.items():
        attr, validator = w["fields"][name]
        cleaned = validator(raw)
        if getattr(row, attr) != cleaned:
            setattr(row, attr, cleaned)
            changed[name] = cleaned
    if not changed:
        return {"ok": True, "changed": changed}
    # updated_at 维护 — Job/Employee/HrCase 等模型带该列,
    # 与各自手写 PUT 端点同口径。
    if hasattr(row, "updated_at"):
        row.updated_at = _now()
    # after_change:事务内治理(如岗位版本快照 / 发布状态回退)
    if w.get("after_change") is not None:
        w["after_change"](db, row, changed, current)
    write_audit(db, actor=current, entity_type=w["audit"], entity_id=rid, action="update", detail={"fields": changed}, request=request)
    db.commit()
    # after_commit:异步副作用(如岗位重开后重投匹配评估),
    # 必须在 commit 后,worker 才能读到新数据。
    if w.get("after_commit") is not None:
        w["after_commit"](db, row, changed, current)
    return {"ok": True, "changed": changed}


# ---------- relations ----------

def _hit(rid, label, secondary=None):
    return {"id": rid, "label": label, "secondary": secondary}


def _rel_candidate(db, u, scope, rid):
    # 关联实体也要套 org scope(与 /api/applications、/api/interviews 的
    # Job+Cand 双过滤同口径)—— 父记录可见不代表其关联项都该暴露标题。
    apps_stmt = (
        select(JobApplication.id, Job.title, JobApplication.stage)
        .join(Job, Job.id == JobApplication.job_id)
        .where(JobApplication.tenant_id == u.tenant_id, JobApplication.candidate_id == rid)
    )
    apps_stmt = apply_org_filter(apps_stmt, org_column=Job.org_unit_id, scope=scope)
    apps = db.execute(apps_stmt.limit(10)).all()
    ivs_stmt = (
        select(Interview.id, Job.title, Interview.status)
        .join(Job, Job.id == Interview.job_id)
        .where(Interview.tenant_id == u.tenant_id, Interview.candidate_id == rid)
    )
    ivs_stmt = apply_org_filter(ivs_stmt, org_column=Job.org_unit_id, scope=scope)
    ivs = db.execute(ivs_stmt.limit(10)).all()
    resumes_n = db.scalar(
        select(func.count(Resume.id)).where(Resume.tenant_id == u.tenant_id, Resume.candidate_id == rid)
    ) or 0
    out = [
        {"key": "applications", "label": "管道申请", "items": [_hit(r.id, r[1], r.stage) for r in apps]},
        {"key": "interviews", "label": "面试", "items": [_hit(r.id, r[1], r.status) for r in ivs]},
    ]
    # Offer 数据按 hiring.py 口径仅 manage_offers 可见 —— view_reports 不能穿透
    if has_permission(u, PERM_MANAGE_OFFERS):
        offers = db.execute(
            select(JobOffer.id, JobOffer.status).where(
                JobOffer.tenant_id == u.tenant_id, JobOffer.candidate_id == rid
            ).limit(10)
        ).all()
        out.append({"key": "offers", "label": "Offer", "items": [_hit(r.id, r.status) for r in offers]})
    out.append({"key": "resumes", "label": "简历", "count": resumes_n})
    return out


def _rel_job(db, u, scope, rid):
    # 候选人侧 org scope —— 同 _applications/_interviews builder 口径
    apps_stmt = (
        select(JobApplication.id, Candidate.name, JobApplication.stage)
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(JobApplication.tenant_id == u.tenant_id, JobApplication.job_id == rid)
    )
    apps_stmt = apply_org_filter(apps_stmt, org_column=Candidate.org_unit_id, scope=scope)
    apps = db.execute(apps_stmt.limit(10)).all()
    ivs_stmt = (
        select(Interview.id, Candidate.name, Interview.status)
        .join(Candidate, Candidate.id == Interview.candidate_id)
        .where(Interview.tenant_id == u.tenant_id, Interview.job_id == rid)
    )
    ivs_stmt = apply_org_filter(ivs_stmt, org_column=Candidate.org_unit_id, scope=scope)
    ivs = db.execute(ivs_stmt.limit(10)).all()
    top_stmt = (
        select(ResumeJobMatch.id, Candidate.name, ResumeJobMatch.score)
        .join(Resume, Resume.id == ResumeJobMatch.resume_id)
        .join(Candidate, Candidate.id == Resume.candidate_id)
        .where(
            ResumeJobMatch.tenant_id == u.tenant_id,
            ResumeJobMatch.job_id == rid,
            ResumeJobMatch.score.is_not(None),
        )
    )
    top_stmt = apply_org_filter(top_stmt, org_column=Candidate.org_unit_id, scope=scope)
    top = db.execute(top_stmt.order_by(ResumeJobMatch.score.desc()).limit(5)).all()
    return [
        {"key": "applications", "label": "管道申请", "items": [_hit(r.id, r[1], r.stage) for r in apps]},
        {"key": "interviews", "label": "面试", "items": [_hit(r.id, r[1], r.status) for r in ivs]},
        {"key": "top_matches", "label": "高分匹配", "items": [_hit(r.id, r[1], r.score) for r in top]},
    ]


def _rel_employee(db, u, scope, rid):
    reqs = db.execute(
        select(EmployeeRequest.id, EmployeeRequest.subject, EmployeeRequest.status)
        .where(EmployeeRequest.tenant_id == u.tenant_id, EmployeeRequest.employee_id == rid)
        .limit(10)
    ).all()
    cases = db.execute(
        select(HrCase.id, HrCase.subject, HrCase.status)
        .where(HrCase.tenant_id == u.tenant_id, HrCase.requester_employee_id == rid)
        .limit(10)
    ).all()
    leaves = db.execute(
        select(LeaveApplication.id, LeaveApplication.status)
        .where(LeaveApplication.tenant_id == u.tenant_id, LeaveApplication.employee_id == rid)
        .limit(10)
    ).all()
    goals = db.execute(
        select(Goal.id, Goal.title, Goal.status)
        .where(Goal.tenant_id == u.tenant_id, Goal.employee_id == rid)
        .limit(10)
    ).all()
    return [
        {"key": "requests", "label": "员工申请", "items": [_hit(r.id, r[1], r.status) for r in reqs]},
        {"key": "cases", "label": "工单", "items": [_hit(r.id, r[1], r.status) for r in cases]},
        {"key": "leave", "label": "请假", "items": [_hit(r.id, "请假", r.status) for r in leaves]},
        {"key": "goals", "label": "目标", "items": [_hit(r.id, r[1], r.status) for r in goals]},
    ]


def _rel_application(db, u, scope, rid):
    row = db.get(JobApplication, rid)
    cand = db.get(Candidate, row.candidate_id) if row else None
    job = db.get(Job, row.job_id) if row else None
    # 关联两侧实体分别过 tenant + org 可见性(applications.py _check_scope 同口径),
    # 出界的不报 404,直接从穿透结果里剔除
    return [
        {"key": "candidate", "label": "候选人", "items": [
            _hit(cand.id, cand.name) for cand in [cand]
            if cand and cand.tenant_id == u.tenant_id and visible_org_ids(scope, cand.org_unit_id)
        ]},
        {"key": "job", "label": "岗位", "items": [
            _hit(job.id, job.title) for job in [job]
            if job and job.tenant_id == u.tenant_id and visible_org_ids(scope, job.org_unit_id)
        ]},
    ]


def _rel_interview(db, u, scope, rid):
    row = db.get(Interview, rid)
    cand = db.get(Candidate, row.candidate_id) if row else None
    job = db.get(Job, row.job_id) if row else None
    turns = db.scalar(
        select(func.count(InterviewTurn.id)).where(InterviewTurn.interview_id == rid)
    ) or 0
    return [
        {"key": "candidate", "label": "候选人", "items": [
            _hit(cand.id, cand.name) for cand in [cand]
            if cand and cand.tenant_id == u.tenant_id and visible_org_ids(scope, cand.org_unit_id)
        ]},
        {"key": "job", "label": "岗位", "items": [
            _hit(job.id, job.title) for job in [job]
            if job and job.tenant_id == u.tenant_id and visible_org_ids(scope, job.org_unit_id)
        ]},
        {"key": "turns", "label": "问答轮次", "count": turns},
    ]


def _rel_case(db, u, scope, rid):
    row = db.get(HrCase, rid)
    emp = db.get(Employee, row.requester_employee_id) if row and row.requester_employee_id else None
    updates = db.scalar(
        select(func.count(HrCaseUpdate.id)).where(
            HrCaseUpdate.tenant_id == u.tenant_id, HrCaseUpdate.case_id == rid
        )
    ) or 0
    return [
        {"key": "requester", "label": "提交人", "items": [
            _hit(emp.id, emp.name, emp.employee_no) for emp in [emp]
            if emp and emp.tenant_id == u.tenant_id
        ]},
        {"key": "updates", "label": "留言", "count": updates},
    ]


def _rel_room(db, u, scope, rid):
    # 预约标题常含候选人名 —— 与 _reservations builder 同口径:interview
    # 预约跟随候选人 org,出界即隐藏
    rs_stmt = (
        select(Reservation.id, Reservation.title, Reservation.starts_at)
        .outerjoin(Interview, Interview.id == Reservation.interview_id)
        .outerjoin(Candidate, Candidate.id == Interview.candidate_id)
        .where(Reservation.tenant_id == u.tenant_id, Reservation.room_id == rid, Reservation.status == "active")
    )
    rs_stmt = apply_org_filter(rs_stmt, org_column=Candidate.org_unit_id, scope=scope)
    rs = db.execute(rs_stmt.order_by(Reservation.starts_at.desc()).limit(10)).all()
    return [
        {"key": "reservations", "label": "进行中预约", "items": [_hit(r.id, r[1], str(r[2])) for r in rs]},
    ]


def _rel_campaign(db, u, scope, rid):
    rows = db.execute(
        select(TalentCampaignSend.status, func.count())
        .where(
            TalentCampaignSend.tenant_id == u.tenant_id,
            TalentCampaignSend.campaign_id == rid,
        )
        .group_by(TalentCampaignSend.status)
    ).all()
    return [
        {"key": "sends", "label": "发送明细", "items": [{"id": s, "label": s, "secondary": n} for s, n in rows]},
    ]


RELATIONS: dict[str, dict] = {
    "talents": {"pk": Candidate.id, "fn": _rel_candidate},
    "jobs": {"pk": Job.id, "fn": _rel_job},
    "employees": {"pk": Employee.id, "fn": _rel_employee},
    "applications": {"pk": JobApplication.id, "fn": _rel_application},
    "interviews": {"pk": Interview.id, "fn": _rel_interview},
    "cases": {"pk": HrCase.id, "fn": _rel_case},
    "rooms": {"pk": Room.id, "fn": _rel_room},
    "campaigns": {"pk": TalentCampaign.id, "fn": _rel_campaign},
}


@router.get("/{entity}/{rid}/relations")
def entity_relations(
    entity: str,
    rid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """实体关联穿透 — 等价 MintHCM ListRelationshipsTool。"""
    spec = _spec(entity, current)
    if spec["feature"]:
        require_feature(spec["feature"])()
    rel = RELATIONS.get(entity)
    if rel is None:
        raise HTTPException(400, "该实体不支持关系查询")
    scope = get_org_scope(db, current)
    # 先按 search 口径确认该行可见(org/范围外 404,防探测)
    stmt = spec["build"](db, current, scope, None)
    exists = db.execute(stmt.where(rel["pk"] == rid).limit(1)).first()
    if exists is None:
        raise HTTPException(404, "记录不存在")
    return {"entity": entity, "id": rid, "relations": rel["fn"](db, current, scope, rid)}
