"""通用实体检索(MintHCM AI Tools search_records/count_records 移植)。

白名单注册表驱动,**不暴露通用 ORM**——每个 entity 有独立 query builder,
tenant 过滤 + 按实体 org scope + 权限映射,统一返回 {id,label,secondary}。

- ``GET /entities`` → 可检索实体清单(assistant 工具发现)
- ``GET /entities/{entity}/search?q=&limit=``
- ``GET /entities/{entity}/count?q=``
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Announcement,
    Candidate,
    Employee,
    EmployeeRequest,
    HrCase,
    Interview,
    Job,
    JobApplication,
    KnowledgeArticle,
    Reservation,
    Room,
    Skill,
    User,
)
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    apply_org_filter,
    get_org_scope,
    has_permission,
    visible_employee_ids,
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


def _talents(db, u, scope, q):
    stmt = select(Candidate.id, Candidate.name.label("label"), Candidate.display_email.label("secondary")).where(
        Candidate.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(Candidate.display_email, q)))
    return stmt


def _jobs(db, u, scope, q):
    stmt = select(Job.id, Job.title.label("label"), Job.status.label("secondary")).where(
        Job.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Job.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Job.title, q))
    return stmt


def _employees(db, u, scope, q):
    stmt = select(Employee.id, Employee.name.label("label"), Employee.employee_no.label("secondary")).where(
        Employee.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Employee.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Employee.name, q), _like(Employee.employee_no, q)))
    return stmt


def _interviews(db, u, scope, q):
    stmt = (
        select(Interview.id, Candidate.name.label("label"), Job.title.label("secondary"))
        .join(Candidate, Candidate.id == Interview.candidate_id)
        .join(Job, Job.id == Interview.job_id)
        .where(Interview.tenant_id == u.tenant_id)
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(Job.title, q)))
    return stmt


def _applications(db, u, scope, q):
    stmt = (
        select(JobApplication.id, Candidate.name.label("label"), JobApplication.stage.label("secondary"))
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(JobApplication.tenant_id == u.tenant_id)
    )
    stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(or_(_like(Candidate.name, q), _like(JobApplication.stage, q)))
    return stmt


def _rooms(db, u, scope, q):
    stmt = select(Room.id, Room.name.label("label"), Room.location.label("secondary")).where(
        Room.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Room.org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Room.name, q))
    return stmt


def _reservations(db, u, scope, q):
    # 预约无 org —— 跟随所约对象:room 看 Room.org_unit_id;
    # interview 预约看候选人 org(标题常含候选人名,与 ics.py 同一收敛口径);
    # resource 预约无 org 维度,按公共可见。两个 org 来源分别套
    # apply_org_filter:任一非空 org 出界即隐藏。
    stmt = (
        select(Reservation.id, Reservation.title.label("label"), Reservation.status.label("secondary"))
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


def _announcements(db, u, scope, q):
    stmt = select(Announcement.id, Announcement.title.label("label"), Announcement.status.label("secondary")).where(
        Announcement.tenant_id == u.tenant_id
    )
    stmt = apply_org_filter(stmt, org_column=Announcement.audience_org_unit_id, scope=scope)
    if q:
        stmt = stmt.where(_like(Announcement.title, q))
    return stmt


def _cases(db, u, scope, q):
    stmt = (
        select(HrCase.id, HrCase.subject.label("label"), HrCase.status.label("secondary"))
        .where(HrCase.tenant_id == u.tenant_id)
    )
    # 与 /api/cases 列表同口径:行级 scope 按汇报线收敛(admin/hr 不限)
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(HrCase.requester_employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(_like(HrCase.subject, q))
    return stmt


def _requests(db, u, scope, q):
    stmt = (
        select(EmployeeRequest.id, EmployeeRequest.subject.label("label"), EmployeeRequest.status.label("secondary"))
        .where(EmployeeRequest.tenant_id == u.tenant_id)
    )
    # 与 /api/requests 列表同口径
    emp_ids = visible_employee_ids(db, u)
    if emp_ids is not None:
        stmt = stmt.where(EmployeeRequest.employee_id.in_(emp_ids or ["__none__"]))
    if q:
        stmt = stmt.where(_like(EmployeeRequest.subject, q))
    return stmt


def _knowledge(db, u, scope, q):
    stmt = (
        select(KnowledgeArticle.id, KnowledgeArticle.title.label("label"), KnowledgeArticle.category.label("secondary"))
        .where(KnowledgeArticle.tenant_id == u.tenant_id, KnowledgeArticle.status == "published")
    )
    if q:
        stmt = stmt.where(_like(KnowledgeArticle.title, q))
    return stmt


def _skills(db, u, scope, q):
    stmt = select(Skill.id, Skill.name.label("label"), Skill.category.label("secondary")).where(
        Skill.tenant_id == u.tenant_id
    )
    if q:
        stmt = stmt.where(_like(Skill.name, q))
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
