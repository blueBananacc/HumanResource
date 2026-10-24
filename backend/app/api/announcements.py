"""公司公告 API。

- ``GET/POST /announcements`` 列表/创建(写 manage_employees,读 view_reports)
- ``GET/PUT/DELETE /announcements/{id}`` 详情/更新/删除
- ``POST /announcements/{id}/publish|archive`` 状态流转
- ``GET /announcements/{id}/reads`` 已读名单
- ESS ``GET /me/announcements`` + ``POST /me/announcements/{id}/read``
  —— 已发布未过期且受众匹配(null=全公司,或员工所在 org 及其祖先 org)。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Announcement,
    AnnouncementRead,
    Employee,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    ensure_can_see,
    get_linked_employee_id,
    get_org_scope,
    require_permission,
)
from app.services.timeutil import local_now

router = APIRouter(
    prefix="/announcements",
    tags=["announcements"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

me_router = APIRouter(
    prefix="/me/announcements",
    tags=["announcements-ess"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_STATUSES = {"draft", "published", "archived"}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class AnnouncementIn(BaseModel):
    title: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=20000)
    audience_org_unit_id: str | None = None
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _wall_clock(cls, v: datetime | None) -> datetime | None:
        # 与 rooms._naive 同口径:tz-aware 输入丢弃偏移取表达墙钟,
        # 否则带时区后缀的请求会把过期时间存偏(me_* 侧按 local_now 比较)
        return v.replace(tzinfo=None) if v is not None else v


class AnnouncementOut(BaseModel):
    id: str
    title: str
    content: str
    audience_org_unit_id: str | None
    audience_name: str | None = None
    status: str
    published_at: datetime | None
    expires_at: datetime | None
    created_by: str | None
    read_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class MeAnnouncementOut(BaseModel):
    id: str
    title: str
    content: str
    audience_name: str | None = None
    published_at: datetime | None
    expires_at: datetime | None
    read: bool
    created_at: datetime


# ============ helpers ============


def _get(db: Session, aid: str, current: User, scope) -> Announcement:
    a = db.get(Announcement, aid)
    if a is None or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "公告不存在")
    ensure_can_see(scope, a.audience_org_unit_id)
    return a


def _audience_name(db: Session, a: Announcement) -> str | None:
    if not a.audience_org_unit_id:
        return None
    org = db.get(OrgUnit, a.audience_org_unit_id)
    return org.name if org else None


def _out(db: Session, a: Announcement) -> AnnouncementOut:
    reads = count_total(db, select(AnnouncementRead).where(AnnouncementRead.announcement_id == a.id))
    return AnnouncementOut(
        id=a.id,
        title=a.title,
        content=a.content,
        audience_org_unit_id=a.audience_org_unit_id,
        audience_name=_audience_name(db, a),
        status=a.status,
        published_at=a.published_at,
        expires_at=a.expires_at,
        created_by=a.created_by,
        read_count=reads,
        created_at=a.created_at,
    )


# ============ 管理端 ============


@router.get("", response_model=PageOut[AnnouncementOut])
def list_announcements(
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, q = paging
    scope = get_org_scope(db, current)
    stmt = select(Announcement).where(Announcement.tenant_id == current.tenant_id)
    stmt = apply_q_ilike(stmt, q, Announcement.title)
    # audience 为空的公告全员可见 → 对 scoped 用户恒可见;
    # audience 非空的公告按 org scope 过滤
    if scope is not None:
        if scope:
            stmt = stmt.where(
                (Announcement.audience_org_unit_id.in_(scope))
                | (Announcement.audience_org_unit_id.is_(None))
            )
        else:
            stmt = stmt.where(Announcement.audience_org_unit_id.is_(None))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Announcement.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(items=[_out(db, a) for a in rows], total=total, limit=limit, offset=offset)


@router.post("", response_model=AnnouncementOut, status_code=status.HTTP_201_CREATED)
def create_announcement(
    body: AnnouncementIn,
    request: Request,
    publish: bool = False,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    scope = get_org_scope(db, current)
    if body.audience_org_unit_id:
        org = db.get(OrgUnit, body.audience_org_unit_id)
        if org is None or org.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "受众组织不存在")
        ensure_can_see(scope, body.audience_org_unit_id)
    a = Announcement(
        tenant_id=current.tenant_id,
        title=body.title,
        content=body.content,
        audience_org_unit_id=body.audience_org_unit_id,
        expires_at=body.expires_at,
        created_by=current.id,
    )
    if publish:
        a.status = "published"
        a.published_at = _now()
    db.add(a)
    db.flush()
    write_audit(
        db, actor=current, entity_type="announcement", entity_id=a.id,
        action="create", detail={"publish": publish}, request=request,
    )
    db.commit()
    return _out(db, a)


@router.get("/{aid}", response_model=AnnouncementOut)
def get_announcement(
    aid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return _out(db, _get(db, aid, current, get_org_scope(db, current)))


@router.put("/{aid}", response_model=AnnouncementOut)
def update_announcement(
    aid: str,
    body: AnnouncementIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    scope = get_org_scope(db, current)
    a = _get(db, aid, current, scope)
    if a.status == "archived":
        raise HTTPException(status.HTTP_409_CONFLICT, "已归档公告不可编辑")
    if body.audience_org_unit_id:
        org = db.get(OrgUnit, body.audience_org_unit_id)
        if org is None or org.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "受众组织不存在")
        ensure_can_see(scope, body.audience_org_unit_id)
    a.title = body.title
    a.content = body.content
    a.audience_org_unit_id = body.audience_org_unit_id
    a.expires_at = body.expires_at
    a.updated_at = _now()
    write_audit(db, actor=current, entity_type="announcement", entity_id=a.id, action="update", request=request)
    db.commit()
    return _out(db, a)


@router.post("/{aid}/status", response_model=AnnouncementOut)
def set_status(
    aid: str,
    request: Request,
    to: Literal["published", "archived", "draft"],
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    a = _get(db, aid, current, get_org_scope(db, current))
    a.status = to
    if to == "published" and a.published_at is None:
        a.published_at = _now()
    a.updated_at = _now()
    write_audit(db, actor=current, entity_type="announcement", entity_id=a.id, action=f"status:{to}", request=request)
    db.commit()
    return _out(db, a)


@router.delete("/{aid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_announcement(
    aid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    a = _get(db, aid, current, get_org_scope(db, current))
    write_audit(db, actor=current, entity_type="announcement", entity_id=a.id, action="delete", request=request)
    db.delete(a)
    db.commit()


@router.get("/{aid}/reads")
def list_reads(
    aid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    _get(db, aid, current, get_org_scope(db, current))
    rows = db.execute(
        select(AnnouncementRead, User.email)
        .join(User, User.id == AnnouncementRead.user_id)
        .where(AnnouncementRead.announcement_id == aid)
        .order_by(AnnouncementRead.read_at.desc())
    ).all()
    return [
        {"user_id": r.user_id, "email": email, "read_at": r.read_at}
        for r, email in rows
    ]


# ============ ESS ============


def _employee_org_chain(db: Session, emp_org_id: str | None) -> set[str]:
    """员工所在 org + 全部祖先 org id。

    公告受众 = 某 org 时按"该 org 及其全部下属"投递(与管理端 apply_org_filter
    的 descendant 语义一致):员工在受众 org 的子树任意位置都算命中。
    """
    chain: set[str] = set()
    cur = emp_org_id
    while cur and cur not in chain:
        chain.add(cur)
        org = db.get(OrgUnit, cur)
        cur = org.parent_id if org else None
    return chain


def _visible_for_employee(a: Announcement, emp_org_chain: set[str]) -> bool:
    if a.audience_org_unit_id is None:
        return True
    return a.audience_org_unit_id in emp_org_chain


@me_router.get("", response_model=list[MeAnnouncementOut])
def me_announcements(
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    emp_id = get_linked_employee_id(db, current)
    emp = db.get(Employee, emp_id) if emp_id else None
    org_chain = _employee_org_chain(db, emp.org_unit_id if emp else None)
    # expires_at 是管理员按本地墙钟录入的(datetime-local 原样落库),与"现在"
    # 比较必须用同一口径 —— 部署时区当前时间,而不是 UTC。
    now = local_now()
    rows = db.scalars(
        select(Announcement)
        .where(
            Announcement.tenant_id == current.tenant_id,
            Announcement.status == "published",
            (Announcement.expires_at.is_(None)) | (Announcement.expires_at > now),
        )
        .order_by(Announcement.published_at.desc())
    ).all()
    read_ids = set(
        db.scalars(
            select(AnnouncementRead.announcement_id).where(
                AnnouncementRead.user_id == current.id
            )
        ).all()
    )
    return [
        MeAnnouncementOut(
            id=a.id,
            title=a.title,
            content=a.content,
            audience_name=_audience_name(db, a),
            published_at=a.published_at,
            expires_at=a.expires_at,
            read=a.id in read_ids,
            created_at=a.created_at,
        )
        for a in rows
        if _visible_for_employee(a, org_chain)
    ]


@me_router.post("/{aid}/read", status_code=status.HTTP_204_NO_CONTENT)
def me_mark_read(
    aid: str,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    a = db.get(Announcement, aid)
    emp_id = get_linked_employee_id(db, current)
    emp = db.get(Employee, emp_id) if emp_id else None
    org_chain = _employee_org_chain(db, emp.org_unit_id if emp else None)
    if (
        a is None
        or a.tenant_id != current.tenant_id
        or a.status != "published"
        or (a.expires_at is not None and a.expires_at <= local_now())
        or not _visible_for_employee(a, org_chain)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "公告不存在")
    existing = db.scalars(
        select(AnnouncementRead).where(
            AnnouncementRead.announcement_id == aid,
            AnnouncementRead.user_id == current.id,
        )
    ).first()
    if existing is None:
        db.add(
            AnnouncementRead(
                tenant_id=current.tenant_id,
                announcement_id=aid,
                user_id=current.id,
                read_at=_now(),
            )
        )
        try:
            db.commit()
        except IntegrityError:
            # 并发重复已读 → 唯一约束兜底,打点本就幂等,忽略即可
            db.rollback()
