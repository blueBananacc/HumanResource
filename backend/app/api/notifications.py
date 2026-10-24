"""站内通知 + 触发式任务(watchers)+ 报表订阅 — 全部按登录用户行级过滤。

- ``GET /me/notifications`` 列表(unread 优先)
- ``GET /me/notifications/unread-count``
- ``POST /me/notifications/{id}/read`` / ``POST /me/notifications/read-all``
- ``GET/POST /me/watchers`` + toggle/delete —— 触发式任务属本人
- ``GET/POST /me/report-subscriptions`` + toggle/delete —— 订阅属本人
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import (
    AiWatcher,
    Notification,
    ReportSubscription,
    User,
)
from app.infra.db import get_db
from app.services.permissions import PERM_VIEW_REPORTS, has_permission
from app.services.watchers import WATCHER_KINDS, WATCHER_PERMS

router = APIRouter(prefix="/me", tags=["notifications"])

REPORT_KINDS = ("pipeline_funnel", "weekly_summary", "pending_items")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ notifications ============


class NotificationOut(BaseModel):
    id: str
    kind: str
    title: str
    body: str | None
    link: str | None
    read_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/notifications", response_model=list[NotificationOut])
def list_notifications(
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    stmt = (
        select(Notification)
        .where(
            Notification.tenant_id == current.tenant_id,
            Notification.user_id == current.id,
        )
        .order_by(Notification.read_at.isnot(None), Notification.created_at.desc())
        .limit(50)
    )
    if unread_only:
        stmt = stmt.where(Notification.read_at.is_(None))
    return db.scalars(stmt).all()


@router.get("/notifications/unread-count")
def unread_count(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    n = db.execute(
        select(func.count(Notification.id)).where(
            Notification.tenant_id == current.tenant_id,
            Notification.user_id == current.id,
            Notification.read_at.is_(None),
        )
    ).scalar_one()
    return {"count": n}


def _own_notification(db: Session, nid: str, current: User) -> Notification:
    n = db.get(Notification, nid)
    if n is None or n.tenant_id != current.tenant_id or n.user_id != current.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通知不存在")
    return n


@router.post("/notifications/{nid}/read", response_model=NotificationOut)
def mark_read(
    nid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    n = _own_notification(db, nid, current)
    if n.read_at is None:
        n.read_at = _now()
        db.commit()
    return n


@router.post("/notifications/read-all")
def mark_all_read(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    n = db.execute(
        select(func.count(Notification.id)).where(
            Notification.tenant_id == current.tenant_id,
            Notification.user_id == current.id,
            Notification.read_at.is_(None),
        )
    ).scalar_one()
    db.query(Notification).filter(
        Notification.tenant_id == current.tenant_id,
        Notification.user_id == current.id,
        Notification.read_at.is_(None),
    ).update({"read_at": _now()})
    db.commit()
    return {"marked": n}


# ============ watchers ============


class WatcherOut(BaseModel):
    id: str
    kind: str
    params: dict
    is_active: bool
    last_checked_at: datetime | None
    last_fired_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class WatcherIn(BaseModel):
    kind: str
    params: dict = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in WATCHER_KINDS:
            raise ValueError(f"未知 watcher 类型,可选:{','.join(WATCHER_KINDS)}")
        return v


@router.get("/watchers", response_model=list[WatcherOut])
def list_watchers(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    return db.scalars(
        select(AiWatcher).where(
            AiWatcher.tenant_id == current.tenant_id,
            AiWatcher.user_id == current.id,
        )
    ).all()


@router.post("/watchers", response_model=WatcherOut, status_code=status.HTTP_201_CREATED)
def create_watcher(
    body: WatcherIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    required = WATCHER_PERMS.get(body.kind)
    if required is not None and not has_permission(current, required):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "该提醒类型权限不足")
    # 同类 watcher 每人一个就够 — 重复创建直接返回已有的
    existing = db.scalars(
        select(AiWatcher).where(
            AiWatcher.tenant_id == current.tenant_id,
            AiWatcher.user_id == current.id,
            AiWatcher.kind == body.kind,
        )
    ).first()
    if existing:
        existing.params = body.params
        existing.is_active = True
        db.commit()
        return existing
    w = AiWatcher(
        tenant_id=current.tenant_id,
        user_id=current.id,
        kind=body.kind,
        params=body.params,
    )
    db.add(w)
    db.commit()
    return w


def _own_watcher(db: Session, wid: str, current: User) -> AiWatcher:
    w = db.get(AiWatcher, wid)
    if w is None or w.tenant_id != current.tenant_id or w.user_id != current.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "watcher 不存在")
    return w


@router.post("/watchers/{wid}/toggle", response_model=WatcherOut)
def toggle_watcher(
    wid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    w = _own_watcher(db, wid, current)
    w.is_active = not w.is_active
    db.commit()
    return w


@router.delete("/watchers/{wid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_watcher(
    wid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    w = _own_watcher(db, wid, current)
    db.delete(w)
    db.commit()


# ============ report subscriptions ============


class SubOut(BaseModel):
    id: str
    kind: str
    weekday: int
    hour: int
    is_active: bool
    last_sent_at: datetime | None

    model_config = {"from_attributes": True}


class SubIn(BaseModel):
    kind: str
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in REPORT_KINDS:
            raise ValueError(f"未知报表类型,可选:{','.join(REPORT_KINDS)}")
        return v


@router.get("/report-subscriptions", response_model=list[SubOut])
def list_subs(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    return db.scalars(
        select(ReportSubscription).where(
            ReportSubscription.tenant_id == current.tenant_id,
            ReportSubscription.user_id == current.id,
        )
    ).all()


@router.post("/report-subscriptions", response_model=SubOut, status_code=status.HTTP_201_CREATED)
def upsert_sub(
    body: SubIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    # 三种报表内容都是租户级聚合数据 → 需要 view_reports(employee 角色无此位)
    if not has_permission(current, PERM_VIEW_REPORTS):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "报表订阅需要 view_reports 权限")
    s = db.scalars(
        select(ReportSubscription).where(
            ReportSubscription.tenant_id == current.tenant_id,
            ReportSubscription.user_id == current.id,
            ReportSubscription.kind == body.kind,
        )
    ).first()
    if s:
        s.weekday = body.weekday
        s.hour = body.hour
        s.is_active = True
        db.commit()
        return s
    s = ReportSubscription(
        tenant_id=current.tenant_id,
        user_id=current.id,
        kind=body.kind,
        weekday=body.weekday,
        hour=body.hour,
    )
    db.add(s)
    try:
        db.commit()
    except IntegrityError:
        # 并发同 kind 订阅 → 唯一约束兜底,按 upsert 语义返回已有行
        db.rollback()
        s = db.scalars(
            select(ReportSubscription).where(
                ReportSubscription.tenant_id == current.tenant_id,
                ReportSubscription.user_id == current.id,
                ReportSubscription.kind == body.kind,
            )
        ).one()
    return s


@router.post("/report-subscriptions/{sid}/toggle", response_model=SubOut)
def toggle_sub(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    s = db.get(ReportSubscription, sid)
    if s is None or s.tenant_id != current.tenant_id or s.user_id != current.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "订阅不存在")
    s.is_active = not s.is_active
    db.commit()
    return s


@router.delete("/report-subscriptions/{sid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sub(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    s = db.get(ReportSubscription, sid)
    if s is None or s.tenant_id != current.tenant_id or s.user_id != current.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "订阅不存在")
    db.delete(s)
    db.commit()
