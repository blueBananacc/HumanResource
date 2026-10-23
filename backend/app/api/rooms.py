"""会议室 / 资源 / 预约 API(参考 MintHCM Rooms/Resources/Reservations)。

- ``GET/POST/PUT/DELETE /rooms``        会议室(manage_settings)
- ``GET/POST/PUT/DELETE /resources``    资源(manage_settings)
- ``GET    /reservations``              日历条(按周/区间)
- ``POST   /reservations``              预约:同对象时间重叠 → 409
- ``PUT    /reservations/{id}``         改期/标题
- ``POST   /reservations/{id}/cancel``  取消(软删)
- ``GET    /interviewers/availability`` 面试官在区间内与自身预约的冲突数
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import (
    Interview,
    Reservation,
    Resource,
    Room,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_SETTINGS,
    PERM_VIEW_REPORTS,
    PERM_WRITE_INTERVIEW,
    get_org_scope,
    require_permission,
)

router = APIRouter(prefix="/reservations", tags=["reservations"])
rooms_router = APIRouter(prefix="/rooms", tags=["reservations"])
resources_router = APIRouter(prefix="/resources", tags=["reservations"])
interviewers_router = APIRouter(prefix="/interviewers", tags=["reservations"])

RESOURCE_TYPES = Resource.TYPES


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


# ============ schemas ============


class RoomIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    location: str | None = Field(None, max_length=128)
    capacity: int = Field(4, ge=1, le=500)
    org_unit_id: str | None = None
    is_active: bool = True


class RoomOut(BaseModel):
    id: str
    name: str
    location: str | None
    capacity: int
    org_unit_id: str | None
    is_active: bool


class ResourceIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: str = "other"
    is_active: bool = True


class ResourceOut(BaseModel):
    id: str
    name: str
    type: str
    is_active: bool


class ReservationIn(BaseModel):
    kind: str = "room"
    room_id: str | None = None
    resource_id: str | None = None
    interview_id: str | None = None
    title: str = Field(..., min_length=1, max_length=128)
    starts_at: datetime
    ends_at: datetime
    interviewer_ids: list[str] = Field(default_factory=list, max_length=20)


class ReservationOut(BaseModel):
    id: str
    kind: str
    room_id: str | None
    room_name: str | None
    resource_id: str | None
    resource_name: str | None
    interview_id: str | None
    title: str
    starts_at: datetime
    ends_at: datetime
    interviewer_ids: list
    interviewer_names: list[str]
    status: str
    created_by: str | None


# ============ helpers ============


def _room(db: Session, tenant_id: str, rid: str) -> Room:
    r = db.get(Room, rid)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(404, "会议室不存在")
    return r


def _resource(db: Session, tenant_id: str, rid: str) -> Resource:
    r = db.get(Resource, rid)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(404, "资源不存在")
    return r


def _reservation(db: Session, tenant_id: str, rid: str) -> Reservation:
    r = db.get(Reservation, rid)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(404, "预约不存在")
    return r


def _overlapping(
    db: Session, tenant_id: str, *, room_id=None, resource_id=None,
    starts_at: datetime, ends_at: datetime, exclude_id=None,
) -> Reservation | None:
    """同对象 + status=active 的时间重叠检查。"""
    stmt = select(Reservation).where(
        Reservation.tenant_id == tenant_id,
        Reservation.status == "active",
        Reservation.starts_at < ends_at,
        Reservation.ends_at > starts_at,
    )
    if room_id:
        stmt = stmt.where(Reservation.room_id == room_id)
    if resource_id:
        stmt = stmt.where(Reservation.resource_id == resource_id)
    if exclude_id:
        stmt = stmt.where(Reservation.id != exclude_id)
    return db.scalars(stmt).first()


def _to_out(
    db: Session, r: Reservation, rooms: dict, resources: dict, users: dict
) -> ReservationOut:
    return ReservationOut(
        id=r.id, kind=r.kind,
        room_id=r.room_id,
        room_name=rooms.get(r.room_id),
        resource_id=r.resource_id,
        resource_name=resources.get(r.resource_id),
        interview_id=r.interview_id,
        title=r.title, starts_at=r.starts_at, ends_at=r.ends_at,
        interviewer_ids=r.interviewer_ids or [],
        interviewer_names=[users.get(uid, uid[:8]) for uid in (r.interviewer_ids or [])],
        status=r.status, created_by=r.created_by,
    )


# ============ 会议室 CRUD ============


@rooms_router.get("", response_model=list[RoomOut])
def list_rooms(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    scope = get_org_scope(db, current)
    stmt = select(Room).where(Room.tenant_id == current.tenant_id)
    if scope and not scope.is_all and scope.org_ids:
        stmt = stmt.where(
            (Room.org_unit_id.is_(None)) | (Room.org_unit_id.in_(scope.org_ids))
        )
    return [
        RoomOut(
            id=r.id, name=r.name, location=r.location, capacity=r.capacity,
            org_unit_id=r.org_unit_id, is_active=r.is_active,
        )
        for r in db.scalars(stmt.order_by(Room.name))
    ]


@rooms_router.post("", response_model=RoomOut, status_code=201)
def create_room(
    body: RoomIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = Room(tenant_id=current.tenant_id, **body.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)
    write_audit(db, actor=current, entity_type="room", entity_id=r.id,
                action="create", detail={"name": r.name}, request=request)
    db.commit()
    return RoomOut(
        id=r.id, name=r.name, location=r.location, capacity=r.capacity,
        org_unit_id=r.org_unit_id, is_active=r.is_active,
    )


@rooms_router.put("/{room_id}", response_model=RoomOut)
def update_room(
    room_id: str,
    body: RoomIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _room(db, current.tenant_id, room_id)
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    write_audit(db, actor=current, entity_type="room", entity_id=r.id,
                action="update", detail={"name": r.name}, request=request)
    db.commit()
    return RoomOut(
        id=r.id, name=r.name, location=r.location, capacity=r.capacity,
        org_unit_id=r.org_unit_id, is_active=r.is_active,
    )


@rooms_router.delete("/{room_id}", status_code=204)
def delete_room(
    room_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _room(db, current.tenant_id, room_id)
    db.delete(r)
    write_audit(db, actor=current, entity_type="room", entity_id=room_id,
                action="delete", detail={"name": r.name}, request=request)
    db.commit()


# ============ 资源 CRUD ============


@resources_router.get("", response_model=list[ResourceOut])
def list_resources(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return [
        ResourceOut(id=r.id, name=r.name, type=r.type, is_active=r.is_active)
        for r in db.scalars(
            select(Resource)
            .where(Resource.tenant_id == current.tenant_id)
            .order_by(Resource.name)
        )
    ]


@resources_router.post("", response_model=ResourceOut, status_code=201)
def create_resource(
    body: ResourceIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    if body.type not in RESOURCE_TYPES:
        raise HTTPException(400, f"type 必须是 {RESOURCE_TYPES} 之一")
    r = Resource(tenant_id=current.tenant_id, **body.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)
    write_audit(db, actor=current, entity_type="resource", entity_id=r.id,
                action="create", detail={"name": r.name}, request=request)
    db.commit()
    return ResourceOut(id=r.id, name=r.name, type=r.type, is_active=r.is_active)


@resources_router.put("/{resource_id}", response_model=ResourceOut)
def update_resource(
    resource_id: str,
    body: ResourceIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _resource(db, current.tenant_id, resource_id)
    if body.type not in RESOURCE_TYPES:
        raise HTTPException(400, f"type 必须是 {RESOURCE_TYPES} 之一")
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    write_audit(db, actor=current, entity_type="resource", entity_id=r.id,
                action="update", detail={"name": r.name}, request=request)
    db.commit()
    return ResourceOut(id=r.id, name=r.name, type=r.type, is_active=r.is_active)


@resources_router.delete("/{resource_id}", status_code=204)
def delete_resource(
    resource_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_MANAGE_SETTINGS)),
):
    r = _resource(db, current.tenant_id, resource_id)
    db.delete(r)
    write_audit(db, actor=current, entity_type="resource", entity_id=resource_id,
                action="delete", detail={"name": r.name}, request=request)
    db.commit()


# ============ 预约 ============


def _reservation_names(db: Session, tenant_id: str, rows: list[Reservation]):
    room_ids = {r.room_id for r in rows if r.room_id}
    res_ids = {r.resource_id for r in rows if r.resource_id}
    user_ids = {uid for r in rows for uid in (r.interviewer_ids or [])}
    rooms = {
        x.id: x.name
        for x in db.scalars(select(Room).where(Room.id.in_(room_ids)))
    } if room_ids else {}
    resources = {
        x.id: x.name
        for x in db.scalars(select(Resource).where(Resource.id.in_(res_ids)))
    } if res_ids else {}
    users = {
        x.id: x.email
        for x in db.scalars(
            select(User).where(User.tenant_id == tenant_id, User.id.in_(user_ids))
        )
    } if user_ids else {}
    return rooms, resources, users


@router.get("", response_model=list[ReservationOut])
def list_reservations(
    start: datetime = Query(...),
    end: datetime = Query(...),
    room_id: str | None = Query(None),
    kind: str | None = Query(None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    """按时间区间列出日历条(含已取消,前端可灰色显示)。"""
    stmt = select(Reservation).where(
        Reservation.tenant_id == current.tenant_id,
        Reservation.starts_at < _naive(end),
        Reservation.ends_at > _naive(start),
    )
    if room_id:
        stmt = stmt.where(Reservation.room_id == room_id)
    if kind:
        stmt = stmt.where(Reservation.kind == kind)
    scope = get_org_scope(db, current)
    if scope and not scope.is_all and scope.org_ids:
        stmt = stmt.outerjoin(Room, Room.id == Reservation.room_id).where(
            (Reservation.room_id.is_(None))
            | (Room.org_unit_id.is_(None))
            | (Room.org_unit_id.in_(scope.org_ids))
        )
    rows = db.scalars(stmt.order_by(Reservation.starts_at)).all()
    rooms, resources, users = _reservation_names(db, current.tenant_id, rows)
    return [_to_out(db, r, rooms, resources, users) for r in rows]


@router.post("", response_model=ReservationOut, status_code=201)
def create_reservation(
    body: ReservationIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    if body.kind not in Reservation.KINDS:
        raise HTTPException(400, f"kind 必须是 {Reservation.KINDS} 之一")
    starts_at = _naive(body.starts_at)
    ends_at = _naive(body.ends_at)
    if ends_at <= starts_at:
        raise HTTPException(400, "结束时间必须晚于开始时间")
    if body.room_id:
        room = _room(db, current.tenant_id, body.room_id)
        if not room.is_active:
            raise HTTPException(400, "会议室已停用")
    if body.resource_id:
        res = _resource(db, current.tenant_id, body.resource_id)
        if not res.is_active:
            raise HTTPException(400, "资源已停用")
    if not body.room_id and not body.resource_id and body.kind != "interview":
        raise HTTPException(400, "预约必须占一个会议室或资源")
    if body.interview_id:
        iv = db.get(Interview, body.interview_id)
        if not iv or iv.tenant_id != current.tenant_id:
            raise HTTPException(404, "面试不存在")

    clash = _overlapping(
        db, current.tenant_id, room_id=body.room_id, resource_id=body.resource_id,
        starts_at=starts_at, ends_at=ends_at,
    )
    if clash:
        raise HTTPException(
            409,
            f"时间段冲突:与「{clash.title}」"
            f"({clash.starts_at:%H:%M}-{clash.ends_at:%H:%M})重叠",
        )

    r = Reservation(
        tenant_id=current.tenant_id, kind=body.kind,
        room_id=body.room_id, resource_id=body.resource_id,
        interview_id=body.interview_id, title=body.title.strip(),
        starts_at=starts_at, ends_at=ends_at,
        interviewer_ids=body.interviewer_ids,
        created_by=current.id,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    write_audit(db, actor=current, entity_type="reservation", entity_id=r.id,
                action="create",
                detail={"title": r.title, "kind": r.kind,
                        "starts_at": starts_at.isoformat()},
                request=request)
    db.commit()
    rooms, resources, users = _reservation_names(db, current.tenant_id, [r])
    return _to_out(db, r, rooms, resources, users)


@router.put("/{rid}", response_model=ReservationOut)
def update_reservation(
    rid: str,
    body: ReservationIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    r = _reservation(db, current.tenant_id, rid)
    if r.status == "cancelled":
        raise HTTPException(400, "已取消的预约不能修改")
    starts_at = _naive(body.starts_at)
    ends_at = _naive(body.ends_at)
    if ends_at <= starts_at:
        raise HTTPException(400, "结束时间必须晚于开始时间")
    if body.room_id:
        _room(db, current.tenant_id, body.room_id)
    if body.resource_id:
        _resource(db, current.tenant_id, body.resource_id)
    clash = _overlapping(
        db, current.tenant_id, room_id=body.room_id, resource_id=body.resource_id,
        starts_at=starts_at, ends_at=ends_at, exclude_id=r.id,
    )
    if clash:
        raise HTTPException(
            409,
            f"时间段冲突:与「{clash.title}」"
            f"({clash.starts_at:%H:%M}-{clash.ends_at:%H:%M})重叠",
        )
    r.kind = body.kind
    r.room_id = body.room_id
    r.resource_id = body.resource_id
    r.interview_id = body.interview_id
    r.title = body.title.strip()
    r.starts_at = starts_at
    r.ends_at = ends_at
    r.interviewer_ids = body.interviewer_ids
    r.updated_at = _now()
    write_audit(db, actor=current, entity_type="reservation", entity_id=r.id,
                action="update", detail={"title": r.title}, request=request)
    db.commit()
    rooms, resources, users = _reservation_names(db, current.tenant_id, [r])
    return _to_out(db, r, rooms, resources, users)


@router.post("/{rid}/cancel", response_model=ReservationOut)
def cancel_reservation(
    rid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    r = _reservation(db, current.tenant_id, rid)
    if r.status == "cancelled":
        return _to_out(db, r, *_reservation_names(db, current.tenant_id, [r]))
    r.status = "cancelled"
    r.updated_at = _now()
    write_audit(db, actor=current, entity_type="reservation", entity_id=r.id,
                action="cancel", detail={"title": r.title}, request=request)
    db.commit()
    rooms, resources, users = _reservation_names(db, current.tenant_id, [r])
    return _to_out(db, r, rooms, resources, users)


# ============ 面试官可用性 ============


@interviewers_router.get("/availability")
def interviewer_availability(
    start: datetime = Query(...),
    end: datetime = Query(...),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    """每位同租户用户在区间内的预约冲突数(含其作为面试官的条数)。"""
    members = db.scalars(
        select(User).where(
            User.tenant_id == current.tenant_id,
            User.status == "active",
        )
    ).all()
    rows = db.scalars(
        select(Reservation).where(
            Reservation.tenant_id == current.tenant_id,
            Reservation.status == "active",
            Reservation.starts_at < _naive(end),
            Reservation.ends_at > _naive(start),
        )
    ).all()
    out = []
    for u in members:
        busy = [r for r in rows if u.id in (r.interviewer_ids or [])]
        out.append({
            "user_id": u.id,
            "name": u.email,
            "role": u.role,
            "busy_count": len(busy),
            "busy_titles": [r.title for r in busy[:3]],
        })
    return out
