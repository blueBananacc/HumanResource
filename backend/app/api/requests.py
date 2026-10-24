"""员工申请枢纽 API(MintHCM Requests 移植)。

- ESS ``/me/requests``:提交/列表/取消/留言(行级=本人员工档案)
- HR ``/requests``:队列/审批/留言(manage_employees + employee scope)
- 审批决定写 Notification 给提交人绑定的账号。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    EmployeeRequest,
    Notification,
    RequestComment,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    can_decide_for_employee,
    require_linked_employee,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(
    prefix="/requests",
    tags=["employee-requests"],
    dependencies=[Depends(require_feature("hr.suite"))],
)
me_router = APIRouter(
    prefix="/me/requests",
    tags=["employee-requests-ess"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_DECIDABLE = ("pending",)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class RequestIn(BaseModel):
    kind: str = "general"
    subject: str = Field(min_length=1, max_length=128)
    detail: str | None = Field(default=None, max_length=5000)

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in EmployeeRequest.KINDS:
            raise ValueError(f"未知类型,可选:{','.join(EmployeeRequest.KINDS)}")
        return v


class CommentIn(BaseModel):
    body: str = Field(min_length=1, max_length=2000)


class DecideIn(BaseModel):
    action: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=500)


class CommentOut(BaseModel):
    id: str
    author_user_id: str | None
    author_email: str | None = None
    body: str
    created_at: datetime


class RequestOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str | None = None
    kind: str
    subject: str
    detail: str | None
    status: str
    decided_by: str | None
    decided_at: datetime | None
    decision_note: str | None
    comments: list[CommentOut] = []
    created_at: datetime
    updated_at: datetime


# ============ helpers ============


def _out(db: Session, r: EmployeeRequest, *, with_comments: bool = True) -> RequestOut:
    emp = db.get(Employee, r.employee_id)
    comments: list[CommentOut] = []
    if with_comments:
        rows = db.execute(
            select(RequestComment, User.email)
            .join(User, User.id == RequestComment.author_user_id, isouter=True)
            .where(RequestComment.request_id == r.id)
            .order_by(RequestComment.created_at)
        ).all()
        comments = [
            CommentOut(
                id=c.id, author_user_id=c.author_user_id, author_email=email,
                body=c.body, created_at=c.created_at,
            )
            for c, email in rows
        ]
    return RequestOut(
        id=r.id,
        employee_id=r.employee_id,
        employee_name=emp.name if emp else None,
        kind=r.kind,
        subject=r.subject,
        detail=r.detail,
        status=r.status,
        decided_by=r.decided_by,
        decided_at=r.decided_at,
        decision_note=r.decision_note,
        comments=comments,
        created_at=r.created_at,
        updated_at=r.updated_at,
    )


def _notify_requester(db: Session, r: EmployeeRequest, title: str, body: str) -> None:
    """给提交人的绑定账号写通知。"""
    emp = db.get(Employee, r.employee_id)
    if emp and emp.user_id:
        db.add(
            Notification(
                tenant_id=r.tenant_id,
                user_id=emp.user_id,
                kind="request",
                title=title,
                body=body,
                link="/me/requests",
            )
        )


# ============ ESS ============


@me_router.get("", response_model=list[RequestOut])
def me_list(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    emp_id = require_linked_employee(db, current)
    rows = db.scalars(
        select(EmployeeRequest)
        .where(
            EmployeeRequest.tenant_id == current.tenant_id,
            EmployeeRequest.employee_id == emp_id,
        )
        .order_by(EmployeeRequest.created_at.desc())
        .limit(100)
    ).all()
    return [_out(db, r) for r in rows]


@me_router.post("", response_model=RequestOut, status_code=status.HTTP_201_CREATED)
def me_create(
    body: RequestIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp_id = require_linked_employee(db, current)
    r = EmployeeRequest(
        tenant_id=current.tenant_id,
        employee_id=emp_id,
        kind=body.kind,
        subject=body.subject,
        detail=body.detail,
    )
    db.add(r)
    db.flush()
    write_audit(db, actor=current, entity_type="employee_request", entity_id=r.id, action="create", request=request)
    db.commit()
    return _out(db, r)


def _own_request(db: Session, rid: str, current: User) -> EmployeeRequest:
    emp_id = require_linked_employee(db, current)
    r = db.get(EmployeeRequest, rid)
    if r is None or r.tenant_id != current.tenant_id or r.employee_id != emp_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    return r


@me_router.post("/{rid}/cancel", response_model=RequestOut)
def me_cancel(
    rid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    r = _own_request(db, rid, current)
    if r.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, "仅待审批申请可取消")
    r.status = "cancelled"
    r.updated_at = _now()
    db.commit()
    return _out(db, r)


@me_router.post("/{rid}/comments", response_model=RequestOut, status_code=status.HTTP_201_CREATED)
def me_comment(
    rid: str,
    body: CommentIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    r = _own_request(db, rid, current)
    if r.status in ("cancelled",):
        raise HTTPException(status.HTTP_409_CONFLICT, "已取消的申请不能留言")
    db.add(
        RequestComment(
            tenant_id=current.tenant_id,
            request_id=r.id,
            author_user_id=current.id,
            body=body.body,
        )
    )
    r.updated_at = _now()
    db.commit()
    return _out(db, r)


# ============ HR 队列 ============


@router.get("", response_model=PageOut[RequestOut])
def list_requests(
    status_filter: str | None = None,
    kind: str | None = None,
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    limit, offset, q = paging
    stmt = select(EmployeeRequest).where(EmployeeRequest.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(EmployeeRequest.status == status_filter)
    if kind:
        stmt = stmt.where(EmployeeRequest.kind == kind)
    stmt = apply_q_ilike(stmt, q, EmployeeRequest.subject)
    # employee scope:hr/admin 看全部;hiring_manager 只看下属+自己
    emp_ids = visible_employee_ids(db, current)
    if emp_ids is not None:
        stmt = stmt.where(EmployeeRequest.employee_id.in_(emp_ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(EmployeeRequest.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PageOut(items=[_out(db, r, with_comments=False) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/{rid}", response_model=RequestOut)
def get_request(
    rid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    r = db.get(EmployeeRequest, rid)
    if r is None or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    emp_ids = visible_employee_ids(db, current)
    if emp_ids is not None and r.employee_id not in emp_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    return _out(db, r)


@router.post("/{rid}/decide", response_model=RequestOut)
def decide(
    rid: str,
    body: DecideIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    r = db.get(EmployeeRequest, rid)
    if r is None or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    if r.status not in _DECIDABLE:
        raise HTTPException(status.HTTP_409_CONFLICT, "该申请已处理")
    # 与 leave/expenses 审批同一口径:不能自批;hiring_manager 仅直属下属
    if not can_decide_for_employee(db, current, r.employee_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权审批该员工的申请")
    r.status = body.action
    r.decided_by = current.id
    r.decided_at = _now()
    r.decision_note = body.note
    r.updated_at = _now()
    _notify_requester(
        db, r,
        "申请已批准" if body.action == "approved" else "申请被驳回",
        f"「{r.subject}」{('已批准' if body.action == 'approved' else '被驳回')}"
        + (f":{body.note}" if body.note else ""),
    )
    write_audit(
        db, actor=current, entity_type="employee_request", entity_id=r.id,
        action=f"decide:{body.action}", detail={"note": body.note}, request=request,
    )
    db.commit()
    return _out(db, r)


@router.post("/{rid}/comments", response_model=RequestOut, status_code=status.HTTP_201_CREATED)
def hr_comment(
    rid: str,
    body: CommentIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    r = db.get(EmployeeRequest, rid)
    if r is None or r.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    emp_ids = visible_employee_ids(db, current)
    if emp_ids is not None and r.employee_id not in emp_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "申请不存在")
    db.add(
        RequestComment(
            tenant_id=current.tenant_id,
            request_id=r.id,
            author_user_id=current.id,
            body=body.body,
        )
    )
    r.updated_at = _now()
    db.commit()
    return _out(db, r)
