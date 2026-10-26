"""费用报销 API(HR 管理端)。

- ``GET/POST/DELETE /expenses/types``             报销类型
- ``GET /expenses/claims``                         报销单列表(scope=team)
- ``GET  /expenses/claims/{id}``                   详情(含 items)
- ``POST /expenses/claims/{id}/approve|reject|pay`` 审批 + 标记已付款
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    ExpenseClaim,
    ExpenseItem,
    ExpenseType,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_APPROVE_EXPENSES,
    PERM_MANAGE_EXPENSES,
    PERM_VIEW_REPORTS,
    get_linked_employee_id,
    require_permission,
    team_ids_with_self,
    visible_employee_ids,
)

router = APIRouter(prefix="/expenses", tags=["hris-expenses"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TypeIn(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _v(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("名称不能为空")
        return v


class TypeOut(BaseModel):
    id: str
    name: str


class ItemOut(BaseModel):
    id: str
    type_id: str | None
    type_name: str | None
    occurred_on: date
    amount: float
    note: str | None


class ClaimOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    title: str
    total: float
    status: str
    items: list[ItemOut]
    approver_id: str | None
    decided_at: datetime | None
    decision_note: str | None
    own_request: bool
    created_at: datetime


def _maps(db: Session, tenant_id: str):
    emps = {e.id: e for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)
    )}
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
    )}
    types = {t.id: t for t in db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == tenant_id)
    )}
    return emps, orgs, types


def _claim_out(
    c: ExpenseClaim, items: list[ExpenseItem], emps, orgs, types,
    me_id: str | None = None,
) -> ClaimOut:
    e = emps.get(c.employee_id)
    return ClaimOut(
        id=c.id, employee_id=c.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        title=c.title, total=c.total, status=c.status,
        items=[ItemOut(
            id=i.id, type_id=i.type_id,
            type_name=types[i.type_id].name if i.type_id in types else None,
            occurred_on=i.occurred_on, amount=i.amount, note=i.note,
        ) for i in sorted(items, key=lambda x: x.occurred_on)],
        approver_id=c.approver_id, decided_at=c.decided_at,
        decision_note=c.decision_note,
        own_request=bool(e and e.user_id == me_id),
        created_at=c.created_at,
    )


# ============ types ============


@router.get("/types", response_model=list[TypeOut])
def list_types(
    db: Session = Depends(get_db), current: User = Depends(require_permission(PERM_VIEW_REPORTS))
):
    rows = db.scalars(
        select(ExpenseType).where(ExpenseType.tenant_id == current.tenant_id)
    ).all()
    return [TypeOut(id=t.id, name=t.name) for t in rows]


@router.post("/types", response_model=TypeOut, status_code=status.HTTP_201_CREATED)
def create_type(
    body: TypeIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EXPENSES)),
):
    t = ExpenseType(tenant_id=current.tenant_id, name=body.name)
    db.add(t)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "类型已存在")
    return TypeOut(id=t.id, name=t.name)


@router.delete("/types/{type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_type(
    type_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EXPENSES)),
):
    t = db.get(ExpenseType, type_id)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "类型不存在")
    used = db.scalars(
        select(ExpenseItem.id).where(ExpenseItem.type_id == t.id).limit(1)
    ).first()
    if used:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该类型下已有报销明细,不能删除")
    db.delete(t)
    db.commit()


# ============ claims ============


@router.get("/claims", response_model=PageOut[ClaimOut])
def list_claims(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    scope: Literal["all", "team"] = "all",
    employee_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(ExpenseClaim).where(ExpenseClaim.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(ExpenseClaim.status == status_filter)
    if employee_id:
        stmt = stmt.where(ExpenseClaim.employee_id == employee_id)
    if scope == "team":
        stmt = stmt.where(
            ExpenseClaim.employee_id.in_(
                team_ids_with_self(db, current) or ["__none__"]
            )
        )
    elif current.role not in ("admin", "hr"):
        ids = visible_employee_ids(db, current)
        stmt = stmt.where(ExpenseClaim.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(ExpenseClaim.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps, orgs, types = _maps(db, current.tenant_id)
    items = db.scalars(
        select(ExpenseItem).where(
            ExpenseItem.claim_id.in_([c.id for c in rows] or ["__none__"])
        )
    ).all()
    by_claim: dict[str, list[ExpenseItem]] = {}
    for i in items:
        by_claim.setdefault(i.claim_id, []).append(i)
    return PageOut(
        items=[_claim_out(c, by_claim.get(c.id, []), emps, orgs, types, current.id)
               for c in rows],
        total=total, limit=limit, offset=offset,
    )


@router.get("/claims/{claim_id}", response_model=ClaimOut)
def get_claim(
    claim_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    c = db.get(ExpenseClaim, claim_id)
    if not c or c.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "报销单不存在")
    ids = visible_employee_ids(db, current)
    if ids is not None and c.employee_id not in ids:
        # 与列表口径一致:非 admin/hr 仅见直属下属+自己,404 不暴露存在性
        raise HTTPException(status.HTTP_404_NOT_FOUND, "报销单不存在")
    emps, orgs, types = _maps(db, current.tenant_id)
    items = list(db.scalars(
        select(ExpenseItem).where(ExpenseItem.claim_id == c.id)
    ))
    return _claim_out(c, items, emps, orgs, types, current.id)


def _can_decide(db: Session, user: User, c: ExpenseClaim) -> bool:
    emp = db.get(Employee, c.employee_id)
    if not emp:
        return False
    # 职责分离:不能批自己提交的报销单
    if emp.user_id == user.id:
        return False
    if user.role in ("admin", "hr"):
        return True
    if user.role == "hiring_manager":
        mgr = get_linked_employee_id(db, user)
        return bool(mgr and emp.reports_to_id == mgr)
    return False


def _apply_claim_decision(
    db: Session, c: ExpenseClaim, action: str, actor: User,
    note: str | None, request: Request | None = None,
) -> None:
    """批准/驳回副作用(报销只改状态,付款走单独 /pay)。审批中心引擎
    与本模块散装端点共用 —— 不 commit。"""
    c.status = action
    c.approver_id = actor.id
    c.decided_at = _now()
    c.decision_note = note
    write_audit(
        db, actor=actor, entity_type="expense_claim", entity_id=c.id,
        action=action, detail={"total": c.total}, request=request,
    )


def _decide(
    db: Session, request: Request, current: User,
    claim_id: str, to: Literal["approved", "rejected"], note: str | None,
) -> ClaimOut:
    # FOR UPDATE:并发审批只允许一方通过状态检查
    c = db.execute(
        select(ExpenseClaim).where(ExpenseClaim.id == claim_id).with_for_update()
    ).scalar_one_or_none()
    if not c or c.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "报销单不存在")
    if c.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已处理")
    from app.services.approvals import decide_on_instance, instance_for

    inst = instance_for(db, current.tenant_id, "expense_claim", c.id)
    if inst is not None:
        decide_on_instance(db, inst, current, to, note, request)
    else:
        if not _can_decide(db, current, c):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "只能审批直属下属的报销,不能自批")
        _apply_claim_decision(db, c, to, current, note, request)
    db.commit()
    emps, orgs, types = _maps(db, current.tenant_id)
    items = list(db.scalars(
        select(ExpenseItem).where(ExpenseItem.claim_id == c.id)
    ))
    return _claim_out(c, items, emps, orgs, types)


class DecideIn(BaseModel):
    note: str | None = None


@router.post("/claims/{claim_id}/approve", response_model=ClaimOut)
def approve_claim(
    claim_id: str, body: DecideIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_EXPENSES)),
):
    return _decide(db, request, current, claim_id, "approved", body.note)


@router.post("/claims/{claim_id}/reject", response_model=ClaimOut)
def reject_claim(
    claim_id: str, body: DecideIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_EXPENSES)),
):
    return _decide(db, request, current, claim_id, "rejected", body.note)


@router.post("/claims/{claim_id}/pay", response_model=ClaimOut)
def pay_claim(
    claim_id: str, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EXPENSES)),
):
    c = db.get(ExpenseClaim, claim_id)
    if not c or c.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "报销单不存在")
    if c.status != "approved":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅已批准的报销可标记付款")
    c.status = "paid"
    write_audit(
        db, actor=current, entity_type="expense_claim", entity_id=c.id,
        action="pay", detail={"total": c.total}, request=request,
    )
    db.commit()
    emps, orgs, types = _maps(db, current.tenant_id)
    items = list(db.scalars(
        select(ExpenseItem).where(ExpenseItem.claim_id == c.id)
    ))
    return _claim_out(c, items, emps, orgs, types, current.id)
