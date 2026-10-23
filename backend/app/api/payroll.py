"""薪酬 API — 轻量工资条(登记制,不做自动核算)。

- ``GET/POST /payroll/slips``             列表(按期间/员工)/新建
- ``GET/PUT/DELETE /payroll/slips/{id}``  详情/编辑(draft)/删除(draft)
- ``POST /payroll/slips/{id}/issue``      draft → issued(员工端可见)
- ``POST /payroll/slips/{id}/retract``    issued → draft(发错回收)
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import Employee, OrgUnit, SalarySlip, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_PAYROLL,
    require_permission,
)

router = APIRouter(prefix="/payroll", tags=["hris-payroll"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SlipItemIn(BaseModel):
    label: str
    kind: Literal["earning", "deduction"]
    amount: float = Field(..., ge=0)


class SlipIn(BaseModel):
    employee_id: str
    period: str = Field(..., pattern=r"^\d{4}-\d{2}$")
    items: list[SlipItemIn] = Field(default_factory=list)
    currency: str = "CNY"
    note: str | None = None

    @field_validator("period")
    @classmethod
    def _v_period(cls, v: str) -> str:
        y, m = v.split("-")
        if not (1 <= int(m) <= 12):
            raise ValueError("期间月份不合法")
        return v


class SlipItemOut(BaseModel):
    label: str
    kind: str
    amount: float


class SlipOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    period: str
    gross: float
    deductions: float
    net: float
    currency: str
    items: list[SlipItemOut] | None
    status: str
    issued_at: datetime | None
    note: str | None
    created_at: datetime


def _maps(db: Session, tenant_id: str):
    emps = {e.id: e for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)
    )}
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
    )}
    return emps, orgs


def _slip_out(s: SalarySlip, emps, orgs) -> SlipOut:
    e = emps.get(s.employee_id)
    return SlipOut(
        id=s.id, employee_id=s.employee_id,
        employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        period=s.period, gross=s.gross, deductions=s.deductions, net=s.net,
        currency=s.currency, items=s.items, status=s.status,
        issued_at=s.issued_at, note=s.note, created_at=s.created_at,
    )


def _totals(items: list[SlipItemIn]) -> tuple[float, float, float]:
    gross = sum(i.amount for i in items if i.kind == "earning")
    ded = sum(i.amount for i in items if i.kind == "deduction")
    return gross, ded, gross - ded


@router.get("/slips", response_model=PageOut[SlipOut])
def list_slips(
    pagination: tuple = Depends(paginate_params),
    period: str | None = None,
    employee_id: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    limit, offset, _ = pagination
    stmt = select(SalarySlip).where(SalarySlip.tenant_id == current.tenant_id)
    if period:
        stmt = stmt.where(SalarySlip.period == period)
    if employee_id:
        stmt = stmt.where(SalarySlip.employee_id == employee_id)
    if status_filter:
        stmt = stmt.where(SalarySlip.status == status_filter)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(SalarySlip.period.desc(), SalarySlip.created_at.desc())
        .limit(limit).offset(offset)
    ).all()
    emps, orgs = _maps(db, current.tenant_id)
    return PageOut(
        items=[_slip_out(s, emps, orgs) for s in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/slips", response_model=SlipOut, status_code=status.HTTP_201_CREATED)
def create_slip(
    body: SlipIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    gross, ded, net = _totals(body.items)
    s = SalarySlip(
        tenant_id=current.tenant_id, employee_id=emp.id, period=body.period,
        gross=gross, deductions=ded, net=net, currency=body.currency,
        items=[i.model_dump() for i in body.items], note=body.note,
    )
    db.add(s)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "该员工此期间的工资条已存在"
        )
    db.refresh(s)
    write_audit(
        db, actor=current, entity_type="salary_slip", entity_id=s.id,
        action="create", detail={"period": s.period, "employee": emp.name},
        request=request,
    )
    db.commit()
    emps, orgs = _maps(db, current.tenant_id)
    return _slip_out(s, emps, orgs)


def _slip(db: Session, sid: str, tenant_id: str) -> SalarySlip:
    s = db.get(SalarySlip, sid)
    if not s or s.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "工资条不存在")
    return s


@router.get("/slips/{sid}", response_model=SlipOut)
def get_slip(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    s = _slip(db, sid, current.tenant_id)
    emps, orgs = _maps(db, current.tenant_id)
    return _slip_out(s, emps, orgs)


@router.put("/slips/{sid}", response_model=SlipOut)
def update_slip(
    sid: str,
    body: SlipIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    s = _slip(db, sid, current.tenant_id)
    if s.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已发放的工资条先回收再改")
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    gross, ded, net = _totals(body.items)
    s.employee_id = emp.id
    s.period = body.period
    s.items = [i.model_dump() for i in body.items]
    s.gross, s.deductions, s.net = gross, ded, net
    s.currency = body.currency
    s.note = body.note
    write_audit(
        db, actor=current, entity_type="salary_slip", entity_id=s.id,
        action="update", detail={"net": s.net}, request=request,
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "该员工此期间的工资条已存在"
        )
    db.refresh(s)
    emps, orgs = _maps(db, current.tenant_id)
    return _slip_out(s, emps, orgs)


@router.delete("/slips/{sid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_slip(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    s = _slip(db, sid, current.tenant_id)
    if s.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已发放的工资条先回收再删")
    write_audit(
        db, actor=current, entity_type="salary_slip", entity_id=s.id,
        action="delete", detail={"period": s.period}, request=request,
    )
    db.delete(s)
    db.commit()


@router.post("/slips/{sid}/issue", response_model=SlipOut)
def issue_slip(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    s = _slip(db, sid, current.tenant_id)
    if s.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅草稿可发放")
    s.status = "issued"
    s.issued_at = _now()
    write_audit(
        db, actor=current, entity_type="salary_slip", entity_id=s.id,
        action="issue", detail={"period": s.period}, request=request,
    )
    db.commit()
    db.refresh(s)
    emps, orgs = _maps(db, current.tenant_id)
    return _slip_out(s, emps, orgs)


@router.post("/slips/{sid}/retract", response_model=SlipOut)
def retract_slip(
    sid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    s = _slip(db, sid, current.tenant_id)
    if s.status != "issued":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅已发放的可回收")
    s.status = "draft"
    s.issued_at = None
    write_audit(
        db, actor=current, entity_type="salary_slip", entity_id=s.id,
        action="retract", detail={"period": s.period}, request=request,
    )
    db.commit()
    db.refresh(s)
    emps, orgs = _maps(db, current.tenant_id)
    return _slip_out(s, emps, orgs)
