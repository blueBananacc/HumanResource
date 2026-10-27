"""职位管理(Position Management,Workday staffing model)。

职位 = 独立于任职人的"座位":open→filled→(frozen/closed)。
- fill:任职人上岗(position.incumbent + employee.position_id 双向回填)
- vacate:离职/调动释放(清两边,status 回 open)
- frozen:冻结(保留职位不可占坑);closed:撤销
- 空缺率 = open / 非 closed 总数(北森"编制达成"的等价口径)

- ``GET  /positions``                  列表 + stats(open/filled/frozen/空缺率/按组织)
- ``POST /positions``                  新建(默认 open)
- ``PATCH /positions/{id}``            编辑字段 / 状态流转(frozen↔open/closed)
- ``POST /positions/{id}/fill``        占坑(校验:员工 active + 无现任 + 职位 open)
- ``POST /positions/{id}/vacate``      释放(可选 note,如离职原因)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import Employee, OrgUnit, Position, SalaryBand, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_MANAGE_ORG,
    has_permission,
    visible_employee_ids,
)

router = APIRouter(
    prefix="/positions", tags=["positions"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

POSITION_STATUSES = ("open", "filled", "frozen", "closed")


def _can_view(current: User) -> bool:
    return has_permission(current, PERM_MANAGE_ORG) or has_permission(
        current, PERM_MANAGE_EMPLOYEES
    )


def _check_write(current: User) -> None:
    if not has_permission(current, PERM_MANAGE_ORG):
        raise HTTPException(403, "没有职位管理权限")


def _pos(db: Session, pid: str, tenant_id: str) -> Position:
    p = db.get(Position, pid)
    if not p or p.tenant_id != tenant_id:
        raise HTTPException(404, "职位不存在")
    return p


class PositionIn(BaseModel):
    title: str = Field(min_length=1, max_length=128)
    designation: str | None = Field(default=None, max_length=128)
    org_unit_id: str | None = None
    salary_band_id: str | None = None
    note: str | None = Field(default=None, max_length=500)


class PositionPatchIn(PositionIn):
    title: str | None = Field(default=None, max_length=128)
    status: str | None = None


class PositionOut(BaseModel):
    id: str
    title: str
    designation: str | None = None
    org_unit_id: str | None = None
    org_name: str | None = None
    salary_band_id: str | None = None
    band_name: str | None = None
    status: str
    incumbent_id: str | None = None
    incumbent_name: str | None = None
    opened_at: str | None = None
    filled_at: str | None = None
    created_at: str


class PositionListOut(BaseModel):
    items: list[PositionOut]
    stats: dict[str, Any]


def _out(p: Position, org: str | None, band: str | None, inc: str | None) -> PositionOut:
    return PositionOut(
        id=p.id, title=p.title, designation=p.designation,
        org_unit_id=p.org_unit_id, org_name=org,
        salary_band_id=p.salary_band_id, band_name=band,
        status=p.status, incumbent_id=p.incumbent_id, incumbent_name=inc,
        opened_at=str(p.opened_at or "")[:16] or None,
        filled_at=str(p.filled_at or "")[:16] or None,
        created_at=str(p.created_at or "")[:16],
    )


@router.get("", response_model=PositionListOut)
def list_positions(
    org_unit_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not _can_view(current):
        raise HTTPException(403, "没有职位查看权限")
    q = select(Position).where(Position.tenant_id == current.tenant_id)
    if org_unit_id:
        q = q.where(Position.org_unit_id == org_unit_id)
    if status:
        q = q.where(Position.status == status)
    rows = db.scalars(q.order_by(Position.created_at.desc())).all()

    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)).all()}
    bands = {b.id: b.name for b in db.scalars(
        select(SalaryBand).where(SalaryBand.tenant_id == current.tenant_id)).all()}
    inc_ids = [p.incumbent_id for p in rows if p.incumbent_id]
    emps = {e.id: e.name for e in db.scalars(
        select(Employee).where(Employee.id.in_(inc_ids or ["-"]))).all()}

    by_status = {"open": 0, "filled": 0, "frozen": 0, "closed": 0}
    by_org: dict[str, dict[str, int]] = {}
    items = []
    for p in rows:
        by_status[p.status] = by_status.get(p.status, 0) + 1
        if p.status != "closed":
            org_key = orgs.get(p.org_unit_id) or "未分配"
            slot = by_org.setdefault(org_key, {"open": 0, "filled": 0, "frozen": 0})
            slot[p.status] = slot.get(p.status, 0) + 1
        items.append(_out(
            p, orgs.get(p.org_unit_id), bands.get(p.salary_band_id),
            emps.get(p.incumbent_id),
        ))
    live = by_status["open"] + by_status["filled"] + by_status["frozen"]
    stats = {
        "total": len(rows),
        **by_status,
        "vacancy_rate": round(by_status["open"] / live * 100, 1) if live else 0.0,
        "by_org": by_org,
    }
    return PositionListOut(items=items, stats=stats)


@router.post("", status_code=201, response_model=PositionOut)
def create_position(
    body: PositionIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    if body.org_unit_id:
        org = db.get(OrgUnit, body.org_unit_id)
        if not org or org.tenant_id != current.tenant_id:
            raise HTTPException(400, "部门不存在")
    if body.salary_band_id:
        band = db.get(SalaryBand, body.salary_band_id)
        if not band or band.tenant_id != current.tenant_id:
            raise HTTPException(400, "薪资带宽不存在")
    p = Position(
        tenant_id=current.tenant_id, title=body.title.strip(),
        designation=body.designation, org_unit_id=body.org_unit_id,
        salary_band_id=body.salary_band_id, note=body.note,
        status="open", opened_at=datetime.now(), created_by=current.id,
    )
    db.add(p)
    db.flush()
    write_audit(db, actor=current, entity_type="position",
                entity_id=p.id, action="create", request=request)
    db.commit()
    return _out(p, None, None, None)


@router.patch("/{pid}", response_model=PositionOut)
def update_position(
    pid: str, body: PositionPatchIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pos(db, pid, current.tenant_id)
    if body.title is not None:
        p.title = body.title.strip()
    for f in ("designation", "org_unit_id", "salary_band_id", "note"):
        v = getattr(body, f)
        if v is not None:
            setattr(p, f, v)
    if body.status is not None:
        if body.status not in POSITION_STATUSES:
            raise HTTPException(400, "无效状态")
        if body.status == "closed" and p.incumbent_id:
            raise HTTPException(400, "职位有人在编,请先释放")
        if body.status == "open" and p.status == "filled":
            raise HTTPException(400, "在编职位不能直接改 open,请用 vacate")
        if body.status == "closed":
            p.closed_at = datetime.now()
        p.status = body.status
    write_audit(db, actor=current, entity_type="position",
                entity_id=p.id, action="update",
                detail={"status": p.status}, request=request)
    db.commit()
    org = db.get(OrgUnit, p.org_unit_id) if p.org_unit_id else None
    band = db.get(SalaryBand, p.salary_band_id) if p.salary_band_id else None
    inc = db.get(Employee, p.incumbent_id) if p.incumbent_id else None
    return _out(p, org.name if org else None,
                band.name if band else None, inc.name if inc else None)


class FillIn(BaseModel):
    employee_id: str


@router.post("/{pid}/fill", response_model=PositionOut)
def fill_position(
    pid: str, body: FillIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pos(db, pid, current.tenant_id)
    if p.status == "frozen":
        raise HTTPException(400, "职位已冻结,先解冻")
    if p.status == "closed":
        raise HTTPException(400, "职位已撤销")
    if p.incumbent_id:
        raise HTTPException(409, "职位已有人在编")
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    scope = visible_employee_ids(db, current)
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在或无权操作")
    if emp.status != "active":
        raise HTTPException(400, "员工非在职状态")
    if emp.position_id and emp.position_id != p.id:
        # 换岗:先释放旧职位
        old = db.get(Position, emp.position_id)
        if old and old.incumbent_id == emp.id:
            old.incumbent_id = None
            old.status = "open"
    p.incumbent_id = emp.id
    p.status = "filled"
    p.filled_at = datetime.now()
    emp.position_id = p.id
    write_audit(db, actor=current, entity_type="position",
                entity_id=p.id, action="fill",
                detail={"employee_id": emp.id}, request=request)
    write_audit(db, actor=current, entity_type="employee",
                entity_id=emp.id, action="position_fill",
                detail={"position_id": p.id, "title": p.title}, request=request)
    db.commit()
    return _out(p, None, None, emp.name)


@router.post("/{pid}/vacate", response_model=PositionOut)
def vacate_position(
    pid: str, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pos(db, pid, current.tenant_id)
    if not p.incumbent_id:
        raise HTTPException(400, "职位本就空缺")
    emp = db.get(Employee, p.incumbent_id)
    emp_id = p.incumbent_id
    p.incumbent_id = None
    p.status = "open"
    p.filled_at = None
    if emp and emp.position_id == p.id:
        emp.position_id = None
    write_audit(db, actor=current, entity_type="position",
                entity_id=p.id, action="vacate",
                detail={"employee_id": emp_id}, request=request)
    db.commit()
    return _out(p, None, None, None)


def release_employee_position(db: Session, emp: Employee) -> None:
    """员工离职/调岗时释放其占坑职位(供 employee/lifecycle 调用)。"""
    if not emp.position_id:
        return
    p = db.get(Position, emp.position_id)
    if p and p.incumbent_id == emp.id:
        p.incumbent_id = None
        p.status = "open"
        p.filled_at = None
    emp.position_id = None
