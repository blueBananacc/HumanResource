"""排班 API(H3)。

- ``GET/POST/PUT/DELETE /shifts``            班次定义(早/晚班 + 每班需人数)
- ``GET/POST/DELETE /shifts/assignments``    排班明细(按月/员工过滤)
- ``POST /shifts/generate``                  生成排班(dry_run 预览 / 落库)
- ``GET  /shifts/gaps``                      覆盖缺口(含「排班但请假」虚覆盖)
- ``GET  /shifts/cover-candidates``          某天可替班候选人
- ``POST /shifts/auto-cover``                自动替班 + 补缺
- ``POST /shifts/assignments/{id}/reassign`` 手工指定替班人
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import Employee, Shift, ShiftAssignment, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_ATTENDANCE,
    PERM_VIEW_REPORTS,
    require_permission,
)
from app.services.scheduling import (
    auto_cover,
    cover_candidates,
    find_gaps,
    generate_schedule,
    month_range,
)

router = APIRouter(prefix="/shifts", tags=["hris-shifts"],
                 dependencies=[Depends(require_feature("hr.suite"))])


def _min_to_hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


class ShiftOut(BaseModel):
    id: str
    name: str
    start_min: int
    end_min: int
    time_range: str
    required: int
    color: str | None
    active: bool


def _shift_out(s: Shift) -> ShiftOut:
    return ShiftOut(
        id=s.id, name=s.name, start_min=s.start_min, end_min=s.end_min,
        time_range=f"{_min_to_hhmm(s.start_min)}-{_min_to_hhmm(s.end_min)}",
        required=s.required, color=s.color, active=s.active,
    )


class ShiftIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    start_min: int = Field(540, ge=0, lt=1440)
    end_min: int = Field(1020, ge=0, le=1440)
    required: int = Field(1, ge=1, le=99)
    color: str | None = None
    active: bool = True


@router.get("", response_model=list[ShiftOut])
def list_shifts(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    rows = db.scalars(
        select(Shift).where(Shift.tenant_id == current.tenant_id)
        .order_by(Shift.start_min)).all()
    return [_shift_out(s) for s in rows]


@router.post("", response_model=ShiftOut, status_code=status.HTTP_201_CREATED)
def create_shift(
    body: ShiftIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = Shift(tenant_id=current.tenant_id, **body.model_dump())
    db.add(s)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(409, "同名班次已存在")
    db.refresh(s)
    return _shift_out(s)


@router.put("/{shift_id}", response_model=ShiftOut)
def update_shift(
    shift_id: str,
    body: ShiftIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = db.get(Shift, shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    for k, v in body.model_dump().items():
        setattr(s, k, v)
    db.commit()
    return _shift_out(s)


@router.delete("/{shift_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shift(
    shift_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    s = db.get(Shift, shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    has = db.scalars(
        select(ShiftAssignment.id).where(ShiftAssignment.shift_id == shift_id)
        .limit(1)).first()
    if has:
        # 有排班引用的班次不物理删,停用保住历史
        s.active = False
        db.commit()
        return
    db.delete(s)
    db.commit()


# ===== 排班明细 =====

class AssignmentOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    shift_id: str
    shift_name: str
    day: date
    source: str
    note: str | None


def _assign_out(a: ShiftAssignment, emps: dict, shifts: dict) -> AssignmentOut:
    return AssignmentOut(
        id=a.id, employee_id=a.employee_id,
        employee_name=emps.get(a.employee_id, "?"),
        shift_id=a.shift_id, shift_name=shifts.get(a.shift_id, "?"),
        day=a.day, source=a.source, note=a.note,
    )


def _name_maps(db: Session, tenant_id: str):
    emps = {e.id: e.name for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)).all()}
    shifts = {s.id: s.name for s in db.scalars(
        select(Shift).where(Shift.tenant_id == tenant_id)).all()}
    return emps, shifts


@router.get("/assignments", response_model=PageOut[AssignmentOut])
def list_assignments(
    pagination: tuple = Depends(paginate_params),
    month: str | None = Query(None, description="YYYY-MM"),
    employee_id: str | None = None,
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    start, end = month_range(month)
    stmt = select(ShiftAssignment).where(
        ShiftAssignment.tenant_id == current.tenant_id,
        ShiftAssignment.day >= start, ShiftAssignment.day <= end)
    if employee_id:
        stmt = stmt.where(ShiftAssignment.employee_id == employee_id)
    if org_unit_id:
        emp_ids = select(Employee.id).where(
            Employee.tenant_id == current.tenant_id,
            Employee.org_unit_id == org_unit_id)
        stmt = stmt.where(ShiftAssignment.employee_id.in_(emp_ids))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(ShiftAssignment.day, ShiftAssignment.shift_id)
        .limit(limit).offset(offset)).all()
    emps, shifts = _name_maps(db, current.tenant_id)
    return PageOut(
        items=[_assign_out(a, emps, shifts) for a in rows],
        total=total, limit=limit, offset=offset)


class AssignmentIn(BaseModel):
    employee_id: str
    shift_id: str
    day: date
    note: str | None = None


@router.post("/assignments", response_model=AssignmentOut,
             status_code=status.HTTP_201_CREATED)
def create_assignment(
    body: AssignmentIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    s = db.get(Shift, body.shift_id)
    if not s or s.tenant_id != current.tenant_id:
        raise HTTPException(404, "班次不存在")
    a = ShiftAssignment(tenant_id=current.tenant_id, source="manual",
                        **body.model_dump())
    db.add(a)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(409, "该员工当天已有排班")
    emps, shifts = _name_maps(db, current.tenant_id)
    return _assign_out(a, emps, shifts)


@router.delete("/assignments/{assign_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assignment(
    assign_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    a = db.get(ShiftAssignment, assign_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(404, "排班不存在")
    db.delete(a)
    db.commit()


# ===== 生成 / 缺口 / 替班 =====

class GenerateIn(BaseModel):
    month: str | None = Field(None, description="YYYY-MM,默认当月")
    org_unit_id: str | None = None
    shift_ids: list[str] | None = None
    dry_run: bool = True


@router.post("/generate")
def generate(
    body: GenerateIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    start, end = month_range(body.month)
    out = generate_schedule(
        db, current.tenant_id, start=start, end=end,
        org_unit_id=body.org_unit_id, shift_ids=body.shift_ids,
        dry_run=body.dry_run, actor_id=current.id)
    if not body.dry_run:
        write_audit(db, actor=current, entity_type="shift_assignment",
                    entity_id="-", action="generate",
                    detail={"month": start.isoformat()[:7],
                            "planned": out["planned"]},
                    request=request)
        db.commit()
    out["month"] = start.isoformat()[:7]
    return out


@router.get("/gaps")
def gaps(
    month: str | None = Query(None, description="YYYY-MM"),
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    start, end = month_range(month)
    return {"month": start.isoformat()[:7],
            "gaps": find_gaps(db, current.tenant_id, start=start, end=end,
                              org_unit_id=org_unit_id)}


@router.get("/cover-candidates")
def candidates(
    day: date,
    shift_id: str | None = None,
    org_unit_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    return {"day": day.isoformat(),
            "candidates": cover_candidates(
                db, current.tenant_id, day=day, shift_id=shift_id,
                org_unit_id=org_unit_id)}


class AutoCoverIn(BaseModel):
    month: str | None = Field(None, description="YYYY-MM,默认当月")
    dry_run: bool = False


@router.post("/auto-cover")
def cover(
    body: AutoCoverIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    start, end = month_range(body.month)
    out = auto_cover(db, current.tenant_id, start=start, end=end,
                     actor_id=current.id, dry_run=body.dry_run)
    if not body.dry_run:
        write_audit(db, actor=current, entity_type="shift_assignment",
                    entity_id="-", action="auto_cover",
                    detail={"month": start.isoformat()[:7],
                            "covered": len(out["covered"]),
                            "filled": len(out["filled"])},
                    request=request)
        db.commit()
    out["month"] = start.isoformat()[:7]
    return out


class ReassignIn(BaseModel):
    employee_id: str
    note: str | None = None


@router.post("/assignments/{assign_id}/reassign", response_model=AssignmentOut)
def reassign(
    assign_id: str,
    body: ReassignIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_ATTENDANCE)),
):
    a = db.get(ShiftAssignment, assign_id)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(404, "排班不存在")
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id or emp.status != "active":
        raise HTTPException(404, "替班员工不存在或已离职")
    clash = db.scalars(
        select(ShiftAssignment).where(
            ShiftAssignment.tenant_id == current.tenant_id,
            ShiftAssignment.employee_id == body.employee_id,
            ShiftAssignment.day == a.day, ShiftAssignment.id != a.id)
    ).first()
    if clash:
        raise HTTPException(409, "替班员工当天已有排班")
    write_audit(db, actor=current, entity_type="shift_assignment",
                entity_id=a.id, action="reassign",
                detail={"day": str(a.day), "from": a.employee_id,
                        "to": body.employee_id},
                request=request)
    a.employee_id = body.employee_id
    a.source = "cover"
    a.note = body.note or "手工替班"
    db.commit()
    emps, shifts = _name_maps(db, current.tenant_id)
    return _assign_out(a, emps, shifts)
