"""绩效 API:目标 + 考核周期 + 考核表。

- ``GET/POST/PUT/DELETE /performance/goals``          目标(HR 管全员,员工端 /me/goals)
- ``GET/POST /performance/cycles``                    考核周期(draft→open→closed)
- ``POST /performance/cycles/{id}/open|close``
- ``GET/POST /performance/appraisals``                考核表;POST 为周期批量铺表
- ``POST /performance/appraisals/{id}/review``        上级/HR 评分
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    AppraisalCycle,
    Employee,
    Goal,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_PERFORMANCE,
    PERM_VIEW_REPORTS,
    require_permission,
    visible_employee_ids,
)

router = APIRouter(prefix="/performance", tags=["hris-performance"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _emps(db: Session, tenant_id: str):
    return {e.id: e for e in db.scalars(
        select(Employee).where(Employee.tenant_id == tenant_id)
    )}


# ============ goals ============


class GoalIn(BaseModel):
    employee_id: str
    title: str
    description: str = ""
    due_date: date | None = None


class GoalOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    title: str
    description: str
    due_date: date | None
    progress: int
    status: str


def _goal_out(g: Goal, emps: dict) -> GoalOut:
    e = emps.get(g.employee_id)
    return GoalOut(
        id=g.id, employee_id=g.employee_id,
        employee_name=e.name if e else "?", title=g.title,
        description=g.description, due_date=g.due_date,
        progress=g.progress, status=g.status,
    )


@router.get("/goals", response_model=PageOut[GoalOut])
def list_goals(
    pagination: tuple = Depends(paginate_params),
    employee_id: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(Goal).where(Goal.tenant_id == current.tenant_id)
    if employee_id:
        stmt = stmt.where(Goal.employee_id == employee_id)
    if status_filter:
        stmt = stmt.where(Goal.status == status_filter)
    ids = visible_employee_ids(db, current)
    if ids is not None:
        stmt = stmt.where(Goal.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Goal.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps = _emps(db, current.tenant_id)
    return PageOut(items=[_goal_out(g, emps) for g in rows],
                   total=total, limit=limit, offset=offset)


@router.post("/goals", response_model=GoalOut, status_code=status.HTTP_201_CREATED)
def create_goal(
    body: GoalIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    g = Goal(
        tenant_id=current.tenant_id, employee_id=emp.id,
        title=body.title.strip(), description=body.description,
        due_date=body.due_date,
    )
    db.add(g)
    db.commit()
    db.refresh(g)
    return _goal_out(g, _emps(db, current.tenant_id))


@router.patch("/goals/{goal_id}", response_model=GoalOut)
def update_goal(
    goal_id: str,
    body: GoalIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    g = db.get(Goal, goal_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "目标不存在")
    g.title = body.title.strip()
    g.description = body.description
    g.due_date = body.due_date
    db.commit()
    return _goal_out(g, _emps(db, current.tenant_id))


@router.delete("/goals/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_goal(
    goal_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    g = db.get(Goal, goal_id)
    if not g or g.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "目标不存在")
    db.delete(g)
    db.commit()


# ============ cycles ============


class CycleIn(BaseModel):
    name: str
    period_start: date
    period_end: date


class CycleOut(BaseModel):
    id: str
    name: str
    period_start: date
    period_end: date
    status: str
    appraisal_count: int = 0


@router.get("/cycles", response_model=list[CycleOut])
def list_cycles(
    db: Session = Depends(get_db), current: User = Depends(require_permission(PERM_VIEW_REPORTS))
):
    rows = db.scalars(
        select(AppraisalCycle).where(
            AppraisalCycle.tenant_id == current.tenant_id
        ).order_by(AppraisalCycle.created_at.desc())
    ).all()
    out = []
    for c in rows:
        n = len(db.scalars(
            select(Appraisal.id).where(Appraisal.cycle_id == c.id)
        ).all())
        out.append(CycleOut(
            id=c.id, name=c.name, period_start=c.period_start,
            period_end=c.period_end, status=c.status, appraisal_count=n,
        ))
    return out


@router.post("/cycles", response_model=CycleOut, status_code=status.HTTP_201_CREATED)
def create_cycle(
    body: CycleIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    c = AppraisalCycle(
        tenant_id=current.tenant_id, name=body.name.strip(),
        period_start=body.period_start, period_end=body.period_end,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return CycleOut(id=c.id, name=c.name, period_start=c.period_start,
                    period_end=c.period_end, status=c.status)


def _cycle(db: Session, cid: str, tenant_id: str) -> AppraisalCycle:
    c = db.get(AppraisalCycle, cid)
    if not c or c.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "考核周期不存在")
    return c


@router.post("/cycles/{cid}/open", response_model=CycleOut)
def open_cycle(
    cid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    c = _cycle(db, cid, current.tenant_id)
    if c.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅草稿可开启")
    c.status = "open"
    write_audit(db, actor=current, entity_type="appraisal_cycle", entity_id=c.id,
                action="open", detail={}, request=request)
    db.commit()
    return CycleOut(id=c.id, name=c.name, period_start=c.period_start,
                    period_end=c.period_end, status=c.status)


@router.post("/cycles/{cid}/close", response_model=CycleOut)
def close_cycle(
    cid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    c = _cycle(db, cid, current.tenant_id)
    if c.status != "open":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅开启中的周期可关闭")
    c.status = "closed"
    write_audit(db, actor=current, entity_type="appraisal_cycle", entity_id=c.id,
                action="close", detail={}, request=request)
    db.commit()
    return CycleOut(id=c.id, name=c.name, period_start=c.period_start,
                    period_end=c.period_end, status=c.status)


# ============ appraisals ============


class AppraisalIn(BaseModel):
    cycle_id: str
    employee_ids: list[str] = Field(..., min_length=1)


class AppraisalOut(BaseModel):
    id: str
    cycle_id: str
    cycle_name: str
    employee_id: str
    employee_name: str
    org_unit_name: str | None
    status: str
    self_review: str | None
    self_rating: int | None
    manager_review: str | None
    manager_rating: int | None
    updated_at: datetime


def _appraisal_out(a: Appraisal, cycles: dict, emps: dict, orgs: dict) -> AppraisalOut:
    e = emps.get(a.employee_id)
    c = cycles.get(a.cycle_id)
    return AppraisalOut(
        id=a.id, cycle_id=a.cycle_id, cycle_name=c.name if c else "?",
        employee_id=a.employee_id, employee_name=e.name if e else "?",
        org_unit_name=orgs.get(e.org_unit_id) if e and e.org_unit_id else None,
        status=a.status, self_review=a.self_review, self_rating=a.self_rating,
        manager_review=a.manager_review, manager_rating=a.manager_rating,
        updated_at=a.updated_at,
    )


def _perf_maps(db: Session, tenant_id: str):
    cycles = {c.id: c for c in db.scalars(
        select(AppraisalCycle).where(AppraisalCycle.tenant_id == tenant_id)
    )}
    emps = _emps(db, tenant_id)
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
    )}
    return cycles, emps, orgs


@router.get("/appraisals", response_model=PageOut[AppraisalOut])
def list_appraisals(
    pagination: tuple = Depends(paginate_params),
    cycle_id: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(Appraisal).where(Appraisal.tenant_id == current.tenant_id)
    if cycle_id:
        stmt = stmt.where(Appraisal.cycle_id == cycle_id)
    if status_filter:
        stmt = stmt.where(Appraisal.status == status_filter)
    ids = visible_employee_ids(db, current)
    if ids is not None:
        stmt = stmt.where(Appraisal.employee_id.in_(ids or ["__none__"]))
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Appraisal.updated_at.desc()).limit(limit).offset(offset)
    ).all()
    cycles, emps, orgs = _perf_maps(db, current.tenant_id)
    return PageOut(
        items=[_appraisal_out(a, cycles, emps, orgs) for a in rows],
        total=total, limit=limit, offset=offset,
    )


@router.post("/appraisals", response_model=list[AppraisalOut],
             status_code=status.HTTP_201_CREATED)
def spawn_appraisals(
    body: AppraisalIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    cycle = _cycle(db, body.cycle_id, current.tenant_id)
    if cycle.status != "open":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅开启中的周期可铺表")
    made: list[Appraisal] = []
    for emp_id in body.employee_ids:
        emp = db.get(Employee, emp_id)
        if not emp or emp.tenant_id != current.tenant_id:
            continue
        a = Appraisal(tenant_id=current.tenant_id, cycle_id=cycle.id,
                      employee_id=emp.id)
        db.add(a)
        try:
            db.flush()
            made.append(a)
        except IntegrityError:
            db.rollback()
            continue
    db.commit()
    cycles, emps, orgs = _perf_maps(db, current.tenant_id)
    return [_appraisal_out(a, cycles, emps, orgs) for a in made]


class ReviewIn(BaseModel):
    manager_review: str
    manager_rating: int = Field(..., ge=1, le=5)


@router.post("/appraisals/{aid}/review", response_model=AppraisalOut)
def review_appraisal(
    aid: str,
    body: ReviewIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PERFORMANCE)),
):
    a = db.get(Appraisal, aid)
    if not a or a.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "考核表不存在")
    if a.status != "self_submitted":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "员工尚未提交自评或已评定"
        )
    a.manager_review = body.manager_review.strip()
    a.manager_rating = body.manager_rating
    a.reviewer_id = current.id
    a.status = "reviewed"
    a.updated_at = _now()
    write_audit(
        db, actor=current, entity_type="appraisal", entity_id=a.id,
        action="review", detail={"rating": a.manager_rating}, request=request,
    )
    db.commit()
    cycles, emps, orgs = _perf_maps(db, current.tenant_id)
    return _appraisal_out(a, cycles, emps, orgs)
