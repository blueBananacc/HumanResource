"""薪酬带宽 API(MintHCM SalaryRanges 移植)。

- CRUD ``/salary-bands``:读写均 ``manage_payroll`` —— 带宽属薪酬敏感数据,
  不对只读/view_reports 的全量非员工角色(viewer/interviewer/hiring_manager)
  开放;org scope 过滤照常生效。
- ``GET /salary-bands/match?designation=&org_unit_id=``:命中带宽,
  排序 org 专属 > designation 精确 > 通用。
- ``GET /salary-bands/check?designation=&amount=``:offer 定薪时的带宽校验
  (不阻塞,仅返回参考结论)。
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import OrgUnit, SalaryBand, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_PAYROLL,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(
    prefix="/salary-bands",
    tags=["salary-bands"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _today() -> date:
    return datetime.now(UTC).date()


# ============ schemas ============


class BandIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    designation: str | None = Field(default=None, max_length=128)
    org_unit_id: str | None = None
    currency: str = Field(default="CNY", min_length=1, max_length=8)
    gross_min: int = Field(ge=0)
    gross_max: int = Field(ge=0)
    employer_cost_min: int | None = Field(default=None, ge=0)
    employer_cost_max: int | None = Field(default=None, ge=0)
    effective_from: date | None = None
    effective_to: date | None = None
    is_active: bool = True

    @field_validator("gross_max")
    @classmethod
    def _v_range(cls, v: int, info) -> int:
        if "gross_min" in info.data and v < info.data["gross_min"]:
            raise ValueError("gross_max 不能小于 gross_min")
        return v

    @field_validator("employer_cost_max")
    @classmethod
    def _v_cost_range(cls, v: int | None, info) -> int | None:
        lo = info.data.get("employer_cost_min")
        if v is not None and lo is not None and v < lo:
            raise ValueError("employer_cost_max 不能小于 employer_cost_min")
        return v


class BandOut(BaseModel):
    id: str
    name: str
    designation: str | None
    org_unit_id: str | None
    org_unit_name: str | None = None
    currency: str
    gross_min: int
    gross_max: int
    employer_cost_min: int | None
    employer_cost_max: int | None
    effective_from: date | None
    effective_to: date | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


def _out(db: Session, b: SalaryBand) -> BandOut:
    org = db.get(OrgUnit, b.org_unit_id) if b.org_unit_id else None
    return BandOut(
        id=b.id,
        name=b.name,
        designation=b.designation,
        org_unit_id=b.org_unit_id,
        org_unit_name=org.name if org else None,
        currency=b.currency,
        gross_min=b.gross_min,
        gross_max=b.gross_max,
        employer_cost_min=b.employer_cost_min,
        employer_cost_max=b.employer_cost_max,
        effective_from=b.effective_from,
        effective_to=b.effective_to,
        is_active=b.is_active,
        created_at=b.created_at,
    )


def _get(db: Session, bid: str, current: User, scope) -> SalaryBand:
    b = db.get(SalaryBand, bid)
    if b is None or b.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "带宽不存在")
    ensure_can_see(scope, b.org_unit_id)
    return b


def _validate_org(db: Session, org_id: str | None, current: User, scope) -> None:
    if not org_id:
        return
    org = db.get(OrgUnit, org_id)
    if org is None or org.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "组织不存在")
    ensure_can_see(scope, org_id)


# ============ CRUD ============


@router.get("", response_model=PageOut[BandOut])
def list_bands(
    paging: tuple = Depends(paginate_params),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    limit, offset, q = paging
    scope = get_org_scope(db, current)
    stmt = select(SalaryBand).where(SalaryBand.tenant_id == current.tenant_id)
    stmt = apply_q_ilike(stmt, q, SalaryBand.name, SalaryBand.designation)
    if scope is not None:
        if scope:
            stmt = stmt.where(
                (SalaryBand.org_unit_id.in_(scope)) | (SalaryBand.org_unit_id.is_(None))
            )
        else:
            stmt = stmt.where(SalaryBand.org_unit_id.is_(None))
    total = count_total(db, stmt)
    rows = db.scalars(stmt.order_by(SalaryBand.name).limit(limit).offset(offset)).all()
    return PageOut(items=[_out(db, b) for b in rows], total=total, limit=limit, offset=offset)


@router.post("", response_model=BandOut, status_code=status.HTTP_201_CREATED)
def create_band(
    body: BandIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    scope = get_org_scope(db, current)
    _validate_org(db, body.org_unit_id, current, scope)
    b = SalaryBand(tenant_id=current.tenant_id, **body.model_dump())
    db.add(b)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "同名带宽已存在") from None
    write_audit(db, actor=current, entity_type="salary_band", entity_id=b.id, action="create", request=request)
    db.commit()
    return _out(db, b)


@router.put("/{bid}", response_model=BandOut)
def update_band(
    bid: str,
    body: BandIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    scope = get_org_scope(db, current)
    b = _get(db, bid, current, scope)
    _validate_org(db, body.org_unit_id, current, scope)
    for k, v in body.model_dump().items():
        setattr(b, k, v)
    b.updated_at = datetime.now(UTC).replace(tzinfo=None)
    # 与 create 同一口径:重命名撞上 (tenant_id,name) 唯一约束 → 409 而非 500
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "同名带宽已存在") from None
    write_audit(db, actor=current, entity_type="salary_band", entity_id=b.id, action="update", request=request)
    db.commit()
    return _out(db, b)


@router.delete("/{bid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_band(
    bid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    b = _get(db, bid, current, get_org_scope(db, current))
    write_audit(db, actor=current, entity_type="salary_band", entity_id=b.id, action="delete", request=request)
    db.delete(b)
    db.commit()


# ============ match / check ============


def _band_score(b: SalaryBand, designation: str | None, org_id: str | None) -> int | None:
    """匹配优先级:0=org 专属 1=designation 精确 2=通用;None=不匹配。

    org 专属带宽(org_unit_id 非空)只在查询方给出同一 org 时命中 —— 不能
    落到通用档(2),否则 B 组织的带宽会被当成全公司带宽返回给 A 组织
    或未带 org 的查询(offer 校验恒传 org_id=None)。
    """
    if not b.is_active:
        return None
    today = _today()
    if b.effective_from and b.effective_from > today:
        return None
    if b.effective_to and b.effective_to < today:
        return None
    if b.org_unit_id is not None:
        if org_id and b.org_unit_id == org_id:
            return 0
        return None
    if designation and b.designation and b.designation == designation:
        return 1
    if not b.designation:
        return 2
    return None


@router.get("/match")
def match_bands(
    designation: str | None = Query(None, max_length=128),
    org_unit_id: str | None = Query(None),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    scope = get_org_scope(db, current)
    rows = db.scalars(
        select(SalaryBand).where(
            SalaryBand.tenant_id == current.tenant_id,
            SalaryBand.is_active.is_(True),
        )
    ).all()
    scored = [
        (s, b)
        for b in rows
        if (s := _band_score(b, designation, org_unit_id)) is not None
    ]
    scored.sort(key=lambda x: x[0])
    return [
        _out(db, b)
        for s, b in scored
        if scope is None or b.org_unit_id is None or b.org_unit_id in scope
    ]


@router.get("/check")
def check_salary(
    designation: str | None = Query(None, max_length=128),
    org_unit_id: str | None = Query(None),
    amount: int = Query(ge=0),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    """offer 定薪参考:返回命中最优先带宽与 amount 是否在区间内。"""
    scope = get_org_scope(db, current)
    rows = db.scalars(
        select(SalaryBand).where(
            SalaryBand.tenant_id == current.tenant_id,
            SalaryBand.is_active.is_(True),
        )
    ).all()
    scored = [
        (s, b)
        for b in rows
        if (s := _band_score(b, designation, org_unit_id)) is not None
        and (scope is None or b.org_unit_id is None or b.org_unit_id in scope)
    ]
    if not scored:
        return {"band": None, "within": None, "amount": amount}
    scored.sort(key=lambda x: x[0])
    band = scored[0][1]
    return {
        "band": _out(db, band),
        "within": band.gross_min <= amount <= band.gross_max,
        "amount": amount,
    }
