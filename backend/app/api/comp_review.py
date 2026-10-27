"""调薪建议矩阵(Comp Review Lite,Workday merit matrix 的建议侧)。

Workday 做法:绩效评级 × compa-ratio(现薪/带宽中点)→ 建议调幅矩阵,
评审人在建议值上调整 → 批准 → 生效。我们只做**建议+决策**层,
不碰算薪本身(项目原则:payroll 是聚合输入,不做计算引擎)。

- 绩效带 = 最近已评定 Appraisal.manager_rating → 1低/2中/3高
- 现薪 = 最新一期 issued SalarySlip.gross;无→中档+estimated
- compa-ratio = gross / 匹配 SalaryBand 中点 → low(<0.8)/mid/high(>1.2)
- 矩阵默认值(可租户覆写 prompt 式?——用固定默认值,理由见 doc):
    高绩效×低带宽比 12% | 高绩效×中 8% | 高绩效×高 5%
    中绩效×低 8%       | 中绩效×中 5% | 中绩效×高 2%
    低绩效×低 4%       | 低绩效×中 2% | 低绩效×高 0%

- ``GET  /comp-review/matrix``        3×3 矩阵落位 + 汇总统计
- ``POST /comp-review/decisions``     记录/调整评审决策(draft)
- ``POST /comp-review/decisions/{id}/approve`` 批准(approved)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    CompReviewDecision,
    Employee,
    OrgUnit,
    SalaryBand,
    SalarySlip,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import PERM_MANAGE_PAYROLL, require_permission

router = APIRouter(
    prefix="/comp-review", tags=["comp-review"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_HI, _LO = 3.75, 2.5
MATRIX: dict[tuple[int, str], float] = {
    (3, "low"): 12.0, (3, "mid"): 8.0, (3, "high"): 5.0,
    (2, "low"): 8.0, (2, "mid"): 5.0, (2, "high"): 2.0,
    (1, "low"): 4.0, (1, "mid"): 2.0, (1, "high"): 0.0,
}
RATIO_LABEL = {"low": "低于带宽 (<0.8)", "mid": "带宽内 (0.8-1.2)", "high": "高于带宽 (>1.2)"}


def _perf_band(rating: float | None) -> int:
    if rating is None:
        return 2
    return 3 if rating >= _HI else 2 if rating >= _LO else 1


def _ratio_band(ratio: float | None) -> str:
    if ratio is None:
        return "mid"
    return "low" if ratio < 0.8 else "high" if ratio > 1.2 else "mid"


def _match_band(
    emp: Employee, bands: list[SalaryBand]
) -> SalaryBand | None:
    """匹配薪资带宽:designation 精确 > 通用带宽;org 匹配 > 全公司。"""
    cands = [
        b for b in bands
        if (b.designation is None or b.designation == emp.designation)
        and (b.org_unit_id is None or b.org_unit_id == emp.org_unit_id)
    ]
    specific = [b for b in cands if b.designation and b.org_unit_id]
    return (specific or [b for b in cands if b.designation] or cands or [None])[0]


def _cycle(c: str | None) -> str:
    return (c or "").strip() or str(datetime.now().year)


class MatrixEmp(BaseModel):
    id: str
    name: str
    designation: str | None = None
    org_name: str | None = None
    perf_band: int
    ratio_band: str
    gross: float | None = None
    band_mid: float | None = None
    compa_ratio: float | None = None
    suggested_pct: float
    estimated: bool = False
    decision_id: str | None = None
    decision_status: str | None = None
    final_pct: float | None = None
    final_amount: float | None = None


class MatrixOut(BaseModel):
    cycle: str
    matrix: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    employees: list[MatrixEmp] = Field(default_factory=list)


@router.get("/matrix", response_model=MatrixOut)
def matrix(
    cycle: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    cyc = _cycle(cycle)
    emps = db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id, Employee.status == "active"
        ).order_by(Employee.name)
    ).all()
    emp_ids = [e.id for e in emps]

    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)).all()}
    bands = db.scalars(
        select(SalaryBand).where(
            SalaryBand.tenant_id == current.tenant_id,
            SalaryBand.is_active.is_(True),
        )
    ).all()

    # 每员工最近 manager_rating + 最新 issued slip
    ratings: dict[str, float] = {}
    for a in db.scalars(
        select(Appraisal)
        .where(Appraisal.tenant_id == current.tenant_id,
               Appraisal.employee_id.in_(emp_ids or ["-"]),
               Appraisal.manager_rating.is_not(None))
        .order_by(Appraisal.id.desc())
    ).all():
        ratings.setdefault(a.employee_id, float(a.manager_rating))
    latest_slip: dict[str, SalarySlip] = {}
    for s in db.scalars(
        select(SalarySlip)
        .where(SalarySlip.tenant_id == current.tenant_id,
               SalarySlip.employee_id.in_(emp_ids or ["-"]),
               SalarySlip.status == "issued")
        .order_by(SalarySlip.period.desc())
    ).all():
        latest_slip.setdefault(s.employee_id, s)

    decisions = {d.employee_id: d for d in db.scalars(
        select(CompReviewDecision).where(
            CompReviewDecision.tenant_id == current.tenant_id,
            CompReviewDecision.cycle == cyc,
        )
    ).all()}

    items: list[MatrixEmp] = []
    cost_sum, pct_sum, decided = 0.0, 0.0, 0
    for e in emps:
        pb = _perf_band(ratings.get(e.id))
        slip = latest_slip.get(e.id)
        band = _match_band(e, bands)
        gross = float(slip.gross) if slip else None
        mid = (
            (float(band.gross_min) + float(band.gross_max)) / 2
            if band else None
        )
        ratio = round(gross / mid, 3) if gross and mid else None
        rb = _ratio_band(ratio)
        sug = MATRIX[(pb, rb)]
        est = gross is None or mid is None
        d = decisions.get(e.id)
        eff_pct = d.final_pct if d and d.final_pct is not None else sug
        if not est and gross:
            cost_sum += gross * eff_pct / 100
            pct_sum += eff_pct
        if d:
            decided += 1
        items.append(MatrixEmp(
            id=e.id, name=e.name, designation=e.designation,
            org_name=orgs.get(e.org_unit_id),
            perf_band=pb, ratio_band=rb,
            gross=gross, band_mid=mid, compa_ratio=ratio,
            suggested_pct=sug, estimated=est,
            decision_id=d.id if d else None,
            decision_status=d.status if d else None,
            final_pct=d.final_pct if d else None,
            final_amount=d.final_amount if d else None,
        ))

    return MatrixOut(
        cycle=cyc,
        matrix={f"{p}_{r}": v for (p, r), v in MATRIX.items()},
        stats={
            "headcount": len(items),
            "covered": len([i for i in items if not i.estimated]),
            "decided": decided,
            "avg_suggested_pct": round(pct_sum / max(1, len([i for i in items if not i.estimated])), 2),
            "est_cost": round(cost_sum),
        },
        employees=items,
    )


class DecisionIn(BaseModel):
    employee_id: str
    cycle: str | None = None
    final_pct: float | None = Field(default=None, ge=0, le=100)
    final_amount: float | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=500)


class DecisionOut(BaseModel):
    id: str
    employee_id: str
    cycle: str
    perf_band: int
    ratio_band: str
    suggested_pct: float
    final_pct: float | None = None
    final_amount: float | None = None
    status: str
    note: str | None = None


def _dout(d: CompReviewDecision) -> DecisionOut:
    return DecisionOut(
        id=d.id, employee_id=d.employee_id, cycle=d.cycle,
        perf_band=d.perf_band, ratio_band=d.ratio_band,
        suggested_pct=d.suggested_pct, final_pct=d.final_pct,
        final_amount=d.final_amount, status=d.status, note=d.note,
    )


@router.post("/decisions", status_code=201, response_model=DecisionOut)
def decide(
    body: DecisionIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    cyc = _cycle(body.cycle)
    # 重算该员工的矩阵落位作为快照依据
    ratings = {
        a.employee_id: float(a.manager_rating)
        for a in db.scalars(
            select(Appraisal)
            .where(Appraisal.tenant_id == current.tenant_id,
                   Appraisal.employee_id == emp.id,
                   Appraisal.manager_rating.is_not(None))
            .order_by(Appraisal.id.desc())
            .limit(1)
        ).all()
    }
    pb = _perf_band(ratings.get(emp.id))
    slip = db.scalar(
        select(SalarySlip)
        .where(SalarySlip.tenant_id == current.tenant_id,
               SalarySlip.employee_id == emp.id,
               SalarySlip.status == "issued")
        .order_by(SalarySlip.period.desc())
        .limit(1)
    )
    band = _match_band(
        emp,
        db.scalars(select(SalaryBand).where(
            SalaryBand.tenant_id == current.tenant_id,
            SalaryBand.is_active.is_(True),
        )).all(),
    )
    gross = float(slip.gross) if slip else None
    mid = (float(band.gross_min) + float(band.gross_max)) / 2 if band else None
    rb = _ratio_band(round(gross / mid, 3) if gross and mid else None)

    d = db.scalar(
        select(CompReviewDecision).where(
            CompReviewDecision.tenant_id == current.tenant_id,
            CompReviewDecision.employee_id == emp.id,
            CompReviewDecision.cycle == cyc,
        )
    )
    if d is None:
        d = CompReviewDecision(
            tenant_id=current.tenant_id, employee_id=emp.id, cycle=cyc,
        )
        db.add(d)
    if d.status == "approved":
        raise HTTPException(409, "已批准的决策不可再调整")
    d.perf_band, d.ratio_band = pb, rb
    d.suggested_pct = MATRIX[(pb, rb)]
    d.final_pct, d.final_amount = body.final_pct, body.final_amount
    if body.note is not None:
        d.note = body.note
    d.decided_by = current.id
    db.flush()
    write_audit(db, actor=current, entity_type="comp_review_decision",
                entity_id=d.id, action="decide",
                detail={"employee_id": emp.id, "final_pct": body.final_pct},
                request=request)
    db.commit()
    return _dout(d)


@router.post("/decisions/{did}/approve", response_model=DecisionOut)
def approve(
    did: str, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_PAYROLL)),
):
    d = db.get(CompReviewDecision, did)
    if not d or d.tenant_id != current.tenant_id:
        raise HTTPException(404, "决策不存在")
    if d.status == "approved":
        raise HTTPException(409, "已批准")
    d.status = "approved"
    d.decided_by = current.id
    write_audit(db, actor=current, entity_type="comp_review_decision",
                entity_id=d.id, action="approve", request=request)
    db.commit()
    return _dout(d)
