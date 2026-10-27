"""人才盘点九宫格 + 人才池(对标北森继任云)。

北森流程:评价结果 → 系统自动初盘落位 → 校准会拖拽调格 → 一键入池 →
池内分阶段梯队(入池/培养中/就绪/出池)+ 饱和度看板。我们的数据映射:

- 绩效 X = 最近一条已评定 Appraisal.manager_rating(1-5)
- 潜力 Y = 已确认技能均级 ×0.6 + 目标完成均值(0-100→0-5) ×0.4
- 档带阈值:>=3.75 高 / 2.5-3.75 中 / <2.5 低;无数据→中档+estimated
- 校准会拖拽 = POST calibrate 覆盖 band(记录校准人/时间/备注)
- 一键入池 = POST pools/{id}/members(盘点页批量)

- ``GET  /talent-review/nine-box``                 初盘+校准后九宫格
- ``POST /talent-review/nine-box/{eid}/calibrate`` 校准覆盖(可传 null 重置)
- ``GET/POST/PATCH/DELETE /talent-pools``          人才池 CRUD
- ``GET  /talent-pools/{id}``                      池详情+饱和度
- ``POST /talent-pools/{id}/members``              一键入池
- ``PATCH/DELETE /talent-pools/{id}/members/{eid}`` 阶段流转/出池
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Appraisal,
    Employee,
    EmployeeSkill,
    Goal,
    NineBoxAssessment,
    OrgUnit,
    TalentPool,
    TalentPoolMember,
    User,
)
from app.infra.db import get_db
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_MANAGE_PERFORMANCE,
    has_permission,
    visible_employee_ids,
)
from app.services.audit import write_audit
from fastapi import Request

router = APIRouter(
    prefix="/talent-review", tags=["talent-review"],
    dependencies=[Depends(require_feature("hr.suite"))],
)
pools = APIRouter(
    prefix="/talent-pools", tags=["talent-pools"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

_HI = 3.75
_LO = 2.5


def _band(score: float | None) -> int:
    if score is None:
        return 2
    return 3 if score >= _HI else 2 if score >= _LO else 1


def _check_view(db: Session, current: User) -> set[str] | None:
    if not (
        has_permission(current, PERM_MANAGE_PERFORMANCE)
        or has_permission(current, PERM_MANAGE_EMPLOYEES)
    ):
        raise HTTPException(403, "没有人才盘点权限")
    return visible_employee_ids(db, current)


def _check_write(current: User) -> None:
    if not has_permission(current, PERM_MANAGE_PERFORMANCE):
        raise HTTPException(403, "没有盘点校准权限")


def _auto_scores(
    db: Session, tenant_id: str, emp_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """聚合三源信号→每员工 {perf, potential, perf_src, potential_src}。"""
    out: dict[str, dict[str, Any]] = {
        eid: {"perf": None, "potential": None, "perf_src": "", "potential_src": ""}
        for eid in emp_ids
    }
    if not emp_ids:
        return out

    # 绩效:每员工最近一条 reviewed Appraisal 的 manager_rating
    apprs = db.scalars(
        select(Appraisal)
        .where(Appraisal.tenant_id == tenant_id, Appraisal.employee_id.in_(emp_ids))
        .order_by(Appraisal.id.desc())
    ).all()
    for a in apprs:
        slot = out[a.employee_id]
        if slot["perf"] is None and a.manager_rating is not None:
            slot["perf"] = float(a.manager_rating)
            slot["perf_src"] = "最近绩效评定"
    # 简化:取每个员工最新一条(任意状态)带 manager_rating 的
    for a in apprs:
        slot = out[a.employee_id]
        if slot["perf"] is None and a.self_rating is not None:
            slot["perf"] = float(a.self_rating)
            slot["perf_src"] = "自评分(暂代)"

    # 潜力 0.6*技能均级 + 0.4*目标完成
    skill_rows = db.execute(
        select(EmployeeSkill.employee_id, func.avg(EmployeeSkill.level))
        .where(
            EmployeeSkill.tenant_id == tenant_id,
            EmployeeSkill.employee_id.in_(emp_ids),
            EmployeeSkill.status == "confirmed",
        )
        .group_by(EmployeeSkill.employee_id)
    ).all()
    skill_avg = {r[0]: float(r[1]) for r in skill_rows}
    goal_rows = db.execute(
        select(Goal.employee_id, func.avg(Goal.progress))
        .where(Goal.tenant_id == tenant_id, Goal.employee_id.in_(emp_ids))
        .group_by(Goal.employee_id)
    ).all()
    goal_avg = {r[0]: float(r[1]) / 20.0 for r in goal_rows}  # 0-100 → 0-5
    for eid in emp_ids:
        parts, weights, srcs = [], [], []
        if eid in skill_avg:
            parts.append(skill_avg[eid]); weights.append(0.6); srcs.append("技能")
        if eid in goal_avg:
            parts.append(goal_avg[eid]); weights.append(0.4); srcs.append("目标")
        if parts:
            w = sum(weights)
            out[eid]["potential"] = round(sum(p * x for p, x in zip(parts, weights)) / w, 2)
            out[eid]["potential_src"] = "+".join(srcs)
    return out


class CellEmployee(BaseModel):
    id: str
    name: str
    designation: str | None = None
    org_name: str | None = None
    perf_score: float | None = None
    potential_score: float | None = None
    perf_band: int
    potential_band: int
    auto_perf_band: int
    auto_potential_band: int
    calibrated: bool = False
    estimated: bool = False
    pools: list[str] = Field(default_factory=list)


class NineBoxOut(BaseModel):
    cycle: str
    counts: dict[str, int] = Field(default_factory=dict)
    employees: list[CellEmployee] = Field(default_factory=list)


def _cycle(cycle: str | None) -> str:
    return (cycle or "").strip() or f"{datetime.now().year}-H{1 if datetime.now().month <= 6 else 2}"


@router.get("/nine-box", response_model=NineBoxOut)
def nine_box(
    cycle: str | None = Query(default=None),
    org_unit_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    scope = _check_view(db, current)
    cyc = _cycle(cycle)
    q = select(Employee).where(
        Employee.tenant_id == current.tenant_id, Employee.status == "active"
    )
    if scope is not None:
        q = q.where(Employee.id.in_(scope))
    if org_unit_id:
        q = q.where(Employee.org_unit_id == org_unit_id)
    emps = db.scalars(q.order_by(Employee.name)).all()

    org_names = {
        o.id: o.name
        for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)
        ).all()
    }
    scores = _auto_scores(db, current.tenant_id, [e.id for e in emps])

    assessments = {
        a.employee_id: a
        for a in db.scalars(
            select(NineBoxAssessment).where(
                NineBoxAssessment.tenant_id == current.tenant_id,
                NineBoxAssessment.cycle == cyc,
            )
        ).all()
    }
    # 池归属
    pool_rows = db.execute(
        select(TalentPoolMember.employee_id, TalentPool.name)
        .join(TalentPool, TalentPool.id == TalentPoolMember.pool_id)
        .where(
            TalentPoolMember.tenant_id == current.tenant_id,
            TalentPoolMember.stage != "exited",
        )
    ).all()
    emp_pools: dict[str, list[str]] = {}
    for eid, pname in pool_rows:
        emp_pools.setdefault(eid, []).append(pname)

    items: list[CellEmployee] = []
    counts = {"3_3": 0}
    dirty = False
    for e in emps:
        sc = scores[e.id]
        auto_pb, auto_tb = _band(sc["perf"]), _band(sc["potential"])
        asm = assessments.get(e.id)
        if asm is None:
            asm = NineBoxAssessment(
                tenant_id=current.tenant_id, employee_id=e.id, cycle=cyc,
                auto_perf=sc["perf"], auto_potential=sc["potential"],
                perf_band=auto_pb, potential_band=auto_tb,
                calibrated=False,
            )
            db.add(asm)
            dirty = True
        estimated = sc["perf"] is None and sc["potential"] is None
        items.append(CellEmployee(
            id=e.id, name=e.name, designation=e.designation,
            org_name=org_names.get(e.org_unit_id),
            perf_score=sc["perf"], potential_score=sc["potential"],
            perf_band=asm.perf_band, potential_band=asm.potential_band,
            auto_perf_band=auto_pb, auto_potential_band=auto_tb,
            calibrated=bool(asm.calibrated), estimated=estimated,
            pools=emp_pools.get(e.id, []),
        ))
        counts[f"{asm.perf_band}_{asm.potential_band}"] = (
            counts.get(f"{asm.perf_band}_{asm.potential_band}", 0) + 1
        )
    if dirty:
        db.commit()
    counts.pop("3_3", None) if counts["3_3"] == 0 else None
    return NineBoxOut(cycle=cyc, counts=counts, employees=items)


class CalibrateIn(BaseModel):
    perf_band: int | None = Field(default=None, ge=1, le=3)
    potential_band: int | None = Field(default=None, ge=1, le=3)
    note: str | None = Field(default=None, max_length=500)
    cycle: str | None = None


@router.post("/nine-box/{emp_id}/calibrate")
def calibrate(
    emp_id: str, body: CalibrateIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    _check_write(current)
    emp = db.get(Employee, emp_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    scope = visible_employee_ids(db, current)
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在")
    cyc = _cycle(body.cycle)
    asm = db.scalar(
        select(NineBoxAssessment).where(
            NineBoxAssessment.tenant_id == current.tenant_id,
            NineBoxAssessment.employee_id == emp.id,
            NineBoxAssessment.cycle == cyc,
        )
    )
    if asm is None:
        asm = NineBoxAssessment(
            tenant_id=current.tenant_id, employee_id=emp.id, cycle=cyc
        )
        db.add(asm)

    if body.perf_band is None and body.potential_band is None:
        # 重置回自动落位
        sc = _auto_scores(db, current.tenant_id, [emp.id])[emp.id]
        asm.perf_band = _band(sc["perf"])
        asm.potential_band = _band(sc["potential"])
        asm.calibrated = False
        asm.calibrated_by = None
        asm.calibrated_at = None
        action = "nine_box_reset"
    else:
        asm.perf_band = body.perf_band or asm.perf_band
        asm.potential_band = body.potential_band or asm.potential_band
        asm.calibrated = True
        asm.calibrated_by = current.id
        asm.calibrated_at = datetime.now()
        action = "nine_box_calibrate"
    if body.note is not None:
        asm.note = body.note
    db.flush()
    write_audit(
        db, actor=current, entity_type="nine_box", entity_id=asm.id,
        action=action,
        detail={"employee_id": emp.id, "perf": asm.perf_band, "potential": asm.potential_band},
        request=request,
    )
    db.commit()
    return {
        "ok": True, "perf_band": asm.perf_band,
        "potential_band": asm.potential_band, "calibrated": asm.calibrated,
    }


# ==================== 人才池 ====================

STAGES = ("in", "developing", "ready", "exited")
POOL_KINDS = ("succession", "high_potential", "custom")


class PoolIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    kind: str = Field(default="custom")
    target_designation: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1000)


class PoolOut(BaseModel):
    id: str
    name: str
    kind: str
    target_designation: str | None = None
    description: str | None = None
    member_count: int = 0
    saturation: dict[str, int] = Field(default_factory=dict)


def _pool(db: Session, pool_id: str, tenant_id: str) -> TalentPool:
    p = db.get(TalentPool, pool_id)
    if not p or p.tenant_id != tenant_id:
        raise HTTPException(404, "人才池不存在")
    return p


def _saturation(db: Session, pool_id: str) -> dict[str, int]:
    rows = db.execute(
        select(TalentPoolMember.stage, func.count())
        .where(TalentPoolMember.pool_id == pool_id)
        .group_by(TalentPoolMember.stage)
    ).all()
    return {s: c for s, c in rows}


def _pool_out(db: Session, p: TalentPool) -> PoolOut:
    sat = _saturation(db, p.id)
    active = sum(c for s, c in sat.items() if s != "exited")
    return PoolOut(
        id=p.id, name=p.name, kind=p.kind,
        target_designation=p.target_designation, description=p.description,
        member_count=active, saturation=sat,
    )


@pools.get("", response_model=list[PoolOut])
def list_pools(
    db: Session = Depends(get_db), current: User = Depends(get_current_user)
):
    _check_view(db, current)
    rows = db.scalars(
        select(TalentPool)
        .where(TalentPool.tenant_id == current.tenant_id)
        .order_by(TalentPool.created_at)
    ).all()
    return [_pool_out(db, p) for p in rows]


@pools.post("", status_code=201, response_model=PoolOut)
def create_pool(
    body: PoolIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    if body.kind not in POOL_KINDS:
        raise HTTPException(400, "无效的池类型")
    p = TalentPool(
        tenant_id=current.tenant_id, name=body.name.strip(), kind=body.kind,
        target_designation=body.target_designation, description=body.description,
        created_by=current.id,
    )
    db.add(p)
    db.flush()
    write_audit(db, actor=current, entity_type="talent_pool",
                entity_id=p.id, action="create", request=request)
    db.commit()
    return _pool_out(db, p)


@pools.patch("/{pool_id}", response_model=PoolOut)
def update_pool(
    pool_id: str, body: PoolIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pool(db, pool_id, current.tenant_id)
    p.name, p.kind = body.name.strip(), body.kind
    p.target_designation, p.description = body.target_designation, body.description
    write_audit(db, actor=current, entity_type="talent_pool",
                entity_id=p.id, action="update", request=request)
    db.commit()
    return _pool_out(db, p)


@pools.delete("/{pool_id}", status_code=204)
def delete_pool(
    pool_id: str, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pool(db, pool_id, current.tenant_id)
    write_audit(db, actor=current, entity_type="talent_pool",
                entity_id=p.id, action="delete", request=request)
    db.delete(p)
    db.commit()


class MemberOut(BaseModel):
    employee_id: str
    name: str
    designation: str | None = None
    org_name: str | None = None
    stage: str
    readiness: int | None = None
    note: str | None = None
    perf_band: int | None = None
    potential_band: int | None = None
    created_at: str = ""


class PoolDetailOut(PoolOut):
    members: list[MemberOut] = Field(default_factory=list)


@pools.get("/{pool_id}", response_model=PoolDetailOut)
def pool_detail(
    pool_id: str, cycle: str | None = Query(default=None),
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_view(db, current)
    p = _pool(db, pool_id, current.tenant_id)
    cyc = _cycle(cycle)
    mems = db.scalars(
        select(TalentPoolMember)
        .where(TalentPoolMember.pool_id == p.id)
        .order_by(TalentPoolMember.created_at)
    ).all()
    emp_ids = [m.employee_id for m in mems]
    emps = {
        e.id: e for e in db.scalars(
            select(Employee).where(Employee.id.in_(emp_ids or ["-"]))
        ).all()
    }
    org_names = {
        o.id: o.name for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == current.tenant_id)
        ).all()
    }
    bands = {
        a.employee_id: a for a in db.scalars(
            select(NineBoxAssessment).where(
                NineBoxAssessment.tenant_id == current.tenant_id,
                NineBoxAssessment.cycle == cyc,
            )
        ).all()
    }
    members = [
        MemberOut(
            employee_id=m.employee_id,
            name=emps[m.employee_id].name if m.employee_id in emps else "?",
            designation=emps[m.employee_id].designation if m.employee_id in emps else None,
            org_name=org_names.get(emps[m.employee_id].org_unit_id) if m.employee_id in emps else None,
            stage=m.stage, readiness=m.readiness, note=m.note,
            perf_band=bands[m.employee_id].perf_band if m.employee_id in bands else None,
            potential_band=bands[m.employee_id].potential_band if m.employee_id in bands else None,
            created_at=str(m.created_at or "")[:16],
        )
        for m in mems if m.employee_id in emps
    ]
    return PoolDetailOut(**_pool_out(db, p).model_dump(), members=members)


class AddMembersIn(BaseModel):
    employee_ids: list[str] = Field(min_length=1, max_length=100)
    readiness: int | None = Field(default=None, ge=1, le=3)
    note: str | None = Field(default=None, max_length=500)


@pools.post("/{pool_id}/members", status_code=201)
def add_members(
    pool_id: str, body: AddMembersIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pool(db, pool_id, current.tenant_id)
    scope = visible_employee_ids(db, current)
    existing = {
        m.employee_id for m in db.scalars(
            select(TalentPoolMember).where(TalentPoolMember.pool_id == p.id)
        ).all()
    }
    added = 0
    for eid in dict.fromkeys(body.employee_ids):
        emp = db.get(Employee, eid)
        if not emp or emp.tenant_id != current.tenant_id:
            continue
        if scope is not None and eid not in scope:
            continue
        if eid in existing:
            continue
        db.add(TalentPoolMember(
            tenant_id=current.tenant_id, pool_id=p.id, employee_id=eid,
            readiness=body.readiness, note=body.note, added_by=current.id,
        ))
        added += 1
    write_audit(db, actor=current, entity_type="talent_pool",
                entity_id=p.id, action="add_members",
                detail={"count": added}, request=request)
    db.commit()
    return {"added": added, **_pool_out(db, p).model_dump()}


class MemberPatchIn(BaseModel):
    stage: str | None = None
    readiness: int | None = Field(default=None, ge=1, le=3)
    note: str | None = Field(default=None, max_length=500)


@pools.patch("/{pool_id}/members/{emp_id}", response_model=MemberOut)
def patch_member(
    pool_id: str, emp_id: str, body: MemberPatchIn, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pool(db, pool_id, current.tenant_id)
    m = db.scalar(
        select(TalentPoolMember).where(
            TalentPoolMember.pool_id == p.id,
            TalentPoolMember.employee_id == emp_id,
        )
    )
    if not m:
        raise HTTPException(404, "成员不存在")
    if body.stage is not None:
        if body.stage not in STAGES:
            raise HTTPException(400, "无效阶段")
        m.stage = body.stage
        m.exited_at = datetime.now() if body.stage == "exited" else None
    if body.readiness is not None:
        m.readiness = body.readiness
    if body.note is not None:
        m.note = body.note
    write_audit(db, actor=current, entity_type="talent_pool_member",
                entity_id=m.id, action="update",
                detail={"stage": m.stage}, request=request)
    db.commit()
    emp = db.get(Employee, emp_id)
    return MemberOut(
        employee_id=emp_id, name=emp.name if emp else "?",
        designation=emp.designation if emp else None,
        stage=m.stage, readiness=m.readiness, note=m.note,
        created_at=str(m.created_at or "")[:16],
    )


@pools.delete("/{pool_id}/members/{emp_id}", status_code=204)
def remove_member(
    pool_id: str, emp_id: str, request: Request,
    db: Session = Depends(get_db), current: User = Depends(get_current_user),
):
    _check_write(current)
    p = _pool(db, pool_id, current.tenant_id)
    m = db.scalar(
        select(TalentPoolMember).where(
            TalentPoolMember.pool_id == p.id,
            TalentPoolMember.employee_id == emp_id,
        )
    )
    if not m:
        raise HTTPException(404, "成员不存在")
    write_audit(db, actor=current, entity_type="talent_pool_member",
                entity_id=m.id, action="remove", request=request)
    db.delete(m)
    db.commit()
