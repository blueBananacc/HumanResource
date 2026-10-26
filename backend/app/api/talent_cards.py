"""P 批次 — 人才画像卡 API。

- ``GET  /employees/{id}/talent-card``            聚合卡(本人视图自动脱敏)
- ``POST /employees/{id}/skills``                 手动加/改技能(直接 confirmed)
- ``POST /employees/{id}/skill-suggestions/confirm`` 批量确认 AI 建议
- ``POST /employees/{id}/skill-suggestions/reject``  驳回(删行)
- ``GET/POST /employees/{id}/insights``           洞察(会议纪要/复盘/随手评)
- ``GET  /me/talent-card``                        ESS 本人卡(脱敏版)

可见性:suggested 技能与 manager_only 洞察对「本人」隐藏。
写操作权限:manage_employees 或该员工直属上级。
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    EmployeeSkill,
    InsightNote,
    OrgUnit,
    ProjectMember,
    Skill,
    User,
)
from app.infra.db import get_db
from app.services import profile_update, talent_card
from app.services.audit import write_audit
from app.services.permissions import (
    get_linked_employee_id, has_permission, require_linked_employee,
    visible_employee_ids,
)
from fastapi import Request

router = APIRouter(tags=["talent-card"], dependencies=[Depends(require_feature("hr.suite"))])


def _get_employee(db: Session, emp_id: str, current: User) -> Employee:
    emp = db.get(Employee, emp_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    scope = visible_employee_ids(db, current)
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在或无权访问")
    return emp


def _can_curate(db: Session, emp: Employee, current: User) -> bool:
    if has_permission(current, "manage_employees"):
        return True
    own = get_linked_employee_id(db, current)
    return bool(own and emp.reports_to_id == own)


@router.get("/employees/{emp_id}/talent-card")
def get_card(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    return talent_card.build_card(
        db, emp, viewer_employee_id=get_linked_employee_id(db, current)
    )


@router.get("/me/talent-card")
def my_card(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp_id = require_linked_employee(db, current)
    emp = db.get(Employee, emp_id)
    return talent_card.build_card(db, emp, viewer_employee_id=emp_id)


class SkillIn(BaseModel):
    name: str = Field(max_length=64)
    level: int = Field(default=3, ge=1, le=5)
    note: str | None = Field(default=None, max_length=200)


@router.post("/employees/{emp_id}/skills", status_code=201)
def add_skill(
    emp_id: str, payload: SkillIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    skill = profile_update.get_or_create_skill(
        db, tenant_id=current.tenant_id, name=payload.name
    )
    row = profile_update.upsert_employee_skill(
        db, tenant_id=current.tenant_id, employee_id=emp.id,
        skill_id=skill.id, level=payload.level, source="manual",
        status="confirmed", note=payload.note, created_by=current.id,
    )
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=row.id, action="upsert", request=request)
    db.commit()
    return {"id": row.id, "skill_id": skill.id, "name": skill.name,
            "level": row.level, "status": row.status}


class SuggestActionIn(BaseModel):
    skill_ids: list[str] | None = None  # None = 全部


def _pending(db: Session, emp_id: str, skill_ids: list[str] | None):
    q = select(EmployeeSkill).where(
        EmployeeSkill.employee_id == emp_id,
        EmployeeSkill.status == "suggested",
    )
    if skill_ids:
        q = q.where(EmployeeSkill.skill_id.in_(skill_ids))
    return db.scalars(q).all()


@router.post("/employees/{emp_id}/skill-suggestions/confirm")
def confirm_suggestions(
    emp_id: str, payload: SuggestActionIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    rows = _pending(db, emp.id, payload.skill_ids)
    for r in rows:
        r.status = "confirmed"
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=emp.id, action=f"confirm_x{len(rows)}", request=request)
    db.commit()
    return {"confirmed": len(rows)}


@router.post("/employees/{emp_id}/skill-suggestions/reject")
def reject_suggestions(
    emp_id: str, payload: SuggestActionIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    rows = _pending(db, emp.id, payload.skill_ids)
    for r in rows:
        db.delete(r)
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=emp.id, action=f"reject_x{len(rows)}", request=request)
    db.commit()
    return {"rejected": len(rows)}


# ============ 洞察(会议纪要手动上传入口在此) ============


class InsightIn(BaseModel):
    kind: Literal["meeting", "project_review", "manager_note", "milestone"] = "manager_note"
    title: str | None = Field(default=None, max_length=200)
    body: str = Field(min_length=1, max_length=8000)
    visibility: Literal["manager_only", "employee_visible"] = "manager_only"
    source_ref: dict[str, Any] | None = None
    extract_skills: bool = True  # 是否触发画像建议管线


@router.get("/employees/{emp_id}/insights")
def list_insights(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    q = select(InsightNote).where(InsightNote.employee_id == emp.id)
    if get_linked_employee_id(db, current) == emp.id:
        q = q.where(InsightNote.visibility == "employee_visible")
    rows = db.scalars(q.order_by(InsightNote.created_at.desc()).limit(50)).all()
    return {"items": [{
        "id": n.id, "kind": n.kind, "title": n.title, "body": n.body,
        "visibility": n.visibility, "source_ref": n.source_ref,
        "created_at": str(n.created_at or ""),
    } for n in rows]}


@router.post("/employees/{emp_id}/insights", status_code=201)
def add_insight(
    emp_id: str, payload: InsightIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    note = InsightNote(
        tenant_id=current.tenant_id, employee_id=emp.id, kind=payload.kind,
        title=payload.title, body=payload.body,
        author_user_id=current.id, source_ref=payload.source_ref,
        visibility=payload.visibility,
    )
    db.add(note)
    db.flush()
    write_audit(db, actor=current, entity_type="insight_note",
                entity_id=note.id, action="create", request=request)
    suggestions = {"suggested": [], "skipped": []}
    if payload.extract_skills:
        suggestions = profile_update.suggest_from_text(
            db, tenant_id=current.tenant_id, employee=emp,
            text=f"{payload.title or ''}\n{payload.body}",
            source="insight",
            evidence={"kind": "insight", "id": note.id},
            created_by=current.id,
        )
    db.commit()
    return {"id": note.id, **suggestions}


# ============ AI 选人:确定性技能预筛(LLM 在结果上做 topK/理由) ============


class TalentSearchIn(BaseModel):
    """skills: [{"name": "TypeScript", "min_level": 3}];

    name 对技能字典做模糊匹配 —— 「AI coding 强」由 LLM 先拆成
    技能名片段(如 ["AI", "Python"])再传进来,这里只做确定性过滤。
    """

    skills: list[dict[str, Any]] = Field(default_factory=list)
    include_suggested: bool = False  # 待确认标签默认不计入匹配
    org_unit_id: str | None = None
    only_active: bool = True
    limit: int = Field(default=20, ge=1, le=50)


@router.post("/talent/search")
def search_talent(
    payload: TalentSearchIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    reqs: list[tuple[str, int]] = []
    for s in payload.skills:
        frag = str(s.get("name") or "").strip()[:64]
        if frag:
            reqs.append((frag, min(5, max(1, int(s.get("min_level") or 1)))))
    if not reqs:
        raise HTTPException(422, "skills 不能为空 —— 至少给一个技能名片段")

    # 1) 片段 → 字典技能 id(每段命中的技能集合)
    dict_rows = db.scalars(
        select(Skill).where(
            Skill.tenant_id == current.tenant_id,
            or_(*[Skill.name.ilike(f"%{n}%") for n, _ in reqs]),
        )
    ).all()
    frag_map: list[tuple[str, int, set[str]]] = []  # (片段, min_level, skill_ids)
    for frag, lv in reqs:
        ids = {s.id for s in dict_rows if frag.lower() in s.name.lower()}
        frag_map.append((frag, lv, ids))
    all_ids = {sid for _, _, ids in frag_map for sid in ids}
    if not all_ids:
        return {"items": [], "matched": {"requested": [n for n, _ in reqs], "dict_hits": []}}

    # 2) 员工技能行过滤(状态/等级/范围)
    statuses = ["confirmed"] + (["suggested"] if payload.include_suggested else [])
    rows = db.execute(
        select(EmployeeSkill, Skill.name)
        .join(Skill, Skill.id == EmployeeSkill.skill_id)
        .where(
            EmployeeSkill.tenant_id == current.tenant_id,
            EmployeeSkill.skill_id.in_(all_ids),
            EmployeeSkill.status.in_(statuses),
        )
    ).all()

    # 3) 按员工聚合命中片段数
    scope = visible_employee_ids(db, current)
    hit: dict[str, dict] = {}
    for es, sname in rows:
        if scope is not None and es.employee_id not in scope:
            continue
        entry = hit.setdefault(es.employee_id, {"frags": set(), "skills": []})
        for frag, lv, ids in frag_map:
            if es.skill_id in ids and es.level >= lv:
                entry["frags"].add(frag)
        entry["skills"].append({
            "name": sname, "level": es.level, "source": es.source,
            "status": es.status,
        })
    if not hit:
        return {"items": [], "matched": {"requested": [n for n, _ in reqs],
                                         "dict_hits": [s.name for s in dict_rows]}}

    emp_ids = [eid for eid, e in hit.items() if e["frags"]]
    q = select(Employee).where(
        Employee.tenant_id == current.tenant_id, Employee.id.in_(emp_ids)
    )
    if payload.only_active:
        q = q.where(Employee.status == "active")
    if payload.org_unit_id:
        q = q.where(Employee.org_unit_id == payload.org_unit_id)
    emps = {e.id: e for e in db.scalars(q).all()}

    org_names = dict(
        db.execute(select(OrgUnit.id, OrgUnit.name).where(
            OrgUnit.tenant_id == current.tenant_id)).all()
    )
    proj_counts = dict(
        db.execute(
            select(ProjectMember.employee_id, func.count())
            .where(
                ProjectMember.tenant_id == current.tenant_id,
                ProjectMember.employee_id.in_(list(emps)),
                ProjectMember.left_at.is_(None),
            )
            .group_by(ProjectMember.employee_id)
        ).all()
    )

    items = []
    for eid, e in emps.items():
        h = hit[eid]
        skills = sorted(h["skills"], key=lambda x: -x["level"])
        items.append({
            "employee_id": eid,
            "name": e.name,
            "designation": e.designation,
            "org_unit": org_names.get(e.org_unit_id or ""),
            "matched_fragments": sorted(h["frags"]),
            "match_count": len(h["frags"]),
            "matched_skills": skills,
            "active_projects": proj_counts.get(eid, 0),
        })
    items.sort(key=lambda x: (-x["match_count"],
                              -sum(s["level"] for s in x["matched_skills"])))
    return {
        "items": items[: payload.limit],
        "matched": {
            "requested": [n for n, _ in reqs],
            "dict_hits": [s.name for s in dict_rows],
        },
    }
