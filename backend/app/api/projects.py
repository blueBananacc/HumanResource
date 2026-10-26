"""P 批次 — 轻量项目 API(人才归属载体)。

- ``GET/POST /projects``                列表 / 创建
- ``GET/PUT /projects/{id}``            详情 / 更新(status→done 触发成员画像建议)
- ``DELETE /projects/{id}``             删除
- ``POST /projects/{id}/members``       加成员(joined_via: manual/ai_match)
- ``DELETE /projects/{id}/members/{emp_id}``  移除成员
- ``PUT /projects/{id}/members/{emp_id}``     角色/投入度/贡献总结

可见:全部登录用户可看项目;写操作需 manage_employees 或项目 owner。
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import Employee, Project, ProjectMember, User
from app.infra.db import get_db
from app.services import profile_update
from app.services.audit import write_audit
from app.services.permissions import get_linked_employee_id, has_permission

router = APIRouter(
    prefix="/projects", tags=["projects"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _out(p: Project, db: Session) -> dict[str, Any]:
    owner = db.get(Employee, p.owner_employee_id) if p.owner_employee_id else None
    cnt = db.execute(
        select(ProjectMember.id).where(
            ProjectMember.project_id == p.id, ProjectMember.left_at.is_(None)
        )
    ).all()
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "status": p.status, "owner_employee_id": p.owner_employee_id,
        "owner_name": owner.name if owner else None,
        "start_date": str(p.start_date or ""), "end_date": str(p.end_date or ""),
        "required_skills": p.required_skills or [],
        "member_count": len(cnt), "created_at": str(p.created_at or ""),
    }


def _get(db: Session, pid: str, current: User) -> Project:
    p = db.get(Project, pid)
    if not p or p.tenant_id != current.tenant_id:
        raise HTTPException(404, "项目不存在")
    return p


def _can_write(db: Session, p: Project | None, current: User) -> bool:
    if has_permission(current, "manage_employees"):
        return True
    own = get_linked_employee_id(db, current)
    return bool(p and own and p.owner_employee_id == own)


@router.get("")
def list_projects(
    status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    stmt = select(Project).where(Project.tenant_id == current.tenant_id)
    if status:
        stmt = stmt.where(Project.status == status)
    if q:
        stmt = stmt.where(Project.name.ilike(f"%{q}%"))
    rows = db.scalars(stmt.order_by(Project.created_at.desc()).limit(100)).all()
    return {"items": [_out(p, db) for p in rows]}


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    status: Literal["draft", "active", "done", "cancelled"] = "draft"
    owner_employee_id: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    required_skills: list[dict[str, Any]] = Field(default_factory=list)


@router.post("", status_code=201)
def create_project(
    payload: ProjectIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if not has_permission(current, "manage_employees"):
        # 主管也可建项目(自动成为 owner)
        if get_linked_employee_id(db, current) is None:
            raise HTTPException(403, "权限不足")
    p = Project(
        tenant_id=current.tenant_id, name=payload.name,
        description=payload.description, status=payload.status,
        owner_employee_id=payload.owner_employee_id
        or get_linked_employee_id(db, current),
        start_date=payload.start_date, end_date=payload.end_date,
        required_skills=payload.required_skills, created_by=current.id,
    )
    db.add(p)
    db.flush()
    write_audit(db, actor=current, entity_type="project",
                entity_id=p.id, action="create", request=request)
    db.commit()
    return _out(p, db)


@router.get("/{pid}")
def get_project(
    pid: str, db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    p = _get(db, pid, current)
    members = db.execute(
        select(ProjectMember, Employee)
        .join(Employee, Employee.id == ProjectMember.employee_id)
        .where(ProjectMember.project_id == p.id)
    ).all()
    out = _out(p, db)
    out["members"] = [{
        "employee_id": e.id, "name": e.name, "designation": e.designation,
        "role": m.role, "allocation_pct": m.allocation_pct,
        "joined_via": m.joined_via, "contribution": m.contribution_summary,
        "joined_at": str(m.joined_at or "")[:10],
        "left_at": str(m.left_at or "")[:10] if m.left_at else None,
    } for m, e in members]
    return out


@router.put("/{pid}")
def update_project(
    pid: str, payload: ProjectIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    p = _get(db, pid, current)
    if not _can_write(db, p, current):
        raise HTTPException(403, "权限不足")
    was_done = p.status == "done"
    for f in ("name", "description", "status", "owner_employee_id",
              "start_date", "end_date", "required_skills"):
        setattr(p, f, getattr(payload, f))
    suggested = 0
    if payload.status == "done" and not was_done:
        suggested = profile_update.suggest_from_project_done(db, p)
    write_audit(db, actor=current, entity_type="project",
                entity_id=p.id, action="update", request=request)
    db.commit()
    return {**_out(p, db), "skills_suggested": suggested}


@router.delete("/{pid}")
def delete_project(
    pid: str, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    p = _get(db, pid, current)
    if not _can_write(db, p, current):
        raise HTTPException(403, "权限不足")
    write_audit(db, actor=current, entity_type="project",
                entity_id=p.id, action="delete", request=request)
    db.delete(p)
    db.commit()
    return {"ok": True}


class MemberIn(BaseModel):
    employee_id: str
    role: str | None = Field(default=None, max_length=64)
    allocation_pct: int = Field(default=100, ge=1, le=100)
    joined_via: Literal["manual", "ai_match"] = "manual"


@router.post("/{pid}/members", status_code=201)
def add_member(
    pid: str, payload: MemberIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    p = _get(db, pid, current)
    if not _can_write(db, p, current):
        raise HTTPException(403, "权限不足")
    emp = db.get(Employee, payload.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    dup = db.scalars(
        select(ProjectMember).where(
            ProjectMember.project_id == p.id,
            ProjectMember.employee_id == emp.id,
            ProjectMember.left_at.is_(None),
        )
    ).first()
    if dup:
        raise HTTPException(409, "该员工已在项目中")
    m = ProjectMember(
        tenant_id=current.tenant_id, project_id=p.id, employee_id=emp.id,
        role=payload.role, allocation_pct=payload.allocation_pct,
        joined_via=payload.joined_via,
    )
    db.add(m)
    db.flush()
    write_audit(db, actor=current, entity_type="project_member",
                entity_id=m.id, action="add", request=request)
    db.commit()
    return {"id": m.id, "employee_id": emp.id, "name": emp.name}


class MemberPatch(BaseModel):
    role: str | None = None
    allocation_pct: int | None = Field(default=None, ge=1, le=100)
    contribution_summary: str | None = None
    leave: bool = False


@router.put("/{pid}/members/{emp_id}")
def patch_member(
    pid: str, emp_id: str, payload: MemberPatch, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    p = _get(db, pid, current)
    if not _can_write(db, p, current):
        raise HTTPException(403, "权限不足")
    m = db.scalars(
        select(ProjectMember).where(
            ProjectMember.project_id == p.id,
            ProjectMember.employee_id == emp_id,
            ProjectMember.left_at.is_(None),
        )
    ).first()
    if not m:
        raise HTTPException(404, "成员不存在")
    if payload.leave:
        m.left_at = datetime.now()
    else:
        if payload.role is not None:
            m.role = payload.role
        if payload.allocation_pct is not None:
            m.allocation_pct = payload.allocation_pct
        if payload.contribution_summary is not None:
            m.contribution_summary = payload.contribution_summary
    write_audit(db, actor=current, entity_type="project_member",
                entity_id=m.id, action="update", request=request)
    db.commit()
    return {"ok": True}
