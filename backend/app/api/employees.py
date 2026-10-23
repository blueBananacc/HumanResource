"""员工主数据 API(HR 管理端)。

- ``GET    /employees``                 员工列表(分页/搜索/部门/状态过滤)
- ``POST   /employees``                 创建员工(manage_employees)
- ``GET    /employees/{id}``            员工详情(聚合:资料+假期余额+最近考勤)
- ``PUT    /employees/{id}``            更新员工
- ``DELETE /employees/{id}``            删除员工(有业务数据则拒绝)
- ``GET    /employees/org-chart``       组织架构图数据(按 reports_to 展开)
- ``POST   /employees/{id}/account``    绑定/创建员工账号(role=employee)
- ``DELETE /employees/{id}/account``    解绑员工账号

设计:员工账号是 User(role="employee") 挂在 employee.user_id 上;
ESS 端点在 ``hr_me.py``(/api/me/*),行级过滤到本人。
"""
from __future__ import annotations

import re
import secrets
from datetime import UTC, date, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, apply_q_ilike, count_total, paginate_params
from app.api.license import require_feature
from app.domain.models import Employee, OrgUnit, User
from app.infra.db import get_db
from app.infra.security import hash_password
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    require_permission,
)

router = APIRouter(prefix="/employees", tags=["hris"], dependencies=[Depends(require_feature("hr.suite"))])

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _v_email(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip().lower()
    if not v:
        return None
    if not _EMAIL_RE.match(v):
        raise ValueError("邮箱格式不合法")
    return v


# ============ schemas ============


class EmployeeOut(BaseModel):
    id: str
    employee_no: str
    name: str
    org_unit_id: str | None
    org_unit_name: str | None
    designation: str | None
    reports_to_id: str | None
    reports_to_name: str | None
    user_id: str | None
    work_email: str | None
    phone: str | None
    hire_date: date | None
    employment_type: str
    status: str
    personal: dict[str, Any] | None
    source_candidate_id: str | None
    has_account: bool
    created_at: datetime
    updated_at: datetime


class EmployeeIn(BaseModel):
    employee_no: str | None = None
    name: str
    org_unit_id: str | None = None
    designation: str | None = None
    reports_to_id: str | None = None
    work_email: str | None = None
    phone: str | None = None
    hire_date: date | None = None
    employment_type: Literal[
        "full_time", "part_time", "intern", "contractor"
    ] = "full_time"
    status: Literal["active", "leave", "suspended", "terminated"] = "active"
    personal: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("姓名不能为空")
        return v

    @field_validator("employee_no")
    @classmethod
    def _v_no(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("work_email")
    @classmethod
    def _v_work_email(cls, v: str | None) -> str | None:
        return _v_email(v)


class AccountIn(BaseModel):
    email: str
    password: str | None = Field(None, min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _v_account_email(cls, v: str) -> str:
        v2 = _v_email(v)
        if not v2:
            raise ValueError("邮箱格式不合法")
        return v2


class AccountOut(BaseModel):
    user_id: str
    email: str
    password: str | None = None  # 仅创建时返回一次


class OrgChartNode(BaseModel):
    id: str
    name: str
    designation: str | None
    org_unit_name: str | None
    status: str
    children: list[OrgChartNode] = Field(default_factory=list)


# ============ helpers ============


def _emp(db: Session, emp_id: str, tenant_id: str) -> Employee:
    e = db.get(Employee, emp_id)
    if not e or e.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    return e


def _name_maps(db: Session, tenant_id: str):
    orgs = {
        o.id: o.name
        for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
        )
    }
    emps = {
        e.id: e.name
        for e in db.scalars(
            select(Employee).where(Employee.tenant_id == tenant_id)
        )
    }
    return orgs, emps


def _serialize(e: Employee, orgs: dict, emps: dict) -> EmployeeOut:
    return EmployeeOut(
        id=e.id,
        employee_no=e.employee_no,
        name=e.name,
        org_unit_id=e.org_unit_id,
        org_unit_name=orgs.get(e.org_unit_id) if e.org_unit_id else None,
        designation=e.designation,
        reports_to_id=e.reports_to_id,
        reports_to_name=emps.get(e.reports_to_id) if e.reports_to_id else None,
        user_id=e.user_id,
        work_email=e.work_email,
        phone=e.phone,
        hire_date=e.hire_date,
        employment_type=e.employment_type,
        status=e.status,
        personal=e.personal,
        source_candidate_id=e.source_candidate_id,
        has_account=e.user_id is not None,
        created_at=e.created_at,
        updated_at=e.updated_at,
    )


def next_employee_no(db: Session, tenant_id: str) -> str:
    """取下一个可用工号 — 按现有 EMP#### 最大序号 +1,删人留洞也不撞号。"""
    nos = db.scalars(
        select(Employee.employee_no).where(Employee.tenant_id == tenant_id)
    ).all()
    n = max(
        (int(no[3:]) for no in nos if re.fullmatch(r"EMP\d{4,}", no)),
        default=0,
    )
    return f"EMP{n + 1:04d}"


def _has_dependents(db: Session, emp_id: str, tenant_id: str) -> str | None:
    """返回阻碍删除的第一个依赖名,无依赖返回 None。"""
    from app.domain.models import (
        Appraisal,
        Attendance,
        EmployeeCheckin,
        ExpenseClaim,
        Goal,
        LeaveAllocation,
        LeaveApplication,
        LeaveLedgerEntry,
        LifecycleEvent,
        SalarySlip,
    )

    checks = [
        (LeaveApplication, "请假申请"),
        (LeaveAllocation, "假期额度"),
        (LeaveLedgerEntry, "假期台账"),
        (Attendance, "考勤记录"),
        (EmployeeCheckin, "打卡记录"),
        (LifecycleEvent, "生命周期事件"),
        (Goal, "目标"),
        (Appraisal, "考核表"),
        (ExpenseClaim, "报销单"),
        (SalarySlip, "工资条"),
    ]
    for model, label in checks:
        if db.scalars(
            select(model.id)
            .where(model.tenant_id == tenant_id, model.employee_id == emp_id)
            .limit(1)
        ).first():
            return label
    return None


# ============ endpoints ============


@router.get("", response_model=PageOut[EmployeeOut])
def list_employees(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    org_unit_id: str | None = None,
    unlinked: bool = Query(False, description="仅看未绑定账号"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    limit, offset, q = pagination
    stmt = select(Employee).where(Employee.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(Employee.status == status_filter)
    if org_unit_id:
        stmt = stmt.where(Employee.org_unit_id == org_unit_id)
    if unlinked:
        stmt = stmt.where(Employee.user_id.is_(None))
    stmt = apply_q_ilike(stmt, q, Employee.name, Employee.employee_no, Employee.work_email, Employee.designation)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(Employee.created_at.desc()).limit(limit).offset(offset)
    ).all()
    orgs, emps = _name_maps(db, current.tenant_id)
    return PageOut(
        items=[_serialize(e, orgs, emps) for e in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=EmployeeOut, status_code=status.HTTP_201_CREATED)
def create_employee(
    body: EmployeeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    if body.org_unit_id:
        org = db.get(OrgUnit, body.org_unit_id)
        if not org or org.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "部门不存在")
    if body.reports_to_id:
        mgr = _emp(db, body.reports_to_id, current.tenant_id)
        _ = mgr
    e = Employee(
        tenant_id=current.tenant_id,
        employee_no=body.employee_no or next_employee_no(db, current.tenant_id),
        name=body.name,
        org_unit_id=body.org_unit_id,
        designation=body.designation,
        reports_to_id=body.reports_to_id,
        work_email=body.work_email,
        phone=body.phone,
        hire_date=body.hire_date,
        employment_type=body.employment_type,
        status=body.status,
        personal=body.personal,
    )
    db.add(e)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "工号已存在")
    db.refresh(e)
    write_audit(
        db, actor=current, entity_type="employee", entity_id=e.id,
        action="create", detail={"name": e.name, "employee_no": e.employee_no},
        request=request,
    )
    db.commit()
    orgs, emps = _name_maps(db, current.tenant_id)
    return _serialize(e, orgs, emps)


@router.get("/org-chart", response_model=list[OrgChartNode])
def org_chart(
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    rows = db.scalars(
        select(Employee).where(
            Employee.tenant_id == current.tenant_id,
            Employee.status != "terminated",
        )
    ).all()
    orgs, _ = _name_maps(db, current.tenant_id)
    nodes = {
        e.id: OrgChartNode(
            id=e.id, name=e.name, designation=e.designation,
            org_unit_name=orgs.get(e.org_unit_id) if e.org_unit_id else None,
            status=e.status, children=[],
        )
        for e in rows
    }
    roots: list[OrgChartNode] = []
    for e in rows:
        node = nodes[e.id]
        if e.reports_to_id and e.reports_to_id in nodes:
            nodes[e.reports_to_id].children.append(node)
        else:
            roots.append(node)
    return roots


@router.get("/{emp_id}", response_model=EmployeeOut)
def get_employee(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    e = _emp(db, emp_id, current.tenant_id)
    orgs, emps = _name_maps(db, current.tenant_id)
    return _serialize(e, orgs, emps)


@router.put("/{emp_id}", response_model=EmployeeOut)
def update_employee(
    emp_id: str,
    body: EmployeeIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    e = _emp(db, emp_id, current.tenant_id)
    if body.reports_to_id == e.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "上级不能是自己")
    # 防环:新上级不能挂到自己的子孙下面
    if body.reports_to_id:
        cursor = _emp(db, body.reports_to_id, current.tenant_id)
        while cursor.reports_to_id:
            if cursor.reports_to_id == e.id:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "不能把员工挂到自己的下级下面"
                )
            cursor = _emp(db, cursor.reports_to_id, current.tenant_id)
    if body.org_unit_id:
        org = db.get(OrgUnit, body.org_unit_id)
        if not org or org.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "部门不存在")
    before = {"name": e.name, "org_unit_id": e.org_unit_id, "status": e.status}
    for f in (
        "name", "org_unit_id", "designation", "reports_to_id", "phone",
        "hire_date", "employment_type", "status",
    ):
        setattr(e, f, getattr(body, f))
    # personal 只在显式传值时更新 — 前端编辑表单不带这个字段,
    # 全量覆盖会把员工在自助端填的住址/紧急联系人清空。
    if "personal" in body.model_fields_set:
        e.personal = body.personal
    e.work_email = body.work_email
    if body.employee_no:
        e.employee_no = body.employee_no
    e.updated_at = datetime.now(UTC).replace(tzinfo=None)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "工号已存在")
    write_audit(
        db, actor=current, entity_type="employee", entity_id=e.id,
        action="update",
        detail={"before": before, "after": {"name": e.name, "status": e.status}},
        request=request,
    )
    db.commit()
    db.refresh(e)
    orgs, emps = _name_maps(db, current.tenant_id)
    return _serialize(e, orgs, emps)


@router.delete("/{emp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee(
    emp_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    e = _emp(db, emp_id, current.tenant_id)
    blocker = _has_dependents(db, e.id, current.tenant_id)
    if blocker:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"该员工已有{blocker}数据,不能删除;可改为 status=terminated",
        )
    # 清理绑定账号(仅删除专为 ESS 创建的 employee 账号,复用的 HR 账号保留只解绑)
    if e.user_id:
        u = db.get(User, e.user_id)
        if u and u.role == "employee":
            db.delete(u)
        e.user_id = None
    # 清理下级的 reports_to
    for sub in db.scalars(
        select(Employee).where(Employee.reports_to_id == e.id)
    ):
        sub.reports_to_id = None
    write_audit(
        db, actor=current, entity_type="employee", entity_id=e.id,
        action="delete", detail={"name": e.name}, request=request,
    )
    db.delete(e)
    db.commit()


@router.post("/{emp_id}/account", response_model=AccountOut)
def bind_account(
    emp_id: str,
    body: AccountIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """给员工绑定登录账号。

    - 邮箱已有用户 → 直接挂上(保留其原有角色:hiring_manager/hr 等也能绑
      员工档案,同时具备管理端权限和 /me/* 自助入口 — 上级审批下属就靠这个)
    - 邮箱不存在 → 新建 role=employee 的自助账号,返回一次性密码
    """
    e = _emp(db, emp_id, current.tenant_id)
    email = body.email
    existing_user = db.scalars(select(User).where(User.email == email)).first()
    password_out: str | None = None
    if existing_user:
        if existing_user.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "邮箱已被其他租户占用")
        taken = db.scalars(
            select(Employee.id).where(
                Employee.user_id == existing_user.id, Employee.id != e.id
            )
        ).first()
        if taken:
            raise HTTPException(status.HTTP_409_CONFLICT, "该账号已绑定其他员工")
        # 之前解绑时被停用的 ESS 账号,重新绑定时恢复登录
        if existing_user.role == "employee" and existing_user.status == "disabled":
            existing_user.status = "active"
        user = existing_user
    else:
        password_out = body.password or secrets.token_urlsafe(9)
        user = User(
            tenant_id=current.tenant_id,
            email=email,
            password_hash=hash_password(password_out),
            role="employee",
            org_unit_id=e.org_unit_id,
        )
        db.add(user)
        db.flush()
    e.user_id = user.id
    db.commit()
    write_audit(
        db, actor=current, entity_type="employee", entity_id=e.id,
        action="bind_account", detail={"user_id": user.id, "email": email},
        request=request,
    )
    db.commit()
    return AccountOut(user_id=user.id, email=email, password=password_out)


@router.delete("/{emp_id}/account", status_code=status.HTTP_204_NO_CONTENT)
def unbind_account(
    emp_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    e = _emp(db, emp_id, current.tenant_id)
    if e.user_id:
        bound = db.get(User, e.user_id)
        write_audit(
            db, actor=current, entity_type="employee", entity_id=e.id,
            action="unbind_account", detail={"user_id": e.user_id},
            request=request,
        )
        e.user_id = None
        # ESS 账号解绑后没有任何可达数据,停用防止裸账号继续登录
        if bound is not None and bound.role == "employee":
            bound.status = "disabled"
        db.commit()


OrgChartNode.model_rebuild()
