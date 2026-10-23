"""员工生命周期 API:入职 / 调动 / 晋升 / 离职清单。

- ``GET/POST/PUT/DELETE /lifecycle/templates``   任务模板
- ``GET/POST /lifecycle/events``                 事件列表 / 发起(按模板铺任务)
- ``GET  /lifecycle/events/{id}``                事件详情(含任务)
- ``POST /lifecycle/events/{id}/cancel``         取消
- ``POST /lifecycle/tasks/{id}/toggle``          勾选任务;全勾→事件 completed

发起事件时若 kind=transfer/promotion,完成后回写 employee 的
org_unit_id / designation(payload 里携带 to_* 字段)。
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.domain.models import (
    Employee,
    LifecycleEvent,
    LifecycleTask,
    LifecycleTemplate,
    OrgUnit,
    User,
)
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_MANAGE_LIFECYCLE,
    PERM_VIEW_REPORTS,
    require_permission,
)

router = APIRouter(prefix="/lifecycle", tags=["hris-lifecycle"], dependencies=[Depends(require_feature("hr.suite"))])

_EVENT_KINDS = ("onboarding", "transfer", "promotion", "separation")


class TemplateIn(BaseModel):
    kind: Literal["onboarding", "transfer", "promotion", "separation"]
    name: str
    tasks: list[str] = Field(default_factory=list)
    is_default: bool = False


class TemplateOut(BaseModel):
    id: str
    kind: str
    name: str
    tasks: list[str]
    is_default: bool


class EventIn(BaseModel):
    employee_id: str
    kind: Literal["onboarding", "transfer", "promotion", "separation"]
    effective_date: date | None = None
    template_id: str | None = None
    payload: dict | None = None
    note: str | None = None


class TaskOut(BaseModel):
    id: str
    title: str
    done: bool
    done_at: datetime | None
    sort: int


class EventOut(BaseModel):
    id: str
    employee_id: str
    employee_name: str
    kind: str
    status: str
    effective_date: date | None
    payload: dict | None
    note: str | None
    tasks: list[TaskOut]
    created_at: datetime
    completed_at: datetime | None


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _event_out(ev: LifecycleEvent, emp_name: str, tasks: list[LifecycleTask]) -> EventOut:
    return EventOut(
        id=ev.id, employee_id=ev.employee_id, employee_name=emp_name,
        kind=ev.kind, status=ev.status, effective_date=ev.effective_date,
        payload=ev.payload, note=ev.note,
        tasks=[TaskOut(id=t.id, title=t.title, done=t.done,
                       done_at=t.done_at, sort=t.sort)
               for t in sorted(tasks, key=lambda x: x.sort)],
        created_at=ev.created_at, completed_at=ev.completed_at,
    )


def _tasks_of(db: Session, event_id: str) -> list[LifecycleTask]:
    return list(db.scalars(
        select(LifecycleTask).where(LifecycleTask.event_id == event_id)
    ))


# ============ templates ============


@router.get("/templates", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db), current: User = Depends(require_permission(PERM_VIEW_REPORTS))
):
    rows = db.scalars(
        select(LifecycleTemplate).where(
            LifecycleTemplate.tenant_id == current.tenant_id
        )
    ).all()
    return [TemplateOut(
        id=t.id, kind=t.kind, name=t.name,
        tasks=[x.get("title", "") if isinstance(x, dict) else str(x)
               for x in (t.tasks or [])],
        is_default=t.is_default,
    ) for t in rows]


@router.post("/templates", response_model=TemplateOut,
             status_code=status.HTTP_201_CREATED)
def create_template(
    body: TemplateIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LIFECYCLE)),
):
    t = LifecycleTemplate(
        tenant_id=current.tenant_id, kind=body.kind, name=body.name.strip(),
        tasks=[{"title": x.strip()} for x in body.tasks if x.strip()],
        is_default=body.is_default,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return TemplateOut(
        id=t.id, kind=t.kind, name=t.name,
        tasks=[x["title"] for x in t.tasks], is_default=t.is_default,
    )


@router.delete("/templates/{tpl_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(
    tpl_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LIFECYCLE)),
):
    t = db.get(LifecycleTemplate, tpl_id)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "模板不存在")
    db.delete(t)
    db.commit()


# ============ events ============


@router.get("/events", response_model=PageOut[EventOut])
def list_events(
    pagination: tuple = Depends(paginate_params),
    kind: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    employee_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(LifecycleEvent).where(
        LifecycleEvent.tenant_id == current.tenant_id
    )
    if kind:
        stmt = stmt.where(LifecycleEvent.kind == kind)
    if status_filter:
        stmt = stmt.where(LifecycleEvent.status == status_filter)
    if employee_id:
        stmt = stmt.where(LifecycleEvent.employee_id == employee_id)
    total = count_total(db, stmt)
    events = db.scalars(
        stmt.order_by(LifecycleEvent.created_at.desc()).limit(limit).offset(offset)
    ).all()
    emps = {e.id: e.name for e in db.scalars(
        select(Employee).where(Employee.tenant_id == current.tenant_id)
    )}
    tasks = db.scalars(
        select(LifecycleTask).where(
            LifecycleTask.event_id.in_([e.id for e in events] or ["__none__"])
        )
    ).all()
    by_ev: dict[str, list[LifecycleTask]] = {}
    for t in tasks:
        by_ev.setdefault(t.event_id, []).append(t)
    return PageOut(
        items=[_event_out(ev, emps.get(ev.employee_id, "?"),
                          by_ev.get(ev.id, [])) for ev in events],
        total=total, limit=limit, offset=offset,
    )


@router.post("/events", response_model=EventOut, status_code=status.HTTP_201_CREATED)
def create_event(
    body: EventIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LIFECYCLE)),
):
    emp = db.get(Employee, body.employee_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    open_same = db.scalars(
        select(LifecycleEvent.id).where(
            LifecycleEvent.employee_id == emp.id,
            LifecycleEvent.kind == body.kind,
            LifecycleEvent.status == "in_progress",
        ).limit(1)
    ).first()
    if open_same:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该员工已有进行中的同类事件")

    payload = dict(body.payload or {})
    # transfer / promotion 自动带上 from_*
    if body.kind == "transfer":
        payload.setdefault("from_org_unit_id", emp.org_unit_id)
        to_id = payload.get("to_org_unit_id")
        if to_id:
            org = db.get(OrgUnit, to_id)
            if not org or org.tenant_id != current.tenant_id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "目标部门不存在")
    if body.kind == "promotion":
        payload.setdefault("from_designation", emp.designation)
    if body.kind == "separation":
        payload.setdefault("from_status", emp.status)

    ev = LifecycleEvent(
        tenant_id=current.tenant_id, employee_id=emp.id, kind=body.kind,
        effective_date=body.effective_date, payload=payload, note=body.note,
        created_by=current.id,
    )
    db.add(ev)
    db.flush()
    # 模板铺任务;没模板也给个默认兜底
    tpl = None
    if body.template_id:
        tpl = db.get(LifecycleTemplate, body.template_id)
        if not tpl or tpl.tenant_id != current.tenant_id or tpl.kind != body.kind:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "模板不匹配")
    if tpl is None:
        tpl = db.scalars(
            select(LifecycleTemplate).where(
                LifecycleTemplate.tenant_id == current.tenant_id,
                LifecycleTemplate.kind == body.kind,
                LifecycleTemplate.is_default.is_(True),
            ).limit(1)
        ).first()
    task_titles = [x.get("title", "") for x in (tpl.tasks or [])] if tpl else []
    if not task_titles:
        task_titles = {
            "onboarding": ["发放offer并确认入职日期", "创建系统账号", "开通门禁/邮箱",
                           "设备领取", "入职培训", "试用期目标确认"],
            "separation": ["交接文档归档", "回收设备与权限", "禁用系统账号",
                           "离职面谈", "薪资结算确认"],
            "transfer": ["新部门负责人确认", "权限与群组调整", "HR系统部门更新"],
            "promotion": ["晋升审批确认", "职级/薪资变更", "公告通知"],
        }.get(body.kind, [])
    for i, title in enumerate(task_titles):
        db.add(LifecycleTask(
            tenant_id=current.tenant_id, event_id=ev.id, title=title, sort=i,
        ))
    # separation 立即标记员工离职态?不 — 等任务完成时再落。保留 in_progress。
    write_audit(
        db, actor=current, entity_type="lifecycle_event", entity_id=ev.id,
        action="create", detail={"kind": ev.kind, "employee": emp.name},
        request=request,
    )
    db.commit()
    db.refresh(ev)
    return _event_out(ev, emp.name, _tasks_of(db, ev.id))


@router.get("/events/{event_id}", response_model=EventOut)
def get_event(
    event_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    ev = db.get(LifecycleEvent, event_id)
    if not ev or ev.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "事件不存在")
    emp = db.get(Employee, ev.employee_id)
    return _event_out(ev, emp.name if emp else "?", _tasks_of(db, ev.id))


@router.post("/events/{event_id}/cancel", response_model=EventOut)
def cancel_event(
    event_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LIFECYCLE)),
):
    ev = db.get(LifecycleEvent, event_id)
    if not ev or ev.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "事件不存在")
    if ev.status != "in_progress":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "事件已结束")
    ev.status = "cancelled"
    write_audit(
        db, actor=current, entity_type="lifecycle_event", entity_id=ev.id,
        action="cancel", detail={}, request=request,
    )
    db.commit()
    emp = db.get(Employee, ev.employee_id)
    return _event_out(ev, emp.name if emp else "?", _tasks_of(db, ev.id))


@router.post("/tasks/{task_id}/toggle", response_model=EventOut)
def toggle_task(
    task_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_LIFECYCLE)),
):
    t = db.get(LifecycleTask, task_id)
    if not t or t.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    # 锁事件行:并发勾选必须串行,否则每个请求都看不到别人刚提交的
    # done=True,"全勾→完成"会被跳过(事件永远卡在 in_progress)
    ev = db.execute(
        select(LifecycleEvent)
        .where(LifecycleEvent.id == t.event_id)
        .with_for_update()
    ).scalar_one()
    if not ev or ev.status != "in_progress":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "事件已结束,任务不能改动")
    t.done = not t.done
    t.done_at = _now() if t.done else None
    # 全部勾选 → 事件完成,并落业务副作用
    tasks = _tasks_of(db, ev.id)
    if all(x.done for x in tasks) and ev.status == "in_progress":
        ev.status = "completed"
        ev.completed_at = _now()
        _apply_event_effect(db, ev)
    write_audit(
        db, actor=current, entity_type="lifecycle_task", entity_id=t.id,
        action="toggle", detail={"done": t.done, "event_id": ev.id},
        request=request,
    )
    db.commit()
    emp = db.get(Employee, ev.employee_id)
    return _event_out(ev, emp.name if emp else "?", tasks)


def _apply_event_effect(db: Session, ev: LifecycleEvent) -> None:
    """事件完成的业务副作用:transfer/promotion/separation 回写员工档案。"""
    emp = db.get(Employee, ev.employee_id)
    if not emp:
        return
    p = ev.payload or {}
    if ev.kind == "transfer" and p.get("to_org_unit_id"):
        emp.org_unit_id = p["to_org_unit_id"]
        # 绑定账号的 org_unit 跟随
        if emp.user_id:
            u = db.get(User, emp.user_id)
            if u:
                u.org_unit_id = emp.org_unit_id
    elif ev.kind == "promotion" and p.get("to_designation"):
        emp.designation = p["to_designation"]
    elif ev.kind == "separation":
        emp.status = "terminated"
        # 离职即禁用员工账号
        if emp.user_id:
            u = db.get(User, emp.user_id)
            if u:
                u.status = "disabled"
    emp.updated_at = _now()
