"""AI 报表执行(注册表 kind → 结构化 dict)。

白名单 kind → 结构化 dict 数据。与 ``workers/tasks/notify.py`` 的订阅 digest
口径同源但更结构化:订阅发邮件文本,这里返回机器可读数据供 assistant 解读。

- ``GET /report-defs`` → 当前用户可跑的 kind 清单(权限+feature 过滤)
- ``POST /report-defs/{kind}/run {params}`` → ``{kind, generated_at, data}``
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Candidate,
    Employee,
    EmployeeRequest,
    HrCase,
    Interview,
    Job,
    JobApplication,
    Notification,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.infra.license.state import is_feature_enabled
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    apply_org_filter,
    get_org_scope,
    has_permission,
    visible_employee_ids,
)

router = APIRouter(prefix="/report-defs", tags=["report-defs"])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _r_pipeline_funnel(db: Session, u: User, p: dict) -> dict:
    scope = get_org_scope(db, u)
    # 管道卡跟随候选人 org(与 _applications builder / applications.py 同口径)
    funnel_stmt = (
        select(JobApplication.stage, func.count())
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(JobApplication.tenant_id == u.tenant_id)
    )
    funnel_stmt = apply_org_filter(funnel_stmt, org_column=Candidate.org_unit_id, scope=scope)
    rows = db.execute(funnel_stmt.group_by(JobApplication.stage)).all()
    jobs_stmt = select(func.count(Job.id)).where(Job.tenant_id == u.tenant_id, Job.status == "open")
    jobs_stmt = apply_org_filter(jobs_stmt, org_column=Job.org_unit_id, scope=scope)
    open_jobs = db.scalar(jobs_stmt) or 0
    return {
        "stages": [{"stage": s, "count": n} for s, n in sorted(rows)],
        "total": sum(n for _, n in rows),
        "open_jobs": open_jobs,
    }


def _r_weekly_summary(db: Session, u: User, p: dict) -> dict:
    # days 是 LLM 侧自由参数 — 非数字回退默认 7,而不是 ValueError 炸 500
    try:
        days = int(p.get("days") or 7)
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 30))
    since = _now() - timedelta(days=days)
    scope = get_org_scope(db, u)
    apps_stmt = (
        select(func.count(JobApplication.id))
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(JobApplication.tenant_id == u.tenant_id, JobApplication.created_at >= since)
    )
    apps_stmt = apply_org_filter(apps_stmt, org_column=Candidate.org_unit_id, scope=scope)
    new_apps = db.scalar(apps_stmt) or 0
    ivs_stmt = (
        select(func.count(Interview.id))
        .join(Candidate, Candidate.id == Interview.candidate_id)
        .where(
            Interview.tenant_id == u.tenant_id,
            Interview.status == "done",
            Interview.finished_at >= since,
        )
    )
    ivs_stmt = apply_org_filter(ivs_stmt, org_column=Candidate.org_unit_id, scope=scope)
    done_ivs = db.scalar(ivs_stmt) or 0
    new_employees = 0
    if is_feature_enabled(db, "hr.suite"):
        emp_stmt = select(func.count(Employee.id)).where(
            Employee.tenant_id == u.tenant_id, Employee.created_at >= since
        )
        emp_stmt = apply_org_filter(emp_stmt, org_column=Employee.org_unit_id, scope=scope)
        new_employees = db.scalar(emp_stmt) or 0
    return {
        "days": days,
        "new_applications": new_apps,
        "interviews_done": done_ivs,
        "new_employees": new_employees,
    }


def _r_pending_items(db: Session, u: User, p: dict) -> dict:
    emp_ids = visible_employee_ids(db, u)
    open_status = ("new", "assigned", "in_progress")
    cases_q = select(func.count(HrCase.id)).where(
        HrCase.tenant_id == u.tenant_id, HrCase.status.in_(open_status)
    )
    reqs_q = select(func.count(EmployeeRequest.id)).where(
        EmployeeRequest.tenant_id == u.tenant_id, EmployeeRequest.status == "pending"
    )
    if emp_ids is not None:
        cases_q = cases_q.where(HrCase.requester_employee_id.in_(emp_ids or ["__none__"]))
        reqs_q = reqs_q.where(EmployeeRequest.employee_id.in_(emp_ids or ["__none__"]))
    unread = db.scalar(
        select(func.count(Notification.id)).where(
            Notification.user_id == u.id, Notification.read_at.is_(None)
        )
    ) or 0
    return {
        "open_cases": db.scalar(cases_q) or 0,
        "pending_requests": db.scalar(reqs_q) or 0,
        "unread_notifications": unread,
    }


def _r_headcount(db: Session, u: User, p: dict) -> dict:
    scope = get_org_scope(db, u)
    # org scope 与 /api/employees 列表同口径 —— 非 admin 的 org 绑定用户
    # 不该拿到跨 org 的编制数
    by_status_stmt = (
        select(Employee.status, func.count())
        .where(Employee.tenant_id == u.tenant_id)
    )
    by_status_stmt = apply_org_filter(by_status_stmt, org_column=Employee.org_unit_id, scope=scope)
    by_status = db.execute(by_status_stmt.group_by(Employee.status)).all()
    # left join 口径:本租户 org 全列出(含 0 人 org),where 不能把 outer join 的
    # NULL 侧过滤掉 —— 否则 0 人组织消失、by_org 合计对不上 total。
    # scope 非 None 时只列出 scope 内的 org 节点(scope=[] → 只剩"未分配"桶)。
    by_org_stmt = (
        select(OrgUnit.name, func.count(Employee.id))
        .join(Employee, Employee.org_unit_id == OrgUnit.id, isouter=True)
        .where(OrgUnit.tenant_id == u.tenant_id)
        .where(or_(Employee.tenant_id == u.tenant_id, Employee.id.is_(None)))
    )
    if scope is not None:
        by_org_stmt = by_org_stmt.where(OrgUnit.id.in_(scope or ["__none__"]))
    by_org = db.execute(by_org_stmt.group_by(OrgUnit.name)).all()
    # 未分配组织的员工单列,与 by_status 口径对齐
    unassigned = db.scalar(
        select(func.count(Employee.id)).where(
            Employee.tenant_id == u.tenant_id, Employee.org_unit_id.is_(None)
        )
    ) or 0
    org_rows = [{"org": n, "count": c} for n, c in sorted(by_org)]
    if unassigned:
        org_rows.append({"org": "(未分配)", "count": unassigned})
    return {
        "by_status": [{"status": s, "count": n} for s, n in sorted(by_status)],
        "by_org": org_rows,
        "total": sum(n for _, n in by_status),
    }


REPORT_DEFS: dict[str, dict] = {
    "pipeline_funnel": {
        "label": "招聘管道漏斗",
        "description": "各阶段管道卡数量 + 开放岗位数",
        "perm": PERM_VIEW_REPORTS,
        "feature": None,
        "run": _r_pipeline_funnel,
    },
    "weekly_summary": {
        "label": "近期摘要",
        "description": "近 N 天新增管道卡/完成面试/入职员工(days 参数,默认 7)",
        "perm": PERM_VIEW_REPORTS,
        "feature": None,
        "run": _r_weekly_summary,
    },
    "pending_items": {
        "label": "待办提醒",
        "description": "当前用户的未结工单/待审批申请/未读通知",
        "perm": None,
        "feature": None,
        "run": _r_pending_items,
    },
    "headcount": {
        "label": "员工编制",
        "description": "员工按状态/组织分布",
        "perm": PERM_MANAGE_EMPLOYEES,
        "feature": "hr.suite",
        "run": _r_headcount,
    },
}


class RunIn(BaseModel):
    params: dict = Field(default_factory=dict)


@router.get("")
def list_defs(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return [
        {
            "kind": k,
            "label": d["label"],
            "description": d["description"],
        }
        for k, d in REPORT_DEFS.items()
        if (d["perm"] is None or has_permission(current, d["perm"]))
        and (not d["feature"] or is_feature_enabled(db, d["feature"]))
    ]


@router.post("/{kind}/run")
def run_def(
    kind: str,
    body: RunIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    d = REPORT_DEFS.get(kind)
    if d is None:
        raise HTTPException(404, f"未知报表:{kind}")
    if d["perm"] is not None and not has_permission(current, d["perm"]):
        raise HTTPException(403, "权限不足")
    if d["feature"]:
        require_feature(d["feature"])()
    return {
        "kind": kind,
        "label": d["label"],
        "generated_at": _now().isoformat(),
        "data": d["run"](db, current, body.params or {}),
    }
