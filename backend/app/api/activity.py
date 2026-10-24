"""活动流(MintHCM SugarFeed 移植):聚合 audit + 业务事件 → 统一时间线。

``GET /api/activity?days=7&limit=50`` → Dashboard「近期动态」卡。
数据源:
- AuditLog(实体 create/update/transition,映射 verb)
- JobApplicationEvent(管道阶段变迁)
- Reservation(新预约)
- SurveyResponse(新答卷)

每类源取最近 limit 条后在内存归并排序 — 数据量小,不建 UNION SQL。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.license import require_feature
from app.domain.models import (
    Announcement,
    Appraisal,
    Attendance,
    AttendanceRequest,
    AuditLog,
    Candidate,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    Goal,
    HrCase,
    Interview,
    InterviewTurn,
    Job,
    JobApplication,
    JobApplicationEvent,
    JobOffer,
    JobRequisition,
    LeaveAllocation,
    LeaveApplication,
    LifecycleEvent,
    LifecycleTask,
    OrgUnit,
    OvertimeRequest,
    Reservation,
    Resume,
    Room,
    SalaryBand,
    SalarySlip,
    SurveyLink,
    SurveyResponse,
    User,
)
from app.infra.db import get_db
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    apply_org_filter,
    get_org_scope,
    require_permission,
    visible_org_ids,
)

router = APIRouter(
    prefix="/activity",
    tags=["activity"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


class ActivityItem(BaseModel):
    at: datetime
    actor_name: str | None
    verb: str
    entity_kind: str
    entity_label: str | None
    link: str | None
    # AI 留痕:audit.via 透传("assistant"/"mcp"),None = 人工操作
    via: str | None = None


_VERBS = {
    "create": "创建了",
    "update": "更新了",
    "delete": "删除了",
    "publish": "发布了",
    "close": "关闭了",
    "revoke": "吊销了",
}


def _verb(action: str | None) -> str:
    if not action:
        return "操作了"
    for k, v in _VERBS.items():
        if k in action:
            return v
    if action.startswith("status:"):
        return "流转了"
    if action.startswith("decide:"):
        return "审批了"
    return action


_ENTITY_LINKS = {
    "candidate": "/talents/{id}",
    "job": "/jobs/{id}",
    "interview": "/interviews/{id}",
    "survey": "/surveys",
    "announcement": "/announcements",
    "hr_case": "/cases",
    "employee_request": "/requests",
    "talent_campaign": "/campaigns",
    "room": "/interviews/schedule",
    "salary_band": "/salary-bands",
}


# audit detail 缺 title/name 时的兜底 label —— 否则前端渲染出
#「xxx 删除了 」这种悬空动词。未登记类型退回 entity_type 原文。
_ENTITY_KIND_LABEL = {
    "candidate": "候选人",
    "job": "岗位",
    "interview": "面试",
    "interview_turn": "面试题",
    "employee": "员工",
    "resume": "简历",
    "job_application": "管道卡",
    "job_offer": "Offer",
    "job_requisition": "编制申请",
    "org_unit": "组织",
    "room": "会议室",
    "reservation": "预约",
    "announcement": "公告",
    "hr_case": "工单",
    "hr_case_update": "工单进展",
    "employee_request": "员工申请",
    "salary_band": "薪酬带宽",
    "talent_campaign": "外联活动",
    "survey": "问卷",
    "survey_link": "问卷链接",
    "survey_response": "答卷",
    "knowledge_article": "知识库文章",
    "leave_application": "请假申请",
    "leave_allocation": "假期额度",
    "attendance": "考勤",
    "attendance_request": "补卡申请",
    "overtime_request": "加班申请",
    "salary_slip": "工资条",
    "expense_claim": "报销",
    "goal": "目标",
    "appraisal": "绩效",
    "lifecycle_event": "入离职事件",
    "lifecycle_task": "入离职任务",
    "employee_account": "员工账号",
    "tenant": "租户",
    "email_template": "邮件模板",
    "smtp_account": "SMTP 账户",
    "email_account": "邮箱账户",
    "question_item": "题目",
    "question_set": "题组",
    "skill": "技能",
    "ai_watcher": "提醒",
    "report_subscription": "报表订阅",
    "mcp_token": "MCP 令牌",
}


def _audit_label(a: AuditLog) -> str:
    detail = a.detail or {}
    return (
        detail.get("title")
        or detail.get("name")
        or _ENTITY_KIND_LABEL.get(a.entity_type, a.entity_type)
    )


# scoped 用户的 audit 行过滤:按实体登记 org 归属,逐类型一次批量查询。
# - _DIRECT:实体自带 org 列(audience_org_unit_id 语义同 org);
# - _VIA_EMPLOYEE / _VIA_CANDIDATE:经 FK 挂到 Employee/Candidate.org_unit_id;
# - interview_turn / lifecycle_task / reservation:二级挂接,在
#   _visible_audit_ids 里单独解析;
# - org_unit 审计的实体即 org 节点自身 → org = entity_id;
# - "report" 审计的 entity_id 即 interview.id → 与 "interview" 共用解析。
# 未登记的类型(配置/字典类,label 不含人名或 org 数据)按公共处理 ——
# 与 apply_org_filter「org 为空 = 全租户共享」同一妥协口径。
_DIRECT_ORG = {
    "announcement": (Announcement, Announcement.audience_org_unit_id),
    "candidate": (Candidate, Candidate.org_unit_id),
    "employee": (Employee, Employee.org_unit_id),
    "job": (Job, Job.org_unit_id),
    "job_requisition": (JobRequisition, JobRequisition.org_unit_id),
    "org_unit": (OrgUnit, OrgUnit.id),
    "room": (Room, Room.org_unit_id),
    "salary_band": (SalaryBand, SalaryBand.org_unit_id),
}
_VIA_EMPLOYEE = {
    "appraisal": (Appraisal, Appraisal.employee_id),
    "attendance": (Attendance, Attendance.employee_id),
    "attendance_request": (AttendanceRequest, AttendanceRequest.employee_id),
    "employee_request": (EmployeeRequest, EmployeeRequest.employee_id),
    "expense_claim": (ExpenseClaim, ExpenseClaim.employee_id),
    "goal": (Goal, Goal.employee_id),
    "hr_case": (HrCase, HrCase.requester_employee_id),
    "leave_allocation": (LeaveAllocation, LeaveAllocation.employee_id),
    "leave_application": (LeaveApplication, LeaveApplication.employee_id),
    "lifecycle_event": (LifecycleEvent, LifecycleEvent.employee_id),
    "overtime_request": (OvertimeRequest, OvertimeRequest.employee_id),
    "salary_slip": (SalarySlip, SalarySlip.employee_id),
}
_VIA_CANDIDATE = {
    "interview": (Interview, Interview.candidate_id),
    "job_application": (JobApplication, JobApplication.candidate_id),
    "job_offer": (JobOffer, JobOffer.candidate_id),
    "report": (Interview, Interview.candidate_id),
    "resume": (Resume, Resume.candidate_id),
    "survey_link": (SurveyLink, SurveyLink.candidate_id),
}
_SCOPED_TYPES = (
    set(_DIRECT_ORG)
    | set(_VIA_EMPLOYEE)
    | set(_VIA_CANDIDATE)
    | {"interview_turn", "lifecycle_task", "reservation"}
)


def _visible_audit_ids(
    db: Session,
    tenant_id: str,
    audits: list[AuditLog],
    scope: list[str] | None,
) -> set[str]:
    """返回 scoped 用户可见的 audit id 集合。

    已登记类型的实体按 org 判定;实体已删(查不到 org)保守隐藏;
    未登记类型视为公共可见。``scope is None``(admin)直通。
    """
    if scope is None:
        return {a.id for a in audits}

    by_type: dict[str, set[str]] = {}
    for a in audits:
        by_type.setdefault(a.entity_type, set()).add(a.entity_id)

    # (entity_type, entity_id) → org 值;reservation 存 (room_org, cand_org) 元组
    org_of: dict[tuple[str, str], object] = {}

    for et, (model, org_col) in _DIRECT_ORG.items():
        ids = by_type.get(et)
        if not ids:
            continue
        for eid, org in db.execute(
            select(model.id, org_col).where(
                model.tenant_id == tenant_id, model.id.in_(ids)
            )
        ):
            org_of[(et, eid)] = org
    for et, (model, emp_col) in _VIA_EMPLOYEE.items():
        ids = by_type.get(et)
        if not ids:
            continue
        for eid, org in db.execute(
            select(model.id, Employee.org_unit_id)
            .join(Employee, Employee.id == emp_col)
            .where(model.tenant_id == tenant_id, model.id.in_(ids))
        ):
            org_of[(et, eid)] = org
    for et, (model, cand_col) in _VIA_CANDIDATE.items():
        ids = by_type.get(et)
        if not ids:
            continue
        for eid, org in db.execute(
            select(model.id, Candidate.org_unit_id)
            .join(Candidate, Candidate.id == cand_col)
            .where(model.tenant_id == tenant_id, model.id.in_(ids))
        ):
            org_of[(et, eid)] = org

    # interview_turn → Interview → Candidate(turn 表本身无 tenant 列,
    # tenant 过滤落在外层 Interview 上)
    ids = by_type.get("interview_turn")
    if ids:
        for eid, org in db.execute(
            select(InterviewTurn.id, Candidate.org_unit_id)
            .join(Interview, Interview.id == InterviewTurn.interview_id)
            .join(Candidate, Candidate.id == Interview.candidate_id)
            .where(Interview.tenant_id == tenant_id, InterviewTurn.id.in_(ids))
        ):
            org_of[("interview_turn", eid)] = org
    # lifecycle_task → LifecycleEvent → Employee
    ids = by_type.get("lifecycle_task")
    if ids:
        for eid, org in db.execute(
            select(LifecycleTask.id, Employee.org_unit_id)
            .join(LifecycleEvent, LifecycleEvent.id == LifecycleTask.event_id)
            .join(Employee, Employee.id == LifecycleEvent.employee_id)
            .where(LifecycleTask.tenant_id == tenant_id, LifecycleTask.id.in_(ids))
        ):
            org_of[("lifecycle_task", eid)] = org
    # reservation:room / interview→candidate 任一挂 org 都要可见
    ids = by_type.get("reservation")
    if ids:
        for rid, room_org, cand_org in db.execute(
            select(Reservation.id, Room.org_unit_id, Candidate.org_unit_id)
            .outerjoin(Room, Room.id == Reservation.room_id)
            .outerjoin(Interview, Interview.id == Reservation.interview_id)
            .outerjoin(Candidate, Candidate.id == Interview.candidate_id)
            .where(Reservation.tenant_id == tenant_id, Reservation.id.in_(ids))
        ):
            org_of[("reservation", rid)] = (room_org, cand_org)

    out = set()
    for a in audits:
        key = (a.entity_type, a.entity_id)
        if key in org_of:
            orgs = org_of[key]
            if not isinstance(orgs, tuple):
                orgs = (orgs,)
            if all(visible_org_ids(scope, o) for o in orgs):
                out.add(a.id)
        elif a.entity_type not in _SCOPED_TYPES:
            # 未登记类型 → 公共可见;已登记但实体已删(查不到 org)→ 保守隐藏
            out.add(a.id)
    return out


def _orgs_visible(scope: list[str] | None, *orgs: str | None) -> bool:
    """多 org 来源的可见性:所有非空 org 都在 scope 内才可见。"""
    return all(visible_org_ids(scope, o) for o in orgs)


@router.get("", response_model=list[ActivityItem])
def activity_feed(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    scope = get_org_scope(db, current)
    items: list[ActivityItem] = []

    # 1) audit 里的业务写(排除 auth 类噪音);scoped 用户按实体 org 过滤
    audits = db.execute(
        select(AuditLog)
        .where(
            AuditLog.tenant_id == current.tenant_id,
            AuditLog.created_at >= since,
            AuditLog.entity_type.notin_(("user", "auth", "sso_config", "sso_login", "llm_provider", "license")),
        )
        .order_by(AuditLog.created_at.desc())
        .limit(limit * 2)
    ).scalars().all()
    visible_ids = _visible_audit_ids(db, current.tenant_id, audits, scope)
    for a in audits:
        if a.id not in visible_ids:
            continue
        link_tpl = _ENTITY_LINKS.get(a.entity_type)
        items.append(
            ActivityItem(
                at=a.created_at,
                actor_name=a.actor_email,
                verb=_verb(a.action),
                entity_kind=a.entity_type,
                entity_label=_audit_label(a),
                link=link_tpl.format(id=a.entity_id) if link_tpl else None,
                via=a.via,
            )
        )

    # 2) 管道阶段变迁(event → application → candidate 取候选人名)
    ev_stmt = (
        select(JobApplicationEvent, Candidate.name, User.email)
        .join(JobApplication, JobApplication.id == JobApplicationEvent.application_id)
        .join(Candidate, Candidate.id == JobApplication.candidate_id, isouter=True)
        .join(User, User.id == JobApplicationEvent.actor_user_id, isouter=True)
        .where(
            JobApplicationEvent.tenant_id == current.tenant_id,
            JobApplicationEvent.created_at >= since,
        )
        .order_by(JobApplicationEvent.created_at.desc())
        .limit(limit)
    )
    ev_stmt = apply_org_filter(ev_stmt, org_column=Candidate.org_unit_id, scope=scope)
    evs = db.execute(ev_stmt).all()
    for e, cname, actor_email in evs:
        items.append(
            ActivityItem(
                at=e.created_at,
                actor_name=actor_email,
                verb=f"从 {e.from_stage or '-'} 流转到 {e.to_stage}",
                entity_kind="job_application",
                entity_label=cname,
                link="/pipeline",
            )
        )

    # 3) 新预约 / 新答卷
    # 预约无 org —— 与 ics.py 同口径:room 看会议室 org,interview 预约看
    # 候选人 org(标题常含候选人名);resource 预约无 org 维度,公共可见。
    rsvs = db.execute(
        select(Reservation, Room.org_unit_id, Candidate.org_unit_id)
        .outerjoin(Room, Room.id == Reservation.room_id)
        .outerjoin(Interview, Interview.id == Reservation.interview_id)
        .outerjoin(Candidate, Candidate.id == Interview.candidate_id)
        .where(Reservation.tenant_id == current.tenant_id, Reservation.created_at >= since)
        .order_by(Reservation.created_at.desc())
        .limit(limit)
    ).all()
    for r, room_org, cand_org in rsvs:
        if not _orgs_visible(scope, room_org, cand_org):
            continue
        items.append(
            ActivityItem(
                at=r.created_at,
                actor_name=None,
                verb="新建了预约",
                entity_kind="reservation",
                entity_label=r.title,
                link="/interviews/schedule",
            )
        )
    resp = db.scalars(
        select(SurveyResponse)
        .where(SurveyResponse.tenant_id == current.tenant_id, SurveyResponse.submitted_at >= since)
        .order_by(SurveyResponse.submitted_at.desc())
        .limit(limit)
    ).all()
    for r in resp:
        items.append(
            ActivityItem(
                at=r.submitted_at,
                actor_name=None,
                verb="提交了问卷",
                entity_kind="survey_response",
                entity_label=None,
                link="/surveys",
            )
        )

    items.sort(key=lambda x: x.at, reverse=True)
    return items[:limit]
