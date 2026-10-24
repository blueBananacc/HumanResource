"""触发式任务(watchers)—— AI-native 轻量工作流。

每个 ``AiWatcher`` 属于一个用户;``kind`` 注册在 :data:`EVALUATORS`,评估器签名::

    def evaluate(db, watcher) -> list[dict]   # {"title","body","link"}

返回空 list = 未触发;非空 → 逐条写 Notification(同日同 link 查重)。
调度由 ``app.main._watch_loop`` 周期驱动(redis 锁,单实例跑)。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    AiWatcher,
    AttendanceFlag,
    Candidate,
    Employee,
    EmployeeRequest,
    HrCase,
    Interview,
    JobApplication,
    Notification,
    Reservation,
    User,
)
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    PERM_VIEW_REPORTS,
    apply_org_filter,
    get_linked_employee_id,
    get_org_scope,
    has_permission,
    visible_employee_ids,
)
from app.services.timeutil import local_day_start_utc, local_now

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _days(w: AiWatcher, default: int = 3) -> int:
    try:
        return max(1, int(w.params.get("days", default)))
    except (TypeError, ValueError):
        return default


def _eval_candidate_stalled(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """管道卡在 interview/screening 超过 params.days 天。

    按 owner 的 org scope 收敛 —— 通知正文带候选人名,不能把
    范围外组织的候选人推给受限用户。
    """
    days = _days(w)
    cutoff = _now() - timedelta(days=days)
    owner = db.get(User, w.user_id)
    if owner is None:
        return []
    stmt = (
        select(JobApplication, Candidate.name)
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .where(
            JobApplication.tenant_id == w.tenant_id,
            JobApplication.stage.in_(("screening", "interview")),
            JobApplication.stage_changed_at < cutoff,
        )
    )
    stmt = apply_org_filter(
        stmt, org_column=Candidate.org_unit_id, scope=get_org_scope(db, owner)
    )
    rows = db.execute(stmt.limit(20)).all()
    return [
        {
            "title": "候选人停滞提醒",
            "body": f"「{name}」在 {app.stage} 阶段已停留超 {days} 天",
            "link": f"/talents/{app.candidate_id}",
            "dedupe_key": f"stalled:{app.id}",
        }
        for app, name in rows
    ]


def _eval_interview_pending(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """已完成但无 summary 报告的面试(finished 超 1 天)。

    按 owner org scope 收敛 —— 与面试列表 / ICS 同一口径。
    """
    cutoff = _now() - timedelta(days=1)
    owner = db.get(User, w.user_id)
    if owner is None:
        return []
    stmt = (
        select(Interview)
        .join(Candidate, Candidate.id == Interview.candidate_id)
        .where(
            Interview.tenant_id == w.tenant_id,
            Interview.status == "done",
            Interview.finished_at < cutoff,
            Interview.summary.is_(None),
        )
    )
    stmt = apply_org_filter(
        stmt, org_column=Candidate.org_unit_id, scope=get_org_scope(db, owner)
    )
    rows = db.scalars(stmt.limit(20)).all()
    return [
        {
            "title": "面试待出报告",
            "body": "有面试已结束超过 1 天仍未生成报告",
            "link": f"/interviews/{r.id}/report" if r.id else "/interviews",
            "dedupe_key": f"pending-report:{r.id}",
        }
        for r in rows
    ]


def _eval_case_open_hours(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """分配给本人的工单超过 params.hours 仍未解决。"""
    try:
        hours = max(1, int(w.params.get("hours", 24)))
    except (TypeError, ValueError):
        hours = 24
    cutoff = _now() - timedelta(hours=hours)
    rows = db.scalars(
        select(HrCase).where(
            HrCase.tenant_id == w.tenant_id,
            HrCase.assignee_user_id == w.user_id,
            HrCase.status.in_(("new", "assigned", "in_progress")),
            HrCase.created_at < cutoff,
        )
    ).all()
    return [
        {
            "title": "工单超时未结",
            "body": f"工单 HR-{r.case_no}「{r.subject}」已开 {hours}h+",
            "link": "/cases",
            "dedupe_key": f"case-overdue:{r.id}",
        }
        for r in rows
    ]


def _eval_request_pending(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """申请超过 params.days 天仍 pending。

    数据范围按 owner 权限收敛:
    - ``manage_employees``(HR/admin)→ 全租户审批队列积压;
    - 其他角色 → 仅本人提交的申请,链接指到 /me/requests。
    """
    days = _days(w)
    cutoff = _now() - timedelta(days=days)
    owner = db.get(User, w.user_id)
    stmt = select(EmployeeRequest).where(
        EmployeeRequest.tenant_id == w.tenant_id,
        EmployeeRequest.status == "pending",
        EmployeeRequest.created_at < cutoff,
    )
    is_hr = owner is not None and has_permission(owner, PERM_MANAGE_EMPLOYEES)
    if not is_hr:
        emp_id = get_linked_employee_id(db, owner) if owner else None
        if emp_id is None:
            return []
        stmt = stmt.where(EmployeeRequest.employee_id == emp_id)
    rows = db.scalars(stmt.limit(20)).all()
    return [
        {
            "title": "员工申请待审批",
            "body": f"「{r.subject}」已等待 {days} 天",
            "link": "/requests" if is_hr else "/me/requests",
            "dedupe_key": f"req-pending:{r.id}",
        }
        for r in rows
    ]


def _eval_reservation_upcoming(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """本人预约即将开始(params.minutes 内)。"""
    try:
        minutes = max(1, int(w.params.get("minutes", 60)))
    except (TypeError, ValueError):
        minutes = 60
    # starts_at 存的是客户端墙钟(app.api.rooms._naive 约定),与"现在"比较
    # 必须用同一口径 —— 部署时区的当前时间,而不是 UTC。
    now = local_now()
    until = now + timedelta(minutes=minutes)
    rows = db.scalars(
        select(Reservation).where(
            Reservation.tenant_id == w.tenant_id,
            Reservation.created_by == w.user_id,
            Reservation.status == "active",
            Reservation.starts_at >= now,
            Reservation.starts_at <= until,
        )
    ).all()
    return [
        {
            "title": "预约即将开始",
            "body": f"「{r.title}」{r.starts_at.strftime('%H:%M')} 开始",
            "link": "/interviews/schedule",
            "dedupe_key": f"rsv-soon:{r.id}",
        }
        for r in rows
    ]


def _eval_late_repeat(db: Session, w: AiWatcher) -> list[dict[str, Any]]:
    """范围内员工近 params.days 天未免计迟到 ≥ params.count 次。"""
    days = _days(w, default=30)
    try:
        count = max(1, int(w.params.get("count", 3)))
    except (TypeError, ValueError):
        count = 3
    cutoff = (_now() - timedelta(days=days)).date()
    owner = db.get(User, w.user_id)
    if owner is None:
        return []
    emp_ids = visible_employee_ids(db, owner)
    stmt = (
        select(AttendanceFlag.employee_id, func.count(AttendanceFlag.id))
        .where(
            AttendanceFlag.tenant_id == w.tenant_id,
            AttendanceFlag.kind == "late_in",
            AttendanceFlag.excused.is_(False),
            AttendanceFlag.day >= cutoff,
        )
        .group_by(AttendanceFlag.employee_id)
        .having(func.count(AttendanceFlag.id) >= count)
    )
    if emp_ids is not None:
        stmt = stmt.where(AttendanceFlag.employee_id.in_(emp_ids or ["__none__"]))
    rows = db.execute(stmt.limit(20)).all()
    names = {
        e.id: e.name
        for e in db.scalars(
            select(Employee).where(Employee.id.in_([r[0] for r in rows]))
        ).all()
    } if rows else {}
    return [
        {
            "title": "多次迟到提醒",
            "body": f"「{names.get(emp_id, '?')}」近 {days} 天已迟到 {cnt} 次",
            "link": "/attendance/flags",
            "dedupe_key": f"late-repeat:{emp_id}:{cutoff.isoformat()}",
        }
        for emp_id, cnt in rows
    ]


EVALUATORS = {
    "candidate_stalled": _eval_candidate_stalled,
    "interview_pending_report": _eval_interview_pending,
    "case_open_hours": _eval_case_open_hours,
    "request_pending_days": _eval_request_pending,
    "reservation_upcoming": _eval_reservation_upcoming,
    "late_repeat": _eval_late_repeat,
}

WATCHER_KINDS = tuple(EVALUATORS.keys())

# 每个 kind 的创建权限位:None = 数据本身按 owner 行级收敛,任何登录用户可建;
# 其余要求 owner 具备对应权限(在 create_watcher 与 run_watchers 双层校验,
# 防止权限被降后旧 watcher 继续推送越权数据)。
WATCHER_PERMS: dict[str, str | None] = {
    "candidate_stalled": PERM_VIEW_REPORTS,
    "interview_pending_report": PERM_VIEW_REPORTS,
    # 受理人恒为 manage_employees(assign 端点校验)— 无此权限的用户建
    # 该 watcher 永远打不响,创建期直接 403 比静默空跑更可解释
    "case_open_hours": PERM_MANAGE_EMPLOYEES,
    "request_pending_days": None,
    "reservation_upcoming": None,
    "late_repeat": PERM_VIEW_REPORTS,
}


def _already_notified_today(db: Session, w: AiWatcher, dedupe_key: str) -> bool:
    # "今天"按部署时区口径;created_at 是 UTC 时间戳,日界先换算回 UTC 再比
    day_start = local_day_start_utc()
    return (
        db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == w.user_id,
                Notification.kind == f"watch:{w.kind}",
                Notification.link == dedupe_key,
                Notification.created_at >= day_start,
            )
        ).scalar_one()
        > 0
    )


def run_watchers(db: Session) -> int:
    """跑一轮所有 active watcher,返回触发的通知数。"""
    fired = 0
    watchers = db.scalars(select(AiWatcher).where(AiWatcher.is_active.is_(True))).all()
    for w in watchers:
        evaluator = EVALUATORS.get(w.kind)
        if evaluator is None:
            continue
        required = WATCHER_PERMS.get(w.kind)
        if required is not None:
            owner = db.get(User, w.user_id)
            if owner is None or not has_permission(owner, required):
                continue
        try:
            hits = evaluator(db, w)
        except Exception:  # noqa: BLE001
            logger.exception("watcher %s(%s)评估失败", w.id, w.kind)
            continue
        for hit in hits:
            dedupe = hit.get("dedupe_key") or hit["link"]
            # dedupe_key 存进 link 的 fragment:既保留可导航性,又能按 link 精确查重
            link = f"{hit.get('link') or '/'}#{dedupe}"
            if _already_notified_today(db, w, link):
                continue
            db.add(
                Notification(
                    tenant_id=w.tenant_id,
                    user_id=w.user_id,
                    kind=f"watch:{w.kind}",
                    title=hit["title"],
                    body=hit.get("body"),
                    link=link,
                )
            )
            w.last_fired_at = _now()
            fired += 1
        w.last_checked_at = _now()
    db.commit()
    return fired
