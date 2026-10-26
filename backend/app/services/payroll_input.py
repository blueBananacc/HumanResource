# -*- coding: utf-8 -*-
"""薪酬底表(E4)— 月度算薪输入聚合(report kind ``payroll_input``)。

不算薪,只把算薪要的输入聚成每人一行,供财务/外部系统对接:

- 应出勤工作日(租户工作日历)
- 考勤:出勤/请假/缺勤天数、工时合计、未免计异常条数
- 请假:本月批准且覆盖本月的申请,按与月交集的工作日数计入
- 加班:本月批准的小时数 + 折算调休天数
- 报销:本月批准的单据总额
- 排班:本月班次天数

口径说明:
- 请假天数按「申请区间 ∩ 月区间 ∩ 工作日」逐日计,跨月单只算落在本月的
  部分 —— 与台账扣减口径不同(台账按整单 -days),底表要的是按月切片。
- 报销按 decided_at 落月(批准才计入),加班按 work_date 落月。
- 全员在册 + 本月内离职的都出(离职当月仍要算薪);hire_date 晚于月末
  的跳过。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Attendance,
    AttendanceFlag,
    Employee,
    ExpenseClaim,
    LeaveApplication,
    LeaveType,
    OrgUnit,
    OvertimeRequest,
    ShiftAssignment,
    User,
)
from app.services.permissions import get_org_scope, apply_org_filter
from app.services.work_calendar import count_workdays


def _month_range(month: str | None) -> tuple[date, date]:
    if month:
        y, m = int(month[:4]), int(month[5:7])
        start = date(y, m, 1)
    else:
        today = date.today()
        start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    end = (start.replace(day=28) + timedelta(days=8)).replace(day=1) - timedelta(days=1)
    return start, end


def _overlap_workdays(
    db: Session, tenant_id: str, a_start: date, a_end: date,
    m_start: date, m_end: date,
) -> float:
    start, end = max(a_start, m_start), min(a_end, m_end)
    if start > end:
        return 0.0
    return float(count_workdays(db, tenant_id, start, end))


def build_payroll_input(
    db: Session, user: User, month: str | None = None
) -> dict[str, Any]:
    m_start, m_end = _month_range(month)
    month_key = m_start.strftime("%Y-%m")
    scope = get_org_scope(db, user)

    emp_stmt = (
        select(Employee)
        .where(
            Employee.tenant_id == user.tenant_id,
            # 月末后入职的不出;status 含 terminated(离职当月仍算薪)
            (Employee.hire_date.is_(None)) | (Employee.hire_date <= m_end),
        )
        .order_by(Employee.employee_no)
    )
    emp_stmt = apply_org_filter(emp_stmt, org_column=Employee.org_unit_id, scope=scope)
    emps = db.scalars(emp_stmt).all()
    if not emps:
        return {"month": month_key, "workdays": 0, "rows": [], "total": 0}

    emp_ids = [e.id for e in emps]
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == user.tenant_id))}

    workdays = count_workdays(db, user.tenant_id, m_start, m_end)

    # --- 考勤:按状态计数 + 工时合计 ---
    att_rows: dict[str, dict[str, Any]] = {}
    for emp_id, st, n, hrs in db.execute(
        select(Attendance.employee_id, Attendance.status,
               func.count(), func.coalesce(func.sum(Attendance.work_hours), 0.0))
        .where(Attendance.tenant_id == user.tenant_id,
               Attendance.employee_id.in_(emp_ids),
               Attendance.day >= m_start, Attendance.day <= m_end)
        .group_by(Attendance.employee_id, Attendance.status)
    ):
        d = att_rows.setdefault(emp_id, {"by_status": {}, "hours": 0.0})
        d["by_status"][st] = n
        d["hours"] += float(hrs or 0)

    # --- 未免计考勤异常条数 ---
    flag_rows = dict(db.execute(
        select(AttendanceFlag.employee_id, func.count())
        .where(AttendanceFlag.tenant_id == user.tenant_id,
               AttendanceFlag.employee_id.in_(emp_ids),
               AttendanceFlag.day >= m_start, AttendanceFlag.day <= m_end,
               AttendanceFlag.excused.is_(False))
        .group_by(AttendanceFlag.employee_id)
    ).all())

    # --- 请假:批准单 ∩ 月 的工作日数,按类型分桶 ---
    lt_map = {lt.id: lt.name for lt in db.scalars(
        select(LeaveType).where(LeaveType.tenant_id == user.tenant_id))}
    leave_apps = db.scalars(
        select(LeaveApplication).where(
            LeaveApplication.tenant_id == user.tenant_id,
            LeaveApplication.employee_id.in_(emp_ids),
            LeaveApplication.status == "approved",
            LeaveApplication.start_date <= m_end,
            LeaveApplication.end_date >= m_start,
        )
    ).all()
    leave_days: dict[str, float] = {}
    leave_by_type: dict[str, dict[str, float]] = {}
    for a in leave_apps:
        d = _overlap_workdays(db, user.tenant_id, a.start_date, a.end_date,
                              m_start, m_end)
        if d <= 0:
            continue
        leave_days[a.employee_id] = leave_days.get(a.employee_id, 0.0) + d
        by_t = leave_by_type.setdefault(a.employee_id, {})
        name = lt_map.get(a.leave_type_id, "?")
        by_t[name] = by_t.get(name, 0.0) + d

    # --- 加班(批准,work_date 落月)---
    ot_rows: dict[str, dict[str, float]] = {}
    for emp_id, hrs, days in db.execute(
        select(OvertimeRequest.employee_id,
               func.coalesce(func.sum(OvertimeRequest.hours), 0.0),
               func.coalesce(func.sum(OvertimeRequest.comp_days), 0.0))
        .where(OvertimeRequest.tenant_id == user.tenant_id,
               OvertimeRequest.employee_id.in_(emp_ids),
               OvertimeRequest.status == "approved",
               OvertimeRequest.work_date >= m_start,
               OvertimeRequest.work_date <= m_end)
        .group_by(OvertimeRequest.employee_id)
    ):
        ot_rows[emp_id] = {"hours": float(hrs or 0), "comp_days": float(days or 0)}

    # --- 报销(批准,decided_at 落月)---
    exp_rows = dict(db.execute(
        select(ExpenseClaim.employee_id, func.coalesce(func.sum(ExpenseClaim.total), 0.0))
        .where(ExpenseClaim.tenant_id == user.tenant_id,
               ExpenseClaim.employee_id.in_(emp_ids),
               ExpenseClaim.status == "approved",
               func.date_trunc("month", ExpenseClaim.decided_at) == m_start)
        .group_by(ExpenseClaim.employee_id)
    ).all())

    # --- 排班天数 ---
    shift_rows = dict(db.execute(
        select(ShiftAssignment.employee_id, func.count())
        .where(ShiftAssignment.tenant_id == user.tenant_id,
               ShiftAssignment.employee_id.in_(emp_ids),
               ShiftAssignment.day >= m_start, ShiftAssignment.day <= m_end)
        .group_by(ShiftAssignment.employee_id)
    ).all())

    rows = []
    for e in emps:
        att = att_rows.get(e.id, {"by_status": {}, "hours": 0.0})
        by_status = att["by_status"]
        rows.append({
            "employee_id": e.id,
            "employee_no": e.employee_no,
            "name": e.name,
            "org": orgs.get(e.org_unit_id),
            "status": e.status,
            "workdays": workdays,
            "present_days": by_status.get("present", 0)
            + by_status.get("remote", 0),
            "leave_days": leave_days.get(e.id, 0.0),
            "leave_by_type": leave_by_type.get(e.id, {}),
            "absent_days": by_status.get("absent", 0),
            "work_hours": round(att["hours"], 2),
            "unexcused_flags": flag_rows.get(e.id, 0),
            "overtime_hours": ot_rows.get(e.id, {}).get("hours", 0.0),
            "overtime_comp_days": ot_rows.get(e.id, {}).get("comp_days", 0.0),
            "expense_total": float(exp_rows.get(e.id, 0.0) or 0),
            "shift_days": shift_rows.get(e.id, 0),
        })
    return {
        "month": month_key,
        "workdays": workdays,
        "total": len(rows),
        "rows": rows,
    }
