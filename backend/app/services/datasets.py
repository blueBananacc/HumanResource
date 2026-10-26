# -*- coding: utf-8 -*-
"""白名单数据集查询(E5)— NL 报表的确定性执行层。

LLM 只负责把自然语言映射到 ``{dataset, group_by, metrics, filters}``,
数据集/维度/指标全部白名单 —— 没有任何自由字段路径,SQL 由这里组装。

- ``GET /api/datasets`` 返回各数据集的可用维度/指标/过滤字段(供 LLM/UI 选择)
- ``POST /api/datasets/query`` 执行:行数封顶 200,无 group_by 时退化成
  单行总计
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Attendance,
    AttendanceFlag,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    LeaveApplication,
    LeaveType,
    OrgUnit,
    OvertimeRequest,
    ShiftAssignment,
    User,
)
from app.services.permissions import apply_org_filter, get_org_scope


def _month_bounds(v: str) -> tuple[date, date]:
    try:
        y, m = int(v[:4]), int(v[5:7])
        start = date(y, m, 1)
    except (ValueError, IndexError):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"month 须为 YYYY-MM:{v}") from None
    end = (start.replace(day=28) + timedelta(days=8)
           ).replace(day=1) - timedelta(days=1)
    return start, end


# dims 取值列:org 维度统一存 Employee.org_unit_id,输出时翻译 OrgUnit.name;
# 其余直接是模型列(employees 数据集 org 列就是它自己的 org_unit_id)
DATASETS: dict[str, dict[str, Any]] = {
    "employees": {
        "label": "员工",
        "model": Employee,
        "date_field": "hire_date",
        "dims": {
            "org": Employee.org_unit_id,
            "status": Employee.status,
        },
        "metrics": {"count": ("人数", None)},
        "filters": ("status", "org_unit_id"),
    },
    "leave_applications": {
        "label": "请假申请",
        "model": LeaveApplication,
        "date_field": "start_date",
        "dims": {
            "status": LeaveApplication.status,
            "leave_type": LeaveApplication.leave_type_id,
            "org": Employee.org_unit_id,
        },
        "metrics": {"count": ("笔数", None), "sum_days": ("天数", LeaveApplication.days)},
        "filters": ("status", "leave_type_id", "org_unit_id"),
        "join_employee": True,
    },
    "attendance": {
        "label": "考勤日汇总",
        "model": Attendance,
        "date_field": "day",
        "dims": {
            "status": Attendance.status,
            "org": Employee.org_unit_id,
        },
        "metrics": {
            "count": ("天数", None),
            "sum_hours": ("工时", Attendance.work_hours),
        },
        "filters": ("status", "org_unit_id"),
        "join_employee": True,
    },
    "attendance_flags": {
        "label": "考勤异常",
        "model": AttendanceFlag,
        "date_field": "day",
        "dims": {
            "kind": AttendanceFlag.kind,
            "excused": AttendanceFlag.excused,
            "org": Employee.org_unit_id,
        },
        "metrics": {"count": ("条数", None)},
        "filters": ("kind", "excused", "org_unit_id"),
        "join_employee": True,
    },
    "overtime_requests": {
        "label": "加班申请",
        "model": OvertimeRequest,
        "date_field": "work_date",
        "dims": {"status": OvertimeRequest.status, "org": Employee.org_unit_id},
        "metrics": {
            "count": ("笔数", None),
            "sum_hours": ("小时", OvertimeRequest.hours),
            "sum_comp_days": ("调休天数", OvertimeRequest.comp_days),
        },
        "filters": ("status", "org_unit_id"),
        "join_employee": True,
    },
    "expense_claims": {
        "label": "报销单",
        "model": ExpenseClaim,
        "date_field": "decided_at",
        "dims": {"status": ExpenseClaim.status, "org": Employee.org_unit_id},
        "metrics": {
            "count": ("笔数", None),
            "sum_total": ("金额", ExpenseClaim.total),
        },
        "filters": ("status", "org_unit_id"),
        "join_employee": True,
    },
    "shift_assignments": {
        "label": "排班",
        "model": ShiftAssignment,
        "date_field": "day",
        "dims": {
            "shift": ShiftAssignment.shift_id,
            "source": ShiftAssignment.source,
            "org": Employee.org_unit_id,
        },
        "metrics": {"count": ("班次天数", None)},
        "filters": ("source", "shift_id", "org_unit_id"),
        "join_employee": True,
    },
    "employee_requests": {
        "label": "员工申请",
        "model": EmployeeRequest,
        "date_field": "created_at",
        "dims": {"kind": EmployeeRequest.kind, "status": EmployeeRequest.status,
                 "org": Employee.org_unit_id},
        "metrics": {"count": ("笔数", None)},
        "filters": ("kind", "status", "org_unit_id"),
        "join_employee": True,
    },
}

_ROW_CAP = 200


def describe_datasets() -> dict:
    """给 LLM/UI 的可用维度/指标/过滤器清单。"""
    return {
        name: {
            "label": d["label"],
            "dims": list(d["dims"].keys()),
            "metrics": list(d["metrics"].keys()),
            "filters": list(d["filters"]) + ["month", "date_from", "date_to"],
        }
        for name, d in DATASETS.items()
    }


def query_dataset(db: Session, user: User, spec: dict) -> dict:
    name = spec.get("dataset")
    d = DATASETS.get(name)
    if d is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"未知数据集:{name}(可选:{','.join(DATASETS)})")
    model = d["model"]
    dims = spec.get("group_by") or []
    metrics = spec.get("metrics") or ["count"]
    filters = spec.get("filters") or {}

    for dim in dims:
        if dim not in d["dims"]:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{name} 不支持维度 {dim}(可选:{','.join(d['dims'])})")
    for m in metrics:
        if m not in d["metrics"]:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{name} 不支持指标 {m}(可选:{','.join(d['metrics'])})")
    for f in filters:
        base = f if f not in ("month", "date_from", "date_to") else None
        if base is None:
            continue
        if f not in d["filters"]:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{name} 不支持过滤 {f}(可选:{','.join(d['filters'])})")

    # 行级 org scope 先算:scope 非 None 时即使没选 org 维度也要 join
    # Employee 才能按下属 org 过滤
    scope = get_org_scope(db, user)
    need_org = "org" in dims or "org_unit_id" in filters or scope is not None
    # employees 数据集模型即 Employee,org 列就是自身的,无需 join
    join_emp = bool(d.get("join_employee")) and need_org

    # select 列表:维度列 + 指标表达式
    dim_cols = [d["dims"][x].label(f"dim_{i}") for i, x in enumerate(dims)]
    metric_cols = []
    for i, m in enumerate(metrics):
        label, col = d["metrics"][m]
        expr = func.count() if col is None else func.coalesce(func.sum(col), 0.0)
        metric_cols.append(expr.label(f"m_{i}"))

    stmt = select(*dim_cols, *metric_cols).select_from(model)
    if join_emp:
        stmt = stmt.join(Employee, Employee.id == model.employee_id)
    stmt = stmt.where(model.tenant_id == user.tenant_id)

    org_col = (model.org_unit_id if model is Employee
               else Employee.org_unit_id)
    stmt = apply_org_filter(stmt, org_column=org_col, scope=scope)

    # 过滤:month / date_from / date_to 作用在数据集日期字段;其余 eq 白名单列
    date_col = getattr(model, d["date_field"])
    if filters.get("month"):
        s, e = _month_bounds(str(filters["month"]))
        stmt = stmt.where(date_col >= s, date_col <= e)
    if filters.get("date_from"):
        stmt = stmt.where(date_col >= date.fromisoformat(str(filters["date_from"])))
    if filters.get("date_to"):
        stmt = stmt.where(date_col <= date.fromisoformat(str(filters["date_to"])))
    for f, v in filters.items():
        if f in ("month", "date_from", "date_to"):
            continue
        col = (Employee.org_unit_id if f == "org_unit_id" and model is not Employee
               else getattr(model, f))
        stmt = stmt.where(col.in_(v) if isinstance(v, list) else col == v)

    if dim_cols:
        stmt = stmt.group_by(*dim_cols).limit(_ROW_CAP + 1)

    rows = db.execute(stmt).all()
    truncated = len(rows) > _ROW_CAP
    rows = rows[:_ROW_CAP]

    # dim 值翻译成可读名:org → OrgUnit.name,leave_type → LeaveType.name,
    # shift → Shift.name
    dim_names: list[tuple[int, dict[Any, str] | None]] = []
    id_maps = _id_name_maps(db, user.tenant_id, dims, d)
    for i, dim in enumerate(dims):
        dim_names.append((i, id_maps.get(dim)))

    out_rows = []
    for r in rows:
        item: dict[str, Any] = {}
        for i, dim in enumerate(dims):
            raw = getattr(r, f"dim_{i}")
            m = dim_names[i][1]
            item[dim] = m.get(raw, raw) if m else raw
        for i, m in enumerate(metrics):
            item[m] = getattr(r, f"m_{i}")
        out_rows.append(item)

    return {
        "dataset": name,
        "label": d["label"],
        "dims": dims,
        "metrics": metrics,
        "rows": out_rows,
        "total_rows": len(out_rows),
        "truncated": truncated,
    }


def _id_name_maps(
    db: Session, tenant_id: str, dims: list[str], d: dict
) -> dict[str, dict[Any, str]]:
    """维度 id → 名称表(org/leave_type/shift 三类翻译)。"""
    from app.domain.models import Shift

    maps: dict[str, dict[Any, str]] = {}
    if "org" in dims or any(x == "org" for x in dims):
        maps["org"] = {o.id: o.name for o in db.scalars(
            select(OrgUnit).where(OrgUnit.tenant_id == tenant_id))}
    if "leave_type" in dims:
        maps["leave_type"] = {lt.id: lt.name for lt in db.scalars(
            select(LeaveType).where(LeaveType.tenant_id == tenant_id))}
    if "shift" in dims:
        maps["shift"] = {s.id: s.name for s in db.scalars(
            select(Shift).where(Shift.tenant_id == tenant_id))}
    return maps
