"""workforce: 工作日历/节假日 + 补卡/加班申请 + 假期政策 + leave_types.is_comp

Revision ID: 0031
Revises: 0030
Create Date: 2026-10-26

对应 Frappe HRMS 的 Holiday List / Attendance Request / Overtime→Comp Leave /
Leave Policy 四个域:

- work_calendars      租户工作日历(每周工作日 + 日标准工时)
- holidays            节假日 / 调休补班日
- attendance_requests 补卡申请(批准 → 补写 checkin 并重算当日考勤)
- overtime_requests   加班申请(批准 → 折算调休额度写 allocation + 台账)
- leave_policies      假期政策(items JSON + 结转上限;assign/run 发放)
- employees.leave_policy_id / leave_types.is_comp 两个新列
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- 工作日历 / 节假日 --------------------------------------------------
    op.create_table(
        "work_calendars",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True, unique=True),
        sa.Column("workdays", sa.JSON(), nullable=False),
        sa.Column("workday_hours", sa.Float, nullable=False, server_default="8"),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "holidays",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False, server_default="holiday"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "day", name="uq_holidays_tenant_day"),
    )

    # ---- 补卡 / 加班申请 ----------------------------------------------------
    op.create_table(
        "attendance_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("approver_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("decided_at", sa.DateTime),
        sa.Column("decision_note", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "overtime_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("work_date", sa.Date(), nullable=False),
        sa.Column("hours", sa.Float, nullable=False),
        sa.Column("comp_days", sa.Float),
        sa.Column("leave_type_id", sa.String(36), sa.ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("approver_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("decided_at", sa.DateTime),
        sa.Column("decision_note", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ---- 假期政策 -----------------------------------------------------------
    op.create_table(
        "leave_policies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("carry_forward", sa.Float),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_leave_policies_tenant_name"),
    )
    op.add_column(
        "employees",
        sa.Column("leave_policy_id", sa.String(36), sa.ForeignKey("leave_policies.id", ondelete="SET NULL")),
    )
    op.add_column(
        "leave_types",
        sa.Column("is_comp", sa.Boolean, nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("leave_types", "is_comp")
    op.drop_column("employees", "leave_policy_id")
    op.drop_table("leave_policies")
    op.drop_table("overtime_requests")
    op.drop_table("attendance_requests")
    op.drop_table("holidays")
    op.drop_table("work_calendars")
