"""hrms suite: 员工核心 + 假勤 + 生命周期 + 需求审批/Offer + 绩效/费用/工资条

Revision ID: 0030
Revises: 0029
Create Date: 2026-10-26

HRIS 领域模型(docs/design/hrms-suite-design.md)。一次建全部表:
- employees                  员工主数据(核心,挂 user/org/candidate)
- leave_types / leave_allocations / leave_applications / leave_ledger_entries
- employee_checkins / attendances
- lifecycle_templates / lifecycle_events / lifecycle_tasks
- job_requisitions / job_offers
- goals / appraisal_cycles / appraisals
- expense_types / expense_claims / expense_items
- salary_slips
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- 员工核心 ----------------------------------------------------------
    op.create_table(
        "employees",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_no", sa.String(32), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("org_unit_id", sa.String(36), sa.ForeignKey("org_units.id", ondelete="SET NULL"), index=True),
        sa.Column("designation", sa.String(128)),
        sa.Column("reports_to_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="SET NULL"), index=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), unique=True),
        sa.Column("work_email", sa.String(256), index=True),
        sa.Column("phone", sa.String(64)),
        sa.Column("hire_date", sa.Date()),
        sa.Column("employment_type", sa.String(32), nullable=False, server_default="full_time"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active", index=True),
        sa.Column("personal", sa.JSON()),
        sa.Column("source_candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "employee_no", name="uq_employees_tenant_no"),
    )

    # ---- 假期 --------------------------------------------------------------
    op.create_table(
        "leave_types",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("annual_days", sa.Float, nullable=False, server_default="0"),
        sa.Column("is_paid", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_leave_types_tenant_name"),
    )
    op.create_table(
        "leave_allocations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("leave_type_id", sa.String(36), sa.ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("days", sa.Float, nullable=False),
        sa.Column("note", sa.String(256)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "leave_type_id", "year", name="uq_leave_alloc_emp_type_year"),
    )
    op.create_table(
        "leave_applications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("leave_type_id", sa.String(36), sa.ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_date", sa.Date, nullable=False),
        sa.Column("end_date", sa.Date, nullable=False),
        sa.Column("days", sa.Float, nullable=False),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("approver_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("decided_at", sa.DateTime),
        sa.Column("decision_note", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "leave_ledger_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("leave_type_id", sa.String(36), sa.ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("delta_days", sa.Float, nullable=False),
        sa.Column("ref_type", sa.String(32), nullable=False),
        sa.Column("ref_id", sa.String(36), nullable=False),
        sa.Column("note", sa.String(256)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ---- 考勤 --------------------------------------------------------------
    op.create_table(
        "employee_checkins",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ts", sa.DateTime, nullable=False, index=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("method", sa.String(16), nullable=False, server_default="web"),
        sa.Column("note", sa.String(256)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "attendances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("day", sa.Date, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="present", index=True),
        sa.Column("first_in", sa.DateTime),
        sa.Column("last_out", sa.DateTime),
        sa.Column("work_hours", sa.Float),
        sa.Column("source", sa.String(16), nullable=False, server_default="checkin"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "day", name="uq_attendance_emp_day"),
    )

    # ---- 生命周期 ----------------------------------------------------------
    op.create_table(
        "lifecycle_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("tasks", sa.JSON, nullable=False),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "lifecycle_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(24), nullable=False, index=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="in_progress", index=True),
        sa.Column("effective_date", sa.Date),
        sa.Column("payload", sa.JSON),
        sa.Column("note", sa.Text),
        sa.Column("created_by", sa.String(36)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime),
    )
    op.create_table(
        "lifecycle_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("event_id", sa.String(36), sa.ForeignKey("lifecycle_events.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("done", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("done_at", sa.DateTime),
        sa.Column("sort", sa.Integer, nullable=False, server_default="0"),
    )

    # ---- 招聘借力 -----------------------------------------------------------
    op.create_table(
        "job_requisitions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("org_unit_id", sa.String(36), sa.ForeignKey("org_units.id", ondelete="SET NULL")),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("headcount", sa.Integer, nullable=False, server_default="1"),
        sa.Column("level", sa.String(32), nullable=False, server_default="intermediate"),
        sa.Column("reason", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("requested_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("approver_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("decided_at", sa.DateTime),
        sa.Column("decision_note", sa.Text),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "job_offers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="SET NULL")),
        sa.Column("position", sa.String(128)),
        sa.Column("salary", sa.JSON),
        sa.Column("start_date", sa.Date),
        sa.Column("letter", sa.Text),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft", index=True),
        sa.Column("sent_at", sa.DateTime),
        sa.Column("responded_at", sa.DateTime),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="SET NULL")),
        sa.Column("created_by", sa.String(36)),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # ---- 绩效 ---------------------------------------------------------------
    op.create_table(
        "goals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("due_date", sa.Date),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active", index=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "appraisal_cycles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("period_start", sa.Date, nullable=False),
        sa.Column("period_end", sa.Date, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft", index=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "appraisals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("cycle_id", sa.String(36), sa.ForeignKey("appraisal_cycles.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("self_review", sa.Text),
        sa.Column("self_rating", sa.Integer),
        sa.Column("manager_review", sa.Text),
        sa.Column("manager_rating", sa.Integer),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft", index=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("cycle_id", "employee_id", name="uq_appraisal_cycle_emp"),
    )

    # ---- 费用 ---------------------------------------------------------------
    op.create_table(
        "expense_types",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_expense_types_tenant_name"),
    )
    op.create_table(
        "expense_claims",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("total", sa.Float, nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("approver_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("decided_at", sa.DateTime),
        sa.Column("decision_note", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "expense_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("claim_id", sa.String(36), sa.ForeignKey("expense_claims.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("type_id", sa.String(36), sa.ForeignKey("expense_types.id", ondelete="SET NULL")),
        sa.Column("occurred_on", sa.Date, nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("note", sa.String(256)),
    )

    # ---- 工资条(轻量登记) ----------------------------------------------------
    op.create_table(
        "salary_slips",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("period", sa.String(7), nullable=False),
        sa.Column("gross", sa.Float, nullable=False, server_default="0"),
        sa.Column("deductions", sa.Float, nullable=False, server_default="0"),
        sa.Column("net", sa.Float, nullable=False, server_default="0"),
        sa.Column("currency", sa.String(8), nullable=False, server_default="CNY"),
        sa.Column("items", sa.JSON),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft", index=True),
        sa.Column("issued_at", sa.DateTime),
        sa.Column("note", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "period", name="uq_salary_slip_emp_period"),
    )


def downgrade() -> None:
    for t in (
        "salary_slips", "expense_items", "expense_claims", "expense_types",
        "appraisals", "appraisal_cycles", "goals",
        "job_offers", "job_requisitions",
        "lifecycle_tasks", "lifecycle_events", "lifecycle_templates",
        "attendances", "employee_checkins",
        "leave_ledger_entries", "leave_applications", "leave_allocations", "leave_types",
        "employees",
    ):
        op.drop_table(t)
