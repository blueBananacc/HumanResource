"""hr_cases + hr_case_updates(T3-4 HR 工单)

Revision ID: 0043
Revises: 0042
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0043"
down_revision = "0042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hr_cases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("case_no", sa.Integer, nullable=False),
        sa.Column("subject", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False, server_default="other"),
        sa.Column("priority", sa.String(16), nullable=False, server_default="normal"),
        sa.Column("status", sa.String(16), nullable=False, server_default="new"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("requester_employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("assignee_user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("resolution", sa.Text, nullable=True),
        sa.Column("resolved_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "case_no", name="uq_hr_cases_tenant_no"),
    )
    op.create_table(
        "hr_case_updates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("case_id", sa.String(36), sa.ForeignKey("hr_cases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("author_user_id", sa.String(36), nullable=True),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("is_internal", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("hr_case_updates")
    op.drop_table("hr_cases")
