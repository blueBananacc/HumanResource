"""employee_requests + request_comments(T3-2 员工申请枢纽)

Revision ID: 0041
Revises: 0040
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0041"
down_revision = "0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(32), nullable=False, server_default="general"),
        sa.Column("subject", sa.String(128), nullable=False),
        sa.Column("detail", sa.Text, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("decided_by", sa.String(36), nullable=True),
        sa.Column("decided_at", sa.DateTime, nullable=True),
        sa.Column("decision_note", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "request_comments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("request_id", sa.String(36), sa.ForeignKey("employee_requests.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("author_user_id", sa.String(36), nullable=True),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("request_comments")
    op.drop_table("employee_requests")
