"""audit_logs.via:标记操作来源(assistant/mcp)

Revision ID: 0049
Revises: 0048
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0049"
down_revision = "0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_logs", sa.Column("via", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("audit_logs", "via")
