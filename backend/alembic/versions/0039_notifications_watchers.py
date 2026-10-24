"""notifications + ai_watchers(T3-0 通知与触发式任务基建)

Revision ID: 0039
Revises: 0038
Create Date: 2026-10-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(32), nullable=False, server_default="system"),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("body", sa.String(512), nullable=True),
        sa.Column("link", sa.String(256), nullable=True),
        sa.Column("read_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_notifications_user_unread", "notifications", ["user_id", "read_at"])
    op.create_table(
        "ai_watchers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("params", JSONB, nullable=False, server_default="{}"),
        sa.Column("last_checked_at", sa.DateTime, nullable=True),
        sa.Column("last_fired_at", sa.DateTime, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("ai_watchers")
    op.drop_index("ix_notifications_user_unread", table_name="notifications")
    op.drop_table("notifications")
