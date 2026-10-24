"""rooms + resources + reservations:会议室/资源预约

Revision ID: 0036
Revises: 0035
Create Date: 2026-10-23

冲突校验在 API 层
(同对象 + status=active + 时间重叠 → 409)。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rooms",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("org_unit_id", sa.String(36), sa.ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("location", sa.String(128), nullable=True),
        sa.Column("capacity", sa.Integer, nullable=False, server_default="4"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "resources",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("type", sa.String(16), nullable=False, server_default="other"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "reservations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(16), nullable=False, index=True),
        sa.Column("room_id", sa.String(36), sa.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("resource_id", sa.String(36), sa.ForeignKey("resources.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("interview_id", sa.String(36), sa.ForeignKey("interviews.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("starts_at", sa.DateTime, nullable=False),
        sa.Column("ends_at", sa.DateTime, nullable=False),
        sa.Column("interviewer_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active", index=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("reservations")
    op.drop_table("resources")
    op.drop_table("rooms")
