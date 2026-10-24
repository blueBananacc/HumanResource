"""talent_campaigns + talent_campaign_sends(T3-7 人才再激活)

Revision ID: 0046
Revises: 0045
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0046"
down_revision = "0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "talent_campaigns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("filter", JSONB, nullable=False, server_default="{}"),
        sa.Column("subject", sa.String(128), nullable=False),
        sa.Column("body_text", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("sent_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("done_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "talent_campaign_sends",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("campaign_id", sa.String(36), sa.ForeignKey("talent_campaigns.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("to_email", sa.String(256), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("error", sa.String(256), nullable=True),
        sa.Column("sent_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("campaign_id", "candidate_id", name="uq_campaign_sends_one"),
    )


def downgrade() -> None:
    op.drop_table("talent_campaign_sends")
    op.drop_table("talent_campaigns")
