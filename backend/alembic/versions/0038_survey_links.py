"""survey_links(一次性公开链接)+ survey_responses 绑定 + rooms.equipment

Revision ID: 0038
Revises: 0037
Create Date: 2026-10-24

审查修复:
- 公开问卷改为一次性 token 链接(sha256 入库,可绑候选人、有过期时间),
  不再允许直接凭 survey id 匿名访问。
- rooms 补 equipment 字段(方案里就有,一期遗漏)。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "survey_links",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("survey_id", sa.String(36), sa.ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("used_at", sa.DateTime, nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("token_hash", name="uq_survey_links_token"),
    )
    op.create_index("ix_survey_links_token_hash", "survey_links", ["token_hash"])
    op.add_column(
        "survey_responses",
        sa.Column("link_id", sa.String(36), sa.ForeignKey("survey_links.id", ondelete="SET NULL"), nullable=True),
    )
    op.add_column(
        "survey_responses",
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True),
    )
    op.add_column(
        "rooms",
        sa.Column("equipment", JSONB, nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("rooms", "equipment")
    op.drop_column("survey_responses", "candidate_id")
    op.drop_column("survey_responses", "link_id")
    op.drop_index("ix_survey_links_token_hash", table_name="survey_links")
    op.drop_table("survey_links")
