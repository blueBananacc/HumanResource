"""drop interviews.verify_phone_last4

Revision ID: 0028
Revises: 0027
Create Date: 2026-10-21

候选人手机末 4 位身份核对下线:链接本身就是唯一凭证(invite_token_hash),
不再要求候选人额外输入手机末 4 位。/verify 端点退化为"直接签发 session token"。
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("interviews", "verify_phone_last4")


def downgrade() -> None:
    op.add_column(
        "interviews",
        sa.Column("verify_phone_last4", sa.String(4), nullable=True),
    )
