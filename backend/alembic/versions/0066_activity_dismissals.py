"""动态流"仅移除此条"标记:activity_dismissals

Revision ID: 0066
Revises: 0065
"""
from alembic import op
import sqlalchemy as sa

revision = "0066"
down_revision = "0065"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "activity_dismissals",
        sa.Column("item_key", sa.String(80), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dismissed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("activity_dismissals")
