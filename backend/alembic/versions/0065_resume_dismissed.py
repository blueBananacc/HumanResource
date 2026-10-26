"""简历消息"仅移除"标记:resume_read_states.dismissed_at

Revision ID: 0065
Revises: 0064
"""
from alembic import op
import sqlalchemy as sa

revision = "0065"
down_revision = "0064"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("resume_read_states", sa.Column("dismissed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("resume_read_states", "dismissed_at")
