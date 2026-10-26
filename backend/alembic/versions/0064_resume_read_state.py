"""概览简历已读标记:resume_read_states(按用户维度)

Revision ID: 0064
Revises: 0063
"""
from alembic import op
import sqlalchemy as sa

revision = "0064"
down_revision = "0063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resume_read_states",
        sa.Column("resume_id", sa.String(36), sa.ForeignKey("resumes.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("read_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_resume_read_states_tenant", "resume_read_states", ["tenant_id"])
    op.create_index("ix_resume_read_states_user", "resume_read_states", ["user_id"])


def downgrade() -> None:
    op.drop_table("resume_read_states")
