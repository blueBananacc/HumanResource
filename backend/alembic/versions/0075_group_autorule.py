"""candidate group auto-enrollment rules

Revision ID: 0075
Revises: 0074
Create Date: 2026-10-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0075"
down_revision = "0074"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidate_groups",
        sa.Column("auto_rule", sa.JSON, nullable=True),
    )
    op.add_column(
        "candidate_group_members",
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
    )


def downgrade() -> None:
    op.drop_column("candidate_group_members", "source")
    op.drop_column("candidate_groups", "auto_rule")
