"""salary slip employee confirmation (薪人薪事 工资条签收)

Revision ID: 0077
Revises: 0076
Create Date: 2026-10-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0077"
down_revision = "0076"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "salary_slips",
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "salary_slips",
        sa.Column(
            "confirmed_by_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("salary_slips", "confirmed_by_id")
    op.drop_column("salary_slips", "confirmed_at")
