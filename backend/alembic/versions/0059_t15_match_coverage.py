"""t15: keyword coverage sub-score on matches

Revision ID: 0059
Revises: 0058
Create Date: 2026-10-26

- resume_job_matches.coverage  — 确定性关键词覆盖 {score, hit, missing, total}
"""
from alembic import op
import sqlalchemy as sa


revision = "0059"
down_revision = "0058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resume_job_matches",
        sa.Column("coverage", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("resume_job_matches", "coverage")
