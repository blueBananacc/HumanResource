"""t12: gate + evidence-first scoring columns

Revision ID: 0057
Revises: 0056
Create Date: 2026-07-11

- jobs.gate_profile       — 惰性派生的门槛题+评分权重(按 JD 指纹缓存)
- resume_job_matches.gate       — 门槛判定明细(answers/disqualifiers)
- resume_job_matches.criteria   — 逐维 {evidence, level} 锚定评分
- resume_job_matches.red_flags  — 高权重维度 level=0 的红旗
"""
from alembic import op
import sqlalchemy as sa


revision = "0057"
down_revision = "0056"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("gate_profile", sa.JSON(), nullable=True))
    op.add_column("resume_job_matches", sa.Column("gate", sa.JSON(), nullable=True))
    op.add_column("resume_job_matches", sa.Column("criteria", sa.JSON(), nullable=True))
    op.add_column("resume_job_matches", sa.Column("red_flags", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("resume_job_matches", "red_flags")
    op.drop_column("resume_job_matches", "criteria")
    op.drop_column("resume_job_matches", "gate")
    op.drop_column("jobs", "gate_profile")
