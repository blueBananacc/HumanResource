"""resume_job_matches.suggestion_dismissed_at

Revision ID: 0029
Revises: 0028
Create Date: 2026-10-21

面试页"推荐面试"区:score >= 90 的匹配自动生成待确认面试项,HR 点"忽略"
后写入本字段,该匹配不再出现。
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resume_job_matches",
        sa.Column("suggestion_dismissed_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("resume_job_matches", "suggestion_dismissed_at")
