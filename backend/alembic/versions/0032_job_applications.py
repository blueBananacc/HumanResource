"""job_applications + job_application_events:申请管道卡与变迁历史

Revision ID: 0032
Revises: 0031
Create Date: 2026-10-23

参考 MintHCM Candidatures(状态机 + Kanban),收敛为
applied/screening/interview/offer/hired/rejected/withdrawn 七态。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_applications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("stage", sa.String(16), nullable=False, server_default="applied", index=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("owner_user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("rejected_reason", sa.Text, nullable=True),
        sa.Column("stage_changed_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "candidate_id", "job_id", name="uq_applications_tenant_cand_job"),
    )
    op.create_table(
        "job_application_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("application_id", sa.String(36), sa.ForeignKey("job_applications.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("from_stage", sa.String(16), nullable=True),
        sa.Column("to_stage", sa.String(16), nullable=False),
        sa.Column("actor_user_id", sa.String(36), nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("job_application_events")
    op.drop_table("job_applications")
