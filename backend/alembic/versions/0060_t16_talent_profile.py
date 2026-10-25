"""t16: talent profiles + sourcing runs

Revision ID: 0060
Revises: 0059
Create Date: 2026-10-26

- talent_profiles  可复用人才画像(spec:skills/keywords/min_years/education)
- sourcing_runs    人才库扫描留痕(审计 + 渠道效果统计)
"""
from alembic import op
import sqlalchemy as sa


revision = "0060"
down_revision = "0059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "talent_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("scoring_rule_id", sa.String(36), sa.ForeignKey("scoring_rules.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "sourcing_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("talent_profiles.id", ondelete="SET NULL"), nullable=True),
        sa.Column("spec_snapshot", sa.JSON(), nullable=True),
        sa.Column("scanned_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prefiltered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_scored", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("returned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("sourcing_runs")
    op.drop_table("talent_profiles")
