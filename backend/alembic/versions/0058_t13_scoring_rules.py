"""t13: declarative scoring rule engine

Revision ID: 0058
Revises: 0057
Create Date: 2026-10-26

- scoring_rules                — 租户级声明式评分规则(维度×评估器)
- jobs.scoring_rule_id         — 岗位挂规则(NULL → 走 T12 五维路径)
- resume_job_matches.rule_snapshot — 评估时冻结的规则快照(可回溯)
"""
from alembic import op
import sqlalchemy as sa


revision = "0058"
down_revision = "0057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scoring_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.add_column(
        "jobs",
        sa.Column("scoring_rule_id", sa.String(36), nullable=True),
    )
    op.create_foreign_key(
        "fk_jobs_scoring_rule_id",
        "jobs",
        "scoring_rules",
        ["scoring_rule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column(
        "resume_job_matches",
        sa.Column("rule_snapshot", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("resume_job_matches", "rule_snapshot")
    op.drop_constraint("fk_jobs_scoring_rule_id", "jobs", type_="foreignkey")
    op.drop_column("jobs", "scoring_rule_id")
    op.drop_table("scoring_rules")
