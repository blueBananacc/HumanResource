"""automation: date_field trigger + create_case action; approval auto-approve

Rippling-inspired: 规则引擎从"仅事件触发"扩展为"事件+定时(日期字段)
双触发",动作新增 create_case;审批策略增加 auto_approve 条件组 ——
命中即提交即批准(自动裁决),实例打 auto_approved 标记。

Revision ID: 0079
Revises: 0078
Create Date: 2026-10-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0079"
down_revision = "0078"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "automation_rules",
        sa.Column("trigger_kind", sa.String(16), nullable=False,
                  server_default="audit"),
    )
    op.add_column(
        "automation_rules",
        sa.Column("schedule", sa.JSON(), nullable=True),
    )
    op.add_column(
        "automation_rules",
        sa.Column("action_config", sa.JSON(), nullable=True),
    )
    op.add_column(
        "approval_policies",
        sa.Column("auto_approve", sa.JSON(), nullable=True),
    )
    op.add_column(
        "approval_instances",
        sa.Column("auto_approved", sa.Boolean(), nullable=False,
                  server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("approval_instances", "auto_approved")
    op.drop_column("approval_policies", "auto_approve")
    op.drop_column("automation_rules", "action_config")
    op.drop_column("automation_rules", "schedule")
    op.drop_column("automation_rules", "trigger_kind")
