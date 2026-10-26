"""P1 SOP 任务链:recruit_runs + recruit_run_steps

Revision ID: 0068
Revises: 0067
"""
from alembic import op
import sqlalchemy as sa

revision = "0068"
down_revision = "0067"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recruit_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("kind", sa.String(40), nullable=False, server_default="full_pipeline"),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_recruit_runs_tenant_id", "recruit_runs", ["tenant_id"])
    op.create_index("ix_recruit_runs_status", "recruit_runs", ["status"])
    op.create_index("ix_recruit_runs_created_at", "recruit_runs", ["created_at"])

    op.create_table(
        "recruit_run_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("recruit_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(60), nullable=False),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("agent_label", sa.String(60), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("link", sa.String(300), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_recruit_run_steps_run_id", "recruit_run_steps", ["run_id"])
    op.create_index("ix_recruit_run_steps_status", "recruit_run_steps", ["status"])


def downgrade() -> None:
    op.drop_table("recruit_run_steps")
    op.drop_table("recruit_runs")
