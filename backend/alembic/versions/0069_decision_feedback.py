"""P3 记忆中枢:decision_feedback 决策埋点表

Revision ID: 0069
Revises: 0068
"""
from alembic import op
import sqlalchemy as sa

revision = "0069"
down_revision = "0068"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "decision_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_id", sa.String(36), sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("application_id", sa.String(36), sa.ForeignKey("job_applications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("from_stage", sa.String(16), nullable=True),
        sa.Column("to_stage", sa.String(16), nullable=False),
        sa.Column("match_score", sa.Integer(), nullable=True),
        sa.Column("criteria_snapshot", sa.JSON(), nullable=True),
        sa.Column("decided_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_decision_feedback_tenant_id", "decision_feedback", ["tenant_id"])
    op.create_index("ix_decision_feedback_job_id", "decision_feedback", ["job_id"])
    op.create_index("ix_decision_feedback_candidate_id", "decision_feedback", ["candidate_id"])
    op.create_index("ix_decision_feedback_application_id", "decision_feedback", ["application_id"])
    op.create_index("ix_decision_feedback_decision", "decision_feedback", ["decision"])
    op.create_index("ix_decision_feedback_created_at", "decision_feedback", ["created_at"])


def downgrade() -> None:
    op.drop_table("decision_feedback")
