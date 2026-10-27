"""two-tier dedup flags + candidate lock + attendance confirmation

Revision ID: 0076
Revises: 0075
Create Date: 2026-10-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0076"
down_revision = "0075"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("candidates", sa.Column("dedup_flag", sa.String(16), nullable=True))
    op.add_column("candidates", sa.Column("dedup_peer_id", sa.String(36), nullable=True))
    op.add_column("candidates", sa.Column("locked_by_id", sa.String(36), nullable=True))
    op.add_column("candidates", sa.Column("locked_at", sa.DateTime, nullable=True))
    op.add_column("candidates", sa.Column("locked_until", sa.DateTime, nullable=True))
    op.create_index("ix_candidates_dedup_flag", "candidates", ["dedup_flag"])

    op.create_table(
        "attendance_confirmations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "employee_id",
            sa.String(36),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("month", sa.String(7), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("confirmed_by_id", sa.String(36), nullable=True),
        sa.Column("resolved_by_id", sa.String(36), nullable=True),
        sa.Column("resolved_note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("employee_id", "month", name="uq_attconfirm_emp_month"),
    )
    op.create_index("ix_attendance_confirmations_tenant", "attendance_confirmations", ["tenant_id"])
    op.create_index("ix_attendance_confirmations_employee", "attendance_confirmations", ["employee_id"])
    op.create_index("ix_attendance_confirmations_status", "attendance_confirmations", ["status"])


def downgrade() -> None:
    op.drop_table("attendance_confirmations")
    op.drop_index("ix_candidates_dedup_flag", table_name="candidates")
    op.drop_column("candidates", "locked_until")
    op.drop_column("candidates", "locked_at")
    op.drop_column("candidates", "locked_by_id")
    op.drop_column("candidates", "dedup_peer_id")
    op.drop_column("candidates", "dedup_flag")
