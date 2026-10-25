"""h3:排班域 shifts + shift_assignments

Revision ID: 0061_h3_shifts
Revises: 0060_t16_talent_profile
Create Date: 2026-02-26
"""
from alembic import op
import sqlalchemy as sa

revision = "0061"
down_revision = "0060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shifts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("start_min", sa.Integer, nullable=False, server_default="540"),
        sa.Column("end_min", sa.Integer, nullable=False, server_default="1020"),
        sa.Column("required", sa.Integer, nullable=False, server_default="1"),
        sa.Column("color", sa.String(16), nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_shifts_tenant_name"),
    )
    op.create_table(
        "shift_assignments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("employee_id", sa.String(36),
                  sa.ForeignKey("employees.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("shift_id", sa.String(36),
                  sa.ForeignKey("shifts.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("day", sa.Date, nullable=False, index=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "employee_id", "day",
                            name="uq_shiftassign_emp_day"),
    )


def downgrade() -> None:
    op.drop_table("shift_assignments")
    op.drop_table("shifts")
