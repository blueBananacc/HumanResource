"""shift suite: rules + availability + templates + swap requests (喔趣 排班2.0)

Revision ID: 0078
Revises: 0077
Create Date: 2026-10-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0078"
down_revision = "0077"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shift_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, unique=True),
        sa.Column("max_consecutive_days", sa.Integer(),
                  nullable=False, server_default="6"),
        sa.Column("max_weekly_hours", sa.Float(),
                  nullable=False, server_default="44"),
        sa.Column("min_rest_minutes", sa.Integer(),
                  nullable=False, server_default="720"),
        sa.Column("respect_availability", sa.Boolean(),
                  nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "employee_availability",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("employee_id", sa.String(36),
                  sa.ForeignKey("employees.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("weekday", sa.Integer(), nullable=False),
        sa.Column("preference", sa.String(16),
                  nullable=False, server_default="available"),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "employee_id", "weekday",
                            name="uq_emp_avail_one"),
    )
    op.create_table(
        "schedule_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("period_days", sa.Integer(),
                  nullable=False, server_default="7"),
        sa.Column("pattern", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(),
                  nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "shift_swap_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("requester_employee_id", sa.String(36),
                  sa.ForeignKey("employees.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("target_employee_id", sa.String(36),
                  sa.ForeignKey("employees.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("status", sa.String(20), nullable=False,
                  server_default="pending_peer", index=True),
        sa.Column("requester_note", sa.Text(), nullable=True),
        sa.Column("peer_note", sa.Text(), nullable=True),
        sa.Column("approver_id", sa.String(36),
                  sa.ForeignKey("users.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("decision_note", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("shift_swap_requests")
    op.drop_table("schedule_templates")
    op.drop_table("employee_availability")
    op.drop_table("shift_rules")
