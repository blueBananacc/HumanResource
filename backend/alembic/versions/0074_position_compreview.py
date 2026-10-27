"""position management + comp review decisions

Revision ID: 0074
Revises: 0073
Create Date: 2026-10-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0074"
down_revision = "0073"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "positions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("designation", sa.String(128), nullable=True),
        sa.Column("org_unit_id", sa.String(36), sa.ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True),
        sa.Column("salary_band_id", sa.String(36), sa.ForeignKey("salary_bands.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("incumbent_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True),
        sa.Column("opened_at", sa.DateTime, nullable=True),
        sa.Column("filled_at", sa.DateTime, nullable=True),
        sa.Column("closed_at", sa.DateTime, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_positions_tenant_id", "positions", ["tenant_id"])
    op.create_index("ix_positions_org_unit_id", "positions", ["org_unit_id"])
    op.create_index("ix_positions_status", "positions", ["status"])

    op.add_column(
        "employees",
        sa.Column("position_id", sa.String(36), sa.ForeignKey("positions.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_employees_position_id", "employees", ["position_id"])

    op.create_table(
        "comp_review_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("cycle", sa.String(32), nullable=False),
        sa.Column("perf_band", sa.Integer, nullable=False, server_default="2"),
        sa.Column("ratio_band", sa.String(8), nullable=False, server_default="mid"),
        sa.Column("suggested_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("final_pct", sa.Float, nullable=True),
        sa.Column("final_amount", sa.Float, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("decided_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "employee_id", "cycle", name="uq_comp_review_emp_cycle"),
    )
    op.create_index("ix_comp_review_decisions_tenant_id", "comp_review_decisions", ["tenant_id"])
    op.create_index("ix_comp_review_decisions_employee_id", "comp_review_decisions", ["employee_id"])


def downgrade() -> None:
    op.drop_table("comp_review_decisions")
    op.drop_index("ix_employees_position_id", "employees")
    op.drop_column("employees", "position_id")
    op.drop_table("positions")
