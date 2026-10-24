"""salary_bands(T3-3 薪酬带宽)

Revision ID: 0042
Revises: 0041
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0042"
down_revision = "0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "salary_bands",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("designation", sa.String(128), nullable=True),
        sa.Column("org_unit_id", sa.String(36), sa.ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True),
        sa.Column("currency", sa.String(8), nullable=False, server_default="CNY"),
        sa.Column("gross_min", sa.Integer, nullable=False),
        sa.Column("gross_max", sa.Integer, nullable=False),
        sa.Column("employer_cost_min", sa.Integer, nullable=True),
        sa.Column("employer_cost_max", sa.Integer, nullable=True),
        sa.Column("effective_from", sa.Date, nullable=True),
        sa.Column("effective_to", sa.Date, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_salary_bands_tenant_name"),
    )


def downgrade() -> None:
    op.drop_table("salary_bands")
