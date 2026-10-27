"""talent review: nine-box assessments + talent pools

Revision ID: 0073_talent_review
Revises: 0072_skill_graph
Create Date: 2026-10-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0073"
down_revision = "0072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nine_box_assessments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("cycle", sa.String(32), nullable=False, server_default=""),
        sa.Column("auto_perf", sa.Float, nullable=True),
        sa.Column("auto_potential", sa.Float, nullable=True),
        sa.Column("perf_band", sa.Integer, nullable=False, server_default="2"),
        sa.Column("potential_band", sa.Integer, nullable=False, server_default="2"),
        sa.Column("calibrated", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("calibrated_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("calibrated_at", sa.DateTime, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_nine_box_assessments_tenant_id", "nine_box_assessments", ["tenant_id"])
    op.create_index("ix_nine_box_assessments_employee_id", "nine_box_assessments", ["employee_id"])

    op.create_table(
        "talent_pools",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False, server_default="custom"),
        sa.Column("target_designation", sa.String(128), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_talent_pools_tenant_id", "talent_pools", ["tenant_id"])

    op.create_table(
        "talent_pool_members",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pool_id", sa.String(36), sa.ForeignKey("talent_pools.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(16), nullable=False, server_default="in"),
        sa.Column("readiness", sa.Integer, nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("added_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("exited_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_talent_pool_members_tenant_id", "talent_pool_members", ["tenant_id"])
    op.create_index("ix_talent_pool_members_pool_id", "talent_pool_members", ["pool_id"])
    op.create_index("ix_talent_pool_members_employee_id", "talent_pool_members", ["employee_id"])


def downgrade() -> None:
    op.drop_table("talent_pool_members")
    op.drop_table("talent_pools")
    op.drop_table("nine_box_assessments")
