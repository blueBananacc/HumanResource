"""E3 审批中心:approval_policies / approval_instances / approval_steps
+ users.delegate_user_id(审批代理人)

Revision ID: 0062
Revises: 0061
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0062"
down_revision = "0061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "approval_policies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("request_type", sa.String(32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("conditions", JSONB, nullable=False, server_default="[]"),
        sa.Column("steps", JSONB, nullable=False, server_default="[]"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_approval_policies_tenant", "approval_policies", ["tenant_id"])
    op.create_index("ix_approval_policies_type", "approval_policies", ["request_type"])

    op.create_table(
        "approval_instances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_type", sa.String(32), nullable=False),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("policy_id", sa.String(36), sa.ForeignKey("approval_policies.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("current_step_idx", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("tenant_id", "request_type", "request_id", name="uq_approval_inst_req"),
    )
    op.create_index("ix_approval_inst_tenant", "approval_instances", ["tenant_id"])
    op.create_index("ix_approval_inst_reqid", "approval_instances", ["request_id"])
    op.create_index("ix_approval_inst_status", "approval_instances", ["status"])

    op.create_table(
        "approval_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("instance_id", sa.String(36), sa.ForeignKey("approval_instances.id", ondelete="CASCADE"), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("resolver", sa.String(16), nullable=False),
        sa.Column("resolver_key", sa.String(64), nullable=True),
        sa.Column("label", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("decided_by_id", sa.String(36), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.UniqueConstraint("instance_id", "idx", name="uq_approval_step_idx"),
    )
    op.create_index("ix_approval_steps_tenant", "approval_steps", ["tenant_id"])
    op.create_index("ix_approval_steps_inst", "approval_steps", ["instance_id"])

    op.add_column("users", sa.Column(
        "delegate_user_id", sa.String(36),
        sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    ))


def downgrade() -> None:
    op.drop_column("users", "delegate_user_id")
    op.drop_table("approval_steps")
    op.drop_table("approval_instances")
    op.drop_table("approval_policies")
