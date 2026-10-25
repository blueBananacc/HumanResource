"""t10 production-grade idempotency keys

Revision ID: 0055
Revises: 0054
Create Date: 2026-10-26

T10:
- idempotency_keys — 可重试写操作的"一行一锁":
  (tenant_id, scope, key) 唯一元组即锁;request_hash 区分重放与键复用;
  in_progress/completed/failed 状态机;终态行存响应供回放;
  locked_at 死锁接管;expires_at 回放窗+定时清扫。
"""

from alembic import op
import sqlalchemy as sa

revision = "0055"
down_revision = "0054"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "idempotency_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("scope", sa.String(64), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="in_progress"
        ),
        sa.Column("response_status", sa.Integer, nullable=True),
        sa.Column("response_body", sa.JSON, nullable=True),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("locked_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "scope", "key", name="uq_idem_tenant_scope_key"
        ),
    )
    op.create_index("ix_idempotency_keys_tenant_id", "idempotency_keys", ["tenant_id"])
    op.create_index("ix_idempotency_keys_user_id", "idempotency_keys", ["user_id"])
    op.create_index("ix_idempotency_keys_scope", "idempotency_keys", ["scope"])
    op.create_index("ix_idempotency_keys_status", "idempotency_keys", ["status"])
    op.create_index("ix_idempotency_keys_expires_at", "idempotency_keys", ["expires_at"])


def downgrade() -> None:
    op.drop_table("idempotency_keys")
