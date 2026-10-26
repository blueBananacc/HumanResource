"""P0 文档引擎产物:generated_documents

Revision ID: 0067
Revises: 0066
"""
from alembic import op
import sqlalchemy as sa

revision = "0067"
down_revision = "0066"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generated_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("file_name", sa.String(200), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime", sa.String(120), nullable=False),
        sa.Column("storage_key", sa.String(400), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("created_by_user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_generated_documents_tenant_id", "generated_documents", ["tenant_id"])
    op.create_index("ix_generated_documents_kind", "generated_documents", ["kind"])
    op.create_index("ix_generated_documents_created_at", "generated_documents", ["created_at"])


def downgrade() -> None:
    op.drop_table("generated_documents")
