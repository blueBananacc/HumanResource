"""知识检索日志(兜底回复标注 → 知识缺口盘点)

Revision ID: 0071
Revises: 0070
"""
from alembic import op
import sqlalchemy as sa

revision = "0071"
down_revision = "0070"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_query_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query", sa.String(512), nullable=False),
        sa.Column("hits_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="ui"),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_knowledge_query_logs_tenant_id", "knowledge_query_logs", ["tenant_id"])
    op.create_index("ix_knowledge_query_logs_created_at", "knowledge_query_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_query_logs_created_at", table_name="knowledge_query_logs")
    op.drop_index("ix_knowledge_query_logs_tenant_id", table_name="knowledge_query_logs")
    op.drop_table("knowledge_query_logs")
