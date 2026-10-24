"""knowledge_articles(T3-5 知识库,assistant RAG 语料)

Revision ID: 0044
Revises: 0043
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0044"
down_revision = "0043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_articles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("category", sa.String(32), nullable=False, server_default="other"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tags", JSONB, nullable=False, server_default="[]"),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("updated_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("knowledge_articles")
