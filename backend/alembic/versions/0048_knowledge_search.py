"""knowledge:AI 剧本标记 + embedding 混合检索;pg_trgm;llm embedding_model

Revision ID: 0048
Revises: 0047
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0048"
down_revision = "0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.add_column("knowledge_articles", sa.Column("is_playbook", sa.Boolean, nullable=False, server_default="false"))
    op.add_column("knowledge_articles", sa.Column("embedding", JSONB, nullable=True))
    op.add_column("llm_providers", sa.Column("embedding_model", sa.String(128), nullable=True))
    op.execute(
        "CREATE INDEX ix_knowledge_articles_title_trgm "
        "ON knowledge_articles USING gin (title gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_knowledge_articles_content_trgm "
        "ON knowledge_articles USING gin (content gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_knowledge_articles_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_knowledge_articles_title_trgm")
    op.drop_column("llm_providers", "embedding_model")
    op.drop_column("knowledge_articles", "embedding")
    op.drop_column("knowledge_articles", "is_playbook")
