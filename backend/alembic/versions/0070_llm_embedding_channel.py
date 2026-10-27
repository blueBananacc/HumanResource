"""LLMProvider 独立 embedding 通道:embedding_base_url + embedding_api_key_encrypted

embedding 服务可与 chat 不同厂商(如 chat=DeepSeek、embedding=HF/SiliconFlow),
两列均可空:留空时回退共用 base_url / api_key。

Revision ID: 0070
Revises: 0069
"""
from alembic import op
import sqlalchemy as sa

revision = "0070"
down_revision = "0069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_providers",
        sa.Column("embedding_base_url", sa.String(512), nullable=True),
    )
    op.add_column(
        "llm_providers",
        sa.Column("embedding_api_key_encrypted", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_providers", "embedding_api_key_encrypted")
    op.drop_column("llm_providers", "embedding_base_url")
