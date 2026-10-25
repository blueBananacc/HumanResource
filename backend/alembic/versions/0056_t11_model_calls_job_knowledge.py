"""t11 model call logs + job knowledge chunks

Revision ID: 0056
Revises: 0055
Create Date: 2026-10-25

T11:
- model_call_logs:逐次 LLM 调用留痕(call_type/provider/model/prompt 版本/
  input digest/归一化输出/原始输出/错误分类/延迟),独立 session 写入,
  业务回滚不丢;租户级隔离 + ops API 供排障
- job_knowledge_sources / job_knowledge_chunks:岗位级知识源(text/file),
  抽取→切块(<=40)→best-effort embedding,喂 match/question/interview prompt
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0056"
down_revision = "0055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_call_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("call_type", sa.String(48), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False, server_default=""),
        sa.Column("model", sa.String(128), nullable=False, server_default=""),
        sa.Column("prompt_key", sa.String(64), nullable=True),
        sa.Column("prompt_version", sa.String(32), nullable=True),
        sa.Column("schema_version", sa.String(16), nullable=False, server_default="v1"),
        sa.Column("input_digest", sa.String(16), nullable=False, server_default=""),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("error_kind", sa.String(32), nullable=False, server_default=""),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_model_call_logs_tenant_created", "model_call_logs", ["tenant_id", "created_at"]
    )
    op.create_index(
        "ix_model_call_logs_tenant_type", "model_call_logs", ["tenant_id", "call_type"]
    )
    op.create_index("ix_model_call_logs_tenant_id", "model_call_logs", ["tenant_id"])
    op.create_index("ix_model_call_logs_user_id", "model_call_logs", ["user_id"])
    op.create_index("ix_model_call_logs_created_at", "model_call_logs", ["created_at"])

    op.create_table(
        "job_knowledge_sources",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(8), nullable=False, server_default="text"),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("file_name", sa.String(255), nullable=True),
        sa.Column("mime", sa.String(128), nullable=True),
        sa.Column("storage_key", sa.String(512), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="indexing"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "extraction_status", sa.String(16), nullable=False, server_default="none"
        ),
        sa.Column("extraction_method", sa.String(32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_job_knowledge_sources_tenant_id", "job_knowledge_sources", ["tenant_id"]
    )
    op.create_index("ix_job_knowledge_sources_job_id", "job_knowledge_sources", ["job_id"])
    op.create_index("ix_job_knowledge_sources_status", "job_knowledge_sources", ["status"])

    op.create_table(
        "job_knowledge_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_id", sa.String(36), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "embedding",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_jk_chunks_job_src_idx",
        "job_knowledge_chunks",
        ["job_id", "source_id", "chunk_index"],
    )
    op.create_index("ix_job_knowledge_chunks_job_id", "job_knowledge_chunks", ["job_id"])
    op.create_index(
        "ix_job_knowledge_chunks_source_id", "job_knowledge_chunks", ["source_id"]
    )


def downgrade() -> None:
    op.drop_table("job_knowledge_chunks")
    op.drop_table("job_knowledge_sources")
    op.drop_table("model_call_logs")
