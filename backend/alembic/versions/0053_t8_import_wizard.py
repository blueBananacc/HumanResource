"""t8 ai import wizard

Revision ID: 0053
Revises: 0052
Create Date: 2026-10-25

T8 AI 导入向导:
- import_jobs — 一次文件导入的全生命周期:原始文件字节 + 列名 +
  AI/启发式映射 + 校验结果 + 执行汇总;check/run 时按需重解析
"""

from alembic import op
import sqlalchemy as sa

revision = "0053"
down_revision = "0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "import_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("dataset", sa.String(32), nullable=False),
        sa.Column("file_name", sa.String(256), nullable=False),
        sa.Column("file_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("mapping", sa.JSON(), nullable=True),
        sa.Column("mapping_meta", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="uploaded"),
        sa.Column("check_result", sa.JSON(), nullable=True),
        sa.Column("run_summary", sa.JSON(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_import_jobs_tenant_id", "import_jobs", ["tenant_id"])
    op.create_index("ix_import_jobs_dataset", "import_jobs", ["dataset"])


def downgrade() -> None:
    op.drop_table("import_jobs")
