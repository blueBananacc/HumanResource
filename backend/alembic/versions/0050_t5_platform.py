"""t5 platform: saved_filters + favorites + dictionary_items + job_runs

Revision ID: 0050
Revises: 0049
Create Date: 2026-10-24

MintHCM 移植 T5 平台甜点:
- saved_filters    — SavedSearch,per-user 列表页命名筛选
- favorites        — 记录星标(多态 entity+entity_id)
- dictionary_items — Dictionaries,可配置下拉枚举
- job_runs         — SchedulersJobs,后台任务运行打点
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0050"
down_revision = "0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "saved_filters",
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
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("params", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "page", "name", name="uq_saved_filters_user_page_name"),
    )
    op.create_index("ix_saved_filters_tenant_id", "saved_filters", ["tenant_id"])
    op.create_index("ix_saved_filters_user_id", "saved_filters", ["user_id"])

    op.create_table(
        "favorites",
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
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "entity", "entity_id", name="uq_favorites_user_entity"),
    )
    op.create_index("ix_favorites_tenant_id", "favorites", ["tenant_id"])
    op.create_index("ix_favorites_user_id", "favorites", ["user_id"])
    op.create_index("ix_favorites_entity_id", "favorites", ["entity_id"])

    op.create_table(
        "dictionary_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("dict_key", sa.String(64), nullable=False),
        sa.Column("value", sa.String(64), nullable=False),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("sort", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "dict_key", "value", name="uq_dictionary_items_key_value"
        ),
    )
    op.create_index("ix_dictionary_items_tenant_id", "dictionary_items", ["tenant_id"])
    op.create_index("ix_dictionary_items_dict_key", "dictionary_items", ["dict_key"])

    op.create_table(
        "job_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("message", sa.String(256), nullable=True),
    )
    op.create_index("ix_job_runs_tenant_id", "job_runs", ["tenant_id"])
    op.create_index("ix_job_runs_name", "job_runs", ["name"])
    op.create_index("ix_job_runs_name_started", "job_runs", ["name", "started_at"])


def downgrade() -> None:
    op.drop_table("job_runs")
    op.drop_table("dictionary_items")
    op.drop_table("favorites")
    op.drop_table("saved_filters")
