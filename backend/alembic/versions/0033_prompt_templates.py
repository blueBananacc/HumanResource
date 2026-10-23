"""prompt_templates + prompt_template_versions:租户级 AI 提示词模板

Revision ID: 0033
Revises: 0032
Create Date: 2026-10-23

参考 MintHCM AIPromptTemplates:按用途 key 管理模板,``{{var}}`` 占位。
没有自定义行时服务用代码内置默认;每次保存写版本表可回滚。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("key", sa.String(48), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("updated_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "key", name="uq_prompt_templates_tenant_key"),
    )
    op.create_table(
        "prompt_template_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("template_id", sa.String(36), sa.ForeignKey("prompt_templates.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("prompt_template_versions")
    op.drop_table("prompt_templates")
