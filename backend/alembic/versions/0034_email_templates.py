"""email_templates:租户级邮件模板(subject + text + html,{{var}} 占位)

Revision ID: 0034
Revises: 0033
Create Date: 2026-10-23

参考 SuiteCRM EmailTemplates。无自定义行时 worker 回落内置文案;
``app.services.mail_templates`` 提供注册表与渲染。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("key", sa.String(48), nullable=False),
        sa.Column("subject", sa.String(256), nullable=False),
        sa.Column("body_text", sa.Text, nullable=False),
        sa.Column("body_html", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("updated_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "key", name="uq_email_templates_tenant_key"),
    )


def downgrade() -> None:
    op.drop_table("email_templates")
