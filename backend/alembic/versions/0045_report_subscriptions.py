"""report_subscriptions(T3-6 定时报表)+ report.digest 邮件模板 seed

Revision ID: 0045
Revises: 0044
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0045"
down_revision = "0044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "report_subscriptions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("weekday", sa.Integer, nullable=False),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("last_sent_at", sa.DateTime, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "user_id", "kind", name="uq_report_subs_one"),
    )
    # report.digest 邮件模板 seed(各租户行:用 generate_series 方式对每个
    # 既有 tenant 插一份;新租户保存前仍回落 EMAIL_SPECS 默认值)
    op.execute(
        """
        INSERT INTO email_templates (id, tenant_id, key, subject, body_text, body_html, is_active, created_at, updated_at)
        SELECT gen_random_uuid()::text, t.id, 'report.digest',
               '{{digest_title}}', '{{digest_body}}', NULL, true, now(), now()
        FROM tenants t
        WHERE NOT EXISTS (
            SELECT 1 FROM email_templates et
            WHERE et.tenant_id = t.id AND et.key = 'report.digest'
        )
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM email_templates WHERE key = 'report.digest'")
    op.drop_table("report_subscriptions")
