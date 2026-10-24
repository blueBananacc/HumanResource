"""t6 ai-native: prompt_templates.scope_module + doc_templates

Revision ID: 0051
Revises: 0050
Create Date: 2026-10-24

T6 AI 产品化:
- prompt_templates.scope_module — AIPromptTemplates.root_module,按模块隔离的提示词覆盖
- doc_templates — MailMerge 智能替代,AI 起草文档模板(内置注册表 + 租户覆盖/自定义)
"""

from alembic import op
import sqlalchemy as sa

revision = "0051"
down_revision = "0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "prompt_templates",
        sa.Column("scope_module", sa.String(48), nullable=False, server_default=""),
    )
    # 唯一键扩到 scope_module:同一 key 可并存 全局("") + 各模块覆盖
    op.drop_constraint(
        "uq_prompt_templates_tenant_key", "prompt_templates", type_="unique"
    )
    op.create_unique_constraint(
        "uq_prompt_templates_tenant_key",
        "prompt_templates",
        ["tenant_id", "key", "scope_module"],
    )

    op.create_table(
        "doc_templates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("entity", sa.String(32), nullable=False, server_default="any"),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "key", name="uq_doc_templates_tenant_key"),
    )
    op.create_index("ix_doc_templates_tenant_id", "doc_templates", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("doc_templates")
    op.drop_constraint(
        "uq_prompt_templates_tenant_key", "prompt_templates", type_="unique"
    )
    op.create_unique_constraint(
        "uq_prompt_templates_tenant_key", "prompt_templates", ["tenant_id", "key"]
    )
    op.drop_column("prompt_templates", "scope_module")
