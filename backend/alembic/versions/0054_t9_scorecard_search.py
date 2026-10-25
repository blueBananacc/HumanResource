"""t9 interview scorecards + declarative global search

Revision ID: 0054
Revises: 0053
Create Date: 2026-10-26

T9:
- interview_scorecards / interview_feedbacks — 人类面试官的结构化评分卡
  (逐维评分 + 录用表态 + AI 聚合),与 AI Interview 互补
- search_index_definitions / search_documents — 声明式全局搜索:
  定义是租户可覆盖的数据,文档是可丢弃的投影(entity_id 不带外键)
"""

from alembic import op
import sqlalchemy as sa

revision = "0054"
down_revision = "0053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_scorecards",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "candidate_id",
            sa.String(36),
            sa.ForeignKey("candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "interview_id",
            sa.String(36),
            sa.ForeignKey("interviews.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("criteria", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("interviewer_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("scheduled_at", sa.DateTime(), nullable=True),
        sa.Column("location", sa.String(256), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("ai_summary", sa.JSON(), nullable=True),
        sa.Column(
            "created_by",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_sc_tenant", "interview_scorecards", ["tenant_id"])
    op.create_index("ix_sc_candidate", "interview_scorecards", ["candidate_id"])
    op.create_index("ix_sc_job", "interview_scorecards", ["job_id"])
    op.create_index("ix_sc_status", "interview_scorecards", ["status"])

    op.create_table(
        "interview_feedbacks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scorecard_id",
            sa.String(36),
            sa.ForeignKey("interview_scorecards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "interviewer_user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.String(16), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "scorecard_id", "interviewer_user_id",
            name="uq_interview_feedback_card_user"
        ),
    )
    op.create_index("ix_fb_tenant", "interview_feedbacks", ["tenant_id"])
    op.create_index("ix_fb_scorecard", "interview_feedbacks", ["scorecard_id"])

    op.create_table(
        "search_index_definitions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity", sa.String(64), nullable=False),
        sa.Column("label", sa.String(128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("fields", sa.JSON(), nullable=True),
        sa.Column("title_template", sa.String(512), nullable=True),
        sa.Column("body_template", sa.String(1024), nullable=True),
        sa.Column("keyword_fields", sa.JSON(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "entity", name="uq_search_def_tenant_entity"),
    )
    op.create_index("ix_sdef_tenant", "search_index_definitions", ["tenant_id"])

    op.create_table(
        "search_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(512), nullable=False, server_default=""),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("keywords", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("source_updated_at", sa.DateTime(), nullable=True),
        sa.Column("definition_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("indexed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "entity", "entity_id", name="uq_search_doc_entity_row"
        ),
    )
    op.create_index("ix_sdoc_tenant", "search_documents", ["tenant_id"])
    op.create_index("ix_sdoc_entity", "search_documents", ["entity"])


def downgrade() -> None:
    op.drop_table("search_documents")
    op.drop_table("search_index_definitions")
    op.drop_table("interview_feedbacks")
    op.drop_table("interview_scorecards")
