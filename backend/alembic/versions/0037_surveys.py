"""surveys + survey_questions + survey_responses + survey_response_answers

Revision ID: 0037
Revises: 0036
Create Date: 2026-10-23

答卷防重复:UNIQUE(survey_id, respondent_key)。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "surveys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft", index=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "survey_questions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("survey_id", sa.String(36), sa.ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("text", sa.String(256), nullable=False),
        sa.Column("options", JSONB, nullable=False, server_default="[]"),
        sa.Column("required", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("order", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_table(
        "survey_responses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("survey_id", sa.String(36), sa.ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("respondent_key", sa.String(64), nullable=False),
        sa.Column("respondent_name", sa.String(64), nullable=True),
        sa.Column("happiness", sa.Integer, nullable=True),
        sa.Column("submitted_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("survey_id", "respondent_key", name="uq_survey_resp_key"),
    )
    op.create_table(
        "survey_response_answers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("response_id", sa.String(36), sa.ForeignKey("survey_responses.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("question_id", sa.String(36), sa.ForeignKey("survey_questions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("value", JSONB, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("survey_response_answers")
    op.drop_table("survey_responses")
    op.drop_table("survey_questions")
    op.drop_table("surveys")
