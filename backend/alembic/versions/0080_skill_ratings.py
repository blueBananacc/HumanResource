"""多评价者技能评分:skill_ratings / skill_rating_requests

Degreed 式多源评分 + iCIMS 反馈请求流:员工/经理/HR 发起评分请求,
同事打分落 skill_ratings(每人每技能一条)并同步 upsert
employee_skills(source=self/manager/peer, confirmed),卡片展示分歧。

Revision ID: 0080
Revises: 0079
"""
from alembic import op
import sqlalchemy as sa

revision = "0080"
down_revision = "0079"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "skill_rating_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("skill_id", sa.String(36), sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rater_employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rater_kind", sa.String(16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("rating_id", sa.String(36), nullable=True),
        sa.Column("requested_by_id", sa.String(36), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_srr_tenant", "skill_rating_requests", ["tenant_id"])
    op.create_index("ix_srr_emp", "skill_rating_requests", ["employee_id"])
    op.create_index("ix_srr_rater", "skill_rating_requests", ["rater_employee_id"])
    op.create_index("ix_srr_status", "skill_rating_requests", ["status"])

    op.create_table(
        "skill_ratings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("skill_id", sa.String(36), sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rater_employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rater_kind", sa.String(16), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(36),
                  sa.ForeignKey("skill_rating_requests.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "skill_id", "rater_employee_id",
                            name="uq_skill_ratings_rater"),
    )
    op.create_index("ix_skill_ratings_tenant", "skill_ratings", ["tenant_id"])
    op.create_index("ix_skill_ratings_emp", "skill_ratings", ["employee_id"])
    op.create_index("ix_skill_ratings_kind", "skill_ratings", ["rater_kind"])

    # rating_id 反向指回 rating(建表顺序限制,这里补 FK)
    op.create_foreign_key(
        "fk_srr_rating", "skill_rating_requests", "skill_ratings",
        ["rating_id"], ["id"], ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_srr_rating", "skill_rating_requests", type_="foreignkey")
    op.drop_table("skill_ratings")
    op.drop_table("skill_rating_requests")
