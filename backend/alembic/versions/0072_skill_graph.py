"""技能图谱:skills 加 parent_id(层级边)+ aliases(同义归并)

Revision ID: 0072
Revises: 0071
"""
from alembic import op
import sqlalchemy as sa

revision = "0072"
down_revision = "0071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column(
            "parent_id",
            sa.String(36),
            sa.ForeignKey("skills.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("skills", sa.Column("aliases", sa.JSON(), nullable=True))
    op.create_index("ix_skills_parent_id", "skills", ["parent_id"])


def downgrade() -> None:
    op.drop_index("ix_skills_parent_id", table_name="skills")
    op.drop_column("skills", "aliases")
    op.drop_column("skills", "parent_id")
