"""P 批次人才画像:employee_skills / projects / project_members / insight_notes

Revision ID: 0063
Revises: 0062
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0063"
down_revision = "0062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_skills",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("skill_id", sa.String(36), sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("source", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("status", sa.String(16), nullable=False, server_default="suggested"),
        sa.Column("evidence", JSONB, nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "skill_id", name="uq_employee_skills_pair"),
    )
    op.create_index("ix_employee_skills_tenant", "employee_skills", ["tenant_id"])
    op.create_index("ix_employee_skills_emp", "employee_skills", ["employee_id"])
    op.create_index("ix_employee_skills_source", "employee_skills", ["source"])
    op.create_index("ix_employee_skills_status", "employee_skills", ["status"])

    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("owner_employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("required_skills", JSONB, nullable=False, server_default="[]"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_projects_tenant", "projects", ["tenant_id"])
    op.create_index("ix_projects_status", "projects", ["status"])

    op.create_table(
        "project_members",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(64), nullable=True),
        sa.Column("allocation_pct", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("joined_via", sa.String(16), nullable=False, server_default="manual"),
        sa.Column("contribution_summary", sa.Text(), nullable=True),
        sa.Column("joined_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("left_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("project_id", "employee_id", name="uq_project_members_pair"),
    )
    op.create_index("ix_project_members_tenant", "project_members", ["tenant_id"])
    op.create_index("ix_project_members_project", "project_members", ["project_id"])
    op.create_index("ix_project_members_emp", "project_members", ["employee_id"])

    op.create_table(
        "insight_notes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", sa.String(36), sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False, server_default="manager_note"),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("author_user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_ref", JSONB, nullable=True),
        sa.Column("visibility", sa.String(24), nullable=False, server_default="manager_only"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_insight_notes_tenant", "insight_notes", ["tenant_id"])
    op.create_index("ix_insight_notes_emp", "insight_notes", ["employee_id"])
    op.create_index("ix_insight_notes_kind", "insight_notes", ["kind"])


def downgrade() -> None:
    op.drop_table("insight_notes")
    op.drop_table("project_members")
    op.drop_table("projects")
    op.drop_table("employee_skills")
