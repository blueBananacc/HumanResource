"""t7 automation + attendance deepening

Revision ID: 0052
Revises: 0051
Create Date: 2026-11-02

T7 事件驱动自动化 + 考勤深化:
- automation_rules / automation_runs — write_audit 漏斗上的规则引擎,
  (rule_id, event_key=audit_log.id) 唯一键保证同一事件对同一规则只执行一次
- attendance_policies — 每租户一行:宽限分钟/IP白名单/迟到早退开关/异常通知
- attendance_flags — 迟到/早退异常台账,(attendance_id, kind) 唯一
- work_calendars.work_start/work_end — 期望作息,迟到早退判定基准
- employee_checkins.ip — 打卡来源 IP 留痕
"""

from alembic import op
import sqlalchemy as sa

revision = "0052"
down_revision = "0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "automation_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("trigger", sa.String(64), nullable=False),
        sa.Column("watch_fields", sa.JSON(), nullable=True),
        sa.Column("conditions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("action", sa.String(16), nullable=False, server_default="notify"),
        sa.Column("recipients", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("title_tpl", sa.String(256), nullable=False),
        sa.Column("body_tpl", sa.String(1024), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_by",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_fired_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_automation_rules_tenant_id", "automation_rules", ["tenant_id"])
    op.create_index("ix_automation_rules_entity_type", "automation_rules", ["entity_type"])

    op.create_table(
        "automation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "rule_id",
            sa.String(36),
            sa.ForeignKey("automation_rules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_key", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(36), nullable=False),
        sa.Column("trigger", sa.String(64), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("recipients", sa.JSON(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("error", sa.String(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("rule_id", "event_key", name="uq_autorun_rule_event"),
    )
    op.create_index("ix_automation_runs_tenant_id", "automation_runs", ["tenant_id"])
    op.create_index("ix_automation_runs_rule_id", "automation_runs", ["rule_id"])

    op.create_table(
        "attendance_policies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("grace_in_minutes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("grace_out_minutes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("track_late_early", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("ip_check_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("allowed_ips", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("notify_on_flag", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_attendance_policies_tenant_id", "attendance_policies", ["tenant_id"])

    op.create_table(
        "attendance_flags",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "attendance_id",
            sa.String(36),
            sa.ForeignKey("attendances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "employee_id",
            sa.String(36),
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("excused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "excused_by_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("excused_note", sa.String(256), nullable=True),
        sa.Column("excused_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("attendance_id", "kind", name="uq_attflag_att_kind"),
    )
    op.create_index("ix_attendance_flags_tenant_id", "attendance_flags", ["tenant_id"])
    op.create_index("ix_attendance_flags_attendance_id", "attendance_flags", ["attendance_id"])
    op.create_index("ix_attendance_flags_employee_id", "attendance_flags", ["employee_id"])

    op.add_column(
        "work_calendars",
        sa.Column("work_start", sa.Time(), nullable=False, server_default="09:00:00"),
    )
    op.add_column(
        "work_calendars",
        sa.Column("work_end", sa.Time(), nullable=False, server_default="18:00:00"),
    )
    op.add_column(
        "employee_checkins",
        sa.Column("ip", sa.String(45), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("employee_checkins", "ip")
    op.drop_column("work_calendars", "work_end")
    op.drop_column("work_calendars", "work_start")
    op.drop_table("attendance_flags")
    op.drop_table("attendance_policies")
    op.drop_table("automation_runs")
    op.drop_table("automation_rules")
