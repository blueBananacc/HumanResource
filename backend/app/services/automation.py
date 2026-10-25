"""事件驱动自动化 — 挂在 ``write_audit`` 漏斗上的租户级规则引擎。

数据流::

    业务写操作 → write_audit(result=success)
        └→ dispatch(db, log, actor)
             ├ 实体白名单 ENTITY_REGISTRY 过滤(未注册实体直接返回,
             │  audit_log/automation_rule/notification 等天然不触发)
             ├ 匹配 automation_rules(tenant, entity_type, trigger==audit.action,
             │   trigger 别名覆盖 import/bulk_* 等变体)
             ├ watch_fields:非空时与 detail.fields 键取交集
             ├ conditions:快照字段白名单条件,AND 语义
             ├ recipients:actor / role / user / record_owner / record_manager
             ├ notify → Notification 同事务落库
             │   email  → automation_runs 记 payload,after_commit → celery
             └ automation_runs 记账((rule_id, event_key=audit.id) 唯一 →
                 同一事件对同一规则幂等)

安全约束:
- dispatch 整体 try/except — 规则评估失败绝不影响业务事务
- ``suspend_automation(db)`` contextmanager:批量导入等场景防通知风暴
- 收件人/快照字段全部走注册表白名单,不支持任意 ORM 路径
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import event, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Announcement,
    Attendance,
    AttendanceRequest,
    AuditLog,
    AutomationRule,
    AutomationRun,
    Candidate,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    HrCase,
    Interview,
    InterviewScorecard,
    Job,
    JobApplication,
    JobOffer,
    LeaveApplication,
    Notification,
    OvertimeRequest,
    User,
)
from app.services.timeutil import utc_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 实体注册表:key = 审计 entity_type;每个实体声明可快照字段(条件/模板可用)、
# 关联员工/负责人解析器与前端链接。规则只能指向注册表内的实体和字段。
# ---------------------------------------------------------------------------


def _scalar(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _snapshot(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """从 ORM 对象取白名单字段成扁平 dict;已删除对象尽力而为。"""
    out: dict[str, Any] = {"id": getattr(obj, "id", None)}
    for f in fields:
        try:
            out[f] = _scalar(getattr(obj, f, None))
        except Exception:  # noqa: BLE001 - deleted/expired 对象属性可能刷新失败
            out[f] = None
    return out


def _emp_id(attr: str):
    return lambda obj: getattr(obj, attr, None)


def _owner(attr: str):
    return lambda obj: getattr(obj, attr, None)


# fields: 快照白名单(条件 field 校验也用它)
# employee_attr: 指向 employees.id 的属性(record_manager 解析用)
# owner_attr: 指向 users.id 的属性(record_owner 解析用)
ENTITY_REGISTRY: dict[str, dict[str, Any]] = {
    "candidate": {
        "model": Candidate,
        "fields": ("name", "status", "display_email", "display_phone", "tags",
                   "is_blacklisted", "org_unit_id"),
        "employee_attr": None,
        "owner_attr": "owner_user_id",
        "link": lambda o: f"/talents/{o.id}",
        "label": "候选人",
    },
    "job": {
        "model": Job,
        "fields": ("title", "status", "level", "publish_status", "org_unit_id"),
        "employee_attr": None,
        "owner_attr": None,
        "link": lambda o: f"/jobs/{o.id}",
        "label": "岗位",
    },
    "job_application": {
        "model": JobApplication,
        "fields": ("stage", "source", "candidate_id", "job_id", "rejected_reason"),
        "employee_attr": None,
        "owner_attr": "owner_user_id",
        "link": lambda o: f"/pipeline",
        "label": "管道申请",
    },
    "job_offer": {
        "model": JobOffer,
        "fields": ("status", "position", "candidate_id", "job_id", "start_date"),
        "employee_attr": "employee_id",
        "owner_attr": "created_by",
        "link": lambda o: "/offers",
        "label": "Offer",
    },
    "interview": {
        "model": Interview,
        "fields": ("status", "mode", "modality", "candidate_id", "job_id"),
        "employee_attr": None,
        "owner_attr": None,
        "link": lambda o: "/interviews",
        "label": "面试",
    },
    "employee": {
        "model": Employee,
        "fields": ("name", "employee_no", "status", "designation", "work_email",
                   "employment_type", "org_unit_id", "hire_date"),
        "employee_attr": "id",
        "owner_attr": "user_id",
        "link": lambda o: f"/employees/{o.id}",
        "label": "员工",
    },
    "leave_application": {
        "model": LeaveApplication,
        "fields": ("status", "start_date", "end_date", "days", "leave_type_id",
                   "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": "approver_id",
        "link": lambda o: "/leave",
        "label": "请假申请",
    },
    "employee_request": {
        "model": EmployeeRequest,
        "fields": ("kind", "subject", "status", "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": "decided_by",
        "link": lambda o: "/requests",
        "label": "员工申请",
    },
    "hr_case": {
        "model": HrCase,
        "fields": ("case_no", "subject", "kind", "priority", "status",
                   "requester_employee_id"),
        "employee_attr": "requester_employee_id",
        "owner_attr": "assignee_user_id",
        "link": lambda o: "/cases",
        "label": "HR 工单",
    },
    "attendance": {
        "model": Attendance,
        "fields": ("day", "status", "work_hours", "source", "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": None,
        "link": lambda o: "/attendance",
        "label": "考勤日结",
    },
    "attendance_request": {
        "model": AttendanceRequest,
        "fields": ("kind", "ts", "status", "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": "approver_id",
        "link": lambda o: "/attendance/requests",
        "label": "补卡申请",
    },
    "overtime_request": {
        "model": OvertimeRequest,
        "fields": ("work_date", "hours", "status", "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": "approver_id",
        "link": lambda o: "/overtime",
        "label": "加班申请",
    },
    "expense_claim": {
        "model": ExpenseClaim,
        "fields": ("title", "total", "status", "employee_id"),
        "employee_attr": "employee_id",
        "owner_attr": "approver_id",
        "link": lambda o: "/expenses",
        "label": "报销单",
    },
    "announcement": {
        "model": Announcement,
        "fields": ("title", "status", "audience_org_unit_id"),
        "employee_attr": None,
        "owner_attr": None,
        "link": lambda o: "/announcements",
        "label": "公告",
    },
    "interview_scorecard": {
        "model": InterviewScorecard,
        "fields": ("title", "status", "candidate_id", "job_id"),
        "employee_attr": None,
        "owner_attr": "created_by",
        "link": lambda o: f"/talents/{o.candidate_id}",
        "label": "面试评分卡",
    },
}

ENTITY_TYPES = tuple(ENTITY_REGISTRY.keys())

# trigger(存的就是审计 action 名)的等价组:create 规则同样命中批量导入产生的
# 行,update 覆盖 link/unlink/status:* 这类关联与状态变更,保持"规则即意图"的直觉。
TRIGGER_ALIASES: dict[str, set[str]] = {
    "create": {"create", "import", "bulk_create"},
    "update": {"update", "transition", "link", "unlink", "bulk_update",
               "status_change"},
    "delete": {"delete", "bulk_delete"},
}

TRIGGERS = tuple(TRIGGER_ALIASES.keys())

# trigger 是开放词表:create/update/delete 走等价组,其它审计动作名(approve /
# reject / status:xxx / bind_account…)走精确匹配。校验只约束形态不封死词表。
_TRIGGER_RE = re.compile(r"^[a-z0-9_:]{1,64}$")


def _action_matches(trigger: str, action: str) -> bool:
    """trigger 命中判定:精确匹配 → 等价组 → update 兼容 status:* 一族。"""
    if trigger == action:
        return True
    if action in TRIGGER_ALIASES.get(trigger, set()):
        return True
    if trigger == "update" and action.split(":", 1)[0] in (
            "status", "transition", "link", "unlink"):
        return True
    return False

CONDITION_OPS = ("eq", "ne", "gt", "lt", "ge", "le", "contains", "in", "exists")
RECIPIENT_KINDS = ("actor", "role", "user", "record_owner", "record_manager")
ACTIONS = ("notify", "email")


# ---------------------------------------------------------------------------
# 条件评估 / 模板渲染 / 收件人解析
# ---------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def eval_condition(cond: dict[str, Any], snap: dict[str, Any]) -> bool:
    field, op = cond.get("field"), cond.get("op")
    want = cond.get("value")
    got = snap.get(field)
    if op == "exists":
        return got is not None and got != ""
    if op in ("gt", "lt", "ge", "le"):
        a, b = _num(got), _num(want)
        if a is None or b is None:
            return False
        return {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b}[op]
    if op == "contains":
        if isinstance(got, (list, tuple)):
            return want in got
        return got is not None and str(want) in str(got)
    if op == "in":
        if isinstance(want, (list, tuple)):
            return got in want
        return got is not None and str(got) in [s.strip() for s in str(want).split(",")]
    if op == "eq":
        return (got is None and want is None) or str(got) == str(want) or got == want
    if op == "ne":
        return not eval_condition({"field": field, "op": "eq", "value": want}, snap)
    return False


def eval_conditions(conds: list[dict[str, Any]], snap: dict[str, Any]) -> bool:
    return all(eval_condition(c, snap) for c in (conds or []))


def render_tpl(tpl: str | None, snap: dict[str, Any], actor: User | None) -> str:
    """``{{field}}`` → 快照值;``{{actor}}`` → 操作者邮箱。未知键留空。"""
    if not tpl:
        return ""
    out = tpl
    for k, v in snap.items():
        out = out.replace("{{" + k + "}}", "" if v is None else str(v))
    out = out.replace("{{actor}}", actor.email if actor else "")
    return out


def _manager_user_ids(db: Session, employee_id: str | None) -> list[str]:
    """员工的直属上级对应的登录用户。"""
    if not employee_id:
        return []
    emp = db.get(Employee, employee_id)
    if emp is None or not emp.reports_to_id:
        return []
    mgr = db.get(Employee, emp.reports_to_id)
    return [mgr.user_id] if mgr and mgr.user_id else []


def resolve_recipients(
    db: Session, rule: AutomationRule, obj: Any, actor: User | None, spec: dict
) -> list[User]:
    """recipients JSON → User 列表(去重,排除 deactivated)。"""
    ids: set[str] = set()
    emp_attr = spec.get("employee_attr")
    owner_attr = spec.get("owner_attr")
    for r in rule.recipients or []:
        kind = r.get("kind")
        if kind == "actor" and actor is not None:
            ids.add(actor.id)
        elif kind == "user" and r.get("value"):
            ids.add(r["value"])
        elif kind == "role" and r.get("value"):
            ids.update(
                db.scalars(
                    select(User.id).where(
                        User.tenant_id == rule.tenant_id,
                        User.role == r["value"],
                        User.status == "active",
                    )
                ).all()
            )
        elif kind == "record_owner" and owner_attr and obj is not None:
            uid = getattr(obj, owner_attr, None)
            if uid:
                ids.add(uid)
        elif kind == "record_manager" and emp_attr and obj is not None:
            ids.update(_manager_user_ids(db, getattr(obj, emp_attr, None)))
    if not ids:
        return []
    return [
        u for u in db.scalars(
            select(User).where(User.id.in_(ids), User.status == "active")
        ).all()
        if u.tenant_id == rule.tenant_id
    ]


# ---------------------------------------------------------------------------
# 挂起 / commit 钩子 / 分发
# ---------------------------------------------------------------------------


@contextmanager
def suspend_automation(db: Session):
    """批量写场景挂起自动化分发(防通知风暴)。"""
    prev = db.info.get("automation_suspended", False)
    db.info["automation_suspended"] = True
    try:
        yield
    finally:
        db.info["automation_suspended"] = prev


def _register_commit_hook(db: Session) -> None:
    """每个 Session 只挂一次:after_commit 把排队中的 email run 交给 celery。"""
    if db.info.get("_automation_hooked"):
        return
    db.info["_automation_hooked"] = True

    @event.listens_for(db, "after_commit")
    def _flush_email_runs(session: Session) -> None:
        run_ids = session.info.pop("automation_email_runs", [])
        for rid in run_ids:
            try:
                from app.workers.tasks.automation import send_automation_email

                send_automation_email.delay(rid)
            except Exception:  # noqa: BLE001
                logger.exception("automation email run %s 派发失败", rid)


def _snapshot_for(db: Session, spec: dict, log: AuditLog) -> tuple[Any, dict[str, Any]]:
    obj = None
    try:
        obj = db.get(spec["model"], log.entity_id)
    except Exception:  # noqa: BLE001
        obj = None
    if obj is None:
        # 对象拿不到(删除后刷新失败等)——快照降级为 id + 审计 detail
        snap = {"id": log.entity_id}
        if isinstance(log.detail, dict):
            snap.update({k: _scalar(v) for k, v in log.detail.items()
                         if not isinstance(v, (dict, list))})
        return None, snap
    return obj, _snapshot(obj, spec["fields"])


def dispatch(db: Session, *, log: AuditLog, actor: User | None) -> int:
    """按一条审计日志分发自动化,返回执行的规则数。任何异常吞掉不炸业务。"""
    try:
        if db.info.get("automation_suspended"):
            return 0
        if log.result != "success":
            return 0
        spec = ENTITY_REGISTRY.get(log.entity_type)
        if spec is None:
            return 0
        rules = [
            r
            for r in db.scalars(
                select(AutomationRule).where(
                    AutomationRule.tenant_id == log.tenant_id,
                    AutomationRule.entity_type == log.entity_type,
                    AutomationRule.is_active.is_(True),
                )
            ).all()
            if _action_matches(r.trigger, log.action)
        ]
        if not rules:
            return 0
        obj, snap = _snapshot_for(db, spec, log)
        fired = 0
        for rule in rules:
            try:
                if _apply_rule(db, rule, spec, log, obj, snap, actor):
                    fired += 1
            except Exception:  # noqa: BLE001
                logger.exception(
                    "automation rule %s(%s)执行失败", rule.id, rule.name
                )
        return fired
    except Exception:  # noqa: BLE001
        logger.exception("automation dispatch 失败 entity=%s action=%s",
                         log.entity_type, log.action)
        return 0


def _apply_rule(
    db: Session,
    rule: AutomationRule,
    spec: dict,
    log: AuditLog,
    obj: Any,
    snap: dict[str, Any],
    actor: User | None,
) -> bool:
    """单规则评估+执行。幂等:(rule_id, event_key=audit.id) 已有 run 则跳过。"""
    # watch_fields:update 类事件限定变更字段交集;detail 无 fields 键按全变更处理
    if rule.watch_fields:
        fields = (log.detail or {}).get("fields")
        if isinstance(fields, dict) and not (set(fields) & set(rule.watch_fields)):
            return False
    if not eval_conditions(rule.conditions, snap):
        return False
    exists = db.scalars(
        select(AutomationRun.id).where(
            AutomationRun.rule_id == rule.id, AutomationRun.event_key == log.id
        )
    ).first()
    if exists:
        return False

    users = resolve_recipients(db, rule, obj, actor, spec)
    title = render_tpl(rule.title_tpl, snap, actor) or rule.name
    body = render_tpl(rule.body_tpl, snap, actor)
    link = spec["link"](obj) if obj is not None else None

    run = AutomationRun(
        tenant_id=rule.tenant_id,
        rule_id=rule.id,
        event_key=log.id,
        entity_type=log.entity_type,
        entity_id=log.entity_id,
        trigger=log.action,
        action=rule.action,
        recipients=[u.id for u in users],
        payload={"title": title, "body": body},
    )
    if not users:
        run.status = "skipped_no_recipients"
        db.add(run)
        return False
    if rule.action == "email":
        run.payload["to"] = [u.email for u in users if u.email]
        if not run.payload["to"]:
            run.status = "skipped_no_recipients"
            db.add(run)
            return False
        run.status = "queued_email"
        db.info.setdefault("automation_email_runs", [])
        if run.id not in db.info["automation_email_runs"]:
            db.add(run)
            db.flush()  # 需要 run.id 供 after_commit 任务回查
            db.info["automation_email_runs"].append(run.id)
        _register_commit_hook(db)
    else:
        for u in users:
            db.add(
                Notification(
                    tenant_id=rule.tenant_id,
                    user_id=u.id,
                    kind="automation",
                    title=title[:128],
                    body=(body or None) and body[:512],
                    link=link,
                )
            )
        run.status = "ok"
        db.add(run)
    rule.last_fired_at = utc_now()
    return True


# ---------------------------------------------------------------------------
# 规则校验(API 层复用)
# ---------------------------------------------------------------------------


def validate_rule_payload(
    entity_type: str,
    trigger: str,
    conditions: list[dict[str, Any]] | None,
    recipients: list[dict[str, Any]] | None,
    action: str,
) -> str | None:
    """返回错误消息字符串;None = 通过。"""
    spec = ENTITY_REGISTRY.get(entity_type)
    if spec is None:
        return f"不支持的实体:{entity_type}"
    if not trigger or not _TRIGGER_RE.match(trigger):
        return ("trigger 须为小写审计动作名(字母/数字/下划线/冒号,≤64);"
                "常用 create/update/delete,也可用 approve 等具体动作名")
    if action not in ACTIONS:
        return f"action 必须是 {ACTIONS} 之一"
    allowed_fields = set(spec["fields"]) | {"id"}
    for c in conditions or []:
        if c.get("field") not in allowed_fields:
            return f"条件字段 '{c.get('field')}' 不在 {entity_type} 白名单内"
        if c.get("op") not in CONDITION_OPS:
            return f"不支持的算子:{c.get('op')}"
    for r in recipients or []:
        if r.get("kind") not in RECIPIENT_KINDS:
            return f"不支持的收件人类型:{r.get('kind')}"
        if r.get("kind") in ("role", "user") and not r.get("value"):
            return f"收件人类型 {r.get('kind')} 需要 value"
    return None


def schema_meta() -> dict[str, Any]:
    """注册表元数据 — 前端表单和 sidecar 工具描述共用。"""
    return {
        "entities": {
            k: {
                "label": v["label"],
                "fields": list(v["fields"]) + ["id"],
                "recipient_kinds": [
                    k2 for k2 in RECIPIENT_KINDS
                    if k2 not in ("record_owner", "record_manager")
                    or (k2 == "record_owner" and v.get("owner_attr"))
                    or (k2 == "record_manager" and v.get("employee_attr"))
                ],
            }
            for k, v in ENTITY_REGISTRY.items()
        },
        "ops": list(CONDITION_OPS),
        "actions": list(ACTIONS),
        "triggers": list(TRIGGERS),
    }


__all__ = [
    "ENTITY_REGISTRY",
    "ENTITY_TYPES",
    "dispatch",
    "eval_conditions",
    "render_tpl",
    "resolve_recipients",
    "schema_meta",
    "suspend_automation",
    "validate_rule_payload",
]
