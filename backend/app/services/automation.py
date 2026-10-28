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
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.models import (
    Announcement,
    Attendance,
    AttendanceRequest,
    AuditLog,
    AutomationRule,
    AutomationRun,
    Candidate,
    CandidateGroup,
    CandidateGroupMember,
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
    Skill,
    SkillRatingRequest,
    Survey,
    SurveyLink,
    User,
)
from app.infra.invite_tokens import generate_invite_token, hash_invite_token
from app.services.timeutil import local_now, utc_now

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


# 员工 personal JSON 里的日期字段 —— date_field 触发器与条件的取值键
EMP_PERSONAL_DATE_KEYS = ("birth_date", "probation_end", "contract_end",
                        "cert_expire")


# fields: 快照白名单(条件 field 校验也用它)
# employee_attr: 指向 employees.id 的属性(record_manager 解析用)
# owner_attr: 指向 users.id 的属性(record_owner 解析用)
# candidate_attr: 指向 candidates.id 的属性(候选人动作解析用;"id"=实体本身)
# extras: (db, obj) → 额外模板键(候选人/岗位名等,不进条件白名单)


def _job_app_extras(db: Session, o: JobApplication) -> dict[str, Any]:
    """管道申请的模板上下文:候选人名/邮箱 + 岗位标题。"""
    out: dict[str, Any] = {}
    try:
        if o.candidate_id:
            cand = db.get(Candidate, o.candidate_id)
            if cand is not None:
                out["candidate_name"] = cand.name
                out["candidate_email"] = cand.display_email
    except Exception:  # noqa: BLE001
        pass
    try:
        if o.job_id:
            job = db.get(Job, o.job_id)
            if job is not None:
                out["job_title"] = job.title
    except Exception:  # noqa: BLE001
        pass
    return out


def _skill_rating_req_extras(db: Session, o: SkillRatingRequest) -> dict[str, Any]:
    """技能评分请求的模板上下文:技能名 + 被评人/评价人姓名。"""
    out: dict[str, Any] = {}
    try:
        sk = db.get(Skill, o.skill_id)
        if sk is not None:
            out["skill_name"] = sk.name
    except Exception:  # noqa: BLE001
        pass
    try:
        ratee = db.get(Employee, o.employee_id)
        if ratee is not None:
            out["ratee_name"] = ratee.name
    except Exception:  # noqa: BLE001
        pass
    try:
        rater = db.get(Employee, o.rater_employee_id)
        if rater is not None:
            out["rater_name"] = rater.name
    except Exception:  # noqa: BLE001
        pass
    return out


ENTITY_REGISTRY: dict[str, dict[str, Any]] = {
    "candidate": {
        "model": Candidate,
        "fields": ("name", "status", "display_email", "display_phone", "tags",
                   "is_blacklisted", "org_unit_id"),
        "employee_attr": None,
        "owner_attr": "owner_user_id",
        "candidate_attr": "id",
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
        "candidate_attr": "candidate_id",
        "extras": _job_app_extras,
        "link": lambda o: f"/pipeline",
        "label": "管道申请",
    },
    "job_offer": {
        "model": JobOffer,
        "fields": ("status", "position", "candidate_id", "job_id", "start_date"),
        "employee_attr": "employee_id",
        "owner_attr": "created_by",
        "candidate_attr": "candidate_id",
        "link": lambda o: "/offers",
        "label": "Offer",
    },
    "interview": {
        "model": Interview,
        "fields": ("status", "mode", "modality", "candidate_id", "job_id"),
        "employee_attr": None,
        "owner_attr": None,
        "candidate_attr": "candidate_id",
        "link": lambda o: "/interviews",
        "label": "面试",
    },
    "employee": {
        "model": Employee,
        # personal JSON 里的日期键也进快照(条件可引用,值由 extras 补齐)
        "fields": ("name", "employee_no", "status", "designation", "work_email",
                   "employment_type", "org_unit_id", "hire_date",
                   *EMP_PERSONAL_DATE_KEYS),
        "extras": lambda db, o: {
            k: _scalar((o.personal or {}).get(k))
            for k in EMP_PERSONAL_DATE_KEYS
        },
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
        "candidate_attr": "candidate_id",
        "link": lambda o: f"/talents/{o.candidate_id}",
        "label": "面试评分卡",
    },
    "skill_rating_request": {
        "model": SkillRatingRequest,
        "fields": ("status", "skill_id", "employee_id", "rater_employee_id",
                   "rater_kind"),
        "employee_attr": "employee_id",
        "owner_attr": None,
        "extras": _skill_rating_req_extras,
        "link": lambda o: "/me/profile",
        "label": "技能评分请求",
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
ACTIONS = ("notify", "email", "create_case",
           "email_candidate", "add_to_group", "send_survey")

# 作用于「实体关联候选人」的动作 —— Teamtailor 阶段触发器的主力动作:
# email_candidate 给候选人发邮件;add_to_group 进人才分组;send_survey 发问卷链接。
# 这些动作不走 recipients(收件人是候选人),要求实体注册表带 candidate_attr。
CANDIDATE_ACTIONS = ("email_candidate", "add_to_group", "send_survey")

# trigger_kind:audit = 审计事件(默认);date_field = 员工日期字段到达
TRIGGER_KINDS = ("audit", "date_field")

# date_field 可监听的员工日期字段
DATE_FIELDS = {
    "hire_date": "入职日期",
    "birth_date": "生日",
    "probation_end": "转正日期",
    "contract_end": "合同到期",
    "cert_expire": "证件到期",
}

CASE_PRIORITIES = ("low", "normal", "high", "urgent")


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
    snap = _snapshot(obj, spec["fields"])
    extra = spec.get("extras")
    if extra is not None:
        snap.update(extra(db, obj))
    # 审计 detail 里的标量也进模板上下文(from/to_stage 等迁移语境)
    if isinstance(log.detail, dict):
        for k, v in log.detail.items():
            if k not in snap and not isinstance(v, (dict, list)):
                snap[k] = _scalar(v)
    return obj, snap


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
                    AutomationRule.trigger_kind == "audit",
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
    return _fire(
        db, rule, spec, obj, snap, actor,
        event_key=log.id, entity_type=log.entity_type,
        entity_id=log.entity_id, trigger_label=log.action,
    )


def _fire(
    db: Session,
    rule: AutomationRule,
    spec: dict,
    obj: Any,
    snap: dict[str, Any],
    actor: User | None,
    *,
    event_key: str,
    entity_type: str,
    entity_id: str,
    trigger_label: str,
) -> bool:
    """条件评估 → 幂等检查 → 动作执行 → 记账。audit 事件与定时扫描共用。"""
    if not eval_conditions(rule.conditions, snap):
        return False
    exists = db.scalars(
        select(AutomationRun.id).where(
            AutomationRun.rule_id == rule.id, AutomationRun.event_key == event_key
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
        event_key=event_key,
        entity_type=entity_type,
        entity_id=entity_id,
        trigger=trigger_label,
        action=rule.action,
        recipients=[u.id for u in users],
        payload={"title": title, "body": body},
    )
    if rule.action == "create_case":
        _exec_create_case(db, rule, spec, obj, snap, actor, run,
                          title, body, users)
    elif rule.action in CANDIDATE_ACTIONS:
        _exec_candidate_action(db, rule, spec, obj, run, body)
    elif not users:
        run.status = "skipped_no_recipients"
        db.add(run)
        return False
    elif rule.action == "email":
        to = [u.email for u in users if u.email]
        if not to:
            run.status = "skipped_no_recipients"
            db.add(run)
            return False
        _queue_email_run(db, run, to)
    else:
        _notify_users(db, rule, users, title, body, link)
        run.status = "ok"
        db.add(run)
    rule.last_fired_at = utc_now()
    return True


def _queue_email_run(db: Session, run: AutomationRun, to: list[str]) -> None:
    """把 run 挂为 queued_email,after_commit 交 celery 投递。"""
    run.payload["to"] = to
    run.status = "queued_email"
    db.info.setdefault("automation_email_runs", [])
    if run.id not in db.info["automation_email_runs"]:
        db.add(run)
        db.flush()  # 需要 run.id 供 after_commit 任务回查
        db.info["automation_email_runs"].append(run.id)
    _register_commit_hook(db)


def _public_base() -> str:
    """候选人侧公开链接的站点根(public_base_url → cors 兜底)。"""
    if settings.public_base_url:
        return settings.public_base_url.rstrip("/")
    return (settings.cors_origins[0] if settings.cors_origins else "").rstrip("/")


def _resolve_candidate(
    db: Session, spec: dict, obj: Any, tenant_id: str
) -> Candidate | None:
    """按注册表 candidate_attr 从快照实体解析候选人(租户校验)。"""
    attr = spec.get("candidate_attr")
    if attr is None or obj is None:
        return None
    cid = getattr(obj, attr, None)
    if not cid:
        return None
    cand = db.get(Candidate, cid)
    return cand if cand is not None and cand.tenant_id == tenant_id else None


def _exec_candidate_action(
    db: Session,
    rule: AutomationRule,
    spec: dict,
    obj: Any,
    run: AutomationRun,
    body: str,
) -> None:
    """候选人动作:email_candidate / add_to_group / send_survey。

    Teamtailor 阶段触发器主力动作 —— 收件人/目标是从实体解析出的候选人,
    不走 recipients。失败只写 run.error,不抛回业务事务。
    """
    cand = _resolve_candidate(db, spec, obj, rule.tenant_id)
    if cand is None:
        run.status = "error"
        run.error = "candidate_not_found"
        db.add(run)
        return
    cfg = rule.action_config or {}
    run.payload["candidate_id"] = cand.id
    run.payload["candidate_name"] = cand.name

    if rule.action == "add_to_group":
        gid = cfg.get("group_id")
        grp = db.get(CandidateGroup, gid) if gid else None
        if grp is None or grp.tenant_id != rule.tenant_id:
            run.status = "error"
            run.error = "group_not_found"
            db.add(run)
            return
        exists = db.scalars(
            select(CandidateGroupMember.id).where(
                CandidateGroupMember.group_id == gid,
                CandidateGroupMember.candidate_id == cand.id,
            )
        ).first()
        if exists:
            run.payload["already_member"] = True
        else:
            db.add(CandidateGroupMember(
                group_id=gid, candidate_id=cand.id,
                added_by=rule.created_by, source="auto",
            ))
        run.payload["group_id"] = gid
        run.payload["group_name"] = grp.name
        run.status = "ok"
        db.add(run)
        return

    if rule.action == "send_survey":
        sid = cfg.get("survey_id")
        survey = db.get(Survey, sid) if sid else None
        if survey is None or survey.tenant_id != rule.tenant_id:
            run.status = "error"
            run.error = "survey_not_found"
            db.add(run)
            return
        if survey.status != "published":
            run.status = "error"
            run.error = "survey_not_published"
            db.add(run)
            return
        days = cfg.get("expires_in_days", 14)
        try:
            days = max(1, min(90, int(days)))
        except (TypeError, ValueError):
            days = 14
        token = generate_invite_token()
        link = SurveyLink(
            tenant_id=rule.tenant_id, survey_id=survey.id,
            token_hash=hash_invite_token(token),
            candidate_id=cand.id,
            expires_at=utc_now() + timedelta(days=days),
            created_by=rule.created_by,
        )
        db.add(link)
        db.flush()
        url = f"{_public_base()}/s/{token}"
        run.payload["link_id"] = link.id
        run.payload["survey_id"] = survey.id
        run.payload["survey_title"] = survey.title
        run.payload["body"] = f"{body}\n\n问卷链接:{url}".strip()
    # email_candidate / send_survey:有邮箱则走 queued_email。
    # send_survey 无邮箱仍 ok(链接已建,可线下转交);email_candidate 记跳过。
    if cand.display_email:
        _queue_email_run(db, run, [cand.display_email])
    else:
        run.payload["email_skipped"] = True
        run.status = ("ok" if rule.action == "send_survey"
                      else "skipped_no_recipients")
        db.add(run)


def _notify_users(
    db: Session, rule: AutomationRule, users: list[User],
    title: str, body: str, link: str | None,
) -> None:
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


def _exec_create_case(
    db: Session,
    rule: AutomationRule,
    spec: dict,
    obj: Any,
    snap: dict[str, Any],
    actor: User | None,
    run: AutomationRun,
    title: str,
    body: str,
    users: list[User],
) -> None:
    """create_case 动作:自动建 HR 工单(转人工处理通道)。

    - ``subject_tpl``/``description_tpl`` 走同一套 ``{{field}}`` 模板
    - ``requester_employee_id`` 取快照实体的员工关联(员工规则=员工本人)
    - ``assignee``:role → 该角色首个活跃用户;user → 指定账号
    - 收件人(recipients)若配置,同时收到一条工单通知
    """
    cfg = rule.action_config or {}
    kind = cfg.get("kind") if cfg.get("kind") in HrCase.KINDS else "other"
    priority = (cfg.get("priority")
                if cfg.get("priority") in CASE_PRIORITIES else "normal")
    subject = render_tpl(cfg.get("subject_tpl"), snap, actor) or title
    desc = render_tpl(cfg.get("description_tpl"), snap, actor) or body

    emp_id = None
    if spec.get("employee_attr") and obj is not None:
        emp_id = getattr(obj, spec["employee_attr"], None)

    assignee_id = None
    ak, av = cfg.get("assignee_kind"), cfg.get("assignee_value")
    if ak == "user" and av:
        u = db.get(User, av)
        if u and u.tenant_id == rule.tenant_id and u.status == "active":
            assignee_id = u.id
    elif ak == "role" and av:
        assignee_id = db.scalars(
            select(User.id).where(
                User.tenant_id == rule.tenant_id, User.role == av,
                User.status == "active",
            ).order_by(User.created_at)
        ).first()

    case_no = (db.scalar(
        select(func.max(HrCase.case_no)).where(
            HrCase.tenant_id == rule.tenant_id)
    ) or 0) + 1
    case = HrCase(
        tenant_id=rule.tenant_id, case_no=case_no,
        subject=subject[:128], kind=kind, priority=priority,
        status="assigned" if assignee_id else "new",
        description=desc or None,
        requester_employee_id=emp_id,
        assignee_user_id=assignee_id,
    )
    db.add(case)
    db.flush()
    run.payload["case_id"] = case.id
    run.payload["case_no"] = case_no

    notified = {u.id for u in users}
    _notify_users(db, rule, users, title, body, "/cases")
    if assignee_id and assignee_id not in notified:
        _notify_users(db, rule, [db.get(User, assignee_id)],
                      f"自动化工单 HR-{case_no}:{case.subject}",
                      "由自动化规则创建,请跟进处理", "/cases")
    run.status = "ok"
    db.add(run)


# ---------------------------------------------------------------------------
# 定时触发(date_field)— 每日 sweep,watch_loop 调用
# ---------------------------------------------------------------------------


def _rule_date_value(emp: Employee, field: str) -> date | None:
    """员工日期字段取值:hire_date 是列,其余在 personal JSON。"""
    raw = emp.hire_date if field == "hire_date" else (emp.personal or {}).get(field)
    if not raw:
        return None
    try:
        return raw if isinstance(raw, date) else date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _run_rule_today(db: Session, rule: AutomationRule, today: date) -> int:
    """单条 date_field 规则对全租户员工评估执行,返回触发次数。"""
    spec = ENTITY_REGISTRY["employee"]
    sched = rule.schedule or {}
    field, offset = sched.get("field"), sched.get("offset_days") or 0
    if field not in DATE_FIELDS:
        return 0
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return 0
    recurring = bool(sched.get("recurring"))
    fired = 0
    for emp in db.scalars(
        select(Employee).where(
            Employee.tenant_id == rule.tenant_id,
            Employee.status != "terminated",
        )
    ).all():
        base = _rule_date_value(emp, field)
        if base is None:
            continue
        if recurring:
            try:
                base = base.replace(year=today.year)
            except ValueError:  # 2/29 → 平年 2/28
                base = base.replace(year=today.year, day=28)
        if base + timedelta(days=offset) != today:
            continue
        snap = _snapshot(emp, spec["fields"])
        extra = spec.get("extras")
        if extra is not None:
            snap.update(extra(db, emp))
        try:
            if _fire(
                db, rule, spec, emp, snap, None,
                # event_key 与 rule_id 联合唯一,本身只需区分「谁哪天」
                event_key=f"s:{emp.id}:{today.isoformat()}",
                entity_type="employee", entity_id=emp.id,
                trigger_label="date_field",
            ):
                fired += 1
        except Exception:  # noqa: BLE001
            logger.exception(
                "定时规则 %s(%s)对员工 %s 执行失败",
                rule.id, rule.name, emp.id)
    return fired


def run_scheduled(db: Session, today: date | None = None) -> int:
    """date_field 规则日扫:员工日期字段 + offset_days == 今天 → 触发。

    - ``schedule.recurring=true`` → 按今年纪念日算(生日/入职周年;2/29
      平年按 2/28),否则按原始日期一次性触发
    - 幂等:event_key = ``s:{emp_id}:{today}`` 与 rule_id 联合唯一,
      同天同人同规则只执行一次
    - terminated 员工不触发;返回本次执行的动作数
    """
    today = today or local_now().date()
    rules = db.scalars(
        select(AutomationRule).where(
            AutomationRule.trigger_kind == "date_field",
            AutomationRule.is_active.is_(True),
        )
    ).all()
    return sum(_run_rule_today(db, r, today) for r in rules)


def run_rule_now(db: Session, rule: AutomationRule) -> int:
    """手动立即执行单条 date_field 规则(run-now API;Rippling 测试运行)。

    不受 is_active 限制 —— 管理员可以测试未启用规则的产出。
    """
    if rule.trigger_kind != "date_field":
        raise ValueError("not_a_date_field_rule")
    return _run_rule_today(db, rule, local_now().date())


# ---------------------------------------------------------------------------
# 规则校验(API 层复用)
# ---------------------------------------------------------------------------


def validate_rule_payload(
    entity_type: str,
    trigger: str,
    conditions: list[dict[str, Any]] | None,
    recipients: list[dict[str, Any]] | None,
    action: str,
    *,
    trigger_kind: str = "audit",
    schedule: dict[str, Any] | None = None,
    action_config: dict[str, Any] | None = None,
) -> str | None:
    """返回错误消息字符串;None = 通过。"""
    spec = ENTITY_REGISTRY.get(entity_type)
    if spec is None:
        return f"不支持的实体:{entity_type}"
    if trigger_kind not in TRIGGER_KINDS:
        return f"trigger_kind 须为 {TRIGGER_KINDS} 之一"
    if trigger_kind == "date_field":
        if entity_type != "employee":
            return "日期触发目前仅支持员工实体"
        sched = schedule or {}
        if sched.get("field") not in DATE_FIELDS:
            return f"日期字段须为 {','.join(DATE_FIELDS)} 之一"
        off = sched.get("offset_days", 0)
        if isinstance(off, bool) or not isinstance(off, int) or abs(off) > 365:
            return "offset_days 须为 ±365 内的整数"
    elif not trigger or not _TRIGGER_RE.match(trigger):
        return ("trigger 须为小写审计动作名(字母/数字/下划线/冒号,≤64);"
                "常用 create/update/delete,也可用 approve 等具体动作名")
    if action not in ACTIONS:
        return f"action 必须是 {ACTIONS} 之一"
    if action == "create_case":
        cfg = action_config or {}
        if cfg.get("kind") and cfg["kind"] not in HrCase.KINDS:
            return f"工单类型须为 {','.join(HrCase.KINDS)} 之一"
        if cfg.get("priority") and cfg["priority"] not in CASE_PRIORITIES:
            return f"优先级须为 {','.join(CASE_PRIORITIES)} 之一"
        ak = cfg.get("assignee_kind")
        if ak and ak not in ("role", "user"):
            return "assignee_kind 须为 role/user"
        if ak and not cfg.get("assignee_value"):
            return "assignee 需要 value"
    if action in CANDIDATE_ACTIONS:
        if not spec.get("candidate_attr"):
            return f"「{action}」动作要求实体有关联候选人"
        cfg = action_config or {}
        if action == "add_to_group" and not cfg.get("group_id"):
            return "加入候选人分组需要 action_config.group_id"
        if action == "send_survey" and not cfg.get("survey_id"):
            return "发送问卷需要 action_config.survey_id"
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
                "has_candidate": bool(v.get("candidate_attr")),
            }
            for k, v in ENTITY_REGISTRY.items()
        },
        "ops": list(CONDITION_OPS),
        "actions": list(ACTIONS),
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "triggers": list(TRIGGERS),
        "trigger_kinds": [
            {"value": "audit", "label": "业务事件"},
            {"value": "date_field", "label": "日期到达"},
        ],
        "date_fields": [{"value": k, "label": v} for k, v in DATE_FIELDS.items()],
        "case_kinds": list(HrCase.KINDS),
        "case_priorities": list(CASE_PRIORITIES),
    }


__all__ = [
    "ENTITY_REGISTRY",
    "ENTITY_TYPES",
    "DATE_FIELDS",
    "TRIGGER_KINDS",
    "dispatch",
    "eval_conditions",
    "render_tpl",
    "resolve_recipients",
    "run_rule_now",
    "run_scheduled",
    "schema_meta",
    "suspend_automation",
    "validate_rule_payload",
]
