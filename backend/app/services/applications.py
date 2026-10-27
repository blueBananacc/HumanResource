"""申请管道卡服务:阶段推进与历史记录。

阶段顺序(展示用,流转本身不强制单向):
    applied → screening → interview → offer → hired
    rejected / withdrawn 为旁路状态,可回退。

系统联动钩子统一走 ``advance_application``:发起面试、offer send/
respond/withdraw 时把管道卡推进到对应阶段(只前进,不覆盖更靠后的
人工状态 — 比如卡在 offer 的人再建面试,不会把它拉回 interview)。
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import JobApplication, JobApplicationEvent

STAGE_ORDER = ["applied", "screening", "interview", "offer", "hired"]
TERMINAL_STAGES = ("rejected", "withdrawn")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _rank(stage: str) -> int:
    """主线阶段返回次序;旁路状态按 -1 处理(总能被主线阶段覆盖)。"""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return -1


def get_application(
    db: Session, tenant_id: str, candidate_id: str, job_id: str
) -> JobApplication | None:
    return db.scalars(
        select(JobApplication).where(
            JobApplication.tenant_id == tenant_id,
            JobApplication.candidate_id == candidate_id,
            JobApplication.job_id == job_id,
        )
    ).first()


def record_event(
    db: Session,
    app: JobApplication,
    to_stage: str,
    *,
    actor_user_id: str | None,
    note: str | None = None,
) -> JobApplicationEvent:
    ev = JobApplicationEvent(
        tenant_id=app.tenant_id,
        application_id=app.id,
        from_stage=app.stage,
        to_stage=to_stage,
        actor_user_id=actor_user_id,
        note=note,
    )
    db.add(ev)
    app.stage = to_stage
    app.stage_changed_at = _now()
    # 阶段变化可能影响 stage 条件的自动入库规则 — 重评该候选人
    try:
        from app.domain.models import Candidate
        from app.services.group_rules import enroll_candidate

        cand = db.get(Candidate, app.candidate_id)
        if cand is not None:
            enroll_candidate(db, cand)
    except Exception:  # noqa: BLE001 — 规则失败不拖垮管道流转
        import logging

        logging.getLogger(__name__).exception(
            "auto_enroll 失败 candidate=%s", app.candidate_id
        )
    return ev


def advance_application(
    db: Session,
    tenant_id: str,
    candidate_id: str,
    job_id: str,
    to_stage: str,
    *,
    actor_user_id: str | None = None,
    note: str | None = None,
    only_forward: bool = True,
    create_if_missing: bool = False,
) -> JobApplication | None:
    """把 (candidate, job) 的管道卡推进到 ``to_stage``。

    - ``only_forward=True``(默认):当前阶段主线次序已 >= 目标时不回退,
      旁路状态(rejected/withdrawn)则允许被主线阶段覆盖。
    - ``create_if_missing=True``:没有卡时按 source=system 自动建一张
      (联动钩子里的补卡行为)。
    """
    app = get_application(db, tenant_id, candidate_id, job_id)
    if app is None:
        if not create_if_missing:
            return None
        app = JobApplication(
            tenant_id=tenant_id,
            candidate_id=candidate_id,
            job_id=job_id,
            stage="applied",
            source="system",
        )
        db.add(app)
        db.flush()
        record_event(db, app, "applied", actor_user_id=actor_user_id, note="自动建卡")

    if app.stage == to_stage:
        return app
    if only_forward and _rank(app.stage) >= _rank(to_stage) > -1:
        return app
    record_event(db, app, to_stage, actor_user_id=actor_user_id, note=note)
    return app
