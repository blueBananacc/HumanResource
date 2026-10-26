"""P1 AI 招聘官 SOP 任务链 API。

助手发起一条 run(内置步骤模板),逐步 PATCH step 状态推进;
前端步骤链卡片轮询本接口实时呈现「专项 Agent 分工 + 进度」。
run/step 只是可观测状态机,不做编排执行 — 真正干活的是现有工具。
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import RecruitRun, RecruitRunStep, User
from app.infra.db import get_db
from app.services.permissions import PERM_WRITE_JOBS, require_permission

router = APIRouter(
    prefix="/recruit-runs",
    tags=["recruit-runs"],
    dependencies=[Depends(require_feature("hr.suite"))],
)

TERMINAL = ("done", "failed", "skipped")

# kind → 步骤模板 (key, label, agent_label, link)
TEMPLATES: dict[str, list[tuple[str, str, str, str | None]]] = {
    "full_pipeline": [
        ("jd", "确认 JD 与岗位要求", "JD 专员", "/jobs"),
        ("parse", "解析技能要求 / 淘汰红线", "JD 专员", None),
        ("source", "候选人入库与去重", "搜人专员", "/talents"),
        ("score", "触发匹配打分", "筛选专员", None),
        ("shortlist", "输出 Top 候选人", "筛选专员", "/jobs"),
        ("interview", "发起面试", "面试专员", "/interviews"),
        ("report", "生成招聘汇报", "汇报专员", "/reports"),
    ],
}


class StepOut(BaseModel):
    id: str
    seq: int
    key: str
    label: str
    agent_label: str
    status: str
    detail: dict
    link: str | None
    started_at: str | None
    finished_at: str | None


class RunOut(BaseModel):
    id: str
    kind: str
    title: str
    status: str
    context: dict
    created_at: str
    finished_at: str | None
    steps: list[StepOut]


class RunCreate(BaseModel):
    kind: str = "full_pipeline"
    title: str = Field(default="", max_length=200)
    job_id: str | None = None


class StepPatch(BaseModel):
    status: str
    detail: dict | None = None
    link: str | None = None


class RunPatch(BaseModel):
    status: str


def _step_out(s: RecruitRunStep) -> StepOut:
    return StepOut(
        id=s.id,
        seq=s.seq,
        key=s.key,
        label=s.label,
        agent_label=s.agent_label,
        status=s.status,
        detail=s.detail or {},
        link=s.link,
        started_at=s.started_at.isoformat() if s.started_at else None,
        finished_at=s.finished_at.isoformat() if s.finished_at else None,
    )


def _run_out(r: RecruitRun) -> RunOut:
    return RunOut(
        id=r.id,
        kind=r.kind,
        title=r.title,
        status=r.status,
        context=r.context or {},
        created_at=r.created_at.isoformat(),
        finished_at=r.finished_at.isoformat() if r.finished_at else None,
        steps=[_step_out(s) for s in sorted(r.steps, key=lambda x: x.seq)],
    )


def _get_own_run(db: Session, current: User, run_id: str) -> RecruitRun:
    run = db.scalar(
        select(RecruitRun).where(
            RecruitRun.id == run_id,
            RecruitRun.tenant_id == current.tenant_id,
            RecruitRun.user_id == current.id,
        )
    )
    if not run:
        raise HTTPException(404, "run 不存在")
    return run


def _auto_finish(db: Session, run: RecruitRun) -> None:
    """所有步骤到终态 → run 自动收口。"""
    if not run.steps:
        return
    statuses = [s.status for s in run.steps]
    if all(s in TERMINAL for s in statuses):
        run.status = "failed" if "failed" in statuses else "done"
        run.finished_at = datetime.now(UTC).replace(tzinfo=None)


@router.get("/", response_model=list[RunOut])
def list_runs(
    status: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> list[RunOut]:
    q = (
        select(RecruitRun)
        .where(
            RecruitRun.tenant_id == current.tenant_id,
            RecruitRun.user_id == current.id,
        )
        .order_by(RecruitRun.created_at.desc())
        .limit(limit)
    )
    if status:
        q = q.where(RecruitRun.status == status)
    return [_run_out(r) for r in db.scalars(q).all()]


@router.get("/{run_id}", response_model=RunOut)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RunOut:
    return _run_out(_get_own_run(db, current, run_id))


@router.post(
    "/",
    response_model=RunOut,
    dependencies=[Depends(require_permission(PERM_WRITE_JOBS))],
)
def create_run(
    body: RunCreate,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RunOut:
    template = TEMPLATES.get(body.kind)
    if not template:
        raise HTTPException(400, f"未知 kind:{body.kind};可选:{','.join(TEMPLATES)}")
    run = RecruitRun(
        tenant_id=current.tenant_id,
        user_id=current.id,
        kind=body.kind,
        title=body.title or "AI 招聘任务链",
        status="running",
        context={"job_id": body.job_id} if body.job_id else {},
    )
    db.add(run)
    db.flush()
    for i, (key, label, agent, link) in enumerate(template, start=1):
        db.add(
            RecruitRunStep(
                run_id=run.id,
                seq=i,
                key=key,
                label=label,
                agent_label=agent,
                status="pending",
                link=link,
            )
        )
    db.commit()
    db.refresh(run)
    return _run_out(run)


@router.patch("/{run_id}", response_model=RunOut)
def patch_run(
    run_id: str,
    body: RunPatch,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RunOut:
    run = _get_own_run(db, current, run_id)
    if body.status not in RecruitRun.STATUS:
        raise HTTPException(400, f"非法 status:{body.status}")
    run.status = body.status
    if body.status in TERMINAL or body.status == "cancelled":
        run.finished_at = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    db.refresh(run)
    return _run_out(run)


@router.patch("/{run_id}/steps/{step_key}", response_model=RunOut)
def patch_step(
    run_id: str,
    step_key: str,
    body: StepPatch,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> RunOut:
    run = _get_own_run(db, current, run_id)
    if body.status not in RecruitRunStep.STATUS:
        raise HTTPException(400, f"非法 status:{body.status}")
    step = next((s for s in run.steps if s.key == step_key), None)
    if not step:
        raise HTTPException(404, f"step '{step_key}' 不存在")
    now = datetime.now(UTC).replace(tzinfo=None)
    step.status = body.status
    if body.status == "running" and not step.started_at:
        step.started_at = now
    if body.status in TERMINAL or body.status == "waiting_confirm":
        step.finished_at = now if body.status in TERMINAL else None
    if body.detail is not None:
        step.detail = body.detail
    if body.link is not None:
        step.link = body.link
    _auto_finish(db, run)
    db.commit()
    db.refresh(run)
    return _run_out(run)


@router.get("/meta/templates", response_model=dict)
def meta_templates() -> dict:
    """助手/前端用来展示可选剧本的步骤骨架。"""
    return {
        kind: [
            {"seq": i, "key": k, "label": lb, "agent_label": ag, "link": lk}
            for i, (k, lb, ag, lk) in enumerate(steps, start=1)
        ]
        for kind, steps in TEMPLATES.items()
    }
