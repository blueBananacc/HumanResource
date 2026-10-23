"""招聘借力 API:招聘需求审批 + Offer。

- ``GET/POST /requisitions``                        需求列表(hm 提交)/新建
- ``POST /requisitions/{id}/approve|reject``        审批(approve_requisitions)
- ``POST /requisitions/{id}/to-job``                approved → 创建岗位
- ``GET/POST /offers``                              Offer 列表/新建
- ``GET  /offers/{id}``                             详情
- ``POST /offers/{id}/send``                        draft → sent
- ``POST /offers/{id}/respond``                     候选人回应:accepted → 建 Employee+onboarding
- ``POST /offers/{id}/withdraw``                    撤回
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api._pagination import PageOut, count_total, paginate_params
from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Candidate,
    Employee,
    Job,
    JobRequisition,
    JobOffer,
    LifecycleEvent,
    LifecycleTask,
    LifecycleTemplate,
    OrgUnit,
    User,
)
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_APPROVE_REQUISITIONS,
    PERM_MANAGE_OFFERS,
    PERM_WRITE_REQUISITIONS,
    PERM_VIEW_REPORTS,
    require_permission,
)

router = APIRouter(tags=["hris-hiring"])
req_router = APIRouter(prefix="/requisitions", tags=["hris-requisitions"], dependencies=[Depends(require_feature("hr.suite"))])
offer_router = APIRouter(prefix="/offers", tags=["hris-offers"], dependencies=[Depends(require_feature("hr.suite"))])


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _maps(db: Session, tenant_id: str):
    orgs = {o.id: o.name for o in db.scalars(
        select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)
    )}
    users = {u.id: u.email for u in db.scalars(
        select(User).where(User.tenant_id == tenant_id)
    )}
    jobs = {j.id: j.title for j in db.scalars(
        select(Job).where(Job.tenant_id == tenant_id)
    )}
    return orgs, users, jobs


# ============ requisitions ============


class RequisitionIn(BaseModel):
    title: str
    org_unit_id: str | None = None
    headcount: int = Field(1, ge=1, le=500)
    level: Literal["junior", "intermediate", "senior", "expert"] = "intermediate"
    reason: str = ""


class ReqDecisionIn(BaseModel):
    note: str | None = None


class RequisitionOut(BaseModel):
    id: str
    title: str
    org_unit_id: str | None
    org_unit_name: str | None
    headcount: int
    level: str
    reason: str
    status: str
    requested_by: str | None
    requested_by_email: str | None
    approver_id: str | None
    approver_email: str | None
    decided_at: datetime | None
    decision_note: str | None
    job_id: str | None
    job_title: str | None
    created_at: datetime


def _req_out(r: JobRequisition, orgs, users, jobs) -> RequisitionOut:
    return RequisitionOut(
        id=r.id, title=r.title, org_unit_id=r.org_unit_id,
        org_unit_name=orgs.get(r.org_unit_id) if r.org_unit_id else None,
        headcount=r.headcount, level=r.level, reason=r.reason, status=r.status,
        requested_by=r.requested_by,
        requested_by_email=users.get(r.requested_by) if r.requested_by else None,
        approver_id=r.approver_id,
        approver_email=users.get(r.approver_id) if r.approver_id else None,
        decided_at=r.decided_at, decision_note=r.decision_note,
        job_id=r.job_id, job_title=jobs.get(r.job_id) if r.job_id else None,
        created_at=r.created_at,
    )


@req_router.get("", response_model=PageOut[RequisitionOut])
def list_requisitions(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    limit, offset, _ = pagination
    stmt = select(JobRequisition).where(
        JobRequisition.tenant_id == current.tenant_id
    )
    if status_filter:
        stmt = stmt.where(JobRequisition.status == status_filter)
    # 用人经理只看自己提的;admin/hr 全看
    if current.role not in ("admin", "hr"):
        stmt = stmt.where(JobRequisition.requested_by == current.id)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(JobRequisition.created_at.desc()).limit(limit).offset(offset)
    ).all()
    orgs, users, jobs = _maps(db, current.tenant_id)
    return PageOut(
        items=[_req_out(r, orgs, users, jobs) for r in rows],
        total=total, limit=limit, offset=offset,
    )


@req_router.post("", response_model=RequisitionOut, status_code=status.HTTP_201_CREATED)
def create_requisition(
    body: RequisitionIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_REQUISITIONS)),
):
    if body.org_unit_id:
        org = db.get(OrgUnit, body.org_unit_id)
        if not org or org.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "部门不存在")
    r = JobRequisition(
        tenant_id=current.tenant_id, title=body.title.strip(),
        org_unit_id=body.org_unit_id, headcount=body.headcount,
        level=body.level, reason=body.reason, requested_by=current.id,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    write_audit(
        db, actor=current, entity_type="job_requisition", entity_id=r.id,
        action="create", detail={"title": r.title, "headcount": r.headcount},
        request=request,
    )
    db.commit()
    orgs, users, jobs = _maps(db, current.tenant_id)
    return _req_out(r, orgs, users, jobs)


def _req(db: Session, rid: str, tenant_id: str) -> JobRequisition:
    r = db.get(JobRequisition, rid)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "需求不存在")
    return r


@req_router.post("/{rid}/approve", response_model=RequisitionOut)
def approve_requisition(
    rid: str,
    request: Request,
    body: ReqDecisionIn | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_REQUISITIONS)),
):
    r = _req(db, rid, current.tenant_id)
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已处理")
    if r.requested_by == current.id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "不能审批自己提交的招聘需求")
    r.status = "approved"
    r.approver_id = current.id
    r.decided_at = _now()
    r.decision_note = body.note if body else None
    write_audit(
        db, actor=current, entity_type="job_requisition", entity_id=r.id,
        action="approve", detail={}, request=request,
    )
    db.commit()
    db.refresh(r)
    orgs, users, jobs = _maps(db, current.tenant_id)
    return _req_out(r, orgs, users, jobs)


@req_router.post("/{rid}/reject", response_model=RequisitionOut)
def reject_requisition(
    rid: str,
    request: Request,
    body: ReqDecisionIn | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_REQUISITIONS)),
):
    r = _req(db, rid, current.tenant_id)
    if r.status != "pending":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已处理")
    if r.requested_by == current.id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "不能审批自己提交的招聘需求")
    r.status = "rejected"
    r.approver_id = current.id
    r.decided_at = _now()
    r.decision_note = body.note if body else None
    write_audit(
        db, actor=current, entity_type="job_requisition", entity_id=r.id,
        action="reject", detail={}, request=request,
    )
    db.commit()
    db.refresh(r)
    orgs, users, jobs = _maps(db, current.tenant_id)
    return _req_out(r, orgs, users, jobs)


@req_router.post("/{rid}/to-job", response_model=RequisitionOut)
def requisition_to_job(
    rid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_APPROVE_REQUISITIONS)),
):
    r = _req(db, rid, current.tenant_id)
    if r.status != "approved":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "只有已批准的需求才能转岗位")
    if r.job_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已生成岗位")
    job = Job(
        tenant_id=current.tenant_id, title=r.title, level=r.level,
        org_unit_id=r.org_unit_id, publish_status="draft",
        created_by=current.id,
        description=f"由招聘需求「{r.title}」(编制 {r.headcount} 人)创建。",
    )
    db.add(job)
    db.flush()
    r.job_id = job.id
    write_audit(
        db, actor=current, entity_type="job_requisition", entity_id=r.id,
        action="to_job", detail={"job_id": job.id}, request=request,
    )
    db.commit()
    db.refresh(r)
    orgs, users, jobs = _maps(db, current.tenant_id)
    return _req_out(r, orgs, users, jobs)


# ============ offers ============


class OfferIn(BaseModel):
    candidate_id: str
    job_id: str | None = None
    position: str | None = None
    salary: dict | None = None
    start_date: date | None = None
    letter: str | None = None


class OfferOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    job_id: str | None
    job_title: str | None
    position: str | None
    salary: dict | None
    start_date: date | None
    letter: str | None
    status: str
    sent_at: datetime | None
    responded_at: datetime | None
    employee_id: str | None
    created_at: datetime


def _offer_out(o: JobOffer, cands, jobs) -> OfferOut:
    return OfferOut(
        id=o.id, candidate_id=o.candidate_id,
        candidate_name=cands.get(o.candidate_id, "?"),
        job_id=o.job_id, job_title=jobs.get(o.job_id) if o.job_id else None,
        position=o.position, salary=o.salary, start_date=o.start_date,
        letter=o.letter, status=o.status, sent_at=o.sent_at,
        responded_at=o.responded_at, employee_id=o.employee_id,
        created_at=o.created_at,
    )


def _offer_maps(db: Session, tenant_id: str):
    cands = {c.id: c.name for c in db.scalars(
        select(Candidate).where(Candidate.tenant_id == tenant_id)
    )}
    jobs = {j.id: j.title for j in db.scalars(
        select(Job).where(Job.tenant_id == tenant_id)
    )}
    return cands, jobs


@offer_router.get("", response_model=PageOut[OfferOut])
def list_offers(
    pagination: tuple = Depends(paginate_params),
    status_filter: str | None = Query(None, alias="status"),
    candidate_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    limit, offset, _ = pagination
    stmt = select(JobOffer).where(JobOffer.tenant_id == current.tenant_id)
    if status_filter:
        stmt = stmt.where(JobOffer.status == status_filter)
    if candidate_id:
        stmt = stmt.where(JobOffer.candidate_id == candidate_id)
    total = count_total(db, stmt)
    rows = db.scalars(
        stmt.order_by(JobOffer.created_at.desc()).limit(limit).offset(offset)
    ).all()
    cands, jobs = _offer_maps(db, current.tenant_id)
    return PageOut(
        items=[_offer_out(o, cands, jobs) for o in rows],
        total=total, limit=limit, offset=offset,
    )


@offer_router.post("", response_model=OfferOut, status_code=status.HTTP_201_CREATED)
def create_offer(
    body: OfferIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    cand = db.get(Candidate, body.candidate_id)
    if not cand or cand.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "候选人不存在")
    if body.job_id:
        job = db.get(Job, body.job_id)
        if not job or job.tenant_id != current.tenant_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "岗位不存在")
    if not body.letter:
        body.letter = (
            f"录用通知书\n\n{cand.name} 您好,\n\n"
            f"我们诚挚邀请您加入,岗位:{body.position or '待定'}。"
            f"预计入职日期:{body.start_date or '待沟通'}。\n\n"
            f"期待您的加入!"
        )
    o = JobOffer(
        tenant_id=current.tenant_id, candidate_id=cand.id, job_id=body.job_id,
        position=body.position, salary=body.salary, start_date=body.start_date,
        letter=body.letter, created_by=current.id,
    )
    db.add(o)
    db.commit()
    db.refresh(o)
    write_audit(
        db, actor=current, entity_type="job_offer", entity_id=o.id,
        action="create", detail={"candidate": cand.name}, request=request,
    )
    db.commit()
    cands, jobs = _offer_maps(db, current.tenant_id)
    return _offer_out(o, cands, jobs)


def _offer(db: Session, oid: str, tenant_id: str, *, for_update: bool = False) -> JobOffer:
    stmt = select(JobOffer).where(JobOffer.id == oid)
    if for_update:
        # 状态机变更带副作用(建员工/生命周期),先锁行防并发重复回应
        stmt = stmt.with_for_update()
    o = db.execute(stmt).scalar_one_or_none()
    if not o or o.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Offer 不存在")
    return o


@offer_router.get("/{oid}", response_model=OfferOut)
def get_offer(
    oid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    o = _offer(db, oid, current.tenant_id)
    cands, jobs = _offer_maps(db, current.tenant_id)
    return _offer_out(o, cands, jobs)


@offer_router.post("/{oid}/send", response_model=OfferOut)
def send_offer(
    oid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    o = _offer(db, oid, current.tenant_id)
    if o.status != "draft":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅草稿可发出")
    o.status = "sent"
    o.sent_at = _now()
    write_audit(
        db, actor=current, entity_type="job_offer", entity_id=o.id,
        action="send", detail={}, request=request,
    )
    db.commit()
    db.refresh(o)
    cands, jobs = _offer_maps(db, current.tenant_id)
    return _offer_out(o, cands, jobs)


class RespondIn(BaseModel):
    accepted: bool


@offer_router.post("/{oid}/respond", response_model=OfferOut)
def respond_offer(
    oid: str,
    body: RespondIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    """记录候选人回应。accepted → 建 Employee + onboarding 事件。"""
    o = _offer(db, oid, current.tenant_id, for_update=True)
    if o.status != "sent":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅已发出的 Offer 可回应")
    o.status = "accepted" if body.accepted else "declined"
    o.responded_at = _now()
    if body.accepted:
        emp = _employee_from_offer(db, o, current)
        o.employee_id = emp.id
        _spawn_onboarding(db, o, emp, current)
    write_audit(
        db, actor=current, entity_type="job_offer", entity_id=o.id,
        action="respond", detail={"accepted": body.accepted}, request=request,
    )
    db.commit()
    db.refresh(o)
    cands, jobs = _offer_maps(db, current.tenant_id)
    return _offer_out(o, cands, jobs)


def _employee_from_offer(db: Session, o: JobOffer, actor: User) -> Employee:
    """Offer accepted → 员工建档。"""
    from app.api.employees import next_employee_no

    cand = db.get(Candidate, o.candidate_id)
    job = db.get(Job, o.job_id) if o.job_id else None
    emp = Employee(
        tenant_id=o.tenant_id,
        employee_no=next_employee_no(db, o.tenant_id),
        name=cand.name,
        org_unit_id=job.org_unit_id if job else None,
        designation=o.position or (job.title if job else None),
        work_email=cand.display_email,
        phone=cand.display_phone,
        hire_date=o.start_date,
        status="active",
        source_candidate_id=cand.id,
    )
    db.add(emp)
    try:
        db.flush()
    except IntegrityError:
        # employee_no 采用 max+1 生成,并发建档可能撞号 — 回滚后由调用方重试
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "工号分配冲突,请重试")
    # 候选人标记 hired 便于人才库筛选
    if cand.status == "active":
        cand.status = "hired"
    return emp


def _spawn_onboarding(db: Session, o: JobOffer, emp: Employee, actor: User) -> None:
    ev = LifecycleEvent(
        tenant_id=o.tenant_id, employee_id=emp.id, kind="onboarding",
        effective_date=o.start_date, created_by=actor.id,
        note=f"Offer {o.id} 接受",
    )
    db.add(ev)
    db.flush()
    tpl = db.scalars(
        select(LifecycleTemplate).where(
            LifecycleTemplate.tenant_id == o.tenant_id,
            LifecycleTemplate.kind == "onboarding",
            LifecycleTemplate.is_default.is_(True),
        ).limit(1)
    ).first()
    titles = (
        [x.get("title", "") for x in (tpl.tasks or [])]
        if tpl else
        ["发放offer并确认入职日期", "创建系统账号", "开通门禁/邮箱",
         "设备领取", "入职培训", "试用期目标确认"]
    )
    for i, t in enumerate(titles):
        db.add(LifecycleTask(
            tenant_id=o.tenant_id, event_id=ev.id, title=t, sort=i,
        ))


@offer_router.post("/{oid}/withdraw", response_model=OfferOut)
def withdraw_offer(
    oid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_OFFERS)),
):
    o = _offer(db, oid, current.tenant_id)
    if o.status in ("accepted", "declined", "withdrawn"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "已结束")
    o.status = "withdrawn"
    write_audit(
        db, actor=current, entity_type="job_offer", entity_id=o.id,
        action="withdraw", detail={}, request=request,
    )
    db.commit()
    db.refresh(o)
    cands, jobs = _offer_maps(db, current.tenant_id)
    return _offer_out(o, cands, jobs)
