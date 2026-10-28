"""P 批次 — 人才画像卡 API。

- ``GET  /employees/{id}/talent-card``            聚合卡(本人视图自动脱敏)
- ``POST /employees/{id}/skills``                 手动加/改技能(直接 confirmed)
- ``POST /employees/{id}/skill-suggestions/confirm`` 批量确认 AI 建议
- ``POST /employees/{id}/skill-suggestions/reject``  驳回(删行)
- ``GET/POST /employees/{id}/insights``           洞察(会议纪要/复盘/随手评)
- ``GET  /me/talent-card``                        ESS 本人卡(脱敏版)

可见性:suggested 技能与 manager_only 洞察对「本人」隐藏。
写操作权限:manage_employees 或该员工直属上级。
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Employee,
    EmployeeSkill,
    InsightNote,
    Notification,
    OrgUnit,
    ProjectMember,
    Skill,
    SkillRating,
    SkillRatingRequest,
    User,
)
from app.infra.db import get_db
from app.services import profile_update, talent_card
from app.services.audit import write_audit
from app.services.permissions import (
    get_linked_employee_id, has_permission, require_linked_employee,
    visible_employee_ids,
)
from fastapi import Request

router = APIRouter(tags=["talent-card"], dependencies=[Depends(require_feature("hr.suite"))])


def _get_employee(db: Session, emp_id: str, current: User) -> Employee:
    emp = db.get(Employee, emp_id)
    if not emp or emp.tenant_id != current.tenant_id:
        raise HTTPException(404, "员工不存在")
    scope = visible_employee_ids(db, current)
    if scope is not None and emp.id not in scope:
        raise HTTPException(404, "员工不存在或无权访问")
    return emp


def _can_curate(db: Session, emp: Employee, current: User) -> bool:
    if has_permission(current, "manage_employees"):
        return True
    own = get_linked_employee_id(db, current)
    return bool(own and emp.reports_to_id == own)


@router.get("/employees/{emp_id}/talent-card")
def get_card(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    return talent_card.build_card(
        db, emp, viewer_employee_id=get_linked_employee_id(db, current)
    )


@router.get("/me/talent-card")
def my_card(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp_id = require_linked_employee(db, current)
    emp = db.get(Employee, emp_id)
    return talent_card.build_card(db, emp, viewer_employee_id=emp_id)


class SkillIn(BaseModel):
    name: str = Field(max_length=64)
    level: int = Field(default=3, ge=1, le=5)
    note: str | None = Field(default=None, max_length=200)


@router.post("/employees/{emp_id}/skills", status_code=201)
def add_skill(
    emp_id: str, payload: SkillIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    skill = profile_update.get_or_create_skill(
        db, tenant_id=current.tenant_id, name=payload.name
    )
    row = profile_update.upsert_employee_skill(
        db, tenant_id=current.tenant_id, employee_id=emp.id,
        skill_id=skill.id, level=payload.level, source="manual",
        status="confirmed", note=payload.note, created_by=current.id,
    )
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=row.id, action="upsert", request=request)
    db.commit()
    return {"id": row.id, "skill_id": skill.id, "name": skill.name,
            "level": row.level, "status": row.status}


class SuggestActionIn(BaseModel):
    skill_ids: list[str] | None = None  # None = 全部


def _pending(db: Session, emp_id: str, skill_ids: list[str] | None):
    q = select(EmployeeSkill).where(
        EmployeeSkill.employee_id == emp_id,
        EmployeeSkill.status == "suggested",
    )
    if skill_ids:
        q = q.where(EmployeeSkill.skill_id.in_(skill_ids))
    return db.scalars(q).all()


@router.post("/employees/{emp_id}/skill-suggestions/confirm")
def confirm_suggestions(
    emp_id: str, payload: SuggestActionIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    rows = _pending(db, emp.id, payload.skill_ids)
    for r in rows:
        r.status = "confirmed"
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=emp.id, action=f"confirm_x{len(rows)}", request=request)
    db.commit()
    return {"confirmed": len(rows)}


@router.post("/employees/{emp_id}/skill-suggestions/reject")
def reject_suggestions(
    emp_id: str, payload: SuggestActionIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    rows = _pending(db, emp.id, payload.skill_ids)
    for r in rows:
        db.delete(r)
    write_audit(db, actor=current, entity_type="employee_skill",
                entity_id=emp.id, action=f"reject_x{len(rows)}", request=request)
    db.commit()
    return {"rejected": len(rows)}


# ============ 洞察(会议纪要手动上传入口在此) ============


class InsightIn(BaseModel):
    kind: Literal["meeting", "project_review", "manager_note", "milestone"] = "manager_note"
    title: str | None = Field(default=None, max_length=200)
    body: str = Field(min_length=1, max_length=8000)
    visibility: Literal["manager_only", "employee_visible"] = "manager_only"
    source_ref: dict[str, Any] | None = None
    extract_skills: bool = True  # 是否触发画像建议管线


@router.get("/employees/{emp_id}/insights")
def list_insights(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    q = select(InsightNote).where(InsightNote.employee_id == emp.id)
    if get_linked_employee_id(db, current) == emp.id:
        q = q.where(InsightNote.visibility == "employee_visible")
    rows = db.scalars(q.order_by(InsightNote.created_at.desc()).limit(50)).all()
    return {"items": [{
        "id": n.id, "kind": n.kind, "title": n.title, "body": n.body,
        "visibility": n.visibility, "source_ref": n.source_ref,
        "created_at": str(n.created_at or ""),
    } for n in rows]}


@router.post("/employees/{emp_id}/insights", status_code=201)
def add_insight(
    emp_id: str, payload: InsightIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    note = InsightNote(
        tenant_id=current.tenant_id, employee_id=emp.id, kind=payload.kind,
        title=payload.title, body=payload.body,
        author_user_id=current.id, source_ref=payload.source_ref,
        visibility=payload.visibility,
    )
    db.add(note)
    db.flush()
    write_audit(db, actor=current, entity_type="insight_note",
                entity_id=note.id, action="create", request=request)
    suggestions = {"suggested": [], "skipped": []}
    if payload.extract_skills:
        suggestions = profile_update.suggest_from_text(
            db, tenant_id=current.tenant_id, employee=emp,
            text=f"{payload.title or ''}\n{payload.body}",
            source="insight",
            evidence={"kind": "insight", "id": note.id},
            created_by=current.id,
        )
    db.commit()
    return {"id": note.id, **suggestions}


# ============ AI 选人:确定性技能预筛(LLM 在结果上做 topK/理由) ============


class TalentSearchIn(BaseModel):
    """skills: [{"name": "TypeScript", "min_level": 3}];

    name 对技能字典做模糊匹配 —— 「AI coding 强」由 LLM 先拆成
    技能名片段(如 ["AI", "Python"])再传进来,这里只做确定性过滤。
    """

    skills: list[dict[str, Any]] = Field(default_factory=list)
    include_suggested: bool = False  # 待确认标签默认不计入匹配
    org_unit_id: str | None = None
    only_active: bool = True
    limit: int = Field(default=20, ge=1, le=50)


@router.post("/talent/search")
def search_talent(
    payload: TalentSearchIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    reqs: list[tuple[str, int]] = []
    for s in payload.skills:
        frag = str(s.get("name") or "").strip()[:64]
        if frag:
            reqs.append((frag, min(5, max(1, int(s.get("min_level") or 1)))))
    if not reqs:
        raise HTTPException(422, "skills 不能为空 —— 至少给一个技能名片段")

    # 1) 片段 → 字典技能 id(每段命中的技能集合)
    dict_rows = db.scalars(
        select(Skill).where(
            Skill.tenant_id == current.tenant_id,
            or_(*[Skill.name.ilike(f"%{n}%") for n, _ in reqs]),
        )
    ).all()
    frag_map: list[tuple[str, int, set[str]]] = []  # (片段, min_level, skill_ids)
    for frag, lv in reqs:
        ids = {s.id for s in dict_rows if frag.lower() in s.name.lower()}
        frag_map.append((frag, lv, ids))
    all_ids = {sid for _, _, ids in frag_map for sid in ids}
    if not all_ids:
        return {"items": [], "matched": {"requested": [n for n, _ in reqs], "dict_hits": []}}

    # 2) 员工技能行过滤(状态/等级/范围)
    statuses = ["confirmed"] + (["suggested"] if payload.include_suggested else [])
    rows = db.execute(
        select(EmployeeSkill, Skill.name)
        .join(Skill, Skill.id == EmployeeSkill.skill_id)
        .where(
            EmployeeSkill.tenant_id == current.tenant_id,
            EmployeeSkill.skill_id.in_(all_ids),
            EmployeeSkill.status.in_(statuses),
        )
    ).all()

    # 3) 按员工聚合命中片段数
    scope = visible_employee_ids(db, current)
    hit: dict[str, dict] = {}
    for es, sname in rows:
        if scope is not None and es.employee_id not in scope:
            continue
        entry = hit.setdefault(es.employee_id, {"frags": set(), "skills": []})
        for frag, lv, ids in frag_map:
            if es.skill_id in ids and es.level >= lv:
                entry["frags"].add(frag)
        entry["skills"].append({
            "name": sname, "level": es.level, "source": es.source,
            "status": es.status,
        })
    if not hit:
        return {"items": [], "matched": {"requested": [n for n, _ in reqs],
                                         "dict_hits": [s.name for s in dict_rows]}}

    emp_ids = [eid for eid, e in hit.items() if e["frags"]]
    q = select(Employee).where(
        Employee.tenant_id == current.tenant_id, Employee.id.in_(emp_ids)
    )
    if payload.only_active:
        q = q.where(Employee.status == "active")
    if payload.org_unit_id:
        q = q.where(Employee.org_unit_id == payload.org_unit_id)
    emps = {e.id: e for e in db.scalars(q).all()}

    org_names = dict(
        db.execute(select(OrgUnit.id, OrgUnit.name).where(
            OrgUnit.tenant_id == current.tenant_id)).all()
    )
    proj_counts = dict(
        db.execute(
            select(ProjectMember.employee_id, func.count())
            .where(
                ProjectMember.tenant_id == current.tenant_id,
                ProjectMember.employee_id.in_(list(emps)),
                ProjectMember.left_at.is_(None),
            )
            .group_by(ProjectMember.employee_id)
        ).all()
    )

    items = []
    for eid, e in emps.items():
        h = hit[eid]
        skills = sorted(h["skills"], key=lambda x: -x["level"])
        items.append({
            "employee_id": eid,
            "name": e.name,
            "designation": e.designation,
            "org_unit": org_names.get(e.org_unit_id or ""),
            "matched_fragments": sorted(h["frags"]),
            "match_count": len(h["frags"]),
            "matched_skills": skills,
            "active_projects": proj_counts.get(eid, 0),
        })
    items.sort(key=lambda x: (-x["match_count"],
                              -sum(s["level"] for s in x["matched_skills"])))
    return {
        "items": items[: payload.limit],
        "matched": {
            "requested": [n for n, _ in reqs],
            "dict_hits": [s.name for s in dict_rows],
        },
    }


# ============ 多评价者技能评分(Degreed 式多源评分 + iCIMS 反馈请求流) ============
#
# 流:员工/经理/HR 发起 SkillRatingRequest(评价人默认直属上级)→ 评价人在
# 「待我评价」里打分/婉拒 → 提交落 SkillRating(每人每技能一条,幂等 upsert)
# 并同步 upsert EmployeeSkill(source=self/manager/peer, confirmed)。评分提交后
# 不可改(iCIMS 语义):要改分只能取消请求重发,或再次发起新请求覆盖。
# 员工也可不经过请求直接自评(POST /me/skills/{skill_id}/rate)。

RATING_KIND_LABEL = {"self": "自评", "manager": "经理", "peer": "同事"}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _rating_out(db: Session, r: SkillRating) -> dict[str, Any]:
    rater = db.get(Employee, r.rater_employee_id)
    skill = db.get(Skill, r.skill_id)
    return {
        "id": r.id, "skill_id": r.skill_id,
        "skill_name": skill.name if skill else None,
        "employee_id": r.employee_id,
        "rater_employee_id": r.rater_employee_id,
        "rater_name": rater.name if rater else None,
        "rater_kind": r.rater_kind, "level": r.level, "comment": r.comment,
        "request_id": r.request_id,
        "created_at": str(r.created_at or "")[:16],
    }


def _req_out(db: Session, q: SkillRatingRequest) -> dict[str, Any]:
    skill = db.get(Skill, q.skill_id)
    ratee = db.get(Employee, q.employee_id)
    rater = db.get(Employee, q.rater_employee_id)
    return {
        "id": q.id, "skill_id": q.skill_id,
        "skill_name": skill.name if skill else None,
        "employee_id": q.employee_id,
        "ratee_name": ratee.name if ratee else None,
        "rater_employee_id": q.rater_employee_id,
        "rater_name": rater.name if rater else None,
        "rater_kind": q.rater_kind, "note": q.note, "status": q.status,
        "rating_id": q.rating_id,
        "created_at": str(q.created_at or "")[:16],
        "decided_at": str(q.decided_at or "")[:16],
    }


def _notify_emp(
    db: Session, tenant_id: str, emp_id: str | None, *,
    title: str, body: str | None = None, link: str = "/me/profile",
) -> None:
    """给员工的绑定账号写站内通知(未绑账号静默跳过)。"""
    emp = db.get(Employee, emp_id) if emp_id else None
    if emp is not None and emp.user_id:
        db.add(Notification(
            tenant_id=tenant_id, user_id=emp.user_id, kind="skill_rating",
            title=title[:128], body=(body or "")[:512], link=link,
        ))


def _get_task(db: Session, task_id: str, current: User) -> SkillRatingRequest:
    """我作为评价人的请求(不存在/非本人/跨租户 → 404)。"""
    me = require_linked_employee(db, current)
    q = db.get(SkillRatingRequest, task_id)
    if (not q or q.tenant_id != current.tenant_id
            or q.rater_employee_id != me):
        raise HTTPException(404, "评分任务不存在")
    return q


class RatingRequestIn(BaseModel):
    skill_id: str
    rater_employee_id: str | None = None  # None → 被评人直属上级
    note: str | None = Field(default=None, max_length=300)


def _create_rating_request(
    db: Session, *, current: User, ratee: Employee,
    payload: RatingRequestIn, request: Request,
) -> SkillRatingRequest:
    skill = db.get(Skill, payload.skill_id)
    if not skill or skill.tenant_id != current.tenant_id:
        raise HTTPException(404, "技能不存在")
    rater_id = payload.rater_employee_id or ratee.reports_to_id
    if not rater_id:
        raise HTTPException(422, "请指定评价人(该员工没有直属上级)")
    if rater_id == ratee.id:
        raise HTTPException(422, "不能向本人发起他评 — 自评请直接打分")
    rater = db.get(Employee, rater_id)
    if not rater or rater.tenant_id != current.tenant_id:
        raise HTTPException(404, "评价人不存在")
    if rater.status != "active":
        raise HTTPException(422, "评价人已离职,无法接收评分任务")
    if not rater.user_id:
        raise HTTPException(422, "评价人未绑定账号,无法接收评分任务")
    dup = db.scalars(
        select(SkillRatingRequest).where(
            SkillRatingRequest.tenant_id == current.tenant_id,
            SkillRatingRequest.employee_id == ratee.id,
            SkillRatingRequest.skill_id == skill.id,
            SkillRatingRequest.rater_employee_id == rater.id,
            SkillRatingRequest.status == "pending",
        )
    ).first()
    if dup:
        raise HTTPException(409, "已有一条待该评价人处理的评分请求")
    kind = "manager" if rater.id == ratee.reports_to_id else "peer"
    q = SkillRatingRequest(
        tenant_id=current.tenant_id, employee_id=ratee.id, skill_id=skill.id,
        rater_employee_id=rater.id, rater_kind=kind, note=payload.note,
        requested_by_id=current.id,
    )
    db.add(q)
    db.flush()
    write_audit(db, actor=current, entity_type="skill_rating_request",
                entity_id=q.id, action="create", request=request)
    _notify_emp(
        db, current.tenant_id, rater.id,
        title=f"{ratee.name} 请你评价技能「{skill.name}」",
        body=payload.note or "请在「我的 → 待我评价」中完成打分。",
    )
    return q


@router.post("/me/skill-rating-requests", status_code=201)
def request_rating_self(
    payload: RatingRequestIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """员工给自己发起他评请求(默认请直属上级评)。"""
    ratee = db.get(Employee, require_linked_employee(db, current))
    q = _create_rating_request(
        db, current=current, ratee=ratee, payload=payload, request=request)
    db.commit()
    return _req_out(db, q)


@router.post("/employees/{emp_id}/skill-rating-requests", status_code=201)
def request_rating_for(
    emp_id: str, payload: RatingRequestIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """HR/直属上级代员工发起评分请求(校准场景)。"""
    emp = _get_employee(db, emp_id, current)
    if not _can_curate(db, emp, current):
        raise HTTPException(403, "权限不足")
    q = _create_rating_request(
        db, current=current, ratee=emp, payload=payload, request=request)
    db.commit()
    return _req_out(db, q)


@router.get("/me/skill-rating-requests")
def my_rating_requests(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """我发出的评分请求(被评人=我)。"""
    me = require_linked_employee(db, current)
    rows = db.scalars(
        select(SkillRatingRequest).where(
            SkillRatingRequest.tenant_id == current.tenant_id,
            SkillRatingRequest.employee_id == me,
        ).order_by(SkillRatingRequest.created_at.desc()).limit(50)
    ).all()
    return {"items": [_req_out(db, q) for q in rows]}


@router.get("/me/skill-rating-tasks")
def my_rating_tasks(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """待我评价(评价人=我):pending 在前,已处理的按时间倒序附后。"""
    me = require_linked_employee(db, current)
    rows = db.scalars(
        select(SkillRatingRequest).where(
            SkillRatingRequest.tenant_id == current.tenant_id,
            SkillRatingRequest.rater_employee_id == me,
        ).order_by(SkillRatingRequest.created_at.desc()).limit(50)
    ).all()
    items = [_req_out(db, q) for q in rows]
    items.sort(key=lambda x: (x["status"] != "pending", x["created_at"]),
               reverse=False)
    return {"items": items,
            "pending_count": sum(1 for i in items if i["status"] == "pending")}


@router.post("/me/skill-rating-requests/{req_id}/cancel")
def cancel_rating_request(
    req_id: str, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """撤回待处理请求:发起者本人、被评人本人或 HR/直属上级可撤。"""
    me = get_linked_employee_id(db, current)
    q = db.get(SkillRatingRequest, req_id)
    if not q or q.tenant_id != current.tenant_id:
        raise HTTPException(404, "评分请求不存在")
    ratee = db.get(Employee, q.employee_id)
    allowed = (
        q.requested_by_id == current.id
        or (me is not None and q.employee_id == me)
        or _can_curate(db, ratee, current)
    )
    if not allowed:
        raise HTTPException(403, "权限不足")
    if q.status != "pending":
        raise HTTPException(409, "只能撤回待处理的请求")
    q.status = "cancelled"
    q.decided_at = _now()
    write_audit(db, actor=current, entity_type="skill_rating_request",
                entity_id=q.id, action="cancel", request=request)
    skill = db.get(Skill, q.skill_id)
    _notify_emp(db, current.tenant_id, q.rater_employee_id,
                title=f"「{skill.name if skill else ''}」的评分请求已撤回")
    db.commit()
    return _req_out(db, q)


class RateIn(BaseModel):
    level: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=500)


class DeclineIn(BaseModel):
    reason: str | None = Field(default=None, max_length=200)


def _upsert_rating(
    db: Session, *, tenant_id: str, ratee_id: str, skill_id: str,
    rater_id: str, rater_kind: str, level: int, comment: str | None,
    request_id: str | None, created_by: str,
) -> SkillRating:
    """每人每技能一条:重复评分覆盖(最新一次为准)。"""
    r = db.scalars(
        select(SkillRating).where(
            SkillRating.employee_id == ratee_id,
            SkillRating.skill_id == skill_id,
            SkillRating.rater_employee_id == rater_id,
        )
    ).first()
    if r is None:
        r = SkillRating(
            tenant_id=tenant_id, employee_id=ratee_id, skill_id=skill_id,
            rater_employee_id=rater_id, rater_kind=rater_kind, level=level,
            comment=comment, request_id=request_id, created_by=created_by,
        )
        db.add(r)
    else:
        r.level = level
        r.comment = comment
        r.rater_kind = rater_kind
        if request_id:
            r.request_id = request_id
    db.flush()
    return r


def _reflect_to_employee_skill(
    db: Session, *, tenant_id: str, ratee_id: str, skill_id: str,
    rater_kind: str, level: int, rating_id: str, comment: str | None,
    created_by: str,
) -> None:
    """评分回写 EmployeeSkill:人评即 confirmed 证据(source=自评/经理/同事)。"""
    profile_update.upsert_employee_skill(
        db, tenant_id=tenant_id, employee_id=ratee_id, skill_id=skill_id,
        level=level, source=rater_kind, status="confirmed",
        evidence={"kind": "rating", "id": rating_id, "rater_kind": rater_kind},
        note=comment, created_by=created_by,
    )


@router.post("/me/skill-rating-tasks/{task_id}/rate", status_code=201)
def submit_rating(
    task_id: str, payload: RateIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """评价人提交打分 → SkillRating + EmployeeSkill(confirmed) + 请求 completed。"""
    me = require_linked_employee(db, current)
    q = _get_task(db, task_id, current)
    if q.status != "pending":
        raise HTTPException(409, "该请求已处理,不能重复评分")
    r = _upsert_rating(
        db, tenant_id=current.tenant_id, ratee_id=q.employee_id,
        skill_id=q.skill_id, rater_id=me, rater_kind=q.rater_kind,
        level=payload.level, comment=payload.comment,
        request_id=q.id, created_by=current.id,
    )
    _reflect_to_employee_skill(
        db, tenant_id=current.tenant_id, ratee_id=q.employee_id,
        skill_id=q.skill_id, rater_kind=q.rater_kind, level=payload.level,
        rating_id=r.id, comment=payload.comment, created_by=current.id,
    )
    q.status = "completed"
    q.rating_id = r.id
    q.decided_at = _now()
    write_audit(db, actor=current, entity_type="skill_rating_request",
                entity_id=q.id, action="rate",
                detail={"level": payload.level, "rater_kind": q.rater_kind},
                request=request)
    skill = db.get(Skill, q.skill_id)
    rater = db.get(Employee, me)
    _notify_emp(
        db, current.tenant_id, q.employee_id,
        title=f"{rater.name if rater else '同事'} 为你的「{skill.name if skill else ''}」评了 {payload.level} 级",
        body=payload.comment or "",
    )
    # 代发场景(请求人≠被评人):结果也知会请求人
    if q.requested_by_id and q.requested_by_id != current.id:
        ratee = db.get(Employee, q.employee_id)
        if not ratee or ratee.user_id != q.requested_by_id:
            db.add(Notification(
                tenant_id=current.tenant_id, user_id=q.requested_by_id,
                kind="skill_rating",
                title=f"{ratee.name if ratee else ''} 的「{skill.name if skill else ''}」已被 {rater.name if rater else '同事'} 评 {payload.level} 级"[:128],
                body=(payload.comment or "")[:512], link="/me/profile",
            ))
    db.commit()
    return {"request": _req_out(db, q), "rating": _rating_out(db, r)}


@router.post("/me/skill-rating-tasks/{task_id}/decline")
def decline_rating(
    task_id: str, payload: DeclineIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """评价人婉拒 → 请求 declined,通知发起方。"""
    q = _get_task(db, task_id, current)
    if q.status != "pending":
        raise HTTPException(409, "该请求已处理")
    q.status = "declined"
    q.decided_at = _now()
    write_audit(db, actor=current, entity_type="skill_rating_request",
                entity_id=q.id, action="decline", request=request)
    me = require_linked_employee(db, current)
    rater = db.get(Employee, me)
    skill = db.get(Skill, q.skill_id)
    _notify_emp(
        db, current.tenant_id, q.employee_id,
        title=f"{rater.name if rater else '同事'} 婉拒了「{skill.name if skill else ''}」的评分请求",
        body=payload.reason or "",
    )
    db.commit()
    return _req_out(db, q)


@router.post("/me/skills/{skill_id}/rate", status_code=201)
def self_rate(
    skill_id: str, payload: RateIn, request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """自评:不走请求流,直接落 self 评分 + EmployeeSkill(source=self)。"""
    me = require_linked_employee(db, current)
    skill = db.get(Skill, skill_id)
    if not skill or skill.tenant_id != current.tenant_id:
        raise HTTPException(404, "技能不存在")
    r = _upsert_rating(
        db, tenant_id=current.tenant_id, ratee_id=me, skill_id=skill.id,
        rater_id=me, rater_kind="self", level=payload.level,
        comment=payload.comment, request_id=None, created_by=current.id,
    )
    _reflect_to_employee_skill(
        db, tenant_id=current.tenant_id, ratee_id=me, skill_id=skill.id,
        rater_kind="self", level=payload.level, rating_id=r.id,
        comment=payload.comment, created_by=current.id,
    )
    write_audit(db, actor=current, entity_type="skill_rating_request",
                entity_id=r.id, action="self_rate",
                detail={"level": payload.level, "skill_id": skill.id},
                request=request)
    db.commit()
    return _rating_out(db, r)


@router.get("/employees/{emp_id}/skill-ratings")
def list_ratings(
    emp_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """某员工的全部多源评分(按技能分组);可见性同 talent-card。"""
    emp = _get_employee(db, emp_id, current)
    rows = db.scalars(
        select(SkillRating).where(
            SkillRating.employee_id == emp.id,
        ).order_by(SkillRating.created_at.desc()).limit(200)
    ).all()
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_skill.setdefault(r.skill_id, []).append(_rating_out(db, r))
    return {
        "items": [_rating_out(db, r) for r in rows],
        "by_skill": by_skill,
        "pending_requests": [
            _req_out(db, q) for q in db.scalars(
                select(SkillRatingRequest).where(
                    SkillRatingRequest.employee_id == emp.id,
                    SkillRatingRequest.status == "pending",
                ).order_by(SkillRatingRequest.created_at.desc())
            ).all()
        ],
    }
