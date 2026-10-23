"""问卷 API(参考 MintHCM Surveys)。

管理端(登录 + 权限):
- ``GET/POST /surveys``;``PUT /surveys/{id}``(title/desc/题目整组替换)
- ``POST /surveys/{id}/status``  draft→published→closed
- ``POST /surveys/{id}/links``   生成一次性填写链接(可绑候选人)
- ``GET  /surveys/{id}/links``   链接列表(状态:有效/已用/过期)
- ``DELETE /surveys/{id}/links/{lid}``  吊销链接
- ``GET  /surveys/{id}/results`` 每题聚合(选项计数 / 评分均值 / 文本采样)

公开端(免认证,凭一次性链接):
- ``GET  /public/surveys/{token}`` 题目 + 描述(不含任何租户数据)
- ``POST /public/surveys/{token}/responses`` 提交答卷并核销链接
  token 对应 ``survey_links.token_hash``(sha256),一次性 + 过期 + 可吊销;
  ``respondent_key`` 由前端 localStorage 生成 → UNIQUE 双保险防重复
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import (
    Candidate,
    Survey,
    SurveyLink,
    SurveyQuestion,
    SurveyResponse,
    SurveyResponseAnswer,
    User,
)
from app.infra.db import get_db
from app.infra.invite_tokens import generate_invite_token, hash_invite_token
from app.services.audit import write_audit
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    PERM_WRITE_INTERVIEW,
    require_permission,
)

router = APIRouter(prefix="/surveys", tags=["surveys"])
public_router = APIRouter(prefix="/public/surveys", tags=["surveys-public"])

QTYPES = SurveyQuestion.TYPES


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ============ schemas ============


class QuestionIn(BaseModel):
    type: str
    text: str = Field(..., min_length=1, max_length=256)
    options: list[str] = Field(default_factory=list, max_length=20)
    required: bool = True
    # None → 按数组下标定序;客户端全传 0 会导致排序退化到随机 id
    order: int | None = None


class QuestionOut(BaseModel):
    id: str
    type: str
    text: str
    options: list
    required: bool
    order: int


class SurveyIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(None, max_length=2000)


class SurveySaveIn(SurveyIn):
    questions: list[QuestionIn] = Field(default_factory=list, max_length=50)


class SurveyOut(BaseModel):
    id: str
    title: str
    description: str | None
    status: str
    question_count: int
    response_count: int
    created_at: datetime


class SurveyDetailOut(SurveyOut):
    questions: list[QuestionOut]


class StatusIn(BaseModel):
    status: str


class LinkIn(BaseModel):
    candidate_id: str | None = None
    expires_in_days: int = Field(30, ge=1, le=365)


class LinkOut(BaseModel):
    id: str
    status: str  # active / used / expired
    candidate_id: str | None
    candidate_name: str | None
    expires_at: datetime
    used_at: datetime | None
    created_at: datetime


class LinkCreatedOut(LinkOut):
    """明文 token 只在创建响应里出现一次 — 之后库中只留 sha256。"""

    token: str


class ResponseIn(BaseModel):
    respondent_key: str = Field(..., min_length=8, max_length=64)
    respondent_name: str | None = Field(None, max_length=64)
    happiness: int | None = Field(None, ge=1, le=5)
    answers: dict[str, object] = Field(default_factory=dict)


class PublicSurveyOut(BaseModel):
    title: str
    description: str | None
    questions: list[QuestionOut]


class QuestionResult(BaseModel):
    question_id: str
    type: str
    text: str
    answered: int
    option_counts: dict[str, int] | None = None
    avg_rating: float | None = None
    samples: list[str] = []


class RespondentOut(BaseModel):
    name: str | None
    candidate_id: str | None
    submitted_at: datetime
    happiness: int | None


class ResultsOut(BaseModel):
    survey_id: str
    response_count: int
    avg_happiness: float | None
    respondents: list[RespondentOut] = []
    questions: list[QuestionResult]


# ============ helpers ============


def _survey_or_404(db: Session, tenant_id: str, sid: str) -> Survey:
    s = db.get(Survey, sid)
    if not s or s.tenant_id != tenant_id:
        raise HTTPException(404, "问卷不存在")
    return s


def _counts(db: Session, sid: str) -> tuple[int, int]:
    q = db.scalar(
        select(func.count()).select_from(SurveyQuestion).where(
            SurveyQuestion.survey_id == sid
        )
    ) or 0
    r = db.scalar(
        select(func.count()).select_from(SurveyResponse).where(
            SurveyResponse.survey_id == sid
        )
    ) or 0
    return q, r


def _out(db: Session, s: Survey) -> SurveyOut:
    qn, rn = _counts(db, s.id)
    return SurveyOut(
        id=s.id, title=s.title, description=s.description, status=s.status,
        question_count=qn, response_count=rn, created_at=s.created_at,
    )


def _questions(db: Session, sid: str) -> list[SurveyQuestion]:
    return db.scalars(
        select(SurveyQuestion)
        .where(SurveyQuestion.survey_id == sid)
        .order_by(SurveyQuestion.order, SurveyQuestion.id)
    ).all()


def _validate_questions(items: list[QuestionIn]) -> None:
    for q in items:
        if q.type not in QTYPES:
            raise HTTPException(400, f"题目类型必须是 {QTYPES} 之一")
        if q.type in ("single_choice", "multi_choice") and len(q.options) < 2:
            raise HTTPException(400, f"「{q.text[:20]}」至少需要 2 个选项")
        if q.type not in ("single_choice", "multi_choice") and q.options:
            raise HTTPException(400, f"「{q.text[:20]}」不是选择题,不能带选项")


# ============ 管理端 ============


@router.get("", response_model=list[SurveyOut])
def list_surveys(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    rows = db.scalars(
        select(Survey)
        .where(Survey.tenant_id == current.tenant_id)
        .order_by(Survey.created_at.desc())
    ).all()
    return [_out(db, s) for s in rows]


@router.post("", response_model=SurveyDetailOut, status_code=201)
def create_survey(
    body: SurveySaveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    _validate_questions(body.questions)
    s = Survey(
        tenant_id=current.tenant_id, title=body.title.strip(),
        description=body.description, created_by=current.id,
    )
    db.add(s)
    db.flush()
    for i, q in enumerate(body.questions):
        db.add(
            SurveyQuestion(
                tenant_id=current.tenant_id, survey_id=s.id,
                type=q.type, text=q.text.strip(),
                options=q.options, required=q.required,
                order=q.order if q.order is not None else i,
            )
        )
    write_audit(db, actor=current, entity_type="survey", entity_id=s.id,
                action="create", detail={"title": s.title}, request=request)
    db.commit()
    db.refresh(s)
    return SurveyDetailOut(
        **_out(db, s).model_dump(),
        questions=[
            QuestionOut(id=q.id, type=q.type, text=q.text, options=q.options,
                        required=q.required, order=q.order)
            for q in _questions(db, s.id)
        ],
    )


@router.get("/{sid}", response_model=SurveyDetailOut)
def get_survey(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    s = _survey_or_404(db, current.tenant_id, sid)
    return SurveyDetailOut(
        **_out(db, s).model_dump(),
        questions=[
            QuestionOut(id=q.id, type=q.type, text=q.text, options=q.options,
                        required=q.required, order=q.order)
            for q in _questions(db, s.id)
        ],
    )


@router.put("/{sid}", response_model=SurveyDetailOut)
def update_survey(
    sid: str,
    body: SurveySaveIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    s = _survey_or_404(db, current.tenant_id, sid)
    # 仅草稿可编辑:published 改题影响在收的答卷;closed 已有沉淀答卷,
    # 整组替换题目会经 FK CASCADE 连坐删除 survey_response_answers
    if s.status != "draft":
        raise HTTPException(400, "仅草稿状态的问卷可以修改")
    _validate_questions(body.questions)
    s.title = body.title.strip()
    s.description = body.description
    s.updated_at = _now()
    db.query(SurveyQuestion).filter(SurveyQuestion.survey_id == s.id).delete()
    for i, q in enumerate(body.questions):
        db.add(
            SurveyQuestion(
                tenant_id=current.tenant_id, survey_id=s.id,
                type=q.type, text=q.text.strip(),
                options=q.options, required=q.required,
                order=q.order if q.order is not None else i,
            )
        )
    write_audit(db, actor=current, entity_type="survey", entity_id=s.id,
                action="update", detail={"title": s.title}, request=request)
    db.commit()
    return get_survey(sid, db, current)


@router.post("/{sid}/status", response_model=SurveyOut)
def set_status(
    sid: str,
    body: StatusIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    s = _survey_or_404(db, current.tenant_id, sid)
    allowed = {
        "draft": {"published"},
        "published": {"closed"},
        "closed": set(),
    }
    if body.status not in allowed.get(s.status, set()):
        raise HTTPException(400, f"不允许 {s.status} → {body.status}")
    if body.status == "published" and _counts(db, s.id)[0] == 0:
        raise HTTPException(400, "至少一道题才能发布")
    s.status = body.status
    s.updated_at = _now()
    write_audit(db, actor=current, entity_type="survey", entity_id=s.id,
                action="status", detail={"to": body.status}, request=request)
    db.commit()
    return _out(db, s)


# ============ 填写链接(一次性 token)============


def _link_status(link: SurveyLink) -> str:
    if link.used_at is not None:
        return "used"
    if link.expires_at <= _now():
        return "expired"
    return "active"


def _link_out(link: SurveyLink, cand_names: dict[str, str]) -> LinkOut:
    return LinkOut(
        id=link.id,
        status=_link_status(link),
        candidate_id=link.candidate_id,
        candidate_name=cand_names.get(link.candidate_id) if link.candidate_id else None,
        expires_at=link.expires_at,
        used_at=link.used_at,
        created_at=link.created_at,
    )


def _links_out(db: Session, links: list[SurveyLink]) -> list[LinkOut]:
    cand_ids = {l.candidate_id for l in links if l.candidate_id}
    cand_names = (
        {c.id: c.name for c in db.scalars(select(Candidate).where(Candidate.id.in_(cand_ids)))}
        if cand_ids else {}
    )
    return [_link_out(l, cand_names) for l in links]


@router.post("/{sid}/links", response_model=LinkCreatedOut, status_code=201)
def create_link(
    sid: str,
    body: LinkIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    """生成一次性填写链接。``candidate_id`` 可绑定候选人追踪答卷归属。"""
    s = _survey_or_404(db, current.tenant_id, sid)
    if s.status == "closed":
        raise HTTPException(400, "问卷已关闭,无法生成新链接")
    if body.candidate_id:
        cand = db.get(Candidate, body.candidate_id)
        if not cand or cand.tenant_id != current.tenant_id:
            raise HTTPException(404, "候选人不存在")
    token = generate_invite_token()
    link = SurveyLink(
        tenant_id=current.tenant_id,
        survey_id=s.id,
        token_hash=hash_invite_token(token),
        candidate_id=body.candidate_id,
        expires_at=_now() + timedelta(days=body.expires_in_days),
        created_by=current.id,
    )
    db.add(link)
    db.flush()
    write_audit(
        db, actor=current, entity_type="survey_link", entity_id=link.id,
        action="create",
        detail={"survey_id": s.id, "candidate_id": body.candidate_id},
        request=request,
    )
    db.commit()
    db.refresh(link)
    cand_names = (
        {c.id: c.name for c in db.scalars(
            select(Candidate).where(Candidate.id == link.candidate_id)
        )}
        if link.candidate_id else {}
    )
    out = _link_out(link, cand_names)
    return LinkCreatedOut(**out.model_dump(), token=token)


@router.get("/{sid}/links", response_model=list[LinkOut])
def list_links(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    s = _survey_or_404(db, current.tenant_id, sid)
    links = db.scalars(
        select(SurveyLink)
        .where(SurveyLink.survey_id == s.id)
        .order_by(SurveyLink.created_at.desc())
    ).all()
    return _links_out(db, links)


@router.delete("/{sid}/links/{lid}", status_code=204)
def revoke_link(
    sid: str,
    lid: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_WRITE_INTERVIEW)),
):
    """吊销链接:删除后明文 token 立即失效。"""
    s = _survey_or_404(db, current.tenant_id, sid)
    link = db.get(SurveyLink, lid)
    if not link or link.survey_id != s.id:
        raise HTTPException(404, "链接不存在")
    write_audit(
        db, actor=current, entity_type="survey_link", entity_id=lid,
        action="revoke", detail={"survey_id": s.id}, request=request,
    )
    db.delete(link)
    db.commit()


@router.get("/{sid}/results", response_model=ResultsOut)
def results(
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    _p=Depends(require_permission(PERM_VIEW_REPORTS)),
):
    s = _survey_or_404(db, current.tenant_id, sid)
    questions = _questions(db, s.id)
    responses = db.scalars(
        select(SurveyResponse)
        .where(SurveyResponse.survey_id == s.id)
        .order_by(SurveyResponse.submitted_at.desc())
    ).all()
    answers = db.execute(
        select(SurveyResponseAnswer, SurveyResponse)
        .join(SurveyResponse, SurveyResponse.id == SurveyResponseAnswer.response_id)
        .where(SurveyResponseAnswer.tenant_id == current.tenant_id,
               SurveyResponse.survey_id == s.id)
    ).all()

    by_q: dict[str, list] = {}
    for a, _r in answers:
        by_q.setdefault(a.question_id, []).append(a)

    # 答卷归属:自定义称呼优先,其次链接绑定的候选人名
    resp_cand_ids = {r.candidate_id for r in responses if r.candidate_id}
    cand_names = (
        {c.id: c.name for c in db.scalars(
            select(Candidate).where(Candidate.id.in_(resp_cand_ids))
        )}
        if resp_cand_ids else {}
    )
    respondents = [
        RespondentOut(
            name=r.respondent_name or cand_names.get(r.candidate_id),
            candidate_id=r.candidate_id,
            submitted_at=r.submitted_at,
            happiness=r.happiness,
        )
        for r in responses
    ]

    happy = [r.happiness for r in responses if r.happiness is not None]
    out_qs = []
    for q in questions:
        ans = by_q.get(q.id, [])
        item = QuestionResult(
            question_id=q.id, type=q.type, text=q.text, answered=len(ans),
        )
        if q.type in ("single_choice", "multi_choice"):
            counts = {str(o): 0 for o in q.options}
            for a in ans:
                vals = a.value if isinstance(a.value, list) else [a.value]
                for v in vals:
                    key = str(v)
                    counts[key] = counts.get(key, 0) + 1
            item.option_counts = counts
        elif q.type == "rating":
            nums = [a.value for a in ans if isinstance(a.value, int | float)]
            if nums:
                item.avg_rating = round(sum(nums) / len(nums), 2)
        else:
            item.samples = [str(a.value) for a in ans[:5] if a.value]
        out_qs.append(item)

    return ResultsOut(
        survey_id=s.id,
        response_count=len(responses),
        avg_happiness=round(sum(happy) / len(happy), 2) if happy else None,
        respondents=respondents,
        questions=out_qs,
    )


# ============ 公开端(免认证,凭一次性链接)============


def _resolve_link(db: Session, token: str) -> tuple[SurveyLink, Survey]:
    """token → (link, survey)。状态判定全部在这里:
    404 无效 token / 问卷未开放;410 链接过期;409 已核销。"""
    link = db.scalars(
        select(SurveyLink).where(SurveyLink.token_hash == hash_invite_token(token))
    ).first()
    if link is None:
        raise HTTPException(404, "链接无效")
    s = db.get(Survey, link.survey_id)
    if not s or s.status != "published":
        raise HTTPException(404, "问卷不存在或未开放")
    if link.used_at is not None:
        raise HTTPException(409, "链接已使用")
    if link.expires_at <= _now():
        raise HTTPException(410, "链接已过期")
    return link, s


@public_router.get("/{token}", response_model=PublicSurveyOut)
def public_get(token: str, db: Session = Depends(get_db)):
    _link, s = _resolve_link(db, token)
    return PublicSurveyOut(
        title=s.title,
        description=s.description,
        questions=[
            QuestionOut(id=q.id, type=q.type, text=q.text, options=q.options,
                        required=q.required, order=q.order)
            for q in _questions(db, s.id)
        ],
    )


@public_router.post("/{token}/responses", status_code=201)
def public_submit(
    token: str,
    body: ResponseIn,
    db: Session = Depends(get_db),
):
    link, s = _resolve_link(db, token)
    questions = {q.id: q for q in _questions(db, s.id)}

    # 先查重,再校验必填(重复提交直接 409)
    dup = db.scalars(
        select(SurveyResponse).where(
            SurveyResponse.survey_id == s.id,
            SurveyResponse.respondent_key == body.respondent_key,
        )
    ).first()
    if dup:
        raise HTTPException(409, "你已经提交过这份问卷")

    # 校验:必填 + 类型匹配
    for q in questions.values():
        v = body.answers.get(q.id)
        empty = v is None or v == "" or v == []
        if q.required and empty:
            raise HTTPException(400, f"「{q.text[:30]}」为必填")
        if empty:
            continue
        if q.type == "single_choice" and str(v) not in [str(o) for o in q.options]:
            raise HTTPException(400, f"「{q.text[:30]}」答案不在选项中")
        if q.type == "multi_choice" and (
            not isinstance(v, list)
            or len(v) > len(q.options)
            or not all(str(x) in [str(o) for o in q.options] for x in v)
        ):
            raise HTTPException(400, f"「{q.text[:30]}」答案不在选项中")
        if q.type == "rating" and (
            isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5
        ):
            raise HTTPException(400, f"「{q.text[:30]}」需要 1-5 整数评分")
        if q.type in ("short_text", "long_text") and (
            not isinstance(v, str) or len(v) > 5000
        ):
            raise HTTPException(400, f"「{q.text[:30]}」需要不超过 5000 字的文本")

    resp = SurveyResponse(
        tenant_id=s.tenant_id, survey_id=s.id,
        link_id=link.id, candidate_id=link.candidate_id,
        respondent_key=body.respondent_key,
        respondent_name=body.respondent_name,
        happiness=body.happiness,
    )
    db.add(resp)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "你已经提交过这份问卷")

    for qid, v in body.answers.items():
        if qid not in questions or v is None or v == "" or v == []:
            continue
        db.add(
            SurveyResponseAnswer(
                tenant_id=s.tenant_id, response_id=resp.id,
                question_id=qid, value=v,
            )
        )
    # 原子核销:并发提交同一链接时,只有一个 UPDATE 能命中 used_at IS NULL
    consumed = db.execute(
        update(SurveyLink)
        .where(SurveyLink.id == link.id, SurveyLink.used_at.is_(None))
        .values(used_at=_now())
    ).rowcount
    if consumed != 1:
        db.rollback()
        raise HTTPException(409, "链接已使用")
    db.commit()
    return {"ok": True}
