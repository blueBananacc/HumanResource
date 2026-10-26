"""P0 文档引擎 API:生成 + 列表 + 下载。

POST 端点生成二进制文档落对象存储并登记 generated_documents;
统一要求 ``PERM_EXPORT_DATA``(与简历下载同口径),写操作留审计。
"""
from __future__ import annotations

import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.license import require_feature
from app.domain.models import (
    Candidate,
    GeneratedDocument,
    Job,
    QuestionSet,
    Resume,
    User,
)
from app.infra.db import get_db
from app.infra.storage import ObjectNotFoundError, build_object_store
from app.services.audit import write_audit
from app.services.docgen import docx_docs, pptx_docs, store, xlsx_docs
from app.services.permissions import (
    PERM_EXPORT_DATA,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(
    prefix="/exports",
    tags=["exports"],
    dependencies=[
        Depends(require_feature("hr.suite")),
        Depends(require_permission(PERM_EXPORT_DATA)),
    ],
)


class DocOut(BaseModel):
    id: str
    kind: str
    title: str
    file_name: str
    file_size: int
    created_at: str


def _out(d: GeneratedDocument) -> DocOut:
    return DocOut(
        id=d.id,
        kind=d.kind,
        title=d.title,
        file_name=d.file_name,
        file_size=d.file_size,
        created_at=d.created_at.isoformat(),
    )


@router.get("/", response_model=list[DocOut])
def list_documents(
    limit: int = 50,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    return [
        _out(d)
        for d in db.scalars(
            select(GeneratedDocument)
            .where(GeneratedDocument.tenant_id == current.tenant_id)
            .order_by(GeneratedDocument.created_at.desc())
            .limit(min(limit, 200))
        ).all()
    ]


async def _emit(
    db: Session,
    request: Request,
    current: User,
    *,
    kind: str,
    title: str,
    file_name: str,
    data: bytes,
    params: dict,
) -> DocOut:
    doc = await store.save_document(
        db,
        tenant_id=current.tenant_id,
        kind=kind,
        title=title,
        file_name=file_name,
        data=data,
        params=params,
        actor=current,
    )
    write_audit(
        db,
        actor=current,
        entity_type="generated_document",
        entity_id=doc.id,
        action="generate",
        detail={"kind": kind, "title": title, "file_name": file_name, "size": len(data)},
        request=request,
    )
    db.commit()
    return _out(doc)


class PayrollIn(BaseModel):
    period: str = Field(pattern=r"^\d{4}-\d{2}$", description="YYYY-MM")


@router.post("/payroll.xlsx", response_model=DocOut)
async def gen_payroll(
    body: PayrollIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    data = xlsx_docs.payroll_sheet(db, current.tenant_id, body.period)
    return await _emit(
        db, request, current,
        kind="payroll_sheet", title=f"{body.period} 薪酬底表",
        file_name=f"payroll_{body.period}.xlsx", data=data,
        params={"period": body.period},
    )


@router.post("/candidates.xlsx", response_model=DocOut)
async def gen_candidates(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    data = xlsx_docs.candidate_list(db, current.tenant_id, get_org_scope(db, current))
    return await _emit(
        db, request, current,
        kind="candidate_list", title="候选人清单",
        file_name="candidates.xlsx", data=data, params={},
    )


class PipelineIn(BaseModel):
    job_id: str | None = None


@router.post("/pipeline.xlsx", response_model=DocOut)
async def gen_pipeline(
    body: PipelineIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    scope = get_org_scope(db, current)
    if body.job_id:
        job = db.get(Job, body.job_id)
        if not job or job.tenant_id != current.tenant_id:
            raise HTTPException(404, "岗位不存在")
        ensure_can_see(scope, job.org_unit_id)
    data = xlsx_docs.pipeline_sheet(db, current.tenant_id, body.job_id, scope)
    return await _emit(
        db, request, current,
        kind="pipeline", title="招聘管道明细",
        file_name="pipeline.xlsx", data=data,
        params={"job_id": body.job_id},
    )


@router.post("/candidate-report/{candidate_id}", response_model=DocOut)
async def gen_candidate_report(
    candidate_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    c = db.get(Candidate, candidate_id)
    if not c or c.tenant_id != current.tenant_id:
        raise HTTPException(404, "候选人不存在")
    ensure_can_see(get_org_scope(db, current), c.org_unit_id)
    try:
        data = docx_docs.candidate_report(db, current.tenant_id, candidate_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    safe = re.sub(r'[\\/:*?"<>|]', "_", c.name)
    return await _emit(
        db, request, current,
        kind="candidate_report", title=f"人才分析报告:{c.name}",
        file_name=f"candidate_report_{safe}.docx", data=data,
        params={"candidate_id": candidate_id},
    )


class ExamIn(BaseModel):
    with_answers: bool = True


@router.post("/exam-paper/{question_set_id}", response_model=DocOut)
async def gen_exam_paper(
    question_set_id: str,
    body: ExamIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    qs = db.get(QuestionSet, question_set_id)
    if not qs or qs.tenant_id != current.tenant_id:
        raise HTTPException(404, "题集不存在")
    resume = db.get(Resume, qs.resume_id)
    if resume:
        cand = db.get(Candidate, resume.candidate_id)
        if cand:
            ensure_can_see(get_org_scope(db, current), cand.org_unit_id)
    try:
        data = docx_docs.exam_paper(db, current.tenant_id, question_set_id, body.with_answers)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return await _emit(
        db, request, current,
        kind="exam_paper",
        title=f"笔试卷({'含答案' if body.with_answers else '考生版'})",
        file_name=f"exam_{question_set_id[:8]}.docx", data=data,
        params={"question_set_id": question_set_id, "with_answers": body.with_answers},
    )


@router.post("/recruiting-pack", response_model=DocOut)
async def gen_recruiting_pack(
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    data = pptx_docs.recruiting_pack(db, current.tenant_id)
    return await _emit(
        db, request, current,
        kind="recruiting_pack", title="招聘进度汇报",
        file_name="recruiting_pack.pptx", data=data, params={},
    )


@router.get("/{doc_id}/download")
async def download_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
) -> Response:
    doc = db.get(GeneratedDocument, doc_id)
    if not doc or doc.tenant_id != current.tenant_id:
        raise HTTPException(404, "文档不存在")
    try:
        data = await build_object_store().get(doc.storage_key)
    except ObjectNotFoundError:
        raise HTTPException(410, "文件已被清理") from None
    fn = doc.file_name
    ascii_fn = re.sub(r"[^\x20-\x7e]", "_", fn) or "document"
    return Response(
        content=data,
        media_type=doc.mime,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_fn}"; '
                f"filename*=UTF-8''{quote(fn, safe='')}"
            )
        },
    )

