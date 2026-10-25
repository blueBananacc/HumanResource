"""岗位级知识源 API(T11)。

挂在 ``/api/jobs/{job_id}/knowledge-sources`` 下:

- GET    /                     源列表(含状态/chunk 数/抽取信息)
- POST   /text                 添加内联文本源
- POST   /file                 上传文件源(pdf/docx/txt/md,<=5MB)
- POST   /{sid}/reindex        重新索引(文本源重跑切块;文件源重取 ObjectStore)
- DELETE /{sid}                删除源+其 chunks
- GET    /chunks               chunks 预览(召回调试)
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.domain.models import Job, JobKnowledgeChunk, JobKnowledgeSource, User
from app.infra.db import get_db
from app.infra.storage import build_object_store
from app.services.job_knowledge import (
    create_file_source,
    create_text_source,
    delete_source,
    index_source,
)
from app.services.permissions import (
    PERM_WRITE_JOBS,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(prefix="/jobs/{job_id}/knowledge-sources", tags=["job-knowledge"])
logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 5 * 1024 * 1024
_ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt", ".md"}


class SourceOut(BaseModel):
    id: str
    name: str
    kind: str
    status: str
    chunk_count: int
    extraction_status: str
    extraction_method: str | None
    error_message: str | None
    file_name: str | None
    created_at: datetime
    updated_at: datetime


class TextSourceIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=40000)


class ChunkOut(BaseModel):
    id: str
    source_id: str
    source_name: str
    chunk_index: int
    content: str
    has_embedding: bool


def _source_out(s: JobKnowledgeSource) -> SourceOut:
    return SourceOut(
        id=s.id,
        name=s.name,
        kind=s.kind,
        status=s.status,
        chunk_count=s.chunk_count,
        extraction_status=s.extraction_status,
        extraction_method=s.extraction_method,
        error_message=s.error_message,
        file_name=s.file_name,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _get_job(db: Session, job_id: str, current: User) -> Job:
    job = db.get(Job, job_id)
    if not job or job.tenant_id != current.tenant_id:
        raise HTTPException(404, "岗位不存在")
    ensure_can_see(get_org_scope(db, current), job.org_unit_id)
    return job


def _get_source(
    db: Session, job_id: str, sid: str, tenant_id: str
) -> JobKnowledgeSource:
    src = db.get(JobKnowledgeSource, sid)
    if not src or src.job_id != job_id or src.tenant_id != tenant_id:
        raise HTTPException(404, "知识源不存在")
    return src


@router.get("/", response_model=list[SourceOut])
def list_sources(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    _get_job(db, job_id, current)
    rows = db.scalars(
        select(JobKnowledgeSource)
        .where(
            JobKnowledgeSource.job_id == job_id,
            JobKnowledgeSource.tenant_id == current.tenant_id,
        )
        .order_by(JobKnowledgeSource.created_at)
    ).all()
    return [_source_out(s) for s in rows]


@router.post("/text", response_model=SourceOut, status_code=201)
def add_text_source(
    job_id: str,
    body: TextSourceIn,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_JOBS)),
):
    job = _get_job(db, job_id, current)
    src = create_text_source(
        db, job=job, name=body.name.strip(), content=body.content, actor=current
    )
    db.commit()
    return _source_out(src)


@router.post("/file", response_model=SourceOut, status_code=201)
async def add_file_source(
    job_id: str,
    file: UploadFile,
    name: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_JOBS)),
):
    import pathlib
    import uuid

    job = _get_job(db, job_id, current)
    file_name = file.filename or "document"
    ext = pathlib.Path(file_name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext},允许 {sorted(_ALLOWED_EXT)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件为空")
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(400, "文件超过 5MB 上限")

    storage_key = f"job-knowledge/{job.tenant_id}/{job_id}/{uuid.uuid4().hex[:12]}{ext}"
    store = build_object_store()
    await store.put(storage_key, data, content_type=file.content_type)
    src = create_file_source(
        db,
        job=job,
        name=(name or file_name).strip()[:128],
        file_name=file_name,
        mime=file.content_type or "application/octet-stream",
        storage_key=storage_key,
        file_bytes=data,
        actor=current,
    )
    db.commit()
    return _source_out(src)


@router.post("/{sid}/reindex", response_model=SourceOut)
async def reindex_source(
    job_id: str,
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_JOBS)),
):
    _get_job(db, job_id, current)
    src = _get_source(db, job_id, sid, current.tenant_id)
    file_bytes = None
    if src.kind == "file":
        if not src.storage_key:
            raise HTTPException(409, "文件源缺少存储引用,请删除后重新上传")
        try:
            file_bytes = await build_object_store().get(src.storage_key)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(409, f"无法读取原始文件: {e}") from e
    src.status = "indexing"
    index_source(db, src, file_bytes=file_bytes, actor=current)
    db.commit()
    return _source_out(src)


@router.delete("/{sid}", status_code=204)
async def remove_source(
    job_id: str,
    sid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_WRITE_JOBS)),
):
    _get_job(db, job_id, current)
    src = _get_source(db, job_id, sid, current.tenant_id)
    if src.storage_key:
        try:
            await build_object_store().delete(src.storage_key)
        except Exception as e:  # noqa: BLE001
            logger.warning("知识源文件删除失败 key=%s err=%s", src.storage_key, e)
    delete_source(db, src, actor=current)
    db.commit()


@router.get("/chunks", response_model=list[ChunkOut])
def list_chunk_preview(
    job_id: str,
    source_id: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    _get_job(db, job_id, current)
    stmt = (
        select(JobKnowledgeChunk, JobKnowledgeSource.name)
        .join(
            JobKnowledgeSource,
            (JobKnowledgeSource.id == JobKnowledgeChunk.source_id)
            & (JobKnowledgeSource.tenant_id == current.tenant_id),
        )
        .where(
            JobKnowledgeChunk.job_id == job_id,
            JobKnowledgeChunk.tenant_id == current.tenant_id,
        )
        .order_by(JobKnowledgeChunk.source_id, JobKnowledgeChunk.chunk_index)
        .limit(80)
    )
    if source_id:
        stmt = stmt.where(JobKnowledgeChunk.source_id == source_id)
    return [
        ChunkOut(
            id=c.id,
            source_id=c.source_id,
            source_name=name,
            chunk_index=c.chunk_index,
            content=c.content,
            has_embedding=c.embedding is not None,
        )
        for c, name in db.execute(stmt).all()
    ]
