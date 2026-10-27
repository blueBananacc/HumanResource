"""岗位级知识 chunks(T11)。

移植 recruit `jobs/service.js` 的 `splitKnowledgeChunks` + `indexKnowledgeSource`,
适配我方 SQLAlchemy/多租户:

- 岗位可挂多个知识源(kind=text 内联文本 / kind=file 上传文件)
- 索引:文本规范化 → 段落贪心装箱切块(<=40 块/源) → best-effort embedding
  → ``job_knowledge_chunks`` 可丢弃投影 → 状态机 indexing→ready/failed
- 召回 ``job_knowledge_context``:有 query 且 chunk 带 embedding → cosine top-N;
  否则按 (source, idx) 顺序截断。无 ready 源时返回空列表,调用方零回归
- 审计 ``job_knowledge.indexed`` / ``index_failed`` 写 ``write_audit``,
  进 T7 自动化漏斗
"""
from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domain.models import Job, JobKnowledgeChunk, JobKnowledgeSource, User
from app.services.audit import write_audit
from app.services.embedding import cosine, embed_text, embed_texts
from app.services.resume_parser import _extract_text_by_format

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 900
_CHUNK_CAP = 40
_MAX_SOURCE_CHARS = 40000

_EXTRACTION_METHODS = {
    ".pdf": "pymupdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "txt",
}


_HEADING_RE = re.compile(r"^#{1,6}\s")


def split_chunks_meta(
    text: str, size: int = _CHUNK_SIZE, cap: int = _CHUNK_CAP
) -> list[dict[str, Any]]:
    """结构感知分块(P2):返回 [{content, section}]。

    - markdown 标题行(#...)开新块,标题与其正文粘在同一块里;
    - 每块携带所在小节的标题文本(最近一个上游标题),写入 chunk meta,
      检索命中时可回溯「哪份文档的哪一节」;
    - 其余规则同旧版:规范化→空行分段→贪心装箱→超长硬切。
    """
    normalized = re.sub(r"[\x00]", " ", text or "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[dict[str, Any]] = []
    current = ""
    cur_section: str | None = None  # 最近见到的标题(文本)
    chunk_section: str | None = None  # 当前块所属小节

    def _flush() -> None:
        nonlocal current
        if current:
            chunks.append({"content": current, "section": chunk_section})
            current = ""

    for para in paragraphs or [normalized]:
        first_line = para.split("\n", 1)[0].strip()
        is_heading = _HEADING_RE.match(first_line) is not None
        if is_heading:
            _flush()
            cur_section = first_line.lstrip("#").strip() or None
            chunk_section = cur_section
            current = para
        elif not current:
            current = para
            chunk_section = cur_section
        elif len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}"
        else:
            _flush()
            current = para
            chunk_section = cur_section
        while len(current) > size:
            chunks.append({"content": current[:size], "section": chunk_section})
            current = current[size:]
    _flush()
    return chunks[:cap]


def split_chunks(text: str, size: int = _CHUNK_SIZE, cap: int = _CHUNK_CAP) -> list[str]:
    """移植 recruit splitKnowledgeChunks:规范化→空行分段→贪心装箱→超长硬切。
    P2 起底层走结构感知分块,本签名保留以兼容调用方/测试。"""
    return [c["content"] for c in split_chunks_meta(text, size, cap)]


def _extract_source_text(
    source: JobKnowledgeSource, file_bytes: bytes | None
) -> tuple[str, str, str | None]:
    """返回 (text, extraction_status, method)。

    status: ok / failed / none(text 源不需要抽取)
    """
    if source.kind == "text":
        return (source.content or "").strip(), "none", None
    if file_bytes is None:
        return "", "failed", None
    import pathlib

    ext = pathlib.Path(source.file_name or "").suffix.lower()
    method = _EXTRACTION_METHODS.get(ext, "txt")
    try:
        text = _extract_text_by_format(
            source.file_name or "file.txt",
            source.mime or "text/plain",
            file_bytes,
            pdf_pages=50,
        )
        return text.strip()[:_MAX_SOURCE_CHARS], "ok", method
    except Exception as e:  # noqa: BLE001
        logger.warning("知识源抽取失败 source=%s err=%s", source.id, e)
        return "", "failed", method


def index_source(
    db: Session,
    source: JobKnowledgeSource,
    *,
    file_bytes: bytes | None = None,
    actor: User | None = None,
) -> JobKnowledgeSource:
    """索引一个知识源:抽取→切块→写 chunks→更新状态。幂等(删旧插新)。"""
    text, ext_status, ext_method = _extract_source_text(source, file_bytes)
    chunk_objs = split_chunks_meta(text)
    chunks = [c["content"] for c in chunk_objs]
    ready = bool(chunks) and (
        source.kind == "text" or (ext_status == "ok" and text)
    )

    db.execute(
        delete(JobKnowledgeChunk).where(
            JobKnowledgeChunk.job_id == source.job_id,
            JobKnowledgeChunk.source_id == source.id,
        )
    )
    if ready:
        embeddings = embed_texts(db, source.tenant_id, chunks) or []
        for idx, c in enumerate(chunk_objs):
            db.add(
                JobKnowledgeChunk(
                    tenant_id=source.tenant_id,
                    job_id=source.job_id,
                    source_id=source.id,
                    chunk_index=idx,
                    content=c["content"],
                    meta={
                        "source_name": source.name,
                        "section": c.get("section"),
                        "extraction_status": ext_status,
                        "extraction_method": ext_method,
                    },
                    embedding=embeddings[idx] if idx < len(embeddings) else None,
                )
            )

    source.status = "ready" if ready else "failed"
    source.chunk_count = len(chunks) if ready else 0
    source.extraction_status = ext_status
    source.extraction_method = ext_method
    source.error_message = (
        None
        if ready
        else ("未抽取到有效文本" if source.kind == "file" else "文本内容为空")
    )
    if actor is not None:
        write_audit(
            db,
            actor=actor,
            entity_type="job_knowledge_source",
            entity_id=source.id,
            action="job_knowledge.indexed" if ready else "job_knowledge.index_failed",
            detail={
                "job_id": source.job_id,
                "chunks": source.chunk_count,
                "extraction_status": ext_status,
                "extraction_method": ext_method,
            },
            result="success" if ready else "failed",
        )
    return source


def create_text_source(
    db: Session, *, job: Job, name: str, content: str, actor: User
) -> JobKnowledgeSource:
    source = JobKnowledgeSource(
        tenant_id=job.tenant_id,
        job_id=job.id,
        name=name[:128],
        kind="text",
        content=content,
        status="indexing",
        created_by=actor.id,
    )
    db.add(source)
    db.flush()
    return index_source(db, source, actor=actor)


def create_file_source(
    db: Session,
    *,
    job: Job,
    name: str,
    file_name: str,
    mime: str,
    storage_key: str,
    file_bytes: bytes,
    actor: User,
) -> JobKnowledgeSource:
    source = JobKnowledgeSource(
        tenant_id=job.tenant_id,
        job_id=job.id,
        name=name[:128],
        kind="file",
        file_name=file_name,
        mime=mime,
        storage_key=storage_key,
        status="indexing",
        created_by=actor.id,
    )
    db.add(source)
    db.flush()
    return index_source(db, source, file_bytes=file_bytes, actor=actor)


def delete_source(
    db: Session, source: JobKnowledgeSource, *, actor: User | None = None
) -> None:
    db.execute(
        delete(JobKnowledgeChunk).where(
            JobKnowledgeChunk.job_id == source.job_id,
            JobKnowledgeChunk.source_id == source.id,
        )
    )
    if actor is not None:
        write_audit(
            db,
            actor=actor,
            entity_type="job_knowledge_source",
            entity_id=source.id,
            action="job_knowledge.deleted",
            detail={"job_id": source.job_id, "name": source.name},
        )
    db.delete(source)


def list_chunks(
    db: Session, job_id: str, *, tenant_id: str, limit: int = 40
) -> list[JobKnowledgeChunk]:
    return list(
        db.scalars(
            select(JobKnowledgeChunk)
            .where(
                JobKnowledgeChunk.job_id == job_id,
                JobKnowledgeChunk.tenant_id == tenant_id,
            )
            .order_by(JobKnowledgeChunk.source_id, JobKnowledgeChunk.chunk_index)
            .limit(limit)
        )
    )


_QVEC_TTL_S = 600
_qvec_cache: dict[tuple[str, str], tuple[float, list[float] | None]] = {}


def _query_vec(db: Session, tenant_id: str, query: str) -> list[float] | None:
    """query embedding 带 10min TTL 缓存 —— 批量评估(50 简历×同岗位)不重复打 embedding API。"""
    import time

    key = (tenant_id, query[:512])
    hit = _qvec_cache.get(key)
    if hit and time.monotonic() - hit[0] < _QVEC_TTL_S:
        return hit[1]
    vec = embed_text(db, tenant_id, query)
    _qvec_cache[key] = (time.monotonic(), vec)
    if len(_qvec_cache) > 512:
        _qvec_cache.clear()
    return vec


def job_knowledge_context(
    db: Session,
    job: Job,
    *,
    query: str | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """召回岗位知识片段供 prompt 注入。

    有 query 且存在 embedding → cosine top-N;否则按序截断。
    返回 [{source_name, content, score}];无 ready 源时 []。
    """
    ready_sources = db.scalars(
        select(JobKnowledgeSource.id).where(
            JobKnowledgeSource.job_id == job.id,
            JobKnowledgeSource.tenant_id == job.tenant_id,
            JobKnowledgeSource.status == "ready",
        )
    ).all()
    if not ready_sources:
        return []
    chunks = list(
        db.scalars(
            select(JobKnowledgeChunk).where(
                JobKnowledgeChunk.job_id == job.id,
                JobKnowledgeChunk.tenant_id == job.tenant_id,
                JobKnowledgeChunk.source_id.in_(ready_sources),
            )
        )
    )
    if not chunks:
        return []

    scored: list[tuple[float, JobKnowledgeChunk]] = []
    qvec = _query_vec(db, job.tenant_id, query) if query else None
    if qvec:
        for c in chunks:
            scored.append((cosine(qvec, c.embedding), c))
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        chunks.sort(key=lambda c: (c.source_id, c.chunk_index))
        scored = [(0.0, c) for c in chunks]

    return [
        {
            "source_name": (c.meta or {}).get("source_name", "岗位资料"),
            "content": c.content,
            "score": round(s, 4),
        }
        for s, c in scored[:limit]
    ]


def knowledge_context_text(
    db: Session, job: Job, *, query: str | None = None, limit: int = 6,
    max_chars: int = 3000,
) -> str:
    """拼成可直接塞进 prompt 的文本块;无知识返回 ""。"""
    items = job_knowledge_context(db, job, query=query, limit=limit)
    if not items:
        return ""
    parts: list[str] = []
    used = 0
    for it in items:
        block = f"【{it['source_name']}】{it['content']}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    if not parts:
        return ""
    return "岗位参考资料:\n" + "\n\n".join(parts)
