"""生成文档的统一落库:对象存储 + generated_documents 记录。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.domain.models import GeneratedDocument, User
from app.infra.storage import build_object_store

MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


async def save_document(
    db: Session,
    *,
    tenant_id: str,
    kind: str,
    title: str,
    file_name: str,
    data: bytes,
    params: dict[str, Any],
    actor: User | None,
) -> GeneratedDocument:
    ext = file_name.rsplit(".", 1)[-1].lower()
    mime = MIME.get(ext, "application/octet-stream")
    now = datetime.utcnow()
    storage_key = (
        f"docs/{tenant_id}/{now.year:04d}/{now.month:02d}/{uuid.uuid4().hex}.{ext}"
    )
    await build_object_store().put(storage_key, data, content_type=mime)
    doc = GeneratedDocument(
        tenant_id=tenant_id,
        kind=kind,
        title=title,
        file_name=file_name,
        file_size=len(data),
        mime=mime,
        storage_key=storage_key,
        params=params,
        created_by_user_id=actor.id if actor else None,
    )
    db.add(doc)
    db.flush()
    return doc
