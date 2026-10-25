"""AI 导入向导 API(T8)。

四步:上传 → AI/启发式列映射 → 校验预览 → 执行+错误文件。
旧 ``/employees/import`` ``/talents/import`` 保留:一键 CSV 快速通道,
列名精确匹配的老文件不必走进向导。

权限:按数据集映射(manage_employees / write_resumes);AI 建议在无
``assistant.agent`` license 时自动降级为启发式,向导不因 LLM 中断。
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.idempotency import idempotent
from app.domain.models import ImportJob, OrgUnit, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.imports import (
    IMPORT_DATASETS,
    MAX_BYTES,
    MAX_ROWS,
    DatasetSpec,
    UploadError,
    ai_check_note,
    ai_suggest,
    apply_mapping,
    check_rows,
    error_csv,
    heuristic_suggest,
    parse_file,
    run_import,
    sample_values,
    _ctx,
    _normalize_choice,
)
from app.services.permissions import has_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/imports", tags=["imports"])


def _spec(key: str) -> DatasetSpec:
    spec = IMPORT_DATASETS.get(key)
    if spec is None:
        raise HTTPException(404, f"未知数据集: {key}")
    return spec


def _perm(current: User, spec: DatasetSpec) -> None:
    if not has_permission(current, spec.permission):
        raise HTTPException(403, f"没有 {spec.label} 导入权限")


def _suggest_options(db: Session, tenant_id: str) -> dict[str, list[str]]:
    return {
        "org_units": sorted(
            db.scalars(
                select(OrgUnit.name).where(OrgUnit.tenant_id == tenant_id)
            ).all()
        )
    }


def _job(db: Session, current: User, job_id: str) -> ImportJob:
    j = db.scalars(
        select(ImportJob).where(
            ImportJob.id == job_id, ImportJob.tenant_id == current.tenant_id
        )
    ).first()
    if j is None:
        raise HTTPException(404, "导入任务不存在")
    return j


def _job_json(j: ImportJob, spec: DatasetSpec | None = None) -> dict:
    out = {
        "id": j.id, "dataset": j.dataset, "file_name": j.file_name,
        "columns": j.columns, "row_count": j.row_count,
        "mapping": j.mapping, "mapping_meta": j.mapping_meta,
        "status": j.status, "check_result": j.check_result,
        "run_summary": j.run_summary,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "finished_at": j.finished_at.isoformat() if j.finished_at else None,
    }
    if spec is not None:
        out["dataset_label"] = spec.label
    return out


# ---------------------------------------------------------------- datasets

@router.get("/datasets")
def list_datasets(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    opts = _suggest_options(db, current.tenant_id)
    from app.infra.license.state import is_feature_enabled
    return {
        "datasets": [
            {**spec.spec_json(opts),
             "allowed": has_permission(current, spec.permission)}
            for spec in IMPORT_DATASETS.values()
        ],
        "ai_enabled": is_feature_enabled(db, "assistant.agent"),
        "max_rows": MAX_ROWS,
    }


@router.get("/datasets/{key}/template")
def dataset_template(
    key: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    spec = _spec(key)
    _perm(current, spec)
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f.key for f in spec.fields])
    w.writerow([f.sample for f in spec.fields])
    return Response(
        content="﻿" + buf.getvalue(),  # BOM:Excel 打开不乱码
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{key}_template.csv"'
        },
    )


# ---------------------------------------------------------------- upload

@router.post("", status_code=201)
async def create_import(
    request: Request,
    file: UploadFile,
    dataset: str = Form(...),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    spec = _spec(dataset)
    _perm(current, spec)
    raw = await file.read()
    try:
        columns, rows = parse_file(file.filename or "", raw)
    except UploadError as e:
        raise HTTPException(400, str(e)) from e
    job = ImportJob(
        tenant_id=current.tenant_id,
        dataset=spec.key,
        file_name=file.filename or "upload",
        file_bytes=raw,
        columns=columns,
        row_count=len(rows),
        created_by=current.id,
        status="uploaded",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    out = _job_json(job, spec)
    out["preview"] = rows[:3]
    out["suggested_mapping"] = heuristic_suggest(columns, spec.fields)
    return out


@router.get("")
def list_imports(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    limit: int = 50,
):
    jobs = db.scalars(
        select(ImportJob)
        .where(ImportJob.tenant_id == current.tenant_id)
        .order_by(ImportJob.created_at.desc())
        .limit(min(limit, 200))
    ).all()
    return [
        _job_json(j, IMPORT_DATASETS.get(j.dataset)) for j in jobs
    ]


@router.get("/{job_id}")
def get_import(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    out = _job_json(j, spec)
    columns, rows = parse_file(j.file_name, j.file_bytes)
    out["preview"] = rows[:5]
    return out


# ---------------------------------------------------------------- mapping

@router.post("/{job_id}/suggest")
def suggest(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    _, rows = parse_file(j.file_name, j.file_bytes)
    mapping, meta, warnings = ai_suggest(
        db, current.tenant_id, spec, j.columns, rows
    )
    j.mapping = mapping
    j.mapping_meta = meta
    j.status = "mapped"
    write_audit(
        db, actor=current, entity_type="import_job", entity_id=j.id,
        action="ai_suggest",
        detail={"dataset": spec.key,
                "ai_fields": sum(1 for m in meta.values() if m.get("source") == "ai")},
        request=None,
    )
    db.commit()
    return {"mapping": mapping, "meta": meta, "warnings": warnings}


class MappingIn(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict)


@router.put("/{job_id}/mapping")
def put_mapping(
    job_id: str,
    body: MappingIn,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    valid_keys = {f.key for f in spec.fields}
    mapping = {
        k: (v.strip() if v else "")
        for k, v in body.mapping.items()
        if k in valid_keys and (not v.strip() or v.strip() in j.columns)
    }
    taken = set()
    for k, v in mapping.items():
        if v and v in taken:
            mapping[k] = ""
        elif v:
            taken.add(v)
    old = dict(j.mapping or {})
    meta = dict(j.mapping_meta or {})
    for k in mapping:
        if mapping[k] != old.get(k, ""):
            meta[k] = {"confidence": 1.0, "source": "user", "reason": ""}
        else:
            meta.setdefault(
                k, {"confidence": 1.0, "source": "user", "reason": ""}
            )
    j.mapping = mapping
    j.mapping_meta = meta
    j.status = "mapped"
    db.commit()
    return {"mapping": mapping, "meta": meta}


# ---------------------------------------------------------------- check/run

@router.post("/{job_id}/check")
def check(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    if not j.mapping:
        raise HTTPException(400, "请先完成列映射(suggest 或 PUT mapping)")
    missing = [
        f.label for f in spec.fields
        if f.required and not j.mapping.get(f.key)
    ]
    if missing:
        raise HTTPException(400, f"必填字段未映射: {', '.join(missing)}")

    _, rows = parse_file(j.file_name, j.file_bytes)
    mapped = apply_mapping(rows, j.mapping, spec.fields)
    ctx = _ctx(db, current.tenant_id)
    for row in mapped:
        _normalize_choice(row, spec)
    result = check_rows(db, spec, mapped, ctx)
    note = ai_check_note(db, current.tenant_id, spec, result)

    j.check_result = {
        "valid": result["valid"],
        "error_count": result["error_count"],
        "errors": result["errors"][:500],
        "will_create": result["will_create"],
        "ai_note": note,
    }
    j.status = "checked"
    db.commit()
    return j.check_result


@router.post("/{job_id}/run")
@idempotent(scope="imports.run", resource_type="import_job")
def run(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    if not j.mapping:
        raise HTTPException(400, "请先完成列映射")
    summary = run_import(db, j, spec)
    j.run_summary = summary
    j.status = "done"
    j.finished_at = datetime.utcnow()
    write_audit(
        db, actor=current, entity_type=spec.entity_type, entity_id=j.id,
        action="import",
        detail={
            "dataset": spec.key, "file": j.file_name,
            "created": summary["created"], "errors": summary["errors"],
        },
        request=request,
    )
    db.commit()
    return summary


@router.get("/{job_id}/errors.csv")
def errors_csv(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    text = error_csv(j, spec)
    if text is None:
        raise HTTPException(404, "无错误记录(未校验或全部通过)")
    return Response(
        content="﻿" + text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="import_errors_{j.id[:8]}.csv"'
        },
    )


@router.delete("/{job_id}", status_code=204)
def delete_import(
    job_id: str,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    j = _job(db, current, job_id)
    spec = _spec(j.dataset)
    _perm(current, spec)
    db.delete(j)
    db.commit()
