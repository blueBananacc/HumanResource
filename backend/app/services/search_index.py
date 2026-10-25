"""声明式全局搜索(T9)—— 索引维护 + 查询。

三条架构性质(参考 Gauzy plugin-search):

1. **声明是数据不是代码**:``SEARCH_SOURCES`` 是代码注册表默认声明;
   ``search_index_definitions`` 存租户覆盖(is_active/权重/模板/keywords),
   与 PromptTemplate 同款两级回落。改权重/关实体不需要发版。
2. **索引是可丢弃投影**:``search_documents.entity_id`` 不带外键;
   命中只回 entity+id,读路径经 entities.py ``build()`` 重读真实行,
   权限与行级 scope 永远在域模型侧生效,索引不会泄漏越权数据。
3. **事件驱动**:`write_audit` 漏斗在 automation.dispatch 后调
   ``on_audit`` —— 44 处既有写端点零改动获得索引覆盖;
   ``reindex`` 端点是孤儿清扫与全量重建的兜底。
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import String, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Announcement,
    AuditLog,
    Candidate,
    Employee,
    EmployeeRequest,
    ExpenseClaim,
    HrCase,
    Interview,
    Job,
    KnowledgeArticle,
    SearchDocument,
    SearchIndexDefinition,
)

logger = logging.getLogger(__name__)

TPL_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
DELETE_ACTIONS = {"delete", "bulk_delete"}
MAX_BODY = 4000


def _txt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_txt(x) for x in v if x is not None)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def _render(tpl: str, ctx: dict[str, Any]) -> str:
    return TPL_RE.sub(lambda m: _txt(ctx.get(m.group(1))), tpl or "").strip()


# ---------------------------------------------------------------------------
# 代码注册表 — 默认声明(租户可经 search_index_definitions 覆盖)
# fields: {属性: 权重};context: 模板补充上下文(join 名等);body_fields 顺序即正文序
# ---------------------------------------------------------------------------


def _interview_ctx(db: Session, o: Interview) -> dict[str, Any]:
    cand = db.get(Candidate, o.candidate_id)
    job = db.get(Job, o.job_id)
    return {
        "candidate_name": cand.name if cand else "",
        "job_title": job.title if job else "",
        "status": o.status,
        "modality": o.modality,
    }


def _emp_name(db: Session, emp_id: str | None) -> str:
    e = db.get(Employee, emp_id) if emp_id else None
    return e.name if e else ""


SEARCH_SOURCES: dict[str, dict[str, Any]] = {
    "talents": {
        "label": "候选人",
        "audit_types": {"candidate"},
        "model": Candidate,
        "title": "{{name}}",
        "body": "{{display_email}} {{display_phone}} {{tags}}",
        "fields": {"name": 3, "display_email": 2, "display_phone": 2, "status": 1},
        "keyword_fields": ["status"],
        "link": "/talents/{id}",
    },
    "jobs": {
        "label": "岗位",
        "audit_types": {"job"},
        "model": Job,
        "title": "{{title}}",
        "body": "{{level}} {{status}} {{description}}",
        "fields": {"title": 3, "level": 1, "status": 1, "description": 1},
        "keyword_fields": ["status", "level"],
        "link": "/jobs/{id}",
    },
    "employees": {
        "label": "员工",
        "audit_types": {"employee"},
        "model": Employee,
        "title": "{{name}}",
        "body": "{{employee_no}} {{designation}} {{work_email}}",
        "fields": {
            "name": 3, "employee_no": 2, "designation": 2, "work_email": 2,
            "status": 1,
        },
        "keyword_fields": ["status", "employment_type"],
        "link": "/employees/{id}",
    },
    "interviews": {
        "label": "面试",
        "audit_types": {"interview"},
        "model": Interview,
        "title": "{{candidate_name}} · {{job_title}}",
        "body": "{{status}} {{modality}}",
        "fields": {"candidate_name": 3, "job_title": 3, "status": 1},
        "keyword_fields": ["status"],
        "context": _interview_ctx,
        "link": "/interviews/{id}/report",
    },
    "cases": {
        "label": "工单",
        "audit_types": {"hr_case"},
        "model": HrCase,
        "title": "{{subject}}",
        "body": "{{case_no}} {{kind}} {{status}} {{priority}}",
        "fields": {"subject": 3, "case_no": 2, "kind": 1, "status": 1},
        "keyword_fields": ["status", "kind"],
        "link": "/cases",
    },
    "requests": {
        "label": "员工申请",
        "audit_types": {"employee_request"},
        "model": EmployeeRequest,
        "title": "{{subject}}",
        "body": "{{kind}} {{status}} {{requester}}",
        "fields": {"subject": 3, "kind": 1, "status": 1},
        "keyword_fields": ["status", "kind"],
        "context": lambda db, o: {"requester": _emp_name(db, o.employee_id)},
        "link": "/requests",
    },
    "knowledge": {
        "label": "知识文章",
        "audit_types": {"knowledge_article"},
        "model": KnowledgeArticle,
        "title": "{{title}}",
        "body": "{{category}} {{content}}",
        "fields": {"title": 3, "category": 1, "content": 1, "tags": 1},
        "keyword_fields": ["category"],
        "indexable": lambda o: o.status == "published",
        "link": "/knowledge",
    },
    "announcements": {
        "label": "公告",
        "audit_types": {"announcement"},
        "model": Announcement,
        "title": "{{title}}",
        "body": "{{status}} {{content}}",
        "fields": {"title": 3, "status": 1, "content": 1},
        "keyword_fields": ["status"],
        "indexable": lambda o: o.status == "published",
        "link": "/announcements",
    },
    "expenses": {
        "label": "报销单",
        "audit_types": {"expense_claim"},
        "model": ExpenseClaim,
        "title": "{{title}}",
        "body": "{{status}} {{total}} {{requester}}",
        "fields": {"title": 3, "status": 1},
        "keyword_fields": ["status"],
        "context": lambda db, o: {"requester": _emp_name(db, o.employee_id)},
        "link": "/expenses",
    },
}

# 审计 entity_type → 搜索实体 key
AUDIT_TO_ENTITY: dict[str, str] = {
    at: key for key, spec in SEARCH_SOURCES.items() for at in spec["audit_types"]
}


# ---------------------------------------------------------------------------
# 定义解析:租户覆盖 > 注册表
# ---------------------------------------------------------------------------


def effective_definition(
    db: Session, tenant_id: str, entity: str
) -> dict[str, Any] | None:
    """合并注册表与租户覆盖;实体未知或租户停用 → None。"""
    spec = SEARCH_SOURCES.get(entity)
    if spec is None:
        return None
    row = db.scalars(
        select(SearchIndexDefinition).where(
            SearchIndexDefinition.tenant_id == tenant_id,
            SearchIndexDefinition.entity == entity,
        )
    ).first()
    out = {
        "entity": entity,
        "label": spec["label"],
        "is_active": True,
        "fields": dict(spec["fields"]),
        "title": spec["title"],
        "body": spec["body"],
        "keyword_fields": list(spec.get("keyword_fields") or []),
        "version": 0,
        "overridden": False,
    }
    if row is not None:
        out["overridden"] = True
        out["version"] = row.version
        out["is_active"] = bool(row.is_active)
        if row.label:
            out["label"] = row.label
        if row.fields:
            # 覆盖只允许调注册表已声明字段的权重(防注入任意路径)
            declared = set(spec["fields"])
            for f in row.fields:
                name, w = (f or {}).get("name"), (f or {}).get("weight")
                if name in declared:
                    try:
                        out["fields"][name] = max(0, min(10, int(w)))
                    except (TypeError, ValueError):
                        continue
        if row.title_template:
            out["title"] = row.title_template
        if row.body_template:
            out["body"] = row.body_template
        if row.keyword_fields is not None:
            out["keyword_fields"] = [
                k for k in row.keyword_fields if k in set(spec["fields"])
            ]
    if not out["is_active"]:
        return None
    return out


def _doc_context(db: Session, spec: dict, obj: Any, defn: dict) -> dict[str, Any]:
    ctx = {f: getattr(obj, f, None) for f in defn["fields"]}
    extra = spec.get("context")
    if extra:
        ctx.update(extra(db, obj))
    return ctx


def _source_ts(obj: Any) -> datetime | None:
    return getattr(obj, "updated_at", None) or getattr(obj, "created_at", None)


def upsert_document(
    db: Session, tenant_id: str, entity: str, obj: Any
) -> SearchDocument | None:
    """按生效定义渲染并 upsert 文档;对象不可索引 → 删已有文档。"""
    spec = SEARCH_SOURCES[entity]
    defn = effective_definition(db, tenant_id, entity)
    doc = db.scalars(
        select(SearchDocument).where(
            SearchDocument.tenant_id == tenant_id,
            SearchDocument.entity == entity,
            SearchDocument.entity_id == obj.id,
        )
    ).first()
    if defn is None or not spec.get("indexable", lambda _o: True)(obj):
        if doc is not None:
            db.delete(doc)
        return None
    ctx = _doc_context(db, spec, obj, defn)
    title = _render(defn["title"], ctx)[:512] or _txt(getattr(obj, "id", ""))
    body = _render(defn["body"], ctx)[:MAX_BODY] or None
    keywords = [
        f"{k}:{_txt(ctx.get(k))}"
        for k in defn["keyword_fields"]
        if _txt(ctx.get(k))
    ]
    if doc is None:
        doc = SearchDocument(
            tenant_id=tenant_id, entity=entity, entity_id=obj.id,
            title=title, body=body, keywords=keywords,
            source_updated_at=_source_ts(obj),
            definition_version=defn["version"],
        )
        db.add(doc)
    else:
        doc.title = title
        doc.body = body
        doc.keywords = keywords
        doc.source_updated_at = _source_ts(obj)
        doc.definition_version = defn["version"]
        doc.indexed_at = datetime.now(UTC).replace(tzinfo=None)
    return doc


def remove_document(db: Session, tenant_id: str, entity: str, entity_id: str) -> None:
    doc = db.scalars(
        select(SearchDocument).where(
            SearchDocument.tenant_id == tenant_id,
            SearchDocument.entity == entity,
            SearchDocument.entity_id == entity_id,
        )
    ).first()
    if doc is not None:
        db.delete(doc)


def on_audit(db: Session, log: AuditLog) -> bool:
    """write_audit 漏斗上的索引钩子;任何异常吞掉不炸业务事务。"""
    try:
        entity = AUDIT_TO_ENTITY.get(log.entity_type)
        if entity is None or log.result != "success":
            return False
        spec = SEARCH_SOURCES[entity]
        if log.action in DELETE_ACTIONS:
            remove_document(db, log.tenant_id, entity, log.entity_id)
            return True
        obj = db.get(spec["model"], log.entity_id)
        if obj is None:
            # 源行已不可见(同事务硬删等)→ 兜底删
            remove_document(db, log.tenant_id, entity, log.entity_id)
            return False
        return upsert_document(db, log.tenant_id, entity, obj) is not None
    except Exception:  # noqa: BLE001
        logger.exception(
            "search index on_audit 失败 entity=%s id=%s", log.entity_type, log.entity_id
        )
        return False


# ---------------------------------------------------------------------------
# 查询 / 重建
# ---------------------------------------------------------------------------


def _escape_like(q: str) -> str:
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def doc_hits(
    db: Session, tenant_id: str, q: str, entity: str | None = None,
    limit: int = 200,
) -> list[tuple[SearchDocument, int]]:
    """按权重打分命中文档。title=3 / keywords=2 / body=1。

    返回 [(doc, score)];**调用方负责权限与行级可见性过滤**。
    """
    pat = f"%{_escape_like(q.strip().lower())}%"
    stmt = select(SearchDocument).where(SearchDocument.tenant_id == tenant_id)
    if entity:
        stmt = stmt.where(SearchDocument.entity == entity)
    kw_col = SearchDocument.keywords.cast(String)
    stmt = stmt.where(
        SearchDocument.title.ilike(pat, escape="\\")
        | SearchDocument.body.ilike(pat, escape="\\")
        | kw_col.ilike(pat, escape="\\")
    )
    docs = db.scalars(stmt.limit(limit)).all()
    out = []
    for d in docs:
        score = 0
        if d.title and q.lower() in d.title.lower():
            score += 3
        if d.keywords and q.lower() in str(d.keywords).lower():
            score += 2
        if d.body and q.lower() in d.body.lower():
            score += 1
        if score:
            out.append((d, score))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def reindex(
    db: Session, tenant_id: str, entity: str | None = None
) -> dict[str, Any]:
    """全量/单实体重建 + 孤儿清扫。返回 {entity: {indexed, removed}}。"""
    entities = [entity] if entity else list(SEARCH_SOURCES)
    result: dict[str, Any] = {}
    for key in entities:
        spec = SEARCH_SOURCES.get(key)
        if spec is None:
            continue
        defn = effective_definition(db, tenant_id, key)
        source_ids = set(
            db.scalars(
                select(spec["model"].id).where(spec["model"].tenant_id == tenant_id)
            ).all()
        )
        existing = {
            d.entity_id: d
            for d in db.scalars(
                select(SearchDocument).where(
                    SearchDocument.tenant_id == tenant_id,
                    SearchDocument.entity == key,
                )
            ).all()
        }
        indexed = 0
        if defn is not None:
            for rid in source_ids:
                obj = db.get(spec["model"], rid)
                if obj is None:
                    continue
                doc = existing.get(rid)
                if (
                    doc is not None
                    and doc.source_updated_at == _source_ts(obj)
                    and doc.definition_version == defn["version"]
                ):
                    continue  # 未变跳过
                if upsert_document(db, tenant_id, key, obj) is not None:
                    indexed += 1
        # 孤儿清扫:源行已删/不可索引的文档
        removed = 0
        for rid, doc in existing.items():
            if rid not in source_ids or defn is None:
                db.delete(doc)
                removed += 1
        result[key] = {"indexed": indexed, "removed": removed}
    return result


def index_status(db: Session, tenant_id: str) -> list[dict[str, Any]]:
    """每实体文档数 + 最近索引时间(管理页用)。"""
    out = []
    for key, spec in SEARCH_SOURCES.items():
        docs = db.scalars(
            select(SearchDocument).where(
                SearchDocument.tenant_id == tenant_id,
                SearchDocument.entity == key,
            )
        ).all()
        out.append(
            {
                "entity": key,
                "label": spec["label"],
                "documents": len(docs),
                "last_indexed_at": max(
                    (d.indexed_at for d in docs), default=None
                ),
            }
        )
    return out


__all__ = [
    "AUDIT_TO_ENTITY",
    "DELETE_ACTIONS",
    "SEARCH_SOURCES",
    "doc_hits",
    "effective_definition",
    "index_status",
    "on_audit",
    "reindex",
    "remove_document",
    "upsert_document",
]
