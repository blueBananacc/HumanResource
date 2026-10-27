"""P2 公司级知识库注入:把命中的制度/要求/文化片段拼成 LLM 上下文。

与 ``job_knowledge.knowledge_context_text``(岗位级 chunk)互补 ——
这里是跨岗位的 ``knowledge_articles``(制度/FAQ/文化/题库/代码规范),
供 JD 优化、出题等「需要公司级依据」的场景注入。
命中同时返回溯源信息(文章标题),由调用方自行决定是否透传。
"""
from __future__ import annotations

import re

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.domain.models import KnowledgeArticle

_MAX_ARTICLES = 3
_SNIPPET_CHARS = 400


def _terms(q: str) -> list[str]:
    return [t for t in re.split(r"[\s,;/、,。()()]+", q or "") if len(t) >= 2][:6]


def search_company_kb(
    db: Session, tenant_id: str, query: str, *, limit: int = _MAX_ARTICLES
) -> list[tuple[KnowledgeArticle, float]]:
    """词面召回已发布知识文章:title 命中权重高于 content,按命中词数排序。"""
    terms = _terms(query)
    if not terms:
        return []
    conds = []
    score_parts = []
    for t in terms:
        pat = f"%{t}%"
        conds.append(KnowledgeArticle.title.ilike(pat))
        conds.append(KnowledgeArticle.content.ilike(pat))
        score_parts.append(
            case((KnowledgeArticle.title.ilike(pat), 2.0), else_=0.0)
            + case((KnowledgeArticle.content.ilike(pat), 1.0), else_=0.0)
        )
    lex = sum(score_parts[1:], score_parts[0]).label("lex")
    stmt = (
        select(KnowledgeArticle, lex)
        .where(
            KnowledgeArticle.tenant_id == tenant_id,
            KnowledgeArticle.status == "published",
        )
        .where(or_(*conds))
        .order_by(lex.desc())
        .limit(limit)
    )
    return [(a, float(s or 0)) for a, s in db.execute(stmt).all()]


def _relevant_excerpt(content: str, terms: list[str]) -> str:
    """取第一个命中词附近的片段作为「证据」。"""
    low = content.lower()
    idx = -1
    for t in terms:
        idx = low.find(t.lower())
        if idx >= 0:
            break
    if idx < 0:
        return content[:_SNIPPET_CHARS].strip()
    start = max(0, idx - 80)
    return content[start : idx + _SNIPPET_CHARS].strip()


def company_kb_context(
    db: Session,
    tenant_id: str,
    query: str,
    *,
    limit: int = _MAX_ARTICLES,
    max_chars: int = 1500,
) -> str:
    """拼 LLM 上下文文本;无命中返回 ""(调用方跳过注入即可)。"""
    rows = search_company_kb(db, tenant_id, query, limit=limit)
    if not rows:
        return ""
    terms = _terms(query)
    parts = ["【公司知识库 · 制度/要求(作为依据,引用须可回溯到文章标题)】"]
    for a, _s in rows:
        parts.append(f"《{a.title}》{_relevant_excerpt(a.content, terms)}")
    return "\n\n".join(parts)[:max_chars]
