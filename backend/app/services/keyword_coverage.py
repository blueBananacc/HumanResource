"""关键词覆盖子分(T15)—— JD 技能在简历中的整词覆盖率,零 LLM。

移植 Resume-Matcher ats.py 的确定性分解思路:覆盖分数作为 LLM
criteria 评分的旁证,`missing` 列表直接告诉 HR 缺哪些关键词。
"""
from __future__ import annotations

from typing import Any

from app.domain.models import Resume
from app.services.text_match import keyword_in_text, normalize_skill_key


def _resume_corpus(resume: Resume) -> str:
    """覆盖语料:解析全文 + 抽取技能 + 结构化技能区,一次拼好。"""
    pd = resume.parsed_data or {}
    parts = [resume.parsed_text or ""]
    skills = pd.get("skills") or []
    if skills:
        parts.append(" ".join(str(s) for s in skills))
    stext = ((pd.get("sections") or {}).get("skills_text") or "")
    if stext:
        parts.append(stext)
    return " ".join(parts)


def compute_coverage(
    resume: Resume,
    keywords: list[str] | None,
) -> dict[str, Any] | None:
    """返回 {score 0-100, hit[], missing[]};无有效关键词返回 None。

    关键词去重(casefold)、保留 JD 原序;命中判定用 CJK 整词匹配。
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for k in keywords or []:
        s = str(k).strip()
        key = normalize_skill_key(s)
        if s and key not in seen:
            seen.add(key)
            ordered.append(s)
    if not ordered:
        return None

    corpus = _resume_corpus(resume)
    hit = [k for k in ordered if keyword_in_text(k, corpus)]
    missing = [k for k in ordered if not keyword_in_text(k, corpus)]
    return {
        "score": round(len(hit) / len(ordered) * 100),
        "hit": hit[:10],
        "missing": missing[:10],
        "total": len(ordered),
    }
