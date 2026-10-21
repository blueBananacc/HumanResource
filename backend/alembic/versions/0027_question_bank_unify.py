"""question bank unify: 题目成为唯一实体,题集只是集合

Revision ID: 0027
Revises: 0026
Create Date: 2026-10-21

1. ``question_library_items`` 加 ``question_set_id``(FK → question_sets,CASCADE)/
   ``position`` / ``dimensions``
2. 把各题集 ``questions`` JSON 里的题迁成题库条目并挂到题集下;
   若题库里已有"导出到题库"产生的同题干独立副本,直接认领该条(保留引用次数),不重复插入
3. 迁移后清空 ``question_sets.questions``(降级时从题目重建)
4. 简历题集里题目的分类标签统一为"岗位名 · 候选人姓名"(独立题目按分类标签自动聚合成分类题集)
"""
from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa

from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None

_DIFF = {"初级": "initial", "中级": "intermediate", "高级": "advanced", "专家": "expert"}
_DIFF_LABEL = {v: k for k, v in _DIFF.items()}
_KIND_LABEL = {
    "tech": "技术深度",
    "project": "项目复盘",
    "scenario": "场景排查",
    "design": "系统设计",
    "soft": "软技能",
}


def _kind(dims: list[str]) -> str:
    for d in dims:
        low = d.lower()
        if "设计" in d or "design" in low:
            return "design"
        if "项目" in d or "project" in low:
            return "project"
        if "场景" in d or "排查" in d or "scenario" in low:
            return "scenario"
        if "软" in d or "沟通" in d or "soft" in low:
            return "soft"
        if "技术" in d or "tech" in low:
            return "tech"
    return "tech"


def _diff(raw: str | None, default: str) -> str:
    if raw in _DIFF_LABEL:
        return raw
    return _DIFF.get(raw or "", default if default in _DIFF_LABEL else "intermediate")


question_sets = sa.table(
    "question_sets",
    sa.column("id", sa.String),
    sa.column("tenant_id", sa.String),
    sa.column("resume_id", sa.String),
    sa.column("job_id", sa.String),
    sa.column("level", sa.String),
    sa.column("questions", sa.JSON),
    sa.column("created_by", sa.String),
)
jobs = sa.table("jobs", sa.column("id", sa.String), sa.column("title", sa.String))
resumes = sa.table("resumes", sa.column("id", sa.String), sa.column("candidate_id", sa.String))
candidates = sa.table("candidates", sa.column("id", sa.String), sa.column("name", sa.String))
items = sa.table(
    "question_library_items",
    sa.column("id", sa.String),
    sa.column("tenant_id", sa.String),
    sa.column("question", sa.Text),
    sa.column("answer_points", sa.JSON),
    sa.column("kind", sa.String),
    sa.column("difficulty", sa.String),
    sa.column("category", sa.String),
    sa.column("follow_up", sa.Text),
    sa.column("source", sa.String),
    sa.column("use_count", sa.Integer),
    sa.column("generated_from_job_id", sa.String),
    sa.column("question_set_id", sa.String),
    sa.column("position", sa.Integer),
    sa.column("dimensions", sa.JSON),
    sa.column("created_by", sa.String),
    sa.column("created_at", sa.DateTime),
    sa.column("updated_at", sa.DateTime),
)


def upgrade() -> None:
    op.add_column(
        "question_library_items",
        sa.Column(
            "question_set_id",
            sa.String(36),
            sa.ForeignKey("question_sets.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "question_library_items",
        sa.Column("position", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column("question_library_items", sa.Column("dimensions", sa.JSON, nullable=True))
    op.create_index(
        "ix_question_library_items_question_set_id",
        "question_library_items",
        ["question_set_id"],
    )

    bind = op.get_bind()
    titles = {r.id: r.title for r in bind.execute(sa.select(jobs.c.id, jobs.c.title))}
    cand_names = {
        r.id: r.name
        for r in bind.execute(
            sa.select(resumes.c.id, candidates.c.name).join(
                candidates, candidates.c.id == resumes.c.candidate_id
            )
        )
    }
    now = datetime.utcnow()
    for qs in bind.execute(sa.select(question_sets)).fetchall():
        questions = qs.questions or []
        # 简历题集题目的分类:"岗位名 · 候选人姓名"
        category = " · ".join(
            p for p in (titles.get(qs.job_id) if qs.job_id else None, cand_names.get(qs.resume_id)) if p
        )[:128]
        for pos, q in enumerate(questions):
            if not isinstance(q, dict) or not q.get("question"):
                continue
            dims = [str(d) for d in (q.get("dimensions") or [])]
            # 认领"导出到题库"留下的独立副本
            orphan = bind.execute(
                sa.select(items.c.id)
                .where(
                    items.c.tenant_id == qs.tenant_id,
                    items.c.question == q["question"],
                    items.c.question_set_id.is_(None),
                    items.c.source == "ai_generated",
                )
                .limit(1)
            ).scalar()
            if orphan:
                bind.execute(
                    items.update()
                    .where(items.c.id == orphan)
                    .values(question_set_id=qs.id, position=pos, dimensions=dims, category=category)
                )
                continue
            bind.execute(
                items.insert().values(
                    id=str(uuid.uuid4()),
                    tenant_id=qs.tenant_id,
                    question=q["question"],
                    answer_points=list(q.get("answer_points") or []),
                    kind=_kind(dims),
                    difficulty=_diff(q.get("difficulty"), qs.level),
                    category=category,
                    follow_up=q.get("follow_up") or None,
                    source="ai_generated",
                    use_count=0,
                    generated_from_job_id=qs.job_id,
                    question_set_id=qs.id,
                    position=pos,
                    dimensions=dims,
                    created_by=qs.created_by,
                    created_at=now,
                    updated_at=now,
                )
            )
        if questions:
            bind.execute(
                # sa.null():JSON 列写 None 会存成 JSON 'null' 而非 SQL NULL
                question_sets.update().where(question_sets.c.id == qs.id).values(questions=sa.null())
            )


def downgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(items)
        .where(items.c.question_set_id.is_not(None))
        .order_by(items.c.question_set_id, items.c.position)
    ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r.question_set_id, []).append(
            {
                "question": r.question,
                "answer_points": list(r.answer_points or []),
                "dimensions": list(r.dimensions or [_KIND_LABEL.get(r.kind, r.kind)]),
                "difficulty": _DIFF_LABEL.get(r.difficulty, r.difficulty),
                "follow_up": r.follow_up or "",
            }
        )
    for set_id, qs in grouped.items():
        bind.execute(
            question_sets.update().where(question_sets.c.id == set_id).values(questions=qs)
        )
    # 降级后题集内的题仍保留为独立题目
    op.drop_index("ix_question_library_items_question_set_id", table_name="question_library_items")
    op.drop_column("question_library_items", "dimensions")
    op.drop_column("question_library_items", "position")
    op.drop_column("question_library_items", "question_set_id")
