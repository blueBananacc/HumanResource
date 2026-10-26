"""docx 文档引擎:人才分析报告 / 笔试卷。

人才分析报告对齐 Moka/大易「全景评估报告」结构:
候选人档案 → 逐岗位匹配评估(总分/锚定级/证据引用表/亮点短板/红旗/门槛)
→ 面试记录 → 综合结论。
"""
from __future__ import annotations

from io import BytesIO

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Interview,
    Job,
    QuestionLibraryItem,
    QuestionSet,
    Resume,
    ResumeJobMatch,
)

_ACCENT = RGBColor(0x83, 0x14, 0x3D)
_GRAY = RGBColor(0x6B, 0x6B, 0x66)
_LEVEL_ZH = {0: "未提及/不满足", 1: "弱相关", 2: "有主张未证实", 3: "接近匹配", 4: "完全满足"}


def _base(title: str, subtitle: str) -> Document:
    doc = Document()
    st = doc.styles["Normal"].font
    st.name = "Calibri"
    st.size = Pt(10.5)
    h = doc.add_heading(title, level=0)
    for r in h.runs:
        r.font.color.rgb = _ACCENT
    p = doc.add_paragraph(subtitle)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in p.runs:
        r.font.size = Pt(9)
        r.font.color.rgb = _GRAY
    return doc


def _dump(doc: Document) -> bytes:
    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()


def candidate_report(db: Session, tenant_id: str, candidate_id: str) -> bytes:
    c = db.get(Candidate, candidate_id)
    if not c or c.tenant_id != tenant_id:
        raise ValueError("候选人不存在")
    resumes = db.scalars(
        select(Resume).where(Resume.candidate_id == c.id).order_by(Resume.created_at.desc())
    ).all()
    resume_ids = [r.id for r in resumes]
    skills = next(
        ((r.parsed_data or {}).get("skills") for r in resumes if (r.parsed_data or {}).get("skills")),
        [],
    )
    matches = db.execute(
        select(ResumeJobMatch, Job)
        .join(Job, Job.id == ResumeJobMatch.job_id)
        .where(
            ResumeJobMatch.tenant_id == tenant_id,
            ResumeJobMatch.resume_id.in_(resume_ids or ["-"]),
        )
        .order_by(ResumeJobMatch.score.desc().nullslast())
    ).all()
    interviews = db.scalars(
        select(Interview).where(
            Interview.tenant_id == tenant_id, Interview.candidate_id == c.id
        )
    ).all()

    doc = _base(f"人才分析报告:{c.name}", f"HROnfire AI 评估 · {c.created_at:%Y-%m-%d}")

    # 1. 档案
    doc.add_heading("一、候选人档案", level=1)
    t = doc.add_table(rows=0, cols=4)
    t.style = "Light Grid Accent 1"
    meta = [
        ("姓名", c.name, "状态", c.status),
        ("邮箱", c.display_email or "-", "电话", c.display_phone or "-"),
        ("标签", ", ".join(c.tags or []) or "-", "简历份数", str(len(resumes))),
        ("技能", ", ".join(skills[:20]) or "-", "", ""),
    ]
    for a, b, cc, d in meta:
        row = t.add_row().cells
        row[0].text, row[1].text, row[2].text, row[3].text = a, b, cc, d
        row[0].paragraphs[0].runs[0].font.bold = True
        row[2].paragraphs[0].runs[0].font.bold = True

    # 2. 逐岗位匹配评估
    doc.add_heading("二、岗位匹配评估", level=1)
    if not matches:
        doc.add_paragraph("尚无匹配评估记录。")
    for m, job in matches:
        h = doc.add_heading(f"{job.title} — ", level=2)
        if m.status == "gated":
            h.add_run("门槛淘汰").font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
        elif m.score is not None:
            h.add_run(f"{m.score} 分").font.color.rgb = _ACCENT
        else:
            h.add_run(m.status).font.color.rgb = _GRAY

        if m.criteria:
            t = doc.add_table(rows=1, cols=4)
            t.style = "Light Grid Accent 1"
            for cell, txt in zip(t.rows[0].cells, ["评估维度", "权重", "锚定级", "简历原文证据"]):
                cell.text = txt
                cell.paragraphs[0].runs[0].font.bold = True
            for cr in m.criteria:
                row = t.add_row().cells
                row[0].text = str(cr.get("name") or cr.get("key", ""))
                row[1].text = f"{cr.get('weight', '-')}"
                lvl = cr.get("level")
                row[2].text = f"{lvl} · {_LEVEL_ZH.get(lvl, '')}" if lvl is not None else "-"
                row[3].text = str(cr.get("evidence") or "-")[:240]

        for label, items in (("匹配亮点", m.strengths), ("关键短板", m.gaps)):
            if items:
                p = doc.add_paragraph()
                p.add_run(f"{label}:").bold = True
                for it in items[:5]:
                    doc.add_paragraph(str(it), style="List Bullet")
        flags = m.red_flags or {}
        if flags.get("major") or flags.get("minor"):
            p = doc.add_paragraph()
            p.add_run("红旗提示:").bold = True
            p.add_run(
                " 重度:" + ",".join(flags.get("major") or [])
                + (";轻度:" + ",".join(flags.get("minor") or "") if flags.get("minor") else "")
            )
        gate = m.gate or {}
        if gate.get("disqualifiers"):
            p = doc.add_paragraph()
            p.add_run("门槛未过项:").bold = True
            for dq in gate["disqualifiers"][:4]:
                doc.add_paragraph(
                    f"{dq.get('question', '')} — {dq.get('evidence', '')}",
                    style="List Bullet",
                )
        if m.comment:
            p = doc.add_paragraph()
            p.add_run("AI 总评:").bold = True
            p.add_run(m.comment)

    # 3. 面试记录
    if interviews:
        doc.add_heading("三、面试记录", level=1)
        for iv in interviews:
            p = doc.add_paragraph()
            p.add_run(f"{iv.started_at:%Y-%m-%d} · {iv.status}").bold = True
            s = iv.summary or {}
            if s.get("recommendation"):
                doc.add_paragraph(f"结论:{s['recommendation']}")
            dims = s.get("dimension_scores") or {}
            if dims:
                doc.add_paragraph(
                    "维度分:" + " / ".join(f"{k}={v}" for k, v in dims.items())
                )

    # 4. 免责
    p = doc.add_paragraph()
    r = p.add_run("本报告由 AI 生成,评分均有简历原文证据支撑;最终录用决策由招聘负责人作出。")
    r.font.size = Pt(8)
    r.font.color.rgb = _GRAY
    return _dump(doc)


_KIND_ZH = {
    "tech": "技术深度",
    "project": "项目复盘",
    "scenario": "场景排查",
    "system": "系统设计",
    "soft": "软技能",
    "技术深度": "技术深度",
    "项目复盘": "项目复盘",
    "场景排查": "场景排查",
    "系统设计": "系统设计",
    "软技能": "软技能",
}


def exam_paper(
    db: Session, tenant_id: str, question_set_id: str, with_answers: bool = True
) -> bytes:
    qs = db.get(QuestionSet, question_set_id)
    if not qs or qs.tenant_id != tenant_id:
        raise ValueError("题集不存在")
    items = db.scalars(
        select(QuestionLibraryItem)
        .where(QuestionLibraryItem.question_set_id == question_set_id)
        .order_by(QuestionLibraryItem.position)
    ).all()
    if not items:
        raise ValueError("题集还没有生成题目")
    job = db.get(Job, qs.job_id) if qs.job_id else None

    doc = _base(
        f"笔试卷:{job.title if job else '通用'}",
        f"共 {len(items)} 题 · 难度 {qs.level} · {'含参考答案' if with_answers else '考生版'}",
    )
    doc.add_paragraph("考试说明:本试卷由 AI 结合岗位要求与候选人简历生成,请独立作答。")

    doc.add_heading("题目", level=1)
    kind_counts: dict[str, int] = {}
    for i, it in enumerate(items, 1):
        kind = _KIND_ZH.get(it.kind, it.kind)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        p = doc.add_paragraph()
        p.add_run(f"{i}. ").bold = True
        p.add_run(f"[{kind} · {it.difficulty}] ").font.color.rgb = _GRAY
        p.add_run(it.question)
        if it.follow_up:
            fp = doc.add_paragraph()
            fp.add_run("   追问:").bold = True
            fp.add_run(it.follow_up)
        doc.add_paragraph("   答:" + "_" * 60)

    if with_answers:
        doc.add_page_break()
        doc.add_heading("参考答案要点", level=1)
        for i, it in enumerate(items, 1):
            p = doc.add_paragraph()
            p.add_run(f"{i}. ").bold = True
            for pt in it.answer_points or []:
                doc.add_paragraph(str(pt), style="List Bullet")

    stat = doc.add_paragraph()
    stat.add_run(
        "题型分布:" + " / ".join(f"{k}×{v}" for k, v in kind_counts.items())
    ).font.size = Pt(9)
    return _dump(doc)
