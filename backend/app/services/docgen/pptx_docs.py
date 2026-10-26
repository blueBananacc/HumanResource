"""pptx 文档引擎:招聘进度汇报三件套(Moka BI 模板库「招聘进度」思路)。

版式(16:9,品牌色 #83143D/#EE164F/#FAA32B):
1. 封面(标题+期间+日期)
2. KPI 总览(简历/在招岗位/进行中面试/已发 Offer/入职)
3. 招聘漏斗(各阶段人数,横向条形图 - pptx 原生 shape)
4. Top 候选人(匹配分 Top5 表)
5. 下一步行动(AI 生成的 3 条建议,基于漏斗瓶颈启发式规则)
"""
from __future__ import annotations

from io import BytesIO

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Interview,
    Job,
    JobApplication,
    JobOffer,
    Resume,
    ResumeJobMatch,
)

DARK = RGBColor(0x60, 0x03, 0x2A)
ACCENT = RGBColor(0xEE, 0x16, 0x4F)
ORANGE = RGBColor(0xFA, 0xA3, 0x2B)
CREAM = RGBColor(0xFD, 0xF6, 0xEB)
TEXT = RGBColor(0x3A, 0x3A, 0x37)
GRAY = RGBColor(0x8A, 0x8A, 0x85)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

_STAGE_ZH = [
    ("applied", "已申请"),
    ("screening", "筛选中"),
    ("interview", "面试中"),
    ("offer", "Offer"),
    ("hired", "已入职"),
]


def _textbox(slide, x, y, w, h, text, size=14, bold=False, color=TEXT, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def _rect(slide, x, y, w, h, fill):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def _slide_title(slide, text):
    _rect(slide, 0.6, 0.45, 0.12, 0.55, ACCENT)
    _textbox(slide, 0.9, 0.42, 11.5, 0.7, text, size=26, bold=True, color=DARK)


def _kpi_card(slide, x, y, w, label, value, hint, color):
    card = _rect(slide, x, y, w, 1.55, CREAM)
    tf = card.text_frame
    tf.margin_left = Inches(0.18)
    tf.margin_top = Inches(0.12)
    p1 = tf.paragraphs[0]
    r = p1.add_run()
    r.text = label
    r.font.size = Pt(11)
    r.font.color.rgb = GRAY
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = str(value)
    r2.font.size = Pt(30)
    r2.font.bold = True
    r2.font.color.rgb = color
    p3 = tf.add_paragraph()
    r3 = p3.add_run()
    r3.text = hint
    r3.font.size = Pt(9)
    r3.font.color.rgb = GRAY


def recruiting_pack(db: Session, tenant_id: str) -> bytes:
    total_resumes = db.scalar(
        select(func.count(Resume.id)).where(Resume.tenant_id == tenant_id)
    ) or 0
    open_jobs = db.scalar(
        select(func.count(Job.id)).where(Job.tenant_id == tenant_id, Job.status == "open")
    ) or 0
    active_iv = db.scalar(
        select(func.count(Interview.id)).where(
            Interview.tenant_id == tenant_id, Interview.status == "in_progress"
        )
    ) or 0
    offers = db.scalar(
        select(func.count(JobOffer.id)).where(JobOffer.tenant_id == tenant_id)
    ) or 0
    hired = db.scalar(
        select(func.count(JobApplication.id)).where(
            JobApplication.tenant_id == tenant_id, JobApplication.stage == "hired"
        )
    ) or 0

    stage_counts = dict(
        db.execute(
            select(JobApplication.stage, func.count(JobApplication.id))
            .where(JobApplication.tenant_id == tenant_id)
            .group_by(JobApplication.stage)
        ).all()
    )
    funnel = [(zh, stage_counts.get(k, 0)) for k, zh in _STAGE_ZH]

    top = db.execute(
        select(ResumeJobMatch.score, Candidate.name, Job.title)
        .join(Resume, Resume.id == ResumeJobMatch.resume_id)
        .join(Candidate, Candidate.id == Resume.candidate_id)
        .join(Job, Job.id == ResumeJobMatch.job_id)
        .where(ResumeJobMatch.tenant_id == tenant_id, ResumeJobMatch.score.is_not(None))
        .order_by(ResumeJobMatch.score.desc())
        .limit(5)
    ).all()

    prs = Presentation()
    prs.slide_width = Emu(Inches(13.33))
    prs.slide_height = Emu(Inches(7.5))
    blank = prs.slide_layouts[6]

    # 1 封面
    s = prs.slides.add_slide(blank)
    bg = _rect(s, 0, 0, 13.33, 7.5, DARK)
    _rect(s, 8.9, -1.2, 5.6, 5.6, ACCENT)
    _rect(s, 10.6, 4.6, 3.6, 3.6, ORANGE)
    _textbox(s, 0.9, 2.5, 9.5, 1.2, "招聘进度汇报", size=44, bold=True, color=WHITE)
    _textbox(s, 0.95, 3.7, 9.5, 0.6, "HROnfire · AI 自动生成", size=16, color=RGBColor(0xFF, 0xD8, 0x87))
    from datetime import datetime

    _textbox(s, 0.95, 6.6, 9.5, 0.5, f"生成时间:{datetime.now():%Y-%m-%d}", size=12, color=RGBColor(0xE8, 0xE2, 0xD9))

    # 2 KPI
    s = prs.slides.add_slide(blank)
    _slide_title(s, "一、整体 KPI")
    kpis = [
        ("在库简历", total_resumes, "累计入库", ACCENT),
        ("在招岗位", open_jobs, "status=open", ORANGE),
        ("进行中面试", active_iv, "AI 面试会话", DARK),
        ("已发 Offer", offers, "含已接受", ACCENT),
        ("已入职", hired, "hired 阶段", ORANGE),
    ]
    for i, (label, v, hint, color) in enumerate(kpis):
        _kpi_card(s, 0.7 + i * 2.45, 1.8, 2.25, label, v, hint, color)
    _textbox(s, 0.7, 4.0, 12, 0.5, "数据口径:本租户全量,实时统计", size=11, color=GRAY)

    # 3 漏斗
    s = prs.slides.add_slide(blank)
    _slide_title(s, "二、招聘漏斗")
    max_v = max([c for _, c in funnel] + [1])
    y = 1.7
    for label, count in funnel:
        _textbox(s, 0.7, y + 0.08, 1.5, 0.4, label, size=13, color=TEXT)
        bar_w = max(0.3, count / max_v * 8.6)
        bar = _rect(s, 2.3, y, bar_w, 0.5, ACCENT if count else RGBColor(0xE8, 0xE2, 0xD9))
        _textbox(s, 2.35 + bar_w, y + 0.08, 1.6, 0.4, str(count), size=13, bold=True, color=DARK)
        y += 0.85
    rej = stage_counts.get("rejected", 0)
    wd = stage_counts.get("withdrawn", 0)
    _textbox(s, 0.7, y + 0.15, 12, 0.5, f"另有 已淘汰 {rej} · 已撤回 {wd}", size=11, color=GRAY)

    # 4 Top 候选人
    s = prs.slides.add_slide(blank)
    _slide_title(s, "三、Top 候选人(按 AI 匹配分)")
    if not top:
        _textbox(s, 0.7, 1.8, 12, 0.6, "暂无匹配评分数据", size=14, color=GRAY)
    else:
        rows, cols = len(top) + 1, 3
        tbl_shape = s.shapes.add_table(rows, cols, Inches(0.7), Inches(1.6), Inches(11.9), Inches(0.6 * rows))
        tbl = tbl_shape.table
        tbl.columns[0].width = Inches(4.5)
        tbl.columns[1].width = Inches(4.9)
        tbl.columns[2].width = Inches(2.5)
        for j, h in enumerate(["候选人", "匹配岗位", "AI 匹配分"]):
            cell = tbl.cell(0, j)
            cell.text = h
            cell.text_frame.paragraphs[0].runs[0].font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK
            cell.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
        for i, (score, name, job_title) in enumerate(top, 1):
            tbl.cell(i, 0).text = name or "-"
            tbl.cell(i, 1).text = job_title or "-"
            tbl.cell(i, 2).text = str(score)

    # 5 下一步行动
    s = prs.slides.add_slide(blank)
    _slide_title(s, "四、下一步行动建议")
    actions = []
    if stage_counts.get("applied", 0) > stage_counts.get("interview", 0) * 3:
        actions.append("初筛积压:「已申请」阶段候选人是面试中的 3 倍以上,建议加大筛选/AI 评估覆盖面")
    if open_jobs > 0 and stage_counts.get("interview", 0) == 0:
        actions.append("面试空档:有在招岗位但无进行中面试,建议从高分匹配列表批量发起 AI 面试")
    if stage_counts.get("offer", 0) > 0 and hired == 0:
        actions.append("Offer 转化:已发 Offer 未入职,建议跟进候选人意向并启动保温运营")
    if not actions:
        actions = [
            "漏斗健康:按周节奏复查管道,保持各阶段流转",
            "建议开启「招聘周报订阅」,自动追踪阶段转化趋势",
            "对 Top 候选人优先安排面试,缩短高分人才等待时间",
        ]
    y = 1.7
    for i, a in enumerate(actions[:4], 1):
        _rect(s, 0.7, y, 0.55, 0.55, ORANGE)
        _textbox(s, 0.82, y + 0.05, 0.5, 0.45, str(i), size=18, bold=True, color=DARK)
        _textbox(s, 1.5, y + 0.02, 11, 0.7, a, size=14, color=TEXT)
        y += 0.95
    _textbox(s, 0.7, 6.7, 12, 0.4, "建议由 AI 基于漏斗瓶颈启发式生成,供招聘例会参考", size=10, color=GRAY)

    bio = BytesIO()
    prs.save(bio)
    return bio.getvalue()
