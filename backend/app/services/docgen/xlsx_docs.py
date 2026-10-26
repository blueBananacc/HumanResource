"""xlsx 文档引擎:薪酬底表 / 候选人清单 / 管道明细。

统一风格:首行标题(14pt 加粗),表头深色底白字,斑马纹,
列宽按内容自适应;金额右对齐两位小数。
"""
from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Employee,
    Job,
    JobApplication,
    OrgUnit,
    Resume,
    ResumeJobMatch,
    SalarySlip,
)

_HDR_FILL = PatternFill("solid", fgColor="83143D")
_HDR_FONT = Font(bold=True, color="FFFFFF", size=11)
_TITLE_FONT = Font(bold=True, size=14, color="3A3A37")
_ZEBRA = PatternFill("solid", fgColor="FDF6EB")
_THIN = Border(*[Side(style="thin", color="E8E2D9")] * 4)
_STAGE_ZH = {
    "applied": "已申请",
    "screening": "筛选中",
    "interview": "面试中",
    "offer": "Offer",
    "hired": "已入职",
    "rejected": "已淘汰",
    "withdrawn": "已撤回",
}


def _sheet(title: str, headers: list[str]) -> tuple[Workbook, object]:
    wb = Workbook()
    ws = wb.active
    ws.append([title])
    ws["A1"].font = _TITLE_FONT
    ws.append(headers)
    for c in ws[2]:
        c.fill = _HDR_FILL
        c.font = _HDR_FONT
        c.border = _THIN
        c.alignment = Alignment(horizontal="center", vertical="center")
    return wb, ws


def _finish(ws, rows: list[list], widths: list[float] | None = None) -> bytes:
    for i, row in enumerate(rows):
        ws.append(row)
        for c in ws[ws.max_row]:
            c.border = _THIN
            c.alignment = Alignment(vertical="center", wrap_text=False)
            if i % 2 == 1:
                c.fill = _ZEBRA
    for idx, w in enumerate(widths or [], start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w
    if not widths:
        for idx, col in enumerate(ws.iter_cols(min_row=2), start=1):
            longest = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(idx)].width = min(max(longest * 1.6, 10), 42)
    bio = BytesIO()
    ws.parent.save(bio)
    return bio.getvalue()


def payroll_sheet(db: Session, tenant_id: str, period: str) -> bytes:
    """薪酬底表:某月全部工资条 + 明细项展开。给下游财务/发薪系统用。"""
    slips = db.execute(
        select(SalarySlip, Employee)
        .join(Employee, Employee.id == SalarySlip.employee_id)
        .where(SalarySlip.tenant_id == tenant_id, SalarySlip.period == period)
        .order_by(Employee.employee_no)
    ).all()
    org_names = {
        o.id: o.name
        for o in db.scalars(select(OrgUnit).where(OrgUnit.tenant_id == tenant_id)).all()
    }

    wb, ws = _sheet(f"{period} 薪酬底表", ["工号", "姓名", "部门", "应发合计", "扣款合计", "实发", "状态", "明细"])
    rows = []
    for s, e in slips:
        detail = "; ".join(f"{i.get('label', '')}:{i.get('amount', 0)}" for i in (s.items or []))
        rows.append([
            e.employee_no, e.name, org_names.get(e.org_unit_id, "-"),
            round(s.gross, 2), round(s.deductions, 2), round(s.net, 2),
            "已签发" if s.status == "issued" else "草稿", detail,
        ])
    return _finish(ws, rows, [10, 12, 16, 12, 12, 12, 8, 60])


def candidate_list(db: Session, tenant_id: str, scope: list[str] | None = None) -> bytes:
    """候选人清单:基本信息 + 技能标签 + 最佳匹配分。scope 非 None 时按组织过滤。"""
    stmt = (
        select(Candidate)
        .where(Candidate.tenant_id == tenant_id)
        .order_by(Candidate.created_at.desc())
        .limit(500)
    )
    if scope is not None:
        from app.services.permissions import apply_org_filter

        stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    cands = db.scalars(stmt).all()
    all_resumes = db.scalars(
        select(Resume).where(Resume.tenant_id == tenant_id)
    ).all()
    skills_of = {
        r.candidate_id: (r.parsed_data or {}).get("skills") or []
        for r in all_resumes
    }
    source_of = {r.candidate_id: r.source for r in all_resumes}
    best = {}
    for m in db.scalars(
        select(ResumeJobMatch).where(
            ResumeJobMatch.tenant_id == tenant_id, ResumeJobMatch.score.is_not(None)
        )
    ).all():
        rid = m.resume_id
        best[rid] = max(best.get(rid, 0), m.score or 0)
    resume_owner = {r.id: r.candidate_id for r in db.scalars(select(Resume).where(Resume.tenant_id == tenant_id)).all()}

    wb, ws = _sheet("候选人清单", ["姓名", "邮箱", "电话", "来源渠道", "技能", "标签", "最佳匹配分", "入库时间"])
    rows = []
    for c in cands:
        score = max(
            (s for rid, s in best.items() if resume_owner.get(rid) == c.id), default=None
        )
        rows.append([
            c.name, c.display_email or "-", c.display_phone or "-", source_of.get(c.id, "-"),
            ", ".join(skills_of.get(c.id, [])[:12]),
            ", ".join(c.tags or []),
            score if score is not None else "-",
            c.created_at.strftime("%Y-%m-%d"),
        ])
    return _finish(ws, rows, [12, 26, 14, 10, 50, 24, 12, 12])


def pipeline_sheet(
    db: Session, tenant_id: str, job_id: str | None = None, scope: list[str] | None = None
) -> bytes:
    """管道明细:候选人 × 岗位 申请卡 + 阶段 + 匹配分 + 负责人。"""
    stmt = (
        select(JobApplication, Candidate, Job)
        .join(Candidate, Candidate.id == JobApplication.candidate_id)
        .join(Job, Job.id == JobApplication.job_id)
        .where(JobApplication.tenant_id == tenant_id)
        .order_by(JobApplication.stage_changed_at.desc())
        .limit(500)
    )
    if job_id:
        stmt = stmt.where(JobApplication.job_id == job_id)
    if scope is not None:
        from app.services.permissions import apply_org_filter

        stmt = apply_org_filter(stmt, org_column=Candidate.org_unit_id, scope=scope)
    apps = db.execute(stmt).all()
    cand_resume = {
        r.candidate_id: r.id for r in db.scalars(select(Resume).where(Resume.tenant_id == tenant_id)).all()
    }
    scores = {
        (m.resume_id, m.job_id): m.score
        for m in db.scalars(
            select(ResumeJobMatch).where(ResumeJobMatch.tenant_id == tenant_id)
        ).all()
    }

    title = "招聘管道明细" + (f" · {apps[0][2].title}" if job_id and apps else "")
    wb, ws = _sheet(title, ["候选人", "岗位", "阶段", "匹配分", "渠道", "淘汰原因", "更新时间"])
    rows = []
    for a, c, j in apps:
        rid = cand_resume.get(c.id)
        rows.append([
            c.name, j.title, _STAGE_ZH.get(a.stage, a.stage),
            scores.get((rid, a.job_id), "-") if rid else "-",
            a.source, a.rejected_reason or "-",
            a.stage_changed_at.strftime("%Y-%m-%d %H:%M"),
        ])
    return _finish(ws, rows, [12, 26, 10, 10, 10, 30, 18])
