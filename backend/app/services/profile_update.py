"""P 批次 — 人才画像更新管线。

设计(调研:Moka 智能标签"规则自动打标" + 北森"盘点校准"):

- 所有 AI/证据驱动的技能建议都落 ``status='suggested'``,由 HR/主管在
  卡片页或助手确认后转 ``confirmed`` —— AI 永不直接写 confirmed。
- 唯一键 (employee_id, skill_id):重复证据只升级(level 取高、
  evidence 补注),不插重复行。
- 确定性种子(简历技能、面试维度)直接 confirmed —— 来源可信。

触发点(P2 在 API 层挂载):
- 入职建卡 ``seed_card_from_candidate``
- insight_note 新建 ``suggest_from_text``
- 项目完成 ``suggest_from_project_done``
- 绩效复核 ``suggest_from_appraisal``
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Appraisal, CandidateSkill, Employee, EmployeeSkill, Interview,
    Project, ProjectMember, Skill,
)

logger = logging.getLogger(__name__)


def upsert_employee_skill(
    db: Session, *, tenant_id: str, employee_id: str, skill_id: str,
    level: int = 3, source: str = "manual", status: str = "suggested",
    evidence: dict[str, Any] | None = None, note: str | None = None,
    created_by: str | None = None,
) -> EmployeeSkill:
    """同员工同技能唯一:存在则合并(level 取高、confirmed 不降级)。"""
    level = max(1, min(5, int(level or 3)))
    row = db.scalars(
        select(EmployeeSkill).where(
            EmployeeSkill.employee_id == employee_id,
            EmployeeSkill.skill_id == skill_id,
        )
    ).first()
    if row:
        row.level = max(row.level, level)
        if status == "confirmed":
            row.status = "confirmed"
        if evidence:
            row.evidence = evidence
        if note:
            row.note = note
        return row
    row = EmployeeSkill(
        tenant_id=tenant_id, employee_id=employee_id, skill_id=skill_id,
        level=level, source=source, status=status, evidence=evidence,
        note=note, created_by=created_by,
    )
    db.add(row)
    db.flush()
    return row


def get_or_create_skill(db: Session, *, tenant_id: str, name: str,
                        category: str = "other") -> Skill:
    """按名称查技能字典,缺则建(LLM 提取到的新技能名走这里)。"""
    name = (name or "").strip()[:64]
    row = db.scalars(
        select(Skill).where(Skill.tenant_id == tenant_id, Skill.name == name)
    ).first()
    if row:
        return row
    row = Skill(tenant_id=tenant_id, name=name, category=category)
    db.add(row)
    db.flush()
    return row


def seed_card_from_candidate(db: Session, employee: Employee) -> int:
    """入职建卡:从 source_candidate 迁移确定性证据。

    - CandidateSkill → EmployeeSkill(confirmed, source=resume)
    - 返回迁移条数。LLM 总结放卡片展示层,不在此处做。
    """
    if not employee.source_candidate_id:
        return 0
    rows = db.scalars(
        select(CandidateSkill).where(
            CandidateSkill.candidate_id == employee.source_candidate_id
        )
    ).all()
    n = 0
    for cs in rows:
        upsert_employee_skill(
            db, tenant_id=employee.tenant_id, employee_id=employee.id,
            skill_id=cs.skill_id, level=cs.level, source=cs.source or "resume",
            status="confirmed",
            evidence={"kind": "resume", "id": employee.source_candidate_id},
        )
        n += 1
    return n


def suggest_from_text(
    db: Session, *, tenant_id: str, employee: Employee, text: str,
    source: str, evidence: dict[str, Any] | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    """LLM 从自由文本提取能力证据 → suggested 技能。

    返回 {"suggested": [...], "skipped": [...]};LLM 不可用时返回空并告警。
    """
    from app.integrations.llm.provider import LLMError, call_llm_json

    dict_names = [
        r[0] for r in db.execute(
            select(Skill.name).where(Skill.tenant_id == tenant_id)
        ).all()
    ]
    sys_p = (
        "你是 HR 人才画像标注器。从工作证据文本中提取该员工展现出的能力,"
        "只输出 JSON:{\"skills\": [{\"name\": \"技能名\", \"level\": 1-5, "
        "\"evidence\": \"≤30字出处\"}]}\n"
        "规则:技能名≤12字、可数(如「系统设计」「跨团队沟通」「Python」),"
        "不要写「努力」「认真」这类态度词;level 按证据强度给;"
        "最多 6 条,无证据返回空数组。禁止 JSON 之外的文字。"
    )
    user = (
        f"员工: {employee.name}({employee.designation or ''})\n"
        f"已有技能字典(优先复用同名): {dict_names[:80]}\n"
        f"证据文本(截断): {(text or '')[:3000]}"
    )
    try:
        raw = call_llm_json(
            "talent.suggest",
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user}],
            db=db, tenant_id=tenant_id, required=("skills",),
            temperature=0.2, prompt_key="talent.suggest",
        )
    except LLMError as e:
        logger.warning("画像建议 LLM 失败 emp=%s: %s", employee.id, e)
        return {"suggested": [], "skipped": [], "llm_error": str(e)}

    out, skipped = [], []
    for item in (raw.get("skills") or [])[:6]:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        skill = get_or_create_skill(db, tenant_id=tenant_id, name=name)
        ev = dict(evidence or {})
        if item.get("evidence"):
            ev["quote"] = str(item["evidence"])[:60]
        row = upsert_employee_skill(
            db, tenant_id=tenant_id, employee_id=employee.id,
            skill_id=skill.id, level=item.get("level") or 3,
            source=source, status="suggested", evidence=ev,
            created_by=created_by,
        )
        (skipped if row.status == "confirmed" else out).append(skill.name)
    return {"suggested": out, "skipped": skipped}


def suggest_from_project_done(db: Session, project: Project) -> int:
    """项目完成 → 对每个成员用 contribution/required_skills 提建议。"""
    members = db.scalars(
        select(ProjectMember).where(ProjectMember.project_id == project.id)
    ).all()
    n = 0
    for m in members:
        emp = db.get(Employee, m.employee_id)
        if not emp:
            continue
        text = (
            f"项目「{project.name}」完成。{project.description or ''}\n"
            f"成员角色: {m.role or ''}\n贡献: {m.contribution_summary or ''}"
        )
        r = suggest_from_text(
            db, tenant_id=project.tenant_id, employee=emp, text=text,
            source="project", evidence={"kind": "project", "id": project.id},
        )
        n += len(r.get("suggested") or [])
    return n


def suggest_from_appraisal(db: Session, appraisal: Appraisal) -> int:
    """绩效复核完成 → 从 manager_review 提取能力证据。"""
    emp = db.get(Employee, appraisal.employee_id)
    if not emp or not (appraisal.manager_review or "").strip():
        return 0
    r = suggest_from_text(
        db, tenant_id=appraisal.tenant_id, employee=emp,
        text=appraisal.manager_review, source="appraisal",
        evidence={"kind": "appraisal", "id": appraisal.id},
    )
    return len(r.get("suggested") or [])
