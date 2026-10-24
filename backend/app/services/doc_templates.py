"""文档模板注册表 + 变量构建(T6-4,MintHCM MailMerge 的智能替代)。

``DOC_SPECS`` 是内置模板(录用通知/在职证明/转正通知);租户可在
``doc_templates`` 表写同 key 覆盖内容,或建 ``custom.<uuid>`` key 的自定义模板。
``build_doc_vars`` 产出扁平 ``{{var}}`` 值,渲染时先 ``render_vars`` 预填,
残余占位交 LLM 补全。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Employee,
    Job,
    JobApplication,
    OrgUnit,
    Tenant,
    User,
)
from app.services.ai_summary import get_scoped_row


def _tenant_name(db: Session, u: User) -> str:
    t = db.get(Tenant, u.tenant_id)
    return t.name if t else ""


class DocSpec:
    def __init__(
        self,
        *,
        name: str,
        entity: str,
        body: str,
        variables: list[dict[str, str]],
    ) -> None:
        self.name = name
        self.entity = entity  # candidates / employees / any
        self.body = body
        self.variables = variables


DOC_SPECS: dict[str, DocSpec] = {
    "offer_letter": DocSpec(
        name="录用通知书",
        entity="candidates",
        variables=[
            {"name": "candidate_name", "desc": "候选人姓名"},
            {"name": "job_title", "desc": "岗位名"},
            {"name": "company_name", "desc": "公司名(可空,LLM 通用表述)"},
        ],
        body=(
            "录用通知书\n\n"
            "尊敬的 {{candidate_name}} 女士/先生:\n\n"
            "您好!非常高兴地通知您,经过综合评估,我们决定正式邀请您加入"
            "{{company_name}},担任 {{job_title}} 一职。\n\n"
            "一、岗位信息\n"
            "1. 岗位名称:{{job_title}}\n"
            "2. 汇报对象:{{manager}}\n"
            "3. 工作地点:{{work_location}}\n"
            "4. 到岗日期:{{onboard_date}}\n\n"
            "二、薪酬福利\n"
            "1. 月薪(税前):{{salary}}\n"
            "2. 试用期:{{probation_months}} 个月\n"
            "3. 其他福利:按公司制度执行\n\n"
            "请您在收到本通知后 {{reply_days}} 日内回复确认。期待与您共事!\n\n"
            "{{company_name}} 人力资源部\n{{doc_date}}"
        ),
    ),
    "employment_certificate": DocSpec(
        name="在职证明",
        entity="employees",
        variables=[
            {"name": "employee_name", "desc": "员工姓名"},
            {"name": "employee_no", "desc": "工号"},
            {"name": "designation", "desc": "职位"},
            {"name": "org_name", "desc": "部门"},
            {"name": "hire_date", "desc": "入职日期"},
            {"name": "company_name", "desc": "公司名(可空)"},
        ],
        body=(
            "在职证明\n\n"
            "兹证明 {{employee_name}}(工号:{{employee_no}})系我公司正式员工,"
            "自 {{hire_date}} 起在我公司 {{org_name}} 担任 {{designation}} 职务,"
            "目前在职状态正常。\n\n"
            "本证明仅用于证明上述员工的在职情况,不作其他用途。\n\n"
            "特此证明。\n\n"
            "{{company_name}} 人力资源部\n{{doc_date}}"
        ),
    ),
    "probation_notice": DocSpec(
        name="转正通知书",
        entity="employees",
        variables=[
            {"name": "employee_name", "desc": "员工姓名"},
            {"name": "employee_no", "desc": "工号"},
            {"name": "designation", "desc": "职位"},
            {"name": "org_name", "desc": "部门"},
            {"name": "hire_date", "desc": "入职日期"},
            {"name": "company_name", "desc": "公司名(可空)"},
        ],
        body=(
            "转正通知书\n\n"
            "{{employee_name}} 同事(工号:{{employee_no}}):\n\n"
            "您于 {{hire_date}} 入职我公司 {{org_name}} {{designation}} 岗位,"
            "试用期已满。经考核评估,您在试用期内的工作表现符合岗位要求,"
            "现正式予以转正,转正后职位、薪酬按公司相关制度执行。\n\n"
            "希望您继续努力,与公司共同成长!\n\n"
            "{{company_name}} 人力资源部\n{{doc_date}}"
        ),
    ),
}


def _d(v) -> str | None:
    return v.strftime("%Y-%m-%d") if v is not None else None


def build_doc_vars(
    db: Session, u: User, scope, entity: str, rid: str
) -> dict[str, Any]:
    """按实体产出扁平 {{var}} 值;scope 校验复用 ai_summary.get_scoped_row。"""
    from fastapi import HTTPException

    if entity == "candidates":
        c = get_scoped_row(db, u, scope, "candidates", rid)
        app_row = db.execute(
            JobApplication.__table__.select()
            .where(
                JobApplication.tenant_id == u.tenant_id,
                JobApplication.candidate_id == c.id,
            )
            .order_by(JobApplication.created_at.desc())
            .limit(1)
        ).first()
        job = db.get(Job, app_row.job_id) if app_row else None
        org = db.get(OrgUnit, c.org_unit_id) if c.org_unit_id else None
        return {
            "candidate_name": c.name,
            "job_title": job.title if job else "",
            "org_name": org.name if org else "",
            "company_name": _tenant_name(db, u),
        }
    if entity == "employees":
        e = get_scoped_row(db, u, scope, "employees", rid)
        org = db.get(OrgUnit, e.org_unit_id) if e.org_unit_id else None
        mgr = db.get(Employee, e.reports_to_id) if e.reports_to_id else None
        return {
            "employee_name": e.name,
            "employee_no": e.employee_no,
            "designation": e.designation or "",
            "org_name": org.name if org else "",
            "manager": mgr.name if mgr else "",
            "hire_date": _d(e.hire_date) or "",
            "employment_type": e.employment_type,
            "status": e.status,
            "company_name": _tenant_name(db, u),
        }
    raise HTTPException(400, f"不支持起草文档的实体: {entity}")
