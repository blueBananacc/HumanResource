"""邮件模板:注册表 + 渲染。

发送点::

    out = render_email(db, tenant_id, "interview.invite", vars)
    if out:
        subject, text, html = out
    else:
        ...内置文案兜底...

- ``{{var}}`` 在 subject / body_text 中按原文替换;在 body_html 中
  替换前对值做 ``html.escape``(候选人名等用户输入不进 markup)。
- 未提供的变量原样保留;body_html 为空 → html=None(只发纯文本)。
"""
from __future__ import annotations

import html as _html

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import EmailTemplate
from app.services.prompting import render_vars


class EmailSpec:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        subject: str,
        body_text: str,
        body_html: str | None,
        variables: list[dict[str, str]],
    ) -> None:
        self.name = name
        self.description = description
        self.subject = subject
        self.body_text = body_text
        self.body_html = body_html
        self.variables = variables


EMAIL_SPECS: dict[str, EmailSpec] = {
    "interview.invite": EmailSpec(
        name="面试邀请",
        description="HR 发起远程面试后发给候选人的邀请邮件",
        subject="[面试邀请] {{job_title}}",
        variables=[
            {"name": "candidate_name", "desc": "候选人姓名"},
            {"name": "job_title", "desc": "岗位名"},
            {"name": "invite_url", "desc": "专属面试链接"},
            {"name": "expires_at_label", "desc": "截止时间文本"},
            {"name": "question_count", "desc": "题目数"},
            {"name": "sender_name", "desc": "发件人显示名"},
        ],
        body_text=(
            "{{candidate_name}} 您好:\n"
            "\n"
            "感谢你投递「{{job_title}}」岗位。\n"
            "我们邀请你完成一场 AI 面试 — 全程文字答题,共 {{question_count}} 道题,\n"
            "请在以下时间前完成:{{expires_at_label}}\n"
            "\n"
            "专属链接(请勿转发):\n"
            "{{invite_url}}\n"
            "\n"
            "提交后 AI 将自动评分,结果由 HR 审阅,我们会尽快与你联系。\n"
            "\n"
            "如有疑问,请回复本邮件联系 HR。\n"
            "祝顺利。\n"
            "\n"
            "— {{sender_name}}"
        ),
        body_html=(
            "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "color:#0F1115;line-height:1.6;max-width:560px;'>"
            "<p>{{candidate_name}} 您好:</p>"
            "<p>感谢你投递「<b>{{job_title}}</b>」岗位。<br>"
            "我们邀请你完成一场 AI 面试 — 全程文字答题,共 <b>{{question_count}}</b> 道题,"
            "请在 <b>{{expires_at_label}}</b> 前完成。</p>"
            "<p style='margin:24px 0;'>"
            "<a href='{{invite_url}}' "
            "style='display:inline-block;background:#0F1115;color:#fff;"
            "padding:10px 20px;border-radius:8px;text-decoration:none;'>"
            "开始面试</a></p>"
            "<p style='color:#6B7280;font-size:12px;'>"
            "如果按钮无法点击,请复制以下链接到浏览器打开:<br>"
            "<span style='word-break:break-all;'>{{invite_url}}</span></p>"
            "<p>提交后 AI 将自动评分,结果由 HR 审阅,我们会尽快与你联系。</p>"
            "<p style='color:#6B7280;font-size:12px;border-top:1px solid #E5E7EB;"
            "padding-top:12px;margin-top:24px;'>"
            "— {{sender_name}}</p>"
            "</div>"
        ),
    ),
    "interview.done": EmailSpec(
        name="面试完成通知",
        description="候选人交卷后发给发起面试的 HR",
        subject="[面试完成] {{candidate_name}} · {{job_title}}",
        variables=[
            {"name": "candidate_name", "desc": "候选人姓名"},
            {"name": "job_title", "desc": "岗位名"},
            {"name": "average", "desc": "平均分(可空)"},
            {"name": "recommendation", "desc": "推荐结论(可空)"},
            {"name": "report_url", "desc": "面试报告链接"},
        ],
        body_text=(
            "候选人「{{candidate_name}}」已完成「{{job_title}}」的远程 AI 面试。\n"
            "\n"
            "平均分 {{average}} · 建议:{{recommendation}}\n"
            "\n"
            "查看完整报告:\n"
            "{{report_url}}\n"
            "\n"
            "— HROnfire 自动通知\n"
        ),
        body_html=(
            "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "color:#0F1115;line-height:1.6;max-width:560px;'>"
            "<p>候选人「<b>{{candidate_name}}</b>」已完成「<b>{{job_title}}</b>」"
            "的远程 AI 面试。</p>"
            "<table style='border-collapse:collapse;margin:16px 0;'>"
            "<tr><td style='padding:4px 16px 4px 0;color:#6B7280;'>平均分</td>"
            "<td>{{average}}</td></tr>"
            "<tr><td style='padding:4px 16px 4px 0;color:#6B7280;'>推荐结论</td>"
            "<td>{{recommendation}}</td></tr>"
            "</table>"
            "<p><a href='{{report_url}}' "
            "style='display:inline-block;background:#0F1115;color:#fff;"
            "padding:8px 16px;border-radius:8px;text-decoration:none;'>"
            "查看报告</a></p>"
            "<p style='color:#6B7280;font-size:12px;border-top:1px solid #E5E7EB;"
            "padding-top:12px;margin-top:24px;'>— HROnfire 自动通知</p>"
            "</div>"
        ),
    ),
    "report.digest": EmailSpec(
        name="定时报表摘要",
        description="报表订阅的周期邮件;body 由后端按订阅类型生成",
        subject="{{digest_title}}",
        variables=[
            {"name": "digest_title", "desc": "邮件标题"},
            {"name": "digest_body", "desc": "正文(多行纯文本)"},
        ],
        body_text="{{digest_body}}\n\n— HROnfire 定时报表",
        body_html=None,
    ),
}


def _rendered_vars(vars: dict[str, object], *, escape: bool) -> dict[str, object]:
    if not escape:
        return vars
    return {k: _html.escape(str(v), quote=True) for k, v in vars.items()}


def render_email(
    db: Session, tenant_id: str, key: str, vars: dict[str, object]
) -> tuple[str, str, str | None] | None:
    """查租户自定义模板并渲染 → ``(subject, text, html|None)``;无则 None。"""
    tpl = db.scalars(
        select(EmailTemplate).where(
            EmailTemplate.tenant_id == tenant_id,
            EmailTemplate.key == key,
            EmailTemplate.is_active.is_(True),
        )
    ).first()
    if tpl is None:
        return None
    subject = render_vars(tpl.subject, vars)
    text = render_vars(tpl.body_text, vars)
    html_body = (
        render_vars(tpl.body_html, _rendered_vars(vars, escape=True))
        if tpl.body_html
        else None
    )
    return subject, text, html_body
