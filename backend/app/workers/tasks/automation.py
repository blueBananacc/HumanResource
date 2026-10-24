"""自动化规则邮件投递。

``app.services.automation`` 在业务事务 after_commit 后把 ``queued_email``
状态的 run 交给这里;payload 已在分发时渲染好,本任务只负责 SMTP 发送并
回写 run.status(sent / email_failed),失败 silent — 不影响业务。
"""
from __future__ import annotations

import logging
from typing import Any

from app.api.smtp import _account_to_config, get_active_account
from app.domain.models import AutomationRun
from app.infra.db import SessionLocal
from app.integrations.email.smtp_sender import SMTPSendError, send_email
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _wrap_html(title: str, body: str) -> str:
    esc = (
        body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return (
        "<div style='font-family:system-ui,-apple-system,sans-serif;"
        "max-width:600px;margin:0 auto;padding:24px;'>"
        f"<h3 style='margin:0 0 12px;color:#111827;'>{title}</h3>"
        f"<div style='color:#374151;font-size:14px;line-height:1.7;'>{esc}</div>"
        "<p style='color:#6B7280;font-size:12px;border-top:1px solid #E5E7EB;"
        "padding-top:12px;margin-top:24px;'>— HROnfire 自动化通知</p>"
        "</div>"
    )


@celery_app.task(
    name="app.workers.tasks.automation.send_automation_email",
    bind=True,
    autoretry_for=(),
    max_retries=0,
)
def send_automation_email(self: Any, run_id: str) -> dict[str, Any]:
    db = SessionLocal()
    try:
        run = db.get(AutomationRun, run_id)
        if run is None or run.status != "queued_email":
            return {"ok": False, "reason": "not_queued"}
        payload = run.payload or {}
        to = [e for e in (payload.get("to") or []) if e]
        if not to:
            run.status = "skipped_no_recipients"
            db.commit()
            return {"ok": False, "reason": "no_recipient"}

        smtp = get_active_account(db, run.tenant_id)
        if smtp is None:
            run.status = "email_failed"
            run.error = "smtp_not_configured"
            db.commit()
            return {"ok": False, "reason": "smtp_not_configured"}
        cfg = _account_to_config(smtp)
        if cfg is None:
            run.status = "email_failed"
            run.error = "smtp_decrypt_failed"
            db.commit()
            return {"ok": False, "reason": "decrypt_failed"}

        title = payload.get("title") or "自动化通知"
        text = payload.get("body") or title
        html = _wrap_html(title, text)
        failed: list[str] = []
        for addr in to:
            try:
                send_email(cfg, to=addr, subject=title, text=text, html=html)
            except SMTPSendError as e:
                logger.warning("send_automation_email to=%s err=%s", addr, e)
                failed.append(addr)
        if failed:
            run.status = "email_failed"
            run.error = f"发送失败:{', '.join(failed)}"[:1000]
            smtp.last_status = "error"
            smtp.last_error = run.error[:2000]
            db.commit()
            return {"ok": False, "reason": "send_failed", "failed": failed}

        run.status = "sent"
        run.error = None
        smtp.last_status = "ok"
        smtp.last_error = None
        db.commit()
        return {"ok": True, "to": to}
    finally:
        db.close()
