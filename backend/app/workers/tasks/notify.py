"""通知类异步任务:人才外联批量发送 + 报表订阅邮件。

与 ``tasks/email.py`` 同一模式:任务内开 SessionLocal,SMTP 取租户
active 账户,失败记 last_error 不抛出(批量里逐条记 failed 继续)。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.api.smtp import _account_to_config, get_active_account
from app.domain.models import (
    Candidate,
    JobApplication,
    Notification,
    ReportSubscription,
    TalentCampaign,
    TalentCampaignSend,
    User,
)
from app.infra.db import SessionLocal
from app.integrations.email.smtp_sender import SMTPSendError, send_email
from app.services.mail_templates import render_email
from app.services.timeutil import local_day_start_utc, local_now
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _render_campaign_body(template: str, cand: Candidate | None) -> str:
    name = cand.name if cand else "候选人"
    return template.replace("{{candidate_name}}", name)


@celery_app.task(name="app.workers.tasks.notify.send_talent_campaign")
def send_talent_campaign(campaign_id: str) -> dict:
    db = SessionLocal()
    try:
        c = db.get(TalentCampaign, campaign_id)
        if c is None:
            return {"ok": False, "reason": "not_found"}
        # 幂等:只有 send 端点置 sending 后才执行;重复投递/手动重跑
        # 在终态(done/failed/cancelled)上直接 no-op,不能把 failed 改写成 done
        if c.status != "sending":
            return {"ok": False, "reason": f"status:{c.status}"}
        smtp = get_active_account(db, c.tenant_id)
        cfg = _account_to_config(smtp) if smtp else None
        queued = db.scalars(
            select(TalentCampaignSend).where(
                TalentCampaignSend.campaign_id == c.id,
                TalentCampaignSend.status == "queued",
            )
        ).all()
        sent = failed = 0
        for s in queued:
            # 发送中途被 /cancel 时,sends 行已在库内转 cancelled —— 逐条刷新
            # 防止按快照把已取消的收件人重发出去
            db.refresh(s)
            if s.status != "queued":
                continue
            if cfg is None:
                s.status = "failed"
                s.error = "smtp_not_configured"
                failed += 1
                continue
            cand = db.get(Candidate, s.candidate_id)
            try:
                send_email(
                    cfg,
                    to=s.to_email,
                    subject=c.subject.replace("{{candidate_name}}", cand.name if cand else ""),
                    text=_render_campaign_body(c.body_text, cand),
                )
                s.status = "sent"
                s.sent_at = _now()
                sent += 1
            except SMTPSendError as e:
                s.status = "failed"
                s.error = str(e)[:256]
                failed += 1
            except Exception as e:  # noqa: BLE001
                # 单条脏数据兜底:任何意外异常都只记该条 failed,
                # 绝不能中断循环 —— 否则活动永远卡在 sending 且无错误明细
                logger.warning(
                    "campaign send failed campaign=%s send=%s err=%s", c.id, s.id, e
                )
                s.status = "failed"
                s.error = f"{type(e).__name__}: {e}"[:256]
                failed += 1
        # autoflush=False(SessionLocal 配置):计数前先显式 flush 循环里的状态变更
        db.flush()
        c.sent_count = int(
            db.execute(
                select(func.count(TalentCampaignSend.id)).where(
                    TalentCampaignSend.campaign_id == c.id,
                    TalentCampaignSend.status == "sent",
                )
            ).scalar_one()
        )
        c.failed_count = int(
            db.execute(
                select(func.count(TalentCampaignSend.id)).where(
                    TalentCampaignSend.campaign_id == c.id,
                    TalentCampaignSend.status.in_(("failed", "skipped_no_email")),
                )
            ).scalar_one()
        )
        # cancel 与任务并发:cancel 已把状态置 cancelled,不要覆盖回 done。
        # 直接查列(refresh 会把上面未提交的计数改动一起回滚掉)
        current_status = db.execute(
            select(TalentCampaign.status).where(TalentCampaign.id == c.id)
        ).scalar_one()
        if current_status != "cancelled":
            # 全数失败 → failed(配合 send_campaign 的 failed→queued 重置,可整批重试);
            # 部分成功 → done(失败明细在 sends 行可见)
            c.status = "failed" if failed > 0 and sent == 0 else "done"
            c.done_at = _now()
        db.commit()
        return {"ok": True, "sent": sent, "failed": failed}
    finally:
        db.close()


def _digest_for(db, sub: ReportSubscription) -> tuple[str, str] | None:
    """按 kind 生成 (title, body);无内容返回 None 跳过发送。"""
    now = _now()
    if sub.kind == "pipeline_funnel":
        rows = db.execute(
            select(JobApplication.stage, func.count()).where(
                JobApplication.tenant_id == sub.tenant_id
            ).group_by(JobApplication.stage)
        ).all()
        if not rows:
            return None
        lines = [f"{stage}: {n}" for stage, n in sorted(rows)]
        return ("管道漏斗周报", "当前各阶段候选人分布:\n\n" + "\n".join(lines))
    if sub.kind == "weekly_summary":
        from app.domain.models import Interview

        week_ago = now.replace(hour=0, minute=0, second=0) - timedelta(days=7)
        new_apps = db.execute(
            select(func.count(JobApplication.id)).where(
                JobApplication.tenant_id == sub.tenant_id,
                JobApplication.created_at >= week_ago,
            )
        ).scalar_one()
        done_ivs = db.execute(
            select(func.count(Interview.id)).where(
                Interview.tenant_id == sub.tenant_id,
                Interview.status == "done",
                Interview.finished_at >= week_ago,
            )
        ).scalar_one()
        return ("每周招聘摘要", f"过去 7 天:\n\n- 新管道卡:{new_apps}\n- 完成面试:{done_ivs}")
    if sub.kind == "pending_items":
        from app.domain.models import EmployeeRequest, HrCase

        cases = db.execute(
            select(func.count(HrCase.id)).where(
                HrCase.tenant_id == sub.tenant_id,
                HrCase.status.in_(("new", "assigned", "in_progress")),
            )
        ).scalar_one()
        reqs = db.execute(
            select(func.count(EmployeeRequest.id)).where(
                EmployeeRequest.tenant_id == sub.tenant_id,
                EmployeeRequest.status == "pending",
            )
        ).scalar_one()
        unread = db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == sub.user_id,
                Notification.read_at.is_(None),
            )
        ).scalar_one()
        if not cases and not reqs and not unread:
            return None
        return (
            "待办事项提醒",
            f"当前待办:\n\n- 未结工单:{cases}\n- 待审批申请:{reqs}\n- 未读通知:{unread}",
        )
    return None


@celery_app.task(name="app.workers.tasks.notify.send_report_subscription")
def send_report_subscription(subscription_id: str) -> dict:
    db = SessionLocal()
    try:
        # 行锁 + 执行期查重:_watch_loop 在命中小时内每 tick(5min)都会把
        # 今日未发的订阅重复入队;任务若排队超时,首个副本写 last_sent_at
        # 之前可能已积压多个副本 —— FOR UPDATE 串行化,后到副本读到新值即跳过。
        sub = db.scalars(
            select(ReportSubscription)
            .where(ReportSubscription.id == subscription_id)
            .with_for_update()
        ).first()
        if sub is None or not sub.is_active:
            return {"ok": False, "reason": "inactive"}
        if sub.last_sent_at and sub.last_sent_at >= local_day_start_utc():
            return {"ok": True, "skipped": "sent_today"}
        user = db.get(User, sub.user_id)
        if user is None or not user.email:
            return {"ok": False, "reason": "no_recipient"}
        if user.status != "active":
            # 账号已停用:不再投递;标 last_sent 避免该小时每 tick 重试
            sub.last_sent_at = _now()
            db.commit()
            return {"ok": False, "reason": "user_inactive"}
        from app.services.permissions import PERM_VIEW_REPORTS, has_permission

        if not has_permission(user, PERM_VIEW_REPORTS):
            # 权限被降后不再投递租户聚合数据;标 last_sent 避免每 tick 重试
            sub.last_sent_at = _now()
            db.commit()
            return {"ok": False, "reason": "no_permission"}
        digest = _digest_for(db, sub)
        if digest is None:
            sub.last_sent_at = _now()
            db.commit()
            return {"ok": True, "skipped": "empty"}
        title, body = digest
        rendered = render_email(
            db, sub.tenant_id, "report.digest",
            {"digest_title": title, "digest_body": body},
        )
        subject, text, html = rendered if rendered else (title, body, None)
        smtp = get_active_account(db, sub.tenant_id)
        cfg = _account_to_config(smtp) if smtp else None
        if cfg is None:
            return {"ok": False, "reason": "smtp_not_configured"}
        try:
            send_email(cfg, to=user.email, subject=subject, text=text, html=html)
        except SMTPSendError as e:
            return {"ok": False, "reason": "send_failed", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            # 兜底:渲染/编码等意外异常按发送失败返回,不让任务异常逃逸
            logger.warning("send_report_subscription %s failed: %s", subscription_id, e)
            return {
                "ok": False,
                "reason": "send_failed",
                "error": f"{type(e).__name__}: {e}",
            }
        sub.last_sent_at = _now()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


def due_subscriptions(db) -> list[ReportSubscription]:
    """_watch_loop 调用:当前小时应投递且今日未发过的订阅。

    weekday/hour 是用户按本地时间配置的("每周一 9 点"),与"现在"比较
    用部署时区口径;"今日已发"的日界换算回 UTC 后再与 last_sent_at(UTC
    时间戳)比。
    """
    now = local_now()
    today_start = local_day_start_utc()
    return db.scalars(
        select(ReportSubscription).where(
            ReportSubscription.is_active.is_(True),
            ReportSubscription.weekday == now.weekday(),
            ReportSubscription.hour == now.hour,
            (ReportSubscription.last_sent_at.is_(None))
            | (ReportSubscription.last_sent_at < today_start),
        )
    ).all()
