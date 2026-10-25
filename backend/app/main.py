"""FastAPI 应用入口。

挂载:
- /api/healthz                   健康检查
- /api/auth/{login,me}            鉴权
- /api/license/{status,activate}  授权
- /api/jobs                       岗位
- /api/resumes                    简历
- /api/interviews                 AI 文本面试
- /api/llm/providers              LLM 配置
- /api/email/accounts             邮箱配置 + 拉取

启动时通过 ``lifespan`` 做自检与后台任务:
- prod 环境下 JWT_SECRET 不能是默认占位
- 没有任何用户时,若 ``BOOTSTRAP_ADMIN_*`` 都已配置,创建默认 tenant + admin
- dev 环境若 jobs 表空,seed 3 条示例 JD
- 启动一个后台 asyncio 任务,每 EMAIL_SYNC_INTERVAL_SECONDS 秒拉一次启用的邮箱
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api import activity as activity_api
from app.api import ai as ai_api
from app.api import announcements as announcements_api
from app.api import applications as applications_api
from app.api import assistant as assistant_api
from app.api import attendance as attendance_api
from app.api import audit as audit_api
from app.api import automations as automations_api
from app.api import auth as auth_api
from app.api import campaigns as campaigns_api
from app.api import cases as cases_api
from app.api import email as email_api
from app.api import email_templates as email_templates_api
from app.api import employees as employees_api
from app.api import entities as entities_api
from app.api import expenses as expenses_api
from app.api import dictionaries as dictionaries_api
from app.api import doc_templates as doc_templates_api
from app.api import favorites as favorites_api
from app.api import healthz
from app.api import hiring as hiring_api
from app.api import hr_me as hr_me_api
from app.api import idempotency as idempotency_api
from app.api import imports as imports_api
from app.api import scorecards as scorecards_api
from app.api import scoring_rules as scoring_rules_api
from app.api import search as search_api
from app.api import saved_filters as saved_filters_api
from app.api import scheduler as scheduler_api
from app.api import ics as ics_api
from app.api import interview_invite as interview_invite_api
from app.api import interviews as interviews_api
from app.api import job_governance as job_governance_api
from app.api import job_knowledge as job_knowledge_api
from app.api import jobs as jobs_api
from app.api import knowledge as knowledge_api
from app.api import leave as leave_api
from app.api import license as license_api
from app.api import lifecycle as lifecycle_api
from app.api import llm as llm_api
from app.api import matches as matches_api
from app.api import mcp as mcp_api
from app.api import metrics as metrics_api
from app.api import model_calls as model_calls_api
from app.api import notifications as notifications_api
from app.api import org as org_api
from app.api import overtime as overtime_api
from app.api import payroll as payroll_api
from app.api import performance as performance_api
from app.api import prompt_templates as prompt_templates_api
from app.api import question_library as question_library_api
from app.api import question_sets as question_sets_api
from app.api import report_defs as report_defs_api
from app.api import reports as reports_api
from app.api import requests as requests_api
from app.api import resumes as resumes_api
from app.api import rooms as rooms_api
from app.api import salary_bands as salary_bands_api
from app.api import shifts as shifts_api
from app.api import skills as skills_api
from app.api import sourcing as sourcing_api
from app.api import smtp as smtp_api
from app.api import sso as sso_api
from app.api import surveys as surveys_api
from app.api import talent_profiles as talent_profiles_api
from app.api import talents as talents_api
from app.api import users as users_api
from app.api import voice as voice_api
from app.api import work_calendar as work_calendar_api
from app.config import settings
from app.domain.models import Job, Tenant, User
from app.infra.db import SessionLocal
from app.infra.json_response import UTCJSONResponse
from app.infra.locks import redis_lock
from app.infra.security import hash_password
from app.integrations.llm.registry import seed_from_env
from app.services.email_sync import sync_all_enabled
from app.services.jobruns import record_run

logger = logging.getLogger("free-hire.bootstrap")
logging.basicConfig(level=logging.INFO)


_DEFAULT_JWT = "dev-only-please-change-in-prod-32+chars"
_DEFAULT_BOOTSTRAP_ADMIN_PASSWORD = "admin123456"

# 全局 loop 锁: 多 worker 下只允许其中一个真正进入轮询临界区
LOOP_LOCK_KEY = "free-hire:lock:email-sync:loop"


def _check_jwt_secret() -> None:
    if settings.jwt_secret == _DEFAULT_JWT:
        if settings.environment != "dev":
            raise RuntimeError(
                "JWT_SECRET 仍是默认占位,prod/staging 启动被拒。请在 .env 设置 32+ 字符随机串。"
            )
        logger.warning("JWT_SECRET 使用默认占位(仅 dev 容忍),生产部署必须改。")


def _check_bootstrap_admin_credentials() -> None:
    """生产/预发环境拒绝使用示例管理员密码启动。"""
    if settings.environment == "dev":
        return
    if not settings.bootstrap_admin_password:
        return
    if settings.bootstrap_admin_password == _DEFAULT_BOOTSTRAP_ADMIN_PASSWORD:
        raise RuntimeError(
            "BOOTSTRAP_ADMIN_PASSWORD 仍是默认示例密码,prod/staging 启动被拒。"
            "请在 .env 设置强密码,首次创建管理员后建议清空 BOOTSTRAP_ADMIN_*。"
        )


def _bootstrap_admin() -> None:
    if not (settings.bootstrap_admin_email and settings.bootstrap_admin_password):
        logger.info("未配置 BOOTSTRAP_ADMIN_*,跳过默认管理员创建。")
        return
    db = SessionLocal()
    try:
        existing = db.scalars(select(User).limit(1)).first()
        if existing:
            return
        tenant = Tenant(name="default")
        db.add(tenant)
        db.flush()
        admin = User(
            tenant_id=tenant.id,
            email=settings.bootstrap_admin_email,
            password_hash=hash_password(settings.bootstrap_admin_password),
            role="admin",
        )
        db.add(admin)
        db.commit()
        logger.info("已创建默认管理员: %s (tenant=%s)", admin.email, tenant.id)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _seed_demo_data() -> None:
    """dev 环境若 jobs 表为空,塞 3 个示例 JD,让 UI 初次进入有内容可看。

    生产部署设 ENVIRONMENT=prod 即可跳过。
    """
    if settings.environment != "dev":
        return
    db = SessionLocal()
    try:
        if db.scalars(select(Job).limit(1)).first():
            return
        admin = db.scalars(select(User).where(User.role == "admin").limit(1)).first()
        if not admin:
            return
        seeds = [
            {
                "title": "Python 后端工程师",
                "level": "intermediate",
                "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                "description": (
                    "负责招聘助手后端模块开发,与 AI 面试服务联调。"
                    "要求 3 年以上 Python 经验,熟悉异步框架与 SQL 优化。"
                ),
            },
            {
                "title": "前端工程师 (React)",
                "level": "intermediate",
                "skills": ["React", "TypeScript", "Tailwind", "Vite"],
                "description": (
                    "负责 HR 工作台与候选人面试界面开发,"
                    "对企业级后台体验有审美追求。"
                ),
            },
            {
                "title": "AI 工程师 (LLM 应用)",
                "level": "advanced",
                "skills": ["Python", "PyTorch", "LLM", "RAG", "NLP"],
                "description": (
                    "负责简历结构化抽取、AI 文本/语音面试出题与多维度评分。"
                    "5+ 年经验,熟悉 prompt engineering 与评测体系。"
                ),
            },
        ]
        for s in seeds:
            db.add(
                Job(
                    tenant_id=admin.tenant_id,
                    title=s["title"],
                    level=s["level"],
                    description=s["description"],
                    skills=s["skills"],
                    created_by=admin.id,
                )
            )
        db.commit()
        logger.info("已注入 %d 条示例 JD(dev 环境)", len(seeds))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _check_jwt_secret()
    _check_bootstrap_admin_credentials()
    _bootstrap_admin()
    _seed_demo_data()
    db = SessionLocal()
    try:
        seed_from_env(db)
    finally:
        db.close()
    # 启动后台邮箱同步任务 + watcher/报表订阅调度
    task = asyncio.create_task(_email_sync_loop())
    watch_task = asyncio.create_task(_watch_loop())
    try:
        yield
    finally:
        task.cancel()
        watch_task.cancel()
        for t in (task, watch_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def _email_sync_loop() -> None:
    """后台轮询:每 settings.email_sync_interval_seconds 秒同步一次所有 enabled 账户。

    multi-worker 安全:外层用 redis ``SET NX PX`` 抢全局锁 :data:`LOOP_LOCK_KEY`,
    没拿到 → 跳过本轮(redis 不可达也走这条 fail-close 路径)。下一轮再试。
    单账户级别的进一步去重在 :func:`app.services.email_sync.sync_all_enabled` 内。
    """
    interval = settings.email_sync_interval_seconds
    if interval <= 0:
        logger.info("EMAIL_SYNC_INTERVAL_SECONDS<=0,跳过邮箱后台同步")
        return
    loop_ttl = settings.email_sync_loop_lock_ttl_seconds or interval * 2
    logger.info(
        "启动邮箱后台同步,间隔 %d 秒,loop 锁 TTL %d 秒",
        interval,
        loop_ttl,
    )
    while True:
        try:
            with redis_lock(LOOP_LOCK_KEY, loop_ttl * 1000) as token:
                if token is None:
                    logger.info(
                        "[email-sync-loop] another worker holds the lock or redis down, skip"
                    )
                else:
                    _started = datetime.now(UTC)
                    _err = None
                    try:
                        await sync_all_enabled(SessionLocal)
                    except Exception as e:  # noqa: BLE001
                        _err = e
                        raise
                    finally:
                        _db = SessionLocal()
                        try:
                            record_run(
                                _db,
                                name="email_sync_loop",
                                started_at=_started,
                                status="failure" if _err else "success",
                                message=str(_err)[:256] if _err else None,
                            )
                        finally:
                            _db.close()
        except Exception as e:  # noqa: BLE001
            logger.exception("[email-sync-loop] 异常: %s", e)
        await asyncio.sleep(interval)


WATCH_LOOP_LOCK_KEY = "free-hire:lock:watch-loop"

# 幂等键清扫节流:watch_loop 每 tick(默认 300s)都会跑,清扫每小时一次即可
_last_idem_purge: datetime | None = None


async def _watch_loop() -> None:
    """触发式任务 + 报表订阅调度:默认 300s 一轮,模式同 _email_sync_loop。

    - ``run_watchers``:评估所有 active AiWatcher,命中写 Notification(日内去重)
    - ``due_subscriptions``:本周此时应发的报表订阅 → 逐条 enqueue celery task
    """
    from app.services.watchers import run_watchers
    from app.workers.tasks.notify import due_subscriptions, send_report_subscription

    interval = settings.watch_interval_seconds
    if interval <= 0:
        logger.info("WATCH_INTERVAL_SECONDS<=0,跳过 watcher 调度")
        return
    logger.info("启动 watcher 调度循环,间隔 %d 秒", interval)

    def _tick() -> None:
        """一轮同步工作:redis 锁 + DB 评估 + celery enqueue 全是阻塞调用,
        放 worker 线程跑,不占 API 事件循环。"""
        with redis_lock(WATCH_LOOP_LOCK_KEY, interval * 2000) as token:
            if token is None:
                return
            db = SessionLocal()
            _started = datetime.now(UTC)
            _err = None
            try:
                fired = run_watchers(db)
                if fired:
                    logger.info("[watch-loop] 触发 %d 条通知", fired)
                subs = due_subscriptions(db)
                for s in subs:
                    send_report_subscription.delay(str(s.id))
                # T10:幂等键定时清扫(每小时)
                global _last_idem_purge
                if _last_idem_purge is None or (
                    _started - _last_idem_purge
                ).total_seconds() >= 3600:
                    from app.services.idempotency import purge_expired
                    from app.services.jobruns import record_run as _record_run

                    purged = purge_expired()
                    _last_idem_purge = _started
                    _record_run(
                        db,
                        name="idempotency_purge",
                        started_at=_started,
                        status="success",
                        message=f"purged={purged}" if purged else None,
                    )
                    if purged:
                        logger.info("[watch-loop] 清扫过期幂等键 %d 条", purged)
            except Exception as e:  # noqa: BLE001
                _err = e
                raise
            finally:
                record_run(
                    db,
                    name="watch_loop",
                    started_at=_started,
                    status="failure" if _err else "success",
                    message=str(_err)[:256] if _err else None,
                )
                db.close()

    while True:
        try:
            await asyncio.to_thread(_tick)
        except Exception as e:  # noqa: BLE001
            logger.exception("[watch-loop] 异常: %s", e)
        await asyncio.sleep(interval)


def create_app() -> FastAPI:
    app = FastAPI(
        title="HROnfire API",
        version=settings.app_version,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        # 全局默认: naive UTC datetime 字段输出补 ``Z``, 见 UTCJSONResponse 注释。
        default_response_class=UTCJSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(healthz.router, prefix="/api")
    app.include_router(metrics_api.router, prefix="/api")
    app.include_router(auth_api.router, prefix="/api")
    app.include_router(sso_api.router, prefix="/api")
    app.include_router(users_api.router, prefix="/api")
    app.include_router(org_api.router, prefix="/api")
    app.include_router(audit_api.router, prefix="/api")
    app.include_router(talents_api.router, prefix="/api")
    app.include_router(talents_api.groups_router, prefix="/api")
    app.include_router(license_api.router, prefix="/api")
    app.include_router(jobs_api.router, prefix="/api")
    app.include_router(job_governance_api.router, prefix="/api")
    app.include_router(resumes_api.router, prefix="/api")
    app.include_router(interviews_api.router, prefix="/api")
    app.include_router(interview_invite_api.router, prefix="/api")
    app.include_router(question_sets_api.router, prefix="/api")
    app.include_router(question_library_api.router, prefix="/api")
    app.include_router(matches_api.router, prefix="/api")
    app.include_router(reports_api.router, prefix="/api")
    app.include_router(report_defs_api.router, prefix="/api")
    app.include_router(llm_api.router, prefix="/api")
    app.include_router(email_api.router, prefix="/api")
    app.include_router(smtp_api.router, prefix="/api")
    app.include_router(voice_api.router, prefix="/api")
    app.include_router(assistant_api.router, prefix="/api")
    app.include_router(mcp_api.router, prefix="/api")
    app.include_router(mcp_api.me_router, prefix="/api")
    # HRIS 员工核心(license feature: hr.suite)
    app.include_router(employees_api.router, prefix="/api")
    app.include_router(hr_me_api.router, prefix="/api")
    app.include_router(leave_api.router, prefix="/api")
    app.include_router(attendance_api.router, prefix="/api")
    app.include_router(overtime_api.router, prefix="/api")
    app.include_router(work_calendar_api.router, prefix="/api")
    app.include_router(lifecycle_api.router, prefix="/api")
    app.include_router(hiring_api.req_router, prefix="/api")
    app.include_router(hiring_api.offer_router, prefix="/api")
    app.include_router(applications_api.router, prefix="/api")
    app.include_router(applications_api.pipeline_router, prefix="/api")
    app.include_router(performance_api.router, prefix="/api")
    app.include_router(expenses_api.router, prefix="/api")
    app.include_router(payroll_api.router, prefix="/api")
    app.include_router(prompt_templates_api.router, prefix="/api")
    app.include_router(email_templates_api.router, prefix="/api")
    app.include_router(skills_api.router, prefix="/api")
    app.include_router(skills_api.job_router, prefix="/api")
    app.include_router(skills_api.cand_router, prefix="/api")
    app.include_router(rooms_api.router, prefix="/api")
    app.include_router(rooms_api.rooms_router, prefix="/api")
    app.include_router(rooms_api.resources_router, prefix="/api")
    app.include_router(rooms_api.interviewers_router, prefix="/api")
    app.include_router(surveys_api.router, prefix="/api")
    app.include_router(surveys_api.public_router, prefix="/api")
    # T3 第二批:通知/公告/申请/带宽/工单/知识库/订阅/触达/ICS/活动流/实体检索
    app.include_router(notifications_api.router, prefix="/api")
    app.include_router(announcements_api.router, prefix="/api")
    app.include_router(announcements_api.me_router, prefix="/api")
    app.include_router(requests_api.router, prefix="/api")
    app.include_router(requests_api.me_router, prefix="/api")
    app.include_router(salary_bands_api.router, prefix="/api")
    app.include_router(cases_api.router, prefix="/api")
    app.include_router(cases_api.me_router, prefix="/api")
    app.include_router(knowledge_api.router, prefix="/api")
    app.include_router(knowledge_api.me_router, prefix="/api")
    app.include_router(campaigns_api.router, prefix="/api")
    app.include_router(ics_api.router, prefix="/api")
    app.include_router(activity_api.router, prefix="/api")
    app.include_router(entities_api.router, prefix="/api")
    app.include_router(saved_filters_api.router, prefix="/api")
    app.include_router(favorites_api.router, prefix="/api")
    app.include_router(dictionaries_api.router, prefix="/api")
    app.include_router(scheduler_api.router, prefix="/api")
    app.include_router(automations_api.router, prefix="/api")
    app.include_router(imports_api.router, prefix="/api")
    app.include_router(scorecards_api.router, prefix="/api")
    app.include_router(scoring_rules_api.router, prefix="/api")
    app.include_router(search_api.router, prefix="/api")
    app.include_router(idempotency_api.router, prefix="/api")
    app.include_router(job_knowledge_api.router, prefix="/api")
    app.include_router(model_calls_api.router, prefix="/api")
    app.include_router(ai_api.router, prefix="/api")
    app.include_router(doc_templates_api.router, prefix="/api")
    app.include_router(talent_profiles_api.router, prefix="/api")
    app.include_router(sourcing_api.router, prefix="/api")
    app.include_router(shifts_api.router, prefix="/api")

    return app


app = create_app()
