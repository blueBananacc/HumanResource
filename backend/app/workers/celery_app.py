"""Celery 应用实例。

模块边界:
- 此文件只配置 Celery,**不**要在导入期触发任何 DB / LLM 客户端初始化,
  避免 worker 启动时 import 链拉太长 / 失败原因隐晦。
- 任务定义全部放在 ``app.workers.tasks.*``,通过 ``include`` 参数让 worker
  在 boot 时显式加载;web 进程发任务时也走相同模块路径,task name 自动一致。

启动 worker(prod):
    celery -A app.workers.celery_app worker --loglevel=info \\
        --concurrency=$CELERY_WORKER_CONCURRENCY

启动 worker(dev,docker compose):
    docker compose up -d worker

测试 / 调试(eager 模式,task 同步在调用进程执行):
    CELERY_TASK_ALWAYS_EAGER=true pytest backend/tests/test_celery_resume.py
"""
from __future__ import annotations

import logging

from celery import Celery
from celery.signals import setup_logging

from app.config import settings

# 任务模块清单(boot 时显式加载)
# 增加新任务时:在 app/workers/tasks/ 下加文件,并把 dotted path 加到这里。
# 漏加 = web 端 .delay() 发出的任务到 worker 后被 reject:
#   "Received unregistered task of type 'app.workers.tasks.xxx'..."
_INCLUDED_TASKS = [
    "app.workers.tasks.resume",
    "app.workers.tasks.interview",
    "app.workers.tasks.voice_interview",
    "app.workers.tasks.questions",
    "app.workers.tasks.email",
    "app.workers.tasks.match",
    "app.workers.tasks.notify",
]

celery_app = Celery(
    "free_hire",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
    include=_INCLUDED_TASKS,
)

# 关键配置(集中,避免分散到散文 .conf):
# - task_acks_late: worker 拿到任务后,执行成功才 ack;crash 时任务重投递。
# - worker_prefetch_multiplier=1: 每个 worker 一次只抓 1 个任务,避免长任务
#   把短任务卡在 prefetch 队列里饿死。
# - task_track_started=true: STARTED state 入 backend,前端能区分 PENDING
#   (排队中)与 STARTED (worker 在处理),诊断队列堆积更直观。
# - timezone=UTC + enable_utc=true: 与 Postgres timestamp 列保持一致。
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    task_always_eager=settings.celery_task_always_eager,
    # eager 模式下让异常透传,方便测试断言
    task_eager_propagates=settings.celery_task_always_eager,
    timezone="UTC",
    enable_utc=True,
    # 任务结果 30 天保留,够 UI 排错复盘;再长意义不大且占 redis。
    result_expires=60 * 60 * 24 * 30,
)


@setup_logging.connect
def _configure_celery_logging(**_kwargs: object) -> None:  # noqa: D401
    """让 worker 的 logging 跟 web 端格式一致(都是 logging.basicConfig)。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# ---------- job_runs 打点(T5,所有任务统一走信号,不改任务体)----------
from celery.signals import task_postrun, task_prerun  # noqa: E402

_task_started: dict[str, object] = {}


@task_prerun.connect
def _jobrun_prerun(task_id: str = "", task: object = None, **_kw: object) -> None:
    from datetime import datetime, UTC

    _task_started[task_id] = datetime.now(UTC)


@task_postrun.connect
def _jobrun_postrun(
    task_id: str = "",
    task: object = None,
    retval: object = None,
    state: str = "SUCCESS",
    **_kw: object,
) -> None:
    from datetime import datetime, UTC

    started = _task_started.pop(task_id, None) or datetime.now(UTC)
    try:
        from app.infra.db import SessionLocal
        from app.services.jobruns import record_run

        name = getattr(task, "name", "") or "unknown"
        status = "failure" if state and state != "SUCCESS" else "success"
        msg = None
        if status == "failure":
            msg = str(retval)[:256] if retval is not None else str(state)
        db = SessionLocal()
        try:
            record_run(db, name=name, started_at=started, status=status, message=msg)
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("job_runs 打点失败: %s", e)
