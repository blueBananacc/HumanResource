"""naive datetime 的两套口径:UTC 时间戳 vs 客户端墙钟。

- ``utc_now()`` —— 系统写入的时间戳(created_at / finished_at / read_at /
  last_sent_at / last_fired_at …)一律 UTC naive;``UTCJSONResponse`` 序列化
  补 ``Z``,前端按浏览器时区渲染。

- ``local_now()`` —— 用户录入的排期类字段(预约 starts_at/ends_at、公告
  expires_at、报表订阅 weekday/hour)存的是「客户端墙钟」:前端
  datetime-local 输入的本地时间原样落库,不做时区换算(见
  ``app.api.rooms._naive`` 与 ``Schedule.tsx`` 的 ``toLocal``)。所以
  "这条墙钟时间到了没有"也必须用同一墙钟口径的"现在"来比 —— 即部署
  时区(``settings.app_timezone``)的当前时间;容器跑 UTC 不受影响。

- ``local_day_start_utc()`` —— "今天是否已发/已通知"这类把 UTC 落库
  时间戳按本地日界分桶的查重。若直接拿本地 naive 0 点和 UTC 时间戳比,
  会把时区差混进桶边界,所以先把本地日界换算回 UTC 再比较。
"""
from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from app.config import settings


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def local_now() -> datetime:
    """部署时区(``app_timezone``)当前时间,naive —— 与墙钟字段同一口径。"""
    return datetime.now(ZoneInfo(settings.app_timezone)).replace(tzinfo=None)


def local_day_start_utc() -> datetime:
    """本地今日 0 点对应的 UTC 时刻(naive)。"""
    today = local_now().date()
    return (
        datetime.combine(today, time.min, tzinfo=ZoneInfo(settings.app_timezone))
        .astimezone(UTC)
        .replace(tzinfo=None)
    )
