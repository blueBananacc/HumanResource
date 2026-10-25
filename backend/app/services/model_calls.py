"""逐次 LLM 调用留痕(T11)。

对齐 recruit `audit/writeModelCallLog`:每次模型调用落一行 ``model_call_logs``,
带 call_type / provider / model / prompt 版本 / input digest / 归一化输出 /
原始输出 / 错误分类 / 延迟。

与 audit_logs 的分工:audit 记"谁做了什么业务动作",本服务记"模型当时看到
什么、返回什么、为什么错" —— AI 可观测性层。

**独立 session 提交**:业务事务回滚不丢日志(recruit 的即时写语义);
写入失败只 warning,绝不反噬业务链路。
"""
from __future__ import annotations

import logging
from typing import Any

from app.domain.models import ModelCallLog
from app.infra.db import SessionLocal

logger = logging.getLogger(__name__)

_RAW_MAX = 20000
_ERR_MAX = 500


def write_model_call(
    *,
    tenant_id: str | None,
    call_type: str,
    provider: str,
    model: str,
    input_digest: str,
    user_id: str | None = None,
    prompt_key: str | None = None,
    prompt_version: str | None = None,
    schema_version: str = "v1",
    output: dict[str, Any] | None = None,
    raw_output: str | None = None,
    error_kind: str = "",
    error_message: str | None = None,
    latency_ms: int = 0,
) -> str | None:
    """写一行 model_call_logs,返回行 id;失败返回 None。

    用独立 SessionLocal —— 调用方业务事务回滚不影响日志落库。
    """
    try:
        db = SessionLocal()
        try:
            row = ModelCallLog(
                tenant_id=tenant_id,
                user_id=user_id,
                call_type=call_type[:48],
                provider=(provider or "")[:64],
                model=(model or "")[:128],
                prompt_key=(prompt_key or None),
                prompt_version=(prompt_version or None),
                schema_version=schema_version[:16],
                input_digest=(input_digest or "")[:16],
                output=output,
                raw_output=(raw_output or "")[:_RAW_MAX] or None,
                error_kind=(error_kind or "")[:32],
                error_message=(error_message or "")[:_ERR_MAX] or None,
                latency_ms=int(latency_ms or 0),
            )
            db.add(row)
            db.commit()
            return str(row.id)
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — 留痕失败绝不反噬业务
        logger.exception("[model_calls] 留痕写入失败 call_type=%s", call_type)
        return None
