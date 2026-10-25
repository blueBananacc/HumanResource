"""幂等键策略层(T10) —— 纯函数,无 DB/框架依赖。

移植 Gauzy ``idempotency.policy`` 的请求指纹与回放语义:

- canonical 化:同一次请求的不同拼写(JSON key 乱序、query 参数乱序、
  重解析的 body)必须产生同一 hash —— 否则重试会被误判为"复用 key"
  而被拒绝,这是唯一不可恢复的终局。
- ``build_request_hash`` = sha256([method, path, query, body])。
- 响应存储上限 1MB:超限只存 status,不存半截 body。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import parse_qsl

IDEMPOTENCY_KEY_HEADER = "idempotency-key"
REPLAYED_HEADER = "Idempotency-Replayed"
ORIGINAL_REQUEST_HEADER = "Idempotency-Original-Request"
RETRY_AFTER_HEADER = "Retry-After"

MIN_KEY_LENGTH = 8
MAX_KEY_LENGTH = 255
MAX_STORED_RESPONSE_BYTES = 1024 * 1024

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def normalize_key(value: Any) -> str | None:
    """读 header 值:数组取第一个,trim,空串视为未带。"""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def validate_key(key: str) -> str | None:
    """返回 'too-short' | 'too-long' | None(可用)。"""
    if len(key) < MIN_KEY_LENGTH:
        return "too-short"
    if len(key) > MAX_KEY_LENGTH:
        return "too-long"
    return None


def hash_prefix(h: str | None) -> str:
    """前 8 位十六进制 —— 够区分两个请求,又不足以还原 body。"""
    return (h or "")[:8]


def stable_json(value: Any) -> str:
    """确定性序列化:object key 排序、undefined→跳过。

    ``JSON.stringify`` 保插入序;客户端重建对象 / 代理重序列化时
    key 序可变 —— 排序是同请求同 hash 的前提。
    """
    if value is None or not isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(stable_json(v) for v in value) + "]"
    keys = sorted(k for k, v in value.items() if v is not None or k in value)
    return "{" + ",".join(
        f"{json.dumps(k, ensure_ascii=False)}:{stable_json(value[k])}"
        for k in keys
        if value[k] is not None
    ) + "}"


def canonicalize_path(path: Any) -> str:
    """剥掉 query/fragment —— query 已单独 hash,混进 path 会双份且拼写不一致。"""
    p = str(path or "")
    cut = len(p)
    for ch in ("?", "#"):
        i = p.find(ch)
        if i != -1:
            cut = min(cut, i)
    return p[:cut]


def canonicalize_query(query: Any) -> str:
    """排序 query 参数。重复 key 保值序(?tag=a&tag=b ≠ ?tag=b&tag=a)。"""
    if not query:
        return ""
    if isinstance(query, str):
        q = query.lstrip("?")
        if not q:
            return ""
        grouped: dict[str, list[str]] = {}
        for k, v in parse_qsl(q, keep_blank_values=True):
            grouped.setdefault(k, []).append(v)
        return stable_json(
            {k: (v[0] if len(v) == 1 else v) for k, v in sorted(grouped.items())}
        )
    # dict:值统一为 list 再排序 key;单值降维为标量与上面同形
    grouped = {}
    for k, v in query.items():
        if isinstance(v, (list, tuple)):
            grouped[k] = [str(x) for x in v]
        else:
            grouped[k] = [str(v)]
    return stable_json(
        {k: (v[0] if len(v) == 1 else v) for k, v in sorted(grouped.items())}
    )


def canonicalize_body(raw_body: bytes | str | None, parsed: Any = None) -> str:
    """已解析的 body 是请求的"意思",原始字节只是"拼法"——优先 canonical 结构。

    无解析结果但有原始字节时:能解 JSON 就同规则 canonical,否则原样
    (签名 blob/XML/纯文本的字节即内容)。
    """
    if parsed is not None and parsed != {}:
        return stable_json(parsed)
    if raw_body is None:
        return stable_json(parsed)
    raw = raw_body.decode("utf-8", "replace") if isinstance(raw_body, bytes) else str(raw_body)
    if raw.strip() == "":
        return stable_json(parsed)
    try:
        return stable_json(json.loads(raw))
    except (ValueError, TypeError):
        return raw


def build_request_hash(
    *, method: str, path: str, query: Any = None, raw_body: bytes | str | None = None,
    parsed_body: Any = None,
) -> str:
    canonical = json.dumps(
        [
            str(method or "").upper(),
            canonicalize_path(path),
            canonicalize_query(query),
            canonicalize_body(raw_body, parsed_body),
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialize_for_storage(
    body: Any, cap_bytes: int = MAX_STORED_RESPONSE_BYTES
) -> tuple[Any, bool]:
    """(可回放体, 是否因超限被丢)。超限不截断 —— 半截 JSON 无法回放。"""
    if body is None:
        return None, False
    try:
        serialized = json.dumps(body, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None, True
    if len(serialized.encode("utf-8")) > cap_bytes:
        return None, True
    return json.loads(serialized), False
