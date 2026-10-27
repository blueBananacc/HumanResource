"""两级简历查重 + 候选人锁定(移植 Moka 机制)。

查重两级(与 Moka「精准查重 / 疑似查重」对齐):

- **精准(exact)**:姓名+手机 / 姓名+邮箱 / 手机+邮箱 两通道命中
  → 同一人的多份档案,可一键合并;
- **疑似(suspect)**:仅单通道命中(手机号或邮箱或姓名其一)
  → 候选人挂 ``dedup_flag='suspect'`` + ``dedup_peer_id``,进入待审队列,
  HR 合并或「确认非重复」摘标。

入库钩子(:func:`flag_suspects`)在候选人新建 / 联系方式变更时运行;
历史存量用 ``scan_tenant`` 全量重扫。锁定(:func:`assert_unlocked`)
在候选人变更操作前调用,防止内部抢人 — 锁只挡非持锁人。
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.domain.models import Candidate

logger = logging.getLogger(__name__)

FLAG_SUSPECT = "suspect"
FLAG_DISMISSED = "dismissed"


def _is_dismissed_pair(c: Candidate, peer: Candidate) -> bool:
    """任一侧已对另一侧做过「确认非重复」→ 不再挂标。"""
    return (
        c.dedup_flag == FLAG_DISMISSED
        and c.dedup_peer_id == peer.id
    ) or (
        peer.dedup_flag == FLAG_DISMISSED
        and peer.dedup_peer_id == c.id
    )


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# 两级查重
# ---------------------------------------------------------------------------

def match_channels(a: Candidate, b: Candidate) -> set[str]:
    """两个候选人命中的重复通道:name / email / phone。"""
    ch: set[str] = set()
    if a.name and b.name and a.name.strip() == b.name.strip():
        ch.add("name")
    if a.email_hash and b.email_hash and a.email_hash == b.email_hash:
        ch.add("email")
    if a.phone_hash and b.phone_hash and a.phone_hash == b.phone_hash:
        ch.add("phone")
    return ch


def tier_of(channels: set[str]) -> str | None:
    """命中通道集合 → 精准("exact")/ 疑似("suspect")/ 无。"""
    if not channels:
        return None
    # 精准:任一联系方式 + 另一通道(等价于 Moka 姓名+手机 / 姓名+邮箱 / 手机+邮箱)
    if len(channels) >= 2:
        return "exact"
    return "suspect"


def find_dup_peers(db: Session, c: Candidate) -> list[tuple[Candidate, str, set[str]]]:
    """租户内与 ``c`` 至少单通道重复的其他候选人 → [(peer, tier, channels)]。

    name 用等值兜底 — 单看 name 撞车概率高,但 Moka 疑似级的语义就是
    "宁可多标、由人来确认",所以保留;队列里精准级优先展示。
    """
    conds = [Candidate.name == c.name]
    if c.email_hash:
        conds.append(Candidate.email_hash == c.email_hash)
    if c.phone_hash:
        conds.append(Candidate.phone_hash == c.phone_hash)
    rows = db.scalars(
        select(Candidate).where(
            Candidate.tenant_id == c.tenant_id,
            Candidate.id != c.id,
            or_(*conds),
        ).limit(50)
    ).all()
    out: list[tuple[Candidate, str, set[str]]] = []
    for r in rows:
        ch = match_channels(c, r)
        tier = tier_of(ch)
        if tier:
            out.append((r, tier, sorted(ch)))
    out.sort(key=lambda x: (x[1] != "exact", -len(x[2])))
    return out


def flag_suspects(db: Session, c: Candidate) -> None:
    """入库钩子:新候选人/联系方式变更后,存在重复嫌疑 → 挂疑标。

    已合并源(dedup_flag 手动摘除后)不重复挂标,除非是新命中的对端。
    """
    peers = find_dup_peers(db, c)
    live = [p for p in peers if not _is_dismissed_pair(c, p[0])]
    if not live:
        if peers and c.dedup_flag != FLAG_DISMISSED:
            # 对端全被确认非重复:维持 dismissed 记忆
            c.dedup_flag = FLAG_DISMISSED
            c.dedup_peer_id = peers[0][0].id
        elif not peers and c.dedup_flag == FLAG_SUSPECT:
            c.dedup_flag = None
            c.dedup_peer_id = None
        return
    peer, tier, _ch = live[0]
    if c.dedup_flag == FLAG_SUSPECT and c.dedup_peer_id == peer.id:
        return
    c.dedup_flag = FLAG_SUSPECT
    c.dedup_peer_id = peer.id


def dismiss_suspect(db: Session, c: Candidate) -> None:
    """HR 确认非重复 → 标 dismissed(保留 peer 指向,重扫不再挂标)。"""
    c.dedup_flag = FLAG_DISMISSED


def scan_tenant(
    db: Session, tenant_id: str, *, limit: int = 500
) -> dict[str, Any]:
    """全量重扫:返回 {exact:[pair...], suspect:[pair...]},并把疑似对挂标。

    pair = {candidate, peer, channels};精准对按 (candidate,peer) 去重,
    疑似同理 — 队列页每条只出现一次(保留 created_at 较早者为 candidate)。
    """
    cands = db.scalars(
        select(Candidate)
        .where(Candidate.tenant_id == tenant_id)
        .order_by(Candidate.created_at)
        .limit(limit)
    ).all()
    exact: list[dict[str, Any]] = []
    suspect: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    flagged = 0
    for c in cands:
        peers = find_dup_peers(db, c)
        live = [p for p in peers if not _is_dismissed_pair(c, p[0])]
        if not live:
            if peers:
                c.dedup_flag = FLAG_DISMISSED
                c.dedup_peer_id = peers[0][0].id
            elif c.dedup_flag == FLAG_SUSPECT:
                c.dedup_flag = None
                c.dedup_peer_id = None
            continue
        peer, tier, ch = live[0]
        key = frozenset({c.id, peer.id})
        if key not in seen:
            seen.add(key)
            row = {"candidate": c, "peer": peer, "channels": ch}
            (exact if tier == "exact" else suspect).append(row)
        if c.dedup_flag != FLAG_SUSPECT or c.dedup_peer_id != peer.id:
            c.dedup_flag = FLAG_SUSPECT
            c.dedup_peer_id = peer.id
            flagged += 1
    db.flush()
    return {
        "exact": exact,
        "suspect": suspect,
        "scanned": len(cands),
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# 候选人锁定 / 释放
# ---------------------------------------------------------------------------

DEFAULT_LOCK_DAYS = 7
MAX_LOCK_DAYS = 90


def is_locked(c: Candidate, now: datetime | None = None) -> bool:
    """锁仍生效(``locked_until`` 未过)。过期即自动释放。"""
    if not c.locked_by_id or not c.locked_until:
        return False
    now = now or _now()
    return c.locked_until > now


def lock_candidate(
    db: Session, c: Candidate, user_id: str, *, days: int = DEFAULT_LOCK_DAYS
) -> None:
    now = _now()
    days = max(1, min(days, MAX_LOCK_DAYS))
    c.locked_by_id = user_id
    c.locked_at = now
    c.locked_until = now + timedelta(days=days)


def unlock_candidate(db: Session, c: Candidate) -> None:
    c.locked_by_id = None
    c.locked_at = None
    c.locked_until = None


def assert_unlocked(c: Candidate, user_id: str, *, now: datetime | None = None) -> None:
    """被他人锁定且未过期 → 409。持锁人本人不受限。"""
    if is_locked(c, now) and c.locked_by_id != user_id:
        from fastapi import HTTPException

        raise HTTPException(
            409,
            f"候选人已被锁定(至 {c.locked_until:%Y-%m-%d}),仅持锁人可操作",
        )
