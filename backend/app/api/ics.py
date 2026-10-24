"""ICS 日历导出(MintHCM iCals/vCals 移植)。

- ``GET /interviews/{id}/ics`` → 面试 VEVENT 下载(auth + scope)
- ``GET /reservations/{id}/ics`` → 预约 VEVENT 下载

手写 ~30 行 ICS builder(RFC5545:CRLF 行尾、75 字节折行、,;\n 转义),
不引 icalendar 依赖。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    Candidate,
    Interview,
    Job,
    Reservation,
    Resource,
    Room,
    User,
)
from app.infra.db import get_db
from app.services.permissions import (
    PERM_VIEW_REPORTS,
    ensure_can_see,
    get_org_scope,
    require_permission,
)

router = APIRouter(tags=["ics"])


def _escape(text: str) -> str:
    # RFC5545:先转义 \ 自身(必须先做,否则把后面的转义符再转一次),
    # 再 ; 、 , ;CRLF / 单独 CR 都归一为 \n 转义序列
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """RFC5545 75 字节折行 —— 不拆多字节字符。

    按字节硬切会在 UTF-8 多字节字符中间断开,产生的续行既是非法 ICS、
    join 后 decode 也直接 UnicodeDecodeError。逐字符累计编码长度:
    首行内容 ≤75 字节;续行以 1 个空格开头计入限额 → 内容 ≤74 字节。
    """
    if len(line.encode("utf-8")) <= 75:
        return line
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    limit = 75
    for ch in line:
        n = len(ch.encode("utf-8"))
        if cur_len + n > limit:
            lines.append("".join(cur))
            cur = [ch]
            cur_len = n
            limit = 74
        else:
            cur.append(ch)
            cur_len += n
    lines.append("".join(cur))
    return "\r\n ".join(lines)


def _dt(dt: datetime, *, floating: bool = False) -> str:
    """``floating=True`` → RFC5545 本地浮动时间(无 Z),日历按打开者时区显示。

    预约的 starts_at/ends_at 是「客户端表达的墙钟时间」(见 rooms._naive),
    必须浮动输出;UTC 语义的时间戳(interview.started_at / DTSTAMP)用 Z。
    """
    return dt.strftime("%Y%m%dT%H%M%S" if floating else "%Y%m%dT%H%M%SZ")


def build_ics(
    *,
    uid: str,
    summary: str,
    start: datetime,
    end: datetime,
    location: str | None = None,
    description: str | None = None,
    floating: bool = False,
) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//HROnfire//CN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}@hronfire",
        f"DTSTAMP:{_dt(datetime.now(UTC).replace(tzinfo=None))}",
        f"DTSTART:{_dt(start, floating=floating)}",
        f"DTEND:{_dt(end, floating=floating)}",
        f"SUMMARY:{_escape(summary)}",
    ]
    if location:
        lines.append(f"LOCATION:{_escape(location)}")
    if description:
        lines.append(f"DESCRIPTION:{_escape(description)}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


def _ics_response(ics: str, filename: str) -> Response:
    return Response(
        content=ics,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/interviews/{iid}/ics")
def interview_ics(
    iid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    iv = db.get(Interview, iid)
    if iv is None or iv.tenant_id != current.tenant_id:
        raise HTTPException(404, "面试不存在")
    scope = get_org_scope(db, current)
    cand = db.get(Candidate, iv.candidate_id)
    job = db.get(Job, iv.job_id)
    if cand:
        ensure_can_see(scope, cand.org_unit_id)
    # 面试关联的预约(如有)取其时段,否则给 1 小时占位;
    # 只看 active —— 已取消预约的时段是陈旧数据,不应导进日历
    rsv = db.scalars(
        select(Reservation)
        .where(
            Reservation.interview_id == iid,
            Reservation.status == "active",
        )
        .order_by(Reservation.starts_at)
        .limit(1)
    ).first()
    floating = rsv is not None
    if rsv:
        start, end, loc = rsv.starts_at, rsv.ends_at, None
        if rsv.room_id:
            room = db.get(Room, rsv.room_id)
            loc = room.name if room else None
    else:
        start = iv.finished_at or iv.started_at
        end = start + timedelta(hours=1)
        loc = "远程 AI 面试"
    title = f"面试:{cand.name if cand else '?'} × {job.title if job else '?'}"
    desc = f"AI 面试 · 岗位 {job.title if job else '-'} · 模式 {iv.modality}"
    return _ics_response(
        build_ics(
            uid=f"interview-{iid}", summary=title, start=start, end=end,
            location=loc, description=desc, floating=floating,
        ),
        f"interview-{iid[:8]}.ics",
    )


@router.get("/reservations/{rid}/ics")
def reservation_ics(
    rid: str,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_VIEW_REPORTS)),
):
    r = db.get(Reservation, rid)
    if r is None or r.tenant_id != current.tenant_id:
        raise HTTPException(404, "预约不存在")
    scope = get_org_scope(db, current)
    loc = None
    if r.room_id:
        room = db.get(Room, r.room_id)
        if room:
            ensure_can_see(scope, room.org_unit_id)
            loc = room.name + (f"({room.location})" if room.location else "")
        # room 已删(悬空引用)→ 无法再定 org,按租户内可见放行
    elif r.resource_id:
        res = db.get(Resource, r.resource_id)
        if res:
            # Resource 没有 org 维度 —— 与 rooms.list_reservations 同口径,
            # 租户内公共可见
            loc = res.name
    elif r.interview_id:
        # 不占会议室/资源的面试预约:跟 interview_ics 同一口径,
        # 按候选人的 org_unit 收敛,否则同租户任何账号都能导出标题。
        iv = db.get(Interview, r.interview_id)
        cand = db.get(Candidate, iv.candidate_id) if iv else None
        if cand:
            ensure_can_see(scope, cand.org_unit_id)
    return _ics_response(
        build_ics(
            uid=f"reservation-{rid}",
            summary=r.title,
            start=r.starts_at,
            end=r.ends_at,
            location=loc,
            description=f"HROnfire 预约 · {r.kind}",
            floating=True,
        ),
        f"reservation-{rid[:8]}.ics",
    )
