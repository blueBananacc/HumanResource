"""人事提醒中心 + 人事日历 API(慧人力智能待办 / 2号人事部人事日历同款)。

- ``GET /hr/reminders?days=30``      时间窗聚合:逾期/今天/本周/N 天内分组
- ``GET /hr/calendar?month=YYYY-MM`` 月视图:全部人事事件平铺
- ``POST /hr/reminders/greet``       生日/周年关怀一键送达员工(同日幂等)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.license import require_feature
from app.domain.models import Employee, Notification, User
from app.infra.db import get_db
from app.services.audit import write_audit
from app.services.hr_events import (
    EVENT_META,
    collect_events,
)
from app.services.permissions import (
    PERM_MANAGE_EMPLOYEES,
    require_permission,
)
from app.services.timeutil import local_day_start_utc, local_now

router = APIRouter(
    prefix="/hr",
    tags=["hris-people-events"],
    dependencies=[Depends(require_feature("hr.suite"))],
)


def _out(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        **ev,
        "day": ev["day"].isoformat(),
    }


@router.get("/reminders")
def reminders(
    days: int = Query(30, ge=1, le=365),
    kind: str | None = None,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """时间窗聚合:[已逾期 / 今天 / 本周 / N 天内]。逾期=deadline 类已过日仍有效。"""
    today = local_now().date()
    events = collect_events(db, current.tenant_id, today, today + timedelta(days=days))
    # 逾期 bucket:回看 30 天的 deadline 类 + 仍在办的入离职
    past = collect_events(
        db, current.tenant_id, today - timedelta(days=30), today - timedelta(days=1)
    )
    overdue = [e for e in past if e["category"] == "deadline"]
    if kind:
        events = [e for e in events if e["kind"] == kind]
        overdue = [e for e in overdue if e["kind"] == kind]

    upcoming = [_out(e) for e in events]
    type_counts = {
        k: sum(1 for e in events + overdue if e["kind"] == k)
        for k in EVENT_META
    }
    return {
        "overdue": [_out(e) for e in overdue],
        "today": [e for e in upcoming if _bucket_raw(e) == "today"],
        "week": [e for e in upcoming if _bucket_raw(e) == "week"],
        "later": [e for e in upcoming if _bucket_raw(e) == "later"],
        "type_counts": type_counts,
        "total": len(upcoming) + len(overdue),
        "days": days,
        "kinds": {k: v[0] for k, v in EVENT_META.items()},
    }


def _bucket_raw(e: dict) -> str:
    du = e["days_until"]
    if du == 0:
        return "today"
    if du <= 7:
        return "week"
    return "later"


@router.get("/calendar")
def calendar(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """月视图:该月全部人事事件(周期性事件已展开到当年)。"""
    y, m = month.split("-")
    start = date(int(y), int(m), 1)
    end = (
        date(int(y) + 1, 1, 1) - timedelta(days=1)
        if int(m) == 12
        else date(int(y), int(m) + 1, 1) - timedelta(days=1)
    )
    events = collect_events(db, current.tenant_id, start, end)
    return {
        "month": month,
        "items": [_out(e) for e in events],
        "kinds": {k: v[0] for k, v in EVENT_META.items()},
    }


class GreetIn(BaseModel):
    employee_id: str
    kind: str

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        if v not in ("birthday", "anniversary"):
            raise ValueError("仅支持 birthday/anniversary 祝福")
        return v


@router.post("/reminders/greet", status_code=status.HTTP_200_OK)
def greet(
    body: GreetIn,
    request: Request,
    db: Session = Depends(get_db),
    current: User = Depends(require_permission(PERM_MANAGE_EMPLOYEES)),
):
    """HR 一键送祝福给员工本人(慧人力祝福送达同款)。同日同员工同类型幂等。"""
    e = db.get(Employee, body.employee_id)
    if not e or e.tenant_id != current.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "员工不存在")
    if not e.user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "该员工未关联账号")
    label = EVENT_META[body.kind][0]
    link = f"/me#people-greet:{body.kind}:{e.id}:{local_now().date()}"
    already = db.execute(
        select(func.count(Notification.id)).where(
            Notification.user_id == e.user_id,
            Notification.kind == "people.greeting",
            Notification.link == link,
            Notification.created_at >= local_day_start_utc(),
        )
    ).scalar_one()
    sent = 0
    if not already:
        wish = "生日快乐 🎂" if body.kind == "birthday" else "入职周年快乐 🎉"
        db.add(Notification(
            tenant_id=current.tenant_id, user_id=e.user_id,
            kind="people.greeting",
            title=wish,
            body=f"{current.email} 代表公司送上{label}祝福,感谢你的付出!",
            link=link,
        ))
        sent = 1
    write_audit(
        db, actor=current, entity_type="employee", entity_id=e.id,
        action="greet", detail={"kind": body.kind, "sent": sent},
        request=request,
    )
    db.commit()
    return {"sent": sent}
