"""多评价者技能评分(Degreed 式多源评分 + iCIMS 反馈请求流)

断言:
- 自评:POST /me/skills/{id}/rate → SkillRating(self) + EmployeeSkill confirmed
- 请求流:员工发起(默认直属上级)→ 评价人任务列表 → 打分 → completed +
  SkillRating + EmployeeSkill source=manager + 双方通知
- 校验:重复 pending 409 / 自评他评混用 422 / 未绑账号 422 / 非评价人 404
- 生命周期:decline / cancel 后不可评分;同一评价人重复打分 upsert
- 卡片:talent-card 透出 ratings 数组 + divergent 分歧标记(极差≥2)
- 权限:HR/经理代发请求;跨租户隔离
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-sr")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-sr"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def client():
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"sr-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, role="admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    return u


def _emp(db, t: Tenant, name, *, user=None, reports_to=None,
         status="active") -> m.Employee:
    e = m.Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:8]}",
        name=name, user_id=user.id if user else None,
        reports_to_id=reports_to.id if reports_to else None,
        status=status,
    )
    db.add(e)
    db.commit()
    return e


def _skill(db, t: Tenant, name="Python") -> m.Skill:
    s = m.Skill(tenant_id=t.id, name=name, category="tech")
    db.add(s)
    db.commit()
    return s


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _eskill(db, emp_id, skill_id) -> m.EmployeeSkill | None:
    return db.scalars(
        select(m.EmployeeSkill).where(
            m.EmployeeSkill.employee_id == emp_id,
            m.EmployeeSkill.skill_id == skill_id,
        )
    ).first()


def _notes(db, tid, user_id) -> list[m.Notification]:
    return db.scalars(
        select(m.Notification).where(
            m.Notification.tenant_id == tid,
            m.Notification.user_id == user_id,
            m.Notification.kind == "skill_rating",
        )
    ).all()


def _setup_pair(db, t: Tenant):
    """ratee + 直属经理(都绑账号)+ 技能 → (ratee, mgr, skill, ratee_u, mgr_u)。"""
    ratee_u = _user(db, t, role="employee")
    mgr_u = _user(db, t, role="employee")
    mgr = _emp(db, t, "李经理", user=mgr_u)
    ratee = _emp(db, t, "张三", user=ratee_u, reports_to=mgr)
    skill = _skill(db, t)
    return ratee, mgr, skill, ratee_u, mgr_u


# ---------- 自评 ----------

def test_self_rate_writes_rating_and_skill(db, client):
    """自评落 SkillRating(self) + EmployeeSkill confirmed/source=self。"""
    t = _tenant(db)
    u = _user(db, t, role="employee")
    emp = _emp(db, t, "员工甲", user=u)
    skill = _skill(db, t)

    r = client.post(f"/api/me/skills/{skill.id}/rate", headers=_hdr(u),
                    json={"level": 4, "comment": "独立交付两个项目"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["rater_kind"] == "self" and body["level"] == 4
    assert body["rater_name"] == "员工甲"

    es = _eskill(db, emp.id, skill.id)
    assert es is not None and es.status == "confirmed"
    assert es.source == "self" and es.level == 4
    assert es.evidence["kind"] == "rating"

    # 重复自评 upsert(仍一条)
    r = client.post(f"/api/me/skills/{skill.id}/rate", headers=_hdr(u),
                    json={"level": 5})
    assert r.status_code == 201
    rows = db.scalars(
        select(m.SkillRating).where(m.SkillRating.employee_id == emp.id)
    ).all()
    assert len(rows) == 1 and rows[0].level == 5

    # 卡片透出 ratings
    r = client.get("/api/me/talent-card", headers=_hdr(u))
    item = next(s for s in r.json()["skills_confirmed"]
                if s["skill_id"] == skill.id)
    assert len(item["ratings"]) == 1
    assert item["ratings"][0]["rater_kind"] == "self"
    assert item["divergent"] is False


def test_self_rate_validations(db, client):
    """未绑账号 403;技能不存在/跨租户 404。"""
    t = _tenant(db)
    admin = _user(db, t)  # 未绑员工
    skill = _skill(db, t)
    r = client.post(f"/api/me/skills/{skill.id}/rate", headers=_hdr(admin),
                    json={"level": 3})
    assert r.status_code == 403

    u = _user(db, t, role="employee")
    _emp(db, t, "员工", user=u)
    r = client.post("/api/me/skills/nonexistent/rate", headers=_hdr(u),
                    json={"level": 3})
    assert r.status_code == 404
    # 跨租户技能
    t2 = _tenant(db)
    s2 = _skill(db, t2, "Java")
    r = client.post(f"/api/me/skills/{s2.id}/rate", headers=_hdr(u),
                    json={"level": 3})
    assert r.status_code == 404
    # level 越界
    r = client.post(f"/api/me/skills/{skill.id}/rate", headers=_hdr(u),
                    json={"level": 9})
    assert r.status_code == 422


# ---------- 请求 → 评分主流程 ----------

def test_request_rate_full_flow(db, client):
    """员工请直属上级评分:创建→经理任务→打分→completed+回写+通知。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, mgr_u = _setup_pair(db, t)

    # 员工发起(默认评价人=直属上级)
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id, "note": "请按最近项目表现评"})
    assert r.status_code == 201, r.text
    req = r.json()
    assert req["status"] == "pending"
    assert req["rater_kind"] == "manager"      # 直属上级 → manager
    assert req["rater_name"] == "李经理"
    assert req["skill_name"] == "Python"

    # 评价人收到通知
    assert len(_notes(db, t.id, mgr_u.id)) == 1

    # 经理的待办
    r = client.get("/api/me/skill-rating-tasks", headers=_hdr(mgr_u))
    assert r.status_code == 200
    items = r.json()["items"]
    assert r.json()["pending_count"] == 1
    assert items[0]["id"] == req["id"] and items[0]["ratee_name"] == "张三"

    # 经理打分
    r = client.post(f"/api/me/skill-rating-tasks/{req['id']}/rate",
                    headers=_hdr(mgr_u),
                    json={"level": 5, "comment": "代码质量高"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["request"]["status"] == "completed"
    assert body["rating"]["rater_kind"] == "manager" and body["rating"]["level"] == 5

    # EmployeeSkill 回写 source=manager confirmed
    es = _eskill(db, ratee.id, skill.id)
    assert es is not None and es.status == "confirmed"
    assert es.source == "manager" and es.level == 5
    assert es.evidence["kind"] == "rating"
    assert es.evidence["rater_kind"] == "manager"

    # 被评人收到完成通知
    db.expire_all()
    notes = _notes(db, t.id, ratee_u.id)
    assert any("5" in n.title for n in notes)

    # 请求行回填 rating_id + decided_at
    q = db.get(m.SkillRatingRequest, req["id"])
    assert q.rating_id == body["rating"]["id"] and q.decided_at is not None

    # 评分列表透出
    admin = _user(db, t)
    r = client.get(f"/api/employees/{ratee.id}/skill-ratings",
                   headers=_hdr(admin))
    assert r.status_code == 200
    assert len(r.json()["items"]) == 1

    # 已完成的请求不能再评分
    r = client.post(f"/api/me/skill-rating-tasks/{req['id']}/rate",
                    headers=_hdr(mgr_u), json={"level": 3})
    assert r.status_code == 409


def test_request_defaults_and_kind(db, client):
    """同事评价人 → rater_kind=peer;显式传上级 → manager。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, _ = _setup_pair(db, t)
    peer_u = _user(db, t, role="employee")
    peer = _emp(db, t, "同事丙", user=peer_u)

    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id, "rater_employee_id": peer.id})
    assert r.status_code == 201
    assert r.json()["rater_kind"] == "peer"

    # 显式传上级(与默认等价)→ manager;但已有 pending 给上级的无重复校验对象不同
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id, "rater_employee_id": mgr.id})
    assert r.status_code == 201
    assert r.json()["rater_kind"] == "manager"


def test_request_validation(db, client):
    """校验:重复 pending 409;他评指向本人 422;未绑账号 422;离职 422;
    无上级且未指定 422;跨租户评价人 404。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, _ = _setup_pair(db, t)
    hdr = _hdr(ratee_u)

    # 重复 pending
    body = {"skill_id": skill.id}
    assert client.post("/api/me/skill-rating-requests", headers=hdr,
                       json=body).status_code == 201
    r = client.post("/api/me/skill-rating-requests", headers=hdr, json=body)
    assert r.status_code == 409

    # 他评指向本人
    r = client.post("/api/me/skill-rating-requests", headers=hdr,
                    json={"skill_id": skill.id, "rater_employee_id": ratee.id})
    assert r.status_code == 422 and "自评" in r.json()["detail"]

    # 评价人未绑账号
    orphan = _emp(db, t, "无账号")
    s2 = _skill(db, t, "Go")
    r = client.post("/api/me/skill-rating-requests", headers=hdr,
                    json={"skill_id": s2.id, "rater_employee_id": orphan.id})
    assert r.status_code == 422 and "账号" in r.json()["detail"]

    # 离职评价人
    u3 = _user(db, t, role="employee")
    gone = _emp(db, t, "已离职", user=u3, status="terminated")
    s3 = _skill(db, t, "Rust")
    r = client.post("/api/me/skill-rating-requests", headers=hdr,
                    json={"skill_id": s3.id, "rater_employee_id": gone.id})
    assert r.status_code == 422 and "离职" in r.json()["detail"]

    # 跨租户评价人
    t2 = _tenant(db)
    u2 = _user(db, t2, role="employee")
    alien = _emp(db, t2, "外人", user=u2)
    s4 = _skill(db, t, "TS")
    r = client.post("/api/me/skill-rating-requests", headers=hdr,
                    json={"skill_id": s4.id, "rater_employee_id": alien.id})
    assert r.status_code == 404

    # 无直属上级且不指定 → 422
    solo_u = _user(db, t, role="employee")
    _emp(db, t, "无上级", user=solo_u)
    s5 = _skill(db, t, "SQL")
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(solo_u),
                    json={"skill_id": s5.id})
    assert r.status_code == 422 and "直属上级" in r.json()["detail"]


def test_decline_and_cancel(db, client):
    """婉拒 → declined + 通知;撤回 → cancelled + 任务不可评分。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, mgr_u = _setup_pair(db, t)

    # --- decline ---
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id})
    req1 = r.json()["id"]
    r = client.post(f"/api/me/skill-rating-tasks/{req1}/decline",
                    headers=_hdr(mgr_u), json={"reason": "没合作过"})
    assert r.status_code == 200 and r.json()["status"] == "declined"
    db.expire_all()
    assert any("婉拒" in n.title for n in _notes(db, t.id, ratee_u.id))
    # declined 后不能再评分
    r = client.post(f"/api/me/skill-rating-tasks/{req1}/rate",
                    headers=_hdr(mgr_u), json={"level": 4})
    assert r.status_code == 409

    # --- cancel(被评人撤回)---
    s2 = _skill(db, t, "Java")
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": s2.id})
    req2 = r.json()["id"]
    r = client.post(f"/api/me/skill-rating-requests/{req2}/cancel",
                    headers=_hdr(ratee_u))
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    r = client.post(f"/api/me/skill-rating-tasks/{req2}/rate",
                    headers=_hdr(mgr_u), json={"level": 4})
    assert r.status_code == 409
    # 撤回后可对同评价人重发(pending 查重不挡 cancelled)
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": s2.id})
    assert r.status_code == 201


def test_task_not_mine_404(db, client):
    """别人不能替评价人打分;跨租户任务不可见。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, mgr_u = _setup_pair(db, t)
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id})
    req = r.json()["id"]

    # 其它员工拿不到任务
    other_u = _user(db, t, role="employee")
    _emp(db, t, "路人", user=other_u)
    r = client.post(f"/api/me/skill-rating-tasks/{req}/rate",
                    headers=_hdr(other_u), json={"level": 3})
    assert r.status_code == 404

    # 跨租户
    t2 = _tenant(db)
    alien_u = _user(db, t2, role="employee")
    _emp(db, t2, "外人", user=alien_u)
    r = client.post(f"/api/me/skill-rating-tasks/{req}/rate",
                    headers=_hdr(alien_u), json={"level": 3})
    assert r.status_code == 404
    # 任务列表租户隔离
    r = client.get("/api/me/skill-rating-tasks", headers=_hdr(alien_u))
    assert r.json()["items"] == []


def test_curate_request_and_divergence(db, client):
    """HR 代发请求(peer)+ 多源评分分歧标记(极差≥2)。"""
    t = _tenant(db)
    admin = _user(db, t)
    ratee, mgr, skill, ratee_u, mgr_u = _setup_pair(db, t)
    peer_u = _user(db, t, role="employee")
    peer = _emp(db, t, "同事", user=peer_u)

    # HR 代员工发起同事评分
    r = client.post(f"/api/employees/{ratee.id}/skill-rating-requests",
                    headers=_hdr(admin),
                    json={"skill_id": skill.id,
                          "rater_employee_id": peer.id,
                          "note": "校准用"})
    assert r.status_code == 201, r.text
    assert r.json()["rater_kind"] == "peer"

    # 同事评分 2 级 + 员工自评 5 级 → 极差 3 ≥ 2 → divergent
    client.post(f"/api/me/skill-rating-tasks/{r.json()['id']}/rate",
                headers=_hdr(peer_u), json={"level": 2})
    client.post(f"/api/me/skills/{skill.id}/rate", headers=_hdr(ratee_u),
                json={"level": 5})

    r = client.get(f"/api/employees/{ratee.id}/talent-card",
                   headers=_hdr(admin))
    item = next(s for s in r.json()["skills_confirmed"]
                if s["skill_id"] == skill.id)
    assert item["divergent"] is True
    kinds = {x["rater_kind"] for x in item["ratings"]}
    assert kinds == {"peer", "self"}
    # 经理评 4 补上 → 3 源仍 divergent
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id})
    client.post(f"/api/me/skill-rating-tasks/{r.json()['id']}/rate",
                headers=_hdr(mgr_u), json={"level": 4})
    r = client.get(f"/api/employees/{ratee.id}/talent-card",
                   headers=_hdr(admin))
    item = next(s for s in r.json()["skills_confirmed"]
                if s["skill_id"] == skill.id)
    assert len(item["ratings"]) == 3 and item["divergent"] is True
    # pending 请求透出
    s2 = _skill(db, t, "Go")
    client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                json={"skill_id": s2.id})
    r = client.get(f"/api/employees/{ratee.id}/talent-card",
                   headers=_hdr(admin))
    pend = r.json()["pending_rating_requests"]
    assert len(pend) == 1 and pend[0]["rater_kind"] == "manager"


def test_my_requests_list(db, client):
    """我发出的请求列表按时间倒序,含状态。"""
    t = _tenant(db)
    ratee, mgr, skill, ratee_u, mgr_u = _setup_pair(db, t)
    r = client.post("/api/me/skill-rating-requests", headers=_hdr(ratee_u),
                    json={"skill_id": skill.id})
    req = r.json()["id"]
    r = client.get("/api/me/skill-rating-requests", headers=_hdr(ratee_u))
    items = r.json()["items"]
    assert len(items) == 1 and items[0]["id"] == req
    assert items[0]["status"] == "pending" and items[0]["rater_name"] == "李经理"
