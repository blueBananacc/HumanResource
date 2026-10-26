# -*- coding: utf-8 -*-
"""P 批次 — 人才画像/项目/AI 选人测试。

覆盖:
- upsert_employee_skill 合并语义(唯一键、level 取高、confirmed 不降级)
- 入职建卡 seed_card_from_candidate(CandidateSkill → confirmed)
- 技能 API:手动加技能 confirmed、建议确认/驳回、越权 403
- insight 创建(免 LLM 路径 + mock LLM 提炼建议)
- 画像聚合结构 + 本人视图 suggested 隐藏
- 项目 CRUD/成员(重复 409、离项)、done 触发成员建议
- /talent/search:片段匹配、min_level、suggested 默认不计、排序
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-p")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-p"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base, Candidate, CandidateSkill, Employee, EmployeeSkill,
    Project, ProjectMember, Skill, Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import profile_update  # noqa: E402


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
    t = Tenant(name=f"p-{uuid.uuid4().hex[:8]}")
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


def _emp(db, t: Tenant, name="张三", user_id=None) -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name, designation="工程师", status="active",
        user_id=user_id,
    )
    db.add(e)
    db.commit()
    return e


def _auth(u: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"}


def _skill(db, t: Tenant, name: str) -> Skill:
    s = Skill(tenant_id=t.id, name=name, category="tech")
    db.add(s)
    db.commit()
    return s


# ============ 服务层:合并语义 + 建卡种子 ============


def test_upsert_skill_merges_no_dup(db):
    t = _tenant(db)
    e = _emp(db, t)
    s = _skill(db, t, "Python")
    r1 = profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e.id, skill_id=s.id,
        level=2, source="insight", status="suggested")
    # 同对重复:level 取高、suggested 保留
    r2 = profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e.id, skill_id=s.id,
        level=4, source="project", status="suggested")
    assert r1.id == r2.id and r2.level == 4 and r2.status == "suggested"
    # confirmed 不降级
    r3 = profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e.id, skill_id=s.id,
        level=1, source="manual", status="confirmed")
    assert r3.level == 4 and r3.status == "confirmed"
    n = db.scalars(select(EmployeeSkill).where(
        EmployeeSkill.employee_id == e.id)).all()
    assert len(n) == 1


def test_seed_card_from_candidate(db):
    t = _tenant(db)
    c = Candidate(tenant_id=t.id, name="候选人李")
    db.add(c)
    db.commit()
    s1 = _skill(db, t, "Go")
    s2 = _skill(db, t, "团队协作")
    db.add_all([
        CandidateSkill(tenant_id=t.id, candidate_id=c.id, skill_id=s1.id,
                       level=4, source="resume"),
        CandidateSkill(tenant_id=t.id, candidate_id=c.id, skill_id=s2.id,
                       level=3, source="interview"),
    ])
    e = _emp(db, t, name="李工")
    e.source_candidate_id = c.id
    db.commit()
    n = profile_update.seed_card_from_candidate(db, e)
    assert n == 2
    rows = db.scalars(select(EmployeeSkill).where(
        EmployeeSkill.employee_id == e.id)).all()
    assert len(rows) == 2
    assert all(r.status == "confirmed" for r in rows)
    assert {r.source for r in rows} == {"resume", "interview"}
    # 无 candidate 时幂等
    e2 = _emp(db, t, name="无来源")
    assert profile_update.seed_card_from_candidate(db, e2) == 0


# ============ 技能/建议 API ============


def test_skill_api_and_suggestion_flow(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    e = _emp(db, t)

    r = client.post(f"/api/employees/{e.id}/skills",
                    json={"name": "TypeScript", "level": 4},
                    headers=_auth(admin))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "confirmed" and body["level"] == 4

    # 直接造两条 suggested,走确认/驳回端点
    s_a = _skill(db, t, "Rust")
    s_b = _skill(db, t, "演讲")
    for s in (s_a, s_b):
        profile_update.upsert_employee_skill(
            db, tenant_id=t.id, employee_id=e.id, skill_id=s.id,
            level=3, source="insight", status="suggested")
    db.commit()

    r = client.post(
        f"/api/employees/{e.id}/skill-suggestions/confirm",
        json={"skill_ids": [s_a.id]}, headers=_auth(admin))
    assert r.json()["confirmed"] == 1
    r = client.post(
        f"/api/employees/{e.id}/skill-suggestions/reject",
        json={}, headers=_auth(admin))
    assert r.json()["rejected"] == 1
    left = db.scalars(select(EmployeeSkill).where(
        EmployeeSkill.employee_id == e.id,
        EmployeeSkill.status == "suggested")).all()
    assert left == []


def test_skill_write_forbidden_for_random_employee(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    e = _emp(db, t)
    # 普通员工(绑了别的员工)看不到张三 → scope 404
    u2 = _user(db, t, "employee")
    e2 = _emp(db, t, name="李四", user_id=u2.id)
    r = client.post(f"/api/employees/{e.id}/skills",
                    json={"name": "Hack", "level": 5}, headers=_auth(u2))
    assert r.status_code == 404
    # 直属上级可以
    e.reports_to_id = e2.id
    db.commit()
    r = client.post(f"/api/employees/{e.id}/skills",
                    json={"name": " mentoring ", "level": 3},
                    headers=_auth(u2))
    assert r.status_code == 201


# ============ 洞察(会议纪要上传口) ============


def test_insight_no_llm_and_mocked_llm(db, client, monkeypatch):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    e = _emp(db, t)

    r = client.post(f"/api/employees/{e.id}/insights",
                    json={"kind": "meeting", "body": "周会纪要",
                          "extract_skills": False},
                    headers=_auth(admin))
    assert r.status_code == 201
    assert r.json()["suggested"] == []

    # mock LLM 提炼 → 建议落 suggested
    import app.integrations.llm.provider as prov

    def fake_llm(kind, messages, **kw):
        return {"skills": [{"name": "系统设计", "level": 4,
                            "evidence": "主导了重构方案评审"}]}

    monkeypatch.setattr(prov, "call_llm_json", fake_llm)
    r = client.post(f"/api/employees/{e.id}/insights",
                    json={"kind": "project_review",
                          "body": "张三主导了重构方案评审,跨组对齐顺畅",
                          "extract_skills": True},
                    headers=_auth(admin))
    assert r.status_code == 201
    assert r.json()["suggested"] == ["系统设计"]
    row = db.scalars(
        select(EmployeeSkill)
        .join(Skill, Skill.id == EmployeeSkill.skill_id)
        .where(EmployeeSkill.employee_id == e.id,
               Skill.name == "系统设计")).first()
    assert row is not None and row.status == "suggested"
    assert row.evidence["kind"] == "insight"


# ============ 画像聚合 + 本人视图脱敏 ============


def test_talent_card_shape_and_self_masking(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    eu = _user(db, t, "employee")
    e = _emp(db, t, user_id=eu.id)
    s_c = _skill(db, t, "Python")
    s_s = _skill(db, t, "演讲")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e.id, skill_id=s_c.id,
        level=4, source="manual", status="confirmed")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e.id, skill_id=s_s.id,
        level=3, source="insight", status="suggested")
    db.commit()

    r = client.get(f"/api/employees/{e.id}/talent-card",
                   headers=_auth(admin))
    assert r.status_code == 200
    card = r.json()
    for k in ("employee", "skills_confirmed", "suggested_count",
              "interviews", "projects", "appraisals", "insights",
              "skills_suggested"):
        assert k in card
    assert card["suggested_count"] == 1
    assert len(card["skills_confirmed"]) == 1

    # 本人视图:suggested 不可见
    r = client.get("/api/me/talent-card", headers=_auth(eu))
    assert r.status_code == 200
    mine = r.json()
    assert mine["suggested_count"] == 0
    assert mine["skills_suggested"] == []
    assert len(mine["skills_confirmed"]) == 1


# ============ 项目 CRUD/成员 + done 触发 ============


def test_project_crud_members_and_done_trigger(db, client, monkeypatch):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    e = _emp(db, t)
    h = _auth(admin)

    r = client.post("/api/projects",
                    json={"name": "AI 平台", "required_skills":
                          [{"name": "Python", "min_level": 3}]},
                    headers=h)
    assert r.status_code == 201
    pid = r.json()["id"]

    r = client.post(f"/api/projects/{pid}/members",
                    json={"employee_id": e.id, "role": "后端",
                          "joined_via": "ai_match"}, headers=h)
    assert r.status_code == 201
    # 重复加 → 409
    r = client.post(f"/api/projects/{pid}/members",
                    json={"employee_id": e.id}, headers=h)
    assert r.status_code == 409

    r = client.get(f"/api/projects/{pid}", headers=h)
    assert r.json()["members"][0]["joined_via"] == "ai_match"

    # done → 触发建议管线(mock LLM)
    import app.integrations.llm.provider as prov
    calls = []

    def fake_llm(kind, messages, **kw):
        calls.append(kind)
        return {"skills": [{"name": "Python", "level": 4}]}

    monkeypatch.setattr(prov, "call_llm_json", fake_llm)
    r = client.put(f"/api/projects/{pid}",
                   json={"name": "AI 平台", "status": "done",
                         "required_skills": []}, headers=h)
    assert r.status_code == 200
    assert r.json()["skills_suggested"] == 1
    assert "talent.suggest" in calls

    # 重复置 done 不再触发
    calls.clear()
    r = client.put(f"/api/projects/{pid}",
                   json={"name": "AI 平台", "status": "done",
                         "required_skills": []}, headers=h)
    assert r.json()["skills_suggested"] == 0
    assert calls == []

    # 离项
    r = client.put(f"/api/projects/{pid}/members/{e.id}",
                   json={"leave": True}, headers=h)
    assert r.status_code == 200
    m = db.scalars(select(ProjectMember).where(
        ProjectMember.project_id == pid)).first()
    assert m.left_at is not None


# ============ /talent/search ============


def test_talent_search(db, client):
    t = _tenant(db)
    admin = _user(db, t, "admin")
    e1 = _emp(db, t, name="强Python")
    e2 = _emp(db, t, name="弱Python")
    e3 = _emp(db, t, name="只有建议")
    s_py = _skill(db, t, "Python")
    s_sys = _skill(db, t, "系统设计")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e1.id, skill_id=s_py.id,
        level=5, source="manual", status="confirmed")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e1.id, skill_id=s_sys.id,
        level=4, source="manual", status="confirmed")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e2.id, skill_id=s_py.id,
        level=2, source="manual", status="confirmed")
    profile_update.upsert_employee_skill(
        db, tenant_id=t.id, employee_id=e3.id, skill_id=s_py.id,
        level=5, source="insight", status="suggested")
    db.commit()

    h = _auth(admin)
    # 基础片段匹配:e1/e2 命中,e3(suggested)默认排除
    r = client.post("/api/talent/search",
                    json={"skills": [{"name": "Python"}]}, headers=h)
    assert r.status_code == 200
    items = r.json()["items"]
    names = [i["name"] for i in items]
    assert "强Python" in names and "弱Python" in names
    assert "只有建议" not in names
    # min_level=3 过滤掉弱
    r = client.post("/api/talent/search",
                    json={"skills": [{"name": "Python", "min_level": 3}]},
                    headers=h)
    names = [i["name"] for i in r.json()["items"]]
    assert names == ["强Python"]
    # include_suggested 纳入 e3
    r = client.post("/api/talent/search",
                    json={"skills": [{"name": "Python"}],
                          "include_suggested": True}, headers=h)
    names = [i["name"] for i in r.json()["items"]]
    assert "只有建议" in names
    # 多片段排序:e1 命中两段排第一
    r = client.post("/api/talent/search",
                    json={"skills": [{"name": "Python"},
                                     {"name": "系统设计"}]}, headers=h)
    items = r.json()["items"]
    assert items[0]["name"] == "强Python"
    assert items[0]["match_count"] == 2
    # 无命中片段
    r = client.post("/api/talent/search",
                    json={"skills": [{"name": "不存在的技能XYZ"}]}, headers=h)
    assert r.json()["items"] == []
    # 空 skills → 422
    r = client.post("/api/talent/search", json={"skills": []}, headers=h)
    assert r.status_code == 422
