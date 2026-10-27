# -*- coding: utf-8 -*-
"""技能图谱 + 内部人才匹配:

- graph:parent/aliases 读写、别名归并、环检测、graph 端点
- internal-match:岗位→员工排行(覆盖率×权重+等级+图谱兑现)、
  员工→岗位反向、租户隔离、权限
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-im")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Employee,
    EmployeeSkill,
    Job,
    JobSkill,
    Skill,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _schema():
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
    t = Tenant(name=f"im-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t, role="admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@x.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u):
    return {"Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"}


def _skill(db, t, name, **kw) -> Skill:
    s = Skill(tenant_id=t.id, name=name, category=kw.pop("category", "tech"), **kw)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _emp(db, t, name, **kw) -> Employee:
    e = Employee(
        tenant_id=t.id, employee_no=f"E{uuid.uuid4().hex[:6].upper()}",
        name=name, status="active", designation=kw.pop("designation", None), **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _emp_skill(db, t, emp, skill, level=4, status="confirmed", source="manual"):
    es = EmployeeSkill(
        tenant_id=t.id, employee_id=emp.id, skill_id=skill.id,
        level=level, status=status, source=source,
    )
    db.add(es)
    db.commit()
    return es


# ---------------- 技能图谱 ----------------


def test_skill_graph_fields_and_alias_merge(client, db):
    t = _tenant(db)
    u = _user(db, t)
    # 建父技能 Python,子技能 Django(parent_id + aliases)
    p = client.post("/api/skills", json={"name": "Python", "category": "tech"}, headers=_hdr(u))
    assert p.status_code == 201, p.text
    pid = p.json()["id"]
    c = client.post("/api/skills", json={
        "name": "Django", "category": "tech", "parent_id": pid,
        "aliases": ["django框架", "DjangoFW"],
    }, headers=_hdr(u))
    assert c.status_code == 201, c.text
    assert c.json()["parent_id"] == pid
    assert c.json()["aliases"] == ["django框架", "DjangoFW"]

    # 别名归并:候选人技能按别名 "django框架" 写入 → 归到 Django 词条
    cand = Candidate(tenant_id=t.id, name="张三")
    db.add(cand)
    db.commit()
    r = client.put(f"/api/candidates/{cand.id}/skills", json={
        "items": [{"name": "django框架", "level": 5}],
    }, headers=_hdr(u))
    assert r.status_code == 200, r.text
    assert r.json()[0]["name"] == "Django"  # 归并到 canonical,不产生新词条
    assert db.scalars(__import__("sqlalchemy").select(Skill).where(Skill.tenant_id == t.id)).all().__len__() == 2

    # graph 端点:节点+边+引用计数
    g = client.get("/api/skills/graph", headers=_hdr(u))
    assert g.status_code == 200
    body = g.json()
    assert len(body["edges"]) == 1 and body["edges"][0]["to"] == pid
    django = next(n for n in body["nodes"] if n["name"] == "Django")
    assert django["candidate_refs"] == 1 and django["aliases"] == ["django框架", "DjangoFW"]


def test_skill_graph_cycle_rejected(client, db):
    t = _tenant(db)
    u = _user(db, t)
    a = client.post("/api/skills", json={"name": "A技能"}, headers=_hdr(u)).json()["id"]
    b = client.post("/api/skills", json={"name": "B技能", "parent_id": a}, headers=_hdr(u)).json()["id"]
    # A.parent = B → 环
    r = client.put(f"/api/skills/{a}", json={"name": "A技能", "category": "other", "parent_id": b}, headers=_hdr(u))
    assert r.status_code == 400 and "循环" in r.json()["detail"]
    # 自引用
    r2 = client.put(f"/api/skills/{a}", json={"name": "A技能", "category": "other", "parent_id": a}, headers=_hdr(u))
    assert r2.status_code == 400


# ---------------- 内部匹配 ----------------


def _setup_match(db, t):
    """岗位要 Python(must)+ 系统设计(nice);三个员工:
    e1 直命中双技能 → 满分区;e2 只有 Django(子技能兑现 Python) → 中分;
    e3 无技能 → 不出榜。"""
    py = _skill(db, t, "Python")
    django = _skill(db, t, "Django", parent_id=py.id)
    sysd = _skill(db, t, "系统设计")
    job = Job(tenant_id=t.id, title="后端工程师", status="open")
    db.add(job)
    db.commit()
    db.refresh(job)
    db.add_all([
        JobSkill(tenant_id=t.id, job_id=job.id, skill_id=py.id, importance="must"),
        JobSkill(tenant_id=t.id, job_id=job.id, skill_id=sysd.id, importance="nice"),
    ])
    e1 = _emp(db, t, "直命中", designation="高工")
    e2 = _emp(db, t, "子技能人")
    e3 = _emp(db, t, "无技能人")
    db.commit()
    _emp_skill(db, t, e1, py, level=5)
    _emp_skill(db, t, e1, sysd, level=4)
    _emp_skill(db, t, e2, django, level=5)
    return job, e1, e2, e3, py, django, sysd


def test_internal_match_job_to_employees(client, db):
    t = _tenant(db)
    u = _user(db, t)
    job, e1, e2, e3, py, django, sysd = _setup_match(db, t)

    r = client.get(f"/api/internal-match/jobs/{job.id}", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    items = {i["employee"]["name"]: i for i in body["items"]}
    assert "直命中" in items and "子技能人" in items and "无技能人" not in items
    top = items["直命中"]
    assert top["score"] > items["子技能人"]["score"]
    assert top["must_hit"] == 1 and top["must_total"] == 1
    assert len(top["matched"]) == 2 and len(top["missing"]) == 0
    # e2:子技能兑现,via=child,衰减等级 5×0.8=4.0
    e2m = items["子技能人"]["matched"][0]
    assert e2m["via"] == "child" and e2m["level"] == 4.0 and e2m["via_name"] == "Django"
    assert any(m["name"] == "系统设计" for m in items["子技能人"]["missing"])


def test_internal_match_employee_to_jobs(client, db):
    t = _tenant(db)
    u = _user(db, t)
    job, e1, e2, e3, *_ = _setup_match(db, t)
    r = client.get(f"/api/internal-match/employees/{e2.id}", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills_count"] == 1
    assert len(body["items"]) == 1 and body["items"][0]["job"]["id"] == job.id


def test_internal_match_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    job2, *_ = _setup_match(db, t2)
    r = client.get(f"/api/internal-match/jobs/{job2.id}", headers=_hdr(u1))
    assert r.status_code == 404


def test_internal_match_empty_requirements(client, db):
    t = _tenant(db)
    u = _user(db, t)
    job = Job(tenant_id=t.id, title="无要求岗", status="open")
    db.add(job)
    db.commit()
    db.refresh(job)
    _emp(db, t, "某员工")
    r = client.get(f"/api/internal-match/jobs/{job.id}", headers=_hdr(u))
    assert r.status_code == 200
    assert r.json()["items"] == [] and r.json()["requirements"] == []


def test_internal_match_suggested_excluded(client, db):
    """suggested(未确认)技能不计入匹配 — 校准机制。"""
    t = _tenant(db)
    u = _user(db, t)
    py = _skill(db, t, "Python")
    job = Job(tenant_id=t.id, title="岗", status="open")
    db.add(job)
    db.commit()
    db.refresh(job)
    db.add(JobSkill(tenant_id=t.id, job_id=job.id, skill_id=py.id, importance="must"))
    e = _emp(db, t, "未确认者")
    db.commit()
    _emp_skill(db, t, e, py, level=5, status="suggested")
    r = client.get(f"/api/internal-match/jobs/{job.id}", headers=_hdr(u))
    assert r.json()["items"] == []
