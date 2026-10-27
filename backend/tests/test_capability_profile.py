# -*- coding: utf-8 -*-
"""职业能力画像:多源聚合(match criteria + 评分卡 scores + 技能档案)+ 来源分解。"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-cap")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    CandidateSkill,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    Resume,
    ResumeJobMatch,
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
    t = Tenant(name=f"cap-{uuid.uuid4().hex[:8]}")
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


def _cand(db, t, name="李四") -> Candidate:
    c = Candidate(tenant_id=t.id, name=name, tags=["Python", "微服务"])
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _seed_full(db, t, cand, interviewer):
    """match 逐维 + 评分卡逐维 + 技能档案,三源齐全。"""
    job = Job(tenant_id=t.id, title="后端工程师", status="open")
    res = Resume(
        tenant_id=t.id, candidate_id=cand.id, file_name="lisi.pdf",
        file_size=100, file_mime="application/pdf", storage_key="k/1.pdf",
        source="upload", parsed_data={"skills": ["Django"]},
    )
    db.add_all([job, res])
    db.commit()
    db.refresh(job)
    db.refresh(res)
    db.add(ResumeJobMatch(
        tenant_id=t.id, resume_id=res.id, job_id=job.id, status="done", score=78,
        criteria=[
            {"key": "skills", "name": "关键技能", "weight": 40, "level": 4, "evidence": "5年Python"},
            {"key": "exp", "name": "经验年限", "weight": 30, "level": 3, "evidence": "5年"},
        ],
    ))
    sc = InterviewScorecard(
        tenant_id=t.id, candidate_id=cand.id, job_id=job.id, title="二面", status="open",
        criteria=[{"key": "c1", "name": "关键技能", "dimension": "tech", "weight": 1},
                  {"key": "c2", "name": "沟通表达", "dimension": "soft", "weight": 1}],
    )
    db.add(sc)
    db.commit()
    db.refresh(sc)
    db.add(InterviewFeedback(
        tenant_id=t.id, scorecard_id=sc.id, interviewer_user_id=interviewer.id,
        rating=4, verdict="recommend",
        scores=[{"criterion": "c1", "score": 5, "comment": "底层扎实"},
                {"criterion": "c2", "score": 3, "comment": "尚可"}],
    ))
    skill = Skill(tenant_id=t.id, name="团队协作", category="soft")
    db.add(skill)
    db.commit()
    db.refresh(skill)
    db.add(CandidateSkill(tenant_id=t.id, candidate_id=cand.id, skill_id=skill.id, level=4, source="manual"))
    db.commit()


def test_profile_three_sources_merged(client, db):
    t = _tenant(db)
    u = _user(db, t)
    cand = _cand(db, t)
    _seed_full(db, t, cand, u)

    r = client.get(f"/api/talents/{cand.id}/capability-profile", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluations_total"] == 5  # 2 match + 2 interview + 1 declared
    assert body["sources_count"] == {"match": 2, "interview": 2, "declared": 1}

    dims = {d["name"]: d for d in body["dimensions"]}
    # "关键技能" 合并了 match(level4→5.0) + interview(5.0)
    ks = dims["关键技能"]
    assert ks["count"] == 2 and ks["avg"] == 5.0
    kinds = {s["kind"] for s in ks["sources"]}
    assert kinds == {"match", "interview"}
    assert any("底层扎实" in (s["detail"] or "") for s in ks["sources"])
    # match 证据保留
    assert any("5年Python" in (s["detail"] or "") for s in ks["sources"])
    # 归一化:match level3 → 3.75;interview score3 → 3.0;declared level4 → 4.0
    assert dims["经验年限"]["avg"] == 3.75
    assert dims["沟通表达"]["avg"] == 3.0
    assert dims["团队协作"]["sources"][0]["kind"] == "declared"
    # skills 汇总:声明技能 + tags + 简历解析
    assert "团队协作" in body["skills"] and "Python" in body["skills"] and "Django" in body["skills"]
    assert "关键技能" in body["strengths"]


def test_profile_empty_candidate(client, db):
    t = _tenant(db)
    u = _user(db, t)
    cand = _cand(db, t)
    r = client.get(f"/api/talents/{cand.id}/capability-profile", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluations_total"] == 0
    assert body["dimensions"] == []
    assert body["skills"] == ["Python", "微服务"]  # 至少还有 tags


def test_profile_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    cand2 = _cand(db, t2)
    r = client.get(f"/api/talents/{cand2.id}/capability-profile", headers=_hdr(u1))
    assert r.status_code == 404


def test_profile_ignores_malformed_scores(client, db):
    """评分卡里 score 非数字 / criterion 不在 criteria 名单 — 稳健跳过/回退。"""
    t = _tenant(db)
    u = _user(db, t)
    cand = _cand(db, t)
    sc = InterviewScorecard(
        tenant_id=t.id, candidate_id=cand.id, title="一面", status="open",
        criteria=[{"key": "c1", "name": "技术深度", "dimension": "tech", "weight": 1}],
    )
    db.add(sc)
    db.commit()
    db.refresh(sc)
    db.add(InterviewFeedback(
        tenant_id=t.id, scorecard_id=sc.id, interviewer_user_id=u.id,
        rating=3, verdict="hold",
        scores=[{"criterion": "c1", "score": 4},
                {"criterion": "c1", "score": "bad"},
                {"criterion": "ghost_key", "score": 2}],
    ))
    db.commit()
    r = client.get(f"/api/talents/{cand.id}/capability-profile", headers=_hdr(u))
    assert r.status_code == 200, r.text
    dims = {d["name"]: d for d in r.json()["dimensions"]}
    assert dims["技术深度"]["count"] == 1 and dims["技术深度"]["avg"] == 4.0
    assert dims["ghost_key"]["avg"] == 2.0  # key 回退为名
