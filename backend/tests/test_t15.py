# -*- coding: utf-8 -*-
"""T15 CJK 整词匹配 + 关键词覆盖子分 — 测试。

覆盖:
- keyword_in_text:CJK 子串/非 CJK 词边界/CJK 邻接边界/大小写
- normalize_skill_key
- compute_coverage:去重保序/hit/missing/score/空关键词
- scoring_rules:coverage 评估器方法 + keyword 整词化
- match_evaluator:结果带 coverage
- API:MatchOut 返回 coverage
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t15")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t15"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base, Candidate, Job, Resume, ResumeJobMatch, ScoringRule, Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import scoring_rules as sr  # noqa: E402
from app.services.keyword_coverage import compute_coverage  # noqa: E402
from app.services.text_match import keyword_in_text, normalize_skill_key  # noqa: E402


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
    t = Tenant(name=f"t15-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant) -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role="admin",
        status="active",
    )
    db.add(u)
    db.commit()
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _mk_resume(db, t: Tenant, parsed_text="", skills=None) -> Resume:
    cand = Candidate(tenant_id=t.id, name="c")
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id,
        candidate_id=cand.id,
        file_name="r.pdf",
        file_size=1,
        file_mime="application/pdf",
        storage_key=uuid.uuid4().hex,
        source="upload",
        parse_status="done",
        parsed_text=parsed_text,
        parsed_data={"skills": skills or []},
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _mk_job(db, t: Tenant, skills=None) -> Job:
    j = Job(
        tenant_id=t.id,
        title="后端工程师",
        description="jd",
        skills=skills or [],
        status="open",
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


# ---------------------------------------------------------------------------
# keyword_in_text
# ---------------------------------------------------------------------------


def test_cjk_substring():
    assert keyword_in_text("数据分析", "负责数据分析与报表")
    assert not keyword_in_text("机器学习", "负责数据分析与报表")


def test_ascii_word_boundary():
    assert not keyword_in_text("Go", "熟悉 Django 框架")
    assert not keyword_in_text("java", "javascript 开发经验")
    assert keyword_in_text("Go", "Python, Go, FastAPI")
    assert keyword_in_text("java", "精通 Java 编程")


def test_cjk_adjacent_boundary():
    assert keyword_in_text("Go", "会Go语言,做过Web开发")
    assert keyword_in_text("Python", "熟悉Python和数据分析")


def test_case_and_empty():
    assert keyword_in_text("SQL", "mysql sql server")
    assert not keyword_in_text("", "any")
    assert not keyword_in_text("x", "")


def test_normalize_skill_key():
    assert normalize_skill_key("  Go  Lang ") == "go lang"
    assert normalize_skill_key("PYTHON") == "python"


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


def test_coverage_hit_missing(db):
    t = _tenant(db)
    r = _mk_resume(db, t, parsed_text="精通 Python 与 PostgreSQL,会Go语言",
                   skills=["Python", "Go"])
    cov = compute_coverage(r, ["Python", "Go", "Django", "Rust"])
    assert cov["total"] == 4
    assert set(cov["hit"]) == {"Python", "Go"}
    assert set(cov["missing"]) == {"Django", "Rust"}
    assert cov["score"] == 50


def test_coverage_dedup_and_empty(db):
    t = _tenant(db)
    r = _mk_resume(db, t, parsed_text="python")
    cov = compute_coverage(r, ["Python", "python ", "PYTHON"])
    assert cov["total"] == 1
    assert compute_coverage(r, []) is None
    assert compute_coverage(r, None) is None


def test_coverage_uses_skills_and_sections(db):
    t = _tenant(db)
    cand = Candidate(tenant_id=t.id, name="c")
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id, candidate_id=cand.id, file_name="r.pdf", file_size=1,
        file_mime="application/pdf", storage_key=uuid.uuid4().hex,
        source="upload", parse_status="done", parsed_text="",
        parsed_data={"skills": ["Redis"],
                     "sections": {"skills_text": "Kubernetes, Docker"}},
    )
    db.add(r)
    db.commit()
    cov = compute_coverage(r, ["Redis", "Docker", "Go"])
    assert set(cov["hit"]) == {"Redis", "Docker"}
    assert cov["missing"] == ["Go"]


# ---------------------------------------------------------------------------
# scoring_rules: coverage 评估器 + keyword 整词化
# ---------------------------------------------------------------------------


def test_normalize_coverage_evaluator():
    ev = sr._normalize_evaluator("coverage", {"method": "coverage", "keywords": ["Go", "K8s"]})
    assert ev["method"] == "coverage"
    assert ev["keywords"] == ["Go", "K8s"]


def test_keyword_evaluator_word_boundary(db):
    t = _tenant(db)
    r = _mk_resume(db, t, parsed_text="熟悉 Django 框架开发")
    out = sr._eval_keyword(r.parsed_text, {"method": "keyword", "keywords": ["Go", "Django"], "matchMode": "any"})
    assert "Django" in out["evidence"]
    # Go 不应因子串命中 Django(整词匹配)
    assert "命中: Go" not in out["evidence"]
    assert "缺失" in out["evidence"] and "Go" in out["evidence"].split("缺失")[1]


def test_coverage_evaluator_via_rule(db):
    t = _tenant(db)
    r = _mk_resume(db, t, parsed_text="Python PostgreSQL Redis")
    j = _mk_job(db, t, skills=["Python", "Go"])
    rule = ScoringRule(
        tenant_id=t.id, name="r15",
        dimensions=[{
            "id": "d1", "key": "d1", "name": "技能覆盖", "weight": 100,
            "evaluator": {"method": "coverage"},
        }],
        version=1,
    )
    db.add(rule)
    db.commit()
    res = sr.evaluate_with_rule(db, resume=r, job=j, rule=rule)
    assert res is not None
    crit = res["criteria"][0]
    assert crit["method"] == "coverage"
    assert crit["score"] == 50  # Python 命中,Go 缺失


# ---------------------------------------------------------------------------
# API:MatchOut coverage
# ---------------------------------------------------------------------------


def test_match_api_returns_coverage(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = _mk_resume(db, t, parsed_text="Python")
    j = _mk_job(db, t, skills=["Python"])
    m = ResumeJobMatch(
        tenant_id=t.id, resume_id=r.id, job_id=j.id,
        status="done", score=80,
        coverage={"score": 100, "hit": ["Python"], "missing": [], "total": 1},
    )
    db.add(m)
    db.commit()
    resp = client.get(f"/api/matches/job/{j.id}", headers=_hdr(u))
    assert resp.status_code == 200
    item = next(x for x in resp.json() if x["id"] == m.id)
    assert item["coverage"]["score"] == 100
    assert item["coverage"]["hit"] == ["Python"]


def test_match_coverage_none_ok(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = _mk_resume(db, t)
    j = _mk_job(db, t)
    m = ResumeJobMatch(tenant_id=t.id, resume_id=r.id, job_id=j.id,
                       status="done", score=50)
    db.add(m)
    db.commit()
    resp = client.get(f"/api/matches/job/{j.id}", headers=_hdr(u))
    assert resp.status_code == 200
    item = next(x for x in resp.json() if x["id"] == m.id)
    assert item["coverage"] is None
