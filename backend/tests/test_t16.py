# -*- coding: utf-8 -*-
"""T16 对话式招聘 —— 人才画像 + 人才库扫描 + 面试纪要 测试。

覆盖:
- normalize_spec:字段归一/去重/min_years 容错
- _gate_lite:年限/学历门槛(结构化优先)
- _prefilter_score:coverage×keywords 加权
- sourcing.scan:排序/留痕 SourcingRun/profile use_count
- API:profiles CRUD/tenant 隔离/generate/from-job/scan
- interviews summary-view:纪要聚合
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t16")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t16"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base, Candidate, Interview, InterviewTurn, Job, Resume, SourcingRun,
    TalentProfile, Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import sourcing  # noqa: E402


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
    t = Tenant(name=f"t16-{uuid.uuid4().hex[:8]}")
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


def _mk_resume(db, t: Tenant, name: str, text: str, skills=None, sections=None) -> Resume:
    cand = Candidate(tenant_id=t.id, name=name)
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id, candidate_id=cand.id, file_name=f"{name}.pdf",
        file_size=1, file_mime="application/pdf",
        storage_key=uuid.uuid4().hex, source="upload",
        parse_status="done", parsed_text=text,
        parsed_data={"skills": skills or [], "sections": sections or {}},
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


SPEC = {"skills": ["Python", "Go"], "keywords": ["微服务"], "min_years": 0,
        "education": "", "nice_to_have": []}


# ---------------------------------------------------------------------------
# spec / gate / prefilter
# ---------------------------------------------------------------------------


def test_normalize_spec():
    s = sourcing.normalize_spec({"skills": ["Go", " go ", "Go", ""],
                                 "min_years": "abc", "education": " 本科 "})
    assert s["skills"] == ["Go", "go"] or s["skills"] == ["Go"]
    assert s["min_years"] == 0
    assert s["education"] == "本科"


def test_gate_lite_years_education(db):
    t = _tenant(db)
    secs = {"work_experiences": [{"years": 5}],
            "educations": [{"degree": "硕士"}]}
    r = _mk_resume(db, t, "a", "x", sections=secs)
    assert sourcing._gate_lite(r, {"min_years": 3, "education": "本科"}) is None
    assert sourcing._gate_lite(r, {"min_years": 6, "education": ""}) == "years"
    assert sourcing._gate_lite(r, {"min_years": 0, "education": "博士"}) == "education"


def test_prefilter_score(db):
    t = _tenant(db)
    r = _mk_resume(db, t, "b", "精通 Python 与 Go,做过微服务", skills=["Python", "Go"])
    score, cov, kw = sourcing._prefilter_score(
        r, {"skills": ["Python", "Go", "Rust"], "keywords": ["微服务"]})
    assert cov["score"] == 67  # 2/3
    assert kw == 100.0
    assert score == pytest.approx(0.6 * 67 + 0.4 * 100)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_ranking_and_run(db):
    t = _tenant(db)
    _mk_resume(db, t, "高手", "10年 Python Go 微服务", skills=["Python", "Go"])
    _mk_resume(db, t, "普通", "3年 Django 开发", skills=["Django"])
    p = TalentProfile(tenant_id=t.id, name="后端画像", spec=dict(SPEC),
                      status="active")
    db.add(p)
    db.commit()
    out = sourcing.scan(db, tenant_id=t.id, spec=p.spec, profile=p,
                        limit=5, llm_top_k=2, created_by="u1")
    assert out["scanned"] == 2
    items = out["items"]
    assert items[0]["name"] == "高手"
    assert items[0]["coverage"]["hit"] == ["Python", "Go"]
    # Go 不因 Django 误中(整词)
    assert "Go" in items[1]["coverage"]["missing"]
    db.refresh(p)
    assert p.use_count == 1 and p.last_used_at is not None
    run = db.get(SourcingRun, out["run_id"])
    assert run and run.returned == 2 and run.llm_scored == 2


def test_scan_gate_filters(db):
    t = _tenant(db)
    _mk_resume(db, t, "新手", "1年 Python", skills=["Python"])
    out = sourcing.scan(db, tenant_id=t.id,
                        spec={"skills": ["Python"], "min_years": 5},
                        limit=5, llm_top_k=0)
    assert out["prefiltered"] == 0 and out["items"] == []


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_profile_crud_and_isolation(db, client):
    t, t2 = _tenant(db), _tenant(db)
    u = _user(db, t)
    u2 = _user(db, t2)
    r = client.post("/api/talent-profiles", headers=_hdr(u),
                    json={"name": "AI画像", "spec": {"skills": ["PyTorch"]}})
    assert r.status_code in (200, 201), r.text
    pid = r.json()["id"]
    assert client.get("/api/talent-profiles?q=AI", headers=_hdr(u)).json()[0]["id"] == pid
    # 跨租户 404
    assert client.get(f"/api/talent-profiles/{pid}", headers=_hdr(u2)).status_code == 404
    r = client.put(f"/api/talent-profiles/{pid}", headers=_hdr(u),
                   json={"name": "AI画像v2", "spec": {"skills": ["PyTorch", "Go"]}})
    assert r.json()["spec"]["skills"] == ["PyTorch", "Go"]
    assert client.delete(f"/api/talent-profiles/{pid}", headers=_hdr(u)).status_code == 204


def test_profile_generate_and_from_job(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post("/api/talent-profiles/generate", headers=_hdr(u),
                    json={"prompt": "招个 AI 工程师"})
    assert r.status_code == 200, r.text
    assert "spec" in r.json() and isinstance(r.json()["spec"], dict)

    j = Job(tenant_id=t.id, title="后端工程师", skills=["Python", "Redis"],
            level="intermediate", status="open")
    db.add(j)
    db.commit()
    r = client.post("/api/talent-profiles/from-job", headers=_hdr(u),
                    json={"job_id": j.id})
    assert r.status_code == 200
    assert r.json()["spec"]["skills"] == ["Python", "Redis"]
    assert r.json()["source_job_id"] == j.id


def test_scan_api(db, client):
    t = _tenant(db)
    u = _user(db, t)
    _mk_resume(db, t, "张三", "5年 Python Go", skills=["Python", "Go"])
    p = TalentProfile(tenant_id=t.id, name="p", spec=dict(SPEC), status="active")
    db.add(p)
    db.commit()
    r = client.post("/api/sourcing/scan", headers=_hdr(u),
                    json={"profile_id": p.id, "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"] and body["items"][0]["name"] == "张三"
    assert body["profile"]["id"] == p.id
    runs = client.get("/api/sourcing/runs", headers=_hdr(u)).json()
    assert runs and runs[0]["returned"] == len(body["items"])


def test_scan_api_inline_spec(db, client):
    t = _tenant(db)
    u = _user(db, t)
    _mk_resume(db, t, "李四", "8年 Java Spring", skills=["Java"])
    r = client.post("/api/sourcing/scan", headers=_hdr(u),
                    json={"spec": {"skills": ["Java"]}, "limit": 3})
    assert r.status_code == 200
    assert r.json()["items"][0]["name"] == "李四"


# ---------------------------------------------------------------------------
# summary-view
# ---------------------------------------------------------------------------


def test_summary_view(db, client):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="王五")
    j = Job(tenant_id=t.id, title="岗位X", status="open")
    db.add_all([cand, j])
    db.commit()
    iv = Interview(
        tenant_id=t.id, job_id=j.id, candidate_id=cand.id,
        mode="remote", status="done",
        summary={"recommendation": "推荐进入下一轮",
                 "dimension_scores": {"accuracy": 80},
                 "evidence_quotes": ["q1"]},
    )
    db.add(iv)
    db.commit()
    db.add(InterviewTurn(interview_id=iv.id, idx=1, level="tech",
                         question="讲讲 Python", answer="我熟", scores={"accuracy": 80}))
    db.commit()
    r = client.get(f"/api/interviews/{iv.id}/summary-view", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ai"]["recommendation"] == "推荐进入下一轮"
    assert body["turns"][0]["question"] == "讲讲 Python"
    assert body["candidate_name"] == "王五"
