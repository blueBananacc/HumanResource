# -*- coding: utf-8 -*-
"""Moka 对标功能测试:JD 审查/知识缺口/动态画像/面试副驾/人才地图。

LLM 调用统一 monkeypatch 掉(测试环境无 provider),
副驾走降级路径,审查走 mock 返回 JSON。
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-moka")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    DecisionFeedback,
    InterviewScorecard,
    Job,
    JobApplication,
    ResumeJobMatch,
    ScoringRule,
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
    t = Tenant(name=f"mk-{uuid.uuid4().hex[:8]}")
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


def _job(db, t, **kw) -> Job:
    j = Job(tenant_id=t.id, title=kw.pop("title", "后端工程师"), status="open", **kw)
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


# ---------------- 1. JD 审查 ----------------


def test_jd_review_mocked_llm(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t, description="限男性,35岁以下,有大厂背景优先。能适应高强度加班。")

    import app.api.ai as ai_api

    fake = (
        '{"score": 42, "issues": ['
        '{"dimension": "DEI合规", "severity": "high", "quote": "限男性", "suggestion": "删除性别限制"},'
        '{"dimension": "措辞精准", "severity": "medium", "quote": "大厂背景", "suggestion": "改为知名互联网公司经验"}'
        '], "summary": "存在歧视性表述和模糊措辞"}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)

    r = client.post("/api/ai/jd-review", json={"job_id": j.id}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["score"] == 42
    assert len(body["issues"]) == 2
    assert body["issues"][0]["dimension"] == "DEI合规"
    assert "歧视" in body["summary"]


def test_jd_review_empty_jd_400(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post("/api/ai/jd-review", json={}, headers=_hdr(u))
    assert r.status_code == 400


def test_jd_review_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    j2 = _job(db, t2, description="x")
    r = client.post("/api/ai/jd-review", json={"job_id": j2.id}, headers=_hdr(u1))
    assert r.status_code == 404


# ---------------- 2. 知识缺口 ----------------


def test_knowledge_gaps_aggregation(client, db):
    t = _tenant(db)
    u = _user(db, t)
    # 搜两次不存在的词 → 缺口
    client.post("/api/knowledge/search", json={"q": "量子波动速读", "limit": 3}, headers=_hdr(u))
    client.post("/api/knowledge/search", json={"q": "量子波动速读", "limit": 3}, headers=_hdr(u))
    client.post("/api/knowledge/search", json={"q": "火星殖民福利", "limit": 3}, headers=_hdr(u))

    r = client.get("/api/knowledge/gaps", headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["zero_hit_total"] >= 3
    gap_qs = {g["query"]: g["count"] for g in body["gaps"]}
    assert gap_qs.get("量子波动速读") == 2
    assert gap_qs.get("火星殖民福利") == 1


def test_knowledge_gaps_via_list_search(client, db):
    """UI 实际走的是 GET /articles?q= — 列表零命中同样计入缺口。"""
    t = _tenant(db)
    u = _user(db, t)
    r = client.get(
        "/api/knowledge/articles", params={"q": "反物质年终奖方案"}, headers=_hdr(u)
    )
    assert r.status_code == 200
    r2 = client.get("/api/knowledge/gaps", headers=_hdr(u))
    gap_qs = {g["query"] for g in r2.json()["gaps"]}
    assert "反物质年终奖方案" in gap_qs


# ---------------- 3. 动态职位画像 ----------------


def test_job_ai_profile(client, db):
    t = _tenant(db)
    u = _user(db, t)
    rule = ScoringRule(
        tenant_id=t.id,
        name="技术岗规则",
        dimensions=[
            {"key": "tech", "name": "技术深度", "weight": 40},
            {"key": "comm", "name": "沟通协作", "weight": 30},
        ],
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    j = _job(db, t, scoring_rule_id=rule.id)
    cand = Candidate(tenant_id=t.id, name="王五")
    db.add(cand)
    db.commit()
    db.refresh(cand)
    app_ = JobApplication(tenant_id=t.id, candidate_id=cand.id, job_id=j.id, stage="hired")
    db.add(app_)
    db.commit()
    db.refresh(app_)
    db.add(
        DecisionFeedback(
            tenant_id=t.id, job_id=j.id, candidate_id=cand.id, application_id=app_.id,
            decision="hired", to_stage="hired",
            criteria_snapshot=[{"key": "tech", "name": "技术深度", "level": 4, "weight": 40}],
        )
    )
    from app.domain.models import Resume

    res = Resume(
        tenant_id=t.id, candidate_id=cand.id, file_name="c.pdf",
        file_size=1024, file_mime="application/pdf",
        storage_key=f"r-{uuid.uuid4().hex[:8]}", source="manual",
        parse_status="done",
    )
    db.add(res)
    db.commit()
    db.refresh(res)
    db.add(
        ResumeJobMatch(
            tenant_id=t.id, job_id=j.id, resume_id=res.id,
            score=88, status="done",
        )
    )
    db.commit()

    r = client.get(f"/api/jobs/{j.id}/ai-profile", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rule"]["name"] == "技术岗规则"
    assert body["learned"]["decisions_total"] == 1
    assert body["match_stats"]["evaluated"] == 1
    assert body["match_stats"]["top_score"] == 88
    assert body["one_liner"]


# ---------------- 5. 面试副驾 ----------------


def test_interview_copilot_degraded_fallback(client, db, monkeypatch):
    """LLM 不可用 → 降级路径仍返回结构化追问(不瘫)。"""
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="赵六")
    db.add(cand)
    db.commit()
    db.refresh(cand)
    sc = InterviewScorecard(
        tenant_id=t.id, candidate_id=cand.id, title="一面",
        criteria=[
            {"key": "c1", "name": "技术深度", "dimension": "tech", "weight": 50},
            {"key": "c2", "name": "团队协作", "dimension": "soft", "weight": 50},
        ],
        status="open",
    )
    db.add(sc)
    db.commit()
    db.refresh(sc)

    import app.api.ai as ai_api
    from app.integrations.llm.provider import LLMError

    monkeypatch.setattr(
        ai_api, "call_llm",
        lambda *a, **kw: (_ for _ in ()).throw(LLMError("no provider")),
    )
    r = client.post(
        "/api/ai/interview-copilot",
        json={
            "scorecard_id": sc.id,
            "answer_text": "我主导了一个高并发系统的重构,提升了吞吐量。",
            "asked_dimensions": ["技术深度"],
        },
        headers=_hdr(u),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is True
    assert "技术深度" in body["covered"]
    assert "团队协作" in body["missing"]
    assert len(body["followups"]) >= 1


def test_interview_copilot_llm_path(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    import app.api.ai as ai_api

    fake = (
        '{"missing": ["团队协作"], '
        '"followups": [{"dimension": "团队协作", "question": "那次重构中你如何协调上下游?", "intent": "验证协作深度"}], '
        '"note": "回答偏技术细节,建议补协作维度"}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)
    r = client.post(
        "/api/ai/interview-copilot",
        json={"answer_text": "我负责写代码和部署。"},
        headers=_hdr(u),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["followups"][0]["dimension"] == "团队协作"
    assert body["note"]


# ---------------- 8. 人才地图 ----------------


def test_talent_map(client, db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    c1 = Candidate(tenant_id=t.id, name="甲", tags=["python", "react"])
    c2 = Candidate(tenant_id=t.id, name="乙", tags=["python"])
    db.add_all([c1, c2])
    db.commit()
    db.refresh(c1)
    db.refresh(c2)
    db.add(JobApplication(tenant_id=t.id, candidate_id=c1.id, job_id=j.id, stage="interview"))
    db.add(JobApplication(tenant_id=t.id, candidate_id=c2.id, job_id=j.id, stage="applied"))
    db.commit()

    r = client.get("/api/reports/talent-map", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["candidates_total"] == 2
    stages = {s["stage"]: s["count"] for s in body["by_stage"]}
    assert stages.get("interview") == 1 and stages.get("applied") == 1
    tags = {t_["tag"]: t_["count"] for t_ in body["top_tags"]}
    assert tags.get("python") == 2
    assert body["by_job"][0]["title"] == "后端工程师"
