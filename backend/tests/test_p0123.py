# -*- coding: utf-8 -*-
"""P0-P3 功能测试:文档引擎 / 招聘任务链 / 知识溯源 / 决策学习。

- P0:exports API 六类生成端点 + 下载(文件魔数校验)+ 登记列表
- P1:recruit-runs 创建模板/推步/自动收口/越权隔离/参数校验
- P2:split_chunks_meta 结构感知 / 知识命中溯源字段 / company_kb_context
- P3:record_decision 埋点映射 / weight_suggestions 权重建议方向
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-p0123")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    DecisionFeedback,
    GeneratedDocument,
    Job,
    JobApplication,
    KnowledgeArticle,
    Resume,
    ResumeJobMatch,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import job_knowledge as jk  # noqa: E402
from app.services.company_knowledge import company_kb_context  # noqa: E402
from app.services.decision_learning import (  # noqa: E402
    record_decision,
    weight_suggestions,
)


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
    t = Tenant(name=f"p0123-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, *, role: str = "admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _job(db, t: Tenant) -> Job:
    j = Job(
        tenant_id=t.id,
        title=f"岗位-{uuid.uuid4().hex[:6]}",
        level="intermediate",
        status="open",
        description="负责平台后端开发。",
        skills=["Python", "FastAPI"],
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _cand(db, t: Tenant, name: str = "候选人") -> Candidate:
    c = Candidate(tenant_id=t.id, name=name)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _app_row(db, t: Tenant, job: Job, cand: Candidate) -> JobApplication:
    a = JobApplication(tenant_id=t.id, job_id=job.id, candidate_id=cand.id, stage="applied")
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _resume(db, t: Tenant, cand: Candidate) -> Resume:
    r = Resume(
        tenant_id=t.id,
        candidate_id=cand.id,
        file_name=f"{cand.name}.pdf",
        file_size=1024,
        storage_key=f"test/{uuid.uuid4().hex}",
        file_mime="application/pdf",
        source="upload",
        parse_status="done",
        parsed_text="五年 Python/FastAPI 后端经验",
        parsed_data={"skills": [{"name": "Python"}, {"name": "FastAPI"}]},
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _match(db, t: Tenant, resume: Resume, job: Job, score: int = 80) -> ResumeJobMatch:
    m = ResumeJobMatch(
        tenant_id=t.id,
        resume_id=resume.id,
        job_id=job.id,
        status="done",
        score=score,
        criteria=[
            {"key": "tech", "name": "技术深度", "weight": 0.4, "level": 3, "evidence": "5年 Python"},
            {"key": "comm", "name": "沟通协作", "weight": 0.2, "level": 2, "evidence": "跨团队项目"},
            {"key": "proj", "name": "项目经验", "weight": 0.4, "level": 4, "evidence": "主导重构"},
        ],
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


# ======================= P0:文档引擎 =======================


def test_p0_candidates_xlsx_export_and_download(db, client):
    t = _tenant(db)
    u = _user(db, t)
    _cand(db, t, "导出候选人A")
    r = client.post("/api/exports/candidates.xlsx", headers=_hdr(u), json={})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["file_name"].endswith(".xlsx")
    d = client.get(f"/api/exports/{doc['id']}/download", headers=_hdr(u))
    assert d.status_code == 200
    assert d.content[:2] == b"PK"  # xlsx = zip
    # 登记列表可见
    lst = client.get("/api/exports/", headers=_hdr(u))
    assert any(x["id"] == doc["id"] for x in lst.json())


def test_p0_payroll_and_pipeline_xlsx(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    for path, body in [
        ("/api/exports/payroll.xlsx", {"period": "2026-10"}),
        ("/api/exports/pipeline.xlsx", {"job_id": j.id}),
    ]:
        r = client.post(path, headers=_hdr(u), json=body)
        assert r.status_code == 200, (path, r.text)
        d = client.get(f"/api/exports/{r.json()['id']}/download", headers=_hdr(u))
        assert d.status_code == 200 and d.content[:2] == b"PK"


def test_p0_docx_pptx_exports(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    cand = _cand(db, t, "报告候选人")
    res = _resume(db, t, cand)
    _match(db, t, res, j)
    r = client.post(f"/api/exports/candidate-report/{cand.id}", headers=_hdr(u), json={})
    assert r.status_code == 200, r.text
    d = client.get(f"/api/exports/{r.json()['id']}/download", headers=_hdr(u))
    assert d.content[:2] == b"PK"  # docx = zip
    r2 = client.post("/api/exports/recruiting-pack", headers=_hdr(u), json={})
    assert r2.status_code == 200, r2.text
    d2 = client.get(f"/api/exports/{r2.json()['id']}/download", headers=_hdr(u))
    assert d2.content[:2] == b"PK"  # pptx = zip


def test_p0_export_tenant_isolation(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u1, u2 = _user(db, t1), _user(db, t2)
    r = client.post("/api/exports/candidates.xlsx", headers=_hdr(u1), json={})
    doc_id = r.json()["id"]
    d = client.get(f"/api/exports/{doc_id}/download", headers=_hdr(u2))
    assert d.status_code == 404  # 跨租户不可下载


def test_p0_generated_document_row(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post("/api/exports/candidates.xlsx", headers=_hdr(u), json={})
    row = db.get(GeneratedDocument, r.json()["id"])
    assert row is not None and row.tenant_id == t.id and row.kind == "candidate_list"


# ======================= P1:招聘任务链 =======================


def test_p1_run_lifecycle(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post("/api/recruit-runs/", headers=_hdr(u), json={"title": "全链路招聘"})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "running"
    assert len(run["steps"]) == 7
    assert [s["key"] for s in run["steps"]] == [
        "jd", "parse", "source", "score", "shortlist", "interview", "report",
    ]
    rid = run["id"]
    # 推一步 running → done
    s1 = client.patch(f"/api/recruit-runs/{rid}/steps/jd", headers=_hdr(u), json={"status": "running"})
    assert s1.json()["steps"][0]["status"] == "running"
    s2 = client.patch(
        f"/api/recruit-runs/{rid}/steps/jd",
        headers=_hdr(u),
        json={"status": "done", "detail": {"note": "JD 已建"}},
    )
    st = s2.json()["steps"][0]
    assert st["status"] == "done" and st["detail"]["note"] == "JD 已建" and st["finished_at"]
    # 全到终态 → run 自动 done
    for k in ["parse", "source", "score", "shortlist", "interview", "report"]:
        resp = client.patch(f"/api/recruit-runs/{rid}/steps/{k}", headers=_hdr(u), json={"status": "done"})
    assert resp.json()["status"] == "done" and resp.json()["finished_at"]


def test_p1_run_failed_when_any_step_failed(db, client):
    t = _tenant(db)
    u = _user(db, t)
    rid = client.post("/api/recruit-runs/", headers=_hdr(u), json={}).json()["id"]
    for k in ["jd", "parse", "source", "score", "shortlist", "interview"]:
        client.patch(f"/api/recruit-runs/{rid}/steps/{k}", headers=_hdr(u), json={"status": "done"})
    r = client.patch(f"/api/recruit-runs/{rid}/steps/report", headers=_hdr(u), json={"status": "failed"})
    assert r.json()["status"] == "failed"


def test_p1_run_validation_and_isolation(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u1, u2 = _user(db, t1), _user(db, t2)
    # 未知 kind
    r = client.post("/api/recruit-runs/", headers=_hdr(u1), json={"kind": "nope"})
    assert r.status_code == 400
    rid = client.post("/api/recruit-runs/", headers=_hdr(u1), json={}).json()["id"]
    # 非法 step key / 非法 status
    assert client.patch(f"/api/recruit-runs/{rid}/steps/xx", headers=_hdr(u1), json={"status": "done"}).status_code == 404
    assert client.patch(f"/api/recruit-runs/{rid}/steps/jd", headers=_hdr(u1), json={"status": "bogus"}).status_code == 400
    # 跨用户不可见
    assert client.get(f"/api/recruit-runs/{rid}", headers=_hdr(u2)).status_code == 404
    assert client.get("/api/recruit-runs/", headers=_hdr(u2)).json() == []


# ======================= P2:知识溯源 =======================


def test_p2_split_chunks_meta_sections():
    text = "# 公司介绍\n我们是一家 AI 公司。\n\n# 岗位要求\n会 Python 和 FastAPI。\n\n三年以上经验。"
    metas = jk.split_chunks_meta(text)
    assert len(metas) >= 2
    # 标题开新块:两个小节各成块,section 正确归属
    secs = [c["section"] for c in metas]
    assert "公司介绍" in secs and "岗位要求" in secs
    req_chunk = next(c for c in metas if "会 Python" in c["content"])
    assert req_chunk["section"] == "岗位要求"


def test_p2_search_hit_provenance(db, client):
    t = _tenant(db)
    u = _user(db, t)
    a = KnowledgeArticle(
        tenant_id=t.id,
        title="差旅报销制度",
        category="policy",
        content="# 总则\n报销需附发票。\n\n# 发票要求\n发票抬头必须为公司全称,缺一不可。",
        tags=[],
        status="published",
    )
    db.add(a)
    db.commit()
    r = client.post("/api/knowledge/search", headers=_hdr(u), json={"q": "发票抬头", "limit": 3})
    assert r.status_code == 200
    hits = r.json()
    assert hits
    hit = hits[0]
    assert hit["match_field"] == "content"
    assert hit["paragraph"] is not None and hit["paragraph"] >= 2
    assert hit["section"] == "发票要求"
    assert hit["char_offset"] is not None


def test_p2_company_kb_context(db):
    t = _tenant(db)
    db.add(
        KnowledgeArticle(
            tenant_id=t.id,
            title="前端岗位要求",
            category="other",
            content="React/TypeScript 必须,熟悉设计系统者优先;职级 P5 起。",
            tags=[],
            status="published",
        )
    )
    db.commit()
    ctx = company_kb_context(db, t.id, "前端 React TypeScript 要求")
    assert "《前端岗位要求》" in ctx and "React" in ctx
    # 无命中 → 空串(调用方跳过注入)
    assert company_kb_context(db, t.id, "zzz不存在的词zzz") == ""


# ======================= P3:决策学习 =======================


def test_p3_record_decision_mapping(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    cand = _cand(db, t)
    res = _resume(db, t, cand)
    _match(db, t, res, j, score=88)
    app_row = _app_row(db, t, j, cand)

    fb = record_decision(
        db, tenant_id=t.id, application=app_row,
        from_stage="applied", to_stage="screening", actor_id=u.id,
    )
    assert fb and fb.decision == "advanced"
    assert fb.match_score == 88
    assert fb.criteria_snapshot and fb.criteria_snapshot[0]["key"] == "tech"

    fb2 = record_decision(
        db, tenant_id=t.id, application=app_row,
        from_stage="screening", to_stage="rejected", actor_id=u.id,
    )
    assert fb2 and fb2.decision == "rejected"

    # applied/withdrawn 非决策 → None
    assert (
        record_decision(
            db, tenant_id=t.id, application=app_row,
            from_stage="applied", to_stage="withdrawn", actor_id=u.id,
        )
        is None
    )


def test_p3_transition_endpoint_creates_feedback(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    cand = _cand(db, t)
    app_row = _app_row(db, t, j, cand)
    r = client.post(
        f"/api/applications/{app_row.id}/transition",
        headers=_hdr(u),
        json={"to_stage": "screening"},
    )
    assert r.status_code == 200, r.text
    fb = db.scalar(
        __import__("sqlalchemy").select(DecisionFeedback).where(
            DecisionFeedback.application_id == app_row.id
        )
    )
    assert fb is not None and fb.decision == "advanced" and fb.to_stage == "screening"


def test_p3_weight_suggestions_direction(db):
    t = _tenant(db)
    j = _job(db, t)
    cand = _cand(db, t)
    app_row = _app_row(db, t, j, cand)
    # 造两类决策:tech 高分的被推进,tech 低分的被淘汰 → tech 应提权
    for _ in range(4):
        db.add(
            DecisionFeedback(
                tenant_id=t.id, job_id=j.id, candidate_id=cand.id,
                application_id=app_row.id, decision="advanced",
                to_stage="interview", match_score=90,
                criteria_snapshot=[
                    {"key": "tech", "name": "技术深度", "weight": 0.2, "level": 4},
                    {"key": "comm", "name": "沟通协作", "weight": 0.2, "level": 3},
                ],
            )
        )
    for _ in range(4):
        db.add(
            DecisionFeedback(
                tenant_id=t.id, job_id=j.id, candidate_id=cand.id,
                application_id=app_row.id, decision="rejected",
                to_stage="rejected", match_score=40,
                criteria_snapshot=[
                    {"key": "tech", "name": "技术深度", "weight": 0.2, "level": 1},
                    {"key": "comm", "name": "沟通协作", "weight": 0.2, "level": 3},
                ],
            )
        )
    db.commit()
    out = weight_suggestions(db, t.id, job_id=j.id, days=90)
    assert out["decisions_total"] == 8 and out["positive"] == 4 and out["negative"] == 4
    dims = {d["key"]: d for d in out["dimensions"]}
    tech = dims["tech"]
    assert tech["pos_avg_level"] == 4.0 and tech["neg_avg_level"] == 1.0
    assert tech["gap"] == 3.0
    assert tech["suggested_weight"] > tech["current_weight"]  # 区分度大 → 提权
    comm = dims["comm"]
    assert comm["gap"] == 0.0  # 无区分度 → 不提权


def test_p3_learned_endpoint(db, client):
    from sqlalchemy import select
    from app.domain.models import ScoringRule

    t = _tenant(db)
    u = _user(db, t)
    rule = ScoringRule(
        tenant_id=t.id,
        name="通用规则",
        dimensions=[
            {"id": "d1", "name": "技术深度", "weight": 50,
             "evaluator": {"method": "ai"}},
        ],
        version=1,
        created_by=u.id,
    )
    db.add(rule)
    db.commit()
    r = client.get(f"/api/scoring-rules/{rule.id}/learned?days=90", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rule"]["id"] == rule.id
    assert "decisions_total" in body and "dimensions" in body
    # 其他租户访问 → 404
    t2 = _tenant(db)
    u2 = _user(db, t2)
    assert client.get(f"/api/scoring-rules/{rule.id}/learned", headers=_hdr(u2)).status_code == 404
