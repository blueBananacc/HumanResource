# -*- coding: utf-8 -*-
"""T17 组织面试质量报告 测试。

覆盖:
- build_quality_report:KPI/维度覆盖/missed/通过率/时长/环比
- 维度名归一化合并、self_test 排除、status!=done 排除
- 独立评分卡反馈并入事件、hold 计入 decided 不计通过
- report-defs 注册可见 + run 端点 + 租户隔离 + 空窗口
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t17")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t17"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base, Candidate, Interview, InterviewFeedback, InterviewScorecard, Job,
    Tenant, User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.interview_quality import (  # noqa: E402
    _rec_positive, build_quality_report,
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
    t = Tenant(name=f"t17-{uuid.uuid4().hex[:8]}")
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


NOW = datetime.now()


def _mk_interview(db, t, cand, job, *, dims=None, rec=None, status="done",
                  mode="remote", mins=40, days_ago=1) -> Interview:
    fin = NOW - timedelta(days=days_ago)
    iv = Interview(
        tenant_id=t.id, job_id=job.id, candidate_id=cand.id,
        mode=mode, status=status,
        started_at=fin - timedelta(minutes=mins), finished_at=fin,
        summary={"recommendation": rec, "dimension_scores": dims or {}},
    )
    db.add(iv)
    db.commit()
    return iv


def _mk_card(db, t, cand, user, *, criteria, verdicts, interview_id=None,
             job_id=None) -> InterviewScorecard:
    card = InterviewScorecard(
        tenant_id=t.id, candidate_id=cand.id, job_id=job_id,
        interview_id=interview_id, title="评分卡",
        criteria=criteria, interviewer_ids=[user.id], status="closed",
    )
    db.add(card)
    db.commit()
    for i, (v, scores) in enumerate(verdicts):
        db.add(InterviewFeedback(
            tenant_id=t.id, scorecard_id=card.id, interviewer_user_id=user.id,
            rating=4, verdict=v, scores=scores,
            submitted_at=NOW - timedelta(days=1),
        ))
    db.commit()
    return card


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_rec_positive():
    assert _rec_positive("推荐进入下一轮") is True
    assert _rec_positive("hire") is True
    assert _rec_positive("不推荐") is False
    assert _rec_positive("保留意见") is None
    assert _rec_positive(None) is None


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_quality_report_basic(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c1")
    job = Job(tenant_id=t.id, title="后端", status="open")
    db.add_all([cand, job])
    db.commit()
    # 3 场 done 面试:结构化思维×3 系统设计×2 业务敏感度×1;2 推荐 1 不推荐
    _mk_interview(db, t, cand, job, dims={"结构化思维": 8, "系统设计": 7},
                  rec="推荐进入下一轮", mins=40)
    _mk_interview(db, t, cand, job, dims={"结构化思维": 9, "业务敏感度": 5},
                  rec="推荐", mins=60)
    _mk_interview(db, t, cand, job, dims={"结构化思维": 6, "系统设计": 4},
                  rec="不推荐", mins=40)
    r = build_quality_report(db, u, days=30)
    k = r["kpis"]
    assert k["interviews_total"] == 3
    assert k["assessed_events"] == 3
    assert k["pass_rate"] == 67  # 2/3
    assert k["avg_duration_min"] in (46, 47)  # (40+60+40)/3
    dims = {d["name"]: d for d in r["dimensions"]}
    assert dims["结构化思维"]["coverage"] == 100
    assert dims["系统设计"]["coverage"] == 67
    assert dims["业务敏感度"]["coverage"] == 33
    assert [m["name"] for m in r["missed"]] == ["业务敏感度"]
    # Σhits=6, universe=3, events=3 → 6/9
    assert k["dimension_coverage"] == 67
    assert r["by_job"][0]["interviews"] == 3


def test_quality_report_empty_and_exclusions(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c2")
    job = Job(tenant_id=t.id, title="j", status="open")
    db.add_all([cand, job])
    db.commit()
    # self_test + in_progress 都不算;60 天前的也不在 30 天窗
    _mk_interview(db, t, cand, job, dims={"x": 1}, mode="self_test")
    _mk_interview(db, t, cand, job, dims={"x": 1}, status="in_progress")
    _mk_interview(db, t, cand, job, dims={"x": 1}, days_ago=60)
    r = build_quality_report(db, u, days=30)
    assert r["kpis"]["interviews_total"] == 0
    assert r["kpis"]["assessed_events"] == 0
    assert r["dimensions"] == []


def test_scorecard_feedback_event(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c3")
    job = Job(tenant_id=t.id, title="j2", status="open")
    db.add_all([cand, job])
    db.commit()
    # 独立评分卡(无面试):criteria 2 维,反馈全打,verdict recommend
    _mk_card(db, t, cand, u,
             criteria=[{"key": "k1", "name": "协作能力", "dimension": "soft"},
                       {"key": "k2", "name": "自驱力", "dimension": "general"}],
             verdicts=[("recommend", [{"criterion": "k1", "score": 4},
                                      {"criterion": "k2", "score": 5}])])
    r = build_quality_report(db, u, days=30)
    k = r["kpis"]
    assert k["assessed_events"] == 1  # 独立评分卡算事件,面试量仍 0
    assert k["interviews_total"] == 0
    assert k["pass_rate"] == 100
    dims = {d["name"]: d for d in r["dimensions"]}
    assert dims["协作能力"]["coverage"] == 100
    assert dims["自驱力"]["avg_score"] == 10.0  # 5×2 对齐 10 分制


def test_dim_name_normalize_merge(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c4")
    job = Job(tenant_id=t.id, title="j3", status="open")
    db.add_all([cand, job])
    db.commit()
    # "系统设计" 与 " 系统设计 " 归一并
    _mk_interview(db, t, cand, job, dims={"系统设计": 8}, rec="推荐")
    _mk_interview(db, t, cand, job, dims={" 系统设计 ": 6}, rec="推荐")
    r = build_quality_report(db, u, days=30)
    sysdims = [d for d in r["dimensions"] if "系统设计" in d["name"]]
    assert len(sysdims) == 1 and sysdims[0]["coverage"] == 100
    assert sysdims[0]["scored_in"] == 2


def test_hold_verdict_decided_not_passed(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c5")
    db.add(cand)
    db.commit()
    _mk_card(db, t, cand, u,
             criteria=[{"key": "k1", "name": "X", "dimension": "general"}],
             verdicts=[("hold", [{"criterion": "k1", "score": 3}])])
    r = build_quality_report(db, u, days=30)
    assert r["kpis"]["decided"] == 1 and r["kpis"]["pass_rate"] == 0


# ---------------------------------------------------------------------------
# report-defs API
# ---------------------------------------------------------------------------


def test_report_defs_run_and_isolation(db, client):
    t, t2 = _tenant(db), _tenant(db)
    u, u2 = _user(db, t), _user(db, t2)
    cand = Candidate(tenant_id=t.id, name="c6")
    job = Job(tenant_id=t.id, title="j4", status="open")
    db.add_all([cand, job])
    db.commit()
    _mk_interview(db, t, cand, job, dims={"Python": 8}, rec="推荐")

    defs = client.get("/api/report-defs", headers=_hdr(u)).json()
    assert any(d["kind"] == "interview_quality" for d in defs)

    r = client.post("/api/report-defs/interview_quality/run", headers=_hdr(u),
                    json={"params": {"days": 30}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "interview_quality"
    assert body["data"]["kpis"]["interviews_total"] == 1
    assert body["data"]["dimensions"][0]["name"] == "Python"

    # 跨租户看不到对方数据(事件数 0)
    r2 = client.post("/api/report-defs/interview_quality/run", headers=_hdr(u2),
                     json={"params": {}})
    assert r2.status_code == 200
    assert r2.json()["data"]["kpis"]["assessed_events"] == 0


def test_delta_prev_window(db):
    t = _tenant(db)
    u = _user(db, t)
    cand = Candidate(tenant_id=t.id, name="c7")
    job = Job(tenant_id=t.id, title="j5", status="open")
    db.add_all([cand, job])
    db.commit()
    _mk_interview(db, t, cand, job, dims={"x": 1}, days_ago=1)   # 本窗
    _mk_interview(db, t, cand, job, dims={"x": 1}, days_ago=1)
    _mk_interview(db, t, cand, job, dims={"x": 1}, days_ago=40)  # 前窗(30d 窗)
    r = build_quality_report(db, u, days=30)
    assert r["kpis"]["interviews_total"] == 2
    assert r["kpis"]["interviews_delta"] == "+100%"
