"""AI 绩效助手四件套路测:目标建议 / 评语起草 / 反馈分析 / 流失预警。

LLM 统一 monkeypatch;flight-risk 是纯规则引擎,无 LLM 依赖。
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Appraisal,
    AppraisalCycle,
    Attendance,
    Base,
    Employee,
    EmployeeCheckin,
    Goal,
    InsightNote,
    Survey,
    SurveyQuestion,
    SurveyResponse,
    SurveyResponseAnswer,
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
    t = Tenant(name=f"pa-{uuid.uuid4().hex[:8]}")
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


def _emp(db, t, **kw) -> Employee:
    e = Employee(
        tenant_id=t.id,
        employee_no=kw.pop("no", f"E{uuid.uuid4().hex[:6]}"),
        name=kw.pop("name", "张三"),
        designation=kw.pop("designation", "后端工程师"),
        status="active",
        **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


# ---------------- 1. 目标设定建议 ----------------


def test_goal_suggest_llm(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    db.add(Goal(tenant_id=t.id, employee_id=e.id, title="旧目标", progress=50))
    db.commit()

    import app.api.ai as ai_api

    fake = (
        '{"goals": ['
        '{"title": "Q3 完成结算链路重构", "description": "9月底前上线,延迟 P99<300ms"},'
        '{"title": "输出服务监控手册", "description": "覆盖核心接口,组织内分享一次"}'
        ']}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)

    r = client.post("/api/ai/goal-suggest",
                    json={"employee_id": e.id}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["goals"]) == 2
    assert body["goals"][0]["title"].startswith("Q3")
    assert body["degraded"] is False


def test_goal_suggest_degraded_and_perms(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)

    import app.api.ai as ai_api
    from app.integrations.llm.provider import LLMError

    def boom(*a, **kw):
        raise LLMError("no provider")

    monkeypatch.setattr(ai_api, "call_llm", boom)
    r = client.post("/api/ai/goal-suggest",
                    json={"employee_id": e.id}, headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert len(body["goals"]) >= 2

    # 无 manage_performance 权限的角色 → 403
    viewer = _user(db, t, role="employee")
    r2 = client.post("/api/ai/goal-suggest",
                     json={"employee_id": e.id}, headers=_hdr(viewer))
    assert r2.status_code in (403, 404)


def test_goal_suggest_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    e2 = _emp(db, t2)
    r = client.post("/api/ai/goal-suggest",
                    json={"employee_id": e2.id}, headers=_hdr(u1))
    assert r.status_code == 404


# ---------------- 2. 绩效评语起草 ----------------


def _appraisal_ctx(db, t, e):
    today = date.today()
    cy = AppraisalCycle(
        tenant_id=t.id, name="2025 年中",
        period_start=today - timedelta(days=90),
        period_end=today + timedelta(days=30),
        status="open",
    )
    db.add(cy)
    db.commit()
    a = Appraisal(
        tenant_id=t.id, cycle_id=cy.id, employee_id=e.id,
        self_review="本季度完成了交付,自评优秀。", self_rating=5,
        status="self_submitted",
    )
    db.add(a)
    db.commit()
    return cy, a


def test_appraisal_assist_llm(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    cy, a = _appraisal_ctx(db, t, e)
    db.add(Goal(tenant_id=t.id, employee_id=e.id,
                title="重构上线", progress=100, status="done",
                due_date=cy.period_start + timedelta(days=30)))
    db.add(InsightNote(tenant_id=t.id, employee_id=e.id,
                       kind="project_review", title="重构复盘",
                       body="主导结算重构,提前一周交付。"))
    db.commit()

    import app.api.ai as ai_api

    fake = (
        '{"suggested_rating": 4, '
        '"review_draft": "本周期主导结算重构并提前交付,目标全部达成。",'
        '"strengths": ["交付质量高"], "improvements": ["文档沉淀不足"],'
        '"calibration_note": "自评5分与事实基本相符,建议4-5分。",'
        '"talking_points": ["确认重构收益口径", "下周期挑战目标"]}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)

    r = client.post("/api/ai/appraisal-assist",
                    json={"appraisal_id": a.id}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["suggested_rating"] == 4
    assert "重构" in body["review_draft"]
    assert len(body["talking_points"]) == 2
    assert body["evidence"]["goal_count"] == 1
    assert body["degraded"] is False


def test_appraisal_assist_degraded(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    cy, a = _appraisal_ctx(db, t, e)
    db.add(Goal(tenant_id=t.id, employee_id=e.id, title="G1", progress=90,
                status="done", due_date=cy.period_start + timedelta(days=10)))
    db.commit()

    import app.api.ai as ai_api
    from app.integrations.llm.provider import LLMError

    def boom(*a, **kw):
        raise LLMError("down")

    monkeypatch.setattr(ai_api, "call_llm", boom)
    r = client.post("/api/ai/appraisal-assist",
                    json={"appraisal_id": a.id}, headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert "目标" in body["review_draft"]
    assert body["evidence"]["avg_progress"] == 90


def test_appraisal_assist_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    e2 = _emp(db, t2)
    _cy, a2 = _appraisal_ctx(db, t2, e2)
    r = client.post("/api/ai/appraisal-assist",
                    json={"appraisal_id": a2.id}, headers=_hdr(u1))
    assert r.status_code == 404


# ---------------- 3. 反馈分析 ----------------


def _survey_ctx(db, t):
    s = Survey(tenant_id=t.id, title="员工满意度", status="published")
    db.add(s)
    db.commit()
    q1 = SurveyQuestion(tenant_id=t.id, survey_id=s.id, type="rating",
                        text="工作满意度", order=0)
    q2 = SurveyQuestion(tenant_id=t.id, survey_id=s.id, type="long_text",
                        text="建议", order=1)
    db.add_all([q1, q2])
    db.commit()
    resp = SurveyResponse(tenant_id=t.id, survey_id=s.id,
                          respondent_key=uuid.uuid4().hex, happiness=6)
    db.add(resp)
    db.commit()
    db.add_all([
        SurveyResponseAnswer(tenant_id=t.id, response_id=resp.id,
                             question_id=q1.id, value=3),
        SurveyResponseAnswer(tenant_id=t.id, response_id=resp.id,
                             question_id=q2.id, value="加班太多,希望减负"),
    ])
    db.commit()
    return s


def test_feedback_survey_llm(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    s = _survey_ctx(db, t)

    import app.api.ai as ai_api

    fake = (
        '{"sentiment": "mixed", '
        '"themes": [{"theme": "工作负荷", "support": "high", "quote": "加班太多"}],'
        '"risks": ["加班问题集中"], "actions": ["评估排班", "补人力"]}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)

    r = client.post("/api/ai/feedback-analyze",
                    json={"target": "survey", "survey_id": s.id}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sentiment"] == "mixed"
    assert body["themes"][0]["theme"] == "工作负荷"
    assert body["response_count"] == 1


def test_feedback_employee_llm(client, db, monkeypatch):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    db.add_all([
        InsightNote(tenant_id=t.id, employee_id=e.id, kind="manager_note",
                    body="最近状态一般,沟通变少。"),
        InsightNote(tenant_id=t.id, employee_id=e.id, kind="milestone",
                    body="完成季度OKR复盘。"),
    ])
    db.commit()

    import app.api.ai as ai_api

    fake = (
        '{"sentiment": "mixed", '
        '"themes": [{"theme": "沟通减少", "count": 1, "polarity": "neg"}],'
        '"trajectory": "flat", "summary": "近期互动走弱,建议 1on1。",'
        '"actions": ["安排一对一沟通"]}'
    )
    monkeypatch.setattr(ai_api, "call_llm", lambda *a, **kw: fake)

    r = client.post("/api/ai/feedback-analyze",
                    json={"target": "employee", "employee_id": e.id},
                    headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trajectory"] == "flat"
    assert body["themes"][0]["polarity"] == "neg"


def test_feedback_employee_empty_notes(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t)
    r = client.post("/api/ai/feedback-analyze",
                    json={"target": "employee", "employee_id": e.id},
                    headers=_hdr(u))
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ---------------- 4. 流失预警(规则引擎)----------------


def test_flight_risk_signals(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t, hire_date=date.today() - timedelta(days=900))

    today = date.today()
    # L3:缺勤 3 天 + 请假 4 天(前 30 天 0)
    for i in range(3):
        db.add(Attendance(tenant_id=t.id, employee_id=e.id,
                          day=today - timedelta(days=i + 1), status="absent"))
    for i in range(4):
        db.add(Attendance(tenant_id=t.id, employee_id=e.id,
                          day=today - timedelta(days=10 + i), status="leave"))
    # 打卡在 60 天前有过、近 14 天没有 → no_checkin_14d
    db.add(EmployeeCheckin(tenant_id=t.id, employee_id=e.id,
                           ts=datetime.now() - timedelta(days=40), kind="in"))
    # L2:逾期未结目标
    db.add(Goal(tenant_id=t.id, employee_id=e.id, title="过期目标",
                status="active", progress=10,
                due_date=today - timedelta(days=10)))
    db.commit()

    r = client.get("/api/flight-risk", headers=_hdr(u))
    assert r.status_code == 200, r.text
    rows = r.json()
    mine = [x for x in rows if x["employee_id"] == e.id]
    assert mine, "员工未上榜"
    row = mine[0]
    assert row["level"] in ("medium", "high")
    keys = {s["key"] for s in row["signals"]}
    assert "absent_days" in keys
    assert "goals_overdue" in keys
    assert "no_checkin_14d" in keys
    assert row["score"] >= 30

    # 单人端点
    r2 = client.get(f"/api/flight-risk/{e.id}", headers=_hdr(u))
    assert r2.status_code == 200
    assert r2.json()["employee_id"] == e.id


def test_flight_risk_clean_employee_low(client, db):
    t = _tenant(db)
    u = _user(db, t)
    e = _emp(db, t, hire_date=date.today() - timedelta(days=200))
    today = date.today()
    db.add(EmployeeCheckin(tenant_id=t.id, employee_id=e.id,
                           ts=datetime.now(), kind="in"))
    db.add(Attendance(tenant_id=t.id, employee_id=e.id, day=today, status="present"))
    db.add(InsightNote(tenant_id=t.id, employee_id=e.id, kind="manager_note",
                       body="表现稳定"))
    db.commit()

    r = client.get(f"/api/flight-risk/{e.id}", headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert body["level"] == "low"
    assert body["score"] < 30


def test_flight_risk_perm_and_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    viewer = _user(db, t1, role="employee")
    e2 = _emp(db, t2)
    # 无 manage_employees → 403
    r = client.get("/api/flight-risk", headers=_hdr(viewer))
    assert r.status_code == 403
    # 跨租户 404
    u1 = _user(db, t1)
    r2 = client.get(f"/api/flight-risk/{e2.id}", headers=_hdr(u1))
    assert r2.status_code == 404
