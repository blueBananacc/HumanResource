# -*- coding: utf-8 -*-
"""T12 硬性门槛 Gate + 证据先行评分 + Prompt 注入防御 — 单元/服务/API 三层测试。

覆盖:
- evaluate_gate 纯裁决: FAIL∧sev≥7 才淘汰 / UNCERTAIN 放行 / 去重 / 非法值
- derive_gate_profile: 归一化 / jd_hash 缓存 / 重派生 / fail-open
- aggregate_score: 加权聚合 / 锚定级钳制 / 红旗分级
- evaluate_match: gate 先行 / gated 短路 / 新 criteria 格式 / 旧格式兼容 / 兜底
- 注入防御: guarded 幂等 / DATA_LABEL / 评分 prompt 带守卫
- API: status=gated 过滤 / MatchOut 新字段 / 非法 status 422 / 跨租户
- worker: gated 终态不重跑
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t12")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t12"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Job,
    Resume,
    ResumeJobMatch,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import match_gate  # noqa: E402
from app.services.match_evaluator import evaluate_match as eval_one  # noqa: E402
from app.services.prompting import DATA_LABEL, INJECTION_GUARD, guarded  # noqa: E402
from app.workers.tasks import match as match_worker  # noqa: E402


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
    t = Tenant(name=f"t12-{uuid.uuid4().hex[:8]}")
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


def _job(db, t: Tenant, **kw) -> Job:
    j = Job(
        tenant_id=t.id,
        title=kw.pop("title", f"岗位-{uuid.uuid4().hex[:6]}"),
        level="intermediate",
        status="open",
        description=kw.pop("description", "负责后端服务开发。必须持有效工作许可,base 上海。"),
        skills=kw.pop("skills", ["Python", "FastAPI"]),
        **kw,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _resume(db, t: Tenant, *, parsed_text: str | None = None) -> Resume:
    cand = Candidate(tenant_id=t.id, name=f"候选人-{uuid.uuid4().hex[:6]}")
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id,
        candidate_id=cand.id,
        file_name="r.pdf",
        file_size=100,
        file_mime="application/pdf",
        storage_key=f"t12-{uuid.uuid4().hex}",
        source="upload",
        parse_status="done",
        parsed_text=parsed_text or "5 年 Python / FastAPI 经验,主导微服务架构改造。",
        parsed_data={"skills": ["Python", "FastAPI"]},
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


# ======================= evaluate_gate 纯裁决 =======================


_QS = [
    {"id": "auth", "question": "能合法全职工作?", "severity": 9},
    {"id": "onsite", "question": "能到上海到岗?", "severity": 8},
    {"id": "lang", "question": "会日语?", "severity": 4},
]


def test_gate_fail_high_severity_disqualifies():
    out = match_gate.evaluate_gate(
        _QS,
        [
            {"id": "auth", "verdict": "FAIL", "evidence": "人在海外无工签"},
            {"id": "onsite", "verdict": "PASS", "evidence": "base 上海"},
        ],
    )
    assert out["passed"] is False
    assert len(out["disqualifiers"]) == 1
    assert out["disqualifiers"][0]["id"] == "auth"
    assert out["disqualifiers"][0]["severity"] == 9
    assert "海外" in out["disqualifiers"][0]["evidence"]


def test_gate_fail_low_severity_only_advisory():
    out = match_gate.evaluate_gate(
        _QS, [{"id": "lang", "verdict": "FAIL", "evidence": "不会日语"}]
    )
    assert out["passed"] is True
    assert out["disqualifiers"] == []


def test_gate_uncertain_never_disqualifies():
    out = match_gate.evaluate_gate(
        _QS, [{"id": "auth", "verdict": "UNCERTAIN", "evidence": "未提及"}]
    )
    assert out["passed"] is True
    assert out["disqualifiers"] == []


def test_gate_missing_and_unknown_answers_safe():
    # 没答的题 / 答了不存在的题 都不淘汰
    out = match_gate.evaluate_gate(
        _QS, [{"id": "ghost", "verdict": "FAIL", "evidence": "x"}]
    )
    assert out["passed"] is True
    assert out["answers"] == []


def test_gate_bad_verdict_becomes_uncertain():
    out = match_gate.evaluate_gate(
        _QS, [{"id": "auth", "verdict": "MAYBE", "evidence": "模糊"}]
    )
    assert out["passed"] is True
    assert out["answers"][0]["verdict"] == "UNCERTAIN"


def test_gate_duplicate_answer_dedup_disqualifier():
    out = match_gate.evaluate_gate(
        _QS,
        [
            {"id": "auth", "verdict": "FAIL", "evidence": "e1"},
            {"id": "auth", "verdict": "FAIL", "evidence": "e2"},
        ],
    )
    assert len(out["disqualifiers"]) == 1


# ======================= derive_gate_profile =======================


def _fake_llm(payload: dict):
    """monkeypatch 用的 call_llm_json — 直接返回 dict(绕过真 LLM)。"""

    def _call(call_type, messages, **kw):
        return dict(payload)

    return _call


def test_derive_profile_normalizes_and_caches(db, monkeypatch):
    t = _tenant(db)
    j = _job(db, t)
    calls = [0]

    def fake(call_type, messages, **kw):
        calls[0] += 1
        return {
            "questions": [
                {"id": "a", "question": "问题A?", "severity": 99, "why": "w"},
                {"id": "a", "question": "重复id应去重", "severity": 9},
                {"id": "b", "question": "问题B?", "severity": 3},
            ],
            "emphasis": {"skills": 200, "experience": 1},
        }

    monkeypatch.setattr(match_gate, "call_llm_json", fake)
    p1 = match_gate.derive_gate_profile(db, j)
    assert calls[0] == 1
    # 同 id 去重,severity 钳到 10
    assert [q["id"] for q in p1["questions"]] == ["a", "b"]
    assert p1["questions"][0]["severity"] == 10
    # 权重钳 5-100
    assert p1["emphasis"]["skills"] == 100
    assert p1["emphasis"]["experience"] == 5
    assert p1["jd_hash"]

    # 第二次:jd_hash 命中 → 不再调 LLM
    p2 = match_gate.derive_gate_profile(db, j)
    assert calls[0] == 1
    assert p2["jd_hash"] == p1["jd_hash"]

    # JD 变更 → 指纹变 → 重新派生
    j.description = "全新 JD,要求完全不同的东西。"
    db.commit()
    match_gate.derive_gate_profile(db, j)
    assert calls[0] == 2


def test_derive_profile_llm_error_fail_open(db, monkeypatch):
    from app.integrations.llm.provider import LLMError

    t = _tenant(db)
    j = _job(db, t)

    def boom(call_type, messages, **kw):
        raise LLMError("down")

    monkeypatch.setattr(match_gate, "call_llm_json", boom)
    p = match_gate.derive_gate_profile(db, j)
    assert p["questions"] == []
    assert p["emphasis"]["skills"] == 40  # 默认权重兜底


def test_run_gate_no_questions_and_error_fail_open(db, monkeypatch):
    t = _tenant(db)
    out = match_gate.run_gate(db, resume_text="x", questions=[], tenant_id=t.id)
    assert out["passed"] is True
    assert out["skipped"] == "no_questions"

    from app.integrations.llm.provider import LLMError

    def boom(call_type, messages, **kw):
        raise LLMError("down")

    monkeypatch.setattr(match_gate, "call_llm_json", boom)
    out2 = match_gate.run_gate(db, resume_text="x", questions=_QS, tenant_id=t.id)
    assert out2["passed"] is True
    assert out2["skipped"] == "error"


def test_gate_answer_prompt_hides_severity_and_guards(db, monkeypatch):
    """判定模型不应看到 severity(判定与后果分离),且 prompt 带注入守卫。"""
    t = _tenant(db)
    seen = {}

    def spy(call_type, messages, **kw):
        seen["messages"] = messages
        return {"answers": [{"id": "a", "verdict": "PASS", "evidence": "ok"}]}

    monkeypatch.setattr(match_gate, "call_llm_json", spy)
    qs = [{"id": "a", "question": "能全职?", "severity": 9, "why": "机密后果"}]
    match_gate.run_gate(db, resume_text="简历内容", questions=qs, tenant_id=t.id)

    sys_p, user_p = seen["messages"][0]["content"], seen["messages"][1]["content"]
    assert INJECTION_GUARD in sys_p
    assert DATA_LABEL in user_p
    assert "9" not in user_p  # severity 未下发
    assert "机密后果" not in user_p  # why 未下发
    assert "能全职?" in user_p  # 问题本身在


# ======================= aggregate_score =======================


def test_aggregate_weighted_and_rows():
    emphasis = {"skills": 40, "experience": 25, "project_depth": 15,
                "domain": 10, "stability": 10}
    score, rows, flags = match_gate.aggregate_score(
        {
            "skills": {"evidence": "5年Python", "level": 4},
            "experience": {"evidence": "3年经验", "level": 3},
            "project_depth": {"evidence": "小项目", "level": 2},
            "domain": {"evidence": "未提及", "level": 0},
            "stability": {"evidence": "一年一跳", "level": 1},
        },
        emphasis,
    )
    # (100*40 + 75*25 + 50*15 + 0*10 + 25*10) / 100 = 68.75 → 69
    assert score == 69
    assert [r["level"] for r in rows] == [4, 3, 2, 0, 1]
    assert rows[0]["evidence"] == "5年Python"
    assert flags["major"] == []  # domain 权重 10 < 15
    assert flags["minor"] == []


def test_aggregate_red_flags_by_weight():
    # skills 权重 40 level=0 → major;project_depth 权重 15 level=0 → minor;domain 10 → 无
    score, rows, flags = match_gate.aggregate_score(
        {
            "skills": {"evidence": "未提及", "level": 0},
            "experience": {"evidence": "有", "level": 4},
            "project_depth": {"evidence": "未提及", "level": 0},
            "domain": {"evidence": "未提及", "level": 0},
            "stability": {"evidence": "有", "level": 4},
        },
        {"skills": 40, "experience": 25, "project_depth": 15,
         "domain": 10, "stability": 10},
    )
    assert flags["major"] == ["关键技能匹配"]
    assert flags["minor"] == ["项目深度与业务规模"]


def test_aggregate_clamps_and_missing_cells():
    score, rows, flags = match_gate.aggregate_score(
        {"skills": {"level": 9}, "experience": "not-a-dict"},
        {"skills": 40, "experience": 25, "project_depth": 15,
         "domain": 10, "stability": 10},
    )
    # level 9 钳到 4;非 dict cell → level 0
    lv = {r["key"]: r["level"] for r in rows}
    assert lv["skills"] == 4
    assert lv["experience"] == 0
    assert rows[0]["evidence"] == "未提及"


def test_emphasis_of_defaults_and_clamp():
    em = match_gate.emphasis_of({"emphasis": {"skills": 0, "experience": 500}})
    assert em["skills"] == 5
    assert em["experience"] == 100
    assert em["project_depth"] == 15  # 默认


# ======================= evaluate_match 服务层 =======================


def test_eval_match_gate_passes_then_scores(db):
    """mock 链路: 门槛全 PASS → 进评分, gate 明细随结果返回。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    out = eval_one(db, resume=r, job=j)
    assert out["gated"] is False
    assert out["score"] is not None and 0 <= out["score"] <= 100
    assert out["gate"]["passed"] is True
    assert len(out["gate"]["answers"]) == 2  # mock 画像 2 题
    # 岗位画像已惰性写入
    assert j.gate_profile and j.gate_profile["questions"]


def test_eval_match_gate_fail_short_circuits(db):
    """简历含 gate_fail 标记 → 高 severity FAIL → gated, score=None, 不进评分。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t, parsed_text="gate_fail 无工作许可。Python 3 年。")
    out = eval_one(db, resume=r, job=j)
    assert out["gated"] is True
    assert out["score"] is None
    assert out["criteria"] is None
    assert out["gate"]["passed"] is False
    assert len(out["gate"]["disqualifiers"]) == 1
    assert any("门槛未过" in g for g in out["gaps"])
    # 淘汰也留了 answers 供审计
    assert len(out["gate"]["answers"]) == 2


def test_eval_match_new_criteria_format_aggregated(db, monkeypatch):
    """模型给 criteria{evidence,level} → 代码侧加权出分 + 红旗。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    # 预置画像(避开 derive 调 LLM)
    j.gate_profile = {
        "jd_hash": match_gate._jd_hash(j),
        "questions": [],
        "emphasis": {"skills": 50, "experience": 20, "project_depth": 15,
                     "domain": 10, "stability": 5},
    }
    db.commit()

    def fake(call_type, messages, **kw):
        assert call_type == "match.evaluate"
        return {
            "criteria": {
                "skills": {"evidence": "精通 Python", "level": 4},
                "experience": {"evidence": "3年", "level": 3},
                "project_depth": {"evidence": "未提及", "level": 0},
                "domain": {"evidence": "电商", "level": 2},
                "stability": {"evidence": "3年一司", "level": 3},
            },
            "strengths": ["技能扎实"],
            "gaps": ["项目规模未知"],
            "comment": "整体不错",
        }

    import app.services.match_evaluator as me

    monkeypatch.setattr(me, "call_llm_json", fake)
    out = eval_one(db, resume=r, job=j)
    # (100*50 + 75*20 + 0*15 + 50*10 + 75*5)/100 = 73.75 → 74
    assert out["score"] == 74
    assert len(out["criteria"]) == 5
    assert out["criteria"][0]["weight"] == 50
    assert "项目深度与业务规模" in out["red_flags"]["minor"]  # w=15 level=0
    assert out["gated"] is False


def test_eval_match_old_flat_format_compatible(db, monkeypatch):
    """旧扁平输出 {score,strengths,gaps} 仍兼容 — 用模型直给分。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    j.gate_profile = {"jd_hash": match_gate._jd_hash(j), "questions": [],
                      "emphasis": {}}
    db.commit()

    def fake(call_type, messages, **kw):
        return {"score": 250, "strengths": ["A"], "gaps": ["B"], "comment": "c"}

    import app.services.match_evaluator as me

    monkeypatch.setattr(me, "call_llm_json", fake)
    out = eval_one(db, resume=r, job=j)
    assert out["score"] == 100  # 钳制
    assert out["criteria"] is None
    assert out["red_flags"] is None


def test_eval_match_llm_error_fallback_keeps_gate(db, monkeypatch):
    """评分调用失败 → 技能重叠兜底分;gate 结果仍保留。"""
    from app.integrations.llm.provider import LLMError

    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)

    def boom(call_type, messages, **kw):
        raise LLMError("down")

    import app.services.match_evaluator as me

    monkeypatch.setattr(me, "call_llm_json", boom)
    out = eval_one(db, resume=r, job=j)
    assert out["gated"] is False
    assert out["score"] == 90  # skills 全命中 → 40+50
    assert out["gate"]["passed"] is True


def test_eval_match_prompt_has_injection_guard(db, monkeypatch):
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    j.gate_profile = {"jd_hash": match_gate._jd_hash(j), "questions": [],
                      "emphasis": {}}
    db.commit()
    seen = {}

    def spy(call_type, messages, **kw):
        seen["messages"] = messages
        return {"score": 70, "strengths": [], "gaps": [], "comment": ""}

    import app.services.match_evaluator as me

    monkeypatch.setattr(me, "call_llm_json", spy)
    eval_one(db, resume=r, job=j)
    sys_p = seen["messages"][0]["content"]
    user_p = seen["messages"][1]["content"]
    assert INJECTION_GUARD in sys_p
    assert DATA_LABEL in user_p


# ======================= 注入守卫单元 =======================


def test_guarded_appends_and_idempotent():
    p = "你是评估师。"
    out = guarded(p)
    assert INJECTION_GUARD in out
    assert out.startswith(p)
    # 幂等:二次调用不重复追加
    assert guarded(out) == out
    # 末尾有换行时不多加
    assert guarded("x\n") == "x\n" + INJECTION_GUARD


# ======================= API =======================


def _mk_match(db, t, job, resume, **kw) -> ResumeJobMatch:
    m = ResumeJobMatch(
        tenant_id=t.id, resume_id=resume.id, job_id=job.id,
        status=kw.pop("status", "done"),
        score=kw.pop("score", 80),
        strengths=["A"], gaps=[], comment="c",
        **kw,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def test_api_status_filter_gated(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    r1 = _resume(db, t)
    r2 = _resume(db, t)
    done = _mk_match(db, t, j, r1, status="done", score=88)
    gated = _mk_match(
        db, t, j, r2, status="gated", score=None,
        gate={"passed": False, "answers": [],
              "disqualifiers": [{"id": "a", "question": "工签?", "severity": 9,
                                  "evidence": "无"}]},
    )

    # 默认(done)不含 gated
    r = client.get(f"/api/matches/job/{j.id}", headers=_hdr(u))
    ids = [x["id"] for x in r.json()]
    assert done.id in ids and gated.id not in ids

    # status=gated 只回 gated
    r = client.get(f"/api/matches/job/{j.id}?status=gated", headers=_hdr(u))
    rows = r.json()
    assert [x["id"] for x in rows] == [gated.id]
    assert rows[0]["status"] == "gated"
    assert rows[0]["score"] is None
    assert rows[0]["gate"]["disqualifiers"][0]["question"] == "工签?"


def test_api_status_invalid_422(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    r = client.get(f"/api/matches/job/{j.id}?status=weird", headers=_hdr(u))
    assert r.status_code == 422


def test_api_match_out_new_fields(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    r = _resume(db, t)
    m = _mk_match(
        db, t, j, r,
        gate={"passed": True, "answers": [], "disqualifiers": []},
        criteria=[{"key": "skills", "name": "关键技能匹配", "weight": 40,
                    "level": 4, "evidence": "精通"}],
        red_flags={"major": [], "minor": ["项目深度与业务规模"]},
    )
    out = client.get(f"/api/matches/{m.id}", headers=_hdr(u)).json()
    assert out["gate"]["passed"] is True
    assert out["criteria"][0]["level"] == 4
    assert out["criteria"][0]["evidence"] == "精通"
    assert out["red_flags"]["minor"] == ["项目深度与业务规模"]


def test_api_gated_cross_tenant_404(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    _user(db, t1)
    u2 = _user(db, t2)
    j1 = _job(db, t1)
    r = client.get(f"/api/matches/job/{j1.id}?status=gated", headers=_hdr(u2))
    assert r.status_code == 404


# ======================= worker 终态 =======================


def test_worker_gated_is_terminal(db):
    """gated 的 match 行不被 evaluate_match 重跑。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    m = _mk_match(db, t, j, r, status="gated", score=None)

    payload = match_worker.evaluate_match.delay(m.id).get(timeout=10)
    assert payload["status"] == "already_gated"
    db.refresh(m)
    assert m.status == "gated"


def test_worker_eager_gated_flow(db):
    """eager 全链: resume 含 gate_fail → worker 写回 status=gated + gate 明细。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t, parsed_text="gate_fail 不在岗位城市。")
    m = ResumeJobMatch(
        tenant_id=t.id, resume_id=r.id, job_id=j.id, status="pending"
    )
    db.add(m)
    db.commit()
    db.refresh(m)

    payload = match_worker.evaluate_match.delay(m.id).get(timeout=10)
    assert payload["status"] == "gated"
    db.refresh(m)
    assert m.status == "gated"
    assert m.score is None
    assert m.gate["passed"] is False
    assert m.gate["disqualifiers"][0]["severity"] == 8  # mock 题1 severity 8
    assert m.finished_at is not None
