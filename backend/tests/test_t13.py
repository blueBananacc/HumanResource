# -*- coding: utf-8 -*-
"""T13 声明式评分规则引擎 — 单元/服务/API 三层测试。

覆盖:
- normalize_dimensions: 方法白名单/参数裁剪/id 去重/权重钳制/上限
- estimate_experience_years: "N年"字面/日期区间/至今
- 确定性评估器: keyword(any/all)/duration/boolean/range
- evaluate_with_rule: 确定性+AI 混合/加权聚合/红旗/快照/单维 fail-open
- evaluate_match: 挂规则走规则引擎 / 不挂走 T12 路径(零回归)
- generate_rule_draft / analyze_rule / apply_analysis: 归一化+版本递增
- API: CRUD/租户隔离/generate/analyze/apply/挂岗位校验
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t13")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t13"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Job,
    Resume,
    ScoringRule,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import scoring_rules as sr  # noqa: E402
from app.services.match_evaluator import evaluate_match as eval_one  # noqa: E402


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
    t = Tenant(name=f"t13-{uuid.uuid4().hex[:8]}")
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
        description=kw.pop("description", "负责后端服务开发。"),
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
        storage_key=f"t13-{uuid.uuid4().hex}",
        source="upload",
        parse_status="done",
        parsed_text=parsed_text or "5 年 Python / FastAPI 经验,主导微服务架构改造。本科。",
        parsed_data={"skills": ["Python", "FastAPI"]},
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _rule(db, t: Tenant, dims: list[dict] | None = None, **kw) -> ScoringRule:
    r = ScoringRule(
        tenant_id=t.id,
        name=kw.pop("name", f"规则-{uuid.uuid4().hex[:6]}"),
        description=kw.pop("description", "测试规则"),
        dimensions=dims
        or [
            {
                "id": "skills",
                "name": "关键技能",
                "weight": 50,
                "evaluator": {
                    "method": "keyword",
                    "keywords": ["Python"],
                    "matchMode": "any",
                },
            },
            {
                "id": "fit",
                "name": "综合契合",
                "weight": 50,
                "evaluator": {"method": "ai", "aiPrompt": "评估契合度"},
            },
        ],
        **kw,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


# ======================= normalize_dimensions =======================


def test_normalize_drops_invalid_and_dedupes():
    dims = sr.normalize_dimensions(
        [
            {"name": "技能", "weight": 50, "evaluator": {"method": "keyword"}},
            {"name": "", "weight": 30},  # 无名丢弃
            "garbage",
            {"id": "dup", "name": "A", "evaluator": {"method": "ai"}},
            {"id": "dup", "name": "B", "evaluator": {"method": "ai"}},
        ]
    )
    assert len(dims) == 2  # 无名/非 dict/重复 id 各丢一个
    assert [d["id"] for d in dims].count("dup") == 1


def test_normalize_clamps_weight_and_method():
    dims = sr.normalize_dimensions(
        [
            {"name": "x", "weight": 999, "evaluator": {"method": "bogus"}},
            {"name": "y", "weight": -5, "evaluator": {"method": "range", "minValue": 5, "maxValue": 2}},
        ]
    )
    assert dims[0]["weight"] == 100
    assert dims[0]["evaluator"]["method"] == "ai"  # 非法方法回落 ai
    assert dims[1]["evaluator"]["maxValue"] >= dims[1]["evaluator"]["minValue"]


def test_normalize_keyword_keeps_only_its_params():
    dims = sr.normalize_dimensions(
        [
            {
                "name": "技能",
                "evaluator": {
                    "method": "keyword",
                    "keywords": ["Go", "K8s"],
                    "matchMode": "all",
                    "aiPrompt": "应被裁掉",
                    "minYears": 99,
                },
            }
        ]
    )
    ev = dims[0]["evaluator"]
    assert ev == {"method": "keyword", "keywords": ["Go", "K8s"], "matchMode": "all"}


def test_normalize_max_dimensions():
    dims = sr.normalize_dimensions(
        [{"name": f"d{i}", "evaluator": {"method": "ai"}} for i in range(20)]
    )
    assert len(dims) == sr.MAX_DIMENSIONS


# ======================= estimate_experience_years =======================


def test_estimate_years_literal():
    assert sr.estimate_experience_years("拥有 8 年后端开发经验") == 8.0


def test_estimate_years_date_ranges():
    # 2018-2021 + 2021-至今 → 取区间合计与字面较大者
    yrs = sr.estimate_experience_years("2018.03 - 2021.05 任职 A;2021 - 至今 B")
    assert yrs >= 5.0


def test_estimate_years_empty():
    assert sr.estimate_experience_years("没有任何时间信息") == 0.0


# ======================= 确定性评估器 =======================


def test_keyword_any_and_all(db):
    t = _tenant(db)
    r = _resume(db, t, parsed_text="精通 Python 和 Django,熟悉 PostgreSQL。")
    # 注意:子串匹配,慎用短词('Go' 会被 'Django' 误命中 — 与源实现同)
    any_ev = {"method": "keyword", "keywords": ["Python", "Rust"], "matchMode": "any"}
    all_ev = {"method": "keyword", "keywords": ["Python", "Rust"], "matchMode": "all"}
    s_any = sr._eval_deterministic(r, r.parsed_text, {"evaluator": any_ev})
    s_all = sr._eval_deterministic(r, r.parsed_text, {"evaluator": all_ev})
    assert s_any["score"] == 70  # 1/2 命中 → 50+20
    assert s_all["score"] == 50  # 1/2 → 50
    assert "Python" in s_any["evidence"] and "Rust" in s_any["evidence"]


def test_duration_interpolation(db):
    t = _tenant(db)
    r = _resume(db, t, parsed_text="5 年经验")
    out = sr._eval_deterministic(
        r, r.parsed_text,
        {"evaluator": {"method": "duration", "minYears": 3, "preferredYears": 7}},
    )
    assert out["score"] == 80  # 60 + (5-3)/(7-3)*40
    assert "5.0" in out["evidence"]


def test_boolean_skills_and_education(db):
    t = _tenant(db)
    r = _resume(db, t, parsed_text="本科毕业,5年经验")
    skills_ev = {"method": "boolean", "booleanField": "skills"}
    edu_ev = {"method": "boolean", "booleanField": "education"}
    assert sr._eval_deterministic(r, r.parsed_text, {"evaluator": skills_ev})["score"] == 100
    assert sr._eval_deterministic(r, r.parsed_text, {"evaluator": edu_ev})["score"] == 100
    # 无 parsed_data.skills → 0
    r.parsed_data = {}
    assert sr._eval_deterministic(r, r.parsed_text, {"evaluator": skills_ev})["score"] == 0


def test_range_mapping(db):
    t = _tenant(db)
    r = _resume(db, t, parsed_text="5 年经验")
    out = sr._eval_deterministic(
        r, r.parsed_text,
        {"evaluator": {"method": "range", "minValue": 0, "maxValue": 10}},
    )
    assert out["score"] == 50


# ======================= evaluate_with_rule =======================


def test_rule_eval_mixed_dims(db):
    """keyword(确定性) + ai(批量一次调用) 混合维加权。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t, parsed_text="5 年 Python 经验,做过大型微服务项目")
    rule = _rule(db, t)

    calls = []

    def fake_llm(call_type, messages, **kw):
        calls.append(call_type)
        return {"dims": {"fit": {"evidence": "mock:微服务", "level": 4}}}

    import app.services.scoring_rules as _sr

    orig = _sr.call_llm_json
    _sr.call_llm_json = fake_llm
    try:
        out = sr.evaluate_with_rule(db, resume=r, job=j, rule=rule)
    finally:
        _sr.call_llm_json = orig

    # keyword 命中 Python → any: 100+20 封顶=100;ai level4 → 100 → 加权 100
    assert out["score"] == 100
    assert calls == ["scoring_rule.evaluate"]  # ai 维合并一发
    methods = {c["key"]: c["method"] for c in out["criteria"]}
    assert methods == {"skills": "keyword", "fit": "ai"}
    assert out["rule_snapshot"]["name"] == rule.name
    assert out["rule_snapshot"]["version"] == rule.version


def test_rule_eval_red_flag_on_zero_high_weight(db):
    """确定性维 0 分且权重≥30 → major 红旗。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t, parsed_text="只会 Excel")
    rule = _rule(
        db, t,
        dims=[
            {"id": "skills", "name": "关键技能", "weight": 60,
             "evaluator": {"method": "keyword", "keywords": ["Rust"], "matchMode": "all"}},
            {"id": "edu", "name": "学历", "weight": 40,
             "evaluator": {"method": "boolean", "booleanField": "education"}},
        ],
    )
    out = sr.evaluate_with_rule(db, resume=r, job=j, rule=rule)
    assert out["red_flags"]["major"] == ["关键技能", "学历"]  # 两者 0 分且权重≥30
    assert out["score"] < 50


def test_rule_eval_ai_dim_failopen(db):
    """ai 维 LLM 失败 → 该维 50 不阻断其余维。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    rule = _rule(db, t)

    def boom(*a, **kw):
        from app.integrations.llm.provider import LLMError

        raise LLMError("mock down")

    orig = sr.call_llm_json
    sr.call_llm_json = boom
    try:
        out = sr.evaluate_with_rule(db, resume=r, job=j, rule=rule)
    finally:
        sr.call_llm_json = orig
    fit = next(c for c in out["criteria"] if c["key"] == "fit")
    assert fit["score"] == 50 and "不可用" in fit["evidence"]
    # keyword 维照常算出 → 总分是 (kw_score*50 + 50*50)/100
    kw = next(c for c in out["criteria"] if c["key"] == "skills")
    assert out["score"] == round((kw["score"] * 50 + 50 * 50) / 100)


# ======================= evaluate_match 分路 =======================


def test_match_uses_rule_when_linked(db):
    """Job.scoring_rule_id 挂上 → Gate 后走规则引擎,criteria 带 method。"""
    t = _tenant(db)
    rule = _rule(db, t)
    j = _job(db, t, scoring_rule_id=rule.id)
    r = _resume(db, t)
    out = eval_one(db, resume=r, job=j)
    assert out["gated"] is False
    assert out["rule_snapshot"] is not None
    assert out["rule_snapshot"]["rule_id"] == rule.id
    assert {c["method"] for c in out["criteria"]} <= {"keyword", "ai"}


def test_match_without_rule_t12_path(db):
    """不挂规则 → criteria 是 T12 五维,无 rule_snapshot。"""
    t = _tenant(db)
    j = _job(db, t)
    r = _resume(db, t)
    out = eval_one(db, resume=r, job=j)
    assert out["rule_snapshot"] is None
    keys = {c["key"] for c in (out["criteria"] or [])}
    assert keys <= {"skills", "experience", "project_depth", "domain", "stability"}


def test_rule_snapshot_frozen_after_rule_change(db):
    """评估后改规则,match 里快照仍是旧版本。"""
    t = _tenant(db)
    rule = _rule(db, t)
    j = _job(db, t, scoring_rule_id=rule.id)
    r = _resume(db, t)
    out = eval_one(db, resume=r, job=j)
    assert out["rule_snapshot"]["version"] == 1
    rule.dimensions = sr.normalize_dimensions(
        [{"id": "x", "name": "新维", "evaluator": {"method": "ai"}}]
    )
    rule.version += 1
    db.commit()
    assert out["rule_snapshot"]["version"] == 1  # 快照不被改
    assert out["rule_snapshot"]["dimensions"][0]["id"] == "skills"


# ======================= generate / analyze / apply =======================


def test_generate_draft_normalized(db):
    t = _tenant(db)
    j = _job(db, t, skills=["Go", "Docker"])
    draft = sr.generate_rule_draft(db, j)
    assert draft["dimensions"]
    # mock 生成器返回混合方法;归一化后方法白名单
    assert all(d["evaluator"]["method"] in sr.VALID_METHODS for d in draft["dimensions"])


def test_analyze_and_apply_bumps_version(db):
    t = _tenant(db)
    rule = _rule(db, t)
    j = _job(db, t, scoring_rule_id=rule.id)
    analysis = sr.analyze_rule(db, rule, [j])
    assert analysis["proposedRule"]["dimensions"]
    v0 = rule.version
    sr.apply_analysis(rule, analysis)
    db.commit()
    assert rule.version == v0 + 1
    # mock 草案:稳定性维被加入
    assert any(d["id"] == "stability" for d in rule.dimensions)


# ======================= API =======================


def test_api_crud_and_tenant_isolation(db, client):
    t = _tenant(db)
    u = _user(db, t)
    h = _hdr(u)
    dims = [
        {"name": "技能", "weight": 60, "evaluator": {"method": "keyword", "keywords": ["Go"]}},
        {"name": "年限", "weight": 40, "evaluator": {"method": "duration", "minYears": 3}},
    ]
    res = client.post("/api/scoring-rules", headers=h, json={"name": "后端规则", "dimensions": dims})
    assert res.status_code == 200, res.text
    rid = res.json()["id"]
    assert res.json()["weight_total"] == 100

    # 更新 → version+1
    res = client.put(f"/api/scoring-rules/{rid}", headers=h,
                     json={"name": "后端规则v2", "dimensions": dims})
    assert res.json()["version"] == 2

    # 跨租户不可见
    t2 = _tenant(db)
    u2 = _user(db, t2)
    res = client.get(f"/api/scoring-rules/{rid}", headers=_hdr(u2))
    assert res.status_code == 404

    # 删除
    assert client.delete(f"/api/scoring-rules/{rid}", headers=h).status_code == 204


def test_api_empty_dimensions_400(db, client):
    t = _tenant(db)
    u = _user(db, t)
    res = client.post("/api/scoring-rules", headers=_hdr(u),
                      json={"name": "空", "dimensions": []})
    assert res.status_code == 400


def test_api_generate_endpoint(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t, skills=["Go"])
    res = client.post("/api/scoring-rules/generate", headers=_hdr(u), json={"job_id": j.id})
    assert res.status_code == 200, res.text
    assert res.json()["dimensions"]
    assert res.json()["weight_total"] == 100


def test_api_analyze_requires_linked_job(db, client):
    t = _tenant(db)
    u = _user(db, t)
    rule = _rule(db, t)
    res = client.post(f"/api/scoring-rules/{rule.id}/analyze", headers=_hdr(u))
    assert res.status_code == 400  # 未挂岗位

    j = _job(db, t, scoring_rule_id=rule.id)
    res = client.post(f"/api/scoring-rules/{rule.id}/analyze", headers=_hdr(u))
    assert res.status_code == 200, res.text
    body = res.json()
    assert "proposedRule" in body and body["coverageGaps"]

    res = client.post(
        f"/api/scoring-rules/{rule.id}/apply-analysis",
        headers=_hdr(u),
        json={"analysis": body},
    )
    assert res.status_code == 200, res.text
    assert res.json()["version"] == 2


def test_job_link_rule_validation(db, client):
    """挂不存在的规则 / 别租户规则 → 400。"""
    t = _tenant(db)
    u = _user(db, t)
    res = client.post(
        "/api/jobs/",
        headers=_hdr(u),
        json={"title": "测试岗", "scoring_rule_id": "no-such-id"},
    )
    assert res.status_code == 400

    t2 = _tenant(db)
    other_rule = _rule(db, t2)
    res = client.post(
        "/api/jobs/",
        headers=_hdr(u),
        json={"title": "测试岗", "scoring_rule_id": other_rule.id},
    )
    assert res.status_code == 400

    rule = _rule(db, t)
    res = client.post(
        "/api/jobs/",
        headers=_hdr(u),
        json={"title": "测试岗", "scoring_rule_id": rule.id},
    )
    assert res.status_code == 201, res.text
    assert res.json()["scoring_rule_id"] == rule.id
