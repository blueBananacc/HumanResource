"""T9 测试:结构化面试评分卡 + 声明式全局搜索。

评分卡:建卡(维度规范化/面试官校验)/权限(被指派面试官/view_reports+scope)/
反馈 upsert/冻结/关闭后拒收/聚合(均分·分歧)/AI 合成降级/乐观锁/租户隔离。

搜索:on_audit 事件索引(建→搜→改→删)/reindex 孤儿清扫/租户定义覆盖
(权重·停用)/权限回读收敛/实体过滤/facets。
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t9")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t9"

from app.domain import models as m  # noqa: E402,F401  # 确保新表进 metadata
from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Employee,
    InterviewFeedback,
    InterviewScorecard,
    Job,
    SearchDocument,
    SearchIndexDefinition,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import scorecards as sc_svc  # noqa: E402
from app.services import search_index as si  # noqa: E402


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


def _tenant(db) -> Tenant:
    t = Tenant(name=f"t9-{uuid.uuid4().hex[:8]}")
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


def _candidate(db, t: Tenant, name: str = "链测候选人") -> Candidate:
    c = Candidate(tenant_id=t.id, name=name, display_email=f"{uuid.uuid4().hex[:6]}@x.com")
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _job(db, t: Tenant) -> Job:
    j = Job(tenant_id=t.id, title=f"链测岗{uuid.uuid4().hex[:6]}", status="open")
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _mk_scorecard(db, t: Tenant, cand: Candidate, interviewers: list[str]) -> InterviewScorecard:
    sc = InterviewScorecard(
        tenant_id=t.id,
        candidate_id=cand.id,
        title="一面·技术面",
        criteria=[
            {"key": "sys", "name": "系统设计", "dimension": "tech", "weight": 2},
            {"key": "comm", "name": "沟通表达", "dimension": "soft", "weight": 1},
        ],
        interviewer_ids=interviewers,
        status="open",
    )
    db.add(sc)
    db.commit()
    db.refresh(sc)
    return sc


@pytest.fixture
def client():
    return TestClient(app)


# ======================= 评分卡:服务层 =======================


def test_normalize_criteria_generates_keys():
    out = sc_svc.normalize_criteria(
        [{"name": "系统设计", "dimension": "tech", "weight": 2}, {"name": "沟通"}]
    )
    assert out[0]["key"] and out[1]["key"]
    assert out[1]["dimension"] == "general"
    assert out[0]["weight"] == 2


def test_normalize_criteria_rejects_bad():
    with pytest.raises(ValueError):
        sc_svc.normalize_criteria([{"name": "  "}])
    with pytest.raises(ValueError):
        sc_svc.normalize_criteria([{"name": "x", "dimension": "nope"}])
    with pytest.raises(ValueError):
        sc_svc.normalize_criteria([{"name": "x", "key": "a"}, {"name": "y", "key": "a"}])


def test_aggregate_disagreement(db):
    t = _tenant(db)
    c = _candidate(db, t)
    u1, u2 = _user(db, t), _user(db, t)
    sc = _mk_scorecard(db, t, c, [u1.id, u2.id])
    for u, rating, verdict, s_sys in (
        (u1, 5, "recommend", 5),
        (u2, 2, "reject", 2),
    ):
        db.add(
            InterviewFeedback(
                tenant_id=t.id,
                scorecard_id=sc.id,
                interviewer_user_id=u.id,
                rating=rating,
                verdict=verdict,
                scores=[
                    {"criterion": "sys", "score": s_sys},
                    {"criterion": "comm", "score": 3},
                ],
                submitted_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
    db.commit()
    agg = sc_svc.aggregate(
        sc,
        db.scalars(
            select(InterviewFeedback).where(InterviewFeedback.scorecard_id == sc.id)
        ).all(),
    )
    assert agg["feedback_count"] == 2
    assert agg["avg_rating"] == 3.5
    assert agg["verdict_tally"]["recommend"] == 1
    assert agg["verdict_tally"]["reject"] == 1
    # 系统设计 5 vs 2 → 极差 3 ≥ 2 标分歧;表态 recommend+reject 也标
    kinds = {d["kind"] for d in agg["disagreements"]}
    assert kinds == {"criterion", "verdict"}
    sys_row = next(x for x in agg["criteria"] if x["key"] == "sys")
    assert sys_row["avg"] == 3.5 and sys_row["min"] == 2 and sys_row["max"] == 5


# ======================= 评分卡:API 链路 =======================


def test_scorecard_create_feedback_flow(client, db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    iv1, iv2 = _user(db, t, role="interviewer"), _user(db, t, role="interviewer")
    cand = _candidate(db, t)
    job = _job(db, t)

    r = client.post(
        "/api/scorecards",
        headers=_hdr(hr),
        json={
            "candidate_id": cand.id,
            "job_id": job.id,
            "title": "一面·技术面",
            "criteria": [
                {"name": "系统设计", "dimension": "tech", "weight": 2},
                {"name": "沟通表达", "dimension": "soft"},
            ],
            "interviewer_ids": [iv1.id, iv2.id],
        },
    )
    assert r.status_code == 201, r.text
    sc = r.json()
    keys = {c["key"] for c in sc["criteria"]}
    assert len(keys) == 2
    assert {i["user_id"] for i in sc["interviewers"]} == {iv1.id, iv2.id}

    # 面试官 1 提交反馈
    r = client.post(
        f"/api/scorecards/{sc['id']}/feedback",
        headers=_hdr(iv1),
        json={
            "rating": 4,
            "verdict": "recommend",
            "scores": [
                {"criterion": c["key"], "score": 4, "comment": "结构清晰"}
                for c in sc["criteria"]
            ],
            "notes": "整体不错",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["aggregate"]["feedback_count"] == 1

    # 面试官 2 提交(分歧)
    r = client.post(
        f"/api/scorecards/{sc['id']}/feedback",
        headers=_hdr(iv2),
        json={
            "rating": 2,
            "verdict": "reject",
            "scores": [
                {"criterion": c["key"], "score": 2} for c in sc["criteria"]
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["avg_rating"] == 3.0
    assert len(body["aggregate"]["disagreements"]) >= 1

    # upsert:面试官 2 改判
    r = client.post(
        f"/api/scorecards/{sc['id']}/feedback",
        headers=_hdr(iv2),
        json={"rating": 3, "verdict": "hold", "scores": []},
    )
    assert r.status_code == 200
    assert r.json()["aggregate"]["feedback_count"] == 2
    assert r.json()["aggregate"]["avg_rating"] == 3.5


def test_scorecard_feedback_requires_assignment(client, db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    iv = _user(db, t, role="interviewer")
    outsider = _user(db, t, role="interviewer")
    ess = _user(db, t, role="employee")
    cand = _candidate(db, t)
    sc = _mk_scorecard(db, t, cand, [iv.id])

    # 未被指派的面试官 → 反馈 403(interviewer 有 view_reports 仍可看,但不能评)
    r = client.post(
        f"/api/scorecards/{sc.id}/feedback",
        headers=_hdr(outsider),
        json={"rating": 4, "verdict": "recommend", "scores": []},
    )
    assert r.status_code == 403
    # 无任何权限的 ESS 用户 → 查不到该卡(不泄露存在性)
    assert client.get(f"/api/scorecards/{sc.id}", headers=_hdr(ess)).status_code == 404

    # 关闭后拒收
    db.refresh(sc)
    sc.status = "closed"
    db.commit()
    r = client.post(
        f"/api/scorecards/{sc.id}/feedback",
        headers=_hdr(iv),
        json={"rating": 4, "verdict": "recommend", "scores": []},
    )
    assert r.status_code == 400


def test_scorecard_mine_and_visibility(client, db):
    t = _tenant(db)
    cand = _candidate(db, t)
    iv = _user(db, t, role="interviewer")
    ess = _user(db, t, role="employee")
    sc = _mk_scorecard(db, t, cand, [iv.id])

    # /mine:面试官看得到
    r = client.get("/api/scorecards/mine", headers=_hdr(iv))
    assert r.status_code == 200
    assert any(x["id"] == sc.id for x in r.json())

    # ESS 用户无权限且未被指派 → GET 404 / list 403
    assert client.get(f"/api/scorecards/{sc.id}", headers=_hdr(ess)).status_code == 404
    assert client.get("/api/scorecards", headers=_hdr(ess)).status_code == 403
    assert client.get("/api/scorecards/mine", headers=_hdr(ess)).json() == []


def test_scorecard_criteria_freeze_and_lock(client, db):
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    iv = _user(db, t, role="interviewer")
    cand = _candidate(db, t)
    sc = _mk_scorecard(db, t, cand, [iv.id])
    db.refresh(sc)

    client.post(
        f"/api/scorecards/{sc.id}/feedback",
        headers=_hdr(iv),
        json={"rating": 4, "verdict": "recommend", "scores": []},
    )
    # 有反馈后维度冻结
    r = client.patch(
        f"/api/scorecards/{sc.id}",
        headers=_hdr(hr),
        json={"criteria": [{"name": "新维度"}]},
    )
    assert r.status_code == 400
    # 乐观锁
    stale = sc.updated_at.replace(year=2000).isoformat()
    r = client.patch(
        f"/api/scorecards/{sc.id}",
        headers=_hdr(hr),
        json={"title": "改名", "base_updated_at": stale},
    )
    assert r.status_code == 409


def test_scorecard_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    hr1 = _user(db, t1, role="hr")
    cand2 = _candidate(db, t2)
    sc2 = _mk_scorecard(db, t2, cand2, [])

    # 跨租户 GET/PATCH/DELETE 一律 404
    assert client.get(f"/api/scorecards/{sc2.id}", headers=_hdr(hr1)).status_code == 404
    assert client.patch(
        f"/api/scorecards/{sc2.id}", headers=_hdr(hr1), json={"title": "x"}
    ).status_code == 404
    assert client.delete(f"/api/scorecards/{sc2.id}", headers=_hdr(hr1)).status_code == 404

    # 跨租户候选人 → 404
    r = client.post(
        "/api/scorecards",
        headers=_hdr(hr1),
        json={"candidate_id": cand2.id, "title": "x"},
    )
    assert r.status_code == 404


def test_scorecard_synthesize_fallback(client, db):
    """无真实 LLM → 机械聚合回落,ai_summary 仍落库。"""
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    iv = _user(db, t, role="interviewer")
    cand = _candidate(db, t)
    sc = _mk_scorecard(db, t, cand, [iv.id])
    client.post(
        f"/api/scorecards/{sc.id}/feedback",
        headers=_hdr(iv),
        json={"rating": 5, "verdict": "recommend",
              "scores": [{"criterion": "sys", "score": 5}]},
    )
    r = client.post(f"/api/scorecards/{sc.id}/synthesize", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ai_summary"]["feedback_count"] == 1
    assert body["ai_summary"]["summary"]  # 回落文本非空

    # 无反馈 → 400
    sc2 = _mk_scorecard(db, t, cand, [iv.id])
    assert client.post(
        f"/api/scorecards/{sc2.id}/synthesize", headers=_hdr(hr)
    ).status_code == 400


# ======================= 全局搜索 =======================


def test_on_audit_indexes_candidate(db):
    """write_audit 漏斗:on_audit 把 candidate.create 事件落成搜索文档。"""
    from app.services.audit import write_audit

    t = _tenant(db)
    hr = _user(db, t, role="hr")
    name = f"搜测{uuid.uuid4().hex[:6]}"
    c = _candidate(db, t, name=name)
    write_audit(
        db, actor=hr, entity_type="candidate", entity_id=c.id, action="create",
        detail={"name": name},
    )
    db.commit()
    doc = db.scalars(
        select(SearchDocument).where(
            SearchDocument.tenant_id == t.id,
            SearchDocument.entity == "talents",
            SearchDocument.entity_id == c.id,
        )
    ).first()
    assert doc is not None
    assert name in doc.title
    assert doc.definition_version == 0  # 未覆盖 → 注册表默认

    # 更新事件 → 文档刷新;非索引实体事件 → 无文档无副作用
    c.display_phone = "13900000000"
    write_audit(
        db, actor=hr, entity_type="candidate", entity_id=c.id, action="update",
    )
    write_audit(
        db, actor=hr, entity_type="interview_scorecard", entity_id="x", action="create",
    )
    db.commit()
    db.refresh(doc)
    assert "13900000000" in (doc.body or "")
    assert db.scalars(
        select(SearchDocument).where(SearchDocument.entity == "interview_scorecard")
    ).first() is None


def test_doc_hits_and_entity_filter(db):
    t = _tenant(db)
    token = uuid.uuid4().hex[:6]
    c = _candidate(db, t, name=f"搜测人{token}")
    j = _job(db, t)
    j.title = f"搜测岗{token}"
    db.commit()
    si.reindex(db, t.id)
    db.commit()

    hits = si.doc_hits(db, t.id, token)
    entities = {d.entity for d, _ in hits}
    assert "talents" in entities and "jobs" in entities

    only_t = si.doc_hits(db, t.id, token, entity="talents")
    assert all(d.entity == "talents" for d, _ in only_t)
    assert any(d.entity_id == c.id for d, _ in only_t)


def test_search_api_permission_reread(client, db):
    """索引命中后还要过 build() 权限回读:无权限角色拿不到结果。"""
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    emp_only = _user(db, t, role="employee")
    token = uuid.uuid4().hex[:6]
    _candidate(db, t, name=f"搜测人{token}")
    si.reindex(db, t.id)
    db.commit()

    r = client.get(f"/api/search?q={token}", headers=_hdr(hr))
    assert r.status_code == 200
    assert r.json()["total"] >= 1
    assert any(h["entity"] == "talents" for h in r.json()["items"])
    # facets 给出实体命中数
    assert any(f["entity"] == "talents" and f["count"] >= 1 for f in r.json()["facets"])

    # employee 角色无 view_reports → 命中被权限回读收敛为空
    r2 = client.get(f"/api/search?q={token}", headers=_hdr(emp_only))
    assert r2.status_code == 200
    assert r2.json()["total"] == 0


def test_search_delete_propagates(client, db):
    """DELETE /api/talents/{id} → audit action=delete → 文档同步删除。"""
    t = _tenant(db)
    hr = _user(db, t, role="hr")
    c = _candidate(db, t, name=f"删测{uuid.uuid4().hex[:6]}")
    si.upsert_document(db, t.id, "talents", c)
    db.commit()
    assert db.scalars(
        select(SearchDocument).where(SearchDocument.entity_id == c.id)
    ).first() is not None
    r = client.delete(f"/api/talents/{c.id}", headers=_hdr(hr))
    assert r.status_code in (200, 204), r.text
    assert db.scalars(
        select(SearchDocument).where(
            SearchDocument.tenant_id == t.id, SearchDocument.entity_id == c.id
        )
    ).first() is None


def test_search_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    hr1 = _user(db, t1, role="hr")
    token = uuid.uuid4().hex[:6]
    _candidate(db, t2, name=f"他租户{token}")
    si.reindex(db, t2.id)
    db.commit()

    r = client.get(f"/api/search?q={token}", headers=_hdr(hr1))
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_definition_override_and_disable(client, db):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    token = uuid.uuid4().hex[:6]
    _candidate(db, t, name=f"覆盖测{token}")
    si.reindex(db, t.id)
    db.commit()

    # 关掉 talents 定义 → reindex 清掉该实体文档 → 搜不到
    r = client.patch(
        "/api/search/definitions/talents",
        headers=_hdr(admin),
        json={"is_active": False},
    )
    assert r.status_code == 200
    row = db.scalars(
        select(SearchIndexDefinition).where(
            SearchIndexDefinition.tenant_id == t.id,
            SearchIndexDefinition.entity == "talents",
        )
    ).first()
    assert row is not None and row.is_active is False and row.version == 1

    res = si.reindex(db, t.id)
    db.commit()
    assert res["talents"]["removed"] >= 1
    r = client.get(f"/api/search?q={token}", headers=_hdr(admin))
    assert not any(h["entity"] == "talents" for h in r.json()["items"])

    # 白名单外字段被拒
    r = client.patch(
        "/api/search/definitions/talents",
        headers=_hdr(admin),
        json={"fields": [{"name": "password_hash", "weight": 5}]},
    )
    assert r.status_code == 400


def test_reindex_orphan_sweep(db):
    t = _tenant(db)
    c = _candidate(db, t, name=f"孤儿{uuid.uuid4().hex[:6]}")
    si.reindex(db, t.id)
    db.commit()
    assert db.scalars(
        select(SearchDocument).where(
            SearchDocument.tenant_id == t.id, SearchDocument.entity_id == c.id
        )
    ).first() is not None
    # 直接删源行(不走 audit)→ 文档成孤儿 → reindex 清扫
    db.delete(c)
    db.commit()
    res = si.reindex(db, t.id, "talents")
    db.flush()  # SessionLocal autoflush=False
    assert res["talents"]["removed"] >= 1
    assert db.scalars(
        select(SearchDocument).where(SearchDocument.entity_id == c.id)
    ).first() is None


def test_index_status_and_reindex_endpoint(client, db):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    _candidate(db, t)
    r = client.post("/api/search/reindex", headers=_hdr(admin), json={})
    assert r.status_code == 200
    assert "talents" in r.json()
    r = client.get("/api/search/index-status", headers=_hdr(admin))
    assert r.status_code == 200
    assert any(x["entity"] == "talents" for x in r.json())


def test_employee_model_searchable(db):
    """employees 实体也能建索引(hr.suite feature 在场时 build 仍有数据)。"""
    t = _tenant(db)
    name = f"员搜{uuid.uuid4().hex[:6]}"
    e = Employee(tenant_id=t.id, name=name, employee_no=f"E{uuid.uuid4().hex[:5]}")
    db.add(e)
    db.commit()
    obj = si.upsert_document(db, t.id, "employees", e)
    assert obj is not None and name in obj.title
