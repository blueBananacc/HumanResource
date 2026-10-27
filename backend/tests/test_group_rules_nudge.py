"""人才库自动入库规则 + 面试官工作台/面评催办路测。

规则:字段词表校验 / all|any 语义 / 创建-标签-阶段钩子 / rescan 幂等 / 隔离。
工作台:待办分桶 / 逾期判定 / 手动催办去重 / 提交反馈销账 / 自动扫描。
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    CandidateGroup,
    CandidateGroupMember,
    InterviewScorecard,
    Job,
    JobApplication,
    Notification,
    Skill,
    CandidateSkill,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import scorecards as sc_svc  # noqa: E402


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
    t = Tenant(name=f"gr-{uuid.uuid4().hex[:8]}")
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


def _cand(db, t, name="张三", tags=None) -> Candidate:
    c = Candidate(tenant_id=t.id, name=name, tags=tags)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _job(db, t, title="后端工程师") -> Job:
    j = Job(tenant_id=t.id, title=title, status="open")
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _group(client, u, name="自动池"):
    r = client.post("/api/talent-groups", json={"name": name}, headers=_hdr(u))
    assert r.status_code == 201, r.text
    return r.json()


# ---------------- 自动入库规则 ----------------


def test_rule_crud_and_validation(db, client):
    t = _tenant(db)
    u = _user(db, t)
    g = _group(client, u)

    # 非法字段 / 算子 / 空条件全部 422
    for bad in (
        {"mode": "all", "conditions": [{"field": "nope", "op": "eq", "value": "x"}]},
        {"mode": "all", "conditions": [{"field": "tag", "op": "eq", "value": ["a"]}]},
        {"mode": "all", "conditions": []},
        {"mode": "bad", "conditions": [{"field": "name", "op": "contains", "value": "x"}]},
        {"mode": "all", "conditions": [{"field": "stage", "op": "in", "value": ["nope"]}]},
    ):
        r = client.put(f"/api/talent-groups/{g['id']}/rule",
                       json={"auto_rule": bad}, headers=_hdr(u))
        assert r.status_code == 422, (bad, r.status_code, r.text)

    rule = {
        "mode": "all",
        "conditions": [
            {"field": "tag", "op": "contains_any", "value": ["内推", "985"]},
            {"field": "status", "op": "eq", "value": "active"},
        ],
    }
    r = client.put(f"/api/talent-groups/{g['id']}/rule",
                   json={"auto_rule": rule}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    assert r.json()["auto_rule"]["mode"] == "all"

    # 字段词表端点
    r = client.get("/api/talent-groups/rule-fields", headers=_hdr(u))
    assert r.status_code == 200
    fields = {f["field"] for f in r.json()["fields"]}
    assert {"name", "tag", "skill", "stage", "text", "job"} <= fields

    # 清除规则
    r = client.put(f"/api/talent-groups/{g['id']}/rule",
                   json={"auto_rule": None}, headers=_hdr(u))
    assert r.status_code == 200
    assert r.json()["auto_rule"] is None


def test_rule_tag_match_and_rescan(db, client):
    t = _tenant(db)
    u = _user(db, t)
    c1 = _cand(db, t, "命中甲", tags=["内推", "VIP"])
    c2 = _cand(db, t, "命中乙", tags=["985"])
    _cand(db, t, "不中丙", tags=["社招"])

    g = _group(client, u, "高校池")
    rule = {"mode": "any", "conditions": [
        {"field": "tag", "op": "contains_any", "value": ["985", "内推"]}]}
    r = client.put(f"/api/talent-groups/{g['id']}/rule",
                   json={"auto_rule": rule}, headers=_hdr(u))
    assert r.status_code == 200, r.text

    # preview:不落库
    r = client.post("/api/talent-groups/rule-preview",
                    json={"auto_rule": rule}, headers=_hdr(u))
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 2

    # rescan:命中两人,理由可解释
    r = client.post(f"/api/talent-groups/{g['id']}/rescan", headers=_hdr(u))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["scanned"] >= 3
    assert len(data["added"]) == 2
    assert {a["candidate_id"] for a in data["added"]} == {c1.id, c2.id}
    assert "标签命中" in data["added"][0]["reasons"]

    # 幂等:二次补扫零新增
    r = client.post(f"/api/talent-groups/{g['id']}/rescan", headers=_hdr(u))
    assert r.json()["added"] == []

    # source=auto 且列表带 auto_count
    from sqlalchemy import select
    members = db.scalars(
        select(CandidateGroupMember).where(
            CandidateGroupMember.group_id == g["id"]
        )
    ).all()
    assert all(m.source == "auto" for m in members)
    r = client.get("/api/talent-groups", headers=_hdr(u))
    grp = next(x for x in r.json() if x["id"] == g["id"])
    assert grp["auto_count"] == 2


def test_rule_hooks_tags_and_stage(db, client):
    """标签更新 + 管道流转两处钩子即时入库。"""
    t = _tenant(db)
    u = _user(db, t)
    g = _group(client, u, "面试池")
    rule = {"mode": "any", "conditions": [
        {"field": "tag", "op": "contains_any", "value": ["重点"]},
        {"field": "stage", "op": "in", "value": ["interview"]},
    ]}
    r = client.put(f"/api/talent-groups/{g['id']}/rule",
                   json={"auto_rule": rule}, headers=_hdr(u))
    assert r.status_code == 200

    c = _cand(db, t, "钩子丙")
    # 打标签 → 入组
    r = client.put(f"/api/talents/{c.id}/tags",
                   json={"tags": ["重点"]}, headers=_hdr(u))
    assert r.status_code == 200
    m = db.scalars(
        select(CandidateGroupMember).where(
            CandidateGroupMember.group_id == g["id"],
            CandidateGroupMember.candidate_id == c.id,
        )
    ).first()
    assert m is not None and m.source == "auto"

    # 另一个候选人:管道推到 interview → 入组
    c2 = _cand(db, t, "钩子丁")
    job = _job(db, t)
    app_ = JobApplication(
        tenant_id=t.id, candidate_id=c2.id, job_id=job.id, stage="applied",
    )
    db.add(app_)
    db.commit()
    from app.services.applications import advance_application

    advance_application(db, t.id, c2.id, job.id, "interview")
    db.commit()
    m2 = db.scalars(
        select(CandidateGroupMember).where(
            CandidateGroupMember.group_id == g["id"],
            CandidateGroupMember.candidate_id == c2.id,
        )
    ).first()
    assert m2 is not None


def test_rule_tenant_isolation_and_perms(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u1, u2 = _user(db, t1), _user(db, t2)
    g = _group(client, u1, "隔离池")
    # t2 不可见 t1 的分组端点
    for method, url in (
        ("put", f"/api/talent-groups/{g['id']}/rule"),
        ("post", f"/api/talent-groups/{g['id']}/rescan"),
    ):
        r = getattr(client, method)(
            url, headers=_hdr(u2),
            json={"auto_rule": {"mode": "all", "conditions": [
                {"field": "name", "op": "contains", "value": "x"}]}}
            if method == "put" else None,
        )
        assert r.status_code == 404, (method, r.status_code)


# ---------------- 面试官工作台 + 催办 ----------------


def _scorecard(db, t, cand, interviewer_ids, hours_ago=None):
    sched = (
        datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours_ago)
        if hours_ago is not None else None
    )
    sc = InterviewScorecard(
        tenant_id=t.id, candidate_id=cand.id, title="一面·技术面",
        criteria=[{"key": "k1", "name": "编码", "dimension": "tech", "weight": 1}],
        interviewer_ids=interviewer_ids, scheduled_at=sched, status="open",
    )
    db.add(sc)
    db.commit()
    db.refresh(sc)
    return sc


def test_workbench_buckets_and_feedback(db, client):
    t = _tenant(db)
    hr = _user(db, t, "admin")
    iv = _user(db, t, "manager")
    c = _cand(db, t, "面评戊")

    # 48h 前的面试 → 逾期;2h 前 → 待填不逾期;未来 → upcoming
    sc_over = _scorecard(db, t, c, [iv.id], hours_ago=48)
    sc_due = _scorecard(db, t, c, [iv.id], hours_ago=2)
    sc_future = _scorecard(db, t, c, [iv.id], hours_ago=-5)

    r = client.get("/api/scorecards/workbench", headers=_hdr(iv))
    assert r.status_code == 200, r.text
    data = r.json()
    todo_ids = {i["id"] for i in data["todo"]}
    up_ids = {i["id"] for i in data["upcoming"]}
    assert {sc_over.id, sc_due.id} <= todo_ids
    assert sc_future.id in up_ids
    by_id = {i["id"]: i for i in data["todo"] + data["upcoming"]}
    assert by_id[sc_over.id]["overdue"] is True
    assert by_id[sc_due.id]["overdue"] is False
    assert by_id[sc_over.id]["my_submitted"] is False

    # 提交反馈 → 移出 todo 进 done
    r = client.post(f"/api/scorecards/{sc_due.id}/feedback",
                    json={"rating": 4, "verdict": "recommend",
                          "scores": [{"criterion": "k1", "score": 4}]},
                    headers=_hdr(iv))
    assert r.status_code == 200, r.text
    r = client.get("/api/scorecards/workbench", headers=_hdr(iv))
    data = r.json()
    assert sc_due.id in {i["id"] for i in data["done"]}
    assert sc_due.id not in {i["id"] for i in data["todo"]}


def test_nudge_manual_dedupe_and_ack(db, client):
    t = _tenant(db)
    hr = _user(db, t, "admin")
    iv = _user(db, t, "manager")
    c = _cand(db, t, "催办己")
    sc = _scorecard(db, t, c, [iv.id], hours_ago=30)

    # HR 手动催 → 面试官收到通知
    r = client.post(f"/api/scorecards/{sc.id}/nudge", headers=_hdr(hr))
    assert r.status_code == 200, r.text
    assert len(r.json()["sent"]) == 1
    n = db.scalars(
        select(Notification).where(
            Notification.user_id == iv.id,
            Notification.kind == "scorecard.nudge",
        )
    ).all()
    assert len(n) == 1
    assert sc.id in (n[0].link or "")

    # cooldown 内再催 → 不发第二条
    r = client.post(f"/api/scorecards/{sc.id}/nudge", headers=_hdr(hr))
    assert r.json()["sent"] == []
    n2 = db.scalars(
        select(Notification).where(
            Notification.user_id == iv.id,
            Notification.kind == "scorecard.nudge",
        )
    ).all()
    assert len(n2) == 1

    # 面试官无 HR 权限也能被通知(通知本身是行级投递,无需权限)
    # 但面试官自己不能催别人:manager 无 write_interview → 403
    r = client.post(f"/api/scorecards/{sc.id}/nudge", headers=_hdr(iv))
    assert r.status_code == 403

    # 提交反馈 → 催办通知自动已读(ack)
    r = client.post(f"/api/scorecards/{sc.id}/feedback",
                    json={"rating": 3, "verdict": "hold",
                          "scores": [{"criterion": "k1", "score": 3}]},
                    headers=_hdr(iv))
    assert r.status_code == 200
    db.refresh(n[0])
    assert n[0].read_at is not None


def test_auto_nudge_sweep(db, client):
    """run_scorecard_nudges:逾期卡自动催,提交后不再发。"""
    t = _tenant(db)
    iv = _user(db, t, "manager")
    c = _cand(db, t, "自动催庚")
    sc = _scorecard(db, t, c, [iv.id], hours_ago=30)
    # 未逾期的卡不应被扫到
    sc_fresh = _scorecard(db, t, c, [iv.id], hours_ago=2)

    # 全租户扫描 — 本用例目标卡必须被催(其他测试残留逾期卡也可能被催)
    sent = sc_svc.run_scorecard_nudges(db)
    assert sent >= 1
    nudges = [
        n for n in db.scalars(
            select(Notification).where(
                Notification.user_id == iv.id,
                Notification.kind == "scorecard.nudge",
            )
        ).all()
        if sc.id in (n.link or "")
    ]
    assert len(nudges) == 1
    # 未逾期卡不被扫
    fresh_nudges = [
        n for n in db.scalars(
            select(Notification).where(
                Notification.user_id == iv.id,
                Notification.kind == "scorecard.nudge",
            )
        ).all()
        if sc_fresh.id in (n.link or "")
    ]
    assert fresh_nudges == []

    # 再扫:全部 cooldown 去重 → 0
    assert sc_svc.run_scorecard_nudges(db) == 0

    # 提交反馈 → 再扫仍然 0(无 pending)
    sc_svc.submit_feedback(
        db, scorecard=sc, interviewer_user_id=iv.id,
        rating=4, verdict="recommend", notes=None,
        scores=[{"criterion": "k1", "score": 4, "comment": None}],
    )
    db.commit()
    assert sc_svc.run_scorecard_nudges(db) == 0
