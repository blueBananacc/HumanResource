"""T4-A AI 产品层全链路测试。

覆盖:MCP 令牌 + token-exchange / entities aggregate+update+relations /
report-defs run / knowledge 混合检索+剧本+embedding / audit via 留痕。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t4")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t4"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, OrgUnit, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.license.state import is_feature_enabled  # noqa: E402
from app.infra.security import create_access_token, decode_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


def _tenant(db) -> Tenant:
    t = Tenant(name=f"t4-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, *, role: str = "admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u: User, **claims) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role, **claims)}"
    }


def _internal_headers() -> dict[str, str]:
    """sidecar 调 /api/mcp/token-exchange 的内部密钥头 —— 与 backend
    ``_internal_key()`` 同一算法(HMAC-SHA256(JWT_SECRET, 固定消息串))。"""
    key = hmac.new(
        settings.jwt_secret.encode(), b"hronfire/mcp/token-exchange", hashlib.sha256
    ).hexdigest()
    return {"x-internal-token": key}


def _mk_emp(db, t: Tenant, name=None, **kw):
    e = m.Employee(
        tenant_id=t.id,
        employee_no=f"E{uuid.uuid4().hex[:6]}",
        name=name or f"员工{uuid.uuid4().hex[:4]}",
        employment_type="full_time",
        status="active",
        **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _mk_org(db, t: Tenant, name: str) -> OrgUnit:
    o = OrgUnit(tenant_id=t.id, name=name, kind="department")
    db.add(o)
    db.commit()
    db.refresh(o)
    return o


def _mk_cand(db, t: Tenant, **kw):
    c = m.Candidate(
        tenant_id=t.id,
        name=kw.pop("name", f"候选人{uuid.uuid4().hex[:4]}"),
        status=kw.pop("status", "active"),
        **kw,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _mk_job(db, t: Tenant, **kw):
    j = m.Job(
        tenant_id=t.id,
        title=kw.pop("title", f"岗位{uuid.uuid4().hex[:4]}"),
        level="intermediate",
        status="open",
        **kw,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        for model in (
            m.McpToken, m.AuditLog, m.KnowledgeArticle, m.ExpenseClaim,
            m.SalarySlip, m.JobApplication, m.ResumeJobMatch, m.Resume,
            m.Interview, m.JobOffer, m.EmployeeRequest,
            m.HrCaseUpdate, m.HrCase, m.LeaveApplication, m.Goal,
            m.Notification, m.CandidateSkill, m.Candidate,
            m.JobSkill, m.JobVersion, m.Reservation, m.Room,
            m.Job, m.Employee, User, OrgUnit,
        ):
            db.query(model).filter(model.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


# ============ T4-1 MCP tokens ============


def test_mcp_token_lifecycle(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        # 签发 → 明文一次性返回
        r = client.post("/api/me/mcp-tokens", headers=hdr, json={"name": "claude desktop"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["token"].startswith("hrm_")
        assert len(body["token"]) > 40
        # 列表不含明文
        items = client.get("/api/me/mcp-tokens", headers=hdr).json()
        assert len(items) == 1 and "token" not in items[0]
        # 库里存的是 hash,不是明文
        row = db_session.scalar(select(m.McpToken))
        assert row.token_hash != body["token"]
        assert row.name == "claude desktop"

        # token-exchange → tool_token + 用户上下文(需内部密钥)
        r2 = client.post(
            "/api/mcp/token-exchange", json={"token": body["token"]}, headers=_internal_headers()
        )
        assert r2.status_code == 200, r2.text
        ex = r2.json()
        assert ex["user"]["id"] == u.id and ex["user"]["tenant_id"] == t.id
        # tool_token 能打真实 API(等价本人身份),但寿命被收窄到 15min 而非全局 480min
        r3 = client.get("/api/auth/me", headers={"Authorization": f"Bearer {ex['tool_token']}"})
        assert r3.status_code == 200
        payload = decode_token(ex["tool_token"])
        assert payload["via"] == "mcp"
        ttl = payload["exp"] - datetime.now(UTC).timestamp()
        assert 0 < ttl <= 15 * 60 + 5
        # last_used_at 已记
        db_session.refresh(row)
        assert row.last_used_at is not None

        # 吊销 → exchange 401
        client.delete(f"/api/me/mcp-tokens/{row.id}", headers=hdr)
        r4 = client.post(
            "/api/mcp/token-exchange", json={"token": body["token"]}, headers=_internal_headers()
        )
        assert r4.status_code == 401

        # 假 token → 401
        assert client.post(
            "/api/mcp/token-exchange", json={"token": "hrm_bogus"}, headers=_internal_headers()
        ).status_code == 401
    finally:
        _cleanup(db_session, t)


def test_mcp_exchange_requires_internal_key(db_session):
    """token-exchange 仅 sidecar 内部开放:缺/错内部密钥一律 403 —
    外部拿 PAT 无法直接换全量 JWT(PAT 范围收敛在 MCP 工具面内)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        tok = client.post("/api/me/mcp-tokens", headers=_hdr(u), json={"name": "x"}).json()["token"]
        assert client.post("/api/mcp/token-exchange", json={"token": tok}).status_code == 403
        assert client.post(
            "/api/mcp/token-exchange", json={"token": tok},
            headers={"x-internal-token": "bogus"},
        ).status_code == 403
        r = client.post("/api/mcp/token-exchange", json={"token": tok}, headers=_internal_headers())
        assert r.status_code == 200
    finally:
        _cleanup(db_session, t)


def test_mcp_token_active_cap(db_session):
    """每用户最多 20 个有效令牌;吊销一个即可再签。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        for i in range(20):
            assert client.post("/api/me/mcp-tokens", headers=hdr, json={"name": f"t{i}"}).status_code == 201
        assert client.post("/api/me/mcp-tokens", headers=hdr, json={"name": "overflow"}).status_code == 400
        items = client.get("/api/me/mcp-tokens", headers=hdr).json()
        client.delete(f"/api/me/mcp-tokens/{items[0]['id']}", headers=hdr)
        assert client.post("/api/me/mcp-tokens", headers=hdr, json={"name": "again"}).status_code == 201
    finally:
        _cleanup(db_session, t)


def test_mcp_token_user_isolation(db_session):
    """别人看不到也吊销不了我的令牌。"""
    t = _tenant(db_session)
    u1, u2 = _user(db_session, t), _user(db_session, t)
    client = TestClient(app)
    try:
        mine = client.post("/api/me/mcp-tokens", headers=_hdr(u1), json={"name": "mine"}).json()
        # u2 的列表里没有
        assert client.get("/api/me/mcp-tokens", headers=_hdr(u2)).json() == []
        # u2 吊销我的令牌 → 404
        assert client.delete(f"/api/me/mcp-tokens/{mine['id']}", headers=_hdr(u2)).status_code == 404
    finally:
        _cleanup(db_session, t)


# ============ T4-2 entities aggregate / update / relations ============


def test_entities_aggregate(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    emp = _mk_emp(db_session, t)
    _mk_cand(db_session, t)
    _mk_job(db_session, t)
    for amt in (100.0, 200.0, 300.0):
        db_session.add(m.ExpenseClaim(tenant_id=t.id, employee_id=emp.id, title=f"报销{amt}", total=amt, status="paid"))
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.get("/api/entities/expenses/aggregate?agg=sum&field=total", headers=hdr)
        assert r.status_code == 200, r.text
        assert r.json()["value"] == 600.0
        avg = client.get("/api/entities/expenses/aggregate?agg=avg&field=total", headers=hdr).json()
        assert avg["value"] == 200.0
        cnt = client.get("/api/entities/expenses/aggregate?agg=count", headers=hdr).json()
        assert cnt["value"] == 3
        # agg=count 对所有实体都要可用(不限于 agg_fields 白名单实体)
        assert client.get("/api/entities/talents/aggregate?agg=count", headers=hdr).status_code == 200
        assert client.get("/api/entities/talents/aggregate?agg=count", headers=hdr).json()["value"] == 1
        assert client.get("/api/entities/jobs/aggregate?agg=count", headers=hdr).json()["value"] == 1
        assert client.get("/api/entities/employees/aggregate?agg=count", headers=hdr).json()["value"] == 1
        # 白名单外字段 → 400
        bad = client.get("/api/entities/expenses/aggregate?agg=sum&field=employee_id", headers=hdr)
        assert bad.status_code == 400
        # 无 agg_fields 的实体 → 400
        na = client.get("/api/entities/talents/aggregate?agg=sum&field=x", headers=hdr)
        assert na.status_code == 400
        # 权限不足(employee 无 manage_expenses)
        emp_u = _user(db_session, t, role="employee")
        r5 = client.get("/api/entities/expenses/aggregate?agg=sum&field=total", headers=_hdr(emp_u))
        assert r5.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_entities_update(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    cand = _mk_cand(db_session, t)
    job = _mk_job(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        # 更新白名单字段
        r = client.patch(f"/api/entities/talents/{cand.id}", headers=hdr, json={
            "fields": {"tags": ["重点", "前端"], "status": "archived"},
        })
        assert r.status_code == 200, r.text
        db_session.refresh(cand)
        assert cand.tags == ["重点", "前端"] and cand.status == "archived"
        # 审计留痕 entity_type 用单数约定(与手写端点 / activity feed 一致)
        log = db_session.scalar(select(m.AuditLog).where(m.AuditLog.entity_id == cand.id))
        assert log.action == "update" and log.entity_type == "candidate"
        # "blacklisted" 不是合法 status(黑名单是 is_blacklisted 旗标体系)
        r_bl = client.patch(f"/api/entities/talents/{cand.id}", headers=hdr, json={"fields": {"status": "blacklisted"}})
        assert r_bl.status_code == 400
        # 白名单外字段 → 400
        r2 = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr, json={"fields": {"title": "x"}})
        assert r2.status_code == 400
        # 非法值 → 400
        r3 = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr, json={"fields": {"status": "bogus"}})
        assert r3.status_code == 400
        # jobs.level 词表与 /api/jobs 一致:entry 合法,expert 非法
        with patch("app.api.jobs._enqueue_match_eval"):
            r_lv = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr, json={"fields": {"level": "entry"}})
        assert r_lv.status_code == 200, r_lv.text
        db_session.refresh(job)
        assert job.level == "entry"
        r_lv_bad = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr, json={"fields": {"level": "expert"}})
        assert r_lv_bad.status_code == 400
        # 不存在 → 404
        r4 = client.patch(f"/api/entities/jobs/{uuid.uuid4()}", headers=hdr, json={"fields": {"level": "entry"}})
        assert r4.status_code == 404
        # 权限不足(view_reports only hr? hr 有 write_resumes?测 employee 无权限)
        emp_u = _user(db_session, t, role="employee")
        r5 = client.patch(f"/api/entities/talents/{cand.id}", headers=_hdr(emp_u), json={"fields": {"tags": ["x"]}})
        assert r5.status_code in (403, 404)
        # 不支持更新的实体 → 400
        r6 = client.patch(f"/api/entities/announcements/{uuid.uuid4()}", headers=hdr, json={"fields": {"x": 1}})
        assert r6.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_entities_update_job_governance(db_session):
    """AI 写岗位要走与手写端点同口径的治理:
    内容变更 → JobVersion 快照 + published 回 draft;
    状态 → open / open 状态改内容 → 重投匹配评估;updated_at 维护。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    job = _mk_job(db_session, t)  # server_default: status=open, publish_status=published
    assert job.publish_status == "published"
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        ver_before = db_session.scalar(select(func.count(m.JobVersion.id)).where(m.JobVersion.job_id == job.id)) or 0
        old_updated = job.updated_at
        with patch("app.api.jobs._enqueue_match_eval") as enq:
            r = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr,
                             json={"fields": {"description": "AI 改写后的 JD"}})
        assert r.status_code == 200, r.text
        db_session.refresh(job)
        # 版本快照 + 发布态回退
        ver_after = db_session.scalar(select(func.count(m.JobVersion.id)).where(m.JobVersion.job_id == job.id)) or 0
        assert ver_after == ver_before + 1
        assert job.publish_status == "draft"
        # open 岗位改内容 → 重投匹配
        assert enq.called
        # updated_at 已刷新
        assert job.updated_at > old_updated

        # 暂停后重开 → 也触发匹配评估
        job.status = "paused"
        db_session.commit()
        db_session.refresh(job)
        with patch("app.api.jobs._enqueue_match_eval") as enq2:
            r2 = client.patch(f"/api/entities/jobs/{job.id}", headers=hdr,
                              json={"fields": {"status": "open"}})
        assert r2.status_code == 200 and enq2.called
        # paused → closed 不触发
        job.status = "paused"
        db_session.commit()
        db_session.refresh(job)
        with patch("app.api.jobs._enqueue_match_eval") as enq3:
            client.patch(f"/api/entities/jobs/{job.id}", headers=hdr,
                         json={"fields": {"status": "closed"}})
        assert not enq3.called
    finally:
        _cleanup(db_session, t)


def test_entities_relations(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    cand = _mk_cand(db_session, t)
    job = _mk_job(db_session, t)
    app_row = m.JobApplication(tenant_id=t.id, candidate_id=cand.id, job_id=job.id, stage="applied")
    db_session.add(app_row)
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.get(f"/api/entities/talents/{cand.id}/relations", headers=hdr)
        assert r.status_code == 200, r.text
        rels = {rel["key"]: rel for rel in r.json()["relations"]}
        assert rels["applications"]["items"][0]["id"] == app_row.id
        assert "resumes" in rels and rels["resumes"]["count"] == 0
        # job 侧也能穿透
        r2 = client.get(f"/api/entities/jobs/{job.id}/relations", headers=hdr)
        keys = {rel["key"] for rel in r2.json()["relations"]}
        assert {"applications", "interviews", "top_matches"} <= keys
        # 不存在 → 404;不支持 → 400
        assert client.get(f"/api/entities/talents/{uuid.uuid4()}/relations", headers=hdr).status_code == 404
        assert client.get(f"/api/entities/skills/{uuid.uuid4()}/relations", headers=hdr).status_code == 400

        # 会议室 → 进行中预约(status="active" 才应命中,cancelled 不算)
        room = m.Room(tenant_id=t.id, name="会议室A")
        db_session.add(room)
        db_session.flush()
        rsv = m.Reservation(
            tenant_id=t.id, kind="room", room_id=room.id, title="面试用会议室",
            starts_at=datetime.now(UTC).replace(tzinfo=None),
            ends_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),
        )
        db_session.add(rsv)
        db_session.commit()
        r3 = client.get(f"/api/entities/rooms/{room.id}/relations", headers=hdr)
        assert r3.status_code == 200, r3.text
        rels3 = {rel["key"]: rel for rel in r3.json()["relations"]}
        assert rels3["reservations"]["items"][0]["id"] == rsv.id
    finally:
        _cleanup(db_session, t)


# ============ T4-2 report-defs ============


def test_report_defs(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    _mk_emp(db_session, t)
    job = _mk_job(db_session, t)
    cand = _mk_cand(db_session, t)
    db_session.add(m.JobApplication(tenant_id=t.id, candidate_id=cand.id, job_id=job.id, stage="screening"))
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        defs = client.get("/api/report-defs", headers=hdr).json()
        kinds = {d["kind"] for d in defs}
        assert {"pipeline_funnel", "weekly_summary", "pending_items", "headcount"} <= kinds

        funnel = client.post("/api/report-defs/pipeline_funnel/run", headers=hdr, json={}).json()
        assert funnel["data"]["total"] == 1
        assert funnel["data"]["open_jobs"] == 1

        headcount = client.post("/api/report-defs/headcount/run", headers=hdr, json={}).json()
        assert headcount["data"]["total"] == 1
        assert headcount["data"]["by_status"][0]["count"] == 1
        # 员工无 org → by_org 要落到"(未分配)"桶,不能静默丢数
        orgs = {o["org"]: o["count"] for o in headcount["data"]["by_org"]}
        assert orgs.get("(未分配)") == 1

        pending = client.post("/api/report-defs/pending_items/run", headers=hdr, json={}).json()
        assert pending["data"]["open_cases"] == 0

        weekly = client.post("/api/report-defs/weekly_summary/run", headers=hdr, json={"params": {"days": 7}}).json()
        assert weekly["data"]["new_applications"] == 1

        assert client.post("/api/report-defs/bogus/run", headers=hdr, json={}).status_code == 404
        # headcount 需要 manage_employees — employee 403
        emp_u = _user(db_session, t, role="employee")
        assert client.post("/api/report-defs/headcount/run", headers=_hdr(emp_u), json={}).status_code == 403
        # 但 pending_items 人人可跑
        assert client.post("/api/report-defs/pending_items/run", headers=_hdr(emp_u), json={}).status_code == 200
    finally:
        _cleanup(db_session, t)


# ============ T4-3 playbooks ============


def test_knowledge_playbook(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.post("/api/knowledge/articles", headers=hdr, json={
            "title": "发 Offer 标准流程", "content": "1.查薪酬带宽 2.确认审批人 3.生成 letter",
            "status": "published", "is_playbook": True,
        })
        assert r.status_code == 201, r.text
        aid = r.json()["id"]
        assert r.json()["is_playbook"] is True
        # is_playbook 过滤
        books = client.get("/api/knowledge/articles?is_playbook=true&status_filter=published", headers=hdr).json()
        assert any(a["id"] == aid for a in books["items"])
        # 普通文章不标记
        r2 = client.post("/api/knowledge/articles", headers=hdr, json={
            "title": "年假制度", "content": "正文", "status": "published",
        }).json()
        assert r2["is_playbook"] is False
        # 单篇 get 带 is_playbook
        one = client.get(f"/api/knowledge/articles/{aid}", headers=hdr).json()
        assert one["is_playbook"] is True
    finally:
        _cleanup(db_session, t)


# ============ T4-4 混合检索 ============


def test_knowledge_trgm_search(db_session):
    """pg_trgm 命中 ILIKE 抓不到的近似词。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    db_session.add_all([
        m.KnowledgeArticle(tenant_id=t.id, title="年度休假管理制度", category="policy",
                           content="员工每年享 10 天年假,提前一周申请。", tags=["假期"], status="published"),
        m.KnowledgeArticle(tenant_id=t.id, title="打印机使用说明", category="it",
                           content="三楼打印机重启方法。", tags=[], status="published"),
        m.KnowledgeArticle(tenant_id=t.id, title="草稿休假", category="policy",
                           content="未发布内容不应命中。", tags=[], status="draft"),
    ])
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        # "休假制度" 不是 "年度休假管理制度" 的子串(隔了"管理")→ 旧 ILIKE 抓不到
        assert "休假制度" not in "年度休假管理制度"
        hits = client.post("/api/knowledge/search", headers=hdr, json={"q": "休假制度", "limit": 5}).json()
        assert hits and hits[0]["title"] == "年度休假管理制度"
        assert hits[0]["score"] is not None and hits[0]["score"] > 0
        # draft 不命中
        assert all(h["title"] != "草稿休假" for h in hits)
        # 无关词返回空而非乱给
        none = client.post("/api/knowledge/search", headers=hdr, json={"q": "xyzxyz不相关词", "limit": 5}).json()
        assert none == []
    finally:
        _cleanup(db_session, t)


def test_knowledge_embedding_hybrid(db_session):
    """有 embedding 时语义近似排前;没有也不炸(降级纯文本)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    a1 = m.KnowledgeArticle(tenant_id=t.id, title="差旅报销政策", category="policy",
                            content="差旅住宿每日限额 500 元。", tags=[], status="published")
    a2 = m.KnowledgeArticle(tenant_id=t.id, title="VPN 使用", category="it",
                            content="VPN 客户端下载地址。", tags=[], status="published")
    db_session.add_all([a1, a2])
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        # 无 embedding provider → embedding=None,纯 trgm 也能搜
        hits = client.post("/api/knowledge/search", headers=hdr, json={"q": "差旅报销", "limit": 5}).json()
        assert hits and hits[0]["title"] == "差旅报销政策"

        # 模拟 embedding:给两篇文章语义向量,让 VPN 那篇对"远程接入"语义更近
        # a2 的向量更靠近 query 向量 [1,0,0]
        a1.embedding = [0.0, 1.0]
        a2.embedding = [0.95, 0.05]
        db_session.commit()
        with patch("app.api.knowledge.embed_text", return_value=[1.0, 0.0]):
            hits2 = client.post("/api/knowledge/search", headers=hdr, json={"q": "远程接入", "limit": 5}).json()
        # lex 都不高时,embedding 把 a2 顶到第一
        assert hits2 and hits2[0]["title"] == "VPN 使用"
        # backfill:文章已有 embedding → 0
        r = client.post("/api/knowledge/embeddings/backfill", headers=hdr)
        assert r.status_code == 200
        assert r.json()["embedded"] == 0 and r.json()["remaining"] == 0
    finally:
        _cleanup(db_session, t)


def test_knowledge_backfill_with_stub(db_session):
    """provider 配好 embedding_model 后,backfill 真的能补。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    db_session.add(m.KnowledgeArticle(tenant_id=t.id, title="制度A", category="policy",
                                    content="内容A", tags=[], status="published"))
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        with patch("app.api.knowledge.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            r = client.post("/api/knowledge/embeddings/backfill", headers=hdr)
            assert r.status_code == 200
            assert r.json()["embedded"] == 1 and r.json()["remaining"] == 0
    finally:
        _cleanup(db_session, t)


# ============ T4-5 audit via ============


def test_audit_via_assistant(db_session):
    """tool_token 带 via 声明 → audit.via 落列。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        # 人工操作 → via NULL
        hdr_plain = _hdr(u)
        client.post("/api/knowledge/articles", headers=hdr_plain, json={
            "title": "人工写", "content": "x", "status": "published",
        })
        # assistant token → via="assistant"
        hdr_ai = _hdr(u, via="assistant")
        client.post("/api/knowledge/articles", headers=hdr_ai, json={
            "title": "AI 代写", "content": "x", "status": "published",
        })
        rows = db_session.scalars(
            select(m.AuditLog).where(m.AuditLog.entity_type == "knowledge_article").order_by(m.AuditLog.created_at)
        ).all()
        assert len(rows) == 2
        # 两条同 actor 但 via 不同 —— 用标题区分?detail 里没有标题;顺序断言即可
        assert rows[0].via is None
        assert rows[1].via == "assistant"
    finally:
        _cleanup(db_session, t)


def test_audit_via_mcp(db_session):
    """MCP 兑换的 tool_token 落 via='mcp'。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        tok = client.post("/api/me/mcp-tokens", headers=hdr, json={"name": "x"}).json()["token"]
        ex = client.post("/api/mcp/token-exchange", json={"token": tok}, headers=_internal_headers()).json()
        ai_hdr = {"Authorization": f"Bearer {ex['tool_token']}"}
        client.post("/api/knowledge/articles", headers=ai_hdr, json={
            "title": "MCP 代写", "content": "x", "status": "published",
        })
        row = db_session.scalar(
            select(m.AuditLog).where(m.AuditLog.entity_type == "knowledge_article")
        )
        assert row.via == "mcp"
        # /api/audit/events 输出也带 via
        ev = client.get("/api/audit/events", headers=hdr).json()["items"][0]
        assert ev["via"] == "mcp"
        # /api/activity 活动流条目同样带 via(AI 留痕前端徽标用)
        acts = client.get("/api/activity?days=7", headers=hdr).json()
        art_items = [a for a in acts if a["entity_kind"] == "knowledge_article"]
        assert art_items and art_items[0]["via"] == "mcp"
    finally:
        _cleanup(db_session, t)


# ============ 安全回归:org scope / 权限穿透 / 输入健壮性 ============


def test_relations_org_scope(db_session):
    """org 绑定的非 admin 用户:relations 不得穿透越界 org 的关联实体名称。"""
    t = _tenant(db_session)
    org_a = _mk_org(db_session, t, "A部门")
    org_b = _mk_org(db_session, t, "B部门")
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    db_session.commit()

    job_pub = _mk_job(db_session, t)                      # org NULL → 公共可见
    job_b = _mk_job(db_session, t, org_unit_id=org_b.id)  # 界外岗位
    cand_a = _mk_cand(db_session, t, org_unit_id=org_a.id)
    cand_b = _mk_cand(db_session, t, org_unit_id=org_b.id)
    cand_pub = _mk_cand(db_session, t)
    app_in = m.JobApplication(tenant_id=t.id, candidate_id=cand_a.id, job_id=job_pub.id, stage="applied")
    app_out = m.JobApplication(tenant_id=t.id, candidate_id=cand_b.id, job_id=job_pub.id, stage="applied")
    app_job_out = m.JobApplication(tenant_id=t.id, candidate_id=cand_pub.id, job_id=job_b.id, stage="applied")
    # 面试:界内候选人 + 界外岗位 —— 父记录对 hr 可见,job 项应被剔除
    iv = m.Interview(tenant_id=t.id, job_id=job_b.id, candidate_id=cand_pub.id, status="done")
    room = m.Room(tenant_id=t.id, name="会议室S")
    db_session.add_all([app_in, app_out, app_job_out, iv, room])
    db_session.flush()
    # 界外候选人的面试预约 —— 标题含候选人名,应对 hr 隐藏
    iv_b = m.Interview(tenant_id=t.id, job_id=job_pub.id, candidate_id=cand_b.id, status="done")
    db_session.add(iv_b)
    db_session.flush()
    now = datetime.now(UTC).replace(tzinfo=None)
    db_session.add_all([
        m.Reservation(tenant_id=t.id, kind="interview", room_id=room.id, interview_id=iv_b.id,
                      title="面试-界外候选人", starts_at=now, ends_at=now + timedelta(hours=1)),
        m.Reservation(tenant_id=t.id, kind="room", room_id=room.id, title="普通会议",
                      starts_at=now, ends_at=now + timedelta(hours=2)),
    ])
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(hr)
        rels_of = lambda r: {x["key"]: x for x in r.json()["relations"]}
        sec = lambda rels, k: {i["id"] for i in rels[k]["items"]}

        # jobs→applications:界外候选人(app_out)不出现在 items
        r = client.get(f"/api/entities/jobs/{job_pub.id}/relations", headers=hdr)
        ids = sec(rels_of(r), "applications")
        assert app_in.id in ids and app_out.id not in ids

        # talents→applications:界外岗位的条目整体隐藏(与 applications.py 双 org 过滤同口径)
        r = client.get(f"/api/entities/talents/{cand_pub.id}/relations", headers=hdr)
        assert sec(rels_of(r), "applications") == set()  # 唯一一条申请指向 org-B 岗位

        # applications→candidate/job:候选人可见,界外 job 项剔除
        r = client.get(f"/api/entities/applications/{app_job_out.id}/relations", headers=hdr)
        rels = rels_of(r)
        assert sec(rels, "candidate") == {cand_pub.id}
        assert sec(rels, "job") == set()

        # interviews→job 同理
        r = client.get(f"/api/entities/interviews/{iv.id}/relations", headers=hdr)
        rels = rels_of(r)
        assert sec(rels, "candidate") == {cand_pub.id}
        assert sec(rels, "job") == set()

        # rooms→reservations:绑定界外候选人面试的预约隐藏,普通预约可见
        r = client.get(f"/api/entities/rooms/{room.id}/relations", headers=hdr)
        items = rels_of(r)["reservations"]["items"]
        titles = {i["label"] for i in items}
        assert "普通会议" in titles and "面试-界外候选人" not in titles

        # 界外候选人本人:父检查直接 404
        assert client.get(f"/api/entities/talents/{cand_b.id}/relations", headers=hdr).status_code == 404

        # admin 全景:全部可见(对照组)
        admin = _user(db_session, t)
        r = client.get(f"/api/entities/jobs/{job_pub.id}/relations", headers=_hdr(admin))
        ids = sec(rels_of(r), "applications")
        assert {app_in.id, app_out.id} <= ids
    finally:
        _cleanup(db_session, t)


def test_relations_offers_perm(db_session):
    """offers 段仅 manage_offers 可见 —— viewer(view_reports)看不到 offer 段。"""
    t = _tenant(db_session)
    admin = _user(db_session, t)
    viewer = _user(db_session, t, role="viewer")
    cand = _mk_cand(db_session, t)
    db_session.add(m.JobOffer(tenant_id=t.id, candidate_id=cand.id, status="sent"))
    db_session.commit()
    client = TestClient(app)
    try:
        keys_admin = {x["key"] for x in client.get(
            f"/api/entities/talents/{cand.id}/relations", headers=_hdr(admin)).json()["relations"]}
        assert "offers" in keys_admin
        keys_viewer = {x["key"] for x in client.get(
            f"/api/entities/talents/{cand.id}/relations", headers=_hdr(viewer)).json()["relations"]}
        assert "offers" not in keys_viewer
    finally:
        _cleanup(db_session, t)


def test_report_defs_org_scope(db_session):
    """org 绑定的 hr:报表口径与实体列表一致,只统计 scope 内数据。"""
    t = _tenant(db_session)
    org_a = _mk_org(db_session, t, "A部门")
    org_b = _mk_org(db_session, t, "B部门")
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    db_session.commit()

    _mk_emp(db_session, t, org_unit_id=org_a.id)
    _mk_emp(db_session, t, org_unit_id=org_b.id)   # 界外
    cand_a = _mk_cand(db_session, t, org_unit_id=org_a.id)
    cand_b = _mk_cand(db_session, t, org_unit_id=org_b.id)
    job_a = _mk_job(db_session, t, org_unit_id=org_a.id)
    job_b = _mk_job(db_session, t, org_unit_id=org_b.id)
    db_session.add_all([
        m.JobApplication(tenant_id=t.id, candidate_id=cand_a.id, job_id=job_a.id, stage="applied"),
        m.JobApplication(tenant_id=t.id, candidate_id=cand_b.id, job_id=job_b.id, stage="applied"),
    ])
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(hr)
        funnel = client.post("/api/report-defs/pipeline_funnel/run", headers=hdr, json={}).json()["data"]
        assert funnel["total"] == 1      # 界外候选人的申请不计
        assert funnel["open_jobs"] == 1  # 界外岗位不计

        headcount = client.post("/api/report-defs/headcount/run", headers=hdr, json={}).json()["data"]
        assert headcount["total"] == 1
        org_names = {o["org"] for o in headcount["by_org"]}
        assert "B部门" not in org_names  # 界外 org 节点连名字都不列

        weekly = client.post("/api/report-defs/weekly_summary/run", headers=hdr, json={}).json()["data"]
        assert weekly["new_applications"] == 1
        if is_feature_enabled(db_session, "hr.suite"):
            assert weekly["new_employees"] == 1

        # admin 对照:全部计入
        admin = _user(db_session, t)
        funnel = client.post("/api/report-defs/pipeline_funnel/run", headers=_hdr(admin), json={}).json()["data"]
        assert funnel["total"] == 2 and funnel["open_jobs"] == 2
    finally:
        _cleanup(db_session, t)


def test_report_defs_bad_params(db_session):
    """weekly_summary 的 days 参数是 LLM 自由输入 — 非数字回退 7,不炸 500。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.post("/api/report-defs/weekly_summary/run", headers=hdr, json={"params": {"days": "abc"}})
        assert r.status_code == 200, r.text
        assert r.json()["data"]["days"] == 7
        r2 = client.post("/api/report-defs/weekly_summary/run", headers=hdr, json={"params": {"days": 999}})
        assert r2.json()["data"]["days"] == 30  # 上限 clamp 仍在
    finally:
        _cleanup(db_session, t)


def test_update_talent_touches_last_active(db_session):
    """update_record 改候选人 → last_active_at 刷新(与手写 set_tags 同口径)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    cand = _mk_cand(db_session, t)
    assert cand.last_active_at is None
    client = TestClient(app)
    try:
        r = client.patch(f"/api/entities/talents/{cand.id}", headers=_hdr(u),
                         json={"fields": {"tags": ["重点"]}})
        assert r.status_code == 200, r.text
        db_session.refresh(cand)
        assert cand.last_active_at is not None
    finally:
        _cleanup(db_session, t)


def test_mcp_token_blank_name_rejected(db_session):
    """空白 name 不能签发(token 名必须非空白)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        r = client.post("/api/me/mcp-tokens", headers=_hdr(u), json={"name": "   "})
        assert r.status_code == 422
        r2 = client.post("/api/me/mcp-tokens", headers=_hdr(u), json={"name": "  my token  "})
        assert r2.status_code == 201
        assert r2.json()["name"] == "my token"  # 入库的是 strip 后的值
    finally:
        _cleanup(db_session, t)


def test_via_claim_whitelist(db_session):
    """via 非白名单值不落 audit.via(防未来任意 claim 误落列)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        client.post("/api/knowledge/articles", headers=_hdr(u, via="bogus_origin"), json={
            "title": "x", "content": "x", "status": "published",
        })
        row = db_session.scalar(select(m.AuditLog).where(m.AuditLog.entity_type == "knowledge_article"))
        assert row.via is None
        # 白名单值仍正常落列
        client.post("/api/knowledge/articles", headers=_hdr(u, via="assistant"), json={
            "title": "y", "content": "y", "status": "published",
        })
        rows = db_session.scalars(
            select(m.AuditLog).where(m.AuditLog.entity_type == "knowledge_article").order_by(m.AuditLog.created_at)
        ).all()
        assert rows[1].via == "assistant"
    finally:
        _cleanup(db_session, t)


# ============ review 修复回归 ============


def test_knowledge_brief_param(db_session):
    """brief=true 把 content 截成首部摘要 —— list_playbooks 工具依赖它防止
    全量正文 JSON 超出工具结果上限、靠后剧本的 id 到不了模型。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        long_content = "步骤0:" + "x" * 2000
        r = client.post("/api/knowledge/articles", headers=hdr, json={
            "title": "长剧本", "content": long_content,
            "status": "published", "is_playbook": True,
        })
        aid = r.json()["id"]
        # 默认仍是全文
        full = client.get("/api/knowledge/articles?brief=false", headers=hdr).json()
        assert full["items"][0]["content"] == long_content
        # brief → 截 300 字,其它字段照旧
        brief = client.get("/api/knowledge/articles?brief=true", headers=hdr).json()
        assert len(brief["items"][0]["content"]) == 300
        assert brief["items"][0]["id"] == aid
        assert brief["items"][0]["is_playbook"] is True
        assert brief["total"] == 1
    finally:
        _cleanup(db_session, t)


def test_entities_aggregate_matched(db_session):
    """aggregate 返回 matched 行数 —— 空集合时区分"没数据"与"聚合为 0/空"。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    emp = _mk_emp(db_session, t)
    db_session.add(m.ExpenseClaim(tenant_id=t.id, employee_id=emp.id, title="x", total=100.0, status="paid"))
    db_session.commit()
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        body = client.get("/api/entities/expenses/aggregate?agg=sum&field=total", headers=hdr).json()
        assert body["value"] == 100.0 and body["matched"] == 1
        # 无匹配 → value null + matched 0
        body2 = client.get(
            "/api/entities/expenses/aggregate?agg=sum&field=total&q=不存在词zzz", headers=hdr
        ).json()
        assert body2["value"] is None and body2["matched"] == 0
        cnt = client.get("/api/entities/expenses/aggregate?agg=count", headers=hdr).json()
        assert cnt["matched"] == 1
    finally:
        _cleanup(db_session, t)


def test_entities_expenses_perm_matches_list(db_session):
    """expenses 实体读口径与 /api/expenses/claims 一致:view_reports 进门,
    行级 scope 由 visible_employee_ids 兜(无绑定员工 → 看不到行)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t)
    viewer = _user(db_session, t, role="viewer")
    emp = _mk_emp(db_session, t)
    db_session.add(m.ExpenseClaim(tenant_id=t.id, employee_id=emp.id, title="差旅", total=50.0, status="paid"))
    db_session.commit()
    client = TestClient(app)
    try:
        # viewer 有 view_reports → 200(此前 manage_expenses 门槛会 403,与 UI 列表口径不一致)
        r = client.get("/api/entities/expenses/aggregate?agg=sum&field=total", headers=_hdr(viewer))
        assert r.status_code == 200, r.text
        # viewer 无绑定员工 → 行级 scope 为空 → matched 0(权限面不放大)
        assert r.json()["matched"] == 0
        assert client.get("/api/entities/expenses/count", headers=_hdr(viewer)).json()["count"] == 0
        # admin 全量
        r3 = client.get("/api/entities/expenses/aggregate?agg=sum&field=total", headers=_hdr(admin))
        assert r3.json()["value"] == 50.0 and r3.json()["matched"] == 1
        # employee 无任何权限位 → 仍 403
        emp_u = _user(db_session, t, role="employee")
        assert client.get(
            "/api/entities/expenses/aggregate?agg=sum&field=total", headers=_hdr(emp_u)
        ).status_code == 403
    finally:
        _cleanup(db_session, t)


def test_audit_via_filter_and_facets(db_session):
    """audit events 支持 via 筛选(assistant/mcp/human),facets 返回 vias。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        client.post("/api/knowledge/articles", headers=hdr,
                    json={"title": "人工", "content": "x", "status": "published"})
        client.post("/api/knowledge/articles", headers=_hdr(u, via="assistant"),
                    json={"title": "AI", "content": "x", "status": "published"})
        ai = client.get("/api/audit/events?via=assistant", headers=hdr).json()
        assert ai["total"] == 1 and ai["items"][0]["via"] == "assistant"
        human = client.get("/api/audit/events?via=human", headers=hdr).json()
        assert human["total"] == 1 and human["items"][0]["via"] is None
        assert client.get("/api/audit/events?via=mcp", headers=hdr).json()["total"] == 0
        assert client.get("/api/audit/events?via=bogus", headers=hdr).status_code == 422
        facets = client.get("/api/audit/facets", headers=hdr).json()
        assert facets["vias"] == ["assistant"]
    finally:
        _cleanup(db_session, t)


def test_mcp_proxy_body_limit(db_session):
    """超大 body 在鉴权/转发前就被 413 拒 —— 未认证大请求不消耗内存+转发。"""
    t = _tenant(db_session)
    client = TestClient(app)
    try:
        big = {"jsonrpc": "2.0", "id": 1, "method": "ping", "pad": "x" * 1_300_000}
        r = client.post("/api/mcp", json=big)
        assert r.status_code == 413
    finally:
        _cleanup(db_session, t)


def test_env_provider_embedding_model(db_session, monkeypatch):
    """env 兜底 provider 也带 embedding_model —— .env 配置的租户不再永远纯词面。"""
    from app.integrations.llm.registry import resolve_provider
    t = _tenant(db_session)
    try:
        monkeypatch.setattr(settings, "llm_api_key", "sk-test")
        monkeypatch.setattr(settings, "llm_embedding_model", "text-embedding-3-small")
        p = resolve_provider(db_session, tenant_id=t.id)
        assert p.source == "env" and p.embedding_model == "text-embedding-3-small"
        monkeypatch.setattr(settings, "llm_embedding_model", None)
        assert resolve_provider(db_session, tenant_id=t.id).embedding_model is None
    finally:
        _cleanup(db_session, t)
