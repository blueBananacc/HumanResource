"""T3 第二批全链路测试。

覆盖:notifications+watchers / announcements / employee_requests /
salary_bands(+offer 钩子)/ hr_cases / knowledge / report_subscriptions /
talent_campaigns / ICS 导出 / activity / entities 通用检索。
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t3")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t3"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, OrgUnit, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
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
    t = Tenant(name=f"t3-{uuid.uuid4().hex[:8]}")
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


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


T3_MODELS = (
    m.TalentCampaignSend, m.TalentCampaign,
    m.ReportSubscription, m.AiWatcher, m.Notification,
    m.HrCaseUpdate, m.HrCase, m.RequestComment, m.EmployeeRequest,
    m.SalaryBand, m.AnnouncementRead, m.Announcement, m.KnowledgeArticle,
)


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        for model in T3_MODELS:
            db.query(model).filter(model.tenant_id == t.id).delete()
        db.query(m.Employee).filter(m.Employee.tenant_id == t.id).delete()
        db.query(User).filter(User.tenant_id == t.id).delete()
        db.query(OrgUnit).filter(OrgUnit.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def _mk_emp(client, hdr, **kw) -> dict:
    body = {"name": kw.pop("name", f"员工{uuid.uuid4().hex[:4]}")}
    body.update(kw)
    res = client.post("/api/employees", headers=hdr, json=body)
    assert res.status_code == 201, res.text
    return res.json()


def _mk_ess(client, hdr, name="员工甲"):
    """建员工 + 绑账号 + 登录 → (emp, ess_headers, ess_user_email)"""
    emp = _mk_emp(client, hdr, name=name)
    email = f"ess-{uuid.uuid4().hex[:6]}@example.com"
    acct = client.post(
        f"/api/employees/{emp['id']}/account", headers=hdr, json={"email": email}
    ).json()
    token = client.post(
        "/api/auth/login", json={"email": email, "password": acct["password"]}
    ).json()["access_token"]
    return emp, {"Authorization": f"Bearer {token}"}, email


# ============ notifications + watchers ============


def test_notifications_and_watcher_flow(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.post("/api/me/watchers", headers=hdr, json={
            "kind": "candidate_stalled", "params": {"days": 1},
        })
        assert r.status_code == 201, r.text
        wid = r.json()["id"]
        # 同 kind 再建 = upsert,不重复
        r2 = client.post("/api/me/watchers", headers=hdr, json={
            "kind": "candidate_stalled", "params": {"days": 2},
        })
        assert r2.json()["id"] == wid

        # 直接插一条停滞的管道卡触发 watcher
        cand = m.Candidate(tenant_id=t.id, name="测试候选", display_email="c@x.com")
        job = m.Job(tenant_id=t.id, title="测试岗", level="junior",
                    description="d", skills=[], created_by=u.id)
        db_session.add_all([cand, job])
        db_session.flush()
        app_row = m.JobApplication(
            tenant_id=t.id, candidate_id=cand.id, job_id=job.id,
            stage="interview",
            stage_changed_at=datetime.now() - timedelta(days=5),
        )
        db_session.add(app_row)
        db_session.commit()

        from app.services.watchers import run_watchers

        fired = run_watchers(db_session)
        assert fired >= 1

        r = client.get("/api/me/notifications/unread-count", headers=hdr)
        assert r.json()["count"] >= 1
        r = client.get("/api/me/notifications", headers=hdr)
        items = r.json()
        assert any("停滞" in n["title"] for n in items)
        nid = items[0]["id"]
        client.post(f"/api/me/notifications/{nid}/read", headers=hdr)
        # 他人不可见
        u2 = _user(db_session, t, role="hr")
        assert client.post(
            f"/api/me/notifications/{nid}/read", headers=_hdr(u2)
        ).status_code == 404
        # watcher toggle + delete
        r = client.post(f"/api/me/watchers/{wid}/toggle", headers=hdr)
        assert r.json()["is_active"] is False
        assert client.delete(f"/api/me/watchers/{wid}", headers=hdr).status_code == 204
        assert client.delete(f"/api/me/watchers/{wid}", headers=hdr).status_code == 404
    finally:
        db_session.query(m.JobApplication).filter(m.JobApplication.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ announcements ============


def test_announcement_chain(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        r = client.post("/api/announcements?publish=true", headers=_hdr(admin), json={
            "title": "放假通知", "content": "下周一年会",
        })
        assert r.status_code == 201, r.text
        aid = r.json()["id"]
        assert r.json()["status"] == "published"

        # ESS 列表可见 + 未读标记
        r = client.get("/api/me/announcements", headers=ess)
        row = next(x for x in r.json() if x["id"] == aid)
        assert row["read"] is False
        client.post(f"/api/me/announcements/{aid}/read", headers=ess)
        r = client.get("/api/me/announcements", headers=ess)
        assert next(x for x in r.json() if x["id"] == aid)["read"] is True
        r = client.get(f"/api/announcements/{aid}/reads", headers=_hdr(admin))
        assert len(r.json()) == 1

        # employee 角色无管理权限
        assert client.get("/api/announcements", headers=ess).status_code == 403

        # 受众=org B 的公告,org A 员工不可见
        org_b = OrgUnit(tenant_id=t.id, name="部门B")
        db_session.add(org_b)
        db_session.commit()
        r = client.post("/api/announcements?publish=true", headers=_hdr(admin), json={
            "title": "B部门专属", "content": "x", "audience_org_unit_id": org_b.id,
        })
        aid_b = r.json()["id"]
        r = client.get("/api/me/announcements", headers=ess)
        assert all(x["id"] != aid_b for x in r.json())
        # 受众不可见的公告 read 端点也 404
        assert client.post(
            f"/api/me/announcements/{aid_b}/read", headers=ess
        ).status_code == 404

        # archive 后 ESS 不可见
        client.post(f"/api/announcements/{aid}/status", headers=_hdr(admin), params={"to": "archived"})
        r = client.get("/api/me/announcements", headers=ess)
        assert all(x["id"] != aid for x in r.json())
    finally:
        _cleanup(db_session, t)


# ============ employee_requests ============


def test_request_chain_and_notification(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        r = client.post("/api/me/requests", headers=ess, json={
            "kind": "training", "subject": "申请参加培训", "detail": "想上 FastAPI 课",
        })
        assert r.status_code == 201, r.text
        rid = r.json()["id"]

        # HR 队列可见
        r = client.get("/api/requests?status_filter=pending", headers=_hdr(hr))
        assert any(x["id"] == rid for x in r.json()["items"])
        # employee 角色访问管理端被拒
        assert client.get("/api/requests", headers=ess).status_code == 403

        # HR 留言 + 批准 → 员工收到通知
        client.post(f"/api/requests/{rid}/comments", headers=_hdr(hr), json={"body": "确认中"})
        r = client.post(f"/api/requests/{rid}/decide", headers=_hdr(hr), json={
            "action": "approved", "note": "同意",
        })
        assert r.json()["status"] == "approved"
        r = client.get("/api/me/notifications", headers=ess)
        assert any("批准" in n["title"] for n in r.json())

        # 重复 decide → 409;非 pending 取消 → 409
        assert client.post(
            f"/api/requests/{rid}/decide", headers=_hdr(hr), json={"action": "rejected"}
        ).status_code == 409
        assert client.post(f"/api/me/requests/{rid}/cancel", headers=ess).status_code == 409

        # 他人看不到我的申请(ESS 行级)
        emp2, ess2, _ = _mk_ess(client, _hdr(admin), name="员工乙")
        r = client.get("/api/me/requests", headers=ess2)
        assert all(x["id"] != rid for x in r.json())
    finally:
        _cleanup(db_session, t)


# ============ salary_bands ============


def test_salary_bands_match_and_offer_check(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        # 通用 + designation 两条带宽,验证优先级
        client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "通用带宽", "gross_min": 8000, "gross_max": 15000,
        })
        client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "后端P5", "designation": "后端工程师", "gross_min": 20000, "gross_max": 35000,
        })
        r = client.get("/api/salary-bands/match?designation=后端工程师", headers=_hdr(admin))
        bands = r.json()
        assert bands[0]["name"] == "后端P5"  # designation 精确 > 通用

        r = client.get(
            "/api/salary-bands/check?designation=后端工程师&amount=25000",
            headers=_hdr(admin),
        )
        assert r.json()["within"] is True and r.json()["band"]["name"] == "后端P5"
        r = client.get(
            "/api/salary-bands/check?designation=后端工程师&amount=50000",
            headers=_hdr(admin),
        )
        assert r.json()["within"] is False

        # 同名 → 409
        r = client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "后端P5", "gross_min": 1, "gross_max": 2,
        })
        assert r.status_code == 409

        # gross_max < gross_min → 422
        r = client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "坏带宽", "gross_min": 100, "gross_max": 50,
        })
        assert r.status_code == 422

        # offer 钩子:position=后端工程师 + salary.base 在带内 → within=true
        cand = m.Candidate(tenant_id=t.id, name="Offer候选人")
        db_session.add(cand)
        db_session.commit()
        r = client.post("/api/offers", headers=_hdr(admin), json={
            "candidate_id": cand.id, "position": "后端工程师",
            "salary": {"base": 28000, "currency": "CNY", "period": "month"},
        })
        assert r.status_code == 201, r.text
        oid = r.json()["id"]
        r = client.get(f"/api/offers/{oid}", headers=_hdr(admin))
        chk = r.json()["salary_band_check"]
        assert chk and chk["within"] is True and chk["band_name"] == "后端P5"

        # 非 manage_payroll 角色写被拒
        hr = _user(db_session, t, role="interviewer")
        r = client.post("/api/salary-bands", headers=_hdr(hr), json={
            "name": "越权", "gross_min": 1, "gross_max": 2,
        })
        assert r.status_code == 403
    finally:
        db_session.query(m.JobOffer).filter(m.JobOffer.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ hr_cases ============


def test_case_chain_and_internal_comments(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        r = client.post("/api/me/cases", headers=ess, json={
            "subject": "电脑开不了机", "kind": "it", "priority": "high",
            "description": "按电源无反应",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        assert r.json()["case_no"] == 1

        # 第二个工单 case_no 递增
        r2 = client.post("/api/me/cases", headers=ess, json={
            "subject": "门禁卡失效", "kind": "facility",
        })
        assert r2.json()["case_no"] == 2

        # HR 队列 + 指派(hr 自己受理)→ assigned
        hdr = _hdr(hr)
        r = client.post(f"/api/cases/{cid}/assign", headers=hdr, json={
            "assignee_user_id": hr.id,
        })
        assert r.json()["status"] == "assigned"
        # resolved 必须带 resolution
        r = client.post(f"/api/cases/{cid}/status", headers=hdr, json={"status": "resolved"})
        assert r.status_code == 422
        r = client.post(f"/api/cases/{cid}/status", headers=hdr, json={
            "status": "resolved", "resolution": "已更换电源线",
        })
        assert r.json()["status"] == "resolved"

        # 内部备注员工不可见
        client.post(f"/api/cases/{cid}/updates", headers=hdr, json={
            "body": "内部备注:疑似主板也有问题", "is_internal": True,
        })
        client.post(f"/api/cases/{cid}/updates", headers=hdr, json={
            "body": "已修复,请验证", "is_internal": False,
        })
        r = client.get("/api/me/cases", headers=ess)
        c = next(x for x in r.json() if x["id"] == cid)
        bodies = [u["body"] for u in c["updates"]]
        assert "已修复,请验证" in bodies
        assert not any("内部备注" in b for b in bodies)
        r = client.get(f"/api/cases/{cid}", headers=hdr)
        assert any("内部备注" in u["body"] for u in r.json()["updates"])

        # 员工收到状态通知
        r = client.get("/api/me/notifications", headers=ess)
        assert any("工单" in n["title"] for n in r.json())
    finally:
        _cleanup(db_session, t)


# ============ knowledge ============


def test_knowledge_search_and_ess(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        client.post("/api/knowledge/articles", headers=_hdr(admin), json={
            "title": "年假申请指南", "category": "benefit",
            "content": "年假在 /me 工作台申请,自动扣减额度。",
            "tags": ["年假", "假期"],
        })
        client.post("/api/knowledge/articles", headers=_hdr(admin), json={
            "title": "VPN 使用", "category": "it",
            "content": "远程办公请连公司 VPN。",
        })
        client.post("/api/knowledge/articles", headers=_hdr(admin), json={
            "title": "草稿文章", "category": "faq", "content": "未发布", "status": "draft",
        })

        # 搜索:title 命中排最前;draft 不可见
        r = client.post("/api/knowledge/search", headers=_hdr(admin), json={"q": "年假"})
        hits = r.json()
        assert hits and hits[0]["title"] == "年假申请指南"
        assert all(h["title"] != "草稿文章" for h in hits)

        # ESS 检索同样可见 published
        r = client.get("/api/me/knowledge?q=VPN", headers=ess)
        assert r.json() and r.json()[0]["title"] == "VPN 使用"
        # ESS 不可见 draft
        r = client.get("/api/me/knowledge", headers=ess)
        assert all(x["title"] != "草稿文章" for x in r.json())

        # employee 角色无管理写权限
        r = client.post("/api/knowledge/articles", headers=ess, json={
            "title": "x", "content": "y",
        })
        assert r.status_code == 403
    finally:
        _cleanup(db_session, t)


# ============ report_subscriptions ============


def test_report_subscriptions_crud(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        r = client.post("/api/me/report-subscriptions", headers=_hdr(u), json={
            "kind": "pipeline_funnel", "weekday": 0, "hour": 9,
        })
        assert r.status_code == 201, r.text
        sid = r.json()["id"]
        # upsert:同 kind 更新不重复
        r2 = client.post("/api/me/report-subscriptions", headers=_hdr(u), json={
            "kind": "pipeline_funnel", "weekday": 1, "hour": 10,
        })
        assert r2.json()["id"] == sid and r2.json()["weekday"] == 1
        # 非法 kind → 422
        r = client.post("/api/me/report-subscriptions", headers=_hdr(u), json={
            "kind": "nope", "weekday": 0, "hour": 9,
        })
        assert r.status_code == 422
        r = client.post(f"/api/me/report-subscriptions/{sid}/toggle", headers=_hdr(u))
        assert r.json()["is_active"] is False
        assert client.delete(
            f"/api/me/report-subscriptions/{sid}", headers=_hdr(u)
        ).status_code == 204
    finally:
        _cleanup(db_session, t)


# ============ talent_campaigns ============


def test_campaign_snapshot_and_send_without_smtp(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        c1 = m.Candidate(tenant_id=t.id, name="有邮箱", display_email="a@x.com")
        c2 = m.Candidate(tenant_id=t.id, name="无邮箱")
        db_session.add_all([c1, c2])
        db_session.commit()

        r = client.post("/api/campaigns", headers=_hdr(admin), json={
            "name": "Q4 激活", "filter": {"query": "a@x.com"},
            "subject": "好久不见 {{candidate_name}}",
            "body_text": "Hi {{candidate_name}},我们有个新机会…",
        })
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        sends = r.json()["sends"]
        assert len(sends) == 1 and sends[0]["status"] == "queued"

        r = client.post("/api/campaigns", headers=_hdr(admin), json={
            "name": "全部人", "filter": {},
            "subject": "s", "body_text": "b",
        })
        statuses = {s["status"] for s in r.json()["sends"]}
        assert "queued" in statuses and "skipped_no_email" in statuses

        # 无 SMTP → 全部投递失败 → 活动置 failed(可整批重试);
        # celery eager 与否由运行环境决定,直接调函数覆盖非 eager 路径;
        # eager 下 delay 已在端点内同步跑完,二次调用命中终态幂等 no-op
        from app.workers.tasks.notify import send_talent_campaign

        client.post(f"/api/campaigns/{cid}/send", headers=_hdr(admin))
        send_talent_campaign(cid)
        r = client.get(f"/api/campaigns/{cid}", headers=_hdr(admin))
        assert r.json()["status"] == "failed"
        assert r.json()["failed_count"] >= 1

        # failed → 重试:send 把 failed 的 sends 重置回 queued 再投递
        r = client.post(f"/api/campaigns/{cid}/send", headers=_hdr(admin))
        assert r.status_code == 200, r.text
        send_talent_campaign(cid)  # 再失败仍回 failed,不锁死
        r = client.get(f"/api/campaigns/{cid}", headers=_hdr(admin))
        assert r.json()["status"] == "failed"

        # cancel:queued → cancelled
        r = client.post("/api/campaigns", headers=_hdr(admin), json={
            "name": "可取消", "filter": {"query": "a@x.com"}, "subject": "s", "body_text": "b",
        })
        cid2 = r.json()["id"]
        client.post(f"/api/campaigns/{cid2}/cancel", headers=_hdr(admin))
        r = client.get(f"/api/campaigns/{cid2}", headers=_hdr(admin))
        assert r.json()["status"] == "cancelled"
        assert all(s["status"] != "queued" for s in r.json()["sends"])
    finally:
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ ICS ============


def test_ics_export(db_session):
    t = _tenant(db_session)
    u = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        room = m.Room(tenant_id=t.id, name="测试会议室", location="3F", capacity=4)
        db_session.add(room)
        db_session.flush()
        rsv = m.Reservation(
            tenant_id=t.id, kind="room", room_id=room.id,
            title="测试预约", starts_at=datetime(2026, 11, 2, 14, 0),
            ends_at=datetime(2026, 11, 2, 15, 0), created_by=u.id,
        )
        db_session.add(rsv)
        db_session.commit()

        r = client.get(f"/api/reservations/{rsv.id}/ics", headers=_hdr(u))
        assert r.status_code == 200
        assert "text/calendar" in r.headers["content-type"]
        body = r.text
        # 预约是墙钟语义 → 浮动时间输出(无 Z),日历按打开者时区显示
        assert "BEGIN:VCALENDAR" in body and "DTSTART:20261102T140000\r\n" in body
        assert "DTEND:20261102T150000\r\n" in body
        assert "DTSTART:20261102T140000Z" not in body
        assert "SUMMARY:测试预约" in body and "LOCATION:测试会议室(3F)" in body

        # interview 无预约 → UTC(Z)+ 1 小时占位
        cand = m.Candidate(tenant_id=t.id, name="面试者")
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=u.id)
        db_session.add_all([cand, job])
        db_session.flush()
        iv = m.Interview(
            tenant_id=t.id, candidate_id=cand.id, job_id=job.id,
            started_at=datetime(2026, 11, 2, 8, 0),
        )
        db_session.add(iv)
        db_session.commit()
        r = client.get(f"/api/interviews/{iv.id}/ics", headers=_hdr(u))
        assert r.status_code == 200
        assert "DTSTART:20261102T080000Z" in r.text
        assert "DTEND:20261102T090000Z" in r.text
    finally:
        db_session.query(m.Interview).filter(m.Interview.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Room).filter(m.Room.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ activity ============


def test_activity_feed(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        # 触发一条 audit(公告 create)
        client.post("/api/announcements", headers=_hdr(admin), json={
            "title": "活动流验证", "content": "x",
        })
        r = client.get("/api/activity?days=1", headers=_hdr(admin))
        items = r.json()
        assert any(i["entity_kind"] == "announcement" and "创建" in i["verb"] for i in items)
        # 无 view_reports 角色被拒
        viewer = _user(db_session, t, role="employee")
        assert client.get("/api/activity", headers=_hdr(viewer)).status_code == 403
    finally:
        _cleanup(db_session, t)


# ============ entities ============


def test_entities_search_and_perms(db_session):
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    viewer = _user(db_session, t, role="viewer")
    client = TestClient(app)
    try:
        m_cand = m.Candidate(tenant_id=t.id, name="实体搜索候选", display_email="es@x.com")
        db_session.add(m_cand)
        db_session.commit()

        r = client.get("/api/entities", headers=_hdr(admin))
        keys = [e["key"] for e in r.json()]
        assert "talents" in keys and "jobs" in keys and "cases" in keys

        r = client.get("/api/entities/talents/search?q=实体搜索", headers=_hdr(admin))
        assert any(h["label"] == "实体搜索候选" for h in r.json())
        r = client.get("/api/entities/talents/count?q=实体搜索", headers=_hdr(admin))
        assert r.json()["count"] == 1

        # 未知实体 → 404
        assert client.get("/api/entities/nope/search", headers=_hdr(admin)).status_code == 404

        # viewer 无 manage_employees → employees 实体不可搜也不在列表
        r = client.get("/api/entities", headers=_hdr(viewer))
        assert "employees" not in [e["key"] for e in r.json()]
        assert client.get(
            "/api/entities/employees/search", headers=_hdr(viewer)
        ).status_code == 403
    finally:
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ 评审回归 ============


def test_kind_vocabulary_contract(db_session):
    """前端 / assistant 工具提交的 kind 必须与模型 KINDS 词表一致。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        for k in m.EmployeeRequest.KINDS:
            r = client.post(
                "/api/me/requests", headers=ess,
                json={"kind": k, "subject": f"kind {k}"},
            )
            assert r.status_code == 201, (k, r.text)
        assert client.post(
            "/api/me/requests", headers=ess,
            json={"kind": "it", "subject": "x"},
        ).status_code == 422
        for k in m.HrCase.KINDS:
            r = client.post(
                "/api/me/cases", headers=ess,
                json={"kind": k, "subject": f"case {k}"},
            )
            assert r.status_code == 201, (k, r.text)
        assert client.post(
            "/api/me/cases", headers=ess,
            json={"kind": "benefit", "subject": "x"},
        ).status_code == 422
    finally:
        _cleanup(db_session, t)


def test_salary_band_read_requires_payroll(db_session):
    """match/check/list 是薪酬敏感数据 — 无 manage_payroll 的角色一律 403
    (employee 与 view_reports-only 的 viewer 都不行)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    viewer = _user(db_session, t, role="viewer")
    client = TestClient(app)
    try:
        client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "带宽", "gross_min": 1000, "gross_max": 2000,
        })
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        for hdr in (ess, _hdr(viewer)):
            assert client.get("/api/salary-bands/match", headers=hdr).status_code == 403
            assert client.get(
                "/api/salary-bands/check?amount=1500", headers=hdr
            ).status_code == 403
            assert client.get("/api/salary-bands", headers=hdr).status_code == 403
    finally:
        _cleanup(db_session, t)


def test_band_score_org_isolation(db_session):
    """org 专属"通用"带宽不能被当成全公司带宽命中
    (回归:_band_score 曾把 org_unit_id 非空的带宽落到 score=2 通用档)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        org_a = OrgUnit(tenant_id=t.id, name="部门A")
        org_b = OrgUnit(tenant_id=t.id, name="部门B")
        db_session.add_all([org_a, org_b])
        db_session.commit()
        client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "A部门通用", "org_unit_id": org_a.id,
            "gross_min": 30000, "gross_max": 40000,
        })
        client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "全公司通用", "gross_min": 10000, "gross_max": 20000,
        })
        # 无 org 查询 → 只能命中全公司通用,不能拿 A 的专属带宽
        r = client.get("/api/salary-bands/check?amount=35000", headers=_hdr(admin))
        assert r.json()["band"]["name"] == "全公司通用"
        # org=A → 专属带宽 score 0 优先
        r = client.get(
            f"/api/salary-bands/check?amount=35000&org_unit_id={org_a.id}",
            headers=_hdr(admin),
        )
        assert r.json()["band"]["name"] == "A部门通用"
        # org=B → A 的专属带宽不参与匹配
        r = client.get(
            f"/api/salary-bands/check?amount=35000&org_unit_id={org_b.id}",
            headers=_hdr(admin),
        )
        assert r.json()["band"]["name"] == "全公司通用"
        r = client.get(
            f"/api/salary-bands/match?org_unit_id={org_b.id}", headers=_hdr(admin)
        )
        assert all(b["name"] != "A部门通用" for b in r.json())
    finally:
        _cleanup(db_session, t)


def test_announcement_audience_includes_subtree(db_session):
    """受众=父 org 的公告对子 org 员工可见(回归:精确匹配只认同一 org)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        parent = OrgUnit(tenant_id=t.id, name="技术部")
        db_session.add(parent)
        db_session.commit()
        child = OrgUnit(tenant_id=t.id, name="后端组", parent_id=parent.id)
        db_session.add(child)
        db_session.commit()
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        row = db_session.get(m.Employee, emp["id"])
        row.org_unit_id = child.id
        db_session.commit()

        r = client.post("/api/announcements?publish=true", headers=_hdr(admin), json={
            "title": "技术部专属", "content": "x", "audience_org_unit_id": parent.id,
        })
        aid = r.json()["id"]
        r = client.get("/api/me/announcements", headers=ess)
        assert any(x["id"] == aid for x in r.json())
        assert client.post(
            f"/api/me/announcements/{aid}/read", headers=ess
        ).status_code == 204
    finally:
        _cleanup(db_session, t)


def test_hr_request_detail_includes_comments(db_session):
    """GET /api/requests/{rid} 带留言线程 — 前端 HR 详情页靠它渲染。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin))
        r = client.post("/api/me/requests", headers=ess, json={
            "kind": "general", "subject": "详情测试",
        })
        rid = r.json()["id"]
        client.post(
            f"/api/me/requests/{rid}/comments", headers=ess, json={"body": "补充材料"}
        )
        client.post(
            f"/api/requests/{rid}/comments", headers=_hdr(hr), json={"body": "收到"}
        )
        r = client.get(f"/api/requests/{rid}", headers=_hdr(hr))
        assert r.status_code == 200, r.text
        bodies = [c["body"] for c in r.json()["comments"]]
        assert "补充材料" in bodies and "收到" in bodies

        # decide 后通知 link 指到 /me/requests(不是 /me#requests)
        r = client.post(
            f"/api/requests/{rid}/decide", headers=_hdr(hr),
            json={"action": "approved"},
        )
        assert r.status_code == 200
        r = client.get("/api/me/notifications", headers=ess)
        hit = next(n for n in r.json() if n["kind"] == "request")
        assert hit["link"] == "/me/requests"
    finally:
        _cleanup(db_session, t)


def test_watcher_and_subscription_perms(db_session):
    """watcher kind 级权限 + request_pending_days 对非 HR 收敛到本人。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        emp, ess, _ = _mk_ess(client, _hdr(admin), name="员工甲")
        # employee 不能建招聘类 watcher(管道/面试数据)
        assert client.post("/api/me/watchers", headers=ess, json={
            "kind": "candidate_stalled", "params": {"days": 1},
        }).status_code == 403
        assert client.post("/api/me/watchers", headers=ess, json={
            "kind": "interview_pending_report",
        }).status_code == 403
        # 行级自收口的 watcher 可以
        r = client.post("/api/me/watchers", headers=ess, json={
            "kind": "request_pending_days", "params": {"days": 1},
        })
        assert r.status_code == 201
        wid = r.json()["id"]
        # employee 不能订阅租户级报表
        assert client.post("/api/me/report-subscriptions", headers=ess, json={
            "kind": "pipeline_funnel", "weekday": 0, "hour": 9,
        }).status_code == 403

        # 他人的超期 pending 申请不通知我
        emp2, ess2, _ = _mk_ess(client, _hdr(admin), name="员工乙")
        client.post("/api/me/requests", headers=ess2, json={
            "kind": "training", "subject": "乙的申请",
        })
        req = db_session.scalars(
            select(m.EmployeeRequest).where(m.EmployeeRequest.tenant_id == t.id)
        ).first()
        req.created_at = datetime.now() - timedelta(days=3)
        db_session.commit()
        from app.services.watchers import run_watchers

        run_watchers(db_session)
        r = client.get("/api/me/notifications", headers=ess)
        assert not any("乙的申请" in (n["body"] or "") for n in r.json())

        # 自己的超期申请 → 通知自己
        client.post("/api/me/requests", headers=ess, json={
            "kind": "training", "subject": "甲的申请",
        })
        req2 = db_session.scalars(
            select(m.EmployeeRequest).where(
                m.EmployeeRequest.tenant_id == t.id,
                m.EmployeeRequest.subject == "甲的申请",
            )
        ).first()
        req2.created_at = datetime.now() - timedelta(days=3)
        db_session.commit()
        run_watchers(db_session)
        r = client.get("/api/me/notifications", headers=ess)
        assert any("甲的申请" in (n["body"] or "") for n in r.json())
        client.delete(f"/api/me/watchers/{wid}", headers=ess)
    finally:
        _cleanup(db_session, t)


def test_request_decide_self_approval_blocked(db_session):
    """decide 与 leave/expenses 同一口径:不能审批自己的申请。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        # 给 hr 绑员工档案 → 该 hr 也能从 ESS 侧提申请
        emp = _mk_emp(client, _hdr(admin), name="HR员工")
        row = db_session.get(m.Employee, emp["id"])
        row.user_id = hr.id
        db_session.commit()
        r = client.post("/api/me/requests", headers=_hdr(hr), json={
            "kind": "training", "subject": "自己的申请",
        })
        assert r.status_code == 201
        rid = r.json()["id"]
        # 自批 → 403;另一个 hr → 200
        assert client.post(
            f"/api/requests/{rid}/decide", headers=_hdr(hr),
            json={"action": "approved"},
        ).status_code == 403
        hr2 = _user(db_session, t, role="hr")
        r = client.post(
            f"/api/requests/{rid}/decide", headers=_hdr(hr2),
            json={"action": "approved"},
        )
        assert r.status_code == 200, r.text
    finally:
        _cleanup(db_session, t)


# ============ 评审修复回归:墙钟口径(reservation/expires_at/weekday+hour)============


def test_reservation_upcoming_uses_wall_clock(db_session):
    """Reservation.starts_at 是客户端墙钟(rooms._naive 约定)— watcher 必须
    用部署时区口径的 now 判断,容器跑 UTC 也不能偏移。"""
    t = _tenant(db_session)
    u = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        hdr = _hdr(u)
        r = client.post("/api/me/watchers", headers=hdr, json={
            "kind": "reservation_upcoming", "params": {"minutes": 60},
        })
        assert r.status_code == 201, r.text
        from app.services.timeutil import local_now

        room = m.Room(tenant_id=t.id, name="墙钟会议室", capacity=4)
        db_session.add(room)
        db_session.flush()
        db_session.add(
            m.Reservation(
                tenant_id=t.id, kind="room", room_id=room.id, title="墙钟预约",
                starts_at=local_now() + timedelta(minutes=30),
                ends_at=local_now() + timedelta(minutes=60),
                created_by=u.id,
            )
        )
        db_session.commit()

        from app.services.watchers import run_watchers

        fired = run_watchers(db_session)
        assert fired >= 1
        r = client.get("/api/me/notifications", headers=hdr)
        assert any("预约即将开始" in n["title"] for n in r.json())
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Room).filter(m.Room.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_announcement_expiry_uses_wall_clock(db_session):
    """expires_at 是管理员按本地墙钟录入的(datetime-local),可见性同口径。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        _emp, ess, _ = _mk_ess(client, _hdr(admin))
        from app.services.timeutil import local_now

        for title, offset in (("已过期公告", -1), ("未过期公告", 1)):
            r = client.post(
                "/api/announcements?publish=true", headers=_hdr(admin), json={
                    "title": title, "content": "x",
                    "expires_at": (local_now() + timedelta(hours=offset)).isoformat(),
                },
            )
            assert r.status_code == 201, r.text
        r = client.get("/api/me/announcements", headers=ess)
        titles = [x["title"] for x in r.json()]
        assert "已过期公告" not in titles
        assert "未过期公告" in titles
    finally:
        _cleanup(db_session, t)


def test_due_subscriptions_local_hour(db_session, monkeypatch):
    """weekday/hour 按部署时区理解;last_sent_at(UTC)与本地日界换算后比。"""
    t = _tenant(db_session)
    u = _user(db_session, t, role="hr")
    try:
        from app.workers.tasks import notify as notify_mod

        # 固定"现在"为部署时区 2026-11-03(周二)09:30,整点边界不失真
        monkeypatch.setattr(
            notify_mod, "local_now", lambda: datetime(2026, 11, 3, 9, 30)
        )
        # 本地 11/3 00:00(CST)= UTC 11/2 16:00 —— 日界先换算回 UTC 再比
        monkeypatch.setattr(
            notify_mod, "local_day_start_utc",
            lambda: datetime(2026, 11, 2, 16, 0),
        )

        sub = m.ReportSubscription(
            tenant_id=t.id, user_id=u.id, kind="weekly_summary",
            weekday=1, hour=9, is_active=True,
        )
        other_hour = m.ReportSubscription(
            tenant_id=t.id, user_id=u.id, kind="pipeline_funnel",
            weekday=1, hour=10, is_active=True,
        )
        db_session.add_all([sub, other_hour])
        db_session.commit()

        ids = [s.id for s in notify_mod.due_subscriptions(db_session)]
        assert sub.id in ids and other_hour.id not in ids
        # 本地今天已发过(UTC 11/3 01:00,在日界之后)→ 不再命中
        sub.last_sent_at = datetime(2026, 11, 3, 1, 0)
        db_session.commit()
        assert sub.id not in [s.id for s in notify_mod.due_subscriptions(db_session)]
        # 昨天发过(UTC 11/2 15:00,本地 11/2 23:00,在日界之前)→ 今日再发
        sub.last_sent_at = datetime(2026, 11, 2, 15, 0)
        db_session.commit()
        assert sub.id in [s.id for s in notify_mod.due_subscriptions(db_session)]
    finally:
        _cleanup(db_session, t)


def test_campaign_stale_sending_finalize(db_session):
    """sending 超时且 queued=0(worker 置终态前挂掉)→ send 端点按任务尾部
    口径现场结算,不再永远 409。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        from app.services.timeutil import utc_now

        cand = m.Candidate(tenant_id=t.id, name="陈旧候选", display_email="st@x.com")
        db_session.add(cand)
        db_session.flush()
        c = m.TalentCampaign(
            tenant_id=t.id, name="陈旧活动", filter={}, subject="s", body_text="b",
            status="sending", total=1, sent_count=1,
            started_at=utc_now() - timedelta(hours=2), created_by=admin.id,
        )
        db_session.add(c)
        db_session.flush()
        db_session.add(
            m.TalentCampaignSend(
                tenant_id=t.id, campaign_id=c.id, candidate_id=cand.id,
                to_email="st@x.com", status="sent", sent_at=utc_now(),
            )
        )
        db_session.commit()
        r = client.post(f"/api/campaigns/{c.id}/send", headers=_hdr(admin))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "done" and body["sent_count"] == 1
    finally:
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_ics_escape_cr_and_interview_rsv_scope(db_session):
    """ICS 转义 CR;interview 预约(无 room/resource)按候选人 org 收敛。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        rsv = m.Reservation(
            tenant_id=t.id, kind="room", title="标题\r\n注入",
            starts_at=datetime(2026, 11, 2, 14, 0),
            ends_at=datetime(2026, 11, 2, 15, 0), created_by=admin.id,
        )
        db_session.add(rsv)
        db_session.commit()
        r = client.get(f"/api/reservations/{rsv.id}/ics", headers=_hdr(admin))
        assert r.status_code == 200
        assert "\r\n注入" not in r.text and "\\n" in r.text

        # interview 预约:候选人 org 在 viewer 范围外 → 403;范围内 → 200
        org = OrgUnit(tenant_id=t.id, name="受限组织")
        db_session.add(org)
        db_session.flush()
        cand = m.Candidate(tenant_id=t.id, name="受限候选", org_unit_id=org.id)
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=admin.id)
        db_session.add_all([cand, job])
        db_session.flush()
        iv = m.Interview(tenant_id=t.id, candidate_id=cand.id, job_id=job.id)
        db_session.add(iv)
        db_session.flush()  # id 是 flush 期 _uuid_str 生成的
        rsv_iv = m.Reservation(
            tenant_id=t.id, kind="interview", interview_id=iv.id, title="面试预约",
            starts_at=datetime(2026, 11, 3, 10, 0),
            ends_at=datetime(2026, 11, 3, 11, 0), created_by=admin.id,
        )
        db_session.add(rsv_iv)
        db_session.commit()
        viewer = _user(db_session, t, role="viewer")
        scope = client.get(f"/api/reservations/{rsv_iv.id}/ics", headers=_hdr(admin))
        assert scope.status_code == 200
        out = client.get(f"/api/reservations/{rsv_iv.id}/ics", headers=_hdr(viewer))
        assert out.status_code in (403, 404)
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Interview).filter(m.Interview.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ 评审修复回归:org scope / 权限门 / 重复投递 ============


def test_ics_perm_gate_and_resource_reservation(db_session):
    """ICS 与读列表同口径(无 view_reports → 403);Resource 无 org 维度,
    resource 预约导出不再 500。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        res_row = m.Resource(tenant_id=t.id, name="测试投影仪", type="equipment")
        db_session.add(res_row)
        db_session.flush()
        rsv = m.Reservation(
            tenant_id=t.id, kind="resource", resource_id=res_row.id,
            title="借用投影仪", starts_at=datetime(2026, 11, 2, 14, 0),
            ends_at=datetime(2026, 11, 2, 15, 0), created_by=admin.id,
        )
        db_session.add(rsv)
        db_session.commit()

        r = client.get(f"/api/reservations/{rsv.id}/ics", headers=_hdr(admin))
        assert r.status_code == 200, r.text
        assert "LOCATION:测试投影仪" in r.text

        # employee(零业务权限)→ 403;之前仅 get_current_user,拿到 UUID 即可读
        _emp, ess, _ = _mk_ess(client, _hdr(admin))
        assert client.get(
            f"/api/reservations/{rsv.id}/ics", headers=ess
        ).status_code == 403
        # interview ICS 同样收口(权限判定先于 404)
        assert client.get(
            f"/api/interviews/{rsv.id}/ics", headers=ess
        ).status_code == 403
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Resource).filter(m.Resource.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_activity_feed_org_scope(db_session):
    """活动流按 viewer org 收敛:org B 的候选人事件/audit label 对 org A
    的 scoped 用户不可见;admin 不受影响。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        org_a = OrgUnit(tenant_id=t.id, name="部门A")
        org_b = OrgUnit(tenant_id=t.id, name="部门B")
        db_session.add_all([org_a, org_b])
        db_session.flush()
        viewer = _user(db_session, t, role="viewer")
        viewer.org_unit_id = org_a.id
        cand_b = m.Candidate(
            tenant_id=t.id, name="B组候选", org_unit_id=org_b.id
        )
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=admin.id)
        db_session.add_all([cand_b, job])
        db_session.flush()
        app_row = m.JobApplication(
            tenant_id=t.id, candidate_id=cand_b.id, job_id=job.id, stage="interview",
        )
        db_session.add(app_row)
        db_session.flush()
        db_session.add(m.JobApplicationEvent(
            tenant_id=t.id, application_id=app_row.id,
            from_stage="screening", to_stage="interview",
            actor_user_id=admin.id,
        ))
        db_session.add(m.AuditLog(
            tenant_id=t.id, actor_id=admin.id, actor_email=admin.email,
            entity_type="candidate", entity_id=cand_b.id,
            action="update", detail={"name": "B组候选"},
        ))
        # 无 org 候选人 = 公共可见,scoped 用户也要能看到
        cand_pub = m.Candidate(tenant_id=t.id, name="公共候选")
        db_session.add(cand_pub)
        db_session.flush()
        db_session.add(m.AuditLog(
            tenant_id=t.id, actor_id=admin.id, actor_email=admin.email,
            entity_type="candidate", entity_id=cand_pub.id,
            action="update", detail={"name": "公共候选"},
        ))
        db_session.commit()

        items = client.get("/api/activity?days=1", headers=_hdr(viewer)).json()
        assert all("B组候选" not in (i["entity_label"] or "") for i in items)
        assert any("公共候选" in (i["entity_label"] or "") for i in items)
        # interview 类预约的标题也按候选人 org 收敛
        iv = m.Interview(tenant_id=t.id, candidate_id=cand_b.id, job_id=job.id)
        db_session.add(iv)
        db_session.flush()
        rsv_iv = m.Reservation(
            tenant_id=t.id, kind="interview", interview_id=iv.id,
            title="面试:B组候选",
            starts_at=datetime(2026, 11, 3, 10, 0),
            ends_at=datetime(2026, 11, 3, 11, 0), created_by=admin.id,
        )
        db_session.add(rsv_iv)
        db_session.commit()
        items = client.get("/api/activity?days=1", headers=_hdr(viewer)).json()
        assert all("面试:B组候选" != (i["entity_label"] or "") for i in items)
        items = client.get("/api/activity?days=1", headers=_hdr(admin)).json()
        assert any("B组候选" in (i["entity_label"] or "") for i in items)
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Interview).filter(m.Interview.tenant_id == t.id).delete()
        db_session.query(m.JobApplicationEvent).filter(
            m.JobApplicationEvent.tenant_id == t.id).delete()
        db_session.query(m.AuditLog).filter(m.AuditLog.tenant_id == t.id).delete()
        db_session.query(m.JobApplication).filter(m.JobApplication.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_watcher_candidate_stalled_org_scope(db_session):
    """candidate_stalled watcher 按 owner org 收敛:范围外候选人不推送。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        org_a = OrgUnit(tenant_id=t.id, name="部门A")
        org_b = OrgUnit(tenant_id=t.id, name="部门B")
        db_session.add_all([org_a, org_b])
        db_session.flush()
        hr = _user(db_session, t, role="hr")
        hr.org_unit_id = org_a.id
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=admin.id)
        cand_a = m.Candidate(tenant_id=t.id, name="A组候选", org_unit_id=org_a.id)
        cand_b = m.Candidate(tenant_id=t.id, name="B组候选", org_unit_id=org_b.id)
        db_session.add_all([job, cand_a, cand_b])
        db_session.flush()
        for cand in (cand_a, cand_b):
            db_session.add(m.JobApplication(
                tenant_id=t.id, candidate_id=cand.id, job_id=job.id,
                stage="interview",
                stage_changed_at=datetime.now() - timedelta(days=5),
            ))
        db_session.commit()

        hdr = _hdr(hr)
        r = client.post("/api/me/watchers", headers=hdr, json={
            "kind": "candidate_stalled", "params": {"days": 1},
        })
        assert r.status_code == 201, r.text

        from app.services.watchers import run_watchers

        run_watchers(db_session)
        bodies = [n["body"] or "" for n in
                  client.get("/api/me/notifications", headers=hdr).json()]
        assert any("A组候选" in b for b in bodies)
        assert not any("B组候选" in b for b in bodies)
    finally:
        db_session.query(m.JobApplication).filter(m.JobApplication.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_report_subscription_executes_once_per_day(db_session):
    """send_report_subscription 执行期查重:last_sent_at 已落库的并发副本
    直接跳过,不重复发邮件(任务排队超时时 watch_loop 会重复入队)。"""
    t = _tenant(db_session)
    u = _user(db_session, t, role="hr")
    try:
        sub = m.ReportSubscription(
            tenant_id=t.id, user_id=u.id, kind="weekly_summary",
            weekday=0, hour=9, is_active=True,
        )
        db_session.add(sub)
        db_session.commit()

        from app.workers.tasks.notify import send_report_subscription

        # 无 SMTP → 走到发送前失败,不置 last_sent_at
        r = send_report_subscription(sub.id)
        assert r["ok"] is False and r["reason"] == "smtp_not_configured"
        # 已发过的副本 → skipped(行锁下读到 last_sent_at)
        db_session.refresh(sub)
        sub.last_sent_at = datetime.now()
        db_session.commit()
        r = send_report_subscription(sub.id)
        assert r.get("skipped") == "sent_today"
    finally:
        _cleanup(db_session, t)


def test_entities_reservations_interview_scope(db_session):
    """entities.reservations:interview 预约按候选人 org 收敛
    (标题常含候选人名,与 ics.py 同一口径)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        org_a = OrgUnit(tenant_id=t.id, name="部门A")
        org_b = OrgUnit(tenant_id=t.id, name="部门B")
        db_session.add_all([org_a, org_b])
        db_session.flush()
        cand = m.Candidate(tenant_id=t.id, name="受限候选", org_unit_id=org_b.id)
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=admin.id)
        db_session.add_all([cand, job])
        db_session.flush()
        iv = m.Interview(tenant_id=t.id, candidate_id=cand.id, job_id=job.id)
        db_session.add(iv)
        db_session.flush()
        db_session.add(m.Reservation(
            tenant_id=t.id, kind="interview", interview_id=iv.id,
            title="面试:受限候选", starts_at=datetime(2026, 11, 3, 10, 0),
            ends_at=datetime(2026, 11, 3, 11, 0), created_by=admin.id,
        ))
        db_session.commit()

        viewer = _user(db_session, t, role="viewer")
        viewer.org_unit_id = org_a.id
        db_session.commit()

        hits = client.get(
            "/api/entities/reservations/search", headers=_hdr(viewer)
        ).json()
        assert all("受限候选" not in h["label"] for h in hits)
        hits = client.get(
            "/api/entities/reservations/search", headers=_hdr(admin)
        ).json()
        assert any("受限候选" in h["label"] for h in hits)
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Interview).filter(m.Interview.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


# ============ 评审修复回归:ICS 折行 / 唯一约束 409 / 指派门槛 ============


def test_ics_fold_multibyte_no_500(db_session):
    """RFC5545 折行不得拆开多字节字符
    (回归:中英文混合长标题 → UnicodeDecodeError → 500)。
    每个物理行 ≤75 字节(续行前导空格计入),去掉折行后内容完整。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        title = "前端组 Sprint Planning Meeting Q4 排期会议讨论会议室预留与协调"
        rsv = m.Reservation(
            tenant_id=t.id, kind="room", title=title,
            starts_at=datetime(2026, 11, 2, 14, 0),
            ends_at=datetime(2026, 11, 2, 15, 0), created_by=admin.id,
        )
        db_session.add(rsv)
        db_session.commit()
        r = client.get(f"/api/reservations/{rsv.id}/ics", headers=_hdr(admin))
        assert r.status_code == 200, r.text[:200]
        for line in r.content.split(b"\r\n"):
            assert len(line) <= 75
        # 去掉折行标记后 SUMMARY 内容完整还原
        assert title in r.text.replace("\r\n ", "")
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_interview_ics_skips_cancelled_reservation(db_session):
    """已取消预约的时段不进面试 ICS —— 回落到面试自身时间(UTC,Z 结尾)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        cand = m.Candidate(tenant_id=t.id, name="候选人")
        job = m.Job(tenant_id=t.id, title="岗", level="mid", description="d",
                    skills=[], created_by=admin.id)
        db_session.add_all([cand, job])
        db_session.flush()
        iv = m.Interview(
            tenant_id=t.id, candidate_id=cand.id, job_id=job.id,
            started_at=datetime(2026, 11, 2, 8, 0),
        )
        db_session.add(iv)
        db_session.flush()
        db_session.add(m.Reservation(
            tenant_id=t.id, kind="interview", interview_id=iv.id,
            title="面试", status="cancelled",
            starts_at=datetime(2026, 11, 3, 10, 0),
            ends_at=datetime(2026, 11, 3, 11, 0), created_by=admin.id,
        ))
        db_session.commit()
        r = client.get(f"/api/interviews/{iv.id}/ics", headers=_hdr(admin))
        assert r.status_code == 200
        assert "20261103T100000" not in r.text
        assert "DTSTART:20261102T080000Z" in r.text
    finally:
        db_session.query(m.Reservation).filter(m.Reservation.tenant_id == t.id).delete()
        db_session.query(m.Interview).filter(m.Interview.tenant_id == t.id).delete()
        db_session.query(m.Job).filter(m.Job.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_update_band_duplicate_name_409(db_session):
    """PUT 重命名撞上 (tenant_id,name) 唯一约束 → 409(回归:500 IntegrityError)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        hdr = _hdr(admin)
        assert client.post("/api/salary-bands", headers=hdr, json={
            "name": "A带", "gross_min": 1, "gross_max": 2,
        }).status_code == 201
        b = client.post("/api/salary-bands", headers=hdr, json={
            "name": "B带", "gross_min": 1, "gross_max": 2,
        }).json()
        r = client.put(f"/api/salary-bands/{b['id']}", headers=hdr, json={
            "name": "A带", "gross_min": 1, "gross_max": 3,
        })
        assert r.status_code == 409, r.text
        # 正常重命名不受影响
        r = client.put(f"/api/salary-bands/{b['id']}", headers=hdr, json={
            "name": "B带2", "gross_min": 1, "gross_max": 3,
        })
        assert r.status_code == 200, r.text
    finally:
        _cleanup(db_session, t)


def test_employer_cost_range_validated(db_session):
    """employer_cost_max < employer_cost_min → 422(与 gross 区间同口径)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        r = client.post("/api/salary-bands", headers=_hdr(admin), json={
            "name": "成本带", "gross_min": 1, "gross_max": 2,
            "employer_cost_min": 10, "employer_cost_max": 5,
        })
        assert r.status_code == 422, r.text
    finally:
        _cleanup(db_session, t)


def test_case_assign_requires_queue_permission(db_session):
    """受理人须能看工单队列(manage_employees)——派给看不到的人是无效指派;
    unassign 不再发「已受理」通知;case_open_hours 需要同权限。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    viewer = _user(db_session, t, role="viewer")
    client = TestClient(app)
    try:
        _emp, ess, _ = _mk_ess(client, _hdr(admin))
        cid = client.post("/api/me/cases", headers=ess, json={
            "subject": "工单X", "kind": "it",
        }).json()["id"]
        hdr = _hdr(hr)

        # viewer 看不到 /api/cases → 派给他无意义 → 422
        r = client.post(f"/api/cases/{cid}/assign", headers=hdr,
                        json={"assignee_user_id": viewer.id})
        assert r.status_code == 422, r.text
        # hr 受理正常
        r = client.post(f"/api/cases/{cid}/assign", headers=hdr,
                        json={"assignee_user_id": hr.id})
        assert r.status_code == 200
        # unassign → 200 且不再给提交人发「已受理」
        before = len(client.get("/api/me/notifications", headers=ess).json())
        r = client.post(f"/api/cases/{cid}/assign", headers=hdr,
                        json={"assignee_user_id": None})
        assert r.status_code == 200, r.text
        after = len(client.get("/api/me/notifications", headers=ess).json())
        assert after == before

        # case_open_hours:受理人恒为 manage_employees,员工建了也永远打不响
        r = client.post("/api/me/watchers", headers=ess, json={
            "kind": "case_open_hours", "params": {"hours": 1},
        })
        assert r.status_code == 403
        r = client.post("/api/me/watchers", headers=hdr, json={
            "kind": "case_open_hours", "params": {"hours": 1},
        })
        assert r.status_code == 201
    finally:
        _cleanup(db_session, t)


# ============ 评审修复回归 2:邮件头非法字符 / 指派通知 / 状态通知 / expires_at 墙钟 ============


def test_send_email_header_injection_is_send_error():
    """主题/收件人带 \\r\\n → SMTPSendError(而不是 ValueError 逃逸)。
    EmailMessage 对含换行的 header 抛 ValueError;调用方只接
    SMTPSendError,不转换的话批量任务会被单条脏数据中断。"""
    from app.integrations.email.smtp_sender import (
        SMTPConfig,
        SMTPSendError,
        send_email,
    )

    cfg = SMTPConfig(
        host="127.0.0.1", port=1, use_tls=False,
        username="u", password="p", from_email="noreply@x.com",
    )
    with pytest.raises(SMTPSendError):
        send_email(cfg, to="a@x.com", subject="标题\nBcc: evil@x.com", text="t")
    with pytest.raises(SMTPSendError):
        send_email(cfg, to="a@x.com\nBcc: e@x.com", subject="s", text="t")
    # 正常 subject 不受影响的断言:SMTP 不可达也应是 SMTPSendError(连接失败)
    with pytest.raises(SMTPSendError):
        send_email(cfg, to="a@x.com", subject="正常主题", text="t")


def test_campaign_send_survives_bad_recipient_data(db_session):
    """某条 send 触发非 SMTPSendError(如候选人数据异常)→ 该条记 failed,
    批次继续到终态 —— 活动不再永远卡 sending。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        from app.infra.crypto import encrypt

        # 无 SMTP 时 send 根本不投递(cfg is None → smtp_not_configured),
        # 种子一个假 SMTP 账户让任务真正走到 send_email 路径
        db_session.add(
            m.SMTPAccount(
                tenant_id=t.id, host="127.0.0.1", port=1, use_tls=False,
                username="u", password_encrypted=encrypt("p"),
                from_email="noreply@x.com",
            )
        )
        good = m.Candidate(tenant_id=t.id, name="正常候选人", display_email="ok@x.com")
        bad = m.Candidate(tenant_id=t.id, name="换行\n候选人", display_email="bad@x.com")
        db_session.add_all([good, bad])
        db_session.commit()

        r = client.post("/api/campaigns", headers=_hdr(admin), json={
            "name": "脏数据批次", "filter": {},
            "subject": "你好 {{candidate_name}}", "body_text": "正文",
        })
        cid = r.json()["id"]

        from app.workers.tasks import notify as notify_mod

        # 让 send_email 在「换行」subject 时抛非 SMTPSendError,模拟
        # 逃逸异常;正常发送仍按原实现走(无 SMTP → SMTPSendError → failed)
        real_send = notify_mod.send_email

        def flaky(cfg, *, to, subject, text, html=None):
            if "换行" in subject:
                raise RuntimeError("boom")
            return real_send(cfg, to=to, subject=subject, text=text, html=html)

        notify_mod.send_email = flaky
        try:
            client.post(f"/api/campaigns/{cid}/send", headers=_hdr(admin))
            notify_mod.send_talent_campaign(cid)
        finally:
            notify_mod.send_email = real_send

        body = client.get(f"/api/campaigns/{cid}", headers=_hdr(admin)).json()
        assert body["status"] == "failed"  # 终态,不是永远 sending
        assert all(s["status"] != "queued" for s in body["sends"])
        boom = next(s for s in body["sends"] if s["to_email"] == "bad@x.com")
        assert boom["status"] == "failed" and "RuntimeError" in (boom["error"] or "")
    finally:
        db_session.query(m.SMTPAccount).filter(m.SMTPAccount.tenant_id == t.id).delete()
        db_session.query(m.Candidate).filter(m.Candidate.tenant_id == t.id).delete()
        _cleanup(db_session, t)


def test_announcement_expires_at_tz_normalized(db_session):
    """带时区后缀的 expires_at 按墙钟语义存(丢偏移),与 local_now 比较不错位。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        r = client.post("/api/announcements", headers=_hdr(admin), json={
            "title": "带时区公告", "content": "x",
            "expires_at": "2030-01-02T10:00:00+08:00",
        })
        assert r.status_code == 201, r.text
        # +08:00 的表达墙钟是 10:00 —— 入库应是 naive 10:00,不是 UTC 02:00
        assert r.json()["expires_at"].startswith("2030-01-02T10:00")
    finally:
        _cleanup(db_session, t)


def test_case_assign_notify_only_on_change(db_session):
    """重复指派同一人不再发「已受理」通知;换人指派才发。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    hr2 = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        _emp, ess, _ = _mk_ess(client, _hdr(admin))
        cid = client.post("/api/me/cases", headers=ess, json={
            "subject": "重复指派", "kind": "it",
        }).json()["id"]
        hdr = _hdr(hr)

        def notif_count():
            return len([
                n for n in client.get("/api/me/notifications", headers=ess).json()
                if "已受理" in n["title"]
            ])

        client.post(f"/api/cases/{cid}/assign", headers=hdr,
                    json={"assignee_user_id": hr.id})
        assert notif_count() == 1
        # 同一人再指派 → 不再发
        client.post(f"/api/cases/{cid}/assign", headers=hdr,
                    json={"assignee_user_id": hr.id})
        assert notif_count() == 1
        # 换人 → 再发一条
        client.post(f"/api/cases/{cid}/assign", headers=hdr,
                    json={"assignee_user_id": hr2.id})
        assert notif_count() == 2
    finally:
        _cleanup(db_session, t)


def test_case_status_notify_only_on_change(db_session):
    """重复提交同一状态不发通知;真流转/补写 resolution 才发。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    hr = _user(db_session, t, role="hr")
    client = TestClient(app)
    try:
        _emp, ess, _ = _mk_ess(client, _hdr(admin))
        cid = client.post("/api/me/cases", headers=ess, json={
            "subject": "状态重复", "kind": "it",
        }).json()["id"]
        hdr = _hdr(hr)

        def notif_count():
            return len([
                n for n in client.get("/api/me/notifications", headers=ess).json()
                if "状态更新" in n["title"]
            ])

        client.post(f"/api/cases/{cid}/status", headers=hdr, json={"status": "in_progress"})
        assert notif_count() == 1
        # 同样 in_progress 再来一次 → 不重复通知
        client.post(f"/api/cases/{cid}/status", headers=hdr, json={"status": "in_progress"})
        assert notif_count() == 1
        # 补写 resolution(状态不变)→ 算实质进展,通知
        client.post(f"/api/cases/{cid}/status", headers=hdr,
                    json={"status": "in_progress", "resolution": "已定位电源故障"})
        assert notif_count() == 2
    finally:
        _cleanup(db_session, t)


def test_activity_label_fallback_to_kind(db_session):
    """audit detail 缺 title/name 时 label 兜底到实体中文名,不再悬空。"""
    t = _tenant(db_session)
    admin = _user(db_session, t, role="admin")
    client = TestClient(app)
    try:
        db_session.add(m.AuditLog(
            tenant_id=t.id, actor_id=admin.id, actor_email=admin.email,
            entity_type="candidate", entity_id="no-such-id",
            action="delete", detail={"publish": True},
        ))
        db_session.commit()
        items = client.get("/api/activity?days=1", headers=_hdr(admin)).json()
        hit = next(i for i in items if i["entity_kind"] == "candidate")
        assert hit["entity_label"] == "候选人"
    finally:
        db_session.query(m.AuditLog).filter(m.AuditLog.tenant_id == t.id).delete()
        _cleanup(db_session, t)
