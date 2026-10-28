"""Teamtailor 阶段触发器:job_application stage 条件 + 候选人动作。

覆盖:
- transition 审计 → 规则按 stage/job_id 条件命中 → notify 落库 + run 记账幂等
- email_candidate:收件人为候选人 display_email,模板 {{candidate_name}} 渲染
- add_to_group:自动入组(source=auto),二次迁移 already_member 幂等
- send_survey:自动建 SurveyLink + 候选人邮件排队(问卷链接进 body)
- 校验:候选人动作要求实体关联候选人;add_to_group/send_survey 必填 config;
  候选人动作不要求 recipients;条件字段白名单
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-st")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-st"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402


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
    t = Tenant(name=f"st-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, role="admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
        status="active",
    )
    db.add(u)
    db.commit()
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


def _job_cand(db, t: Tenant, email=None):
    job = m.Job(tenant_id=t.id, title="后端工程师", level="senior",
                description="d")
    cand = m.Candidate(tenant_id=t.id, name="王小明", display_email=email)
    db.add_all([job, cand])
    db.commit()
    return job, cand


def _app(db, t, job, cand, stage="applied") -> m.JobApplication:
    a = m.JobApplication(tenant_id=t.id, job_id=job.id,
                         candidate_id=cand.id, stage=stage)
    db.add(a)
    db.commit()
    return a


def _mk_rule(client, admin, **over) -> dict:
    body = {
        "name": "面试阶段触发器",
        "entity_type": "job_application",
        "trigger": "transition",
        "conditions": over.pop("conditions"),
        "action": over.pop("action", "notify"),
        "title_tpl": over.pop("title_tpl", "进入面试"),
        "recipients": over.pop("recipients",
                               [{"kind": "actor"}]),
        **over,
    }
    r = client.post("/api/automations", headers=_hdr(admin), json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _conds(job_id, stage="interview"):
    return [
        {"field": "stage", "op": "eq", "value": stage},
        {"field": "job_id", "op": "eq", "value": job_id},
    ]


def _transition(client, u, app_id, to_stage):
    r = client.post(f"/api/applications/{app_id}/transition",
                    headers=_hdr(u), json={"to_stage": to_stage})
    assert r.status_code == 200, r.text
    return r.json()


def _runs(db, rule_id) -> list[m.AutomationRun]:
    return db.scalars(
        select(m.AutomationRun).where(m.AutomationRun.rule_id == rule_id)
    ).all()


def test_stage_trigger_notify_fires_on_transition(db, client):
    """迁移到面试 → stage+job_id 条件命中 → 操作者收到通知 + run ok。"""
    t = _tenant(db)
    admin = _user(db, t)
    job, cand = _job_cand(db, t)
    app_row = _app(db, t, job, cand)

    rule = _mk_rule(client, admin, conditions=_conds(job.id),
                    title_tpl="{{candidate_name}} 进入 {{job_title}} 面试")
    _transition(client, admin, app_row.id, "screening")  # 非目标阶段不触发
    assert _runs(db, rule["id"]) == []

    _transition(client, admin, app_row.id, "interview")
    db.expire_all()
    runs = _runs(db, rule["id"])
    assert len(runs) == 1 and runs[0].status == "ok"
    assert runs[0].payload["title"] == "王小明 进入 后端工程师 面试"
    notes = db.scalars(
        select(m.Notification).where(
            m.Notification.tenant_id == t.id,
            m.Notification.kind == "automation",
            m.Notification.title.like("%王小明%"),
        )
    ).all()
    assert len(notes) == 1

    # 其它岗位同阶段不命中 job_id 条件
    job2 = m.Job(tenant_id=t.id, title="前端工程师", level="senior")
    db.add(job2)
    db.commit()
    app2 = _app(db, t, job2, cand)
    _transition(client, admin, app2.id, "interview")
    assert len(_runs(db, rule["id"])) == 1


def test_email_candidate_queues_candidate_email(db, client):
    """email_candidate:收件人为候选人邮箱(无需 recipients),主题模板渲染。"""
    t = _tenant(db)
    admin = _user(db, t)
    job, cand = _job_cand(db, t, email="cand@example.com")
    app_row = _app(db, t, job, cand, stage="screening")

    rule = _mk_rule(
        client, admin,
        conditions=_conds(job.id), action="email_candidate",
        recipients=[],  # 候选人动作不需要 recipients
        title_tpl="面试邀请 - {{job_title}}",
        body_tpl="你好 {{candidate_name}},请确认面试时间",
    )
    _transition(client, admin, app_row.id, "interview")
    db.expire_all()
    runs = _runs(db, rule["id"])
    assert len(runs) == 1
    run = runs[0]
    # eager 模式(全量套件其它模块打开 CELERY_TASK_ALWAYS_EAGER)下邮件
    # 同步尝试发送,无 SMTP → email_failed;隔离运行保持 queued_email
    assert run.status in ("queued_email", "email_failed", "sent")
    assert run.payload["to"] == ["cand@example.com"]
    assert run.payload["title"] == "面试邀请 - 后端工程师"
    assert "你好 王小明" in run.payload["body"]

    # 无邮箱候选人 → skipped_no_recipients 仍记账
    cand2 = m.Candidate(tenant_id=t.id, name="无邮箱")
    db.add(cand2)
    db.commit()
    app2 = _app(db, t, job, cand2)
    _transition(client, admin, app2.id, "interview")
    db.expire_all()
    runs = {r.entity_id: r for r in _runs(db, rule["id"])}
    assert runs[app2.id].status == "skipped_no_recipients"


def test_add_to_group_auto_membership(db, client):
    """add_to_group:迁移即入组 source=auto;重复进入 already_member 幂等。"""
    t = _tenant(db)
    admin = _user(db, t)
    job, cand = _job_cand(db, t)
    app_row = _app(db, t, job, cand, stage="offer")
    grp = m.CandidateGroup(tenant_id=t.id, name="Offer 池")
    db.add(grp)
    db.commit()

    rule = _mk_rule(
        client, admin,
        conditions=_conds(job.id, stage="hired"),
        action="add_to_group", recipients=[],
        action_config={"group_id": grp.id},
        title_tpl="加入 Offer 池",
    )
    _transition(client, admin, app_row.id, "hired")
    db.expire_all()
    mem = db.scalars(
        select(m.CandidateGroupMember).where(
            m.CandidateGroupMember.group_id == grp.id,
            m.CandidateGroupMember.candidate_id == cand.id,
        )
    ).first()
    assert mem is not None and mem.source == "auto"
    runs = _runs(db, rule["id"])
    assert len(runs) == 1 and runs[0].status == "ok"
    assert runs[0].payload["group_name"] == "Offer 池"


def test_send_survey_creates_link_and_email(db, client):
    """send_survey:生成一次性 SurveyLink(绑候选人)+ 邮件附填写链接。"""
    t = _tenant(db)
    admin = _user(db, t)
    job, cand = _job_cand(db, t, email="cand@example.com")
    app_row = _app(db, t, job, cand, stage="screening")
    survey = m.Survey(tenant_id=t.id, title="候选人体验 NPS",
                      status="published")
    db.add(survey)
    db.commit()

    rule = _mk_rule(
        client, admin,
        conditions=_conds(job.id), action="send_survey", recipients=[],
        action_config={"survey_id": survey.id},
        title_tpl="邀请填写体验问卷",
        body_tpl="你好 {{candidate_name}},感谢参加面试",
    )
    _transition(client, admin, app_row.id, "interview")
    db.expire_all()
    links = db.scalars(
        select(m.SurveyLink).where(
            m.SurveyLink.survey_id == survey.id,
            m.SurveyLink.candidate_id == cand.id,
        )
    ).all()
    assert len(links) == 1
    runs = _runs(db, rule["id"])
    assert len(runs) == 1
    run = runs[0]
    # eager 模式下同步发送失败(email_failed)亦视为已投递尝试
    assert run.status in ("queued_email", "email_failed", "sent")
    assert run.payload["to"] == ["cand@example.com"]
    assert "/s/" in run.payload["body"]
    assert run.payload["survey_title"] == "候选人体验 NPS"


def test_candidate_action_validation(db, client):
    """校验:实体无候选人 → 400;缺 config → 400;无关联实体字段条件 → 400。"""
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)

    # job 实体无 candidate_attr → email_candidate 拒绝
    r = client.post("/api/automations", headers=hdr, json={
        "name": "x", "entity_type": "job", "trigger": "update",
        "conditions": [], "action": "email_candidate",
        "title_tpl": "t", "recipients": [],
    })
    assert r.status_code == 400 and "候选人" in r.json()["detail"]

    # add_to_group 缺 group_id
    r = client.post("/api/automations", headers=hdr, json={
        "name": "x", "entity_type": "job_application", "trigger": "transition",
        "conditions": [], "action": "add_to_group",
        "title_tpl": "t", "recipients": [],
    })
    assert r.status_code == 400 and "group_id" in r.json()["detail"]

    # send_survey 缺 survey_id
    r = client.post("/api/automations", headers=hdr, json={
        "name": "x", "entity_type": "job_application", "trigger": "transition",
        "conditions": [], "action": "send_survey",
        "title_tpl": "t", "recipients": [],
    })
    assert r.status_code == 400 and "survey_id" in r.json()["detail"]

    # 条件字段 to_stage 不在白名单(实体字段是 stage)
    r = client.post("/api/automations", headers=hdr, json={
        "name": "x", "entity_type": "job_application", "trigger": "transition",
        "conditions": [{"field": "to_stage", "op": "eq", "value": "interview"}],
        "action": "notify", "title_tpl": "t",
        "recipients": [{"kind": "actor"}],
    })
    assert r.status_code == 400 and "白名单" in r.json()["detail"]

    # schema 透出候选人动作 + has_candidate 标记
    r = client.get("/api/automations/schema", headers=hdr)
    assert r.status_code == 200
    meta = r.json()
    assert set(meta["candidate_actions"]) == {
        "email_candidate", "add_to_group", "send_survey"}
    assert meta["entities"]["job_application"]["has_candidate"] is True
    assert meta["entities"]["job"]["has_candidate"] is False


def test_inactive_rule_not_fired(db, client):
    """停用规则不触发;重新启用后恢复。"""
    t = _tenant(db)
    admin = _user(db, t)
    job, cand = _job_cand(db, t)
    app_row = _app(db, t, job, cand)

    rule = _mk_rule(client, admin, conditions=_conds(job.id))
    r = client.patch(f"/api/automations/{rule['id']}",
                     headers=_hdr(admin), json={"is_active": False})
    assert r.status_code == 200
    _transition(client, admin, app_row.id, "interview")
    assert _runs(db, rule["id"]) == []

    r = client.patch(f"/api/automations/{rule['id']}",
                     headers=_hdr(admin), json={"is_active": True})
    assert r.status_code == 200
    _transition(client, admin, app_row.id, "offer")
    _transition(client, admin, app_row.id, "interview")
    db.expire_all()
    assert len(_runs(db, rule["id"])) == 1


def test_cross_tenant_isolation(db, client):
    """规则只在建它的租户生效。"""
    t1, t2 = _tenant(db), _tenant(db)
    a1, a2 = _user(db, t1), _user(db, t2)
    job1, cand1 = _job_cand(db, t1)
    job2, cand2 = _job_cand(db, t2)
    app1 = _app(db, t1, job1, cand1)
    app2 = _app(db, t2, job2, cand2)

    rule = _mk_rule(client, a1, conditions=_conds(job1.id))
    _transition(client, a2, app2.id, "interview")  # 其它租户迁移不触发
    assert _runs(db, rule["id"]) == []
    _transition(client, a1, app1.id, "interview")
    assert len(_runs(db, rule["id"])) == 1
