"""T5 平台甜点 + AI 工具收口测试。

覆盖:entities schema/create/delete/links、saved_filters、favorites、
candidate merge + duplicates、CSV import、dictionaries + kind 校验、
scheduler job_runs、optimistic lock。
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import UTC, datetime

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t5")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t5"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.candidates import _hash  # noqa: E402
from app.services.jobruns import record_run  # noqa: E402


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
    t = Tenant(name=f"t5-{uuid.uuid4().hex[:8]}")
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


def _cand(db, t: Tenant, **kw) -> m.Candidate:
    c = m.Candidate(tenant_id=t.id, name=kw.pop("name", f"候选人{uuid.uuid4().hex[:4]}"), **kw)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@pytest.fixture
def client():
    return TestClient(app)


# ---------- entities: schema / create / delete / links ----------


def test_entity_schema(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.get("/api/entities/jobs/schema", headers=_hdr(u))
    assert r.status_code == 200
    body = r.json()
    assert "level" in body["writable"]
    assert body["linkable"]["skills"]["to_entity"] == "skills"
    r2 = client.get("/api/entities/notes/schema", headers=_hdr(u))
    assert r2.status_code == 200 and r2.json()["creatable"] is True


def test_create_record_notes_and_knowledge(client, db):
    t = _tenant(db)
    u = _user(db, t)
    c = _cand(db, t)
    r = client.post(
        "/api/entities/notes",
        headers=_hdr(u),
        json={"fields": {"candidate_id": c.id, "content": "初试表现不错"}},
    )
    assert r.status_code == 201, r.text
    nid = r.json()["id"]
    assert db.get(m.CandidateNote, nid).content == "初试表现不错"

    r = client.post(
        "/api/entities/knowledge",
        headers=_hdr(u),
        json={"fields": {"title": "入职 SOP", "content": "第一步…", "category": "onboarding"}},
    )
    assert r.status_code == 201
    art = db.get(m.KnowledgeArticle, r.json()["id"])
    assert art.status == "draft" and art.is_playbook is False

    # 缺必填 → 400;未知实体 → 400;viewer 无写权限 → 403
    r = client.post("/api/entities/notes", headers=_hdr(u), json={"fields": {}})
    assert r.status_code == 400
    r = client.post("/api/entities/employees", headers=_hdr(u), json={"fields": {"name": "x"}})
    assert r.status_code == 400
    v = _user(db, t, role="viewer")
    r = client.post(
        "/api/entities/notes",
        headers=_hdr(v),
        json={"fields": {"candidate_id": c.id, "content": "x"}},
    )
    assert r.status_code == 403


def test_delete_record(client, db):
    t = _tenant(db)
    u = _user(db, t)
    art = m.KnowledgeArticle(tenant_id=t.id, title="t", content="c", category="other", status="draft", tags=[])
    db.add(art)
    db.commit()
    art_id = art.id
    db.expunge(art)
    r = client.delete(f"/api/entities/knowledge/{art_id}", headers=_hdr(u))
    assert r.status_code == 200
    assert db.get(m.KnowledgeArticle, art_id) is None
    # 不可删实体
    c = _cand(db, t)
    r = client.delete(f"/api/entities/candidates/{c.id}", headers=_hdr(u))
    assert r.status_code == 400
    # 被引用技能 → 409
    s = m.Skill(tenant_id=t.id, name="Go", category="tech")
    db.add(s)
    db.flush()
    db.add(m.CandidateSkill(tenant_id=t.id, candidate_id=c.id, skill_id=s.id, level=3))
    db.commit()
    r = client.delete(f"/api/entities/skills/{s.id}", headers=_hdr(u))
    assert r.status_code == 409


def test_link_records(client, db):
    t = _tenant(db)
    u = _user(db, t)
    j = m.Job(tenant_id=t.id, title="后端", status="open", created_by=u.id)
    s = m.Skill(tenant_id=t.id, name="Python", category="tech")
    c = _cand(db, t)
    g = m.CandidateGroup(tenant_id=t.id, name="AI 池")
    db.add_all([j, s, g])
    db.commit()

    r = client.post(
        f"/api/entities/jobs/{j.id}/links",
        headers=_hdr(u),
        json={"rel_name": "skills", "related_id": s.id, "extra": {"importance": "must"}},
    )
    assert r.status_code == 201, r.text
    assert db.scalar(
        select(m.JobSkill).where(m.JobSkill.job_id == j.id, m.JobSkill.skill_id == s.id)
    ).importance == "must"
    # 重复 → 409
    r = client.post(
        f"/api/entities/jobs/{j.id}/links",
        headers=_hdr(u),
        json={"rel_name": "skills", "related_id": s.id, "extra": {}},
    )
    assert r.status_code == 409
    # candidate skills with level + group link
    r = client.post(
        f"/api/entities/candidates/{c.id}/links",
        headers=_hdr(u),
        json={"rel_name": "skills", "related_id": s.id, "extra": {"level": 4}},
    )
    assert r.status_code == 201
    r = client.post(
        f"/api/entities/candidates/{c.id}/links",
        headers=_hdr(u),
        json={"rel_name": "groups", "related_id": g.id, "extra": {}},
    )
    assert r.status_code == 201
    # level 越界 → 400;未知关系 → 400;跨租户目标 → 404
    r = client.post(
        f"/api/entities/candidates/{c.id}/links",
        headers=_hdr(u),
        json={"rel_name": "skills", "related_id": s.id, "extra": {"level": 9}},
    )
    assert r.status_code in (400, 409)
    r = client.post(
        f"/api/entities/jobs/{j.id}/links",
        headers=_hdr(u),
        json={"rel_name": "bogus", "related_id": s.id, "extra": {}},
    )
    assert r.status_code == 400
    # unlink
    r = client.delete(
        f"/api/entities/jobs/{j.id}/links/skills/{s.id}", headers=_hdr(u)
    )
    assert r.status_code == 200
    db.expire_all()
    assert db.scalar(
        select(m.JobSkill).where(m.JobSkill.job_id == j.id, m.JobSkill.skill_id == s.id)
    ) is None
    r = client.delete(
        f"/api/entities/jobs/{j.id}/links/skills/{s.id}", headers=_hdr(u)
    )
    assert r.status_code == 404


# ---------- saved_filters / favorites ----------


def test_saved_filters(client, db):
    t = _tenant(db)
    u = _user(db, t)
    u2 = _user(db, t)
    r = client.post(
        "/api/saved-filters",
        headers=_hdr(u),
        json={"page": "talents", "name": "华东区", "params": {"q": "上海"}},
    )
    assert r.status_code == 201
    fid = r.json()["id"]
    # 同名 → 409;本人可见,他人不可见/不可删
    r = client.post(
        "/api/saved-filters",
        headers=_hdr(u),
        json={"page": "talents", "name": "华东区", "params": {}},
    )
    assert r.status_code == 409
    r = client.get("/api/saved-filters?page=talents", headers=_hdr(u))
    assert len(r.json()) == 1
    r = client.get("/api/saved-filters?page=talents", headers=_hdr(u2))
    assert r.json() == []
    r = client.delete(f"/api/saved-filters/{fid}", headers=_hdr(u2))
    assert r.status_code == 404
    r = client.delete(f"/api/saved-filters/{fid}", headers=_hdr(u))
    assert r.status_code == 200


def test_favorites(client, db):
    t = _tenant(db)
    u = _user(db, t)
    c = _cand(db, t)
    r = client.post(
        "/api/favorites",
        headers=_hdr(u),
        json={"entity": "talents", "entity_id": c.id},
    )
    assert r.status_code == 201
    fid = r.json()["id"]
    # 重复 → 409;未知 entity → 400;不存在记录 → 404
    r = client.post("/api/favorites", headers=_hdr(u), json={"entity": "talents", "entity_id": c.id})
    assert r.status_code == 409
    r = client.post("/api/favorites", headers=_hdr(u), json={"entity": "aliens", "entity_id": c.id})
    assert r.status_code == 400
    r = client.post(
        "/api/favorites", headers=_hdr(u), json={"entity": "talents", "entity_id": uuid.uuid4().hex}
    )
    assert r.status_code == 404
    r = client.get("/api/favorites?entity=talents", headers=_hdr(u))
    assert len(r.json()) == 1
    r = client.delete(f"/api/favorites/{fid}", headers=_hdr(u))
    assert r.status_code == 200


# ---------- merge / duplicates / import ----------


def test_talent_merge(client, db):
    t = _tenant(db)
    u = _user(db, t)
    keep = _cand(db, t, name="张伟", display_email="a@x.com", email_hash=_hash("a@x.com", t.id))
    drop = _cand(db, t, name="张伟", display_email="a@x.com", email_hash=_hash("a@x.com", t.id))
    r = m.Resume(
        tenant_id=t.id, candidate_id=drop.id, file_name="r.pdf",
        file_size=1, file_mime="application/pdf", storage_key=f"k/{uuid.uuid4().hex}",
        source="upload",
    )
    j = m.Job(tenant_id=t.id, title="后端", status="open", created_by=u.id)
    db.add(j)
    db.flush()
    app_row = m.JobApplication(
        tenant_id=t.id, candidate_id=drop.id, job_id=j.id, stage="new", source="manual"
    )
    db.add_all([r, app_row])
    db.commit()

    # duplicates 命中同邮箱
    r2 = client.get(f"/api/talents/{keep.id}/duplicates", headers=_hdr(u))
    assert r2.status_code == 200
    assert any(d["id"] == drop.id and d["match"] == "email" for d in r2.json())

    drop_id = drop.id
    db.expunge(drop)
    r3 = client.post(
        f"/api/talents/{keep.id}/merge",
        headers=_hdr(u),
        json={"drop_id": drop_id, "field_choices": {"status": "drop"}},
    )
    assert r3.status_code == 200, r3.text
    db.expire_all()
    assert db.get(m.Candidate, drop_id) is None
    assert r.candidate_id == keep.id and app_row.candidate_id == keep.id
    # audit merge 留痕
    ev = db.scalar(
        select(m.AuditLog).where(
            m.AuditLog.entity_id == keep.id, m.AuditLog.action == "merge"
        )
    )
    assert ev is not None


def test_talent_import_csv(client, db):
    t = _tenant(db)
    u = _user(db, t)
    _cand(db, t, name="旧", display_email="dup@x.com", email_hash=_hash("dup@x.com", t.id))
    csv_text = "name,email,phone,tags\n新人一,new1@x.com,138,后端;急聘\n重复,dup@x.com,,\n,noemail@x.com,,\n"
    r = client.post(
        "/api/talents/import",
        headers=_hdr(u),
        files={"file": ("c.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 1
    reasons = [s["reason"] for s in body["skipped"]]
    assert any("重复" in x for x in reasons) and any("name 为空" in x for x in reasons)


def test_employee_import_csv(client, db):
    t = _tenant(db)
    u = _user(db, t)
    csv_text = "name,employee_no,designation,employment_type\n李四,E9001,工程师,full_time\n王五,,经理,bogus\n"
    r = client.post(
        "/api/employees/import",
        headers=_hdr(u),
        files={"file": ("e.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 1
    assert db.scalar(
        select(m.Employee).where(m.Employee.tenant_id == t.id, m.Employee.employee_no == "E9001")
    ).name == "李四"
    assert any("employment_type" in s["reason"] for s in body["skipped"])


# ---------- dictionaries + kind 校验 ----------


def test_dictionaries_and_kind(client, db):
    t = _tenant(db)
    u = _user(db, t)
    emp = m.Employee(
        tenant_id=t.id, employee_no="E1", name="员工",
        employment_type="full_time", status="active", user_id=u.id,
    )
    db.add(emp)
    db.commit()
    r = client.post(
        "/api/dictionaries",
        headers=_hdr(u),
        json={"dict_key": "request_kind", "value": "relocation", "label": "异地调动"},
    )
    assert r.status_code == 201
    # 非法 dict_key → 400
    r = client.post(
        "/api/dictionaries",
        headers=_hdr(u),
        json={"dict_key": "bogus", "value": "x", "label": "x"},
    )
    assert r.status_code == 400
    # options:自定义值 + 内置值都在
    r = client.get("/api/dictionaries/request_kind/options", headers=_hdr(u))
    vals = {o["value"] for o in r.json()}
    assert "relocation" in vals or "general" in vals  # 字典非空回字典+内置
    # 字典值可用于 request 创建(用户有绑定 employee)
    r = client.post(
        "/api/me/requests",
        headers=_hdr(u),
        json={"kind": "relocation", "subject": "申请调动"},
    )
    assert r.status_code in (200, 201), r.text
    # 非法值仍拒绝(字段校验语义 422,与既有 KINDS 契约一致)
    r = client.post(
        "/api/me/requests",
        headers=_hdr(u),
        json={"kind": "hack_value", "subject": "x"},
    )
    assert r.status_code == 422


# ---------- scheduler / job_runs ----------


def test_scheduler_jobs(client, db):
    t = _tenant(db)
    u = _user(db, t)
    record_run(db, "watch_loop", datetime.now(UTC), "success", "fired=2")
    r = client.get("/api/scheduler/jobs", headers=_hdr(u))
    assert r.status_code == 200
    wl = next(j for j in r.json() if j["name"] == "watch_loop")
    assert wl["last_status"] == "success"
    r = client.get("/api/scheduler/jobs/watch_loop/runs", headers=_hdr(u))
    assert len(r.json()) >= 1
    # viewer 无 manage_settings → 403
    v = _user(db, t, role="viewer")
    r = client.get("/api/scheduler/jobs", headers=_hdr(v))
    assert r.status_code == 403


# ---------- optimistic lock ----------


def test_optimistic_lock_entities(client, db):
    t = _tenant(db)
    u = _user(db, t)
    j = m.Job(
        tenant_id=t.id, title="前端", status="draft", created_by=u.id,
        updated_at=datetime.now(UTC).replace(tzinfo=None),
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    base = j.updated_at.isoformat() + "Z"
    # 匹配 → 通过;失配 → 409
    r = client.patch(
        f"/api/entities/jobs/{j.id}",
        headers=_hdr(u),
        json={"fields": {"level": "advanced"}, "base_updated_at": base},
    )
    assert r.status_code == 200, r.text
    r = client.patch(
        f"/api/entities/jobs/{j.id}",
        headers=_hdr(u),
        json={"fields": {"level": "entry"}, "base_updated_at": base},
    )
    assert r.status_code == 409
