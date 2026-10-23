"""技能库 / 岗位技能 / 候选人技能测试。

覆盖:
- 字典 CRUD + 同名 409 + 搜索过滤
- 岗位 PUT 替换语义 + name 自动入字典 + must/nice
- 候选人 PUT + level 范围 + 简历导入(import-resume)
- skill-coverage:全池 vs 管道内覆盖数
- 权限:viewer 只读;租户隔离
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-skills")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    CandidateSkill,
    Job,
    JobApplication,
    JobSkill,
    OrgUnit,
    Resume,
    Skill,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from sqlalchemy import select  # noqa: E402


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


@pytest.fixture
def client():
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"sk-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    db.refresh(t)
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


def _job_cand(db, t: Tenant):
    job = Job(tenant_id=t.id, title="后端", level="intermediate", description="d")
    cand = Candidate(tenant_id=t.id, name="李四")
    db.add_all([job, cand])
    db.commit()
    db.refresh(job)
    db.refresh(cand)
    return job, cand


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        for m in (CandidateSkill, JobSkill, JobApplication, Resume, Skill, Candidate, Job, User):
            db.query(m).filter(m.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def test_skill_dict_crud(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/skills", headers=hdr, json={"name": "Python", "category": "tech"})
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        # 同名 → 409
        r = client.post("/api/skills", headers=hdr, json={"name": "Python"})
        assert r.status_code == 409

        # 非法 category → 400
        r = client.post("/api/skills", headers=hdr, json={"name": "X", "category": "bad"})
        assert r.status_code == 400

        client.post("/api/skills", headers=hdr, json={"name": "FastAPI", "category": "tech"})
        r = client.get("/api/skills?q=py", headers=hdr)
        assert {s["name"] for s in r.json()} == {"Python"}
        r = client.get("/api/skills?category=tech", headers=hdr)
        assert len(r.json()) == 2

        r = client.put(f"/api/skills/{sid}", headers=hdr,
                       json={"name": "Python3", "category": "tech", "description": "后端语言"})
        assert r.json()["name"] == "Python3"

        r = client.delete(f"/api/skills/{sid}", headers=hdr)
        assert r.status_code == 204
        r = client.get("/api/skills", headers=hdr)
        assert [s["name"] for s in r.json()] == ["FastAPI"]
    finally:
        _cleanup(db_session, t)


def test_job_skills_put_and_auto_create(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, _c = _job_cand(db_session, t)
    hdr = _hdr(u)
    try:
        # name 自动入字典 + must/nice
        r = client.put(
            f"/api/jobs/{job.id}/skills", headers=hdr,
            json={"items": [
                {"name": "Python", "importance": "must"},
                {"name": "Redis", "importance": "nice"},
            ]},
        )
        assert r.status_code == 200, r.text
        assert len(r.json()) == 2
        assert {s["importance"] for s in r.json()} == {"must", "nice"}
        # 字典里也有了
        r2 = client.get("/api/skills", headers=hdr)
        assert {s["name"] for s in r2.json()} >= {"Python", "Redis"}

        # 替换语义:再 PUT 只剩一个
        r = client.put(
            f"/api/jobs/{job.id}/skills", headers=hdr,
            json={"items": [{"name": "Python", "importance": "must"}]},
        )
        assert len(r.json()) == 1

        # 非法 importance → 400
        r = client.put(
            f"/api/jobs/{job.id}/skills", headers=hdr,
            json={"items": [{"name": "X", "importance": "bogus"}]},
        )
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_candidate_skills_and_resume_import(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    _j, cand = _job_cand(db_session, t)
    hdr = _hdr(u)
    try:
        # 简历里带 skills 的解析数据
        db_session.add(Resume(
            tenant_id=t.id, candidate_id=cand.id, file_name="r.pdf",
            file_size=1, file_mime="application/pdf", storage_key="k",
            parse_status="done",
            parsed_data={"skills": ["Python", "Docker", "沟通能力"]},
        ))
        db_session.commit()

        r = client.post(f"/api/candidates/{cand.id}/skills/import-resume", headers=hdr)
        assert r.status_code == 200
        assert r.json() == {"imported": 3, "total": 3}
        # 再导入不重复
        r = client.post(f"/api/candidates/{cand.id}/skills/import-resume", headers=hdr)
        assert r.json()["imported"] == 0

        r = client.get(f"/api/candidates/{cand.id}/skills", headers=hdr)
        items = {s["name"]: s for s in r.json()}
        assert set(items) == {"Python", "Docker", "沟通能力"}
        assert items["Python"]["source"] == "resume"

        # PUT 整体替换(manual source + level)
        py_id = items["Python"]["skill_id"]
        r = client.put(
            f"/api/candidates/{cand.id}/skills", headers=hdr,
            json={"items": [{"skill_id": py_id, "level": 5}]},
        )
        assert len(r.json()) == 1
        assert r.json()[0]["level"] == 5
        assert r.json()[0]["source"] == "manual"

        # level 越界 → 422
        r = client.put(
            f"/api/candidates/{cand.id}/skills", headers=hdr,
            json={"items": [{"skill_id": py_id, "level": 9}]},
        )
        assert r.status_code == 422
    finally:
        _cleanup(db_session, t)


def test_skill_coverage(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, cand = _job_cand(db_session, t)
    cand2 = Candidate(tenant_id=t.id, name="王五")
    db_session.add(cand2)
    db_session.commit()
    db_session.refresh(cand2)
    hdr = _hdr(u)
    try:
        # 岗位要求 Python(must) + Go(nice)
        client.put(f"/api/jobs/{job.id}/skills", headers=hdr,
                   json={"items": [{"name": "Python", "importance": "must"},
                                    {"name": "Go", "importance": "nice"}]})
        py = db_session.scalars(select(Skill).where(Skill.name == "Python")).one()

        # cand 会 Python 且在管道;cand2 会 Python 不在管道
        db_session.add_all([
            CandidateSkill(tenant_id=t.id, candidate_id=cand.id, skill_id=py.id, level=4),
            CandidateSkill(tenant_id=t.id, candidate_id=cand2.id, skill_id=py.id, level=2),
            JobApplication(tenant_id=t.id, candidate_id=cand.id, job_id=job.id, stage="screening"),
        ])
        db_session.commit()

        r = client.get(f"/api/jobs/{job.id}/skill-coverage", headers=hdr)
        assert r.status_code == 200
        items = {x["name"]: x for x in r.json()}
        assert items["Python"]["pool_count"] == 2
        assert items["Python"]["pipeline_count"] == 1
        assert items["Go"]["pool_count"] == 0
    finally:
        _cleanup(db_session, t)


def test_permissions_and_isolation(client, db_session):
    t1 = _tenant(db_session)
    t2 = _tenant(db_session)
    v = _user(db_session, t1, role="viewer")
    admin = _user(db_session, t1)
    u2 = _user(db_session, t2)
    job, cand = _job_cand(db_session, t1)
    try:
        # viewer:读 OK,写 403
        r = client.get("/api/skills", headers=_hdr(v))
        assert r.status_code == 200
        r = client.post("/api/skills", headers=_hdr(v), json={"name": "X"})
        assert r.status_code == 403
        r = client.put(f"/api/jobs/{job.id}/skills", headers=_hdr(v),
                       json={"items": []})
        assert r.status_code == 403

        # 租户隔离:B 租户看不到 A 的技能 / 岗位技能接口 404
        client.post("/api/skills", headers=_hdr(admin), json={"name": "Secret"})
        r = client.get("/api/skills", headers=_hdr(u2))
        assert r.json() == []
        r = client.get(f"/api/jobs/{job.id}/skills", headers=_hdr(u2))
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t1, t2)


def test_update_skill_dup_name_409(client, db_session):
    """重命名撞已有名字 → 409(之前会直接 IntegrityError 500)。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        client.post("/api/skills", headers=hdr, json={"name": "A"})
        sid = client.post("/api/skills", headers=hdr, json={"name": "B"}).json()["id"]
        r = client.put(f"/api/skills/{sid}", headers=hdr,
                       json={"name": "A", "category": "other"})
        assert r.status_code == 409
        # 改成不撞的名字 → 200
        r = client.put(f"/api/skills/{sid}", headers=hdr,
                       json={"name": "B2", "category": "tech"})
        assert r.status_code == 200
    finally:
        _cleanup(db_session, t)


def test_org_scope_on_job_cand_skills(client, db_session):
    """org-A 的 hr 不能读写 org-B 岗位/候选人的技能档案。"""
    t = _tenant(db_session)
    org_a = OrgUnit(tenant_id=t.id, name="A")
    org_b = OrgUnit(tenant_id=t.id, name="B")
    db_session.add_all([org_a, org_b])
    db_session.commit()
    db_session.refresh(org_a)
    db_session.refresh(org_b)
    job = Job(tenant_id=t.id, title="B岗", level="intermediate",
              description="d", org_unit_id=org_b.id)
    cand = Candidate(tenant_id=t.id, name="B候选人", org_unit_id=org_b.id)
    db_session.add_all([job, cand])
    db_session.commit()
    db_session.refresh(job)
    db_session.refresh(cand)
    hr = _user(db_session, t, role="hr")
    hr.org_unit_id = org_a.id
    db_session.commit()
    hdr = _hdr(hr)
    try:
        r = client.put(f"/api/jobs/{job.id}/skills", headers=hdr,
                       json={"items": [{"name": "X"}]})
        assert r.status_code == 404
        r = client.get(f"/api/jobs/{job.id}/skill-coverage", headers=hdr)
        assert r.status_code == 404
        r = client.put(f"/api/candidates/{cand.id}/skills", headers=hdr,
                       json={"items": [{"name": "X"}]})
        assert r.status_code == 404
        r = client.get(f"/api/candidates/{cand.id}/skills", headers=hdr)
        assert r.status_code == 404
        r = client.post(f"/api/candidates/{cand.id}/skills/import-resume",
                        headers=hdr)
        assert r.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_employee_cannot_list_skills(client, db_session):
    """ESS 员工(零权限)不能列技能字典 — 需要 view_reports。"""
    t = _tenant(db_session)
    emp = _user(db_session, t, role="employee")
    try:
        r = client.get("/api/skills", headers=_hdr(emp))
        assert r.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_name_create_requires_manage_settings(client, db_session):
    """字典归 manage_settings 管:非管理员按 name 提交不存在的技能 → 400,
    只能引用字典已有项;import-resume 例外(简历抽取本就扩充字典)。"""
    t = _tenant(db_session)
    admin = _user(db_session, t)
    hr = _user(db_session, t, role="hr")          # write_resumes,无 manage_settings
    hm = _user(db_session, t, role="hiring_manager")  # write_jobs,无 manage_settings
    job, cand = _job_cand(db_session, t)
    hadm, hhr, hhm = _hdr(admin), _hdr(hr), _hdr(hm)
    try:
        # hiring_manager 按 name 加不存在的技能 → 400,字典不被污染
        r = client.put(f"/api/jobs/{job.id}/skills", headers=hhm,
                       json={"items": [{"name": "Rust", "importance": "must"}]})
        assert r.status_code == 400
        assert client.get("/api/skills", headers=hhm).json() == []

        # 引用字典已有项(skill_id 或 name 命中)→ OK
        client.post("/api/skills", headers=hadm, json={"name": "Python"})
        r = client.put(f"/api/jobs/{job.id}/skills", headers=hhm,
                       json={"items": [{"name": "Python", "importance": "must"}]})
        assert r.status_code == 200

        # admin 按 name → 自动入字典
        r = client.put(f"/api/jobs/{job.id}/skills", headers=hadm,
                       json={"items": [{"name": "Go", "importance": "nice"}]})
        assert r.status_code == 200
        names = {s["name"] for s in client.get("/api/skills", headers=hadm).json()}
        assert "Go" in names

        # hr 走 import-resume:简历抽取自动入字典(例外)
        db_session.add(Resume(
            tenant_id=t.id, candidate_id=cand.id, file_name="r.pdf",
            file_size=1, file_mime="application/pdf", storage_key="k",
            parse_status="done", parsed_data={"skills": ["Docker"]},
        ))
        db_session.commit()
        r = client.post(f"/api/candidates/{cand.id}/skills/import-resume", headers=hhr)
        assert r.json()["imported"] == 1

        # 候选人侧同规则:hr 按 name 新增 → 400
        r = client.put(f"/api/candidates/{cand.id}/skills", headers=hhr,
                       json={"items": [{"name": "K8s", "level": 3}]})
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t)


def test_delete_skill_in_use_guard(client, db_session):
    """被岗位/候选人引用的技能不能硬删 — CASCADE 会连坐抹掉档案。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    job, _c = _job_cand(db_session, t)
    hdr = _hdr(u)
    try:
        client.put(f"/api/jobs/{job.id}/skills", headers=hdr,
                   json={"items": [{"name": "Python", "importance": "must"}]})
        py = db_session.scalars(select(Skill).where(Skill.name == "Python")).one()
        r = client.delete(f"/api/skills/{py.id}", headers=hdr)
        assert r.status_code == 409
        assert "引用" in r.json()["detail"]
        # 解除引用后可删
        client.put(f"/api/jobs/{job.id}/skills", headers=hdr, json={"items": []})
        assert client.delete(f"/api/skills/{py.id}", headers=hdr).status_code == 204
    finally:
        _cleanup(db_session, t)
