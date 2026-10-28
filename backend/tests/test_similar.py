"""``GET /api/talents/{id}/similar`` 相似候选人召回测试(iCIMS Talent Match)。

覆盖:
1. 技能加权命中主导排序(共享技能多的排前)
2. 经验画像参与打分(职位相似/年限接近加分)
3. 银牌候选人标记 + 召回加成
4. 黑名单/归档/无简历候选人不进结果;model 自身不进结果
5. min_score / limit 参数生效
6. 权限:viewer 可读,无权限 401,跨租户 404/隔离
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-similar")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-similar"

from app.domain.models import (  # noqa: E402
    AuditLog,
    Base,
    Candidate,
    CandidateSkill,
    Job,
    JobApplication,
    JobApplicationEvent,
    Resume,
    Skill,
    Tenant,
    User,
)
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


def _make_tenant(db, prefix="similar") -> Tenant:
    t = Tenant(name=f"{prefix}-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _make_user(db, t, *, role="hr") -> User:
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


def _make_candidate(db, t, *, name, skills=None, title=None, years=5, **kw) -> Candidate:
    c = Candidate(tenant_id=t.id, name=name, **kw)
    db.add(c)
    db.flush()
    pd = {"skills": skills or []}
    if title or years:
        pd["sections"] = {
            "work_experiences": [
                {"title": title or "工程师", "company": "某厂",
                 "date_range": f"2020-至今", "years": years, "descriptions": []}
            ],
            "educations": [{"school": "某大学", "degree": "本科"}],
        }
    r = Resume(
        tenant_id=t.id,
        candidate_id=c.id,
        file_name=f"{name}.txt",
        file_size=100,
        file_mime="text/plain",
        storage_key=f"test/{uuid.uuid4().hex[:8]}.txt",
        source="upload",
        parse_status="done",
        parsed_text=f"{name} {' '.join(skills or [])} {title or ''} 工作经验{years}年",
        parsed_data=pd,
    )
    db.add(r)
    db.commit()
    db.refresh(c)
    return c


def _make_skill(db, t, name, level_target=None, cand=None):
    s = Skill(tenant_id=t.id, name=name, category="tech")
    db.add(s)
    db.flush()
    if cand is not None:
        db.add(CandidateSkill(tenant_id=t.id, candidate_id=cand.id, skill_id=s.id,
                              level=level_target or 3))
    db.commit()
    return s


def _cleanup(db, t):
    for m in (JobApplicationEvent, JobApplication, CandidateSkill, AuditLog,
              Resume, Job, Skill, Candidate, User):
        db.query(m).filter(m.tenant_id == t.id).delete(synchronize_session=False)
    db.delete(db.get(Tenant, t.id))
    db.commit()


# ============ 用例 ============


def test_similar_requires_auth(db_session):
    res = TestClient(app).get("/api/talents/x/similar")
    assert res.status_code == 401


def test_similar_ranks_by_shared_skills(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    model = _make_candidate(db_session, t, name="标杆",
                            skills=["Python", "Django", "PostgreSQL"], title="后端工程师", years=5)
    near = _make_candidate(db_session, t, name="相近",
                           skills=["Python", "Django", "PostgreSQL", "Redis"], title="后端工程师", years=6)
    far = _make_candidate(db_session, t, name="无关",
                          skills=["Photoshop", "手绘"], title="视觉设计师", years=3)
    try:
        res = TestClient(app).get(
            f"/api/talents/{model.id}/similar?limit=10", headers=_hdr(u)
        )
        assert res.status_code == 200, res.text
        data = res.json()
        ids = [x["candidate_id"] for x in data["items"]]
        assert model.id not in ids  # model 自身不进结果
        assert ids[0] == near.id, data
        near_item = data["items"][0]
        assert near_item["score"] > 60
        assert set(near_item["shared_skills"]) >= {"Python", "Django", "PostgreSQL"}
        far_item = next((x for x in data["items"] if x["candidate_id"] == far.id), None)
        if far_item:
            assert far_item["score"] < near_item["score"]
        assert data["model"]["candidate_id"] == model.id
        assert data["scanned"] >= 2
    finally:
        _cleanup(db_session, t)


def test_similar_model_skills_from_candidate_skill(db_session):
    """CandidateSkill 实体技能(带 level)比 parsed skills 权重更高。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    model = _make_candidate(db_session, t, name="标杆", skills=[], title="数据分析师", years=4)
    sk = _make_skill(db_session, t, "Python", level_target=5, cand=model)
    twin = _make_candidate(db_session, t, name="同技能",
                           skills=["Python"], title="数据分析师", years=4)
    _make_candidate(db_session, t, name="无关", skills=["电焊"], title="焊工", years=10)
    try:
        res = TestClient(app).get(f"/api/talents/{model.id}/similar", headers=_hdr(u))
        assert res.status_code == 200
        items = {x["candidate_id"]: x for x in res.json()["items"]}
        assert items[twin.id]["score"] >= 80
        assert "Python" in items[twin.id]["shared_skills"]
    finally:
        _cleanup(db_session, t)


def test_silver_medalist_flag(db_session):
    """曾到 interview/offer 后被拒 → silver=True + 召回加成。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    model = _make_candidate(db_session, t, name="标杆", skills=["Java"], title="Java工程师", years=5)
    silver = _make_candidate(db_session, t, name="银牌", skills=["Java"], title="Java工程师", years=5)
    job = Job(tenant_id=t.id, title="后端开发", skills=["Java"])
    db_session.add(job)
    db_session.flush()
    app_row = JobApplication(
        tenant_id=t.id, candidate_id=silver.id, job_id=job.id, stage="rejected",
        rejected_reason="HC 冻结",
    )
    db_session.add(app_row)
    db_session.flush()
    db_session.add(JobApplicationEvent(
        tenant_id=t.id, application_id=app_row.id,
        from_stage="interview", to_stage="offer",
    ))
    db_session.add(JobApplicationEvent(
        tenant_id=t.id, application_id=app_row.id,
        from_stage="offer", to_stage="rejected",
    ))
    db_session.commit()

    plain = _make_candidate(db_session, t, name="普通", skills=["Java"], title="Java工程师", years=5)
    try:
        res = TestClient(app).get(f"/api/talents/{model.id}/similar", headers=_hdr(u))
        assert res.status_code == 200
        items = {x["candidate_id"]: x for x in res.json()["items"]}
        assert items[silver.id]["silver"] is True
        assert items[silver.id]["silver_job_title"] == "后端开发"
        assert items[plain.id]["silver"] is False
        assert items[silver.id]["score"] > items[plain.id]["score"]
    finally:
        _cleanup(db_session, t)


def test_excludes_blacklist_archived_no_resume(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    model = _make_candidate(db_session, t, name="标杆", skills=["Go"], title="后端工程师")
    _make_candidate(db_session, t, name="黑名单", skills=["Go"], is_blacklisted=True)
    _make_candidate(db_session, t, name="归档", skills=["Go"], status="archived")
    no_resume = Candidate(tenant_id=t.id, name="无简历")
    db_session.add(no_resume)
    ok_c = _make_candidate(db_session, t, name="正常", skills=["Go"], title="后端工程师")
    db_session.commit()
    try:
        res = TestClient(app).get(f"/api/talents/{model.id}/similar", headers=_hdr(u))
        ids = [x["candidate_id"] for x in res.json()["items"]]
        assert ids == [ok_c.id]
    finally:
        _cleanup(db_session, t)


def test_min_score_and_limit(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    model = _make_candidate(db_session, t, name="标杆", skills=["Python"], title="工程师")
    _make_candidate(db_session, t, name="像", skills=["Python", "Flask"], title="工程师", years=5)
    _make_candidate(db_session, t, name="不像", skills=["厨艺"], title="厨师", years=20)
    try:
        res = TestClient(app).get(
            f"/api/talents/{model.id}/similar?min_score=50&limit=5", headers=_hdr(u)
        )
        names = [x["name"] for x in res.json()["items"]]
        assert "像" in names
        res2 = TestClient(app).get(
            f"/api/talents/{model.id}/similar?limit=1", headers=_hdr(u)
        )
        assert len(res2.json()["items"]) == 1
    finally:
        _cleanup(db_session, t)


def test_viewer_can_read_cross_tenant_404(db_session):
    t1, t2 = _make_tenant(db_session, "s1"), _make_tenant(db_session, "s2")
    u1 = _make_user(db_session, t1, role="viewer")
    model = _make_candidate(db_session, t1, name="标杆", skills=["X"])
    other = _make_candidate(db_session, t2, name="他租户", skills=["X"])
    try:
        client = TestClient(app)
        assert client.get(f"/api/talents/{model.id}/similar", headers=_hdr(u1)).status_code == 200
        # 跨租户:模型候选人 404;且他租户候选人绝不出现在结果里
        assert client.get(f"/api/talents/{other.id}/similar", headers=_hdr(u1)).status_code == 404
        ids = [x["candidate_id"] for x in
               client.get(f"/api/talents/{model.id}/similar", headers=_hdr(u1)).json()["items"]]
        assert other.id not in ids
    finally:
        _cleanup(db_session, t1)
        _cleanup(db_session, t2)
