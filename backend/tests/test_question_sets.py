"""面试题集 API + worker 任务测试。

走 ``CELERY_TASK_ALWAYS_EAGER=true`` 模式 — POST 创建后任务同步跑完,
返回的 detail 直接含 questions(模拟生产里"轮询拿到 done")。

覆盖:
1. POST 创建 → eager 跑完 → status=done + questions 数量符合
2. GET list 按 resume_id 过滤
3. GET detail 含完整 questions 字段
4. DELETE
5. POST /regen 重新生成
6. viewer 拒绝创建 / 删除 / 重新生成
7. 跨租户隔离
8. mock LLM 出题数据 schema 合规
9. 简历找不到 → 404
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-question-sets")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-question-sets"
settings.celery_task_always_eager = True

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Job,
    QuestionLibraryItem,
    QuestionSet,
    Resume,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.workers.celery_app import celery_app  # noqa: E402, F401


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


def _make_tenant(db, name_prefix: str = "qs-test") -> Tenant:
    t = Tenant(name=f"{name_prefix}-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _make_user(db, t: Tenant, role: str = "hr") -> User:
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


def _make_resume(
    db, t: Tenant, *, parsed_text: str = "Python 工程师 5 年", skills: list[str] | None = None
) -> Resume:
    cand = Candidate(tenant_id=t.id, name="测试候选人")
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id,
        candidate_id=cand.id,
        file_name="r.txt",
        file_size=len(parsed_text),
        file_mime="text/plain",
        storage_key=f"resumes/{t.id}/{uuid.uuid4().hex}.txt",
        source="upload",
        parsed_text=parsed_text,
        parsed_data={"skills": skills or ["Python"]},
        parse_status="done",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def _make_job(db, t: Tenant, u: User, *, title: str = "Python 后端工程师") -> Job:
    j = Job(
        tenant_id=t.id,
        title=title,
        level="intermediate",
        skills=["Python", "FastAPI"],
        description="后端开发",
        created_by=u.id,
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _cleanup(db, t: Tenant) -> None:
    db.query(QuestionLibraryItem).filter(QuestionLibraryItem.tenant_id == t.id).delete()
    db.query(QuestionSet).filter(QuestionSet.tenant_id == t.id).delete()
    db.query(Resume).filter(Resume.tenant_id == t.id).delete()
    db.query(Candidate).filter(Candidate.tenant_id == t.id).delete()
    db.query(Job).filter(Job.tenant_id == t.id).delete()
    db.query(User).filter(User.tenant_id == t.id).delete()
    db.delete(t)
    db.commit()


# ---------------------------- tests ----------------------------


def test_create_question_set_eager_mode_returns_done(db_session):
    """POST 创建 → eager mode 同步跑任务 → 立即拿到 done 状态 + questions。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        res = client.post(
            "/api/question-sets/",
            headers=_hdr(u),
            json={
                "resume_id": r.id,
                "level": "intermediate",
                "count": 5,
                "kinds": ["技术深度", "项目复盘"],
            },
        )
        assert res.status_code == 201, res.text
        body = res.json()
        assert body["status"] == "done"
        assert body["count"] == 5
        assert body["questions"] is not None
        assert len(body["questions"]) == 5
        first = body["questions"][0]
        # schema 合规
        assert first["question"]
        assert isinstance(first["answer_points"], list)
        assert len(first["answer_points"]) >= 1
        assert isinstance(first["dimensions"], list)
        assert first["difficulty"]
    finally:
        _cleanup(db_session, t)


def test_list_filter_by_resume_id(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r1 = _make_resume(db_session, t)
    r2 = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        client.post(
            "/api/question-sets/",
            headers=_hdr(u),
            json={"resume_id": r1.id, "count": 3},
        )
        client.post(
            "/api/question-sets/",
            headers=_hdr(u),
            json={"resume_id": r2.id, "count": 3},
        )

        # 全部
        all_ = client.get("/api/question-sets/", headers=_hdr(u)).json()
        assert all_["total"] == 2
        assert len(all_["items"]) == 2
        # 按 resume_id 过滤
        only1 = client.get(
            f"/api/question-sets/?resume_id={r1.id}", headers=_hdr(u)
        ).json()
        assert only1["total"] == 1
        assert only1["items"][0]["resume_id"] == r1.id
    finally:
        _cleanup(db_session, t)


def test_get_detail_includes_questions(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": r.id, "count": 5},
        ).json()
        detail = client.get(
            f"/api/question-sets/{created['id']}", headers=_hdr(u)
        ).json()
        assert detail["status"] == "done"
        assert isinstance(detail["questions"], list)
        assert detail["resume_file_name"] == r.file_name
        assert detail["candidate_name"]
    finally:
        _cleanup(db_session, t)


def test_delete_question_set(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": r.id, "count": 3},
        ).json()
        res = client.delete(
            f"/api/question-sets/{created['id']}", headers=_hdr(u)
        )
        assert res.status_code == 204
        # 再 GET 应当 404
        res = client.get(
            f"/api/question-sets/{created['id']}", headers=_hdr(u)
        )
        assert res.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_regen_resets_status_and_runs_again(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": r.id, "count": 3},
        ).json()
        old_finished = created["finished_at"]
        # regen
        regenerated = client.post(
            f"/api/question-sets/{created['id']}/regen", headers=_hdr(u)
        ).json()
        # eager mode 下立刻 done
        assert regenerated["status"] == "done"
        assert regenerated["finished_at"] != old_finished
    finally:
        _cleanup(db_session, t)


def test_viewer_cannot_create_or_delete(db_session):
    t = _make_tenant(db_session)
    admin = _make_user(db_session, t, role="admin")
    viewer = _make_user(db_session, t, role="viewer")
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        # admin 创建 ok
        created = client.post(
            "/api/question-sets/", headers=_hdr(admin),
            json={"resume_id": r.id, "count": 3},
        )
        assert created.status_code == 201

        # viewer 创建失败
        res = client.post(
            "/api/question-sets/", headers=_hdr(viewer),
            json={"resume_id": r.id, "count": 3},
        )
        assert res.status_code == 403

        # viewer 删除失败
        res = client.delete(
            f"/api/question-sets/{created.json()['id']}", headers=_hdr(viewer)
        )
        assert res.status_code == 403
    finally:
        _cleanup(db_session, t)


def test_cross_tenant_isolation(db_session):
    a = _make_tenant(db_session, "qs-a")
    b = _make_tenant(db_session, "qs-b")
    ua = _make_user(db_session, a)
    ub = _make_user(db_session, b)
    rb = _make_resume(db_session, b)
    client = TestClient(app)
    try:
        # tenant-A 用户尝试用 tenant-B 的 resume 创建 → 404
        res = client.post(
            "/api/question-sets/", headers=_hdr(ua),
            json={"resume_id": rb.id, "count": 3},
        )
        assert res.status_code == 404

        # tenant-B 自己创建一个,tenant-A 看不到
        client.post(
            "/api/question-sets/", headers=_hdr(ub),
            json={"resume_id": rb.id, "count": 3},
        )
        a_list = client.get("/api/question-sets/", headers=_hdr(ua)).json()
        assert a_list["items"] == []
        assert a_list["total"] == 0
    finally:
        _cleanup(db_session, a)
        _cleanup(db_session, b)


def test_resume_not_found_returns_404(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    client = TestClient(app)
    try:
        res = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": uuid.uuid4().hex, "count": 3},
        )
        assert res.status_code == 404
    finally:
        _cleanup(db_session, t)


def test_invalid_count_422(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        # count 超限
        res = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": r.id, "count": 999},
        )
        assert res.status_code == 422
        # count 负数
        res = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={"resume_id": r.id, "count": 0},
        )
        assert res.status_code == 422
    finally:
        _cleanup(db_session, t)


def test_with_job_id(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t, role="admin")
    r = _make_resume(db_session, t)
    j = _make_job(db_session, t, u)
    client = TestClient(app)
    try:
        res = client.post(
            "/api/question-sets/", headers=_hdr(u),
            json={
                "resume_id": r.id,
                "job_id": j.id,
                "level": "advanced",
                "count": 5,
            },
        )
        assert res.status_code == 201
        body = res.json()
        assert body["job_id"] == j.id
        assert body["job_title"] == j.title
        assert body["level"] == "advanced"
        # 简历题集的题目分类 = "岗位名 · 候选人姓名"
        assert all(q["category"] == f"{j.title} · 测试候选人" for q in body["questions"])
    finally:
        _cleanup(db_session, t)


# ------------------- 题目实体化:题集只是题目的集合 -------------------


def _create_set(client, u, r, count=3) -> dict:
    res = client.post(
        "/api/question-sets/", headers=_hdr(u), json={"resume_id": r.id, "count": count}
    )
    assert res.status_code == 201, res.text
    return res.json()


def test_set_questions_are_library_items(db_session):
    """出题结果直接落为题库条目,题库列表能看到所属题集,并可按题集过滤。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=4)
        assert created["question_count"] == 4
        q0 = created["questions"][0]
        assert q0["id"] and q0["question_set_id"] == created["id"]
        assert q0["source"] == "ai_generated"
        assert q0["kind"] in {"tech", "project", "scenario", "design", "soft"}
        assert q0["difficulty"] in {"initial", "intermediate", "advanced", "expert"}
        assert [q["position"] for q in created["questions"]] == [0, 1, 2, 3]

        # 再手动建一道独立题
        client.post(
            "/api/question-library/", headers=_hdr(u), json={"question": "独立题目一道"}
        )
        lib = client.get("/api/question-library/?limit=50", headers=_hdr(u)).json()
        assert lib["total"] == 5
        in_set = [i for i in lib["items"] if i["question_set_id"]]
        assert len(in_set) == 4
        assert in_set[0]["question_set_label"].startswith("测试候选人")
        standalone = [i for i in lib["items"] if not i["question_set_id"]]
        assert standalone[0]["question_set_label"] is None

        only_set = client.get(
            f"/api/question-library/?question_set_id={created['id']}", headers=_hdr(u)
        ).json()
        assert only_set["total"] == 4
        # 集合列表带实际题数
        sets = client.get("/api/question-sets/", headers=_hdr(u)).json()
        assert sets["items"][0]["question_count"] == 4
    finally:
        _cleanup(db_session, t)


def test_edit_and_delete_item_inside_set(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=3)
        item = created["questions"][1]
        res = client.put(
            f"/api/question-library/{item['id']}",
            headers=_hdr(u),
            json={"question": "改过的题干", "difficulty": "expert"},
        )
        assert res.status_code == 200
        assert res.json()["question_set_id"] == created["id"]
        assert res.json()["position"] == 1

        client.delete(f"/api/question-library/{created['questions'][0]['id']}", headers=_hdr(u))
        detail = client.get(f"/api/question-sets/{created['id']}", headers=_hdr(u)).json()
        assert detail["question_count"] == 2
        assert detail["count"] == 3  # 请求题量不变
        assert detail["questions"][0]["question"] == "改过的题干"
    finally:
        _cleanup(db_session, t)


def test_delete_set_removes_its_questions_only(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=3)
        client.post("/api/question-library/", headers=_hdr(u), json={"question": "独立题目一道"})
        assert client.delete(f"/api/question-sets/{created['id']}", headers=_hdr(u)).status_code == 204
        lib = client.get("/api/question-library/", headers=_hdr(u)).json()
        assert lib["total"] == 1
        assert lib["items"][0]["question"] == "独立题目一道"
    finally:
        _cleanup(db_session, t)


def test_regen_set_replaces_questions(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=3)
        old_ids = {q["id"] for q in created["questions"]}
        regen = client.post(f"/api/question-sets/{created['id']}/regen", headers=_hdr(u)).json()
        assert regen["question_count"] == 3
        assert not old_ids & {q["id"] for q in regen["questions"]}
        lib = client.get("/api/question-library/", headers=_hdr(u)).json()
        assert lib["total"] == 3
    finally:
        _cleanup(db_session, t)


def test_regenerate_single_item(db_session):
    """单题重新生成:内容替换,id / 所属题集 / 顺序不变;viewer 拒绝。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    viewer = _make_user(db_session, t, role="viewer")
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=3)
        item = created["questions"][2]
        res = client.post(
            f"/api/question-library/{item['id']}/regenerate",
            headers=_hdr(u),
            json={"instruction": "更贴近高并发场景"},
        )
        assert res.status_code == 200, res.text
        out = res.json()
        assert out["id"] == item["id"]
        assert out["question_set_id"] == created["id"]
        assert out["position"] == 2
        assert out["question"]
        assert len(out["answer_points"]) >= 1

        # 独立题同样可重新生成(无修改要求)
        solo = client.post(
            "/api/question-library/", headers=_hdr(u), json={"question": "独立题目一道"}
        ).json()
        res = client.post(
            f"/api/question-library/{solo['id']}/regenerate", headers=_hdr(u), json={}
        )
        assert res.status_code == 200
        assert res.json()["source"] == "ai_generated"

        res = client.post(
            f"/api/question-library/{item['id']}/regenerate", headers=_hdr(viewer), json={}
        )
        assert res.status_code == 403
    finally:
        _cleanup(db_session, t)


def _add(client, u, question, category="", kind="tech"):
    res = client.post(
        "/api/question-library/",
        headers=_hdr(u),
        json={"question": question, "category": category, "kind": kind},
    )
    assert res.status_code == 201, res.text
    return res.json()


def test_category_sets_form_at_two_standalone_questions(db_session):
    """同分类独立题目 >= 2 道自动成集;简历题集里的题不参与;分类标签去首尾空格。"""
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        _create_set(client, u, r, count=3)  # 分类 = "测试候选人",不参与分类题集
        first = _add(client, u, "Redis 持久化机制?", "后端基础")
        assert first["in_category_set"] is False
        assert client.get("/api/question-library/categories", headers=_hdr(u)).json() == []

        _add(client, u, "MySQL 索引失效场景?", " 后端基础 ", kind="scenario")
        _add(client, u, "只有一道的分类", "前端")
        cats = client.get("/api/question-library/categories", headers=_hdr(u)).json()
        assert [(c["category"], c["question_count"]) for c in cats] == [("后端基础", 2)]
        assert cats[0]["kinds"] == ["tech", "scenario"]

        # min_count=1 给输入提示用:包含单题分类
        all_cats = client.get(
            "/api/question-library/categories?min_count=1", headers=_hdr(u)
        ).json()
        assert {c["category"] for c in all_cats} == {"后端基础", "前端"}

        lib = client.get("/api/question-library/?limit=50", headers=_hdr(u)).json()
        flags = {i["question"]: i["in_category_set"] for i in lib["items"]}
        assert flags["Redis 持久化机制?"] is True
        assert flags["只有一道的分类"] is False
        assert not any(i["in_category_set"] for i in lib["items"] if i["question_set_id"])

        only = client.get(
            "/api/question-library/?standalone=true&category=后端基础", headers=_hdr(u)
        ).json()
        assert only["total"] == 2
    finally:
        _cleanup(db_session, t)


def test_category_set_rename_merge_and_delete(db_session):
    t = _make_tenant(db_session)
    u = _make_user(db_session, t)
    viewer = _make_user(db_session, t, role="viewer")
    r = _make_resume(db_session, t)
    client = TestClient(app)
    try:
        created = _create_set(client, u, r, count=2)
        _add(client, u, "题 A", "后端")
        _add(client, u, "题 B", "后端")
        _add(client, u, "题 C", "服务端")
        _add(client, u, "题 D", "服务端")

        assert client.post(
            "/api/question-library/categories/rename",
            headers=_hdr(viewer),
            json={"old": "后端", "new": "x"},
        ).status_code == 403

        # 改名成已存在的分类 → 两组合并
        res = client.post(
            "/api/question-library/categories/rename",
            headers=_hdr(u),
            json={"old": "后端", "new": "服务端"},
        )
        assert res.json() == {"updated": 2}
        cats = client.get("/api/question-library/categories", headers=_hdr(u)).json()
        assert [(c["category"], c["question_count"]) for c in cats] == [("服务端", 4)]

        assert client.post(
            "/api/question-library/categories/delete",
            headers=_hdr(viewer),
            json={"category": "服务端"},
        ).status_code == 403
        res = client.post(
            "/api/question-library/categories/delete",
            headers=_hdr(u),
            json={"category": "服务端"},
        )
        assert res.json() == {"deleted": 4}
        # 简历题集里的题不受影响
        lib = client.get("/api/question-library/", headers=_hdr(u)).json()
        assert lib["total"] == 2
        assert all(i["question_set_id"] == created["id"] for i in lib["items"])
    finally:
        _cleanup(db_session, t)


def test_category_sets_tenant_isolated(db_session):
    a = _make_tenant(db_session, "qs-a")
    b = _make_tenant(db_session, "qs-b")
    ua = _make_user(db_session, a)
    ub = _make_user(db_session, b)
    client = TestClient(app)
    try:
        _add(client, ub, "B1", "共享名")
        _add(client, ub, "B2", "共享名")
        assert client.get("/api/question-library/categories", headers=_hdr(ua)).json() == []
        res = client.post(
            "/api/question-library/categories/delete",
            headers=_hdr(ua),
            json={"category": "共享名"},
        )
        assert res.json() == {"deleted": 0}
        assert len(client.get("/api/question-library/categories", headers=_hdr(ub)).json()) == 1
    finally:
        _cleanup(db_session, a)
        _cleanup(db_session, b)


def test_regenerate_cross_tenant_404(db_session):
    a = _make_tenant(db_session, "qs-a")
    b = _make_tenant(db_session, "qs-b")
    ua = _make_user(db_session, a)
    ub = _make_user(db_session, b)
    client = TestClient(app)
    try:
        item = client.post(
            "/api/question-library/", headers=_hdr(ub), json={"question": "B 租户的题"}
        ).json()
        res = client.post(
            f"/api/question-library/{item['id']}/regenerate", headers=_hdr(ua), json={}
        )
        assert res.status_code == 404
    finally:
        _cleanup(db_session, a)
        _cleanup(db_session, b)
