"""问卷测试。

覆盖:
- 创建(题目校验:类型/选项数)+ 列表 + 详情
- 状态机:draft→published→closed,逆向 400,无题不可发布
- 公开端:未发布 404;提交成功;重复提交 409;必填缺失/非法选项 400
- 结果聚合:选项计数 / 评分均值 / 文本采样 / 满意度
- 权限与租户隔离
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-survey")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import update  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    Survey,
    SurveyLink,
    SurveyQuestion,
    SurveyResponse,
    SurveyResponseAnswer,
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


@pytest.fixture
def client():
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"sv-{uuid.uuid4().hex[:8]}")
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


def _cleanup(db, *tenants: Tenant) -> None:
    for t in tenants:
        for m in (
            SurveyResponseAnswer, SurveyResponse, SurveyLink,
            SurveyQuestion, Survey, Candidate, User,
        ):
            db.query(m).filter(m.tenant_id == t.id).delete()
        db.delete(t)
    db.commit()


def _survey_body():
    return {
        "title": "候选人体验调查",
        "description": "面试后 2 分钟",
        "questions": [
            {"type": "rating", "text": "整体满意度", "required": True, "order": 0},
            {"type": "single_choice", "text": "如何知道我们?",
             "options": ["官网", "内推", "猎头"], "required": True, "order": 1},
            {"type": "long_text", "text": "其他建议", "required": False, "order": 2},
        ],
    }


def test_survey_crud_and_status_flow(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/surveys", headers=hdr, json=_survey_body())
        assert r.status_code == 201, r.text
        sid = r.json()["id"]
        assert r.json()["question_count"] == 3
        assert r.json()["status"] == "draft"

        # 无题问卷不可发布
        r2 = client.post("/api/surveys", headers=hdr,
                         json={"title": "空", "questions": []})
        sid2 = r2.json()["id"]
        r = client.post(f"/api/surveys/{sid2}/status", headers=hdr,
                        json={"status": "published"})
        assert r.status_code == 400

        # 单选 <2 选项 → 400
        r = client.post("/api/surveys", headers=hdr, json={
            "title": "x",
            "questions": [{"type": "single_choice", "text": "q", "options": ["只有一个"]}],
        })
        assert r.status_code == 400

        # draft → published → closed;published 不可改
        r = client.post(f"/api/surveys/{sid}/status", headers=hdr, json={"status": "published"})
        assert r.json()["status"] == "published"
        r = client.put(f"/api/surveys/{sid}", headers=hdr, json=_survey_body())
        assert r.status_code == 400
        r = client.post(f"/api/surveys/{sid}/status", headers=hdr, json={"status": "draft"})
        assert r.status_code == 400
        r = client.post(f"/api/surveys/{sid}/status", headers=hdr, json={"status": "closed"})
        assert r.json()["status"] == "closed"
    finally:
        _cleanup(db_session, t)


def test_public_submit_and_results(client, db_session):
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/surveys", headers=hdr, json=_survey_body())
        sid = r.json()["id"]
        q_ids = [q["id"] for q in r.json()["questions"]]
        rating_q, choice_q, text_q = q_ids

        # 未发布 → 公开端 404(链接此时也是死链)
        assert client.get(f"/api/public/surveys/{sid}").status_code == 404
        r = client.post(f"/api/surveys/{sid}/links", headers=hdr, json={})
        assert r.status_code == 201
        token = r.json()["token"]
        assert client.get(f"/api/public/surveys/{token}").status_code == 404
        client.post(f"/api/surveys/{sid}/status", headers=hdr, json={"status": "published"})

        # 公开端拿题(不含租户数据)
        r = client.get(f"/api/public/surveys/{token}")
        assert r.status_code == 200
        assert len(r.json()["questions"]) == 3
        assert "tenant_id" not in r.text

        # 再发一条链接给第二个答卷人
        token2 = client.post(
            f"/api/surveys/{sid}/links", headers=hdr, json={}
        ).json()["token"]

        # 缺必填 → 400
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": "k-" + uuid.uuid4().hex,
            "answers": {rating_q: 5},
        })
        assert r.status_code == 400

        # 非法选项 → 400
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": "k-" + uuid.uuid4().hex,
            "answers": {rating_q: 5, choice_q: "不存在"},
        })
        assert r.status_code == 400

        # 正常提交 ×2(各自一条链接)
        key = "k-" + uuid.uuid4().hex
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": key,
            "respondent_name": "候选人A",
            "happiness": 5,
            "answers": {rating_q: 5, choice_q: "内推", text_q: "流程顺畅"},
        })
        assert r.status_code == 201
        r = client.post(f"/api/public/surveys/{token2}/responses", json={
            "respondent_key": "k-" + uuid.uuid4().hex,
            "happiness": 3,
            "answers": {rating_q: 3, choice_q: "官网"},
        })
        assert r.status_code == 201

        # 重复提交:同 key → 409;同链接 → 已核销 409
        r = client.post(f"/api/public/surveys/{token2}/responses", json={
            "respondent_key": key, "answers": {},
        })
        assert r.status_code == 409
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": "k-" + uuid.uuid4().hex,
            "answers": {rating_q: 5, choice_q: "官网"},
        })
        assert r.status_code == 409

        # 聚合
        r = client.get(f"/api/surveys/{sid}/results", headers=hdr)
        assert r.status_code == 200
        data = r.json()
        assert data["response_count"] == 2
        assert data["avg_happiness"] == 4.0
        assert len(data["respondents"]) == 2
        assert "候选人A" in [r["name"] for r in data["respondents"]]
        by_q = {q["question_id"]: q for q in data["questions"]}
        assert by_q[rating_q]["avg_rating"] == 4.0
        assert by_q[choice_q]["option_counts"] == {"官网": 1, "内推": 1, "猎头": 0}
        assert by_q[text_q]["samples"] == ["流程顺畅"]
    finally:
        _cleanup(db_session, t)


def test_answer_type_validation(client, db_session):
    """答案按题型校验:rating 1-5 非布尔整数;文本必须字符串;多选不得超选项数。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/surveys", headers=hdr, json={
            "title": "校验",
            "questions": [
                {"type": "rating", "text": "评分", "required": False},
                {"type": "long_text", "text": "想法", "required": False},
                {"type": "multi_choice", "text": "多选",
                 "options": ["A", "B"], "required": False},
            ],
        })
        sid = r.json()["id"]
        rating_q, text_q, multi_q = [q["id"] for q in r.json()["questions"]]
        client.post(f"/api/surveys/{sid}/status", headers=hdr,
                    json={"status": "published"})

        # 每次校验失败不耗链接 — 链接是提交成功才核销的
        def submit(answers):
            token = client.post(
                f"/api/surveys/{sid}/links", headers=hdr, json={}
            ).json()["token"]
            return client.post(f"/api/public/surveys/{token}/responses", json={
                "respondent_key": "k-" + uuid.uuid4().hex, "answers": answers,
            })

        # bool 不是合法评分(JSON true → Python True 是 int 子类,必须挡)
        assert submit({rating_q: True}).status_code == 400
        # 越界评分
        assert submit({rating_q: 0}).status_code == 400
        assert submit({rating_q: 99}).status_code == 400
        # 文本题收到对象/数组 → 400
        assert submit({text_q: {"x": 1}}).status_code == 400
        assert submit({text_q: ["a"]}).status_code == 400
        # 多选超出选项数 → 400
        assert submit({multi_q: ["A", "B", "A"]}).status_code == 400
        # 合法提交 → 201
        assert submit({rating_q: 5, text_q: "挺好", multi_q: ["A"]}).status_code == 201
    finally:
        _cleanup(db_session, t)


def test_permissions_isolation(client, db_session):
    t1 = _tenant(db_session)
    t2 = _tenant(db_session)
    v = _user(db_session, t1, role="viewer")
    u2 = _user(db_session, t2)
    admin = _user(db_session, t1)
    try:
        r = client.post("/api/surveys", headers=_hdr(admin), json=_survey_body())
        sid = r.json()["id"]
        client.post(f"/api/surveys/{sid}/status", headers=_hdr(admin),
                    json={"status": "published"})

        # viewer 写 → 403
        r = client.post("/api/surveys", headers=_hdr(v), json=_survey_body())
        assert r.status_code == 403
        # viewer 不能发链接(要 write_interview)
        r = client.post(f"/api/surveys/{sid}/links", headers=_hdr(v), json={})
        assert r.status_code == 403
        # viewer 读结果 → 200(viewer 有 view_reports)
        r = client.get(f"/api/surveys/{sid}/results", headers=_hdr(v))
        assert r.status_code == 200
        r = client.get(f"/api/surveys/{sid}/links", headers=_hdr(v))
        assert r.status_code == 200

        # 租户隔离:B 租户管理端看不到;也发不了 A 的链接
        r = client.get(f"/api/surveys/{sid}", headers=_hdr(u2))
        assert r.status_code == 404
        r = client.post(f"/api/surveys/{sid}/links", headers=_hdr(u2), json={})
        assert r.status_code == 404
        # A 租户发的链接,公开端匿名可访问
        token = client.post(
            f"/api/surveys/{sid}/links", headers=_hdr(admin), json={}
        ).json()["token"]
        r = client.get(f"/api/public/surveys/{token}")
        assert r.status_code == 200
    finally:
        _cleanup(db_session, t1, t2)


def test_closed_survey_not_editable(client, db_session):
    """closed 问卷不可编辑 — 之前允许 PUT,会把已收答卷连坐级联删光。"""
    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    try:
        r = client.post("/api/surveys", headers=hdr, json=_survey_body())
        sid = r.json()["id"]
        q_ids = [q["id"] for q in r.json()["questions"]]
        client.post(f"/api/surveys/{sid}/status", headers=hdr,
                    json={"status": "published"})
        token = client.post(
            f"/api/surveys/{sid}/links", headers=hdr, json={}
        ).json()["token"]
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": "rk-" + uuid.uuid4().hex,
            "answers": {q_ids[0]: 5, q_ids[1]: "内推"},
        })
        assert r.status_code == 201
        client.post(f"/api/surveys/{sid}/status", headers=hdr,
                    json={"status": "closed"})

        # PUT closed → 400,数据保持原样
        r = client.put(f"/api/surveys/{sid}", headers=hdr, json={
            "title": "改题",
            "questions": [{"type": "short_text", "text": "新题"}],
        })
        assert r.status_code == 400
        r = client.get(f"/api/surveys/{sid}/results", headers=hdr)
        assert r.json()["response_count"] == 1
        by_q = {q["question_id"]: q for q in r.json()["questions"]}
        assert by_q[q_ids[0]]["answered"] == 1
        assert by_q[q_ids[0]]["avg_rating"] == 5.0
        # published 同样不可编辑
        r2 = client.post("/api/surveys", headers=hdr, json=_survey_body())
        sid2 = r2.json()["id"]
        client.post(f"/api/surveys/{sid2}/status", headers=hdr,
                    json={"status": "published"})
        assert client.put(f"/api/surveys/{sid2}", headers=hdr,
                          json=_survey_body()).status_code == 400
        # draft 仍可编辑
        r3 = client.post("/api/surveys", headers=hdr, json=_survey_body())
        sid3 = r3.json()["id"]
        assert client.put(f"/api/surveys/{sid3}", headers=hdr,
                          json=_survey_body()).status_code == 200
    finally:
        _cleanup(db_session, t)


def test_link_lifecycle(client, db_session):
    """一次性链接:状态流转 / 过期 / 吊销 / 候选人绑定 / 不可见明文。"""
    from datetime import datetime, timedelta

    t = _tenant(db_session)
    u = _user(db_session, t)
    hdr = _hdr(u)
    cand = Candidate(tenant_id=t.id, name="候选人丙")
    db_session.add(cand)
    db_session.commit()
    db_session.refresh(cand)
    try:
        r = client.post("/api/surveys", headers=hdr, json=_survey_body())
        sid = r.json()["id"]
        q_ids = [q["id"] for q in r.json()["questions"]]
        client.post(f"/api/surveys/{sid}/status", headers=hdr,
                    json={"status": "published"})

        # 绑定候选人的链接
        r = client.post(f"/api/surveys/{sid}/links", headers=hdr,
                        json={"candidate_id": cand.id})
        assert r.status_code == 201
        link = r.json()
        assert link["candidate_name"] == "候选人丙"
        token = link["token"]
        # 明文 token 不在列表接口里出现
        links = client.get(f"/api/surveys/{sid}/links", headers=hdr).json()
        assert len(links) == 1
        assert "token" not in links[0]
        assert links[0]["status"] == "active"

        # 提交一次 → 答卷绑定 candidate_id,链接变 used
        r = client.post(f"/api/public/surveys/{token}/responses", json={
            "respondent_key": "rk-" + uuid.uuid4().hex,
            "answers": {q_ids[0]: 5, q_ids[1]: "内推"},
        })
        assert r.status_code == 201
        links = client.get(f"/api/surveys/{sid}/links", headers=hdr).json()
        assert links[0]["status"] == "used"
        # 已用链接再访问 → 409
        assert client.get(f"/api/public/surveys/{token}").status_code == 409
        # results 的 respondents 带出绑定候选人
        data = client.get(f"/api/surveys/{sid}/results", headers=hdr).json()
        assert data["respondents"][0]["candidate_id"] == cand.id
        assert data["respondents"][0]["name"] == "候选人丙"

        # 已用链接不能吊销?可以 — 吊销历史记录不留有效 token
        lid = links[0]["id"]
        assert client.delete(
            f"/api/surveys/{sid}/links/{lid}", headers=hdr
        ).status_code == 204
        assert client.get(f"/api/public/surveys/{token}").status_code == 404

        # 过期链接:直接改库让 expires_at 落到过去
        token2 = client.post(
            f"/api/surveys/{sid}/links", headers=hdr, json={}
        ).json()["token"]
        db_session.execute(
            update(SurveyLink)
            .where(SurveyLink.tenant_id == t.id)
            .values(expires_at=datetime.now() - timedelta(days=1))
        )
        db_session.commit()
        r = client.get(f"/api/public/surveys/{token2}")
        assert r.status_code == 410
        links = client.get(f"/api/surveys/{sid}/links", headers=hdr).json()
        assert links[0]["status"] == "expired"

        # 绑定不存在的候选人 → 404
        r = client.post(f"/api/surveys/{sid}/links", headers=hdr,
                        json={"candidate_id": "nope"})
        assert r.status_code == 404
        # closed 问卷不能发新链接
        client.post(f"/api/surveys/{sid}/status", headers=hdr,
                    json={"status": "closed"})
        r = client.post(f"/api/surveys/{sid}/links", headers=hdr, json={})
        assert r.status_code == 400
    finally:
        _cleanup(db_session, t)
