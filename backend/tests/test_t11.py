# -*- coding: utf-8 -*-
"""T11 逐次 LLM 调用留痕 + 岗位级知识 chunks — 单元/服务/API 三层测试。

覆盖:
- 切块: 空文/装箱/硬切/40 上限
- 网关: 成功归一化/parse 重试/schema 失败/strict 开关/独立 session 留痕
- 知识源: text/file 索引/失败态/召回排序/删除/隔离
- API: 权限/跨租户/ops 过滤/stats/助手回报端点
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t11")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain import models as m  # noqa: E402,F401  # 确保新表进 metadata
from app.domain.models import (  # noqa: E402
    Base,
    Job,
    JobKnowledgeChunk,
    JobKnowledgeSource,
    ModelCallLog,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.services import job_knowledge as jk  # noqa: E402
from app.services.model_calls import write_model_call  # noqa: E402
from app.integrations.llm import provider as llm  # noqa: E402


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
    t = Tenant(name=f"t11-{uuid.uuid4().hex[:8]}")
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


def _job(db, t: Tenant) -> Job:
    j = Job(
        tenant_id=t.id,
        title=f"岗位-{uuid.uuid4().hex[:6]}",
        level="intermediate",
        status="open",
        description="负责后端服务开发与维护。",
        skills=["Python", "FastAPI"],
    )
    db.add(j)
    db.commit()
    db.refresh(j)
    return j


def _logs(tid: str) -> list[ModelCallLog]:
    s = SessionLocal()
    try:
        return list(
            s.query(ModelCallLog).filter_by(tenant_id=tid).order_by(ModelCallLog.created_at)
        )
    finally:
        s.close()


# ======================= 切块 =======================


def test_split_chunks_empty_and_short():
    assert jk.split_chunks("") == []
    assert jk.split_chunks("   \n\n  ") == []
    assert jk.split_chunks("你好世界") == ["你好世界"]


def test_split_chunks_paragraph_packing():
    paras = "\n\n".join(f"段落{i} " + "x" * 300 for i in range(5))
    chunks = jk.split_chunks(paras, size=900)
    # 每段 ~305 字 → 每块最多装 2 段(2*305+2<900,3*305+4>900)
    assert len(chunks) == 3
    assert all(len(c) <= 900 for c in chunks)


def test_split_chunks_hard_split_and_cap():
    long_text = "x" * 5000
    chunks = jk.split_chunks(long_text, size=900)
    assert chunks[0] == "x" * 900
    # 40 块上限
    huge = "y" * (900 * 60)
    assert len(jk.split_chunks(huge, size=900)) == 40


def test_split_chunks_normalizes_whitespace():
    text = "a  \t b\n\n\n\nc"
    chunks = jk.split_chunks(text)
    assert chunks == ["a b\n\nc"]


# ======================= 网关 call_llm_json =======================


def _msgs(sys_prompt="你是助手", user="hi"):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def test_gateway_success_writes_log(db, monkeypatch):
    t = _tenant(db)
    calls = []

    def fake_chat(msgs, **kw):
        calls.append(kw)
        return '{"score": 88, "extra": 1}'

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = llm.call_llm_json(
        "test.success", _msgs(), db=db, tenant_id=t.id, required=("score",)
    )
    assert out["score"] == 88
    assert len(calls) == 1 and calls[0]["response_json"] is True

    logs = _logs(t.id)
    assert len(logs) == 1
    row = logs[0]
    assert row.call_type == "test.success"
    assert row.error_kind == ""
    assert row.output["score"] == 88
    assert row.raw_output == '{"score": 88, "extra": 1}'
    assert len(row.input_digest) == 16
    assert len(row.prompt_version) == 12
    assert row.provider == "mock"


def test_gateway_normalize_applied(db, monkeypatch):
    t = _tenant(db)
    monkeypatch.setattr(llm, "chat", lambda m, **k: '{"n": "5"}')
    out = llm.call_llm_json(
        "test.norm",
        _msgs(),
        db=db,
        tenant_id=t.id,
        normalize=lambda p: {"n": int(p["n"])},
        required=("n",),
    )
    assert out == {"n": 5}


def test_gateway_parse_error_strict_retry_succeeds(db, monkeypatch):
    t = _tenant(db)
    responses = iter(["not json at all", '{"ok": true}'])
    seen = []

    def fake_chat(msgs, **kw):
        seen.append(msgs)
        return next(responses)

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = llm.call_llm_json("test.retry", _msgs(), db=db, tenant_id=t.id)
    assert out == {"ok": True}
    assert len(seen) == 2
    # 第二次调用带了严格重试提示
    assert seen[1][-1]["role"] == "system"
    assert "JSON" in seen[1][-1]["content"]
    # 且第二次 temperature=0
    logs = _logs(t.id)
    assert len(logs) == 1 and logs[0].error_kind == ""


def test_gateway_schema_error_then_raise(db, monkeypatch):
    t = _tenant(db)
    monkeypatch.setattr(llm, "chat", lambda m, **k: '{"wrong": 1}')
    with pytest.raises(llm.LLMCallError) as ei:
        llm.call_llm_json(
            "test.schema", _msgs(), db=db, tenant_id=t.id, required=("score",)
        )
    assert ei.value.error_kind == "schema_error"
    logs = _logs(t.id)
    assert len(logs) == 1
    assert logs[0].error_kind == "schema_error"
    assert "score" in logs[0].error_message


def test_gateway_llm_error_kind_and_retry(db, monkeypatch):
    t = _tenant(db)
    calls = [0]

    def boom(msgs, **kw):
        calls[0] += 1
        raise llm.LLMError("provider down")

    monkeypatch.setattr(llm, "chat", boom)
    with pytest.raises(llm.LLMCallError) as ei:
        llm.call_llm_json("test.llmerr", _msgs(), db=db, tenant_id=t.id)
    assert ei.value.error_kind == "llm_error"
    assert calls[0] == 2  # llm_error 也会严格重试一次
    logs = _logs(t.id)
    assert logs[0].error_kind == "llm_error"


def test_gateway_retry_strict_off_single_attempt(db, monkeypatch):
    t = _tenant(db)
    calls = [0]

    def fake(m, **kw):
        calls[0] += 1
        return "bad json"

    monkeypatch.setattr(llm, "chat", fake)
    with pytest.raises(llm.LLMCallError):
        llm.call_llm_json(
            "test.noretry", _msgs(), db=db, tenant_id=t.id, retry_strict=False
        )
    assert calls[0] == 1


def test_call_llm_text_logs(db, monkeypatch):
    t = _tenant(db)
    monkeypatch.setattr(llm, "chat", lambda m, **k: "开场白文本")
    out = llm.call_llm("test.text", _msgs(), db=db, tenant_id=t.id)
    assert out == "开场白文本"
    logs = _logs(t.id)
    assert logs[0].call_type == "test.text"
    assert logs[0].output["text"] == "开场白文本"


def test_model_call_log_survives_business_rollback(db):
    t = _tenant(db)
    write_model_call(
        tenant_id=t.id, call_type="test.rb", provider="mock", model="m",
        input_digest="d", latency_ms=3,
    )
    db.rollback()  # 业务会话回滚
    assert len(_logs(t.id)) == 1  # 独立 session 已提交,不受影响


# ======================= 知识源服务层 =======================


def test_text_source_ready_and_chunks(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    src = jk.create_text_source(
        db, job=j, name="团队介绍",
        content="我们负责 AI 招聘产品线。\n\n技术栈 Python/FastAPI/Postgres。",
        actor=u,
    )
    db.commit()
    assert src.status == "ready"
    assert src.chunk_count >= 1
    chunks = jk.list_chunks(db, j.id, tenant_id=t.id)
    assert chunks and chunks[0].meta["source_name"] == "团队介绍"


def test_text_source_empty_fails(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    src = jk.create_text_source(db, job=j, name="空", content="   ", actor=u)
    db.commit()
    assert src.status == "failed"
    assert src.chunk_count == 0
    assert jk.list_chunks(db, j.id, tenant_id=t.id) == []


def test_file_source_txt_indexes(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    data = "公司介绍\n\n我们成立于 2020 年,主营 HR SaaS。".encode()
    src = jk.create_file_source(
        db, job=j, name="公司介绍", file_name="intro.txt",
        mime="text/plain", storage_key="k/1.txt", file_bytes=data, actor=u,
    )
    db.commit()
    assert src.status == "ready"
    assert src.extraction_status == "ok"
    assert src.extraction_method == "txt"
    assert src.chunk_count >= 1


def test_file_source_bad_format_fails(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)

    def bad_extract(fn, mime, data, *, pdf_pages):
        raise ValueError("corrupt pdf")

    import app.services.job_knowledge as mod
    orig = mod._extract_text_by_format
    mod._extract_text_by_format = bad_extract
    try:
        src = jk.create_file_source(
            db, job=j, name="坏文件", file_name="x.pdf",
            mime="application/pdf", storage_key="k/2.pdf",
            file_bytes=b"%PDF corrupt", actor=u,
        )
    finally:
        mod._extract_text_by_format = orig
    db.commit()
    assert src.status == "failed"
    assert src.extraction_status == "failed"


def test_context_ordered_and_text_format(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    jk.create_text_source(db, job=j, name="资料A", content="内容甲。" * 30, actor=u)
    jk.create_text_source(db, job=j, name="资料B", content="内容乙。" * 30, actor=u)
    db.commit()
    ctx = jk.job_knowledge_context(db, j, limit=10)
    assert len(ctx) >= 2
    text = jk.knowledge_context_text(db, j, limit=10)
    assert text.startswith("岗位参考资料:")
    assert "资料A" in text and "资料B" in text
    # 无知识的岗位返回空
    j2 = _job(db, t)
    assert jk.job_knowledge_context(db, j2) == []
    assert jk.knowledge_context_text(db, j2) == ""


def test_context_limit_and_max_chars(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    jk.create_text_source(db, job=j, name="长文", content="段 " + "x" * 800 + "\n\n" + "段二 " + "y" * 800 + "\n\n" + "段三 " + "z" * 800, actor=u)
    db.commit()
    ctx = jk.job_knowledge_context(db, j, limit=1)
    assert len(ctx) == 1
    text = jk.knowledge_context_text(db, j, limit=10, max_chars=100)
    # max_chars 内能装下就装,装不下整块跳过
    assert len(text) < 100 + 100


def test_delete_source_removes_chunks(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    src = jk.create_text_source(db, job=j, name="删", content="内容", actor=u)
    db.commit()
    assert jk.list_chunks(db, j.id, tenant_id=t.id)
    jk.delete_source(db, src, actor=u)
    db.commit()
    assert jk.list_chunks(db, j.id, tenant_id=t.id) == []


def test_reindex_replaces_chunks(db):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    src = jk.create_text_source(db, job=j, name="源", content="第一版内容", actor=u)
    db.commit()
    first = jk.list_chunks(db, j.id, tenant_id=t.id)
    src.content = "第二版内容,更长一些。" * 5
    jk.index_source(db, src, actor=u)
    db.commit()
    second = jk.list_chunks(db, j.id, tenant_id=t.id)
    assert all("第二版" in c.content or c.content != first[0].content for c in second)


def test_tenant_isolation_chunks(db):
    t1, t2 = _tenant(db), _tenant(db)
    u1 = _user(db, t1)
    j1 = _job(db, t1)
    jk.create_text_source(db, job=j1, name="T1资料", content="甲", actor=u1)
    db.commit()
    # 他租户同 job_id 查不到(chunk 有 tenant 列)
    assert jk.list_chunks(db, j1.id, tenant_id=t2.id) == []
    # 上下文召回也按 job.tenant 过滤
    assert jk.job_knowledge_context(db, _job(db, t2)) == []


# ======================= API =======================


def test_api_text_source_crud(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    r = client.post(
        f"/api/jobs/{j.id}/knowledge-sources/text",
        json={"name": "技术栈", "content": "我们用 FastAPI + Postgres 构建。"},
        headers=_hdr(u),
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "ready" and body["chunk_count"] >= 1

    r2 = client.get(f"/api/jobs/{j.id}/knowledge-sources/", headers=_hdr(u))
    assert r2.status_code == 200
    assert len(r2.json()) == 1

    r3 = client.get(f"/api/jobs/{j.id}/knowledge-sources/chunks", headers=_hdr(u))
    assert r3.status_code == 200
    assert r3.json()[0]["source_name"] == "技术栈"

    r4 = client.delete(
        f"/api/jobs/{j.id}/knowledge-sources/{body['id']}", headers=_hdr(u)
    )
    assert r4.status_code == 204
    assert client.get(f"/api/jobs/{j.id}/knowledge-sources/", headers=_hdr(u)).json() == []


def test_api_write_perm_denied(db, client):
    t = _tenant(db)
    viewer = _user(db, t, role="viewer")
    j = _job(db, t)
    r = client.post(
        f"/api/jobs/{j.id}/knowledge-sources/text",
        json={"name": "x", "content": "y"},
        headers=_hdr(viewer),
    )
    assert r.status_code == 403


def test_api_cross_tenant_404(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u2 = _user(db, t2)
    j1 = _job(db, t1)
    r = client.get(f"/api/jobs/{j1.id}/knowledge-sources/", headers=_hdr(u2))
    assert r.status_code == 404


def test_api_reindex(db, client):
    t = _tenant(db)
    u = _user(db, t)
    j = _job(db, t)
    src = jk.create_text_source(db, job=j, name="源", content=" ", actor=u)
    db.commit()
    assert src.status == "failed"
    # 修内容后重索引
    src.content = "现在有内容了。"
    db.commit()
    r = client.post(
        f"/api/jobs/{j.id}/knowledge-sources/{src.id}/reindex", headers=_hdr(u)
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


# ======================= model_call_logs API =======================


def test_model_call_logs_perm_and_filters(db, client):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    viewer = _user(db, t, role="viewer")
    write_model_call(
        tenant_id=t.id, call_type="match.evaluate", provider="db", model="m1",
        input_digest="d1", output={"score": 90}, latency_ms=120,
    )
    write_model_call(
        tenant_id=t.id, call_type="import.column_map", provider="db", model="m1",
        input_digest="d2", error_kind="parse_error", error_message="bad json",
        latency_ms=80,
    )

    # viewer 无权
    assert client.get("/api/model-call-logs", headers=_hdr(viewer)).status_code == 403
    assert client.get("/api/model-call-logs/stats", headers=_hdr(viewer)).status_code == 403

    r = client.get("/api/model-call-logs?limit=50", headers=_hdr(admin))
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2

    r2 = client.get(
        "/api/model-call-logs?call_type=match.evaluate", headers=_hdr(admin)
    )
    assert len(r2.json()["items"]) == 1
    assert r2.json()["items"][0]["output"]["score"] == 90

    r3 = client.get("/api/model-call-logs?has_error=true", headers=_hdr(admin))
    assert len(r3.json()["items"]) == 1
    assert r3.json()["items"][0]["error_kind"] == "parse_error"

    s = client.get("/api/model-call-logs/stats?hours=24", headers=_hdr(admin)).json()
    assert s["total"] == 2 and s["errors"] == 1
    assert s["error_rate"] == 0.5
    types = {x["call_type"] for x in s["by_type"]}
    assert types == {"match.evaluate", "import.column_map"}


def test_model_call_logs_tenant_isolation(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    a1 = _user(db, t1, role="admin")
    write_model_call(
        tenant_id=t1.id, call_type="x", provider="m", model="m", input_digest="d",
    )
    r = client.get("/api/model-call-logs?limit=50", headers=_hdr(a1))
    assert all(i["call_type"] == "x" for i in r.json()["items"])


def test_report_endpoint_uses_caller_identity(db, client):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post(
        "/api/model-call-logs/report",
        json={
            "call_type": "assistant.completion",
            "model": "deepseek/deepseek-chat",
            "input_digest": "abcd1234",
            "latency_ms": 800,
            "output": {"tools": ["hr_get_candidate"], "reply_chars": 42},
        },
        headers=_hdr(u),
    )
    assert r.status_code == 201, r.text
    logs = _logs(t.id)
    assert len(logs) == 1
    row = logs[0]
    assert row.call_type == "assistant.completion"
    assert row.user_id == u.id
    assert row.tenant_id == t.id
    assert row.output["tools"] == ["hr_get_candidate"]
