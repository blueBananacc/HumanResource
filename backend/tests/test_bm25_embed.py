# -*- coding: utf-8 -*-
"""RAG 升级测试:应用层 BM25 词面排序 + 独立 embedding 通道。

- tokenize:CJK bigram/单字 + ASCII 词
- bm25_scores:相关性排序 / 字段加权(title>content)/ 候选过滤 / 空查询
- _hybrid_search:无 embedding 时 BM25 独立可用
- LLMProvider embedding 通道:独立 base_url/key 落库 + 掩码返回 + 留空保留
- embed_texts:独立通道寻址(mock httpx)+ 失败降级 None
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-bm25")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain.models import (  # noqa: E402
    Base,
    KnowledgeArticle,
    LLMProvider,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import embedding as emb_svc  # noqa: E402
from app.services.bm25 import bm25_scores, tokenize  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    # create_all 不建扩展;召回 SQL 的 similarity() 依赖 pg_trgm
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.commit()
        except Exception:
            conn.rollback()
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
    t = Tenant(name=f"bm25-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant) -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role="admin",
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


def _article(db, t: Tenant, title: str, content: str, tags=None) -> KnowledgeArticle:
    a = KnowledgeArticle(
        tenant_id=t.id,
        title=title,
        content=content,
        category="company",
        tags=tags or [],
        status="published",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ---------------- tokenize ----------------


def test_tokenize_cjk_bigrams():
    toks = tokenize("年假制度")
    assert "年假" in toks and "假制" in toks and "制度" in toks
    assert "年" in toks  # 单字也保留


def test_tokenize_ascii_and_empty():
    assert "fastapi" in tokenize("FastAPI 框架")
    assert tokenize("") == []


# ---------------- bm25_scores ----------------


def _corpus():
    return [
        ("a1", "带薪年假制度", ["hr-policy"], "员工入职满一年可享受五天带薪年假,需提前在系统中申请。"),
        ("a2", "报销流程", ["finance"], "差旅报销需在月底前提交发票。"),
        ("a3", "远程办公规范", ["policy"], "每周最多两天远程办公,年假期间不计入。"),
    ]


def test_bm25_ranks_relevant_first():
    scores = bm25_scores("年假申请", _corpus())
    assert scores["a1"] == pytest.approx(1.0)  # 最高分归一化为 1
    assert scores["a1"] > scores.get("a3", 0)
    assert "a2" not in scores  # 无命中词项不出现在结果里


def test_bm25_title_boost():
    """query 只出现在 title 的文档应胜过只出现在 content 的。"""
    corpus = [
        ("t1", "绩效考核办法", [], "本制度适用于全体员工。"),
        ("t2", "员工手册", [], "公司每年进行绩效考核,详见附录。"),
    ]
    scores = bm25_scores("绩效考核", corpus)
    assert scores["t1"] > scores.get("t2", 0)


def test_bm25_candidate_filter_and_empty():
    scores = bm25_scores("年假", _corpus(), candidate_ids={"a3"})
    assert set(scores) == {"a3"}
    assert bm25_scores("", _corpus()) == {}
    assert bm25_scores("qqzzx9", _corpus()) == {}


# ---------------- _hybrid_search(BM25 独立工作)----------------


def test_hybrid_search_bm25_only(db):
    """无 embedding provider 时,BM25 词面排序独立可用。"""
    from app.api.knowledge import _hybrid_search

    t = _tenant(db)
    _article(db, t, "带薪年假制度", "员工入职满一年可享受五天带薪年假,需提前申请审批。")
    _article(db, t, "考勤管理规定", "员工每日上下班打卡,迟到按月累计。")
    _article(db, t, "草稿不可见", "年假相关内容但 status=draft").status = "draft"
    db.commit()

    hits = _hybrid_search(db, t.id, "年假", 5, published_only=True)
    assert hits and hits[0][0].title == "带薪年假制度"
    assert all(a.status == "published" for a, _ in hits)


def test_search_api_bm25(client, db):
    t = _tenant(db)
    u = _user(db, t)
    _article(db, t, "差旅报销标准", "市内交通按票据实报实销,住宿费按职级上限报销。")
    r = client.post("/api/knowledge/search", json={"q": "报销", "limit": 5}, headers=_hdr(u))
    assert r.status_code == 200
    hits = r.json()
    assert hits and hits[0]["title"] == "差旅报销标准"
    assert hits[0]["match_field"] in ("content", "title")


# ---------------- LLMProvider 独立 embedding 通道 ----------------


def test_provider_embedding_channel_roundtrip(client, db):
    t = _tenant(db)
    u = _user(db, t)
    body = {
        "name": "deepseek+hf",
        "api_key": "sk-chat-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek/deepseek-chat",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_base_url": "https://router.huggingface.co/v1",
        "embedding_api_key": "hf_secret_token_123",
    }
    r = client.post("/api/llm/providers", json=body, headers=_hdr(u))
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["embedding_base_url"] == "https://router.huggingface.co/v1"
    assert out["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    # 掩码返回,不泄露明文
    assert out["embedding_api_key_masked"] == "hf_***..._123"
    assert "hf_secret_token_123" not in r.text

    # 留空更新 → 原 key 保留
    r2 = client.put(
        f"/api/llm/providers/{out['id']}",
        json={**body, "api_key": None, "embedding_api_key": None, "name": "deepseek+hf2"},
        headers=_hdr(u),
    )
    assert r2.status_code == 200
    row = db.get(LLMProvider, out["id"])
    from app.infra.crypto import decrypt

    assert decrypt(row.embedding_api_key_encrypted) == "hf_secret_token_123"


def test_embed_texts_independent_channel(client, db, monkeypatch):
    """embedding 走独立 base_url/key,不复用 chat 通道。"""
    t = _tenant(db)
    u = _user(db, t)
    r = client.post(
        "/api/llm/providers",
        json={
            "name": "mix",
            "api_key": "sk-chat",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek/deepseek-chat",
            "embedding_model": "bge-m3",
            "embedding_base_url": "https://router.huggingface.co/v1",
            "embedding_api_key": "hf_key_abc",
        },
        headers=_hdr(u),
    )
    assert r.status_code == 201
    pid = r.json()["id"]
    client.post(f"/api/llm/providers/{pid}/activate", headers=_hdr(u))

    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen["auth"] = kw["headers"]["Authorization"]

        class R:
            status_code = 200

            def json(self):
                return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

        return R()

    monkeypatch.setattr(emb_svc.httpx, "post", fake_post)
    out = emb_svc.embed_texts(db, t.id, ["hello"])
    assert out == [[0.1, 0.2]]
    assert seen["url"] == "https://router.huggingface.co/v1/embeddings"
    assert seen["auth"] == "Bearer hf_key_abc"


def test_embed_texts_falls_back_gracefully(client, db, monkeypatch):
    """未配 embedding_model / 端点失败 → None,不抛异常。"""
    t = _tenant(db)
    u = _user(db, t)
    r = client.post(
        "/api/llm/providers",
        json={"name": "chat-only", "api_key": "sk-1", "model": "deepseek/deepseek-chat"},
        headers=_hdr(u),
    )
    assert r.status_code == 201
    pid = r.json()["id"]
    client.post(f"/api/llm/providers/{pid}/activate", headers=_hdr(u))
    assert emb_svc.embed_texts(db, t.id, ["x"]) is None  # 没配 embedding_model

    # 配了但端点 500 → 仍 None
    client.put(
        f"/api/llm/providers/{pid}",
        json={"name": "chat-only", "model": "deepseek/deepseek-chat", "embedding_model": "m"},
        headers=_hdr(u),
    )

    def boom(*a, **kw):
        raise emb_svc.httpx.HTTPError("down")

    monkeypatch.setattr(emb_svc.httpx, "post", boom)
    assert emb_svc.embed_texts(db, t.id, ["x"]) is None
