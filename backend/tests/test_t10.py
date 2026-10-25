# -*- coding: utf-8 -*-
"""T10 生产级幂等键 —— 服务/策略/API 三层测试。

覆盖:
- 纯策略: key 规范化/校验、canonical body/path/query、request_hash
- 服务: claim/complete/fail/replay/takeover/purge/stats
- API: POST 同键回放、同键异体 409、无键各自执行、GET 忽略、
  确认门 call_id 兜底键、运维 API 权限
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t10")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.idempotency import _fallback_key  # noqa: E402
from app.config import settings  # noqa: E402,F401
from app.domain import models as m  # noqa: E402,F401  # 确保新表在 metadata
from app.domain.models import (  # noqa: E402
    Base,
    Candidate,
    IdempotencyKey,
    Tenant,
    User,
)
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import idempotency as svc  # noqa: E402
from app.services import idempotency_policy as pol  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402


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
    t = Tenant(name=f"t10-{uuid.uuid4().hex[:8]}")
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


def _purge_scope(db, tenant_id: str, scope: str):
    db.query(IdempotencyKey).filter_by(tenant_id=tenant_id, scope=scope).delete()
    db.commit()


@pytest.fixture
def client():
    return TestClient(app)


# ======================= 策略层 =======================


def test_key_normalize_and_validate():
    assert pol.normalize_key("  abc-def_123  ") == "abc-def_123"
    assert pol.normalize_key(123) is None
    assert pol.normalize_key(None) is None
    assert pol.validate_key("short") is not None  # < 8
    assert pol.validate_key("x" * 300) is not None  # > 255
    assert pol.validate_key("k-" + uuid.uuid4().hex) is None


def test_stable_json_canonical():
    # 对象键序无关
    a = pol.stable_json({"b": 1, "a": {"y": 2, "x": 1}})
    b = pol.stable_json({"a": {"x": 1, "y": 2}, "b": 1})
    assert a == b
    # path 剥 query/fragment(query 单独 hash)
    assert pol.canonicalize_path("/api/x?y=1#f") == "/api/x"
    # query 排序稳定
    q1 = pol.canonicalize_query("b=2&a=1")
    q2 = pol.canonicalize_query("a=1&b=2")
    assert q1 == q2


def test_request_hash_sensitive_to_body_and_path():
    h1 = pol.build_request_hash(method="POST", path="/api/x", query="", raw_body=b'{"a":1}')
    h2 = pol.build_request_hash(method="POST", path="/api/x", query="", raw_body=b'{"a":2}')
    h3 = pol.build_request_hash(method="POST", path="/api/y", query="", raw_body=b'{"a":1}')
    h4 = pol.build_request_hash(method="PUT", path="/api/x", query="", raw_body=b'{"a":1}')
    assert h1 != h2 and h1 != h3 and h1 != h4
    # JSON 语义等价(key 序不同)→ 同 hash
    h5 = pol.build_request_hash(method="POST", path="/api/x", query="", raw_body=b'{"a": 1}')
    assert h1 == h5


# ======================= 服务层 =======================


def _claim(tid, uid=None, scope="s.test", key="k-12345678", body=b'{"a":1}'):
    h = pol.build_request_hash(method="POST", path="/api/x", query="", raw_body=body)
    return svc.claim(
        tenant_id=tid, user_id=uid, scope=scope, key=key, request_hash=h, ttl_seconds=3600
    )


def test_claim_replay_cycle(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c1 = _claim(t.id, scope=scope)
    assert c1.outcome == "claimed"
    svc.complete(c1.record.id, response_status=201, response_body={"id": "r1"})
    c2 = _claim(t.id, scope=scope)
    assert c2.outcome == "replayed"
    assert c2.replay_status == 201 and c2.replay_body == {"id": "r1"}


def test_claim_reused_key_different_body(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c1 = _claim(t.id, scope=scope, body=b'{"a":1}')
    svc.complete(c1.record.id, response_status=201, response_body={})
    c2 = _claim(t.id, scope=scope, body=b'{"a":999}')
    assert c2.outcome == "reused_key"
    assert c2.expected_hash_prefix != c2.actual_hash_prefix


def test_in_flight_then_stale_takeover(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c1 = _claim(t.id, scope=scope)
    assert c1.outcome == "claimed"
    # 锁还活着 → in_flight
    c2 = _claim(t.id, scope=scope)
    assert c2.outcome == "in_flight" and c2.retry_after_ms > 0
    # 把锁改成 2 分钟前 → takeover
    row = db.scalars(
        __import__("sqlalchemy").select(IdempotencyKey).where(
            IdempotencyKey.id == c1.record.id
        )
    ).first()
    row.locked_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=121)
    db.commit()
    c3 = _claim(t.id, scope=scope)
    assert c3.outcome == "claimed"


def test_fail_replays_failure(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c1 = _claim(t.id, scope=scope)
    svc.fail(c1.record.id, response_status=422, response_body={"detail": "bad"})
    c2 = _claim(t.id, scope=scope)
    assert c2.outcome == "replayed" and c2.replay_status == 422


def test_scope_and_tenant_isolation(db):
    t1, t2 = _tenant(db), _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    _claim(t1.id, scope=scope)  # t1 占 key
    svc.complete(
        db.scalars(
            __import__("sqlalchemy").select(IdempotencyKey).where(
                IdempotencyKey.tenant_id == t1.id, IdempotencyKey.scope == scope
            )
        ).first().id,
        response_status=201,
        response_body={"x": 1},
    )
    # 同 key 不同 scope → 独立认领
    other = _claim(t1.id, scope=f"{scope}-b")
    assert other.outcome == "claimed"
    # 同 key+scope 不同租户 → 独立
    c_t2 = _claim(t2.id, scope=scope)
    assert c_t2.outcome == "claimed"


def test_purge_and_stats(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c = _claim(t.id, scope=scope)
    svc.complete(c.record.id, response_status=200, response_body=None)
    row = db.get(IdempotencyKey, c.record.id)
    row.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    db.commit()
    n = svc.purge_expired()
    assert n >= 1
    db.expire_all()
    assert db.get(IdempotencyKey, c.record.id) is None
    st = svc.stats(db, t.id)
    assert "by_status" in st and "total" in st


def test_expired_key_claimed_fresh(db):
    t = _tenant(db)
    scope = f"s.{uuid.uuid4().hex[:6]}"
    c = _claim(t.id, scope=scope)
    svc.complete(c.record.id, response_status=200, response_body={"old": True})
    row = db.get(IdempotencyKey, c.record.id)
    row.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)
    db.commit()
    c2 = _claim(t.id, scope=scope)
    assert c2.outcome == "claimed"  # 过期行被惰性回收,重新认领


# ======================= API 集成:entities =======================


def test_entities_post_replay(client, db):
    t = _tenant(db)
    u = _user(db, t)
    key = f"k-{uuid.uuid4().hex}"
    title = f"幂等文章{uuid.uuid4().hex[:6]}"
    body = {"fields": {"title": title, "content": "正文"}}
    hdrs = {**_hdr(u), "Idempotency-Key": key}
    r1 = client.post("/api/entities/knowledge", json=body, headers=hdrs)
    assert r1.status_code == 201, r1.text
    id1 = r1.json()["id"]
    r2 = client.post("/api/entities/knowledge", json=body, headers=hdrs)
    assert r2.status_code == 201
    assert r2.json()["id"] == id1
    assert r2.headers.get("idempotency-replayed") == "true"
    from app.domain.models import KnowledgeArticle

    n = db.query(KnowledgeArticle).filter_by(tenant_id=t.id, title=title).count()
    assert n == 1


def test_entities_same_key_different_body_409(client, db):
    t = _tenant(db)
    u = _user(db, t)
    key = f"k-{uuid.uuid4().hex}"
    hdrs = {**_hdr(u), "Idempotency-Key": key}
    r1 = client.post(
        "/api/entities/knowledge",
        json={"fields": {"title": "A", "content": "c"}},
        headers=hdrs,
    )
    assert r1.status_code == 201
    r2 = client.post(
        "/api/entities/knowledge",
        json={"fields": {"title": "B", "content": "c"}},
        headers=hdrs,
    )
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == "IDEMPOTENCY_KEY_REUSED"


def test_entities_no_key_executes_each_time(client, db):
    t = _tenant(db)
    u = _user(db, t)
    title = f"无键{uuid.uuid4().hex[:6]}"
    body = {"fields": {"title": title, "content": "c"}}
    client.post("/api/entities/knowledge", json=body, headers=_hdr(u))
    client.post("/api/entities/knowledge", json=body, headers=_hdr(u))
    from app.domain.models import KnowledgeArticle

    n = db.query(KnowledgeArticle).filter_by(tenant_id=t.id, title=title).count()
    assert n == 2


def test_get_ignores_key(client, db):
    t = _tenant(db)
    u = _user(db, t)
    key = f"k-{uuid.uuid4().hex}"
    r = client.get(
        "/api/entities/talents/search?q=x", headers={**_hdr(u), "Idempotency-Key": key}
    )
    assert r.status_code == 200
    assert "idempotency-replayed" not in r.headers


def test_entities_patch_replay(client, db):
    t = _tenant(db)
    u = _user(db, t)
    c = Candidate(tenant_id=t.id, name="P0", display_email="p@x.com")
    db.add(c)
    db.commit()
    key = f"k-{uuid.uuid4().hex}"
    hdrs = {**_hdr(u), "Idempotency-Key": key}
    body = {"fields": {"tags": ["a", "b"]}}
    r1 = client.patch(f"/api/entities/talents/{c.id}", json=body, headers=hdrs)
    assert r1.status_code == 200, r1.text
    r2 = client.patch(f"/api/entities/talents/{c.id}", json=body, headers=hdrs)
    assert r2.status_code == 200
    assert r2.headers.get("idempotency-replayed") == "true"


def test_bad_key_400(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post(
        "/api/entities/knowledge",
        json={"fields": {"title": "X", "content": "c"}},
        headers={**_hdr(u), "Idempotency-Key": "!!bad!!"},
    )
    assert r.status_code == 400


def test_failure_not_cached_blocks_retry_replay(client, db):
    """端点抛错 → 存 failed 态;同键重试 = 回放该错误(确定性语义)。"""
    t = _tenant(db)
    u = _user(db, t)
    key = f"k-{uuid.uuid4().hex}"
    hdrs = {**_hdr(u), "Idempotency-Key": key}
    bad = {"fields": {"title": ""}}  # 缺 content + 空 title → 400
    r1 = client.post("/api/entities/knowledge", json=bad, headers=hdrs)
    assert r1.status_code == 400
    r2 = client.post("/api/entities/knowledge", json=bad, headers=hdrs)
    assert r2.status_code == 400
    assert r2.headers.get("idempotency-replayed") == "true"


def test_scope_key_across_users_no_leak(client, db):
    """同租户两个用户用同 key 同 body:第二个用户得到的是'自己键空间'的回放。

    键唯一性是 (tenant, scope, key) —— scope 已含端点;不同 user 不隔离键,
    但 response 是终态确定性回放,不会重复执行,也不会泄露对方数据
    (body 相同 ⇒ 返回同一已创建记录 ID)。
    """
    t = _tenant(db)
    u1 = _user(db, t, role="admin")
    u2 = _user(db, t, role="admin")
    key = f"k-{uuid.uuid4().hex}"
    body = {"fields": {"title": f"共享键{uuid.uuid4().hex[:6]}", "content": "c"}}
    r1 = client.post(
        "/api/entities/knowledge", json=body, headers={**_hdr(u1), "Idempotency-Key": key}
    )
    r2 = client.post(
        "/api/entities/knowledge", json=body, headers={**_hdr(u2), "Idempotency-Key": key}
    )
    assert r2.status_code == 201 and r2.json()["id"] == r1.json()["id"]
    assert r2.headers.get("idempotency-replayed") == "true"


# ======================= 确认门 call_id 兜底 =======================


class _Body:
    def __init__(self, call_id):
        self.call_id = call_id


def test_fallback_key_from_body():
    assert _fallback_key({"body": _Body("call-abc")}, "call_id") == "call-abc"
    assert _fallback_key({"body": {"call_id": "call-x"}}, "call_id") == "call-x"
    assert _fallback_key({"body": _Body(None)}, "call_id") is None
    assert _fallback_key({}, "call_id") is None


# ======================= 运维 API =======================


def test_ops_list_and_peek(client, db):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    viewer = _user(db, t, role="viewer")
    c = _claim(t.id, uid=admin.id, scope="s.ops")
    svc.complete(c.record.id, response_status=200, response_body={"k": 1})

    r = client.get("/api/idempotency-keys", headers=_hdr(admin))
    assert r.status_code == 200
    assert any(k["id"] == c.record.id for k in r.json())
    r2 = client.get(f"/api/idempotency-keys/{c.record.id}", headers=_hdr(admin))
    assert r2.status_code == 200 and r2.json()["status"] == "completed"
    # viewer 无权
    r3 = client.get("/api/idempotency-keys", headers=_hdr(viewer))
    assert r3.status_code == 403


def test_ops_release_resets_key(client, db):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    c = _claim(t.id, scope="s.rel")
    svc.complete(c.record.id, response_status=200, response_body={})
    r = client.delete(f"/api/idempotency-keys/{c.record.id}", headers=_hdr(admin))
    assert r.status_code == 200
    # 释放后同 key 重新认领
    c2 = _claim(t.id, scope="s.rel")
    assert c2.outcome == "claimed"


def test_ops_stats_shape(client, db):
    t = _tenant(db)
    admin = _user(db, t, role="admin")
    _claim(t.id, scope="s.stat")
    r = client.get("/api/idempotency-keys/stats", headers=_hdr(admin))
    assert r.status_code == 200
    body = r.json()
    assert "by_status" in body and "total" in body and "recent_24h" in body
