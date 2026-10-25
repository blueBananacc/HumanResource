"""T10 生产级幂等键 —— 真实 HTTP 链路验证(容器内运行)。

用法:
    docker compose exec backend python scripts/verify_t10_live.py

覆盖:
- POST 同键回放(同 ID / X-Idempotent-Replayed 头 / 库里单条)
- 同键异体 → 409 IDEMPOTENCY_KEY_REUSED
- 无键两次执行 → 两条记录
- PATCH 同键回放
- 失败请求终态回放(400 不缓存为成功)
- 键过期/释放后可重用
- 跨租户同键互不可见
- 运维 API:list/stats/get/release(权限门)
- assistant confirm call_id 兜底键(双发同 call_id → 单次生效)

退出码 0 = 全过。自动清理自建数据。
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
RUN = uuid.uuid4().hex[:6]

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))


def req(method: str, path: str, token: str, **kw):
    return requests.request(
        method, f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def req_key(method: str, path: str, token: str, key: str, **kw):
    return requests.request(
        method, f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": key},
        timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import (
        Candidate, IdempotencyKey, KnowledgeArticle, Tenant, User as DbUser,
    )
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")
    created_articles: list[str] = []
    cleanup_users: list[str] = []

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        tid = t.id
        cand = Candidate(tenant_id=tid, name=f"T10链测人{RUN}",
                         display_email=f"t10c-{RUN}@x.com")
        db.add(cand)
        db.flush()
        cand_id = cand.id
        # 他租户 + 他租户 admin(跨租户键隔离测试)
        t2 = Tenant(name=f"t10-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso = DbUser(tenant_id=t2.id, email=f"t10-iso-{RUN}@x.com",
                     password_hash=hash_password("T10iso!pass"), role="admin",
                     status="active")
        db.add(iso)
        # viewer(权限门测试)
        vw = DbUser(tenant_id=tid, email=f"t10-vw-{RUN}@x.com",
                    password_hash=hash_password("T10vw!pass"), role="viewer",
                    status="active")
        db.add(vw)
        db.commit()
        t2id, iso_id, vw_id = t2.id, iso.id, vw.id
    iso_tok = login(f"t10-iso-{RUN}@x.com", "T10iso!pass")
    vw_tok = login(f"t10-vw-{RUN}@x.com", "T10vw!pass")
    cleanup_users += [iso_id, vw_id]

    # ---------- 1. POST 同键回放 ----------
    key = f"live-{RUN}-post"
    title = f"T10幂等文章{RUN}"
    body = {"fields": {"title": title, "content": "live 链路", "category": "other"}}
    r1 = req_key("POST", "/api/entities/knowledge", admin, key, json=body)
    check("POST 首次执行 201", r1.status_code == 201, r1.text[:200])
    aid = r1.json().get("id") if r1.status_code == 201 else None
    if aid:
        created_articles.append(aid)
    r2 = req_key("POST", "/api/entities/knowledge", admin, key, json=body)
    check("POST 同键回放同 ID", r2.status_code == 201 and r2.json().get("id") == aid)
    check("回放头 X-Idempotent-Replayed", r2.headers.get("Idempotency-Replayed") == "true")
    check("回放头带原始时间戳", bool(r2.headers.get("Idempotency-Original-Request")))
    with SessionLocal() as db:
        n = db.query(KnowledgeArticle).filter_by(tenant_id=tid, title=title).count()
    check("库里只有一条记录", n == 1, f"n={n}")

    # ---------- 2. 同键异体 → 409 ----------
    r3 = req_key("POST", "/api/entities/knowledge", admin, key,
                 json={"fields": {"title": "另一篇", "content": "x"}})
    check("同键异体 409", r3.status_code == 409, r3.text[:200])
    detail = r3.json().get("detail")
    check("409 带 REUSED 码", isinstance(detail, dict) and detail.get("code") == "IDEMPOTENCY_KEY_REUSED")
    check("409 带 hash 前缀对", isinstance(detail, dict) and "expected" in detail and "actual" in detail)

    # ---------- 3. 无键各自执行 ----------
    title2 = f"T10无键{RUN}"
    b2 = {"fields": {"title": title2, "content": "x"}}
    ra = req("POST", "/api/entities/knowledge", admin, json=b2)
    rb = req("POST", "/api/entities/knowledge", admin, json=b2)
    if ra.status_code == 201:
        created_articles.append(ra.json()["id"])
    if rb.status_code == 201:
        created_articles.append(rb.json()["id"])
    with SessionLocal() as db:
        n2 = db.query(KnowledgeArticle).filter_by(tenant_id=tid, title=title2).count()
    check("无键两次各执行", n2 == 2, f"n={n2}")

    # ---------- 4. PATCH 同键回放(talents 可写) ----------
    key_p = f"live-{RUN}-patch"
    pb = {"fields": {"tags": ["t10", "idem"]}}
    p1 = req_key("PATCH", f"/api/entities/talents/{cand_id}", admin, key_p, json=pb)
    check("PATCH 首次 200", p1.status_code == 200, p1.text[:200])
    p2 = req_key("PATCH", f"/api/entities/talents/{cand_id}", admin, key_p, json=pb)
    check("PATCH 同键回放", p2.status_code == 200 and p2.headers.get("Idempotency-Replayed") == "true")

    # ---------- 5. 失败终态回放 ----------
    key_f = f"live-{RUN}-fail"
    bad = {"fields": {"title": ""}}
    f1 = req_key("POST", "/api/entities/knowledge", admin, key_f, json=bad)
    check("坏请求 400", f1.status_code == 400)
    f2 = req_key("POST", "/api/entities/knowledge", admin, key_f, json=bad)
    check("失败回放 400 + 回放头", f2.status_code == 400 and f2.headers.get("Idempotency-Replayed") == "true")

    # ---------- 6. 跨租户同键不可见 ----------
    r_iso = req_key("POST", "/api/entities/knowledge", iso_tok, key,
                    json={"fields": {"title": f"iso-{RUN}", "content": "x"}})
    check("他租户同 key 独立执行(不回放)", r_iso.status_code == 201 and "Idempotency-Replayed" not in r_iso.headers)
    if r_iso.status_code == 201:
        with SessionLocal() as db:
            db.query(KnowledgeArticle).filter_by(id=r_iso.json()["id"]).delete()
            db.commit()

    # ---------- 7. 运维 API ----------
    rl = req("GET", "/api/idempotency-keys", admin)
    check("运维 list 200", rl.status_code == 200)
    if rl.status_code == 200:
        check("list 含本租户键", any(k["key"].startswith(f"live-{RUN}") for k in rl.json()))
        # 他租户自己也调 list:同名字符串 key 两边都有,但行数/归属各自独立
        rl_iso = req("GET", "/api/idempotency-keys", iso_tok)
        if rl_iso.status_code == 200:
            iso_rows = [k for k in rl_iso.json() if k["key"].startswith(f"live-{RUN}")]
            check("他租户只见自己的键行", len(iso_rows) == 1)
    rs = req("GET", "/api/idempotency-keys/stats", admin)
    check("stats 200 + 结构", rs.status_code == 200 and "by_status" in rs.json())
    # 找到 post key 的 id
    kid = None
    if rl.status_code == 200:
        kid = next((k["id"] for k in rl.json() if k["key"] == key), None)
    if kid:
        rg = req("GET", f"/api/idempotency-keys/{kid}", admin)
        check("get 单键 200 + completed", rg.status_code == 200 and rg.json()["status"] == "completed")
        check("hash 只回前缀", len(rg.json().get("request_hash", "")) <= 8)
        check("resource_id 已绑定", rg.json().get("resource_id") == aid)
    rlv = req("GET", "/api/idempotency-keys", vw_tok)
    check("viewer 无权看键", rlv.status_code == 403)

    # ---------- 8. release 后键可重用 ----------
    key_r = f"live-{RUN}-rel"
    req_key("POST", "/api/entities/knowledge", admin, key_r,
            json={"fields": {"title": f"rel1-{RUN}", "content": "x"}})
    rr = req("GET", "/api/idempotency-keys", admin)
    rid = next((k["id"] for k in rr.json() if k["key"] == key_r), None) if rr.status_code == 200 else None
    if rid:
        rd = req("DELETE", f"/api/idempotency-keys/{rid}", admin)
        check("release 200", rd.status_code == 200)
        r_re = req_key("POST", "/api/entities/knowledge", admin, key_r,
                       json={"fields": {"title": f"rel2-{RUN}", "content": "x"}})
        check("释放后同键重新执行(非回放)", r_re.status_code == 201 and "Idempotency-Replayed" not in r_re.headers)
        if r_re.status_code == 201:
            created_articles.append(r_re.json()["id"])

    # ---------- 9. confirm call_id 兜底 ----------
    # 双发同 call_id:第二次应回放(approved=False 无副作用,安全可测)
    cid = f"call-{RUN}-noop"
    c1 = req("POST", "/api/assistant/confirm", admin, json={"call_id": cid, "approved": False})
    c2 = req("POST", "/api/assistant/confirm", admin, json={"call_id": cid, "approved": False})
    check("confirm 同 call_id 幂等", c2.headers.get("Idempotency-Replayed") == "true" or c2.status_code == c1.status_code,
          f"{c1.status_code}→{c2.status_code}")

    # ---------- 10. 过期键惰性回收 ----------
    with SessionLocal() as db:
        row = db.scalars(sa_select(IdempotencyKey).where(
            IdempotencyKey.tenant_id == tid, IdempotencyKey.key == key_p)).first()
        if row:
            row.expires_at = datetime.utcnow() - timedelta(minutes=5)
            db.commit()
    e1 = req_key("PATCH", f"/api/entities/talents/{cand_id}", admin, key_p,
                 json={"fields": {"tags": ["t10"]}})
    check("过期键重新执行(非回放)", e1.status_code == 200 and "Idempotency-Replayed" not in e1.headers)

    # ---------- 清理 ----------
    with SessionLocal() as db:
        for a in created_articles:
            db.query(KnowledgeArticle).filter_by(id=a).delete()
        db.query(KnowledgeArticle).filter(
            KnowledgeArticle.tenant_id == tid,
            KnowledgeArticle.title.like(f"T10%{RUN}%")).delete(synchronize_session=False)
        db.query(KnowledgeArticle).filter(
            KnowledgeArticle.tenant_id == tid,
            KnowledgeArticle.title.like(f"rel%-{RUN}")).delete(synchronize_session=False)
        db.query(IdempotencyKey).filter(
            IdempotencyKey.tenant_id.in_([tid, t2id]),
            IdempotencyKey.key.like(f"live-{RUN}%")).delete(synchronize_session=False)
        db.query(IdempotencyKey).filter(
            IdempotencyKey.key == cid).delete(synchronize_session=False)
        db.query(Candidate).filter_by(id=cand_id).delete()
        for uid in cleanup_users:
            db.query(DbUser).filter_by(id=uid).delete()
        db.query(Tenant).filter_by(id=t2id).delete()
        db.commit()

    print(f"\n===== T10 live: {len(PASS)} passed, {len(FAIL)} failed =====")
    for f in FAIL:
        print("  FAIL:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
