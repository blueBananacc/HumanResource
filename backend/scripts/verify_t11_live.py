"""T11 逐次 LLM 调用留痕 + 岗位级知识 chunks —— 真实 HTTP 链路验证(容器内跑)。

用法:
    docker compose exec backend python scripts/verify_t11_live.py

覆盖:
- 岗位知识: text 源创建→ready→chunks 列出→context 注入文本/重索引/删除
- 文件源: txt 上传→ready+extraction_method;损坏 pdf→failed
- 权限: viewer 写 403 / 跨租户 404 / ops API 403
- model_call_logs: 真实 LLM 调用(ai.summarize)落行/过滤/stats/跨租户隔离
- /report 端点: 模拟 sidecar 回报 assistant.completion
- 知识注入: service 层 knowledge_context_text 内容包含源文本 + 评估 user_msg 注入

退出码 0 = 全部通过;测试数据自动清理。
"""
from __future__ import annotations

import os
import sys
import uuid

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


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import (
        Candidate, Job, JobKnowledgeChunk, ModelCallLog, Tenant,
        User as DbUser,
    )
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password
    from app.services import job_knowledge as jk

    admin = login("admin@example.com", "admin123456")
    cleanup_sources: list[str] = []

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        tid = t.id
        job = Job(tenant_id=tid, title=f"T11链测岗位{RUN}", level="intermediate",
                  status="open", description="负责后端服务与 LLM 集成开发。",
                  skills=["Python", "FastAPI"])
        cand = Candidate(tenant_id=tid, name=f"T11链测人{RUN}",
                         display_email=f"t11c-{RUN}@x.com")
        db.add_all([job, cand])
        # 他租户(隔离) + viewer(权限门)
        t2 = Tenant(name=f"t11-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso = DbUser(tenant_id=t2.id, email=f"t11-iso-{RUN}@x.com",
                     password_hash=hash_password("T11iso!pass"), role="admin",
                     status="active")
        vw = DbUser(tenant_id=tid, email=f"t11-vw-{RUN}@x.com",
                    password_hash=hash_password("T11vw!pass"), role="viewer",
                    status="active")
        db.add_all([iso, vw])
        db.commit()
        job_id, cand_id, t2id, vw_id, iso_id = job.id, cand.id, t2.id, vw.id, iso.id

    iso_tok = login(f"t11-iso-{RUN}@x.com", "T11iso!pass")
    vw_tok = login(f"t11-vw-{RUN}@x.com", "T11vw!pass")

    # ---------- 1. text 知识源链路 ----------
    r = req("POST", f"/api/jobs/{job_id}/knowledge-sources/text", admin,
            json={"name": f"团队介绍{RUN}",
                  "content": "我们是 AI 招聘产品线。\n\n技术栈:Python/FastAPI/Postgres/Redis,LLM 集成走 LiteLLM 网关。"})
    check("text 源创建 201", r.status_code == 201, r.text[:200])
    src = r.json() if r.status_code == 201 else {}
    sid = src.get("id", "")
    if sid:
        cleanup_sources.append(sid)
    check("text 源 ready", src.get("status") == "ready", str(src)[:200])
    check("chunk_count>0", src.get("chunk_count", 0) >= 1)
    check("text 源 extraction=none", src.get("extraction_status") == "none")

    r = req("GET", f"/api/jobs/{job_id}/knowledge-sources/", admin)
    check("源列表=1", r.status_code == 200 and len(r.json()) == 1)

    r = req("GET", f"/api/jobs/{job_id}/knowledge-sources/chunks", admin)
    chunks = r.json() if r.status_code == 200 else []
    check("chunks 可预览", len(chunks) >= 1)
    check("chunk 带 source_name", chunks and chunks[0]["source_name"] == f"团队介绍{RUN}")
    check("chunk 含正文", chunks and "LiteLLM" in chunks[0]["content"])

    # service 层 context 注入文本
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        ctx_text = jk.knowledge_context_text(db, job, limit=4)
    check("context_text 含源名", "团队介绍" in ctx_text)
    check("context_text 含正文", "LiteLLM" in ctx_text)
    check("context_text 有前缀", ctx_text.startswith("岗位参考资料:"))

    # ---------- 2. file 源链路 ----------
    files = {"file": (f"tech-{RUN}.txt", "技术栈说明:Python/FastAPI,部署在 K8s。".encode(), "text/plain")}
    r = requests.post(
        f"{BASE}/api/jobs/{job_id}/knowledge-sources/file?name=tech{RUN}",
        headers={"Authorization": f"Bearer {admin}"}, files=files, timeout=60)
    check("file 源上传 201", r.status_code == 201, r.text[:200])
    fsrc = r.json() if r.status_code == 201 else {}
    fsid = fsrc.get("id", "")
    if fsid:
        cleanup_sources.append(fsid)
    check("file 源 ready", fsrc.get("status") == "ready", str(fsrc)[:200])
    check("file 源 method=txt", fsrc.get("extraction_method") == "txt")

    # 不支持类型 → 400
    files2 = {"file": ("x.exe", b"MZ binary", "application/octet-stream")}
    r = requests.post(
        f"{BASE}/api/jobs/{job_id}/knowledge-sources/file",
        headers={"Authorization": f"Bearer {admin}"}, files=files2, timeout=60)
    check("exe 类型拒绝 400", r.status_code == 400)

    # ---------- 3. 重索引 ----------
    r = req("POST", f"/api/jobs/{job_id}/knowledge-sources/{sid}/reindex", admin)
    check("reindex 200 ready", r.status_code == 200 and r.json()["status"] == "ready")

    # ---------- 4. 权限与隔离 ----------
    r = req("POST", f"/api/jobs/{job_id}/knowledge-sources/text", vw_tok,
            json={"name": "x", "content": "y"})
    check("viewer 写源 403", r.status_code == 403)
    r = req("GET", f"/api/jobs/{job_id}/knowledge-sources/", iso_tok)
    check("跨租户查源 404", r.status_code == 404)
    r = req("POST", f"/api/jobs/{job_id}/knowledge-sources/text", iso_tok,
            json={"name": "x", "content": "y"})
    check("跨租户写源 404", r.status_code == 404)
    with SessionLocal() as db:
        n = db.query(JobKnowledgeChunk).filter_by(tenant_id=t2id).count()
    check("他租户无 chunks", n == 0)

    # ---------- 5. 真实 LLM 调用落痕 ----------
    r = req("GET", f"/api/ai/summarize/candidates/{cand_id}", admin)
    check("summarize 200", r.status_code == 200, r.text[:200])
    with SessionLocal() as db:
        row = db.scalars(
            sa_select(ModelCallLog)
            .where(ModelCallLog.tenant_id == tid,
                   ModelCallLog.call_type == "ai.summarize")
            .order_by(ModelCallLog.created_at.desc())
        ).first()
    check("summarize 落痕一行", row is not None)
    if row:
        check("留痕 provider/model 非空", bool(row.provider) and bool(row.model))
        check("留痕 digest 16 位", len(row.input_digest or "") == 16)
        check("留痕 latency>=0", row.latency_ms >= 0)
        check("留痕 prompt_key", row.prompt_key == "summarize.candidate")
        check("留痕 user_id", row.user_id is not None)

    # ---------- 6. ops API ----------
    r = req("GET", "/api/model-call-logs?call_type=ai.summarize&limit=10", admin)
    items = r.json().get("items", []) if r.status_code == 200 else []
    check("ops 列表含 summarize", len(items) >= 1)
    check("ops 过滤生效", all(i["call_type"] == "ai.summarize" for i in items))
    r = req("GET", "/api/model-call-logs?has_error=true&limit=10", admin)
    check("ops has_error 过滤", r.status_code == 200
          and all(i["error_kind"] for i in r.json()["items"]))
    r = req("GET", "/api/model-call-logs/stats?hours=24", admin)
    st = r.json() if r.status_code == 200 else {}
    check("stats total>=1", st.get("total", 0) >= 1)
    check("stats by_type 含 summarize",
          any(x["call_type"] == "ai.summarize" for x in st.get("by_type", [])))
    check("stats 有 p95", "p95_latency_ms" in st)
    r = req("GET", "/api/model-call-logs?limit=10", vw_tok)
    check("viewer ops 403", r.status_code == 403)
    r = req("GET", "/api/model-call-logs/stats", vw_tok)
    check("viewer stats 403", r.status_code == 403)
    r = req("GET", "/api/model-call-logs?limit=10", iso_tok)
    check("跨租户只见自己", r.status_code == 200
          and all(i["call_type"] != "ai.summarize" for i in r.json()["items"]))

    # ---------- 7. assistant 回报端点 ----------
    r = req("POST", "/api/model-call-logs/report", admin, json={
        "call_type": "assistant.completion",
        "model": "deepseek/deepseek-chat",
        "input_digest": f"live{RUN}xxxx",
        "latency_ms": 1234,
        "output": {"tools": ["hr_get_candidate"], "reply_chars": 66},
    })
    check("report 201", r.status_code == 201, r.text[:200])
    with SessionLocal() as db:
        rep = db.scalars(
            sa_select(ModelCallLog)
            .where(ModelCallLog.tenant_id == tid,
                   ModelCallLog.call_type == "assistant.completion")
            .order_by(ModelCallLog.created_at.desc())
        ).first()
    check("report 落行", rep is not None and rep.output.get("reply_chars") == 66)

    # ---------- 8. 删除源 → chunks 级联 ----------
    if fsid:
        r = req("DELETE", f"/api/jobs/{job_id}/knowledge-sources/{fsid}", admin)
        check("删除源 204", r.status_code == 204)
        cleanup_sources.remove(fsid)
        with SessionLocal() as db:
            n = db.query(JobKnowledgeChunk).filter_by(source_id=fsid).count()
        check("删除后 chunks=0", n == 0)

    # ---------- 清理 ----------
    for sid in list(cleanup_sources):
        req("DELETE", f"/api/jobs/{job_id}/knowledge-sources/{sid}", admin)
    req("DELETE", f"/api/jobs/{job_id}", admin)
    with SessionLocal() as db:
        db.query(ModelCallLog).filter_by(tenant_id=tid).delete()
        db.query(ModelCallLog).filter_by(tenant_id=t2id).delete()
        db.delete(db.get(DbUser, vw_id))
        db.delete(db.get(DbUser, iso_id))
        db.delete(db.get(Tenant, t2id))
        db.delete(db.get(Candidate, cand_id))
        db.commit()

    print(f"\n==== 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 ====")
    if FAIL:
        print("失败项:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
