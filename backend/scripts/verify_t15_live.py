# -*- coding: utf-8 -*-
"""T15 CJK 整词匹配 + 关键词覆盖子分 —— 真实 HTTP 链路验证(容器内运行)。

用法:
    docker exec free-hire-backend-1 python /app/scripts/verify_t15_live.py

链路:
- DB 直插岗位(skills=[Python,Go,Django,Rust])+ 两份简历
  · r_go: 文本含 "Python、Go 语言" → 命中 2,缺 Django/Rust
  · r_dj: 文本含 "Django、Rust"  → Go 不得因子串误中 Django(整词边界)
- POST /api/matches/job/{id}/evaluate-all → 轮询至 done
- 断言 match.coverage:{score,hit,missing,total} + API 返回 coverage
- 清理全量数据
"""
from __future__ import annotations

import os
import sys
import time
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
    return requests.request(method, f"{BASE}{path}",
                            headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import (Candidate, Job, Resume, ResumeJobMatch,
                                   Tenant)
    from app.infra.db import SessionLocal

    admin = login("admin@example.com", "admin123456")

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        job = Job(tenant_id=t.id, title=f"T15后端-{RUN}",
                  description="后端开发", skills=["Python", "Go", "Django", "Rust"],
                  status="open")
        db.add(job)
        db.flush()
        jid = job.id

        def mk_resume(tag: str, text: str) -> str:
            c = Candidate(tenant_id=t.id, name=f"t15-{tag}-{RUN}")
            db.add(c)
            db.flush()
            r = Resume(tenant_id=t.id, candidate_id=c.id, file_name=f"{tag}.pdf",
                       file_size=1, file_mime="application/pdf",
                       storage_key=f"t15-{tag}-{RUN}", source="upload",
                       parse_status="done", parsed_text=text,
                       parsed_data={"skills": [], "sections": {}})
            db.add(r)
            db.flush()
            return r.id

        rid_go = mk_resume("go", "8年后端经验,精通 Python 与 Go 语言,"
                                 "本科 计算机科学,熟悉 PostgreSQL Redis")
        rid_dj = mk_resume("dj", "5年开发经验,熟悉 Django 框架与 Rust,"
                                  "本科 软件工程,熟悉 MySQL")
        db.commit()

    r = req("POST", f"/api/matches/job/{jid}/evaluate-all", admin)
    check("evaluate-all 受理", r.status_code in (200, 202), f"{r.status_code} {r.text[:120]}")

    rows = []
    for _ in range(60):
        rr = req("GET", f"/api/matches/job/{jid}?limit=20", admin)
        rows = [x for x in rr.json() if x["resume_id"] in (rid_go, rid_dj)]
        if len(rows) == 2 and all(x["status"] in ("done", "failed", "gated") for x in rows):
            break
        time.sleep(2)
    check("2 条 match 完成", len(rows) == 2 and all(x["status"] == "done" for x in rows),
          str([(x["resume_id"][-4:], x["status"]) for x in rows]))

    by_rid = {x["resume_id"]: x for x in rows}
    m_go, m_dj = by_rid.get(rid_go) or {}, by_rid.get(rid_dj) or {}

    cov_go = m_go.get("coverage") or {}
    check("go 简历 coverage 存在", bool(cov_go))
    check("go: 命中 Python+Go", set(cov_go.get("hit") or []) == {"Python", "Go"},
          str(cov_go.get("hit")))
    check("go: 缺 Django+Rust", set(cov_go.get("missing") or []) == {"Django", "Rust"},
          str(cov_go.get("missing")))
    check("go: score=50", cov_go.get("score") == 50, str(cov_go.get("score")))
    check("go: total=4", cov_go.get("total") == 4)

    cov_dj = m_dj.get("coverage") or {}
    check("dj 简历 coverage 存在", bool(cov_dj))
    check("dj: 命中 Django+Rust", set(cov_dj.get("hit") or []) == {"Django", "Rust"},
          str(cov_dj.get("hit")))
    check("dj: Go 在缺失(整词边界:不因 Django 误中)",
          "Go" in (cov_dj.get("missing") or []), str(cov_dj))
    check("dj: score=50", cov_dj.get("score") == 50, str(cov_dj.get("score")))

    # 单条 match detail 也带 coverage
    if m_go.get("id"):
        r1 = req("GET", f"/api/matches/{m_go['id']}", admin)
        check("detail coverage 一致", (r1.json().get("coverage") or {}).get("score") == 50)

    # ---- 清理 ----
    with SessionLocal() as db:
        db.query(ResumeJobMatch).filter(ResumeJobMatch.job_id == jid).delete(synchronize_session=False)
        for rid_ in (rid_go, rid_dj):
            rr_ = db.get(Resume, rid_)
            if rr_:
                cid = rr_.candidate_id
                db.delete(rr_)
                cc = db.get(Candidate, cid)
                if cc:
                    db.delete(cc)
        jj = db.get(Job, jid)
        if jj:
            db.delete(jj)
        db.commit()
    print("清理完成")

    print(f"\n=> {len(PASS)} 通过 / {len(FAIL)} 失败")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
