# -*- coding: utf-8 -*-
"""T16 对话式招聘 —— 真实 HTTP 链路验证(容器内运行)。

用法:
    docker exec free-hire-backend-1 python /app/scripts/verify_t16_live.py

链路:
- 画像 CRUD:建/查/改/生成/岗位转画像/跨租户 404
- 扫描:profile_id 扫描(排序+coverage 整词)+ inline spec + 门槛过滤 + runs 留痕
- summary-view:面试纪要聚合
- 清理全量数据
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
    return requests.request(method, f"{BASE}{path}",
                            headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import (Candidate, Interview, InterviewTurn, Job,
                                   Resume, SourcingRun, TalentProfile, Tenant,
                                   User as DbUser)
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        t2 = Tenant(name=f"t16-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso = DbUser(tenant_id=t2.id, email=f"t16-iso-{RUN}@x.com",
                     password_hash=hash_password("T16iso!pass"), role="admin",
                     status="active")
        db.add(iso)
        db.flush()
        t2id = t2.id
        iso_email = iso.email

        def mk_resume(name, text, skills):
            c = Candidate(tenant_id=t.id, name=f"{name}-{RUN}")
            db.add(c)
            db.flush()
            r = Resume(tenant_id=t.id, candidate_id=c.id,
                       file_name=f"{name}.pdf", file_size=1,
                       file_mime="application/pdf",
                       storage_key=f"t16-{name}-{RUN}", source="upload",
                       parse_status="done", parsed_text=text,
                       parsed_data={"skills": skills, "sections": {}})
            db.add(r)
            db.flush()
            return r.id, c.id

        rid_s, cid_s = mk_resume("高手", "10年 Python Go 微服务架构经验,本科",
                                ["Python", "Go"])
        rid_w, cid_w = mk_resume("新手", "1年 Django 入门,大专", ["Django"])
        db.commit()
        tid = t.id

    iso_tok = login(iso_email, "T16iso!pass")

    # ---- 画像 CRUD ----
    r = req("POST", "/api/talent-profiles", admin,
            json={"name": f"AI工程师画像-{RUN}",
                  "spec": {"skills": ["Python", "Go", "Kubernetes"],
                           "keywords": ["微服务"], "min_years": 2,
                           "education": "本科"}})
    check("建画像", r.status_code in (200, 201), f"{r.status_code} {r.text[:120]}")
    pid = r.json().get("id", "")

    r = req("GET", f"/api/talent-profiles?q=AI工程师", admin)
    check("画像列表模糊搜", any(p["id"] == pid for p in r.json()))
    r = req("GET", f"/api/talent-profiles/{pid}", iso_tok)
    check("画像跨租户 404", r.status_code == 404)

    r = req("POST", "/api/talent-profiles/generate", admin,
            json={"prompt": "招个 Go 后端工程师"})
    check("画像 AI 生成草稿", r.status_code == 200 and "spec" in r.json(),
          r.text[:150])

    # ---- 扫描(profile) ----
    r = req("POST", "/api/sourcing/scan", admin,
            json={"profile_id": pid, "limit": 5})
    check("扫描受理", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
    body = r.json() if r.status_code == 200 else {}
    items = body.get("items") or []
    check("短名单返回", len(items) >= 1, str(len(items)))
    top = next((i for i in items if "高手" in i["name"]), None)
    check("高手在列且分数最高", items and items[0]["name"].startswith("高手"),
          str([(i["name"], i["score"]) for i in items]))
    if top:
        check("高手 coverage 命中 Python+Go",
              {"Python", "Go"} <= set(top["coverage"]["hit"]),
              str(top["coverage"]))
        check("核心匹配点非空", bool(top["highlights"]))
    check("新手被本科门槛过滤", not any("新手" in i["name"] for i in items),
          str([(i["name"]) for i in items]))

    # ---- 扫描(inline spec + 留痕) ----
    r = req("POST", "/api/sourcing/scan", admin,
            json={"spec": {"skills": ["Django"], "min_years": 0, "education": ""},
                  "limit": 3})
    check("inline spec 扫描", r.status_code == 200 and
          any("新手" in i["name"] for i in r.json()["items"]))
    r = req("GET", "/api/sourcing/runs", admin)
    check("runs 留痕≥2", len(r.json()) >= 2)

    with SessionLocal() as db:
        p = db.get(TalentProfile, pid)
        check("画像 use_count 已记", (p.use_count or 0) >= 1 and p.last_used_at,
              f"use_count={p.use_count}")

    # ---- summary-view ----
    with SessionLocal() as db:
        j = Job(tenant_id=tid, title=f"T16岗位-{RUN}", status="open")
        db.add(j)
        db.flush()
        iv = Interview(tenant_id=tid, job_id=j.id, candidate_id=cid_s,
                       mode="remote", status="done",
                       summary={"recommendation": "推荐进入下一轮",
                                "dimension_scores": {"accuracy": 85},
                                "evidence_quotes": ["引用1"]})
        db.add(iv)
        db.flush()
        db.add(InterviewTurn(interview_id=iv.id, idx=1, level="tech",
                             question="讲讲 Go", answer="熟", scores={"accuracy": 85}))
        db.commit()
        iv_id, jid = iv.id, j.id
    r = req("GET", f"/api/interviews/{iv_id}/summary-view", admin)
    check("summary-view 200", r.status_code == 200, f"{r.status_code}")
    sv = r.json() if r.status_code == 200 else {}
    check("纪要含 AI 建议", sv.get("ai", {}).get("recommendation") == "推荐进入下一轮")
    check("纪要含逐题", len(sv.get("turns") or []) == 1)
    check("纪要含候选人/岗位", "高手" in sv.get("candidate_name", "") and
          "T16岗位" in sv.get("job_title", ""))
    r = req("GET", f"/api/interviews/{iv_id}/summary-view", iso_tok)
    check("纪要跨租户 404", r.status_code == 404)

    # ---- 清理 ----
    with SessionLocal() as db:
        db.query(InterviewTurn).filter(InterviewTurn.interview_id == iv_id).delete(synchronize_session=False)
        iv = db.get(Interview, iv_id)
        if iv:
            db.delete(iv)
        jj = db.get(Job, jid)
        if jj:
            db.delete(jj)
        db.query(SourcingRun).filter(SourcingRun.tenant_id == tid).delete(synchronize_session=False)
        pp = db.get(TalentProfile, pid)
        if pp:
            db.delete(pp)
        for rid_, cid_ in ((rid_s, cid_s), (rid_w, cid_w)):
            rr = db.get(Resume, rid_)
            if rr:
                db.delete(rr)
            cc = db.get(Candidate, cid_)
            if cc:
                db.delete(cc)
        t2o = db.get(Tenant, t2id)
        if t2o:
            db.query(DbUser).filter(DbUser.tenant_id == t2id).delete(synchronize_session=False)
            db.delete(t2o)
        db.commit()
    print("清理完成")

    print(f"\n=> {len(PASS)} 通过 / {len(FAIL)} 失败")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
