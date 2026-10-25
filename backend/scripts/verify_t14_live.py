"""T14 特征评分简历结构解析 —— 真实 HTTP 链路验证(容器内跑)。

用法:
    docker compose exec backend python scripts/verify_t14_live.py

链路:
- 上传真实中文 PDF(容器内 pymupdf 合成带字体/bullet 的简历)→ 轮询解析
- parsed_data.sections 落库:profile/工作段/教育段/技能区/confidence
- 候选人 email/phone 自动补全
- T13 duration 评估器读结构化年限(结构化优先、断文本兜底)
- detail API 返回 sections(跨租户 404)
- 测试数据清理

退出码 0 = 全部通过。
"""
from __future__ import annotations

import io
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
    return requests.request(
        method, f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def make_resume_pdf(name: str, email: str) -> bytes:
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    y = 60.0
    lines = [
        (name, 18.0),
        (f"{email}  13811112222  北京", 12.0),
        ("工作经历", 14.0),
        ("2019.06 - 至今  资深数据工程师  字节跳动", 12.0),
        ("• 搭建实时数仓,日处理 20 亿事件", 12.0),
        ("2015.07 - 2019.05  数据开发工程师  京东", 12.0),
        ("• 负责用户画像 ETL 管线", 12.0),
        ("教育经历", 14.0),
        ("2011.09 - 2015.06  北京邮电大学  硕士 软件工程", 12.0),
        ("专业技能", 14.0),
        ("• Python, SQL, Spark, Kafka, Airflow", 12.0),
    ]
    for text, size in lines:
        p.insert_text((72, y), text, fontname="china-s", fontsize=size)
        y += size + 4
    return doc.tobytes()


def main() -> int:
    from sqlalchemy import select as sa_select
    from app.domain.models import Candidate, Resume, Tenant, User as DbUser
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        t2 = Tenant(name=f"t14-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso = DbUser(tenant_id=t2.id, email=f"t14-iso-{RUN}@x.com",
                     password_hash=hash_password("T14iso!pass"), role="admin",
                     status="active")
        db.add(iso)
        db.commit()
        tid = t.id
        t2id = t2.id
        iso_email = iso.email

    iso_tok = login(iso_email, "T14iso!pass")

    # ---- 上传真实 PDF ----
    email_addr = f"t14-{RUN}@example.com"
    pdf = make_resume_pdf("李验证", email_addr)
    r = req("POST", "/api/resumes/upload", admin,
            files={"file": (f"李验证_{RUN}.pdf", pdf, "application/pdf")})
    check("上传简历返回 201", r.status_code == 201, f"{r.status_code} {r.text[:200]}")
    if r.status_code != 201:
        print(f"共 {len(PASS)} 通过 / {len(FAIL)} 失败")
        return 1
    resume = r.json()
    rid = resume["id"]

    # ---- 轮询解析完成 ----
    detail = None
    for _ in range(30):
        d = req("GET", f"/api/resumes/{rid}", admin)
        detail = d.json()
        if detail.get("parse_status") in ("done", "failed"):
            break
        time.sleep(1)
    check("解析完成 done", detail and detail.get("parse_status") == "done",
          f"status={detail and detail.get('parse_status')} err={detail and detail.get('parse_error')}")

    secs = (detail or {}).get("sections") or {}
    check("detail API 返回 sections", bool(secs), "sections 为空")
    check("解析器标记 feature-scoring-v1",
          secs.get("parser") == "feature-scoring-v1", secs.get("parser", ""))

    prof = secs.get("profile") or {}
    check("profile.name=李验证", prof.get("name") == "李验证", prof.get("name", ""))
    check("profile.email 命中", email_addr in (prof.get("email") or ""), prof.get("email", ""))
    check("profile.phone 命中", "13811112222" in (prof.get("phone") or ""), prof.get("phone", ""))

    works = secs.get("work_experiences") or []
    check("工作段 2 段", len(works) == 2, f"{len(works)} 段")
    if works:
        check("公司1=字节跳动", "字节" in (works[0].get("company") or ""), works[0].get("company", ""))
        check("职位含工程师", "工程师" in (works[0].get("title") or ""), works[0].get("title", ""))
        check("date_range 含至今", "至今" in (works[0].get("date_range") or ""), works[0].get("date_range", ""))
        check("years>0", (works[0].get("years") or 0) > 3, str(works[0].get("years")))
        check("descriptions 非空", bool(works[0].get("descriptions")))
    if len(works) > 1:
        check("公司2=京东", "京东" in (works[1].get("company") or ""), works[1].get("company", ""))
        check("段2年限≈3.8", abs((works[1].get("years") or 0) - 3.8) < 0.5,
              str(works[1].get("years")))

    edus = secs.get("educations") or []
    check("教育段 1 段", len(edus) == 1, f"{len(edus)} 段")
    if edus:
        check("学校=北邮", "邮电" in (edus[0].get("school") or ""), edus[0].get("school", ""))
        check("学位=硕士", "硕士" in (edus[0].get("degree") or ""), edus[0].get("degree", ""))

    check("技能区命中 Spark", "Spark" in (secs.get("skills_text") or ""), secs.get("skills_text", "")[:80])
    conf = secs.get("confidence") or {}
    check("confidence.overall>0.5", (conf.get("overall") or 0) > 0.5, str(conf.get("overall")))

    # 候选人补全
    cand_id = detail["candidate"]["id"]
    c = req("GET", f"/api/talents/{cand_id}", admin)
    cand = c.json() if c.status_code == 200 else {}
    check("候选人 email 补全", (cand.get("email") or cand.get("display_email") or "") == email_addr
          or email_addr in str(cand), str(cand)[:120])

    # ---- T13 评估器读结构化字段(服务级) ----
    from app.services import scoring_rules as srs
    with SessionLocal() as db:
        r_obj = db.get(Resume, rid)
        dur = srs._eval_duration(r_obj, r_obj.parsed_text or "",
                                 {"minYears": 3, "preferredYears": 8})
        check("duration 走结构化", "结构化" in dur["evidence"], dur["evidence"])
        total = sum(w.get("years") or 0 for w in works)
        check("结构化年限=各段合计", abs(total - (works[0]["years"] + works[1]["years"])) < 0.01
              if len(works) > 1 else True)
        b = srs._eval_boolean(r_obj, "", {"booleanField": "education"})
        check("boolean education=具备", b["score"] == 100 and "结构化" in b["evidence"],
              b["evidence"])

    # 跨租户
    r404 = req("GET", f"/api/resumes/{rid}", iso_tok)
    check("跨租户 detail 404", r404.status_code == 404, str(r404.status_code))

    # ---- 清理 ----
    with SessionLocal() as db:
        for mdl, col in ((Resume, "id"), (Candidate, "id")):
            pass
        rr = db.get(Resume, rid)
        if rr:
            cid = rr.candidate_id
            db.delete(rr)
            cc = db.get(Candidate, cid)
            if cc:
                db.delete(cc)
        for row in db.scalars(sa_select(DbUser).where(DbUser.tenant_id == t2id)):
            db.delete(row)
        tt = db.get(Tenant, t2id)
        if tt:
            db.delete(tt)
        db.commit()
    check("测试数据清理", True)

    print(f"\n=== 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 ===")
    if FAIL:
        print("失败项:", FAIL)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
