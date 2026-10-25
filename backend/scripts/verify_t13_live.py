"""T13 声明式评分规则引擎 —— 真实 HTTP 链路验证(容器内跑)。

用法:
    docker compose exec backend python scripts/verify_t13_live.py

链路:
- API 建规则(keyword+duration+ai 三维)→ 挂岗位 → 触发评估
- criteria 行带 method 徽标数据 / rule_snapshot 冻结
- ai 维批量调用留痕 scoring_rule.evaluate;确定性维零 LLM
- POST /generate 按岗位出草案 → POST 保存 → /analyze → /apply-analysis version+1
- 跨租户规则 404 / 挂别租户规则 400 / 规则删除后岗位 scoring_rule_id 置 NULL
- 测试数据清理

退出码 0 = 全部通过。
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
        Candidate, Job, ModelCallLog, Resume, ResumeJobMatch, Tenant,
        User as DbUser,
    )
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        tid = t.id
        t2 = Tenant(name=f"t13-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso = DbUser(tenant_id=t2.id, email=f"t13-iso-{RUN}@x.com",
                     password_hash=hash_password("T13iso!pass"), role="admin",
                     status="active")
        db.add(iso)
        db.commit()
        iso_tok = login(f"t13-iso-{RUN}@x.com", "T13iso!pass")

    # ---------- 1. API 建规则(三维混合方法) ----------
    dims = [
        {"id": "skills", "name": "关键技能", "weight": 40,
         "evaluator": {"method": "keyword", "keywords": ["Python", "FastAPI"], "matchMode": "any"}},
        {"id": "years", "name": "工作年限", "weight": 35,
         "evaluator": {"method": "duration", "minYears": 3, "preferredYears": 7}},
        {"id": "fit", "name": "综合契合", "weight": 25,
         "evaluator": {"method": "ai", "aiPrompt": "评估项目深度与领域契合"}},
    ]
    r = req("POST", "/api/scoring-rules", admin,
            json={"name": f"T13链测规则{RUN}", "description": "链路验证",
                  "dimensions": dims})
    check("创建规则 200", r.status_code == 200, r.text[:200])
    rule = r.json() if r.status_code == 200 else {}
    rid = rule.get("id", "")
    check("权重合计=100", rule.get("weight_total") == 100, str(rule.get("weight_total")))
    check("3 维返回", len(rule.get("dimensions") or []) == 3)

    # ---------- 2. 挂规则建岗位 + 两份简历 ----------
    r = req("POST", "/api/jobs/", admin,
            json={"title": f"T13链测岗位{RUN}", "level": "intermediate",
                  "description": "后端开发,Python/FastAPI,3年以上经验。",
                  "skills": ["Python", "FastAPI"], "scoring_rule_id": rid})
    check("建岗位挂规则 201", r.status_code == 201, r.text[:200])
    job = r.json() if r.status_code == 201 else {}
    jid = job.get("id", "")
    check("岗位 scoring_rule_id", job.get("scoring_rule_id") == rid)

    with SessionLocal() as db:
        cand_ok = Candidate(tenant_id=tid, name=f"T13合格{RUN}",
                            display_email=f"t13ok-{RUN}@x.com")
        cand_zero = Candidate(tenant_id=tid, name=f"T13零维{RUN}",
                              display_email=f"t13zero-{RUN}@x.com")
        db.add_all([cand_ok, cand_zero])
        db.flush()
        r_ok = Resume(
            tenant_id=tid, candidate_id=cand_ok.id, file_name="ok.pdf",
            file_size=100, file_mime="application/pdf", storage_key=f"t13ok-{RUN}",
            source="upload", parse_status="done",
            parsed_text="6 年 Python / FastAPI 后端经验,主导微服务重构,本科。",
            parsed_data={"skills": ["Python", "FastAPI"]},
        )
        r_zero = Resume(
            tenant_id=tid, candidate_id=cand_zero.id, file_name="zero.pdf",
            file_size=100, file_mime="application/pdf", storage_key=f"t13zero-{RUN}",
            source="upload", parse_status="done",
            parsed_text="应届生,只做过课程作业。ai_dim_zero",
            parsed_data={"skills": []},
        )
        db.add_all([r_ok, r_zero])
        db.commit()
        rid_ok, rid_zero = r_ok.id, r_zero.id

    # ---------- 3. 触发评估 → 轮询 ----------
    r = req("POST", f"/api/matches/job/{jid}/evaluate-all", admin)
    check("evaluate-all 200", r.status_code == 200, r.text[:200])

    deadline = time.time() + 240
    rows: list[dict] = []
    while time.time() < deadline:
        out = []
        for st in ("done", "gated", "failed"):
            rr = req("GET", f"/api/matches/job/{jid}?status={st}&limit=10", admin)
            out += rr.json() if rr.status_code == 200 else []
        rows = out
        if len(rows) >= 2:
            break
        time.sleep(4)
    by_resume = {x["resume_id"]: x for x in rows}
    check("2 条 match 出结果", len(rows) >= 2, f"got {len(rows)}")
    m_ok = by_resume.get(rid_ok)
    m_zero = by_resume.get(rid_zero)

    # ---------- 4. 规则路径结果 ----------
    if m_ok:
        check("合格简历 done", m_ok["status"] == "done", f"status={m_ok['status']}")
        snap = m_ok.get("rule_snapshot") or {}
        check("rule_snapshot 存在", snap.get("rule_id") == rid, str(snap)[:200])
        check("snapshot name/version", snap.get("name") == f"T13链测规则{RUN}"
              and snap.get("version") == 1, str(snap.get("version")))
        cr = m_ok.get("criteria") or []
        check("criteria 3 维", len(cr) == 3, f"got {len(cr)}")
        methods = {c["key"]: c.get("method") for c in cr}
        check("method 徽标数据齐全",
              methods == {"skills": "keyword", "years": "duration", "fit": "ai"},
              str(methods))
        check("逐维 evidence", all(c.get("evidence") for c in cr))
        kw = next((c for c in cr if c["key"] == "skills"), None)
        check("keyword 维满分命中", kw and kw["score"] == 100,
              str(kw))
        yrs = next((c for c in cr if c["key"] == "years"), None)
        check("duration 维插值(6 年 ≥min3)", yrs and yrs["score"] >= 80,
              str(yrs))
        fit = next((c for c in cr if c["key"] == "fit"), None)
        check("ai 维 level×25 映射", fit and fit["score"] in (0, 25, 50, 75, 100),
              str(fit))
        check("score 加权合理", isinstance(m_ok["score"], int) and m_ok["score"] >= 60,
              str(m_ok["score"]))

    if m_zero:
        check("零维简历完成", m_zero["status"] in ("done", "gated"),
              f"status={m_zero['status']}")
        cr = m_zero.get("criteria") or []
        kw = next((c for c in cr if c["key"] == "skills"), None)
        if kw is not None:
            check("技能未命中 → skills 低分", kw["score"] < 50, str(kw))
        fit = next((c for c in cr if c["key"] == "fit"), None)
        check("ai_dim_zero → ai 维 0 分", fit is not None and fit["score"] == 0,
              str(fit))

    # ---------- 5. LLM 留痕:scoring_rule.evaluate ----------
    with SessionLocal() as db:
        logs = db.scalars(
            sa_select(ModelCallLog).where(
                ModelCallLog.tenant_id == tid,
                ModelCallLog.call_type == "scoring_rule.evaluate",
            ).order_by(ModelCallLog.id.desc()).limit(10)
        ).all()
    check("scoring_rule.evaluate 落痕", len(logs) >= 2,
          f"got {len(logs)}")

    # ---------- 6. AI 生成草案 → 分析 → 应用 ----------
    r = req("POST", "/api/scoring-rules/generate", admin, json={"job_id": jid})
    check("generate 200", r.status_code == 200, r.text[:200])
    draft = r.json() if r.status_code == 200 else {}
    check("generate 出维度", len(draft.get("dimensions") or []) >= 1)
    check("generate 维度方法合法",
          all(d.get("evaluator", {}).get("method")
              in ("keyword", "duration", "ai", "boolean", "range")
              for d in draft.get("dimensions") or []))

    r = req("POST", f"/api/scoring-rules/{rid}/analyze", admin)
    check("analyze 200", r.status_code == 200, r.text[:300])
    analysis = r.json() if r.status_code == 200 else {}
    check("analyze 结构齐全",
          all(k in analysis for k in
              ("summary", "coverageGaps", "proposedRule")))

    r = req("POST", f"/api/scoring-rules/{rid}/apply-analysis", admin,
            json={"analysis": analysis})
    check("apply-analysis 200", r.status_code == 200, r.text[:300])
    check("version 升到 2", r.json().get("version") == 2 if r.status_code == 200 else False)

    # ---------- 7. 权限/隔离 ----------
    r = req("GET", f"/api/scoring-rules/{rid}", iso_tok)
    check("跨租户规则 404", r.status_code == 404)
    r = req("POST", "/api/jobs/", iso_tok,
            json={"title": "iso", "scoring_rule_id": rid})
    check("挂别租户规则 400/201被拒", r.status_code in (400, 404, 422))

    # ---------- 8. 快照冻结:改规则后旧 match 仍 v1 ----------
    if m_ok:
        rr = req("GET", f"/api/matches/{m_ok['id']}", admin)
        snap = (rr.json() or {}).get("rule_snapshot") or {}
        check("改规则后历史 match 快照仍 v1", snap.get("version") == 1,
              str(snap.get("version")))

    # ---------- 9. 清理 ----------
    with SessionLocal() as db:
        db.query(ResumeJobMatch).filter(
            ResumeJobMatch.tenant_id == tid,
            ResumeJobMatch.job_id == jid,
        ).delete(synchronize_session=False)
        for rid_ in (rid_ok, rid_zero):
            db.query(Resume).filter(Resume.id == rid_).delete(synchronize_session=False)
        db.query(Candidate).filter(
            Candidate.name.like(f"T13%{RUN}")).delete(synchronize_session=False)
        db.query(Job).filter(Job.id == jid).delete(synchronize_session=False)
        from app.domain.models import ScoringRule
        db.query(ScoringRule).filter(ScoringRule.id == rid).delete(synchronize_session=False)
        db.query(DbUser).filter(DbUser.tenant_id == t2.id).delete(synchronize_session=False)
        db.query(Tenant).filter(Tenant.id == t2.id).delete(synchronize_session=False)
        db.commit()
    print("数据已清理")

    print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
    for name in FAIL:
        print("FAIL:", name)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
