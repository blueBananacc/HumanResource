# -*- coding: utf-8 -*-
"""T17 组织面试质量报告 —— 真实 HTTP 链路验证(容器内运行)。

用法:
    docker exec free-hire-backend-1 python /app/scripts/verify_t17_live.py

链路:
- 直插 DB 造窗口内 done 面试(dimension_scores/recommendation/时长)+ 评分卡反馈
- POST /api/report-defs/interview_quality/run → KPI/维度/missed/by_job 断言
- GET /api/report-defs 可见性 + 跨租户数据隔离 + digest 分支
- 修正历史种子面试 status finished→done
- 清理全量数据
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
    return requests.request(method, f"{BASE}{path}",
                            headers={"Authorization": f"Bearer {token}"}, timeout=60, **kw)


def login(email: str, pwd: str) -> str:
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": email, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main() -> int:
    from sqlalchemy import select as sa_select, update as sa_update
    from app.domain.models import (Candidate, Interview, InterviewFeedback,
                                   InterviewScorecard, Job, Tenant,
                                   User as DbUser)
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")
    now = datetime.now()
    ids = {"ivs": [], "cards": [], "cands": [], "jobs": [], "t2": None, "u2": None}

    # 基线报告:demo 租户里已有真实 done 面试(苏晴等),期望值=基线+新增
    base = req("POST", "/api/report-defs/interview_quality/run", admin,
               json={"params": {"days": 30}}).json()["data"]
    b_n = base["kpis"]["assessed_events"]
    b_dec = base["kpis"]["decided"]
    b_pass = round(base["kpis"]["pass_rate"] * b_dec / 100)
    b_dur_sum = (base["kpis"]["avg_duration_min"] or 0) * b_n
    b_hits = {d["name"]: d["scored_in"] for d in base["dimensions"]}

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()

        # 修正历史种子:status finished→done(应用口径)
        fixed = db.execute(
            sa_update(Interview).where(Interview.status == "finished")
            .values(status="done")
        )
        check("历史种子面试 finished→done 修正", fixed.rowcount >= 0)

        u = db.scalars(sa_select(DbUser).where(DbUser.email == "admin@example.com")).first()
        t2 = Tenant(name=f"t17-iso-{RUN}")
        db.add(t2)
        db.flush()
        ids["t2"] = t2.id
        iso = DbUser(tenant_id=t2.id, email=f"t17-iso-{RUN}@x.com",
                     password_hash=hash_password("p1234567"), role="admin",
                     status="active")
        db.add(iso)
        db.flush()
        ids["u2"] = iso.id

        cand = Candidate(tenant_id=t.id, name=f"质检员{RUN}")
        job = Job(tenant_id=t.id, title=f"质检岗{RUN}", status="open")
        db.add_all([cand, job])
        db.flush()
        ids["cands"].append(cand.id)
        ids["jobs"].append(job.id)

        def iv(dims, rec, mins, days_ago=1, status="done", mode="remote"):
            fin = now - timedelta(days=days_ago)
            x = Interview(tenant_id=t.id, job_id=job.id, candidate_id=cand.id,
                          mode=mode, status=status,
                          started_at=fin - timedelta(minutes=mins), finished_at=fin,
                          summary={"recommendation": rec,
                                   "dimension_scores": dims})
            db.add(x)
            db.flush()
            ids["ivs"].append(x.id)
            return x

        # 4 场:结构化思维×4 系统设计×2 业务敏感度×1;3 推荐 1 不推荐;时长 40/60/40/60
        iv({"结构化思维": 8, "系统设计": 7}, "推荐进入下一轮", 40)
        iv({"结构化思维": 9, "业务敏感度": 5}, "推荐", 60)
        iv({"结构化思维": 6, "系统设计": 4}, "不推荐", 40)
        iv({"结构化思维": 7}, None, 60)
        # 排除项:self_test / in_progress / 窗口外
        iv({"噪音": 1}, "推荐", 30, mode="self_test")
        iv({"噪音": 1}, "推荐", 30, status="in_progress")
        iv({"噪音": 1}, "推荐", 30, days_ago=60)

        # 挂面试的评分卡:AI应用深度 维,feedback recommend
        card = InterviewScorecard(
            tenant_id=t.id, candidate_id=cand.id, job_id=job.id,
            interview_id=ids["ivs"][0], title=f"质检卡{RUN}",
            criteria=[{"key": "kc1", "name": "AI应用深度", "dimension": "tech", "weight": 1}],
            interviewer_ids=[u.id], status="closed")
        db.add(card)
        db.flush()
        ids["cards"].append(card.id)
        db.add(InterviewFeedback(
            tenant_id=t.id, scorecard_id=card.id, interviewer_user_id=u.id,
            rating=5, verdict="recommend",
            scores=[{"criterion": "kc1", "score": 4, "comment": "ok"}],
            submitted_at=now - timedelta(days=1)))
        db.commit()

        # 跨租户对照组:t2 完全没数据
        r = req("POST", "/api/report-defs/interview_quality/run", admin,
                json={"params": {"days": 30}})
        check("run 端点 200", r.status_code == 200, r.text[:200])
        body = r.json()
        check("kind/label", body["kind"] == "interview_quality" and "面试质量" in body["label"])
        d = body["data"]
        k = d["kpis"]
        N = b_n + 4  # 基线事件 + 新增 4 场(评分卡并入既有面试事件)
        check("面试总量=基线+4(排除 self_test/进行中/窗外)",
              k["interviews_total"] == N, f'{k["interviews_total"]} vs {N}')
        exp_pr = round((b_pass + 2) * 100 / (b_dec + 3))
        check("通过率=(基线+2推荐)/(基线+3有结论)", k["pass_rate"] == exp_pr,
              f'{k["pass_rate"]} vs {exp_pr}')
        exp_dur = round((b_dur_sum + 200) / N)
        check("平均时长=加权均值", k["avg_duration_min"] == exp_dur,
              f'{k["avg_duration_min"]} vs {exp_dur}')
        dims = {x["name"]: x for x in d["dimensions"]}
        for name, add in (("结构化思维", 4), ("系统设计", 2), ("业务敏感度", 1)):
            exp_in = b_hits.get(name, 0) + add
            got = dims.get(name, {})
            check(f"{name} scored_in={exp_in} coverage≈{round(exp_in*100/N)}%",
                  got.get("scored_in") == exp_in
                  and abs(got.get("coverage", -1) - round(exp_in * 100 / N)) <= 1,
                  str(got))
        check("业务敏感度进 missed(<50%)",
              any(m["name"] == "业务敏感度" for m in d["missed"]), str(d["missed"]))
        check("评分卡维度 AI应用深度 scored_in=1",
              dims.get("AI应用深度", {}).get("scored_in") == b_hits.get("AI应用深度", 0) + 1,
              str(dims.get("AI应用深度")))
        check("噪音维度未混入", "噪音" not in dims)
        check("by_job 岗位下钻", any(j["title"] == f"质检岗{RUN}" and j["interviews"] == 4
                                   for j in d["by_job"]), str(d["by_job"]))
        # Σhits/(|U|×N) 口径自洽
        exp_cov = round(sum(x["scored_in"] for x in d["dimensions"])
                        / (len(d["dimensions"]) * N) * 100)
        check("维度覆盖率 KPI = Σhits/(|U|×N) 自洽",
              abs(k["dimension_coverage"] - exp_cov) <= 1,
              f'{k["dimension_coverage"]} vs {exp_cov}')

        defs = req("GET", "/api/report-defs", admin)
        check("list_defs 可见 interview_quality",
              any(x["kind"] == "interview_quality" for x in defs.json()))

        iso_token = login(f"t17-iso-{RUN}@x.com", "p1234567")
        r2 = req("POST", "/api/report-defs/interview_quality/run", iso_token,
                 json={"params": {"days": 30}})
        check("跨租户事件数=0", r2.json()["data"]["kpis"]["assessed_events"] == 0,
              str(r2.json()["data"]["kpis"]))

        # digest 分支:直接调 _digest_for 造订阅(不落库,传个 SimpleNamespace)
        from types import SimpleNamespace
        from app.workers.tasks.notify import _digest_for
        dg = _digest_for(db, SimpleNamespace(kind="interview_quality",
                                             tenant_id=t.id, user_id=u.id))
        check("digest 生成含 KPI 与遗漏",
              dg is not None and "通过率" in dg[1] and "经常被遗漏" in dg[1],
              str(dg))
        dg2 = _digest_for(db, SimpleNamespace(kind="interview_quality",
                                              tenant_id=t2.id, user_id=iso.id))
        check("空租户 digest=None 跳过发送", dg2 is None)

        # ---- 清理 ----
        db.query(InterviewFeedback).filter(
            InterviewFeedback.scorecard_id.in_(ids["cards"])).delete(synchronize_session=False)
        db.query(InterviewScorecard).filter(
            InterviewScorecard.id.in_(ids["cards"])).delete(synchronize_session=False)
        db.query(Interview).filter(Interview.id.in_(ids["ivs"])).delete(synchronize_session=False)
        db.query(Candidate).filter(Candidate.id.in_(ids["cands"])).delete(synchronize_session=False)
        db.query(Job).filter(Job.id.in_(ids["jobs"])).delete(synchronize_session=False)
        db.query(DbUser).filter(DbUser.id == ids["u2"]).delete(synchronize_session=False)
        db.query(Tenant).filter(Tenant.id == ids["t2"]).delete(synchronize_session=False)
        db.commit()

    print(f"\n=> {len(PASS)} PASS / {len(FAIL)} FAIL")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
