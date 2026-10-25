"""T9 真实 HTTP 链路验证(容器内运行)。

用法:
    docker compose exec backend python scripts/verify_t9_live.py

覆盖:
- 评分卡:创建(维度规范化+面试官校验)/mine/双面试官提交(分歧)/
  upsert 改判/AI 聚合(降级回落)/维度冻结/关闭拒收/ESS 404/跨租户 404/删除
- 全局搜索:reindex/跨实体命中+facets/实体过滤/更新经 audit 漏斗同步/
  删除同步/ESS 权限回读为空/跨租户隔离/定义停用/字段白名单拒绝/index-status

退出码 0 = 全过。自动清理自建数据。
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
    from app.domain.models import Candidate, Tenant, User as DbUser
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    admin = login("admin@example.com", "admin123456")

    # ---------- 0. 测试数据:候选人 + 岗位 + 2 面试官 + ESS + 他租户 ----------
    iv_ids, ess_id, iso_id = [], None, None
    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        tid = t.id
        cand = Candidate(tenant_id=tid, name=f"T9链测人{RUN}",
                         display_email=f"t9c-{RUN}@example.com")
        db.add(cand)
        for i in (1, 2):
            u = DbUser(tenant_id=tid, email=f"t9-iv{i}-{RUN}@example.com",
                       password_hash=hash_password("T9iv!pass"), role="interviewer",
                       status="active")
            db.add(u)
            db.flush()
            iv_ids.append(u.id)
        ess_u = DbUser(tenant_id=tid, email=f"t9-ess-{RUN}@example.com",
                       password_hash=hash_password("T9ess!pass"), role="employee",
                       status="active")
        db.add(ess_u)
        db.flush()
        ess_id = ess_u.id
        t2 = Tenant(name=f"t9-iso-{RUN}")
        db.add(t2)
        db.flush()
        iso_u = DbUser(tenant_id=t2.id, email=f"t9-iso-{RUN}@example.com",
                       password_hash=hash_password("T9iso!pass"), role="admin",
                       status="active")
        db.add(iso_u)
        db.flush()
        iso_id = iso_u.id
        t2_id = t2.id
        db.commit()
        cand_id = cand.id
    iv1, iv2 = login(f"t9-iv1-{RUN}@example.com", "T9iv!pass"), \
        login(f"t9-iv2-{RUN}@example.com", "T9iv!pass")
    ess = login(f"t9-ess-{RUN}@example.com", "T9ess!pass")
    iso = login(f"t9-iso-{RUN}@example.com", "T9iso!pass")

    r = req("POST", "/api/jobs/", admin,
            json={"title": f"T9链测岗{RUN}", "level": "intermediate"})
    job_id = r.json()["id"] if r.status_code == 201 else None
    check("建岗位", r.status_code == 201, r.text[:200])

    sc_id = None
    try:
        # ================================================== 评分卡
        # ---------- 1. 创建 ----------
        r = req("POST", "/api/scorecards", admin, json={
            "candidate_id": cand_id, "job_id": job_id,
            "title": f"T9一面·技术面{RUN}",
            "criteria": [
                {"name": "系统设计", "dimension": "tech", "weight": 2},
                {"name": "沟通表达", "dimension": "soft"},
                {"name": "Ownership", "dimension": "general"},
            ],
            "interviewer_ids": iv_ids,
        })
        check("创建评分卡 201", r.status_code == 201, r.text[:300])
        sc = r.json() if r.status_code == 201 else {}
        sc_id = sc.get("id")
        keys = [c["key"] for c in sc.get("criteria", [])]
        check("criteria 自动生 key", len(keys) == 3 and all(keys))
        check("面试官就位", {i["user_id"] for i in sc.get("interviewers", [])} == set(iv_ids))

        # 非本租户用户当面试官 → 400
        r = req("POST", "/api/scorecards", admin, json={
            "candidate_id": cand_id, "title": "x", "interviewer_ids": [iso_id]})
        check("他租户用户做面试官 400", r.status_code == 400, r.text[:200])

        # ---------- 2. /mine ----------
        r = req("GET", "/api/scorecards/mine", iv1)
        check("面试官 mine 可见", any(x["id"] == sc_id for x in r.json()), r.text[:200])
        r = req("GET", "/api/scorecards/mine", ess)
        check("ESS mine 为空", r.status_code == 200 and r.json() == [])

        # ---------- 3. 未指派/无权限制 ----------
        r = req("POST", f"/api/scorecards/{sc_id}/feedback", ess,
                json={"rating": 4, "verdict": "recommend", "scores": []})
        check("ESS 交反馈 → 404(卡不可见)", r.status_code == 404, f"{r.status_code}")

        # ---------- 4. 双面试官提交(分歧) ----------
        r = req("POST", f"/api/scorecards/{sc_id}/feedback", iv1, json={
            "rating": 5, "verdict": "recommend",
            "scores": [{"criterion": k, "score": 5, "comment": "很强"} for k in keys],
            "notes": "系统设计非常扎实",
        })
        check("面试官1 提交反馈", r.status_code == 200, r.text[:300])
        check("聚合 1 份", r.json()["aggregate"]["feedback_count"] == 1)

        r = req("POST", f"/api/scorecards/{sc_id}/feedback", iv2, json={
            "rating": 2, "verdict": "reject",
            "scores": [
                {"criterion": keys[0], "score": 2},
                {"criterion": keys[1], "score": 3},
                {"criterion": keys[2], "score": 3},
            ],
        })
        check("面试官2 提交反馈", r.status_code == 200)
        agg = r.json()["aggregate"]
        check("聚合:总均分 3.5", agg["avg_rating"] == 3.5, str(agg))
        check("聚合:维度分歧标记", any(d["kind"] == "criterion" for d in agg["disagreements"]))
        check("聚合:表态分歧标记", any(d["kind"] == "verdict" for d in agg["disagreements"]))

        # 非法维度 key 拒绝
        r = req("POST", f"/api/scorecards/{sc_id}/feedback", iv2, json={
            "rating": 3, "verdict": "hold",
            "scores": [{"criterion": "bogus", "score": 3}],
        })
        check("未知维度 key 400", r.status_code == 400)

        # upsert 改判
        r = req("POST", f"/api/scorecards/{sc_id}/feedback", iv2, json={
            "rating": 3, "verdict": "hold", "scores": []})
        check("反馈 upsert 不重复计数",
              r.status_code == 200 and r.json()["aggregate"]["feedback_count"] == 2)

        # ---------- 5. 维度冻结 ----------
        r = req("PATCH", f"/api/scorecards/{sc_id}", admin,
                json={"criteria": [{"name": "新维度"}]})
        check("有反馈后维度冻结 400", r.status_code == 400)
        r = req("PATCH", f"/api/scorecards/{sc_id}", admin, json={"title": f"T9一面·技术面{RUN}·改"})
        check("非维度字段仍可改", r.status_code == 200)

        # ---------- 6. AI 聚合 ----------
        r = req("POST", f"/api/scorecards/{sc_id}/synthesize", admin)
        check("synthesize 200", r.status_code == 200, r.text[:300])
        summ = r.json().get("ai_summary") or {}
        check("ai_summary 落库", bool(summ.get("summary")) and summ.get("feedback_count") == 2,
              str(summ)[:200])
        # interviewer 角色带 write_interview,可调 synthesize;ESS 无权限
        check("ESS 不可调 synthesize 403",
              req("POST", f"/api/scorecards/{sc_id}/synthesize", ess).status_code == 403)

        # ---------- 7. 乐观锁 ----------
        sc_now = req("GET", f"/api/scorecards/{sc_id}", admin).json()
        r = req("PATCH", f"/api/scorecards/{sc_id}", admin,
                json={"note": "x", "base_updated_at": "2000-01-01T00:00:00"})
        check("乐观锁 409", r.status_code == 409)

        # ---------- 8. 列表过滤 ----------
        r = req("GET", f"/api/scorecards?candidate_id={cand_id}", admin)
        check("按候选人过滤", any(x["id"] == sc_id for x in r.json().get("items", [])))
        r = req("GET", "/api/scorecards?status=closed", admin)
        check("closed 过滤不含 open 卡",
              all(x["status"] == "closed" for x in r.json().get("items", [])))

        # ---------- 9. 关闭后拒收 ----------
        r = req("PATCH", f"/api/scorecards/{sc_id}", admin, json={"status": "closed"})
        check("关闭评分卡", r.status_code == 200 and r.json()["status"] == "closed")
        r = req("POST", f"/api/scorecards/{sc_id}/feedback", iv1,
                json={"rating": 4, "verdict": "recommend", "scores": []})
        check("关闭后拒收反馈 400", r.status_code == 400)

        # ---------- 10. 跨租户 ----------
        check("他租户 GET 404",
              req("GET", f"/api/scorecards/{sc_id}", iso).status_code == 404)
        check("他租户 PATCH 404",
              req("PATCH", f"/api/scorecards/{sc_id}", iso,
                  json={"title": "x"}).status_code == 404)
        check("他租户列表不含本租户卡",
              all(x["id"] != sc_id for x in
                  req("GET", "/api/scorecards", iso).json().get("items", [])))

        # ---------- 11. 审计留痕 ----------
        r = req("GET", "/api/audit/events?entity_type=interview_scorecard&limit=20", admin)
        evs = r.json().get("items", []) if r.status_code == 200 else []
        actions = {e["action"] for e in evs}
        check("审计含 create/feedback/synthesize/update",
              {"create", "feedback", "synthesize", "update"} <= actions, str(actions))

        # ================================================== 全局搜索
        token = f"T9链测人{RUN}"
        # ---------- 12. reindex + 命中 ----------
        r = req("POST", "/api/search/reindex", admin, json={})
        check("reindex 200", r.status_code == 200, r.text[:200])
        r = req("GET", f"/api/search?q={token}", admin)
        body = r.json() if r.status_code == 200 else {}
        check("搜到候选人", any(h["entity"] == "talents" and token in h["title"]
                              for h in body.get("items", [])), r.text[:300])
        check("facets 含 talents",
              any(f["entity"] == "talents" for f in body.get("facets", [])))
        r = req("GET", f"/api/search?q={token}&entity=talents", admin)
        check("entity 过滤生效",
              all(h["entity"] == "talents" for h in r.json().get("items", [])))
        r = req("GET", f"/api/search?q={token}&entity=nope", admin)
        check("未知 entity 404", r.status_code == 404)

        # ---------- 13. 更新经 audit 漏斗同步 ----------
        new_mail = f"t9new-{RUN}@example.com"
        r = req("PATCH", f"/api/entities/talents/{cand_id}", admin,
                json={"fields": {"display_email": new_mail}})
        check("PATCH talents 字段", r.status_code == 200, r.text[:200])
        r = req("GET", f"/api/search?q={new_mail}", admin)
        check("更新后新值可搜(事件增量)",
              any(h["id"] == cand_id for h in r.json().get("items", [])), r.text[:300])

        # ---------- 14. 权限回读收敛 ----------
        r = req("GET", f"/api/search?q={token}", ess)
        check("ESS 搜索被权限回读清零",
              r.status_code == 200 and r.json()["total"] == 0, r.text[:200])
        r = req("GET", f"/api/search?q={token}", iso)
        check("跨租户搜索 0 命中",
              r.status_code == 200 and r.json()["total"] == 0)

        # ---------- 15. 定义管理 ----------
        r = req("GET", "/api/search/definitions", admin)
        check("definitions 列出实体",
              r.status_code == 200 and any(d["entity"] == "talents" for d in r.json()))
        r = req("PATCH", "/api/search/definitions/talents", admin,
                json={"fields": [{"name": "password_hash", "weight": 9}]})
        check("白名单外字段 400", r.status_code == 400)
        r = req("PATCH", "/api/search/definitions/jobs", admin,
                json={"label": "招聘职位", "fields": [{"name": "title", "weight": 5}]})
        check("定义覆盖 200 + version 递增",
              r.status_code == 200 and r.json().get("version", 0) >= 1)
        r = req("GET", "/api/search/index-status", admin)
        check("index-status 返回实体行",
              r.status_code == 200 and any(x["entity"] == "talents" for x in r.json()))
        check("ESS 无权限访问 definitions",
              req("GET", "/api/search/definitions", ess).status_code == 403)
        check("ESS 无权限 reindex",
              req("POST", "/api/search/reindex", ess, json={}).status_code == 403)

        # ---------- 16. 删除同步 ----------
        r = req("DELETE", f"/api/talents/{cand_id}", admin)
        check("删候选人", r.status_code in (200, 204), f"{r.status_code}")
        r = req("GET", f"/api/search?q={new_mail}", admin)
        check("删除后搜不到", r.json().get("total", 0) == 0, r.text[:200])

    finally:
        # ---------- 清理 ----------
        if sc_id:
            req("DELETE", f"/api/scorecards/{sc_id}", admin)
        if job_id:
            req("DELETE", f"/api/jobs/{job_id}", admin)
        with SessionLocal() as db:
            t2 = db.get(Tenant, t2_id)
            if t2:
                db.delete(t2)  # 级联删其下 iso 用户
                db.flush()
            for u in db.scalars(
                    sa_select(DbUser).where(DbUser.email.like(f"t9-%-{RUN}@example.com"))).all():
                db.delete(u)
            db.commit()
        # 候选人若还在(删除失败时兜底)
        try:
            req("DELETE", f"/api/talents/{cand_id}", admin)
        except Exception:
            pass

    print(f"\n===== T9 LIVE: {len(PASS)} passed, {len(FAIL)} failed =====")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  - ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
