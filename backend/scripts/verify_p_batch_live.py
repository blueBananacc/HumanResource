# -*- coding: utf-8 -*-
"""P 批次 live 验证(打真 backend 容器 :8000)。

跑全链路:建员工绑账号 → 手动加技能 → insight(mock 不行,live 用
extract_skills=false + 直接 API 造建议)→ 确认建议 → 项目 CRUD/成员 →
done → talent/search → 本人脱敏视图。

用法: docker exec free-hire-backend-1 python scripts/verify_p_batch_live.py
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import date, timedelta

import urllib.request
import json

BASE = os.environ.get("API_BASE", "http://localhost:8000")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = ""):
    results.append((name, ok, note))
    print(("PASS " if ok else "FAIL ") + name + (f"  {note}" if note else ""))


def req(method: str, path: str, token: str | None = None,
        body: dict | None = None, expect: tuple = (200, 201)):
    r = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(r) as resp:
            code, data = resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        code, data = e.code, json.loads(e.read() or b"{}")
    return code, data


def main():
    # 直接走 DB 造用户/员工(与 verify_e_batch 同风格)
    sys.path.insert(0, "/app")
    from app.domain.models import Employee, Tenant, User  # noqa
    from app.infra.db import SessionLocal  # noqa
    from app.infra.security import create_access_token, hash_password  # noqa

    db = SessionLocal()
    tag = uuid.uuid4().hex[:6]
    t = Tenant(name=f"p-live-{tag}")
    db.add(t)
    db.flush()
    admin = User(tenant_id=t.id, email=f"adm-{tag}@x.com",
                 password_hash=hash_password("p1234567"), role="admin",
                 status="active")
    mgr_u = User(tenant_id=t.id, email=f"mgr-{tag}@x.com",
                 password_hash=hash_password("p1234567"), role="hiring_manager",
                 status="active")
    emp_u = User(tenant_id=t.id, email=f"emp-{tag}@x.com",
                 password_hash=hash_password("p1234567"), role="employee",
                 status="active")
    db.add_all([admin, mgr_u, emp_u])
    db.flush()
    mgr = Employee(tenant_id=t.id, employee_no=f"M{tag}", name="王经理",
                   designation="总监", status="active", user_id=mgr_u.id,
                   hire_date=date.today() - timedelta(days=900))
    db.add(mgr)
    db.flush()  # 先 flush 拿 mgr.id 再挂汇报线
    emp = Employee(tenant_id=t.id, employee_no=f"E{tag}", name="张工",
                   designation="后端工程师", status="active",
                   user_id=emp_u.id, reports_to_id=mgr.id,
                   hire_date=date.today() - timedelta(days=400))
    emp2 = Employee(tenant_id=t.id, employee_no=f"F{tag}", name="刘工",
                    designation="前端工程师", status="active",
                    reports_to_id=mgr.id,
                    hire_date=date.today() - timedelta(days=200))
    db.add_all([emp, emp2])
    db.commit()

    ah = create_access_token(subject=admin.id, email=admin.email, role=admin.role)
    mh = create_access_token(subject=mgr_u.id, email=mgr_u.email, role=mgr_u.role)
    eh = create_access_token(subject=emp_u.id, email=emp_u.email, role=emp_u.role)

    # 1. 手动加技能(admin)
    c, b = req("POST", f"/api/employees/{emp.id}/skills", ah,
               {"name": "Python", "level": 5})
    check("手动加技能 201+confirmed", c == 201 and b.get("status") == "confirmed")
    c, b = req("POST", f"/api/employees/{emp2.id}/skills", ah,
               {"name": "Python", "level": 2})
    check("第二人加技能", c == 201)

    # 2. 画像卡聚合
    c, b = req("GET", f"/api/employees/{emp.id}/talent-card", ah)
    keys = {"employee", "skills_confirmed", "suggested_count", "interviews",
            "projects", "appraisals", "insights", "skills_suggested"}
    check("画像卡 keys", c == 200 and keys <= set(b), str(sorted(b)))

    # 3. insight(免提炼)+ 手动造建议 → 确认
    c, b = req("POST", f"/api/employees/{emp.id}/insights", ah,
               {"kind": "meeting", "title": "架构评审",
                "body": "评审通过,张工主讲", "extract_skills": False})
    check("insight 201", c == 201 and "id" in b)
    # 直接 DB 造一条 suggested
    from app.domain.models import EmployeeSkill, Skill  # noqa
    sk = Skill(tenant_id=t.id, name="系统设计", category="tech")
    db.add(sk)
    db.flush()
    db.add(EmployeeSkill(tenant_id=t.id, employee_id=emp.id, skill_id=sk.id,
                         level=4, source="insight", status="suggested"))
    db.commit()
    c, b = req("POST", f"/api/employees/{emp.id}/skill-suggestions/confirm",
               ah, {})
    check("建议确认", c == 200 and b.get("confirmed") == 1)

    # 4. 项目 CRUD + 成员
    c, b = req("POST", "/api/projects", ah,
               {"name": f"AI平台-{tag}",
                "required_skills": [{"name": "Python", "min_level": 3}]})
    check("建项目 201", c == 201, str(b)[:80])
    pid = b.get("id")
    c, b = req("POST", f"/api/projects/{pid}/members", ah,
               {"employee_id": emp.id, "role": "后端", "joined_via": "ai_match"})
    check("加成员 ai_match", c == 201)
    c, b = req("POST", f"/api/projects/{pid}/members", ah,
               {"employee_id": emp.id})
    check("重复成员 409", c == 409)
    c, b = req("PUT", f"/api/projects/{pid}", ah,
               {"name": f"AI平台-{tag}", "status": "active",
                "required_skills": []})
    check("项目启动", c == 200 and b.get("status") == "active")
    c, b = req("PUT", f"/api/projects/{pid}", ah,
               {"name": f"AI平台-{tag}", "status": "done",
                "required_skills": []})
    check("项目完成(skills_suggested>=0)", c == 200 and "skills_suggested" in b)

    # 5. talent/search
    c, b = req("POST", "/api/talent/search", ah,
               {"skills": [{"name": "Python"}]})
    names = [i["name"] for i in b.get("items", [])]
    check("搜索片段命中", c == 200 and "张工" in names and "刘工" in names,
          str(names))
    c, b = req("POST", "/api/talent/search", ah,
               {"skills": [{"name": "Python", "min_level": 3}]})
    names = [i["name"] for i in b.get("items", [])]
    check("min_level 过滤", c == 200 and names == ["张工"], str(names))
    c, b = req("POST", "/api/talent/search", ah,
               {"skills": [{"name": "Python"}, {"name": "系统设计"}]})
    it = (b.get("items") or [{}])[0]
    check("多片段排序", c == 200 and it.get("name") == "张工"
          and it.get("match_count") == 2, str(it)[:100])

    # 6. 本人脱敏
    c, b = req("GET", "/api/me/talent-card", eh)
    check("本人视图无 suggested", c == 200 and b.get("suggested_count") == 0
          and b.get("skills_suggested") == [])

    # 7. 主管可看下属 + 无权者 404
    c, b = req("GET", f"/api/employees/{emp.id}/talent-card", mh)
    check("主管看下属卡", c == 200 and "skills_suggested" in b
          and b.get("suggested_count", 0) >= 0,
          f"suggested={b.get('suggested_count')}")
    c, b = req("GET", f"/api/employees/{mgr.id}/talent-card", eh)
    check("员工看上级卡 404(scope)", c == 404)

    ok = sum(1 for _, o, _ in results if o)
    print(f"\n== {ok}/{len(results)} passed ==")
    db.close()
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
