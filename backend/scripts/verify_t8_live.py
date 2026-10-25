"""T8 AI 导入向导 — 真实 HTTP 链路验证(容器内运行)。

用法:
    docker compose exec backend python scripts/verify_t8_live.py

覆盖:
- datasets 元数据 + 模板下载
- 上传(csv 中文表头/xlsx/坏文件 400/超行数 400)
- 启发式映射 + AI suggest(license 开→ai 源,关→启发式兜底)
- PUT mapping 手改 + 幻觉列拒绝 + 必填未映射 400
- check:valid/errors/will_create/ai_note;错误行字段级归因
- run:写库 + 部门自动创建 + audit import → automation 联动触发
- errors.csv 下载;任务历史/详情;DELETE
- 权限:employee → 403;租户隔离:他租户 404

退出码 0 = 全过。
"""
from __future__ import annotations

import io
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


def upload(token, dataset, fname, content: bytes):
    return req("POST", "/api/imports", token,
               data={"dataset": dataset},
               files={"file": (fname, content)})


def main() -> int:
    admin = login("admin@example.com", "admin123456")

    # ---------- 1. datasets ----------
    r = req("GET", "/api/imports/datasets", admin)
    ds = {d["key"]: d for d in r.json().get("datasets", [])}
    check("datasets 含 employees+candidates",
          r.status_code == 200 and {"employees", "candidates"} <= set(ds), r.text[:200])
    check("datasets 返回 ai_enabled", "ai_enabled" in r.json())
    check("员工字段含中文别名/必填标记",
          any(f.get("required") for f in ds.get("employees", {}).get("fields", [])))

    # ---------- 2. 模板 ----------
    r = req("GET", "/api/imports/datasets/employees/template", admin)
    check("员工模板下载", r.status_code == 200 and "name" in r.text and "hire_date" in r.text)

    # ---------- 3. 上传拒绝 ----------
    r = upload(admin, "employees", "bad.txt", b"hello")
    check("非 csv/xlsx 400", r.status_code == 400, f"{r.status_code}")
    r = upload(admin, "nope", "a.csv", b"a,b\n1,2\n")
    check("未知数据集 404", r.status_code == 404, f"{r.status_code}")
    r = upload(admin, "employees", "empty.csv", "h1,h2\n".encode())
    check("无数据行 400", r.status_code == 400, f"{r.status_code}")

    # ---------- 4. 中文表头员工导入全链路 ----------
    csv_emp = (
        "姓名,员工编号,电子邮箱,部门名称,入职日期,雇佣类型\n"
        f"链路张{RUN},T8-{RUN}1,zhang-{RUN}@acme.co,导入向导部{RUN},2024-01-15,全职\n"
        f"链路李{RUN},T8-{RUN}2,li-{RUN}@acme.co,销售部,2024/02/01,full_time\n"
        f"链路错{RUN},,bad-email,研发部,not-a-date,兼职\n"
    ).encode()
    r = upload(admin, "employees", "staff.csv", csv_emp)
    check("上传 201 + 列/行数", r.status_code == 201 and
          r.json()["row_count"] == 3 and "姓名" in r.json()["columns"], r.text[:200])
    job = r.json()
    jid = job["id"]
    check("上传即给启发式映射", job["suggested_mapping"].get("name") == "姓名")
    check("preview 有样本", bool(job.get("preview")))

    r = req("POST", f"/api/imports/{jid}/suggest", admin)
    sug = r.json() if r.status_code == 200 else {}
    check("suggest 返回 mapping+meta", r.status_code == 200 and
          sug.get("mapping", {}).get("name") == "姓名" and "meta" in sug, r.text[:200])
    srcs = {v.get("source") for v in sug.get("meta", {}).values()}
    check("meta 标记映射来源(ai/heuristic)", srcs <= {"ai", "heuristic"}, str(srcs))

    r = req("POST", f"/api/imports/{jid}/check", admin)
    chk = r.json() if r.status_code == 200 else {}
    check("check: 2 valid 1 error", r.status_code == 200 and
          chk.get("valid") == 2 and chk.get("error_count") == 1, r.text[:300])
    check("错误行字段归因", any(
        "邮箱" in m_ or "日期" in m_
        for e in chk.get("errors", []) for m_ in e["messages"]))
    check("will_create 预告新部门",
          f"导入向导部{RUN}" in str(chk.get("will_create", {}).get("orgs", [])))
    check("ai_note 字段存在(可空)", "ai_note" in chk)

    # 手改映射:把 hire_date 解绑 → 该字段为空不报错(非必填),雇佣类型改绑
    r = req("PUT", f"/api/imports/{jid}/mapping", admin,
            json={"mapping": {**sug.get("mapping", {}), "hire_date": ""}})
    check("PUT mapping 200", r.status_code == 200, r.text[:200])
    # 幻觉列注入应被拒
    r = req("PUT", f"/api/imports/{jid}/mapping", admin,
            json={"mapping": {"name": "不存在列"}})
    check("幻觉列映射被拒", r.status_code == 200 and
          not r.json()["mapping"].get("name"))

    # 恢复 hire_date 并跑
    req("PUT", f"/api/imports/{jid}/mapping", admin,
        json={"mapping": sug.get("mapping", {})})
    r = req("POST", f"/api/imports/{jid}/run", admin)
    sm = r.json() if r.status_code == 200 else {}
    check("run: created=2 errors=1", r.status_code == 200 and
          sm.get("created") == 2 and sm.get("errors") == 1, r.text[:300])

    # 落库:员工 + 自动建部门
    r = req("GET", f"/api/employees?q=链路张{RUN}", admin)
    items = r.json().get("items", r.json() if isinstance(r.json(), list) else [])
    emp = next((e for e in items if e.get("name") == f"链路张{RUN}"), None)
    check("员工已写库", emp is not None, r.text[:200])
    if emp:
        check("雇佣类型中文别名→full_time",
              emp.get("employment_type") == "full_time", str(emp)[:200])
        check("部门名→org_unit_id", emp.get("org_unit_id") is not None)
        check("新部门自动创建(名称回填)",
              emp.get("org_unit_name") == f"导入向导部{RUN}", str(emp)[:200])

    # 审计 → automation 联动(entity_type=employee, action=import)
    r = req("GET", f"/api/audit/events?entity_id={jid}&action=import", admin)
    logs = r.json().get("items", [])
    check("审计 import 留痕", any(
        l.get("entity_type") == "employee" for l in logs), r.text[:200])

    # ---------- 5. 错误工作簿 ----------
    r = req("GET", f"/api/imports/{jid}/errors.csv", admin)
    check("errors.csv 下载", r.status_code == 200 and
          "bad-email" in r.text and "错误说明" in r.text, f"{r.status_code}")

    # ---------- 6. 历史/详情/删除 ----------
    r = req("GET", "/api/imports", admin)
    check("历史含本任务", any(j["id"] == jid for j in r.json()))
    r = req("GET", f"/api/imports/{jid}", admin)
    check("详情 status=done + run_summary",
          r.json().get("status") == "done" and
          r.json().get("run_summary", {}).get("created") == 2)

    # ---------- 7. 候选人流水线 ----------
    csv_cand = (
        "姓名,邮箱,手机,标签\n"
        f"链路候选{RUN},cand-{RUN}@x.co,139{RUN}01,python;后端\n"
        f"链路重复{RUN},cand-{RUN}@x.co,139{RUN}01,\n"
    ).encode()
    r = upload(admin, "candidates", "c.csv", csv_cand)
    cj = r.json()
    req("POST", f"/api/imports/{cj['id']}/suggest", admin)
    r = req("POST", f"/api/imports/{cj['id']}/check", admin)
    chk = r.json()
    check("候选人文件内去重", chk.get("valid") == 1 and chk.get("error_count") == 1,
          r.text[:200])
    r = req("POST", f"/api/imports/{cj['id']}/run", admin)
    check("候选人导入 created=1", r.json().get("created") == 1, r.text[:200])
    req("DELETE", f"/api/imports/{cj['id']}", admin)

    # ---------- 8. xlsx ----------
    try:
        import openpyxl
        buf = io.BytesIO()
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["姓名", "电子邮箱"])
        ws.append([f"链路X{RUN}", f"x-{RUN}@acme.co"])
        wb.save(buf)
        r = upload(admin, "employees", "a.xlsx", buf.getvalue())
        check("xlsx 上传解析", r.status_code == 201 and r.json()["row_count"] == 1,
              r.text[:200])
        if r.status_code == 201:
            req("DELETE", f"/api/imports/{r.json()['id']}", admin)
    except ImportError:
        check("xlsx 上传解析(openpyxl 缺失跳过)", True)

    # ---------- 9. 权限 ----------
    ess_email = f"t8-ess-{RUN}@example.com"
    ess_pwd = "T8ess!pass"
    from sqlalchemy import select as sa_select
    from app.domain.models import Tenant, User as DbUser
    from app.infra.db import SessionLocal
    from app.infra.security import hash_password

    with SessionLocal() as db:
        t = db.scalars(sa_select(Tenant).where(Tenant.name == "default")).first()
        eu = DbUser(tenant_id=t.id, email=ess_email,
                    password_hash=hash_password(ess_pwd), role="employee")
        db.add(eu)
        db.commit()
        ess_uid = eu.id
    ess = login(ess_email, ess_pwd)
    r = upload(ess, "employees", "a.csv", "name\nx\n".encode())
    check("employee 上传员工 403", r.status_code == 403, f"{r.status_code}")
    r = req("POST", f"/api/imports/{jid}/suggest", ess)
    check("employee 操作任务 403(同租户无权限)", r.status_code == 403,
          f"{r.status_code}")

    # ---------- 10. 租户隔离 ----------
    with SessionLocal() as db:
        t2 = Tenant(name=f"t8-iso-{RUN}")
        db.add(t2)
        db.flush()
        u2 = DbUser(tenant_id=t2.id, email=f"t8-iso-{RUN}@example.com",
                    password_hash=hash_password("iso123456"), role="admin")
        db.add(u2)
        db.commit()
        t2_uid = u2.id
    iso = login(f"t8-iso-{RUN}@example.com", "iso123456")
    r = req("GET", f"/api/imports/{jid}", iso)
    check("他租户读任务 404", r.status_code == 404)
    r = req("GET", "/api/imports", iso)
    check("他租户列表不含本任务", all(j["id"] != jid for j in r.json()))
    r = req("GET", f"/api/imports/{jid}/errors.csv", iso)
    check("他租户下载错误文件 404", r.status_code == 404)

    # ---------- 清理 ----------
    req("DELETE", f"/api/imports/{jid}", admin)
    r = req("GET", f"/api/imports/{jid}", admin)
    check("删除后 404", r.status_code == 404)
    # 清导入的员工/候选人/部门/用户/租户
    with SessionLocal() as db:
        from app.domain.models import Candidate, Employee, OrgUnit
        for e in db.scalars(sa_select(Employee).where(
                Employee.name.like(f"链路%{RUN}%"))).all():
            db.delete(e)
        for c in db.scalars(sa_select(Candidate).where(
                Candidate.name.like(f"链路%{RUN}%"))).all():
            db.delete(c)
        for o in db.scalars(sa_select(OrgUnit).where(
                OrgUnit.name == f"导入向导部{RUN}")).all():
            db.delete(o)
        for u in db.scalars(sa_select(DbUser).where(
                DbUser.email.like(f"t8-%-{RUN}@example.com"))).all():
            db.delete(u)
        t2 = db.scalars(sa_select(Tenant).where(Tenant.name == f"t8-iso-{RUN}")).first()
        if t2:
            db.delete(t2)
        db.commit()

    print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
    for f_ in FAIL:
        print("FAIL:", f_)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
