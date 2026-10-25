"""T8 AI 导入向导测试。

覆盖:文件解析(csv/xlsx/大小/编码)、启发式映射、AI 映射合并
(防幻觉列/防重复占用/降级)、校验(必填/email/日期/choice中文别名/
去重/预建组织)、执行写库+审计+automation 联动、错误 CSV、
权限/租户隔离、模板下载。
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import datetime

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t8")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t8"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import imports as imp  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _create_schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.rollback()
    s.close()


def _tenant(db) -> Tenant:
    t = Tenant(name=f"t8-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant, *, role: str = "admin") -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role=role,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


@pytest.fixture
def client():
    return TestClient(app)


def _upload(client, hdr, dataset, name, content: bytes):
    return client.post(
        "/api/imports",
        headers=hdr,
        data={"dataset": dataset},
        files={"file": (name, content)},
    )


EMP_CSV = (
    "姓名,员工编号,电子邮箱,部门名称,入职日期,雇佣类型\n"
    "张三,E101,zhang@acme.co,研发部,2024-01-15,全职\n"
    "李四,E102,lisi@acme.co,销售部,2024/02/01,full_time\n"
    "王五,,bad-email,研发部,not-a-date,兼职\n"
)


# ---------- 解析 ----------

def test_parse_csv_basic():
    cols, rows = imp.parse_file("a.csv", EMP_CSV.encode())
    assert cols == ["姓名", "员工编号", "电子邮箱", "部门名称", "入职日期", "雇佣类型"]
    assert len(rows) == 3
    assert rows[0]["姓名"] == "张三"


def test_parse_rejects():
    with pytest.raises(imp.UploadError):
        imp.parse_file("a.txt", b"hi")
    with pytest.raises(imp.UploadError):
        imp.parse_file("a.csv", "h1,h2\n".encode())  # 无数据行
    with pytest.raises(imp.UploadError):
        imp.parse_file("a.csv", b"x" * (imp.MAX_BYTES + 1))


def test_parse_xlsx():
    openpyxl = pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["姓名", "电子邮箱"])
    ws.append(["赵六", "zhao@acme.co"])
    wb.save(buf)
    cols, rows = imp.parse_file("a.xlsx", buf.getvalue())
    assert cols == ["姓名", "电子邮箱"]
    assert rows[0]["姓名"] == "赵六"


def test_parse_date_formats():
    from app.services.imports import parse_date_value
    assert parse_date_value("2024-01-15").year == 2024
    assert parse_date_value("2024/02/01").month == 2
    assert parse_date_value("15/03/2024").day == 15
    assert parse_date_value("2024年4月3日").day == 3
    assert parse_date_value("45000") is not None  # Excel 序列日期
    assert parse_date_value("not-a-date") is None
    assert parse_date_value("") is None


# ---------- 映射 ----------

def test_heuristic_chinese_headers():
    spec = imp.IMPORT_DATASETS["employees"]
    m = imp.heuristic_suggest(
        ["姓名", "员工编号", "电子邮箱", "部门名称", "入职日期", "雇佣类型"],
        spec.fields,
    )
    assert m["name"] == "姓名"
    assert m["employee_no"] == "员工编号"
    assert m["work_email"] == "电子邮箱"
    assert m["org_name"] == "部门名称"
    assert m["hire_date"] == "入职日期"
    assert m["employment_type"] == "雇佣类型"


def test_ai_suggest_merge_and_hallucination(db, monkeypatch):
    t = _tenant(db)
    spec = imp.IMPORT_DATASETS["employees"]
    cols = ["E-mail", "部门", "奇怪列"]
    rows = [{"E-mail": "a@b.co", "部门": "研发", "奇怪列": "x"}]
    monkeypatch.setattr(
        "app.infra.license.state.is_feature_enabled", lambda db, f: True
    )
    monkeypatch.setattr(
        "app.integrations.llm.provider.chat",
        lambda *a, **k: (
            '{"mapping": {"work_email": "E-mail", "org_name": "幻觉列"},'
            ' "confidence": {"work_email": 0.95, "org_name": 0.9},'
            ' "warnings": ["ok"]}'
        ),
    )
    mapping, meta, warnings = imp.ai_suggest(db, t.id, spec, cols, rows)
    assert mapping["work_email"] == "E-mail"  # AI 命中
    assert mapping.get("org_name") in ("部门", "")  # 幻觉列被拒 → 启发式补
    assert meta["work_email"]["source"] == "ai"
    assert meta["work_email"]["confidence"] == 0.95


def test_ai_suggest_degrades_without_llm(db, monkeypatch):
    t = _tenant(db)
    spec = imp.IMPORT_DATASETS["employees"]
    monkeypatch.setattr(
        "app.infra.license.state.is_feature_enabled", lambda db, f: False
    )
    mapping, meta, warnings = imp.ai_suggest(
        db, t.id, spec, ["姓名"], [{"姓名": "x"}]
    )
    assert mapping["name"] == "姓名"  # 纯启发式兜底
    assert meta["name"]["source"] == "heuristic"


# ---------- API 链路 ----------

def test_full_flow_employees(client, db, monkeypatch):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    monkeypatch.setattr(
        "app.infra.license.state.is_feature_enabled", lambda db, f: True
    )
    # AI 映射:姓名→name 等,故意漏 org_name 看启发式补位
    monkeypatch.setattr(
        "app.integrations.llm.provider.chat",
        lambda *a, **k: (
            '{"mapping": {"name": "姓名", "work_email": "电子邮箱"},'
            ' "confidence": {"name": 0.9, "work_email": 0.9}}'
            if a[0][0]["content"].startswith("你是数据导入映射")
            else "错误集中在日期格式,请统一为 YYYY-MM-DD"
        ),
    )

    r = _upload(client, hdr, "employees", "staff.csv", EMP_CSV.encode())
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["columns"][0] == "姓名"
    assert job["suggested_mapping"]["name"] == "姓名"

    r = client.post(f"/api/imports/{job['id']}/suggest", headers=hdr)
    assert r.status_code == 200, r.text
    assert r.json()["mapping"]["work_email"] == "电子邮箱"
    assert r.json()["meta"]["work_email"]["source"] == "ai"
    assert r.json()["mapping"]["org_name"] == "部门名称"  # 启发式补位

    r = client.post(f"/api/imports/{job['id']}/check", headers=hdr)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["valid"] == 2
    assert res["error_count"] == 1
    assert res["errors"][0]["row"] == 4
    assert "研发部" in res["will_create"]["orgs"]
    assert res["ai_note"]  # mock LLM 归因

    r = client.post(f"/api/imports/{job['id']}/run", headers=hdr)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["created"] == 2 and s["errors"] == 1
    # 落库验证 + 组织已自动创建
    emps = db.scalars(
        select(m.Employee).where(
            m.Employee.tenant_id == t.id, m.Employee.name.in_(["张三", "李四"])
        )
    ).all()
    assert len(emps) == 2
    org = db.scalar(
        select(m.OrgUnit).where(
            m.OrgUnit.tenant_id == t.id, m.OrgUnit.name == "研发部"
        )
    )
    assert org is not None
    assert all(e.org_unit_id for e in emps)  # 部门名→ID(含自动创建)
    # 雇佣类型中文别名已规范化
    assert {e.employment_type for e in emps} == {"full_time"}
    # audit import → automation 可监听
    log = db.scalar(
        select(m.AuditLog).where(
            m.AuditLog.tenant_id == t.id,
            m.AuditLog.action == "import",
            m.AuditLog.entity_type == "employee",
        )
    )
    assert log is not None and log.detail["created"] == 2
    # 错误 CSV
    r = client.get(f"/api/imports/{job['id']}/errors.csv", headers=hdr)
    assert r.status_code == 200
    assert "bad-email" in r.text and "错误说明" in r.text
    # 任务详情
    r = client.get(f"/api/imports/{job['id']}", headers=hdr)
    assert r.json()["status"] == "done"


def test_check_requires_mapping(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    r = _upload(client, hdr, "employees", "a.csv", EMP_CSV.encode())
    job = r.json()
    r = client.post(f"/api/imports/{job['id']}/check", headers=hdr)
    assert r.status_code == 400


def test_put_mapping_validation(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    r = _upload(client, hdr, "employees", "a.csv", EMP_CSV.encode())
    job = r.json()
    r = client.put(
        f"/api/imports/{job['id']}/mapping",
        headers=hdr,
        json={"mapping": {"name": "幻觉列", "work_email": "电子邮箱"}},
    )
    assert r.status_code == 200
    assert r.json()["mapping"].get("name") in (None, "")  # 不存在列被拒
    assert r.json()["mapping"]["work_email"] == "电子邮箱"
    assert r.json()["meta"]["work_email"]["source"] == "user"


def test_candidates_dedup(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    from app.services.candidates import _hash as ch
    db.add(m.Candidate(
        tenant_id=t.id, name="存量", email_hash=ch("dup@x.co", t.id),
        display_email="dup@x.co",
    ))
    db.commit()
    csv_data = (
        "姓名,邮箱,手机,标签\n"
        "新人,new@x.co,13800000001,python;后端\n"
        "重复人,dup@x.co,,,\n"
        "又重复,new@x.co,13800000001,\n"
    )
    r = _upload(client, hdr, "candidates", "c.csv", csv_data.encode())
    job = r.json()
    client.post(f"/api/imports/{job['id']}/suggest", headers=hdr)
    r = client.post(f"/api/imports/{job['id']}/check", headers=hdr)
    res = r.json()
    assert res["valid"] == 1
    msgs = " ".join(m_ for e in res["errors"] for m_ in e["messages"])
    assert "库中已存在" in msgs and "文件内重复" in msgs
    r = client.post(f"/api/imports/{job['id']}/run", headers=hdr)
    assert r.json()["created"] == 1


def test_permissions_and_tenant_isolation(client, db):
    t1, t2 = _tenant(db), _tenant(db)
    admin = _user(db, t1)
    admin2 = _user(db, t2)
    emp = _user(db, t1, role="employee")
    r = _upload(client, _hdr(emp), "employees", "a.csv", EMP_CSV.encode())
    assert r.status_code == 403
    r = _upload(client, _hdr(admin), "employees", "a.csv", EMP_CSV.encode())
    job = r.json()
    # 跨租户不可见
    r = client.get(f"/api/imports/{job['id']}", headers=_hdr(admin2))
    assert r.status_code == 404


def test_datasets_and_template(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    r = client.get("/api/imports/datasets", headers=hdr)
    assert r.status_code == 200
    keys = {d["key"] for d in r.json()["datasets"]}
    assert keys == {"employees", "candidates"}
    assert "ai_enabled" in r.json()
    r = client.get("/api/imports/datasets/employees/template", headers=hdr)
    assert r.status_code == 200
    assert "name" in r.text and "hire_date" in r.text


def test_history_delete(client, db):
    t = _tenant(db)
    admin = _user(db, t)
    hdr = _hdr(admin)
    r = _upload(client, hdr, "employees", "a.csv", EMP_CSV.encode())
    job = r.json()
    r = client.get("/api/imports", headers=hdr)
    assert any(j["id"] == job["id"] for j in r.json())
    r = client.delete(f"/api/imports/{job['id']}", headers=hdr)
    assert r.status_code == 204
    r = client.get(f"/api/imports/{job['id']}", headers=hdr)
    assert r.status_code == 404
