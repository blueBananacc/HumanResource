"""T6 AI 产品化测试。

覆盖:prompt scope_module 两级回落、field-assist、记录摘要、文档起草、
doc_templates 管理、product_help 知识隔离、租户/权限/审计。
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t6")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t6"

from app.domain import models as m  # noqa: E402
from app.domain.models import Base, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services.prompting import render_prompt  # noqa: E402


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
    t = Tenant(name=f"t6-{uuid.uuid4().hex[:8]}")
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


def _cand(db, t: Tenant, **kw) -> m.Candidate:
    c = m.Candidate(tenant_id=t.id, name=kw.pop("name", f"候选人{uuid.uuid4().hex[:4]}"), **kw)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _emp(db, t: Tenant, **kw) -> m.Employee:
    e = m.Employee(
        tenant_id=t.id,
        employee_no=kw.pop("employee_no", f"E{uuid.uuid4().hex[:6]}"),
        name=kw.pop("name", f"员工{uuid.uuid4().hex[:4]}"),
        **kw,
    )
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


@pytest.fixture
def client():
    return TestClient(app)


# ---------- prompt_templates.scope_module 两级回落 ----------


def test_scope_module_two_level_fallback(client, db):
    t = _tenant(db)
    u = _user(db, t)
    key = "field.jobs.description.draft"

    # 全局覆盖
    r = client.put(
        f"/api/prompt-templates/{key}",
        headers=_hdr(u),
        json={"content": "全局版本 {{title}} 占位测试内容"},
    )
    assert r.status_code == 200, r.text
    # 模块级覆盖
    r = client.put(
        f"/api/prompt-templates/{key}",
        headers=_hdr(u),
        json={"content": "jobs 模块版 {{title}} 占位测试内容", "scope_module": "jobs"},
    )
    assert r.status_code == 200, r.text
    db.expire_all()

    scoped = render_prompt(db, t.id, key, module="jobs", title="X")
    assert scoped is not None and "jobs 模块版" in scoped
    fallback = render_prompt(db, t.id, key, module="cases", title="X")
    assert fallback is not None and "全局版本" in fallback
    plain = render_prompt(db, t.id, key, title="X")
    assert plain is not None and "全局版本" in plain

    # list 输出 scoped_count + module 分组字段
    r = client.get("/api/prompt-templates", headers=_hdr(u))
    assert r.status_code == 200
    item = next(x for x in r.json() if x["key"] == key)
    assert item["module"] == "jobs"
    assert item["scoped_count"] == 1
    assert item["is_custom"] is True

    # 删模块覆盖后回落全局
    r = client.post(
        f"/api/prompt-templates/{key}/reset?scope_module=jobs", headers=_hdr(u)
    )
    assert r.status_code == 200
    db.expire_all()
    assert "全局版本" in (render_prompt(db, t.id, key, module="jobs", title="X") or "")


def test_prompt_templates_list_has_t6_keys(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.get("/api/prompt-templates", headers=_hdr(u))
    assert r.status_code == 200
    keys = {x["key"] for x in r.json()}
    for k in (
        "field.jobs.description.draft",
        "field.cases.resolution.draft",
        "field.generic.polish",
        "summarize.candidate",
        "summarize.employee",
        "summarize.case",
        "doc.draft",
    ):
        assert k in keys, k


# ---------- T6-1 field-assist ----------


def test_field_assist_and_audit(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.post(
        "/api/ai/field-assist",
        headers=_hdr(u),
        json={
            "target": "jobs.description",
            "intent": "draft",
            "fields": {"title": "后端工程师", "level": "advanced"},
            "instruction": "语气活泼",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["text"]
    assert r.json()["key"] == "field.jobs.description.draft"

    audit = db.scalar(
        select(m.AuditLog).where(
            m.AuditLog.tenant_id == t.id,
            m.AuditLog.entity_type == "ai_call",
            m.AuditLog.action == "field_assist",
        )
    )
    assert audit is not None


def test_field_assist_perm_and_generic_fallback(client, db):
    t = _tenant(db)
    v = _user(db, t, role="viewer")
    # viewer 有 view_reports 但没有 manage_employees → cases.resolution 403
    r = client.post(
        "/api/ai/field-assist",
        headers=_hdr(v),
        json={"target": "cases.resolution", "fields": {}},
    )
    assert r.status_code == 403
    # 未知 target → field.generic.* 回落,viewer 可用
    r = client.post(
        "/api/ai/field-assist",
        headers=_hdr(v),
        json={"target": "whatever.x", "intent": "draft", "fields": {"a": "1"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["key"] == "field.generic.draft"


# ---------- T6-2 记录摘要 ----------


def test_summarize_candidate(client, db):
    t = _tenant(db)
    u = _user(db, t)
    c = _cand(db, t, name="测试摘要")
    db.add(
        m.CandidateNote(
            tenant_id=t.id,
            candidate_id=c.id,
            author_id=u.id,
            author_email=u.email,
            content="沟通顺畅",
        )
    )
    db.commit()

    r = client.get(f"/api/ai/summarize/candidates/{c.id}", headers=_hdr(u))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entity"] == "candidates" and body["summary"]

    audit = db.scalar(
        select(m.AuditLog).where(
            m.AuditLog.tenant_id == t.id,
            m.AuditLog.entity_type == "ai_call",
            m.AuditLog.action == "summarize",
        )
    )
    assert audit is not None


def test_summarize_tenant_isolation_and_perm(client, db):
    t = _tenant(db)
    other = _tenant(db)
    u2 = _user(db, other)
    c = _cand(db, t)
    # 跨租户 → 404(防探测)
    r = client.get(f"/api/ai/summarize/candidates/{c.id}", headers=_hdr(u2))
    assert r.status_code == 404
    # viewer 看 employees 摘要 → 403(需 manage_employees)
    v = _user(db, t, role="viewer")
    e = _emp(db, t)
    r = client.get(f"/api/ai/summarize/employees/{e.id}", headers=_hdr(v))
    assert r.status_code == 403
    # 不支持实体 → 400
    u = _user(db, t)
    r = client.get("/api/ai/summarize/jobs/x", headers=_hdr(u))
    assert r.status_code == 400


# ---------- T6-4 文档起草 ----------


def test_doc_templates_list_and_draft(client, db):
    t = _tenant(db)
    u = _user(db, t)
    # 轻量列表(view_reports 可用)
    r = client.get("/api/ai/doc-templates", headers=_hdr(u))
    assert r.status_code == 200
    keys = {x["key"] for x in r.json()}
    assert {"offer_letter", "employment_certificate", "probation_notice"} <= keys

    c = _cand(db, t, name="张伟")
    r = client.post(
        "/api/ai/doc-draft",
        headers=_hdr(u),
        json={"key": "offer_letter", "entity": "candidates", "record_id": c.id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["draft"]
    assert body["vars"]["candidate_name"] == "张伟"

    # offer_letter 绑 candidates,对 employees 用 → 400
    e = _emp(db, t)
    r = client.post(
        "/api/ai/doc-draft",
        headers=_hdr(u),
        json={"key": "offer_letter", "entity": "employees", "record_id": e.id},
    )
    assert r.status_code == 400
    # 不存在的模板 → 404
    r = client.post(
        "/api/ai/doc-draft",
        headers=_hdr(u),
        json={"key": "nope", "entity": "candidates", "record_id": c.id},
    )
    assert r.status_code == 404


def test_doc_templates_crud(client, db):
    t = _tenant(db)
    u = _user(db, t)
    r = client.get("/api/doc-templates", headers=_hdr(u))
    assert r.status_code == 200
    builtin = next(x for x in r.json() if x["key"] == "offer_letter")
    assert builtin["is_builtin"] and not builtin["is_custom"]

    # 覆盖内置
    r = client.put(
        "/api/doc-templates/offer_letter",
        headers=_hdr(u),
        json={"name": "我司录用函", "entity": "candidates", "body": "{{candidate_name}} 您好…（足够长）"},
    )
    assert r.status_code == 200 and r.json()["is_custom"]

    # 新建自定义
    r = client.put(
        "/api/doc-templates/custom.abc1",
        headers=_hdr(u),
        json={"name": "自定义证明", "entity": "any", "body": "兹证明 {{var}}（足够长内容）"},
    )
    assert r.status_code == 200 and not r.json()["is_builtin"]

    # 轻量列表也能看到自定义项
    r = client.get("/api/ai/doc-templates", headers=_hdr(u))
    assert "custom.abc1" in {x["key"] for x in r.json()}

    # 起草走租户覆盖版
    c = _cand(db, t, name="覆盖验证")
    r = client.post(
        "/api/ai/doc-draft",
        headers=_hdr(u),
        json={"key": "offer_letter", "entity": "candidates", "record_id": c.id},
    )
    assert r.status_code == 200

    # 内置不能 DELETE(400),reset 回默认
    r = client.delete("/api/doc-templates/offer_letter", headers=_hdr(u))
    assert r.status_code == 400
    r = client.post("/api/doc-templates/offer_letter/reset", headers=_hdr(u))
    assert r.status_code == 200 and not r.json()["is_custom"]

    # 自定义可 DELETE
    r = client.delete("/api/doc-templates/custom.abc1", headers=_hdr(u))
    assert r.status_code == 204


# ---------- product_help 知识隔离 ----------


def test_product_help_isolation(client, db):
    t = _tenant(db)
    u = _user(db, t)
    # API 侧建不了 product_help(category 校验器不含) → 422
    r = client.post(
        "/api/knowledge/articles",
        headers=_hdr(u),
        json={"title": "X", "category": "product_help", "content": "y"},
    )
    assert r.status_code == 422

    # 直写库(模拟灌库脚本产物)
    art = m.KnowledgeArticle(
        tenant_id=t.id,
        title="HROnfire 使用手册 > 批量导入",
        category="product_help",
        content="进入员工列表点击批量导入,选择 CSV 文件…",
        tags=["product_help"],
        status="published",
    )
    db.add(art)
    db.commit()

    # 按 category 限定召回只命中产品文档
    r = client.post(
        "/api/knowledge/search",
        headers=_hdr(u),
        json={"q": "批量导入", "category": "product_help"},
    )
    assert r.status_code == 200
    assert any(h["id"] == art.id for h in r.json())
    assert all(h["category"] == "product_help" for h in r.json())

    # 普通检索不混入产品文档?——不,product_help 属知识库召回面;
    # 隔离语义是"写只读 + 可按类过滤"。校验编辑/删除被拒:
    r = client.put(
        f"/api/knowledge/articles/{art.id}",
        headers=_hdr(u),
        json={"title": "改", "category": "faq", "content": "x"},
    )
    assert r.status_code == 409
    r = client.delete(f"/api/knowledge/articles/{art.id}", headers=_hdr(u))
    assert r.status_code == 409


def test_seed_slicer():
    from scripts.seed_product_help import slice_markdown

    md = "# 手册\n\n## 第一章\n\n章下正文。\n\n### 1.1 小节\n\n小节内容。\n\n## 第二章\n\n第二章正文。"
    secs = slice_markdown("测试手册", md)
    titles = [t for t, _ in secs]
    assert "测试手册 > 第一章" in titles
    assert "测试手册 > 第一章 > 1.1 小节" in titles
    assert "测试手册 > 第二章" in titles
