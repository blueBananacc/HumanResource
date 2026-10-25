# -*- coding: utf-8 -*-
"""T14 特征评分式简历结构解析器 — 单元/集成/API 三层测试。

覆盖:
- read_pdf_items / read_docx_items / read_txt_items:span→TextItem
- group_lines_into_sections:中英关键词/粗体标题
- divide_section_into_subsections:行首日期主路径/行距兜底
- extract_profile/work/education/projects:特征评分字段抽取
- structure_resume:fail-open / confidence / 各格式分发
- parse_resume 集成:sections 进 ParseResult
- scoring_rules:结构化优先评估(duration/boolean/range)
- API:detail 返回 sections
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id-t14")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402

settings.machine_fingerprint_override = "test-machine-id-t14"
settings.celery_task_always_eager = True

from app.domain.models import Base, Candidate, Resume, Tenant, User  # noqa: E402
from app.infra.db import SessionLocal, engine  # noqa: E402
from app.infra.security import create_access_token, hash_password  # noqa: E402
from app.main import app  # noqa: E402
from app.services import resume_structurer as st  # noqa: E402
from app.services import scoring_rules as sr  # noqa: E402
from app.services.resume_parser import parse_resume  # noqa: E402


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


@pytest.fixture
def client():
    return TestClient(app)


def _tenant(db) -> Tenant:
    t = Tenant(name=f"t14-{uuid.uuid4().hex[:8]}")
    db.add(t)
    db.commit()
    return t


def _user(db, t: Tenant) -> User:
    u = User(
        tenant_id=t.id,
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password("p1234567"),
        role="admin",
        status="active",
    )
    db.add(u)
    db.commit()
    return u


def _hdr(u: User) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(subject=u.id, email=u.email, role=u.role)}"
    }


# ---------------------------------------------------------------------------
# fixtures:合成 PDF / TXT
# ---------------------------------------------------------------------------


def _make_pdf(lines: list[tuple[str, float]]) -> bytes:
    """lines: [(text, size)] — 用 pymupdf 生成单页 PDF(中文字体 china-s)。"""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    p = doc.new_page()
    y = 60.0
    for text, size in lines:
        p.insert_text((72, y), text, fontname="china-s", fontsize=size)
        y += size + 4
    return doc.tobytes()


_SAMPLE_LINES = [
    ("张三", 18.0),
    ("zhangsan@example.com  13812345678  上海", 12.0),
    ("工作经历", 14.0),
    ("2020.03 - 至今  高级后端工程师  阿里巴巴", 12.0),
    ("• 负责订单系统重构,QPS 提升 3 倍", 12.0),
    ("2016.07 - 2020.02  后端开发工程师  美团", 12.0),
    ("• 开发骑手调度系统", 12.0),
    ("教育经历", 14.0),
    ("2012.09 - 2016.06  复旦大学  本科 计算机科学", 12.0),
    ("专业技能", 14.0),
    ("• Python, Go, FastAPI, PostgreSQL, Redis", 12.0),
]

_SAMPLE_TXT = "\n".join(t for t, _ in _SAMPLE_LINES)


# ---------------------------------------------------------------------------
# 纯函数层
# ---------------------------------------------------------------------------


def test_txt_items_and_line_grouping():
    items = st.read_txt_items(_SAMPLE_TXT)
    assert len(items) == 11
    lines = st.group_items_into_lines(items)
    assert len(lines) == 11


def test_sections_grouping_txt():
    lines = st.group_items_into_lines(st.read_txt_items(_SAMPLE_TXT))
    sections = st.group_lines_into_sections(lines)
    assert "profile" in sections
    names = list(sections.keys())
    assert any("工作" in n for n in names)
    assert any("教育" in n for n in names)
    assert any("技能" in n or "专业" in n for n in names)


def test_subsections_split_by_date_prefix():
    lines = st.group_items_into_lines(st.read_txt_items(_SAMPLE_TXT))
    sections = st.group_lines_into_sections(lines)
    work_lines = st._section_lines_by_keywords(sections, st._WORK_KW)
    subs = st.divide_section_into_subsections(work_lines)
    assert len(subs) == 2  # 阿里 + 美团各一段


def test_extract_profile_fields():
    lines = st.group_items_into_lines(st.read_txt_items(_SAMPLE_TXT))
    sections = st.group_lines_into_sections(lines)
    profile, _ = st.extract_profile(sections["profile"])
    assert profile["name"] == "张三"
    assert "zhangsan@example.com" in profile["email"]
    assert "13812345678" in profile["phone"]


def test_extract_work_two_entries():
    lines = st.group_items_into_lines(st.read_txt_items(_SAMPLE_TXT))
    sections = st.group_lines_into_sections(lines)
    work_lines = st._section_lines_by_keywords(sections, st._WORK_KW)
    works, _ = st.extract_work(work_lines)
    assert len(works) == 2
    assert works[0]["company"] == "阿里巴巴"
    assert "工程师" in works[0]["title"]
    assert works[0]["date_range"].startswith("2020.03")
    assert works[0]["years"] > 4  # 2020→今
    assert works[1]["company"] == "美团"
    assert abs(works[1]["years"] - 3.6) < 0.2
    assert works[0]["descriptions"]  # bullet 提取


def test_extract_education():
    lines = st.group_items_into_lines(st.read_txt_items(_SAMPLE_TXT))
    sections = st.group_lines_into_sections(lines)
    edu_lines = st._section_lines_by_keywords(sections, st._EDU_KW)
    edus, _ = st.extract_education(edu_lines)
    assert len(edus) == 1
    assert "复旦" in edus[0]["school"]
    assert "本科" in edus[0]["degree"]


def test_structure_resume_txt_full():
    res = st.structure_resume("r.txt", "text/plain", b"", _SAMPLE_TXT)
    assert res is not None
    assert res["parser"] == "feature-scoring-v1"
    assert res["profile"]["name"] == "张三"
    assert len(res["work_experiences"]) == 2
    assert len(res["educations"]) == 1
    assert "Python" in res["skills_text"]
    assert res["confidence"]["overall"] > 0.5


def test_structure_resume_pdf_full():
    pdf = _make_pdf(_SAMPLE_LINES)
    res = st.structure_resume("r.pdf", "application/pdf", pdf, "")
    assert res is not None
    assert res["profile"]["name"] == "张三"
    assert len(res["work_experiences"]) == 2
    assert res["work_experiences"][0]["company"] == "阿里巴巴"
    assert res["educations"][0]["school"].startswith("复旦")
    assert res["confidence"]["overall"] >= 0.8


def test_structure_resume_failopen():
    assert st.structure_resume("r.pdf", "application/pdf", b"not a pdf", "") is None
    assert st.structure_resume("x.txt", "text/plain", b"", "") is None


def test_years_from_range():
    assert st._years_from_range("2020.03 - 至今") > 4
    assert abs(st._years_from_range("2016.07 - 2020.02") - 3.6) < 0.2
    assert st._years_from_range("无日期") == 0.0


# ---------------------------------------------------------------------------
# parse_resume 集成
# ---------------------------------------------------------------------------


def test_parse_resume_txt_sections():
    res = parse_resume("张三简历.txt", "text/plain", _SAMPLE_TXT.encode("utf-8"))
    assert res.sections is not None
    assert res.sections["profile"]["name"] == "张三"
    assert res.email == "zhangsan@example.com"
    assert len(res.sections["work_experiences"]) == 2


# ---------------------------------------------------------------------------
# scoring_rules 结构化优先
# ---------------------------------------------------------------------------


def _mk_resume(db, t: Tenant, sections=None, parsed_text="") -> Resume:
    cand = Candidate(tenant_id=t.id, name="c")
    db.add(cand)
    db.commit()
    r = Resume(
        tenant_id=t.id,
        candidate_id=cand.id,
        file_name="r.pdf",
        file_size=1,
        file_mime="application/pdf",
        storage_key=uuid.uuid4().hex,
        source="upload",
        parse_status="done",
        parsed_text=parsed_text,
        parsed_data={
            "skills": ["Python"],
            "sections": sections,
        },
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def test_duration_prefers_structured(db):
    secs = {
        "work_experiences": [
            {"company": "A", "title": "工程师", "date_range": "2018-2020", "years": 2.0},
            {"company": "B", "title": "经理", "date_range": "2020-至今", "years": 6.0},
        ]
    }
    r = _mk_resume(db, _tenant(db), sections=secs, parsed_text="没有任何年份字样")
    ev = {"minYears": 5, "preferredYears": 10}
    out = sr._eval_duration(r, "", ev)
    # 结构化合计 8 年 → 落在 [5,10] → 60 + 3/5*40 = 84
    assert out["score"] == 84
    assert "结构化" in out["evidence"]


def test_duration_fallback_text(db):
    r = _mk_resume(db, _tenant(db), sections=None, parsed_text="3年经验")
    out = sr._eval_duration(r, "3年经验", {"minYears": 1, "preferredYears": 5})
    assert out["score"] > 0
    assert "文本估计" in out["evidence"]


def test_boolean_education_structured(db):
    secs = {"educations": [{"school": "复旦", "degree": "本科"}]}
    r = _mk_resume(db, _tenant(db), sections=secs, parsed_text="")
    out = sr._eval_boolean(r, "", {"booleanField": "education"})
    assert out["score"] == 100
    assert "结构化" in out["evidence"]


def test_range_structured(db):
    secs = {"work_experiences": [{"years": 4.0}, {"years": 4.0}]}
    r = _mk_resume(db, _tenant(db), sections=secs)
    out = sr._eval_range(r, "", {"minValue": 0, "maxValue": 10})
    assert out["score"] == 80


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_detail_returns_sections(db, client):
    t = _tenant(db)
    u = _user(db, t)
    secs = {"profile": {"name": "张三"}, "work_experiences": [{"company": "阿里"}],
            "confidence": {"overall": 0.9}, "parser": "feature-scoring-v1"}
    r = _mk_resume(db, t, sections=secs, parsed_text="x")
    resp = client.get(f"/api/resumes/{r.id}", headers=_hdr(u))
    assert resp.status_code == 200
    body = resp.json()
    assert body["sections"]["profile"]["name"] == "张三"
    assert body["sections"]["confidence"]["overall"] == 0.9


def test_api_detail_cross_tenant_denied(db, client):
    t1, t2 = _tenant(db), _tenant(db)
    u2 = _user(db, t2)
    r = _mk_resume(db, t1, parsed_text="x")
    resp = client.get(f"/api/resumes/{r.id}", headers=_hdr(u2))
    assert resp.status_code == 404
