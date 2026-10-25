"""特征评分式简历结构解析器(T14)—— LLM 前置确定性预处理层。

移植自 open-resume ``parse-resume-from-pdf``(MIT),本地化为中英双语:
1. ``read_pdf_items``:pymupdf ``get_text("dict")`` → TextItem{text,font,bold,x,y}
2. ``group_items_into_lines`` → ``group_lines_into_sections``
3. ``divide_section_into_subsections``(行距>典型行距×1.4;无坐标时按行首日期)
4. 字段抽取:FeatureSet 正负特征打分取最高分

DOCX/TXT 没有坐标/字体,走同一套管线但 bold=False、行=段落;
docx 额外读取 ``<w:b/>`` run 粗体增强标题检测。

产出 ``sections`` dict 进 ``parsed_data["sections"]``;任何失败 fail-open
返回 None —— 下游(T13 评估器)回退全文正则,零回归。
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据类型
# ---------------------------------------------------------------------------


@dataclass
class TextItem:
    text: str
    bold: bool = False
    size: float = 0.0
    x: float = 0.0
    y: float = 0.0
    page: int = 0


Line = list[TextItem]
Lines = list[Line]
# 特征集: (判别函数, 分值, 是否返回匹配文本)
FeatureSet = tuple[Callable[[TextItem], Any], int, bool]


def _fs(fn: Callable[[TextItem], Any], score: int, ret: bool = False) -> FeatureSet:
    return (fn, score, ret)


# ---------------------------------------------------------------------------
# 特征评分系统(移植 feature-scoring-system.ts)
# ---------------------------------------------------------------------------


def _highest_scored(
    items: list[TextItem],
    feature_sets: list[FeatureSet],
    *,
    empty_if_not_positive: bool = True,
    concat_ties: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """逐项打分,返回 (最高分文本, 调试用 scores)。"""
    scores: list[dict[str, Any]] = [
        {"text": it.text, "score": 0, "match": False} for it in items
    ]
    for i, it in enumerate(items):
        for fn, score, ret in feature_sets:
            res = fn(it)
            if not res:
                continue
            if ret:
                # re.Match 支持 [0] → group(0);list/tuple 同理
                try:
                    text = res[0]
                except (IndexError, TypeError, KeyError):
                    text = it.text
            else:
                text = it.text
            if it.text == text:
                scores[i]["score"] += score
                if ret:
                    scores[i]["match"] = True
            else:
                scores.append({"text": text, "score": score, "match": True})

    best = float("-inf")
    winners: list[str] = []
    for s in scores:
        if s["score"] >= best:
            if s["score"] > best:
                winners = []
            winners.append(s["text"])
            best = s["score"]

    if empty_if_not_positive and best <= 0:
        return "", scores
    if not winners:
        return "", scores
    text = winners[0] if not concat_ties else " ".join(t.strip() for t in winners)
    return (text or ""), scores


# ---------------------------------------------------------------------------
# Step1-2: items → lines → sections
# ---------------------------------------------------------------------------


def read_pdf_items(data: bytes, max_pages: int = 50) -> list[TextItem]:
    """pymupdf dict 模式抽 span 级文本项(带字体/坐标)。"""
    import pymupdf  # type: ignore[import-not-found]

    items: list[TextItem] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[arg-type]
        for pno, page in enumerate(doc):
            if pno >= max_pages:
                break
            try:
                raw = page.get_text("dict")
            except Exception as e:  # noqa: BLE001
                logger.warning("structurer: 第 %d 页 dict 提取失败: %s", pno, e)
                continue
            for block in raw.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").replace("-­‐", "-")
                        if not text.strip():
                            continue
                        font = (span.get("font") or "").lower()
                        flags = int(span.get("flags") or 0)
                        origin = span.get("origin") or [0.0, 0.0]
                        items.append(
                            TextItem(
                                text=text.strip(),
                                bold=("bold" in font) or bool(flags & 16),
                                size=float(span.get("size") or 0),
                                x=float(origin[0]),
                                y=float(origin[1]),
                                page=pno,
                            )
                        )
    return items


def read_docx_items(data: bytes) -> list[TextItem]:
    """DOCX:段落=行;``<w:b/>`` run 标记 bold(增强标题检测)。"""
    try:
        from lxml import etree
    except ImportError:
        return []

    W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            parts = [n for n in zf.namelist() if n == "word/document.xml" or (
                n.startswith(("word/header", "word/footer")) and n.endswith(".xml")
            )]
            xmls = [zf.read(n) for n in parts]
    except (zipfile.BadZipFile, KeyError):
        return []

    items: list[TextItem] = []

    def _in_fallback(node) -> bool:
        anc = node.getparent()
        while anc is not None:
            q = etree.QName(anc)
            if q.namespace == MC_NS and q.localname == "Fallback":
                return True
            anc = anc.getparent()
        return False

    def _walk(node) -> None:
        for child in node:
            q = etree.QName(child)
            if q.namespace == MC_NS and q.localname == "AlternateContent":
                target = next(
                    (c for c in child
                     if etree.QName(c).namespace == MC_NS and etree.QName(c).localname == "Choice"),
                    None,
                ) or next(
                    (c for c in child
                     if etree.QName(c).namespace == MC_NS and etree.QName(c).localname == "Fallback"),
                    None,
                )
                if target is not None:
                    _walk(target)
                continue
            if q.namespace == W_NS and q.localname == "p":
                texts: list[str] = []
                bold = False
                for t in child.iter():
                    tg = etree.QName(t)
                    if tg.namespace != W_NS or _in_fallback(t):
                        continue
                    if tg.localname == "t" and t.text:
                        texts.append(t.text)
                    elif tg.localname == "b":
                        # <w:b/> 或 <w:b w:val="true"/>
                        val = t.get(f"{{{W_NS}}}val")
                        if val not in ("0", "false", "off"):
                            bold = True
                line = "".join(texts).strip()
                if line:
                    items.append(TextItem(text=line, bold=bold))
                continue
            _walk(child)

    for xml in xmls:
        try:
            root = etree.fromstring(xml)
        except Exception:  # noqa: BLE001
            continue
        _walk(root)
    return items


def read_txt_items(text: str) -> list[TextItem]:
    return [TextItem(text=ln.strip()) for ln in text.splitlines() if ln.strip()]


def group_items_into_lines(items: list[TextItem]) -> Lines:
    """按 (page, y) 聚合 span 成行;无坐标的文本项本来就一项一行。"""
    if not items:
        return []
    if all(it.y == 0.0 for it in items):
        return [[it] for it in items]

    by_key: dict[tuple[int, int], Line] = {}
    for it in items:
        key = (it.page, round(it.y))
        by_key.setdefault(key, []).append(it)
    lines: Lines = []
    # PyMuPDF 坐标系 y 向下递增(左上原点),升序即阅读顺序
    for (page, _y), line_items in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append(sorted(line_items, key=lambda i: i.x))
    return lines


_SECTION_KEYWORDS = [
    # 中文
    "工作经历", "工作经验", "任职经历", "实习经历", "项目经验", "项目经历",
    "教育经历", "教育背景", "专业技能", "技能", "证书", "语言",
    "自我评价", "个人评价", "个人总结", "自我总结", "求职意向", "获奖",
    # 英文
    "experience", "employment", "education", "project", "skill",
    "summary", "objective", "award", "honor", "certificate", "language",
]
_SECTION_PRIMARY = [
    "experience", "employment", "education", "project", "skill",
    "工作", "任职", "实习", "教育", "项目", "技能", "专业",
]


def _line_text(line: Line) -> str:
    return " ".join(it.text for it in line).strip()


def _is_section_title(line: Line, line_no: int) -> bool:
    if line_no < 1 or not line:
        return False
    text = _line_text(line)
    if not text:
        return False
    first = line[0]
    low = text.lower()

    # 关键词命中(≤8字/≤3词 短行)——中英文通用主路径
    compact = text.replace(" ", "")
    is_short = len(compact) <= 8 or len(text.split()) <= 3
    if is_short and any(k in low for k in _SECTION_KEYWORDS):
        return True

    # 粗体+全大写(英文排版惯例兜底)
    if first.bold and re.search(r"[a-zA-Z]", text) and text.upper() == text:
        return True
    # 粗体 + 中文短行(模板化中文简历兜底)
    if first.bold and len(compact) <= 6 and re.search(r"[一-鿿]", text):
        return True
    return False


def group_lines_into_sections(lines: Lines) -> dict[str, Lines]:
    """首个标题前的行归 profile,其后各标题各成 section。"""
    sections: dict[str, Lines] = {}
    name = "profile"
    buf: Lines = []
    for i, line in enumerate(lines):
        if _is_section_title(line, i):
            sections[name] = buf
            name = _line_text(line)
            buf = []
        else:
            buf.append(line)
    if buf:
        sections[name] = buf
    return sections


def _section_lines_by_keywords(
    sections: dict[str, Lines], keywords: list[str]
) -> Lines:
    out: Lines = []
    for name, lines in sections.items():
        low = name.lower()
        if any(k in low for k in keywords):
            out.extend(lines)
    return out


# ---------------------------------------------------------------------------
# Step3: section → subsections(行距 / 行首日期)
# ---------------------------------------------------------------------------

_BULLETS = "⋅∙🞄•⦁⚫●⬤⚬○·"
_DATE_RANGE_RE = re.compile(
    r"(?:19|20)\d{2}[.\-/年]?\s*\d{0,2}月?\s*(?:-|~|–|—|至|到)\s*"
    r"((?:19|20)\d{2}(?:[.\-/年]?\s*\d{1,2}月?)?|至今|今|Present|Now)",
    re.IGNORECASE,
)


def _starts_with_date(line: Line) -> bool:
    text = _line_text(line)
    return bool(_DATE_RANGE_RE.search(text[:30]))


def divide_section_into_subsections(lines: Lines) -> list[Lines]:
    """行首日期区间为主(简历每段惯例以日期开头)→ 行距×1.4 → 粗体 兜底。"""
    if not lines:
        return []

    # 主路径:行首/行前 30 字符内含日期区间的行是新段起点
    subs: list[Lines] = []
    buf: Lines = []
    for i, line in enumerate(lines):
        if i > 0 and _starts_with_date(line):
            subs.append(buf)
            buf = []
        buf.append(line)
    if buf:
        subs.append(buf)
    # 段数 ≥2 说明日期切分生效;不足再试行距
    if len(subs) <= 1 and any(ln and ln[0].y != 0.0 for ln in lines):
        gaps: dict[int, int] = {}
        ys = [ln[0].y for ln in lines if ln]
        for i in range(1, len(ys)):
            g = round(ys[i] - ys[i - 1])
            gaps[g] = gaps.get(g, 0) + 1
        typical = max(gaps.items(), key=lambda kv: kv[1])[0] if gaps else 0
        threshold = typical * 1.4
        subs = []
        buf = []
        for i, line in enumerate(lines):
            if i > 0 and round(line[0].y - lines[i - 1][0].y) > threshold:
                subs.append(buf)
                buf = []
            buf.append(line)
        if buf:
            subs.append(buf)

    # 兜底2:整段没切开时,粗体行作为新段起点(忽略 bullet)
    if len(subs) <= 1 and len(lines) > 3:
        subs = []
        buf = []
        for i, line in enumerate(lines):
            first = line[0] if line else None
            if (
                i > 0
                and first is not None
                and first.bold
                and not (lines[i - 1] and lines[i - 1][0].bold)
                and not any(b in first.text for b in _BULLETS)
            ):
                subs.append(buf)
                buf = []
            buf.append(line)
        if buf:
            subs.append(buf)
        subs = [s for s in subs if s]
    return subs or [lines]


# ---------------------------------------------------------------------------
# bullet 描述切分(移植 bullet-points.ts)
# ---------------------------------------------------------------------------


def bullet_descriptions(lines: Lines) -> list[str]:
    joined = " ".join(it.text for ln in lines for it in ln).strip()
    if not any(b in joined for b in _BULLETS):
        return [_line_text(ln) for ln in lines if _line_text(ln)]
    common = max(_BULLETS, key=lambda b: joined.count(b))
    idx = joined.find(common)
    if idx >= 0:
        joined = joined[idx:]
    return [t.strip() for t in joined.split(common) if t.strip()]


def _descriptions_line_idx(lines: Lines) -> int | None:
    for i, ln in enumerate(lines):
        if any(any(b in it.text for b in _BULLETS) for it in ln):
            return i
    # 无 bullet:长行(≥8词 或 ≥20中文字)视为描述起点
    for i, ln in enumerate(lines):
        text = _line_text(ln)
        if len(ln) == 1 and (
            len([w for w in text.split() if not re.search(r"\d", w)]) >= 8
            or len(re.sub(r"[^一-鿿]", "", text)) >= 20
        ):
            return i
    return None


# ---------------------------------------------------------------------------
# 字段特征集
# ---------------------------------------------------------------------------


def _has(pattern: str) -> Callable[[TextItem], Any]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda it: rx.search(it.text)


def _has_text(needle: str) -> Callable[[TextItem], Any]:
    return lambda it: needle and needle in it.text


_EMAIL_F = _has(r"\S+@\S+\.\S+")
_PHONE_F = _has(r"(?<!\d)(?:1[3-9]\d{9}|\+\d{1,3}[\s-]?\d{6,14}|\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})(?!\d)")
_NAME_F = _has(r"^[a-zA-Z\s.一-鿿·]{2,20}$")

PROFILE_FEATURE_SETS: dict[str, list[FeatureSet]] = {
    "name": [
        _fs(_NAME_F, 3, True),
        _fs(lambda it: it.bold, 2),
        _fs(lambda it: bool(re.search(r"[a-zA-Z]", it.text)) and it.text.upper() == it.text, 2),
        _fs(lambda it: "@" in it.text, -4),
        _fs(_PHONE_F, -4),
        _fs(lambda it: len(it.text.split()) >= 4 or len(re.sub(r"[^一-鿿]", "", it.text)) > 8, -2),
    ],
    "email": [_fs(_EMAIL_F, 6, True), _fs(lambda it: it.bold, -1)],
    "phone": [_fs(_PHONE_F, 6, True), _fs(lambda it: bool(re.search(r"[a-zA-Z一-鿿]", it.text)), -4)],
    "location": [
        _fs(_has(r"[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}|北京|上海|广州|深圳|杭州|成都|南京|武汉|西安|苏州|重庆|天津"), 4, True),
        _fs(lambda it: "@" in it.text, -4),
        _fs(_PHONE_F, -3),
    ],
    "summary": [
        _fs(lambda it: len(it.text.split()) >= 4 or len(re.sub(r"[^一-鿿]", "", it.text)) >= 12, 4),
        _fs(lambda it: it.bold, -1),
        _fs(lambda it: "@" in it.text, -4),
        _fs(_PHONE_F, -3),
    ],
}

_JOB_TITLES = [
    "engineer", "developer", "manager", "director", "lead", "senior",
    "architect", "analyst", "consultant", "designer", "intern", "specialist",
    "工程师", "经理", "主管", "专员", "总监", "实习生", "实习", "顾问",
    "架构师", "分析师", "设计师", "开发", "测试", "运维", "产品", "运营",
    "负责人", "研究员", "助理", "组长", "cto", "vp",
]


def _has_job_title(it: TextItem) -> bool:
    low = it.text.lower()
    words = low.split()
    return any(t in words for t in _JOB_TITLES) or any(
        t in low for t in _JOB_TITLES if not t.isascii()
    )


_DATE_F = _has(r"(?:19|20)\d{2}")
_MONTH_F = _has(r"(?:19|20)\d{2}[.\-/年]\s*\d{1,2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|月")
_PRESENT_F = _has(r"至今|今|Present|Now")

DATE_FEATURE_SETS: list[FeatureSet] = [
    _fs(_DATE_F, 2, True),
    _fs(_MONTH_F, 1),
    _fs(_PRESENT_F, 1),
    _fs(lambda it: "," in it.text, -1),
]

TITLE_FEATURE_SETS: list[FeatureSet] = [
    _fs(_has_job_title, 4, True),
    _fs(lambda it: bool(re.search(r"\d", it.text)), -4),
    _fs(lambda it: len(it.text.split()) > 5 or len(it.text) > 20, -2),
]

_DEGREE_F = _has(r"博士|硕士|研究生|MBA|本科|学士|大专|专科|高中|bachelor|master|phd|mba|degree")


def extract_profile(lines: Lines) -> tuple[dict[str, str], dict[str, Any]]:
    items = [it for ln in lines for it in ln]
    out: dict[str, str] = {}
    debug: dict[str, Any] = {}
    for field_name, fsets in PROFILE_FEATURE_SETS.items():
        text, scores = _highest_scored(
            items, fsets, concat_ties=(field_name == "summary")
        )
        out[field_name] = text
        debug[field_name] = scores
    return out, debug


_WORK_KW = ["work", "experience", "employment", "job", "工作", "任职", "实习"]
_EDU_KW = ["education", "教育", "学历", "院校"]
_PROJ_KW = ["project", "项目", "作品"]
_SKILL_KW = ["skill", "技能", "专业"]
_SUMMARY_KW = ["summary", "objective", "自我评价", "个人评价", "个人总结", "自我总结", "求职意向"]


def _years_from_range(date_range: str) -> float:
    import datetime as _dt

    now_y = _dt.datetime.now(_dt.UTC).year
    m = _DATE_RANGE_RE.search(date_range)
    if not m:
        return 0.0
    head = re.match(r"((?:19|20)\d{2})(?:[.\-/年]?\s*(\d{1,2}))?", m.group(0))
    tail = m.group(1)
    if not head:
        return 0.0
    start = int(head.group(1))
    start_m = float(head.group(2) or 0)
    tm = re.match(r"((?:19|20)\d{2})(?:[.\-/年]?\s*(\d{1,2}))?", tail)
    end = now_y if not tm else int(tm.group(1))
    end_m = float(tm.group(2) or 0) if tm else 0.0
    months = (end_m - start_m) / 12
    if not (1950 <= start <= now_y and end >= start):
        return 0.0
    return max(0.0, min(end, now_y) - start + months)


def _segments(lines: Lines) -> list[TextItem]:
    """行内按「2+空格/竖线/制表」切段,继承首个 item 的粗体标记。

    中文简历头部信息常挤在一行:「2020.03-至今  高级后端工程师  阿里巴巴」,
    span 级粒度不够,需要再切一次让特征评分在字段级粒度工作。
    """
    out: list[TextItem] = []
    for ln in lines:
        text = _line_text(ln)
        if not text:
            continue
        bold = ln[0].bold if ln else False
        for seg in re.split(r"\s{2,}|\||\t", text):
            seg = seg.strip()
            if seg:
                out.append(TextItem(text=seg, bold=bold))
    return out


def _strip_date(text: str) -> str:
    return _DATE_RANGE_RE.sub("", text).strip(" -~–—至到,|")


def extract_work(lines: Lines) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    debug: list[Any] = []
    for sub in divide_section_into_subsections(lines):
        desc_idx = _descriptions_line_idx(sub)
        desc_idx = desc_idx if desc_idx is not None else min(2, len(sub))
        info_items = _segments(sub[:desc_idx])

        # 日期区间:对全部 info 文本正则取首个区间,比逐 span 打分更稳
        info_text = " ".join(it.text for it in info_items)
        dm = _DATE_RANGE_RE.search(info_text)
        date_range = dm.group(0).strip() if dm else ""

        # 剔除日期段后,职位/公司在剩余段里评分
        rest = [
            TextItem(text=_strip_date(it.text), bold=it.bold)
            for it in info_items
        ]
        rest = [it for it in rest if it.text]
        title, title_scores = _highest_scored(rest, TITLE_FEATURE_SETS)

        company_fsets: list[FeatureSet] = [
            _fs(_has(r"公司|集团|企业|厂|银行|医院|学校|机构|中心|事务所|科技|有限"), 3),
            _fs(lambda it: it.bold, 2),
            _fs(lambda it: len(it.text) <= 24 and not re.search(r"\d", it.text), 1),
            _fs(_has_text(title), -6),
            _fs(_has_job_title, -4),
        ]
        company, company_scores = _highest_scored(
            rest, company_fsets, empty_if_not_positive=False
        )
        if company == title:
            company = ""
        out.append(
            {
                "company": company,
                "title": title,
                "date_range": date_range,
                "years": round(_years_from_range(date_range), 1),
                "descriptions": bullet_descriptions(sub[desc_idx:]),
            }
        )
        debug.append({"title": title_scores, "company": company_scores})
    return out, debug


def extract_education(lines: Lines) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    debug: list[Any] = []
    for sub in divide_section_into_subsections(lines):
        segs = _segments(sub)
        info_text = " ".join(it.text for it in segs)
        dm = _DATE_RANGE_RE.search(info_text)
        date_range = dm.group(0).strip() if dm else ""

        rest = [TextItem(text=_strip_date(it.text), bold=it.bold) for it in segs]
        rest = [it for it in rest if it.text]
        degree, deg_s = _highest_scored(
            rest, [_fs(_DEGREE_F, 5, True), _fs(lambda it: it.bold, -1)]
        )
        school_fsets: list[FeatureSet] = [
            _fs(_has(r"大学|学院|学校|university|college|institute|school"), 5),
            _fs(lambda it: it.bold, 1),
            _fs(_has_text(degree), -4),
            _fs(lambda it: len(it.text) <= 24, 1),
        ]
        school, ss = _highest_scored(rest, school_fsets, empty_if_not_positive=False)
        desc_idx = _descriptions_line_idx(sub) or 1
        out.append(
            {
                "school": school,
                "degree": degree,
                "date_range": date_range,
                "descriptions": bullet_descriptions(sub[desc_idx:]),
            }
        )
        debug.append({"degree": deg_s, "school": ss})
    return out, debug


def extract_projects(lines: Lines) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sub in divide_section_into_subsections(lines):
        desc_idx = _descriptions_line_idx(sub) or 0
        head_items = [it for ln in sub[: max(desc_idx, 1)] for it in ln]
        name, _ = _highest_scored(
            head_items,
            [
                _fs(lambda it: it.bold, 2, True),
                _fs(lambda it: len(it.text) <= 30, 1),
                _fs(_DATE_F, -2),
            ],
            empty_if_not_positive=False,
        )
        dm = _DATE_RANGE_RE.search(" ".join(it.text for it in head_items))
        out.append(
            {
                "name": name,
                "date_range": dm.group(0) if dm else "",
                "descriptions": bullet_descriptions(sub[desc_idx:]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def structure_items(items: list[TextItem]) -> dict[str, Any] | None:
    """主入口:items → 结构化 sections(失败返回 None)。"""
    if not items:
        return None
    try:
        lines = group_items_into_lines(items)
        sections = group_lines_into_sections(lines)

        profile, profile_scores = extract_profile(sections.get("profile") or [])
        work_lines = _section_lines_by_keywords(sections, _WORK_KW)
        edu_lines = _section_lines_by_keywords(sections, _EDU_KW)
        proj_lines = _section_lines_by_keywords(sections, _PROJ_KW)
        skill_lines = _section_lines_by_keywords(sections, _SKILL_KW)
        summary_lines = _section_lines_by_keywords(sections, _SUMMARY_KW)

        works, work_debug = extract_work(work_lines)
        edus, edu_debug = extract_education(edu_lines)
        projects = extract_projects(proj_lines)

        skills_text = ", ".join(bullet_descriptions(skill_lines)).strip(", ")
        summary_text = " ".join(_line_text(ln) for ln in summary_lines).strip()
        if summary_text:
            profile["summary"] = summary_text or profile.get("summary", "")

        def _conf(found: int, total: int) -> float:
            return round(found / total, 2) if total else 0.0

        confidence = {
            "profile": _conf(sum(1 for k in ("name", "email", "phone") if profile.get(k)), 3),
            "work": _conf(sum(1 for w in works if w.get("date_range") or w.get("title")), max(len(works), 1)),
            "education": _conf(sum(1 for e in edus if e.get("school") or e.get("degree")), max(len(edus), 1)),
        }
        confidence["overall"] = round(
            (confidence["profile"] + confidence["work"] + confidence["education"]) / 3, 2
        )

        return {
            "profile": profile,
            "work_experiences": works,
            "educations": edus,
            "projects": projects,
            "skills_text": skills_text,
            "confidence": confidence,
            "section_names": list(sections.keys()),
            "parser": "feature-scoring-v1",
            # 调试数据(供解析报告 UI)
            "debug": {
                "profile_scores": profile_scores,
                "work_scores": work_debug,
                "education_scores": edu_debug,
            },
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("resume_structurer 失败,fail-open: %s", e)
        return None


def structure_resume(file_name: str, mime: str, data: bytes, raw_text: str) -> dict[str, Any] | None:
    """按格式分发:PDF/DOCX 走字体坐标,TXT/其他走纯文本行。"""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    try:
        if mime == "application/pdf" or ext == "pdf":
            return structure_items(read_pdf_items(data))
        if ext == "docx" or "wordprocessingml" in (mime or ""):
            items = read_docx_items(data)
            if items:
                return structure_items(items)
            return structure_items(read_txt_items(raw_text))
        return structure_items(read_txt_items(raw_text))
    except Exception as e:  # noqa: BLE001
        logger.warning("structure_resume 异常 fail-open: %s", e)
        return None
