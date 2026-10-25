"""CJK 感知整词匹配(T15)—— 移植 Resume-Matcher text_match.py。

CJK 字符直接子串匹配(中文词无空格边界);非 CJK 词要求词边界,
CJK 字符同样构成边界 —— "Go" 不匹配 "Django","java" 不匹配
"javascript",但 "会Go语言" 里的 Go 命中。
"""
from __future__ import annotations

import re

# Han(含扩展平面)、平/片假名、韩文音节/字母
_CJK_CHAR_CLASS = (
    r"[ᄀ-ᇿ぀-ヿ㄰-㆏ㇰ-ㇿ"
    r"㐀-鿿ꥠ-꥿가-퟿豈-﫿"
    r"ｦ-ￜ\U0001b000-\U0001b16f\U00020000-\U0003ffff]"
)
_CJK_RE = re.compile(_CJK_CHAR_CLASS)


def keyword_in_text(keyword: str, text: str) -> bool:
    """CJK 子串或有边界的非 CJK 词匹配。"""
    kw = (keyword or "").strip().lower()
    if not kw:
        return False
    txt = (text or "").lower()
    if _CJK_RE.search(kw):
        return kw in txt
    pattern = (
        rf"(?:(?<!\w)|(?<={_CJK_CHAR_CLASS}))"
        rf"{re.escape(kw)}"
        rf"(?:(?!\w)|(?={_CJK_CHAR_CLASS}))"
    )
    return bool(re.search(pattern, txt))


def normalize_skill_key(skill: str) -> str:
    """技能名规范化(大小写/空白不敏感比较)。"""
    return re.sub(r"\s+", " ", (skill or "").strip()).casefold()
