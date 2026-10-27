"""轻量 BM25 词面排序(应用层实现,零外部依赖)。

设计要点:
- 分词:ASCII 小写词 + CJK 单字 + CJK 相邻 bigram(中文检索标准做法,
  不需要 jieba 等重型分词器,对制度类短文档足够)。
- 字段加权:title 词项重复 3 次、tags 重复 2 次、content 1 次,
  等价于 field-boost 的 BM25F 简化版。
- IDF 以「当前租户已发布文章全集」为语料(量小,Python 直算开销可忽略);
  词项只出现在候选池时 IDF 退化仍保持可排序。
- 输出按 max 归一化到 [0,1],供混合检索与语义分相加。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_K1 = 1.2
_B = 0.75

_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """ASCII 词 + CJK 单字 + CJK bigram。"""
    if not text:
        return []
    text = text.lower()
    toks = _ASCII_RE.findall(text)
    cjk = _CJK_RE.findall(text)
    toks.extend(cjk)
    toks.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return toks


def _doc_tokens(title: str, tags: list[str] | None, content: str) -> list[str]:
    """字段加权展开:title×3 + tags×2 + content×1。"""
    toks: list[str] = []
    t = tokenize(title)
    toks.extend(t * 3)
    for tag in tags or []:
        toks.extend(tokenize(tag) * 2)
    toks.extend(tokenize(content))
    return toks


def bm25_scores(
    query: str,
    corpus: list[tuple[str, str, list[str] | None, str]],
    candidate_ids: set[str] | None = None,
) -> dict[str, float]:
    """对候选文档算 BM25 分,返回 {doc_id: 归一化分(0-1]}。

    ``corpus`` 项为 (doc_id, title, tags, content);IDF 在整个 corpus 上统计。
    ``candidate_ids`` 非空时只对这些文档打分(仍用全语料 IDF)。
    """
    q_toks = tokenize(query)
    if not q_toks or not corpus:
        return {}

    stats: list[tuple[str, Counter, int]] = []
    df: Counter = Counter()
    for did, title, tags, content in corpus:
        tf = Counter(_doc_tokens(title, tags, content))
        dl = sum(tf.values())
        stats.append((did, tf, dl))
        for term in tf:
            df[term] += 1

    n = len(stats)
    avgdl = sum(dl for _, _, dl in stats) / n if n else 1.0

    raw: dict[str, float] = {}
    for did, tf, dl in stats:
        if candidate_ids is not None and did not in candidate_ids:
            continue
        score = 0.0
        for term in q_toks:
            f = tf.get(term)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / avgdl))
        if score > 0:
            raw[did] = score

    if not raw:
        return {}
    mx = max(raw.values())
    return {did: s / mx for did, s in raw.items()}
