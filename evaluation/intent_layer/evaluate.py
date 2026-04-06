"""Intent Layer 自动化评测脚本（IntentAnalyzer 版）。

评估 IntentAnalyzer 生成的自然语言意图提示是否包含正确的意图类别。

评估指标：
- Intent Accuracy：提取出的意图集合与预期意图集合精确匹配的比例。
- Multi-intent Coverage（Recall）：提取出的意图覆盖预期意图的比例。
- 平均延迟：单次 analyze 调用的平均耗时。

运行方式（项目根目录）：
  conda activate HR
  set PYTHONPATH=src
  python evaluation/intent_layer/evaluate.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

# 确保 src 在 import 路径中
_project_root = Path(__file__).resolve().parents[2]
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from human_resource.intent.analyzer import IntentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 数据路径 ─────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent / "data.json"
_RESULTS_PATH = Path(__file__).parent / "results.json"

# IntentAnalyzer 支持的意图类别（与 analyzer.py prompt 保持一致）
VALID_INTENTS = {
    "policy_qa",
    "process_inquiry",
    "employee_lookup",
    "memory_recall",
    "chitchat",
    "unknown",
}


def load_test_cases() -> list[dict]:
    """加载评测用例。"""
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 意图提取 ─────────────────────────────────────────────────


def extract_intents_from_hints(hints: str) -> list[str]:
    """从自然语言 intent hints 中提取意图类别关键词。

    策略：
    1. 先尝试匹配 "label1 + label2" 格式（多意图）
    2. 再逐个扫描已知意图类别
    3. 去重、保持出现顺序
    """
    if not hints:
        return ["unknown"]

    found: list[str] = []
    hints_lower = hints.lower()

    for label in VALID_INTENTS:
        if label in hints_lower:
            found.append(label)

    if not found:
        return ["unknown"]

    # 按在 hints 中出现的位置排序，保持顺序
    found.sort(key=lambda x: hints_lower.index(x))
    return found


# ── 指标计算 ─────────────────────────────────────────────────


def compute_intent_accuracy(expected: list[str], actual: list[str]) -> float:
    """Intent Accuracy：精确集合匹配，完全一致返回 1.0，否则 0.0。"""
    return 1.0 if set(expected) == set(actual) else 0.0


def compute_multi_intent_coverage(expected: list[str], actual: list[str]) -> float:
    """Multi-intent Coverage（Recall）：actual 覆盖 expected 的比例。"""
    if not expected:
        return 1.0
    return len(set(expected) & set(actual)) / len(set(expected))


# ── 单条评测 ─────────────────────────────────────────────────


def evaluate_single(
    analyzer: IntentAnalyzer,
    case: dict,
) -> dict:
    """对单条测试用例运行意图分析并计算指标。"""
    query = case["query"]
    expected_intents = case["expected_intents"]

    # 调用 IntentAnalyzer
    start = time.time()
    hints = analyzer.analyze(query)
    latency = time.time() - start

    # 从 hints 提取意图类别
    actual_intents = extract_intents_from_hints(hints)

    # 计算指标
    intent_acc = compute_intent_accuracy(expected_intents, actual_intents)
    coverage = compute_multi_intent_coverage(expected_intents, actual_intents)

    return {
        "query": query,
        "expected_intents": expected_intents,
        "actual_hints": hints,
        "actual_intents": actual_intents,
        "intent_accuracy": intent_acc,
        "multi_intent_coverage": coverage,
        "latency_s": round(latency, 3),
    }


# ── 汇总 ─────────────────────────────────────────────────────


def run_evaluation() -> dict:
    """执行完整评测并输出汇总。"""
    cases = load_test_cases()
    analyzer = IntentAnalyzer()

    results: list[dict] = []
    total_acc = 0.0
    total_cov = 0.0
    total_latency = 0.0

    print("\n" + "=" * 70)
    print("  Intent Layer 评测 — IntentAnalyzer")
    print("=" * 70)

    for i, case in enumerate(cases, 1):
        r = evaluate_single(analyzer, case)
        results.append(r)

        total_acc += r["intent_accuracy"]
        total_cov += r["multi_intent_coverage"]
        total_latency += r["latency_s"]

        status = "✓" if r["intent_accuracy"] == 1.0 else "✗"
        print(
            f"  [{status}] #{i:02d} | Acc={r['intent_accuracy']:.0f} "
            f"Cov={r['multi_intent_coverage']:.2f} "
            f"| {r['latency_s']:.2f}s"
        )
        print(f"       Q: {r['query']}")
        print(f"       Expected: {r['expected_intents']}")
        print(f"       Actual:   {r['actual_intents']}")
        print(f"       Hints:    {r['actual_hints'][:120]}")
        print()

    n = len(cases)
    summary = {
        "total_cases": n,
        "intent_accuracy": round(total_acc / n, 4) if n else 0,
        "multi_intent_coverage": round(total_cov / n, 4) if n else 0,
        "avg_latency_s": round(total_latency / n, 3) if n else 0,
        "total_latency_s": round(total_latency, 3),
    }

    print("=" * 70)
    print(f"  总计: {n} cases")
    print(f"  Intent Accuracy:        {summary['intent_accuracy']:.2%}")
    print(f"  Multi-intent Coverage:  {summary['multi_intent_coverage']:.2%}")
    print(f"  Avg Latency:            {summary['avg_latency_s']:.3f}s")
    print(f"  Total Latency:          {summary['total_latency_s']:.3f}s")
    print("=" * 70)

    # 保存详细结果
    output = {"summary": summary, "results": results}
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存: {_RESULTS_PATH}")

    return output


if __name__ == "__main__":
    run_evaluation()
