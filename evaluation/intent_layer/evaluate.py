"""Intent Layer 自动化评测脚本。

评估指标：
- Intent Accuracy：模型正确识别出用户意图的比例（精确集合匹配）。
- Multi-intent 覆盖率：模型识别出的意图覆盖预期意图的比例（Recall）。

运行方式（项目根目录）：
  conda activate HR
  set PYTHONPATH=src
  python evaluation/intent_layer/evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# 确保 src 在 import 路径中
_project_root = Path(__file__).resolve().parents[2]
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from human_resource.agents.orchestrator import register_default_tools
from human_resource.intent.classifier import IntentClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 数据加载 ─────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent / "data.json"
_RESULTS_PATH = Path(__file__).parent / "results.json"


def load_test_cases() -> list[dict]:
    """加载评测用例。"""
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 指标计算 ─────────────────────────────────────────────────


def compute_intent_accuracy(expected: list[str], actual: list[str]) -> float:
    """Intent Accuracy：预期意图集合与实际意图集合的精确匹配。

    完全一致返回 1.0，否则返回 0.0。
    """
    return 1.0 if set(expected) == set(actual) else 0.0


def compute_multi_intent_coverage(expected: list[str], actual: list[str]) -> float:
    """Multi-intent 覆盖率：实际意图覆盖预期意图的比例（Recall）。

    coverage = |expected ∩ actual| / |expected|
    """
    if not expected:
        return 1.0
    expected_set = set(expected)
    actual_set = set(actual)
    return len(expected_set & actual_set) / len(expected_set)


# ── 单条评测 ─────────────────────────────────────────────────


def evaluate_single(
    classifier: IntentClassifier,
    case: dict,
) -> dict:
    """对单条测试用例运行意图分类并计算指标。"""
    query = case["query"]
    expected_intents = case["expected_intents"]
    expected_tools = case["expected_tools"]

    # 调用分类器
    result = classifier.classify(query)

    # 提取实际意图和工具
    actual_intents = [item.label.value for item in result.intents]
    actual_tools = list(result.requires_tools)

    # 计算指标
    intent_acc = compute_intent_accuracy(expected_intents, actual_intents)
    coverage = compute_multi_intent_coverage(expected_intents, actual_intents)

    return {
        "query": query,
        "expected_intents": expected_intents,
        "expected_tools": expected_tools,
        "actual_intents": actual_intents,
        "actual_tools": actual_tools,
        "intent_accuracy": intent_acc,
        "multi_intent_coverage": coverage,
    }


# ── 主流程 ───────────────────────────────────────────────────


def run_evaluation() -> list[dict]:
    """运行完整评测流程并返回结果列表。"""
    # 注册工具（分类器 prompt 需要工具列表）
    register_default_tools()

    classifier = IntentClassifier()
    cases = load_test_cases()
    results: list[dict] = []

    logger.info("开始评测，共 %d 条用例", len(cases))

    for idx, case in enumerate(cases, 1):
        logger.info("[%d/%d] 评测: %s", idx, len(cases), case["query"])
        record = evaluate_single(classifier, case)
        results.append(record)
        logger.info(
            "  → 意图准确: %.1f | 覆盖率: %.1f | 实际意图: %s",
            record["intent_accuracy"],
            record["multi_intent_coverage"],
            record["actual_intents"],
        )

    return results


def print_summary(results: list[dict]) -> None:
    """打印评测汇总报告。"""
    total = len(results)
    if total == 0:
        print("无评测结果。")
        return

    avg_accuracy = sum(r["intent_accuracy"] for r in results) / total
    avg_coverage = sum(r["multi_intent_coverage"] for r in results) / total

    # 多意图子集统计
    multi_cases = [r for r in results if len(r["expected_intents"]) > 1]
    multi_accuracy = (
        sum(r["intent_accuracy"] for r in multi_cases) / len(multi_cases)
        if multi_cases
        else float("nan")
    )
    multi_coverage = (
        sum(r["multi_intent_coverage"] for r in multi_cases) / len(multi_cases)
        if multi_cases
        else float("nan")
    )

    print("\n" + "=" * 70)
    print("  Intent Layer 评测报告")
    print("=" * 70)
    print(f"  用例总数:              {total}")
    print(f"  Intent Accuracy (avg): {avg_accuracy:.2%}")
    print(f"  Multi-intent 覆盖率:   {avg_coverage:.2%}")
    print("-" * 70)
    print(f"  多意图用例数:          {len(multi_cases)}")
    print(f"  多意图 Accuracy:       {multi_accuracy:.2%}")
    print(f"  多意图 Coverage:       {multi_coverage:.2%}")
    print("=" * 70)

    # 逐条详情
    print("\n逐条详情:")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        status = "✓" if r["intent_accuracy"] == 1.0 else "✗"
        print(f"  [{status}] {i}. {r['query']}")
        print(f"      预期意图: {r['expected_intents']}")
        print(f"      实际意图: {r['actual_intents']}")
        print(f"      预期工具: {r['expected_tools']}")
        print(f"      实际工具: {r['actual_tools']}")
        print(
            f"      Accuracy: {r['intent_accuracy']:.1f}  "
            f"Coverage: {r['multi_intent_coverage']:.1f}"
        )
    print("-" * 70)


def save_results(results: list[dict]) -> None:
    """将评测结果保存到 results.json。"""
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评测结果已保存: %s", _RESULTS_PATH)


def main() -> None:
    start = time.time()
    results = run_evaluation()
    elapsed = time.time() - start

    save_results(results)
    print_summary(results)
    print(f"\n  耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
