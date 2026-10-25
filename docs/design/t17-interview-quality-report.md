# T17:组织面试质量报告 + 洞察页

> 目标:复刻截图场景 —— AI 助手生成「近 30 天组织面试质量报告」(卡片进对话),并可展开为「面试质量洞察」页:KPI(面试总量/维度覆盖率/通过率/平均时长)+ 维度覆盖情况(哪些考察项经常被遗漏)。

## 家底(全部复用)

| 现有 | 作用 |
|---|---|
| `report_defs.py` 注册表(kind → 结构化 dict) | **新报表只需加一个 builder,助手 `run_report` 工具零改动可用** |
| `Interview.summary` = `{recommendation, dimension_scores{dim:score}, evidence_quotes, comment}` | AI 面试的维度分/录用建议数据源 |
| `InterviewScorecard.criteria` = `[{key,name,dimension∈tech/soft/general,weight}]` + `InterviewFeedback.scores`/`verdict` | 人类面试官考察项与表态数据源 |
| `InterviewTurn.scores` = `{accuracy,completeness,clarity}` + `score_status` | 逐题分兜底 |
| `QuestionItem.dimensions` | 题库考察项词表(维度全集参考) |
| `notify.py _digest_for` + `ReportSubscription` | 「报告已生成」主动推送通道(加 kind 即可) |
| `get_org_scope/apply_org_filter` | 组织范围口径,与 reports.py 一致 |

## 一、聚合服务 `services/interview_quality.py`(纯确定性,零 LLM)

`build_quality_report(db, tenant_id, user, days=30) -> dict`:

```python
{
  "days": 30, "generated_at": ...,
  "kpis": {
    "interviews_total": 47,            # 窗口内 status=done 的面试数(mode!=self_test)
    "interviews_delta": "+12%",        # 与前一等长窗口环比
    "dimension_coverage": 78,          # 已考察维度数 / 计划维度数(口径见下)
    "pass_rate": 41,                   # 建议录用(recommendation=hire 或 verdict=recommend)占比
    "avg_duration_min": 48,            # avg(finished_at - started_at)
  },
  "dimensions": [                      # 维度覆盖情况,降序
    {"name": "结构化思维", "coverage": 92, "avg_score": 7.8, "scored_in": 43},
    ...
  ],
  "missed": [{"name": "业务敏感度", "coverage": 38}],  # coverage<50 的子集
  "by_job": [{"job_id","title","interviews","pass_rate"}],   # 岗位下钻(stretch)
}
```

**口径定义**(全部 explainable,可写进卡片注释):

- **维度全集** = 窗口内评分卡 `criteria[].name` ∪ AI summary `dimension_scores` keys(经 `normalize_skill_key` 归一合并「系统设计」/「系统设计能力」类近名);可选并入 `question_library.dimensions` 作为"应考"基线。
- **某维 coverage** = 覆盖该维的面试数 / 参与评分的面试总数(AI: dimension_scores 含该 key;人类: feedback.scores 命中该 criterion)。
- **维度覆盖率 KPI** = Σ各维命中次数 / (维度全集数 × 参与评分的面试数) —— 即"该考的维度里实际考了多少"。
- **通过率** = 结论为推荐(`summary.recommendation` ∈ hire/推荐,或 scorecard `verdict=recommend`) / 有结论的面试数。
- **missed** = coverage < 50% 的维度,卡片区单列「经常被遗漏」。
- org scope:`apply_org_filter(Interview.join(Candidate), Candidate.org_unit_id)` 与 reports.py 同口径;`mode=self_test` 排除。

## 二、注册与入口(零新工具)

- `REPORT_DEFS["interview_quality"] = {label:"面试质量报告", perm:PERM_VIEW_REPORTS, run:...}` —— 助手 `list_report_kinds`/`run_report` 立即可用,**sidecar 零改动**。
- 页面数据源:直接复用 `POST /api/report-defs/interview-quality/run` 即可,不新开端点。
- 主动推送:`notify.py _digest_for` 加 `interview_quality` 分支 → 已有 `create_report_subscription` 工具可订「每周质量报告」→ digest 通知带 `/reports/interview-quality` 深链 —— 即截图 1 的「报告已生成」主动消息。

## 三、助手与卡片

- **system prompt playbook 加一段**:「问面试质量/维度遗漏/通过率 → `run_report(kind=interview_quality, params{days})`,解读时先给 KPI 再给遗漏维度建议(可建议 `suggest_scorecard_criteria`/题库补题)」—— 反馈建议由 LLM 基于结构化数据生成,报告本身零幻觉。
- **ToolResultCard 新卡 `InterviewQualityCard`**(识别 `result.kind=="interview_quality"`):4 个 KPI 格 + 维度条形(top 5 + missed 红条)+「点击展开完整报告」按钮 → `navigate("/reports/interview-quality?days=30")`。

## 四、洞察页 `frontend/src/pages/InterviewQuality.tsx`

按截图 2 复刻:`/reports/interview-quality`:
- 顶部时间选择:近 7/30/90 天(query param 驱动)
- KPI 卡 ×4:面试总量(带环比箭头)/维度覆盖率/通过率/平均时长
- 「维度覆盖情况」条形列表:名称 + 百分比条(≥80 绿 / 50-79 黄 / <50 红)+ 命中场数,副标题「哪些考察项经常被遗漏」—— <50% 维度沉底并标红
- 每条维度可点开 → 该维 `avg_score` + 命中面试列表(链 summary-view)
- 「让 AI 解读」按钮 → 注入助手话术「帮我解读近 30 天面试质量报告」

## 五、测试与验证

- `test_t17`:聚合口径(维度归并/coverage/通过率/时长/环比)、空窗口零值、org scope、self_test 排除、report-defs 注册可见性、租户隔离。
- `verify_t17_live`:造 2 场 done 面试(dimension_scores 不同维)+ 评分卡 feedback → run → 断言 KPI 与维度条形正确。
- ego-browser:助手问「近 30 天面试质量」→ InterviewQualityCard 渲染 → 跳洞察页 → 页面 KPI/条形/遗漏区正确。
- 回归全量。

## 批次

- **T17-1** `services/interview_quality.py` + report_defs 注册 + notify digest
- **T17-2** 洞察页 + InterviewQualityCard + agent playbook
- **T17-3** 测试 + live + ego-browser + DOCS-MAP

遗留修正:此前 ego 演示种子面试 status='finished',实际口径为 'done' —— 实施时一并 `UPDATE`。
