# T13 声明式评分规则引擎 — 设计文档

> 移植 hr-assistant `scoring-utils.ts` / `scoring-rule-analysis-utils.ts` 的声明式
> 评分规则概念,叠加 HROnfire 的多租户 / LLM 网关留痕 / T12 Gate+证据评分体系。

## 1. 目标与定位

T12 的两段式(Gate→五维 criteria 评分)解决了"能不能做 / 做得好不好",
但五个评分维度是**写死的**,权重只能由 gate_profile.emphasis 在固定维度上
微调。T13 把"评什么、怎么评"变成**租户可声明的数据**:

- 规则 = 维度列表,每个维度声明一个**评估器方法**
- `keyword / duration / boolean / range` 是**确定性纯函数**,0 token、可复现、
  命中/缺失即证据
- `ai` 维度走一次**批量 LLM 调用**(所有 ai 维合并为一发),沿用 T12 的
  evidence+level 证据先行格式
- 规则可挂在多个岗位;评估时**快照冻结**进 match 记录,分数永远可回溯到
  当时的规则版本

## 2. 数据模型

### scoring_rules(新表)

| 列 | 类型 | 说明 |
|---|---|---|
| id | str36 | PK |
| tenant_id | str36 FK→tenants | 隔离 |
| name | str(200) | 规则名 |
| description | text | 适用场景说明 |
| dimensions | JSON | `[{id,name,weight,description,evaluator}]` |
| version | int | 每次修改/应用分析 +1 |
| created_by / created_at / updated_at | — | 常规 |

dimension.evaluator 方法(对齐 hr-assistant,参数裁剪):

| method | 参数 | 评估逻辑 | 成本 |
|---|---|---|---|
| keyword | keywords[], matchMode:any/all | 简历文本子串命中;any=命中比例+20 封顶,all=命中比例 | 0 |
| duration | minYears, preferredYears | parsed_data.experience 年限合计线性插值(60→100) | 0 |
| boolean | booleanField: education/experience/projects/skills | parsed_data 字段非空 → 100/0 | 0 |
| range | field: experience_years, minValue, maxValue | 数值区间线性映射 | 0 |
| ai | aiPrompt | 批量 LLM 一次调用 → {evidence, level 0-4} → ×25 | 1 次/候选 |

### jobs.scoring_rule_id(新列,nullable FK→scoring_rules ON DELETE SET NULL)

岗位不挂规则 → 走现有 T12 criteria 路径,零回归。

### resume_job_matches.rule_snapshot(新列,JSON nullable)

评估时冻结:`{rule_id, name, version, dimensions[]}` —— 即使规则后被
修改/删除,历史分数的评分依据完整可查(对齐 hr-assistant ScoringRuleSnapshot)。

criteria 行扩展一个字段 `method`(keyword/duration/boolean/range/ai),
前端按方法给徽标("规则·关键词" vs "规则·AI")。

## 3. 评估流水线

```
resume ─→ Gate (不变) ─→ job.scoring_rule_id?
                            ├─ None  → T12 criteria 路径(不变)
                            └─ Rule  → evaluate_with_rule:
                                        ├─ 确定性维:纯函数 → {score, evidence}
                                        ├─ ai 维:合并 1 次 call_llm_json
                                        │   → {evidence, level} → level*25
                                        ├─ 加权聚合 → score 0-100
                                        ├─ 红旗:level=0/raw=0 ∧ weight≥30 major,≥15 minor
                                        └─ rule_snapshot 落 match
```

- 维度产出统一 `{score:0-100, evidence}`;criteria 行
  `{key,name,weight,method,level,evidence}` —— 确定性维的 level = round(score/25) 供 UI 复用 L0-4 色条。
- ai 维 LLM 失败 → 该维 score=50 + evidence="AI评估暂不可用"**不阻断**其余维度
  (单维 fail-open,比整评 fallback 更细)。
- 整规则路径异常 → `_fallback_match`(与现状一致)。

## 4. API /api/scoring-rules

| 端点 | 说明 |
|---|---|
| GET / POST / PUT / DELETE | 租户内 CRUD;PUT/apply 使 version+1;校验维度 schema、权重再平衡提示 |
| POST /generate `{job_id}` | LLM 按 JD 起草规则(维度+评估器+权重),normalize 后返回草案,前端确认后 POST 保存 |
| POST /{id}/analyze | 规则 + 关联岗位 JD → {summary, coverageGaps, weightAdjustments, industrySignals, recommendedChanges, proposedRule} |
| POST /{id}/apply-analysis `{analysis}` | 应用 proposedRule,version+1 |

生成/分析调用点 `prompt_key=scoring_rule.generate|analyze`,走 call_llm_json
留痕(call_type 新增 scoring_rule.*)。

## 5. 前端

- 新页面 `ScoringRules.tsx`(路由 /scoring-rules,侧栏"招聘"组):
  列表 → 编辑器(维度行:名称/权重/方法/方法参数;权重合计提示)→
  "从岗位 AI 生成" → "规则分析"抽屉(缺口/建议/草案 diff → 一键应用)。
- `JobDetail` / 岗位编辑:scoring_rule 下拉(无=默认五维)。
- `JobMatches` 评分明细:criteria 行加 method 徽标 + 规则名(来自 rule_snapshot)。

## 6. Mock 与测试

- mock provider 新增三分支(分派词避开现有"评分"等触发词,置于匹配分支前):
  规则生成 → 混合方法维度;规则分析 → 草案;ai 维批量评估 → 每维 level+evidence。
- `test_t13.py`:normalize/rebalance、四种确定性评估器、批量 ai 调用一次、
  加权聚合/红旗、快照、CRUD+租户隔离、generate/analyze/apply、
  match_evaluator 挂规则路径、未挂规则零回归。
- `verify_t13_live.py`:API 建规则→生成→分析→应用→挂岗位→触发匹配→
  断言 criteria method/rule_snapshot/留痕。

## 7. 不做(本期)

- 维度拖拽排序 UI、规则模板市场、多规则 A/B;评估器不引入正则/向量方法
  (后续可加 embedding 相似度维)。
