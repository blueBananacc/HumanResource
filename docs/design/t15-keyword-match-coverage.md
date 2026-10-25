# T15 设计:CJK 整词匹配 + 关键词覆盖子分

## 背景

移植 Resume-Matcher 两个确定性机制:

1. `text_match.keyword_in_text` — CJK 感知整词匹配。现有子串匹配有
   已知误伤:`"Go"` 命中 "Django"、"java" 命中 "javascript"、
   "CV" 命中 "CVC"。规则:CJK 字符直接子串;非 CJK 词要求词边界
   (`(?<!\w)`),且 CJK 字符也算边界("会Go语言"里的 Go 命中)。
2. 关键词覆盖子分 — JD `skills` 与简历的整词覆盖率,零 LLM,
   作为 LLM criteria 评分的确定性旁证。

## 替换点(子串 → 整词)

| 位置 | 现状 | 风险 |
|---|---|---|
| `scoring_rules._eval_keyword` | `k.lower() in lower` | Go 命中 Django |
| `scoring_rules._eval_boolean` pats | 同上 | MBA 命中 "cumber..." |
| `resume_parser._extract_skills` | `s.lower() in lower` | 技能表误抽(如 "Go" 在 "Good") |
| `match_evaluator._fallback_match` | `s in text` | 兜底分虚高 |
| `provider.py` mock 命中 | `s.lower() in resume_text` | 测试语义与生产不一致 |

`search_index` 全局搜索保留子串(模糊搜索是预期行为)。

## 关键词覆盖(`keyword_coverage.py`)

```
compute_coverage(resume, job_skills, extra_keywords=[])
  corpus = parsed_text + parsed_data.skills + sections.skills_text
  hit     = [k for k in keywords if keyword_in_text(k, corpus)]
  missing = 其余(保持 JD 原序)
  score   = round(len(hit) / len(keywords) * 100)
  → {score, hit, missing}  keywords 为空 → None
```

- `evaluate_match`:非 gated 与 gated 结果都附 `coverage`
  (零成本,淘汰者也想看缺哪些词);兜底路径同样附
- 存储:`resume_job_matches.coverage` JSONB(迁移 0059)
- API:`MatchOut.coverage`
- T13 规则引擎:新增评估器方法 `coverage`
  (evaluator 无需参数;可用 `keywords` 覆盖默认 job.skills)
  → `_eval_deterministic(resume, text, job, dim)` 签名加 job

## 前端

- `JobMatches` criteria 明细上方:「关键词覆盖」行 — 百分比徽标 +
  `命中 n/m` + 缺失词 chip(最多 8 个,灰底);criteria 行内 method=coverage 同格式展示
- `ScoringRules` 编辑器评估器方法下拉加 `coverage`("关键词覆盖")

## 测试

- `test_t15.py`:keyword_in_text 边界(Go/Django、java/javascript、
  Go语言/会Go、C++)、coverage 计算/空 skills、评估器 coverage 方法、
  evaluate_match 带 coverage、gated 也带、API 输出、迁移列
- live:上传简历+岗位触发匹配 → match.coverage 命中/缺失正确
