# T16 对话式招聘(Conversational Recruiting)设计方案

> 目标:用户在 AI 助手里说「急招一个 AI 工程师」→ 助手复用/新建**人才画像** →
> 扫描人才库给出带核心匹配点的候选短名单(可勾选)→ 一键安排面试 →
> 面试后生成结构化**面试纪要**。对标"招聘 Eva"演示的五屏流程。

## 1. 截图流程 ↔ 现有能力映射

| 演示步骤 | 我方现状 | 缺口 |
|---|---|---|
| 「急招X」→ 回忆起上次画像问是否沿用 | 无画像实体;Job+ScoringRule 可表达画像内容 | **缺 TalentProfile 实体** |
| 「内部人才库上次效果最好」渠道记忆 | resume.source 有来源字段,无效果统计 | 延后(stretch) |
| 扫描人才库 → 5 人短名单 | `search_talents` 按名字模糊搜;匹配是 job 级(evaluate-all→ResumeJobMatch) | **缺跨岗位临时筛选** |
| 每人一句匹配点 + 勾选 + 面试官偏好排序 | match 已有 strengths/gaps/criteria/coverage 可凑「核心匹配点」;面试官偏好无 | **缺可交互候选人卡片** |
| 选人 → 安排面试 | `create_interview` 工具+确认卡已就绪 | 仅接线 |
| 面试纪要/反馈面板 | `Interview.summary` + `synthesize_scorecard`(多面试官维度合成结论)已就绪 | 纪要卡片渲染 + 触发入口 |

## 2. 实体设计:TalentProfile(人才画像)

```python
class TalentProfile(Base):
    tenant_id, name            # "AI 工程师画像"
    spec: JSON = {
        skills: [..],          # 必备技能(喂 coverage)
        keywords: [..],        # 加分关键词(keyword_in_text)
        min_years: 3,          # 年限门槛(gate-lite)
        education: "本科",      # 学历门槛(gate-lite)
        nice_to_have: [..],
    }
    scoring_rule_id: FK→ScoringRule nullable   # 可选:精评规则
    source_job_id: FK→Job nullable             # 画像来源(从岗位/招聘沉淀)
    last_used_at, use_count                     # 「上次X月用过」的对话依据
    status: active/archived, created_by
```

- **画像 ≠ 岗位**:岗位是编制/JD,画像是可复用的"招什么样的人"模板。
- **生成来源**:`POST /api/talent-profiles` 手建;`POST /api/talent-profiles/from-job/{id}`
  从岗位 skills+description 抽取;`POST /api/talent-profiles/generate`(LLM 按一句话 JD 起草,
  走 call_llm_json → T11 留痕)。
- "3月招的 AI 工程师画像还适用吗" = agent 按名称模糊查到 `last_used_at` 最久未用的同名画像。

## 3. 核心服务:人才库扫描(sourcing scan)

`POST /api/sourcing/scan` `{profile_id | inline spec, limit=5}`:

```
候选集 = 租户内 status 正常的候选人最新已解析简历
  → 确定性预筛(零 LLM):
      gate-lite:min_years/education 命中 → 过滤
      coverage = compute_coverage(resume, spec.skills)
      keyword 加分:spec.keywords 整词命中率
      pre_score = 0.6*coverage + 0.4*keyword率 → 排序取 top K(默认 20)
  → 精评(top K 内,复用 T12 evaluate_match 单评,或 profile 挂了 scoring_rule 走规则):
      LLM 只对 top K 调用 → score/strengths/gaps
  → 返回 [{candidate_id, name, score, coverage{score,hit,missing},
           highlights[核心匹配点≤3], gaps≤2, resume_id}]
```

- **不建 Job 不落 match**:扫描是一次性查询;选中安排面试时由 `create_interview`
  需要 job_id —— 方案:画像可绑 `source_job_id`;无岗位时 agent 先建议
  「是否要顺手建个岗位」或允许 interview 的 job_id 为空(放宽校验,记 profile 名到备注)。
- `SourcingRun` 表留痕:{profile_id, spec 快照, scanned_total, returned, duration_ms}
  —— "内部人才库上次效果最好" 的统计原料 + T11 一致的审计哲学。
- 成本控制:预筛全免费;LLM 只打 top K;复用 T15 整词匹配保证精度。

## 4. 助手侧

**新工具**(sidecar tools.ts,全部薄包装 REST):

| 工具 | 写? | 说明 |
|---|---|---|
| `list_talent_profiles` | 读 | 查画像(名称模糊 + last_used_at 排序) |
| `create_talent_profile` | 写 | 从一句话/spec 建画像(走 confirm 卡) |
| `scan_talent_pool` | 读 | 触达 /api/sourcing/scan,返回候选卡数据 |
| `get_interview_summary` | 读 | 纪要:synthesize 输出 + turns 摘要 |
| (可选)`generate_interview_summary` | 写 | 对已完成面试跑 synthesize |

**Agent prompt playbook**(system prompt 追加,不改协议):

```
用户表达招聘意图时:
1. list_talent_profiles 找同名/相似画像 → 有则问"上次的X画像还适用吗"
2. 确认后 scan_talent_pool → 展示短名单,引导勾选
3. 用户指定候选人后 create_interview(confirm 卡)→ 发链接
4. 面试完成后 get_interview_summary → 纪要卡
```

**前端交互卡**(ToolResultCard 扩展,复用 CardShell/ScoreBadge/RowLink):

- `CandidatePickCard`:头像首字母 + 姓名 + ScoreBadge + 核心匹配点(≤3 条 chips)+
  一句风险(gap);行尾 checkbox 本地选中态;卡片底部「安排面试」按钮 →
  往输入框注入 `给 {name}(candidate_id=xx,画像=yy) 安排面试` 让用户确认发送
  (最小改动,零新协议);agent 收到后走既有 create_interview confirm 卡。
- `InterviewSummaryCard`:综合评价 bullet + 维度分 chips + 结论/建议 +
  链接到 `/interviews/{id}/report`。

## 5. 面试纪要

现有 `Interview.summary`(AI 面试自动出)+ `synthesize_scorecard`(人工面试官多份
反馈→合成结论:verdict/优势/风险/建议)已覆盖演示面板的数据源。本批只做:

- `GET /api/interviews/{id}/summary-view`:聚合 summary + scorecard 合成 +
  逐题 transcript 摘要 → 纪要视图数据(若无 scorecard 只用 AI summary)。
- 面试完成后(turns 全部评分)自动触发一次 synthesize?——**否**,保持显式调用,
  agent 在 `get_interview_summary` 发现未完成时提示「可先跑合成」。

## 6. Stretch(本批不做)

- **渠道效果记忆**:SourcingRun 已留痕;后续可统计各 source → hired 转化率,
  agent 开场白引用("内部人才库上次转化率最高")。
- **面试官偏好排序**:记录面试官历史打分分布对短名单重排;需要更多数据积累。
- **外部渠道扫描**(全渠道人才库):需要接外部源,当前只做内部库。

## 7. 实施批次

| 批次 | 内容 | 验证 |
|---|---|---|
| T16-1 | TalentProfile 实体+迁移+CRUD+sourcing scan(确定性预筛+LLM topK 精评)+SourcingRun 留痕 | test_t16 单测+回归 |
| T16-2 | sidecar 5 工具+agent prompt+CandidatePickCard 勾选卡+InterviewSummaryCard | live 脚本+ego 全链路 |
| T16-3 | summary-view 聚合端点(纪要面板数据源) | 并入 T16-2 |

测试沿用既有约定:test_t16(库内直插 profile+简历→scan 排序断言)、
verify_t16_live(HTTP:建画像→扫描→安排面试)、ego-browser(对话→画像沿用提示→
候选卡勾选→安排面试 confirm→纪要卡)。


## 8. iCIMS 批次:相似候选人召回(Talent Match,2026-10)

iCIMS Talent Match 机制:Similarity = skills match + experience match(对 model candidate 的
技能重叠 + 职位/任职年限相近),加银牌候选人(silver medalist:曾深入流程后被拒)召回叙事。

落地:`services/similar.py` + `GET /api/talents/{id}/similar`(view_reports):

- 技能集:CandidateSkill(level 加权)∪ parsed_data.skills → 加权命中率(精确 skill_id
  优先,CJK 关键词兜底)
- 经验画像:最近职位 title(结构化 work_experiences,噪声词剥离+token Jaccard)+
  年限接近度(15 分/年衰减)+ 学历等级差
- 可选语义融合:租户配 embedding 时 cos×100 占 20%,任何失败 fail-open
- 银牌标记:JobApplicationEvent 曾 to_stage∈{interview,offer} 且当前 rejected/withdrawn → +5
- 前端:TalentDetail「相似候选人」卡(ScoreRing+共享技能+银牌徽标);assistant 工具
  `find_similar_candidates`
