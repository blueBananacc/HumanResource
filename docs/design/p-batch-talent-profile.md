# P 批次:员工人才画像 + 内部人才市场

> 定位:HRBP 向的人才全景与内部配置能力 —— 「管理者看清人才全貌,
> AI 按能力标签实时演进,新项目一句话从内部摇人」。
> 调研参考:Moka(智能标签双通道/BP Eva 选人)、HiBob(统一技能框架)、
> 北森数字人才(画像=聚合视图+标签带来源+盘点校准)。

## 一、架构总览

```
业务域(简历/面试/项目/绩效/洞察) ──证据──▶ profile_update 管线
                                            │  确定性种子 → confirmed
                                            │  LLM 提炼   → suggested
                                            ▼
                              employee_skills(唯一键 employee×skill)
                                            │
                              talent_card.build_card 聚合视图
                                            │
              ┌── EmployeeDetail「人才画像」区块(确认/驳回/上传纪要)
              ├── /projects 项目页(登记+成员+done 触发建议)
              └── 助手:get_talent_card / search_talent / create_project /
                  add_project_member / add_insight_note(写全走确认门)
```

**核心不变量(与全系统一致)**:
- AI 建议永不直接落 `confirmed`,一律 `suggested` 待 HR/主管确认
  —— 画像可信度护栏,对应北森「盘点校准」与 Moka 人工标签通道。
- 真相散在业务域,画像是只读聚合;标签必须带 `evidence{kind,id,quote}`。
- 助手写操作(create_project/add_project_member/add_insight_note)
  全部进 `WRITE_TOOLS` 确认门,权限/租户仍由 backend 裁定。

## 二、数据模型(迁移 0063)

| 表 | 要点 |
|---|---|
| `employee_skills` | employee×skill 唯一;`level`1-5;`source` ∈ resume/interview/appraisal/project/insight/manual/ai;`status` ∈ suggested/confirmed;`evidence` JSON 出处 |
| `projects` | 轻量登记:name/desc/status(draft→active→done/cancelled)/时段/`required_skills` JSON/owner。**只做人才归属载体,不做任务管理** |
| `project_members` | project×employee 唯一(在岗);role/allocation_pct/`joined_via`(manual/ai_match 留痕,供事后统计 AI 命中率)/contribution_summary |
| `insight_notes` | 画像证据统一输入口:kind ∈ meeting(会议纪要手动上传)/project_review/manager_note/milestone;body≤8000;visibility manager_only/employee_visible |

## 三、画像更新管线(profile_update.py)

| 触发点 | 方式 | 落点 |
|---|---|---|
| 入职建卡 | `seed_card_from_candidate`:CandidateSkill→EmployeeSkill | confirmed(来源可信:resume/interview) |
| insight 新建 | `suggest_from_text`:LLM `talent.suggest` 提示词,≤6 条,复用技能字典 | suggested |
| 项目置 done | `suggest_from_project_done`:贡献总结+角色喂 LLM | suggested |
| 绩效复核 | `suggest_from_appraisal`:manager_review 喂 LLM | suggested |
| 手动加技能 | API 直写 | confirmed |

`upsert_employee_skill` 合并语义:同对唯一,level 取高,confirmed 不降级,
evidence 覆盖补注。LLM 不可用 → 优雅降级返回 `llm_error`,不影响主流程。

## 四、API

- `GET /employees/{id}/talent-card` — 聚合卡(行级 scope);`GET /me/talent-card` 本人视图(suggested 隐藏、manager_only 洞察过滤)
- `POST /employees/{id}/skills` — 手动加/改(manage_employees 或直属上级)
- `POST /employees/{id}/skill-suggestions/{confirm|reject}` — 批量裁决
- `GET/POST /employees/{id}/insights` — 洞察 CRUD(`extract_skills` 开关)
- `GET/POST/PUT/DELETE /projects[/{id}]` + `/members` — 项目与成员
- `POST /talent/search` — 确定性预筛:skills 片段 → 字典 ILIKE →
  confirmed(可选含 suggested)→ min_level → scope → 按命中片段数+等级和排序;
  LLM 在结果上做 topK 与推荐理由(T16 sourcing 同款两段式)

## 五、前端与助手

- `EmployeeDetail` 嵌入 `TalentCardSection`:头部徽章(入职天数/待确认数)+
  能力标签 cloud(confirmed 实心/suggested 虚线)+ 全部确认/驳回 +
  项目经历(AI 推荐徽章)+ 洞察时间线 + 「加技能」「上传会议纪要」弹窗
  (提交后即时展示 LLM 提取出的建议)
- `/projects` 页:列表+详情(所需技能/成员/AI 推荐标记)+新建/加成员弹窗
  +「标记完成(提炼成员能力)」
- 助手结果卡:`TalentMatchCard`(勾选→注入「把以下员工加入项目」)、
  `get_talent_card` 摘要卡、`list_projects` 列表卡
- playbook 人才画像剧本:需求→拆技能片段→search_talent→带证据 topK→
  add_project_member(ai_match)→ 完成闭环

## 六、验证

- 单测 `test_p123.py` 8/8:合并语义/建卡种子/建议确认驳回/越权 scope/
  insight mock-LLM/本人脱敏/项目 done 幂等触发/search 排序过滤
- live `verify_p_batch_live.py` 16/16
- 全量回归 698/698
- ego-browser:画像区块渲染+建议确认流转+上传纪要真 LLM 提取
  (组件化重构/工程管理/前端架构 → 待确认)+ 项目页成员/状态 +
  助手 search_talent 匹配卡 + add_project_member 确认门 +
  409 已在项目中的优雅报错;console 零错误

## 七、明确不做(后续批次候选)

九宫格盘点/校准会、继任计划、LMS 课程推荐、项目任务管理、
真实会议系统(飞书/腾讯)对接 —— 当前 insight 手动上传入口覆盖输入侧。
