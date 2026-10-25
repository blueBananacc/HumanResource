# H1–H3 人事 AI 助手提效场景设计

> 状态:已实现(2026-10)。五个场景全部走"确定性计算 + LLM 编排/解读"架构,写操作一律确认门 + 幂等键。

## 背景

调研 frappe/hrms、ever-gauzy、Horilla 后,用户放弃大薪酬引擎范围,选择 5 个 AI 助手提效场景:

| 用户话术 | 批次 | 落点 |
|---|---|---|
| 我还有几天年假? | H1 | `get_leave_balance` 工具 |
| 今天有 12 条考勤异常已核对 | H1 | `list_attendance_flags` + `excuse_attendance_flags`(确认门) |
| 给我一份上月入离职情况 | H1 | `hr_movements` report-def + KPI 卡 |
| 按最新讨论方案执行组织调整 | H2 | `POST /org/changes`(dry_run 预览/事务执行) |
| 生成 3 月排班,有请假自动找替班 | H3 | Shift/ShiftAssignment 域 + 生成器 + 替班求解 + 月历页 |

## H1:问答与考勤核对

- **`get_leave_balance`**:无参走 `/me/leave-balances`(本人);带 `employee_id` 走 `/api/leave/balances?employee_id=`(HR 查他人,行级权限由 backend scope 判定)。账号未绑定员工档案时返回明确提示而非报错。
- **`list_attendance_flags`**:`GET /attendance/flags` 新增 `day` 查询参数,支持按天核对。
- **`excuse_attendance_flags`**(WRITE_TOOLS):`POST /attendance/flags/bulk-excuse`。逐 flag 幂等(已免计跳过)、逐条 `can_decide_for_employee` 行级守卫(只能免计直属下属/有考勤权限范围)、写审计。助手 playbook 要求:列出明细 → LLM 可给建议但"是否合理"由用户拍板 → 确认门放行。
- **`hr_movements`**:注册进 `REPORT_DEFS`(`services/hr_movements.py`),纯 SQL 聚合:期初/期末在册、入/离/调/升计数、净增减、离职率、按部门分布。助手 `run_report` 零改动直接驱动,前端 `ToolResultCard` KPI 卡。

## H2:会话式组织调整

「最新讨论方案」= 会话内自然语言方案,不落方案实体。

- **`POST /org/changes`**(admin-only):
  - `dry_run=true`:逐条校验(父节点存在/重名/祖先链环检测 `_ensure_not_descendant` 同款),返回每条的执行计划 + 警告(如"未指定新上级沿用原汇报关系")+ errors。**环在预览阶段即返回 400**。
  - `dry_run=false`:单事务执行;支持 `create_org_unit`/`rename`/`move`/`transfer_employee`/`set_manager`;员工调动自动校正 `reports_to_id`;写 `transfer` LifecycleEvent + 审计。
- 前端 `OrgChangesPreviewCard`:变更清单 + 可执行标记 + 逐条警告。
- 流程:描述方案 → `preview_org_changes` 预览卡 → 确认门 → `apply_org_changes` 执行卡。

## H3:智能排班(新域)

**模型**(迁移 0061):`Shift`(名称/起止分钟/每班需人数/适用部门/开关)、`ShiftAssignment`(员工×日期×班次,source=auto/manual/cover)。

**生成器**(`services/scheduling.py:generate_schedule`):
- 硬约束:工作日历过滤(周末/节假日不排)、已批准请假不排、连续工作天数上限、班次间休息间隔、已有排班不重复填充。
- 软约束:周末班次轮换 + 每人工作日计分均衡(贪心选当班得分最低者)。
- `dry_run` 预览不落库;返回 `planned/plan/understaffed/workdays/shifts/employees/month`。

**替班**:
- `find_gaps`:覆盖不足槽位 + 「排了班但人请假」待替班项。
- `auto_cover`:候选池(同部门优先、当日无假无班、未超连班/休息约束)→ 确定性评分排序 → dry_run 预览或落库,返回 `covered/filled/remaining`。

**API**(`api/shifts.py`,`hr.suite` feature 门):班次 CRUD + `POST /generate` + `GET /gaps` + `GET /cover-candidates` + `POST /auto-cover` + 手动增删 assignment。

**UI**:`/attendance/shifts` 月历表(行=日,列=班次,chip=员工,缺口红标)+ 班次管理抽屉 + 生成/替班按钮;`ToolResultCard` 排班预览卡/替班结果卡。

**已知坑**:`plan` 明细大月可超 sidecar `BODY_LIMIT=8000` 截断 → 尾部 `month/understaffed` 丢失致卡片 JSON 解析失败。`compactPlan()` 压 plan 为前 15 条 + `plan_total` 计数。

## 助手接线

- 工具:`get_leave_balance`、`list_attendance_flags`、`excuse_attendance_flags`★、`preview_org_changes`、`apply_org_changes`★、`list_shifts`、`preview_shift_plan`、`generate_shifts`★、`list_shift_gaps`、`auto_cover_shifts`★(★=确认门)。
- playbook(`agent.ts`):人事段落编排五场景;组织调整必须先预览再执行;排班先预览,替班需确认。

## 验证

- `tests/test_h123.py` 10/10;全量回归 678/678。
- `scripts/verify_h123_live.py` 20/20(含幂等/行级守卫/环检测)。
- ego-browser 端到端(DeepSeek):假期余额(本人降级+HR查人卡片)、7 条异常列表→建议→确认门→批量免计成功、入离职月报 KPI 卡、组织调整预览→确认→执行 2 项、排班预览卡 + 月历页。
