# E 批次:员工/HR 提效(E1–E5)

承接 h123(H1–H3 招聘侧人事助手),本批把 AI 提效延伸到「员工服务 + HR 运营」侧:
E1 员工自助 / E2 入离职编排 / E3 统一审批中心 / E4 薪酬底表 / E5 NL 报表。

## E3 审批中心(本批核心)

**为什么**:此前 leave/overtime/attendance/expense/employee_request 五个域各有散装一级 approve 端点,
无多级链、无代理人、无统一收件箱。

### 模型(迁移 0062)

- `approval_policies`:request_type + priority + conditions[](AND)+ steps[](有序)+ active
- `approval_instances`:request_type+request_id 指回原单 + policy 快照 + current_step + status
- `approval_steps`:instance → 每级(resolver/approver_user/status/decision_note)
- `users.delegate_user_id`:用户级代理人(代收原审批人的待办)

### 路由与裁决 `services/approvals.py`

- `route_request(db, request_type, request_id, applicant)` 在五个创建端点调用:
  有命中策略 → 建 instance+steps(单据仍 pending);无命中 → 保持原一级审批路径(legacy)
- 条件按 request_type 白名单(leave:days/leave_type_id/org_unit_id 等),操作符 eq/ne/gt/lt/ge/le/in
- resolver:`manager`(申请人 reports_to,含代理人并入)/ `role:xxx` / `user:uuid`
- `decide_on_instance`:行锁防并发双裁决;非末级批准 → 推进下一步;末级批准/驳回 →
  `apply_domain_decision` 派发五个域各自的 `_apply_decision`(与旧端点共用同一函数,行为不发散)
- `cancel_instance_for`:员工撤回申请时关闭在途实例,防收件箱残留

### API `api/approvals.py`

- `GET /approvals/inbox`:step 项(我当前级)+ legacy 项(原有行级判定)统一聚合
- `POST /approvals/decide`:instance 模式走引擎,legacy 模式回打原域 `_apply_decision`
- `/approvals/policies` CRUD + `/approvals/delegate` GET/PATCH + `/approvals/history` + `/instances/{type}/{id}`

### 前端 `pages/hris/Approvals.tsx`

待审批列表(类型 tag/申请人/第N/M级/意见框/批准驳回)+ 审批策略 Tab(策略卡 + 新建表单:
名称/类型/优先级/条件下拉按类型白名单/步骤每行 `manager|名` 或 `role:xx|名`)+
代理人设置栏 + "我最近处理的"记录。

## E1 员工自助工具(sidecar)

`get_my_profile` / `get_my_attendance` / `my_request_status` / `get_leave_balance`(H1 已有)/
`create_leave_application` / `create_overtime_request`(后两个 WRITE_TOOLS 确认门)。
全部走 `/me/*` 公共 API,行级"本人"由后端保证,员工未绑档自动降级提示。

## E2 入离职编排

`run_lifecycle_event`(确认门)→ 现有 lifecycle 模板/checklist 引擎;
`preview_lifecycle_checklist` 只读预览卡(任务列表确认后再执行)。

## E4 薪酬底表

`services/payroll_input.py` + report-def `payroll_input`:按月聚合每员工
应出勤/实出勤/请假天数(按类型)/加班小时/异动事件——确定性聚合供财务/算薪系统消费,
**不做算薪引擎**(调研结论:frappe/horilla 均为规则引擎,工程量与风险不匹配当前阶段)。

## E5 NL 报表

`services/datasets.py` 白名单数据集(employees/leave_applications/attendance 等):
`query_dataset(dataset, dimensions, metrics, filters, time_range)` 维度/字段全白名单校验,
非法字段直接拒;`/api/datasets` + `list_datasets`/`query_dataset` 工具,权限与租户由后端判定。

## 验证

- `backend/tests/test_e3.py` 12/12(策略匹配/多级推进/驳回/代理人/撤回关实例/legacy 兼容/行锁)
- 全量回归 690/690;live `verify_e_batch_live.py` 41/41
- ego-browser:审批中心页(收件箱/批准/策略 CRUD/代理人)、助手收件箱卡/薪酬底表卡/数据集卡、
  员工 ESS 链路(余额查询/申请状态/确认门请假→经理收件箱)
