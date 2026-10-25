# AI 导入向导设计文档

> 状态:已实现(T8)。调研对象:master_import 式四步导入向导;
> 本文档描述 HROnfire 的 AI 化实现方案与最终形态。

## 1. 目标与背景

现有 `POST /api/employees/import` / `POST /api/talents/import` 是裸
`csv.DictReader`:列名必须与字段名完全一致,错一行就跳过,没有映射层、
没有错误文件下载。每个新客户上线的第一件事就是灌历史数据——这是
导入失败最高发的环节。

本设计把导入升级为四步向导,并把其中最适合 LLM 的环节 AI 化:

```
上传(dataset+文件) → AI 列映射 → 校验/预览 → 执行+错误文件下载
```

旧 `/import` 端点**保留不动**(test_t5 依赖,也是免 AI 的快速通道)。

## 2. 参考实现要点(已消化)

| 参考点 | 取舍 |
|---|---|
| `ImportJob` 存 file + mapping + edits/skipped + summary | 模型同构;我们用 LargeBinary 存文件字节(无文件存储层) |
| `Dataset` 注册表:Field(key/label/required/aliases/kind/choices/suggest) | 同构 `IMPORT_DATASETS` 注册表 |
| `suggest_mapping` normalize+difflib 启发式 | 保留作**兜底**;LLM 映射为主,启发式补空缺/降级 |
| `apply_mapping` 行号从 2 起(对齐表格行号) | 同 |
| `check` → Checked{valid, errors[{row,messages,by_field}], will_create} | 同构;will_create 简化为 orgs |
| review 页可编辑单元格 + 跳过行 | v1 不做(错误行自动排除,改映射重校验即可) |
| create/update/both 三模式 | v1 只做 create(与现有 import 语义一致:重复跳过) |
| 错误工作簿下载 | 保留:CSV(原列+错误说明列) |

## 3. 数据模型(迁移 0053)

`import_jobs`:

| 列 | 说明 |
|---|---|
| id / tenant_id / created_by | 常规 |
| dataset | `employees` / `candidates` |
| file_name / file_bytes(LargeBinary) | 原文件,≤10MB |
| columns / row_count | 解析出的列名 / 行数 |
| mapping / mapping_meta | `{field_key: column}` + `{field_key: {confidence, source, reason}}` |
| status | `uploaded → mapped → checked → done / failed` |
| check_result | `{valid, errors[], warnings[], will_create, ai_note}` |
| run_summary | `{created, skipped, errors}` |
| created_at / finished_at | |

行上限 2000;文件 ≤10MB。

## 4. 数据集注册表 `IMPORT_DATASETS`

每个数据集声明:`fields`(key/label/required/aliases/kind/choices)、
`suggest_options`(org_name → 租户 OrgUnit 名称列表)、
`validate_row(row, ctx)` → `{field: msg}`、`commit(rows, ctx)` → summary。

- **employees**:name*(别名:姓名/员工姓名/Name)、employee_no、work_email(email)、
  phone(tel)、designation、org_name(suggest)、employment_type(choice:
  full_time/part_time/intern/contractor,接受中文别名映射:全职/兼职/实习/外包)、
  hire_date(date)
- **candidates**:name*、email、phone、tags(`;`/`,`/`、`分隔)、org_name

去重规则沿用现有实现:employee_no(文件内+库内)、候选人 contact hash(email/phone)。
日期解析尝试 `%Y-%m-%d`、`%Y/%m/%d`、`%d/%m/%Y`、`%Y年%m月%d日`、Excel 序列日期。

## 5. API(`api/imports.py`,前缀 `/api/imports`)

| 端点 | 说明 | 权限 |
|---|---|---|
| `GET /datasets` | 数据集+字段 spec+suggest 选项 | 登录 |
| `GET /datasets/{key}/template` | CSV 模板下载 | 登录 |
| `POST /` (multipart) | 上传:解析→建 job,回 columns/row_count/前3行预览 | 数据集权限 |
| `GET /` | 本租户导入历史 | 登录 |
| `GET /{id}` | 详情(映射/校验结果/汇总) | 数据集权限 |
| `POST /{id}/suggest` | **AI 映射**:LLM 为主+启发式兜底,写 mapping/meta | 数据集权限 + `assistant.agent` |
| `PUT /{id}/mapping` | 人工改映射 | 数据集权限 |
| `POST /{id}/check` | 校验:valid/errors/warnings/will_create + AI 错误归因摘要 | 数据集权限 |
| `POST /{id}/run` | 执行导入(valid 行)→ summary,写 audit `import` | 数据集权限 |
| `GET /{id}/errors.csv` | 错误文件下载 | 数据集权限 |
| `DELETE /{id}` | 删除任务记录 | 数据集权限 |

- `suggest`/`check` 里 LLM 不可用时**降级不报错**:suggest 回退纯启发式
  (meta.source=heuristic),check 跳过 AI 摘要。
- `run` 的 audit `action="import"` → T7 automation 的 create 触发器别名
  覆盖,规则可监听"员工导入完成"。

## 6. AI 层

**`import.column_map` 提示词**(PROMPT_REGISTRY, module="imports"):
输入 = 目标字段 spec(含别名/样例/类型)+ 文件列名 + 每列 2-3 个样本值;
输出 `{"mapping": {field_key: "列名"或""}, "confidence": {field_key: 0-1}, "warnings": []}`。
返回后服务端校验:列名必须在文件列集合内(防幻觉)、同一列不重复占用
(保留高置信度一方);LLM 留空/失败的字段由启发式补。

**`import.check_note` 提示词**:输入 = 错误行聚合(字段×错误计数+样例行);
输出一句中文归因摘要 + 修复建议(≤200 字)。错误 0 条时不调。

mock provider 加 `import` 分支:`response_json` 时返回
`{"mapping": {}, "warnings": ["mock"]}` → 走启发式兜底,测试无 key 可跑通;
单元测试用 monkeypatch 直接控制 `chat` 返回值。

## 7. 前端 `/settings/imports`(`ImportWizard.tsx`)

四步 stepper:

1. **选择+上传**:数据集卡片(图标+说明+字段 spec 表)+拖拽/选择文件+
   模板下载;上传后进入映射
2. **映射**:目标字段 → 文件列下拉;AI 建议标置信度徽章(高绿/中黄/低红),
   "AI 智能映射"按钮(无 license 时隐藏,启发式结果仍可人工调);
   必填字段未映射时禁止下一步
3. **校验**:valid/错误计数、错误表(行号+字段+原因)、AI 归因摘要卡片、
   will_create(将新建的组织);"执行导入"按钮;可返回改映射重校验
4. **结果**:created/skipped/errors 汇总、错误 CSV 下载、历史任务列表入口

导航:设置组「数据导入」;Employees/TalentPool 页保留旧快速导入按钮,
加「智能导入 →」链接。

## 8. 依赖与降级

- `openpyxl>=3.1` 加入 pyproject(xlsx 解析;csv 走 stdlib)。容器内先
  `pip install` 供联调,镜像 rebuild 后自然带上。
- 无 pandas——行解析用 `csv` + `openpyxl.load_workbook(read_only)`。
- LLM 任何环节失败:启发式兜底/跳过 AI 摘要,向导不因 LLM 不可用而断。

## 9. 验证矩阵

- 单测 `test_t8.py`:解析(csv/xlsx/超大/空文件/坏编码)、启发式映射、
  AI 映射合并(防幻觉列名/防重复占用/降级)、日期/枚举校验、去重、
  run+audit+automation 联动、错误文件下载、权限/租户隔离、模板下载
- 全量回归
- `verify_t8_live.py`:真实 LLM(stub_llm_server)→ 中文表头文件 →
  AI 映射→校验→执行→错误文件全链路
- ego-browser:向导四步 UI、AI 徽章、错误表、执行汇总、历史列表
