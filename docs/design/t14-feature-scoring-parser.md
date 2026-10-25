# T14 设计:特征评分式简历结构解析器(LLM 前置预处理)

## 背景与目标

`resume_parser.py` 目前只产出 `raw_text + email/phone/skills/name_hint`,
`parsed_data` 无结构化工作经历/教育 —— T13 的 `duration`/`boolean`/`range`
评估器只能全文正则,精度受限(分不出按段任职时长、学历与学校混淆等)。

移植 open-resume `parse-resume-from-pdf` 的特征评分管线到 Python,
作为 LLM 结构化(未来 M2)之前的**确定性预处理层**:

- 产出结构化 `parsed_data.sections`:work_experiences/educations/projects/
  skills_text/summary/profile(name/email/phone/location/summary)
- 每个字段带 `confidence`(特征得分)与 `*_scores` 调试数据 → 前端解析报告
- 零 LLM、零成本、可复现;失败 fail-open(结构化缺省时下游回退全文正则)

## 管线(对齐 open-resume 四段式)

```
PDF → pymupdf get_text("dict") → TextItem{text,font,size,bold,x,y,page}
DOCX/TXT → TextItem(无字体/坐标, bold=False, 行=段落)
  1. items → lines(pymupdf 原生 block/line;文本模式按 \n)
  2. lines → sections(标题检测: 粗体+短行+关键词命中; 中英双语关键词表)
  3. section → subsections(行距>典型行距×1.4;无坐标时按「行首日期区间」切)
  4. 字段抽取:FeatureSet 评分取最高分(正则命中正分,互斥特征负分)
```

### 特征集(本地化后)

- **section 标题**关键词:工作|任职|实习|教育|项目|技能|专业|证书|语言|
  自我评价|个人总结|求职意向|experience|education|project|skill|summary|award
- **日期区间**:`YYYY[.MM/年月]-(至今|YYYY)`,教育段同规则
- **职位 title**:中文职级词(工程师|经理|主管|专员|总监|实习|顾问|架构师|
  分析师|主管|负责人)+英文 JOB_TITLES 子集;负分:含数字/超5词
- **company**:粗体+短行;负分:含已识别 date/title
- **degree**(教育段):博士|硕士|研究生|MBA|本科|学士|大专|专科|高中|
  bachelor|master|phd|mba
- **bullet 描述**:10 种 Unicode bullet + 最高频切分(直接移植)

## 数据结构

`parsed_data.sections`(新键,JSON 列无迁移):

```json
{
  "profile": {"name","email","phone","location","summary"},
  "work_experiences": [{"company","title","date_range","years","descriptions":[]}],
  "educations": [{"school","degree","date_range","descriptions":[]}],
  "projects": [{"name","date_range","descriptions":[]}],
  "skills_text": "技能区原文",
  "confidence": {"work":0.0-1.0,"education":...,"overall":...},
  "parser": "feature-scoring-v1"
}
```

confidence = 各字段最高特征分 / 理论满分 截断到 [0,1]。

## 接入点

1. `resume_parser.py`:新增 `structure_resume(text, source_items)` →
   `ParseResult.sections`;PDF 走 items 路径,DOCX/TXT 走文本路径
   (docx 顺带读取 `<w:b/>` run 粗体标记增强标题检测)
2. `workers/tasks/resume.py` + upload 同步降级:写 `parsed_data["sections"]`
3. `scoring_rules.py` 评估器升级(优先结构化,raw_text 兜底):
   - `duration`/`range`:优先 `sum(work_experiences[].years)`(区间去重)
   - `boolean`:`education` → educations 非空;`experience` → work 非空
   - `keyword`:原文不变(全文匹配本就合理)
4. `api/resumes.py` detail 输出 `sections` + `confidence`
5. 前端简历详情:「解析报告」折叠区 — 分段展示各字段+置信度条

## 边界

- 多栏 PDF:pymupdf block 已按列分组,行内 x 排序即可;乱序严重仍走
  `_maybe_dedup_garbled` 全文兜底
- 无坐标文本(DOCX/TXT):bold 信号弱 → section 检测主要靠关键词;
  subsection 按「行首匹配日期区间」切分
- 结构化失败/为空:下游一律回退现有全文正则,零回归

## 测试

- `test_t14.py`:合成 pymupdf PDF(构造 span)+ DOCX/TXT fixtures;
  section 切分、subsection 行距、字段特征分、confidence、
  评估器结构化优先、API 输出
- live:真实中文简历 fixture 走上传→解析→匹配全链
- ego-browser:简历详情「解析报告」
