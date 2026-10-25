/** HR 业务工具 — 每个工具的 execute 就是对 HROnfire 公共 REST API 的一次调用。
 *
 * 鉴权:backend 在 /chat 请求里签发与当前用户等价的 JWT(tool_token),
 * sidecar 原样回带 — 租户隔离 / 组织可见域 / license feature / 权限位
 * 全部走既有中间件,sidecar 不做任何权限决策。写工具另外过一层
 * "用户在 UI 上确认"的门(见 agent.ts 的 beforeToolCall)。
 */
import { Type } from "@earendil-works/pi-ai";
import type { AgentTool } from "@earendil-works/pi-agent-core";

const BACKEND = process.env.BACKEND_URL || "http://backend:8000";
const BODY_LIMIT = 8000;
// 单文档读取(get_playbook 之类)的专用上限 —— 全文就是工具载荷,
// 通用 8000 会截掉 SOP 后半段且无法续读;30k 覆盖 article.content 的
// 实际体量(字段上限 50k,但真实 SOP 远小于此)同时保护模型上下文。
const DOC_LIMIT = 30000;

async function callApi(
  token: string,
  path: string,
  opts: { method?: string; body?: unknown } = {},
): Promise<unknown> {
  const resp = await fetch(`${BACKEND}${path}`, {
    method: opts.method ?? "GET",
    headers: {
      Authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });
  const text = await resp.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  if (!resp.ok) {
    const detail =
      typeof data === "object" && data && "detail" in data
        ? (data as { detail: unknown }).detail
        : data;
    return {
      error: `HTTP ${resp.status}`,
      detail: typeof detail === "string" ? detail : JSON.stringify(detail),
      hint: hintFor(resp.status),
    };
  }
  return data;
}

function hintFor(status: number): string | undefined {
  if (status === 402) return "该功能未开通(license feature 未启用),告知用户原因";
  if (status === 403) return "当前用户没有该操作的权限,告知用户而不是重试";
  if (status === 404) return "资源不存在,id 可能不对,先用列表类工具查到正确 id";
  return undefined;
}

function toolResult(data: unknown, bodyLimit: number = BODY_LIMIT) {
  const json = JSON.stringify(data, null, 0);
  // callApi 对 HTTP 失败返回 {error,detail,hint} 结构 — 标 isError:
  // pi-agent 认它为工具失败(tool_end is_error / 错误呈现给模型),
  // MCP 侧同步透传给客户端。
  const isError = typeof data === "object" && data !== null && "error" in data;
  return {
    content: [
      {
        type: "text" as const,
        text: json.length > bodyLimit ? json.slice(0, bodyLimit) + "…(截断)" : json,
      },
    ],
    details: {},
    isError,
  };
}

type ToolDef = AgentTool & { label: string };

function makeTools(token: string): ToolDef[] {
  const t = (path: string, opts?: { method?: string; body?: unknown }) =>
    callApi(token, path, opts).then(toolResult);

  return [
    {
      name: "search_talents",
      label: "查询人才库",
      description:
        "搜索候选人(按姓名/邮箱/电话模糊匹配),返回候选人列表含简历数、面试数、最高匹配分、最近活跃时间",
      parameters: Type.Object({
        q: Type.Optional(Type.String({ description: "搜索关键词" })),
        limit: Type.Optional(Type.Number({ description: "返回条数,默认 10" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/talents?q=${encodeURIComponent(p.q ?? "")}&limit=${Number(p.limit ?? 10) || 10}`,
        ),
    },
    {
      name: "list_jobs",
      label: "查询岗位",
      description: "列出岗位(可按 open/paused/closed 过滤),返回岗位 id、名称、等级、技能、状态",
      parameters: Type.Object({
        status: Type.Optional(
          Type.String({ description: "open / paused / closed,默认不限" }),
        ),
        q: Type.Optional(Type.String({ description: "按岗位名搜索" })),
        limit: Type.Optional(Type.Number({ description: "默认 20" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: String(p.limit ?? 20) });
        if (p.status) qs.set("status", p.status);
        if (p.q) qs.set("q", p.q);
        return t(`/api/jobs/?${qs}`);
      },
    },
    {
      name: "get_job_matches",
      label: "查询匹配结果",
      description:
        "查某岗位的简历↔岗位 AI 匹配结果(分数、亮点、短板、评语),按分数倒序",
      parameters: Type.Object({
        job_id: Type.String({ description: "岗位 id" }),
        min_score: Type.Optional(Type.Number({ description: "最低分过滤" })),
        limit: Type.Optional(Type.Number({ description: "默认 20" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/matches/job/${encodeURIComponent(p.job_id)}?min_score=${Number(p.min_score ?? 0) || 0}&limit=${Number(p.limit ?? 20) || 20}`,
        ),
    },
    {
      name: "list_interviews",
      label: "查询面试列表",
      description: "列出远程面试(候选人/岗位可搜索),返回状态、完成度、平均分",
      parameters: Type.Object({
        q: Type.Optional(Type.String({ description: "候选人或岗位名" })),
        limit: Type.Optional(Type.Number({ description: "默认 10" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/interviews/?q=${encodeURIComponent(p.q ?? "")}&limit=${Number(p.limit ?? 10) || 10}`,
        ),
    },
    {
      name: "get_interview_suggestions",
      label: "查询推荐面试",
      description:
        "查 AI 推荐的待确认面试项(匹配分>90 且尚未发起面试的候选人↔岗位对)",
      parameters: Type.Object({}),
      execute: async () => t("/api/interviews/suggested"),
    },
    {
      name: "get_interview_report",
      label: "查看面试报告",
      description:
        "获取某场已完成面试的汇总报告:各维度分数、推荐结论、证据片段、逐题明细",
      parameters: Type.Object({
        interview_id: Type.String({ description: "面试 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/interviews/${encodeURIComponent(p.interview_id)}/report`),
    },
    {
      name: "list_question_items",
      label: "查询题库",
      description: "搜索题库题目(按题干关键词/题型过滤)",
      parameters: Type.Object({
        q: Type.Optional(Type.String({ description: "题干关键词" })),
        kind: Type.Optional(
          Type.String({ description: "题型 tech/project/scenario/soft" }),
        ),
        limit: Type.Optional(Type.Number({ description: "默认 10" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: String(p.limit ?? 10) });
        if (p.q) qs.set("q", p.q);
        if (p.kind) qs.set("kind", p.kind);
        return t(`/api/question-library/?${qs}`);
      },
    },
    {
      name: "dismiss_interview_suggestion",
      label: "忽略推荐面试",
      description: "把某条 AI 推荐面试项标记为不再提示(需要 match_id)",
      parameters: Type.Object({
        match_id: Type.String({ description: "匹配记录 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/matches/${encodeURIComponent(p.match_id)}/dismiss-suggestion`, {
          method: "POST",
        }),
    },
    {
      name: "create_interview",
      label: "发起面试",
      description:
        "为候选人发起远程面试,生成邀请链接。需要候选人 id 和岗位 id;" +
        "如果用户只给了名字,先用 search_talents/list_jobs 查到 id 再调用",
      parameters: Type.Object({
        job_id: Type.String({ description: "岗位 id" }),
        candidate_id: Type.String({ description: "候选人 id" }),
        question_count: Type.Optional(
          Type.Number({ description: "题数 3/5/10,默认 5" }),
        ),
        kinds: Type.Optional(
          Type.Array(Type.String(), {
            description: "题型 tech/project/scenario/soft,默认全选",
          }),
        ),
        expires_in_hours: Type.Optional(
          Type.Number({ description: "链接有效期小时数,默认 48" }),
        ),
        delivery: Type.Optional(
          Type.String({ description: "link / email / both,默认 link" }),
        ),
        notify_email: Type.Optional(
          Type.String({ description: "候选人通知邮箱" }),
        ),
      }),
      execute: async (_id, p: any) =>
        t("/api/interviews/start", {
          method: "POST",
          body: {
            job_id: p.job_id,
            candidate_id: p.candidate_id,
            question_count: p.question_count ?? 5,
            kinds: p.kinds ?? ["tech", "project", "scenario", "soft"],
            expires_in_hours: p.expires_in_hours ?? 48,
            delivery: p.delivery ?? "link",
            notify_email: p.notify_email,
          },
        }),
    },
    {
      name: "generate_questions",
      label: "AI 生成题目",
      description:
        "让 AI 生成面试题并存入题库(可按岗位 id 关联,或归到某个分类)",
      parameters: Type.Object({
        job_id: Type.Optional(Type.String({ description: "岗位 id" })),
        category: Type.Optional(
          Type.String({ description: "题库分类名,留空归到默认" }),
        ),
        kind: Type.Optional(
          Type.String({ description: "tech/project/scenario/soft,默认 tech" }),
        ),
        difficulty: Type.Optional(
          Type.String({
            description: "initial/intermediate/advanced/expert,默认 intermediate",
          }),
        ),
        count: Type.Optional(Type.Number({ description: "1-20,默认 5" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/question-library/generate", {
          method: "POST",
          body: {
            job_id: p.job_id,
            category: p.category ?? "",
            kind: p.kind ?? "tech",
            difficulty: p.difficulty ?? "intermediate",
            count: Math.min(20, Math.max(1, p.count ?? 5)),
          },
        }),
    },
    // ===== T3:知识库 RAG =====
    {
      name: "search_knowledge",
      label: "检索知识库",
      description:
        "在公司知识库(制度/FAQ/福利/IT 指引)里做全文检索;回答员工政策类问题时先查这里,并引用文章标题",
      parameters: Type.Object({
        q: Type.String({ description: "检索关键词" }),
        limit: Type.Optional(Type.Number({ description: "默认 5" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/knowledge/search", {
          method: "POST",
          body: { q: p.q, limit: p.limit ?? 5 },
        }),
    },
    {
      name: "search_product_help",
      label: "检索产品帮助",
      description:
        "在 HROnfire 产品使用手册里检索(系统怎么用/功能在哪/操作步骤);用户问的是'这个系统怎么用'时先查这里,回答时引用章节名",
      parameters: Type.Object({
        q: Type.String({ description: "检索关键词,如'批量导入员工'" }),
        limit: Type.Optional(Type.Number({ description: "默认 5" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/knowledge/search", {
          method: "POST",
          body: { q: p.q, limit: p.limit ?? 5, category: "product_help" },
        }),
    },
    // ===== T3:通用实体元工具 =====
    {
      name: "list_entities",
      label: "列出可检索实体",
      description:
        "列出当前用户可检索的业务实体类型(候选人/岗位/员工/面试/工单/公告等)及各自可搜索字段;不确定查什么时先调它",
      parameters: Type.Object({}),
      execute: async () => t("/api/entities"),
    },
    {
      name: "search_records",
      label: "通用实体搜索",
      description:
        "对任意实体做模糊搜索(先用 list_entities 看可用 entity 与字段),返回 {id,label,secondary}",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key,如 talents/jobs/cases/employees" }),
        q: Type.Optional(Type.String({ description: "关键词" })),
        limit: Type.Optional(Type.Number({ description: "默认 10" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/search?q=${encodeURIComponent(p.q ?? "")}&limit=${Number(p.limit ?? 10) || 10}`,
        ),
    },
    {
      name: "count_records",
      label: "通用实体计数",
      description: "统计某实体有多少条命中(可带关键词),回答'有多少…'类问题",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        q: Type.Optional(Type.String({ description: "关键词" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/count?q=${encodeURIComponent(p.q ?? "")}`,
        ),
    },
    {
      name: "aggregate_records",
      label: "聚合实体字段",
      description:
        "对某实体的数值字段做聚合(sum/avg/min/max/count),回答'总额/平均/最高'类问题;" +
        "可用实体与字段:expenses.total、payslips.gross/net、matches.score、campaigns.total/sent_count/failed_count",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        agg: Type.String({ description: "sum / avg / min / max / count" }),
        field: Type.Optional(Type.String({ description: "聚合字段(count 不用)" })),
        q: Type.Optional(Type.String({ description: "关键词过滤" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ agg: String(p.agg ?? "count") });
        if (p.field) qs.set("field", p.field);
        if (p.q) qs.set("q", p.q);
        return t(`/api/entities/${encodeURIComponent(p.entity)}/aggregate?${qs}`);
      },
    },
    {
      name: "update_record",
      label: "更新业务记录",
      description:
        "更新某条记录的白名单字段(先用 search_records 拿 id);" +
        "可写:talents(display_email/display_phone/status/tags)、jobs(level/description/status)、" +
        "employees(phone/work_email/designation)、cases(priority)、skills(category/description)、" +
        "rooms(name/location/capacity/equipment/is_active)、interviews(notify_email)",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        id: Type.String({ description: "记录 id" }),
        fields: Type.Object({}, {
          additionalProperties: true,
          description: "要更新的字段与值",
        }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/entities/${encodeURIComponent(p.entity)}/${encodeURIComponent(p.id)}`, {
          method: "PATCH",
          body: { fields: p.fields ?? {} },
        }),
    },
    {
      name: "list_relationships",
      label: "穿透关联记录",
      description:
        "查一条记录关联的其它实体(候选人→申请/面试/offer,岗位→申请/面试/高分匹配," +
        "员工→申请/工单/请假/目标,工单→提交人/留言数等);一次调用穿透多类关联",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        id: Type.String({ description: "记录 id" }),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/${encodeURIComponent(p.id)}/relations`,
        ),
    },
    {
      name: "get_record_fields",
      label: "查实体字段schema",
      description:
        "在 create/update/link 前先查:某实体可写字段(必填/类型/枚举值)与可建关系" +
        "(linkable:jobs.skills/candidates.skills/candidates.groups)",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/entities/${encodeURIComponent(p.entity)}/schema`),
    },
    {
      name: "create_record",
      label: "创建业务记录",
      description:
        "创建白名单实体(先 get_record_fields 查必填与类型);" +
        "可建:notes(候选人备注,candidate_id+content)、knowledge(知识库草稿,title+content+category/tags/is_playbook)、" +
        "announcements(公告草稿,title+content+audience_org_unit_id);员工/岗位/工单等走专用工具",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        fields: Type.Object({}, {
          additionalProperties: true,
          description: "字段与值",
        }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/entities/${encodeURIComponent(p.entity)}`, {
          method: "POST",
          body: { fields: p.fields ?? {} },
        }),
    },
    {
      name: "delete_record",
      label: "删除业务记录",
      description:
        "删除白名单实体记录(高危,先确认);" +
        "可删:knowledge/announcements/notes/skills(未被引用)/reservations(取消语义);" +
        "候选人/岗位/员工/工单不可通用删除",
      parameters: Type.Object({
        entity: Type.String({ description: "实体 key" }),
        id: Type.String({ description: "记录 id" }),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/${encodeURIComponent(p.id)}`,
          { method: "DELETE" },
        ),
    },
    {
      name: "link_records",
      label: "建立记录关系",
      description:
        "给两条已有记录建 M:N 关联(先 get_record_fields 查 linkable);" +
        "jobs.skills(extra.importance=must|nice)、candidates.skills(extra.level=1-5)、candidates.groups",
      parameters: Type.Object({
        entity: Type.String({ description: "源实体 key" }),
        id: Type.String({ description: "源记录 id" }),
        rel_name: Type.String({ description: "关系名" }),
        related_id: Type.String({ description: "目标记录 id" }),
        extra: Type.Optional(Type.Object({}, {
          additionalProperties: true,
          description: "join 字段(如 importance/level)",
        })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/${encodeURIComponent(p.id)}/links`,
          {
            method: "POST",
            body: {
              rel_name: p.rel_name,
              related_id: p.related_id,
              extra: p.extra ?? {},
            },
          },
        ),
    },
    {
      name: "unlink_records",
      label: "解除记录关系",
      description: "拆除 link_records 建的 M:N 关联",
      parameters: Type.Object({
        entity: Type.String({ description: "源实体 key" }),
        id: Type.String({ description: "源记录 id" }),
        rel_name: Type.String({ description: "关系名" }),
        related_id: Type.String({ description: "目标记录 id" }),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/entities/${encodeURIComponent(p.entity)}/${encodeURIComponent(p.id)}` +
          `/links/${encodeURIComponent(p.rel_name)}/${encodeURIComponent(p.related_id)}`,
          { method: "DELETE" },
        ),
    },
    {
      name: "list_report_kinds",
      label: "查可跑报表",
      description: "列出当前用户可执行的内置报表类型(pipeline_funnel/weekly_summary/pending_items/headcount)",
      parameters: Type.Object({}),
      execute: async () => t("/api/report-defs"),
    },
    {
      name: "run_report",
      label: "执行内置报表",
      description:
        "运行一个内置报表并返回结构化数据(先 list_report_kinds 查可用 kind);" +
        "回答'招聘漏斗/本周摘要/我的待办/员工编制'类问题后顺带解读",
      parameters: Type.Object({
        kind: Type.String({ description: "报表 kind" }),
        params: Type.Optional(
          Type.Object({}, { additionalProperties: true }),
        ),
      }),
      execute: async (_id, p: any) =>
        t(`/api/report-defs/${encodeURIComponent(p.kind)}/run`, {
          method: "POST",
          body: { params: p.params ?? {} },
        }),
    },
    {
      name: "list_playbooks",
      label: "查流程剧本",
      description:
        "列出知识库里标记为'AI 剧本'的 SOP 文档(租户自定义标准流程);" +
        "执行多步业务(发 offer/审批/发面试/处理工单等)前先查有无剧本",
      parameters: Type.Object({}),
      // brief=true 只取摘要;再投影成 id+title 级字段 —— 全量正文 JSON 会被
      // BODY_LIMIT 截断,导致靠后的剧本连 id 都到不了模型,发现机制失效。
      execute: async () => {
        const data: any = await callApi(
          token,
          "/api/knowledge/articles?is_playbook=true&status_filter=published&limit=50&brief=true",
        );
        if (data && typeof data === "object" && "error" in data) {
          return toolResult(data);
        }
        const items = (data?.items ?? []).map((a: any) => ({
          id: a.id,
          title: a.title,
          category: a.category,
          updated_at: a.updated_at,
        }));
        return toolResult({ total: data?.total ?? 0, items });
      },
    },
    {
      name: "get_playbook",
      label: "读流程剧本",
      description: "读取一篇 AI 剧本全文(用 list_playbooks 返回的 id),拿到后严格按步骤执行",
      parameters: Type.Object({
        id: Type.String({ description: "剧本文章 id" }),
      }),
      // 剧本全文是工具载荷本身 —— 用 DOC_LIMIT,避免 SOP 步骤被静默截断
      execute: async (_id, p: any) =>
        callApi(token, `/api/knowledge/articles/${encodeURIComponent(p.id)}`).then(
          (data) => toolResult(data, DOC_LIMIT),
        ),
    },
    // ===== T3:员工服务 =====
    {
      name: "list_announcements",
      label: "查公司公告",
      description: "列出对当前用户可见的已发布公告(含已读标记)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/announcements"),
    },
    {
      name: "create_request",
      label: "提交员工申请",
      description:
        "帮当前用户提交员工申请(培训/福利/设备/健康/通用);先和用户确认类型与内容再调用",
      parameters: Type.Object({
        kind: Type.String({
          description: "general/training/benefit/equipment/wellness",
        }),
        subject: Type.String({ description: "一句话主题,≤128 字" }),
        detail: Type.Optional(Type.String({ description: "详细说明" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/requests", {
          method: "POST",
          body: { kind: p.kind, subject: p.subject, detail: p.detail },
        }),
    },
    {
      name: "list_my_requests",
      label: "查我的申请",
      description: "列出当前用户提交的员工申请及状态",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/requests"),
    },
    {
      name: "create_case",
      label: "提交 HR 工单",
      description:
        "帮当前用户提交 HR 服务台工单(薪酬/IT/行政/设施/假勤争议等);先和用户确认类型与优先级",
      parameters: Type.Object({
        subject: Type.String({ description: "一句话主题,≤128 字" }),
        kind: Type.Optional(
          Type.String({ description: "payroll/admin/it/facility/leave_dispute/other,默认 other" }),
        ),
        priority: Type.Optional(
          Type.String({ description: "low/normal/high/urgent,默认 normal" }),
        ),
        description: Type.Optional(Type.String({ description: "详细描述" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/cases", {
          method: "POST",
          body: {
            subject: p.subject,
            kind: p.kind ?? "other",
            priority: p.priority ?? "normal",
            description: p.description,
          },
        }),
    },
    {
      name: "list_my_cases",
      label: "查我的工单",
      description: "列出当前用户提交的 HR 工单及处理进展",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/cases"),
    },
    // ===== T3:薪酬带宽 =====
    {
      name: "check_salary_band",
      label: "查薪酬带宽",
      description:
        "查询某职级/岗位的薪酬带宽与建议薪资是否落在区间内(offer 定薪参考)",
      parameters: Type.Object({
        designation: Type.Optional(Type.String({ description: "职级/岗位名" })),
        amount: Type.Number({ description: "月薪金额(数字)" }),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/salary-bands/check?amount=${Number(p.amount) || 0}${p.designation ? `&designation=${encodeURIComponent(p.designation)}` : ""}`,
        ),
    },
    // ===== T3:触发式任务 + 报表订阅 =====
    {
      name: "create_watcher",
      label: "创建自动提醒",
      description:
        "创建触发式提醒:可选类型 candidate_stalled(候选人停滞,params.days)/interview_pending_report(面试未出报告)/case_open_hours(工单超时,params.hours)/request_pending_days(申请积压,params.days)/reservation_upcoming(预约将开始,params.minutes)",
      parameters: Type.Object({
        kind: Type.String({ description: "watcher 类型" }),
        params: Type.Optional(
          Type.Object({}, { additionalProperties: true }),
        ),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/watchers", {
          method: "POST",
          body: { kind: p.kind, params: p.params ?? {} },
        }),
    },
    {
      name: "list_watchers",
      label: "查我的提醒",
      description: "列出当前用户创建的触发式提醒(watcher)及启用状态",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/watchers"),
    },
    {
      name: "delete_watcher",
      label: "删除提醒",
      description: "删除当前用户的一条触发式提醒(先用 list_watchers 拿 id)",
      parameters: Type.Object({
        watcher_id: Type.String({ description: "watcher id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/me/watchers/${encodeURIComponent(p.watcher_id)}`, {
          method: "DELETE",
        }),
    },
    {
      name: "save_filter",
      label: "保存筛选条件",
      description:
        "把自然语言描述的列表筛选条件保存为可复用筛选器;page 是列表页标识(如 talents/cases/employees),params 是该页筛选字段的键值(如 {q:'张', status:'active'})",
      parameters: Type.Object({
        page: Type.String({ description: "列表页标识:talents/cases/employees/jobs" }),
        name: Type.String({ description: "筛选器名称,如'华东在面候选人'" }),
        params: Type.Object({}, {
          additionalProperties: true,
          description: "筛选字段键值,先想清楚该页有哪些筛选参数",
        }),
      }),
      execute: async (_id, p: any) =>
        t("/api/saved-filters", {
          method: "POST",
          body: { page: p.page, name: p.name, params: p.params ?? {} },
        }),
    },
    {
      name: "create_report_subscription",
      label: "订阅定时报表",
      description:
        "创建每周定时邮件报表:kind=pipeline_funnel(管道漏斗)/weekly_summary(每周摘要)/pending_items(待办提醒);weekday 0=周一,hour 0-23",
      parameters: Type.Object({
        kind: Type.String({ description: "报表类型" }),
        weekday: Type.Number({ description: "0=周一…6=周日" }),
        hour: Type.Number({ description: "发送小时 0-23" }),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/report-subscriptions", {
          method: "POST",
          body: { kind: p.kind, weekday: p.weekday, hour: p.hour },
        }),
    },
    {
      name: "list_report_subscriptions",
      label: "查我的报表订阅",
      description: "列出当前用户的每周报表订阅及启用状态",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/report-subscriptions"),
    },
    {
      name: "delete_report_subscription",
      label: "取消报表订阅",
      description: "删除当前用户的一条报表订阅(先用 list_report_subscriptions 拿 id)",
      parameters: Type.Object({
        subscription_id: Type.String({ description: "订阅 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/me/report-subscriptions/${encodeURIComponent(p.subscription_id)}`, {
          method: "DELETE",
        }),
    },
    // ===== T3:HR 侧补充工具 =====
    {
      name: "publish_announcement",
      label: "发布公司公告",
      description:
        "代当前用户发布一条全公司公告(需 manage_employees 权限);先和用户确认标题与内容",
      parameters: Type.Object({
        title: Type.String({ description: "公告标题,≤128 字" }),
        content: Type.String({ description: "公告正文" }),
      }),
      execute: async (_id, p: any) =>
        t("/api/announcements?publish=true", {
          method: "POST",
          body: { title: p.title, content: p.content },
        }),
    },
    {
      name: "list_cases",
      label: "查 HR 工单队列",
      description:
        "列出 HR 工单队列(需 manage_employees 权限);可按 status(new/assigned/in_progress/resolved/closed)过滤",
      parameters: Type.Object({
        status: Type.Optional(Type.String({ description: "状态过滤" })),
        q: Type.Optional(Type.String({ description: "主题关键词" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: "20" });
        if (p.status) qs.set("status_filter", p.status);
        if (p.q) qs.set("q", p.q);
        return t(`/api/cases?${qs}`);
      },
    },
    {
      name: "update_case_status",
      label: "更新工单状态",
      description:
        "更新 HR 工单状态(需 manage_employees 权限);status=resolved 时必须给 resolution",
      parameters: Type.Object({
        case_id: Type.String({ description: "工单 id" }),
        status: Type.String({
          description: "new/assigned/in_progress/resolved/closed",
        }),
        resolution: Type.Optional(
          Type.String({ description: "解决方案(resolved 必填)" }),
        ),
      }),
      execute: async (_id, p: any) =>
        t(`/api/cases/${encodeURIComponent(p.case_id)}/status`, {
          method: "POST",
          body: { status: p.status, resolution: p.resolution },
        }),
    },
    {
      name: "get_activity",
      label: "查近期动态",
      description: "获取租户近期活动流(创建/流转/预约/答卷等),回答'最近发生了什么'",
      parameters: Type.Object({
        days: Type.Optional(Type.Number({ description: "回看天数 1-30,默认 7" })),
        limit: Type.Optional(Type.Number({ description: "条数上限,默认 30" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/activity?days=${Number(p.days ?? 7) || 7}&limit=${Number(p.limit ?? 30) || 30}`,
        ),
    },
    // ===== T9:全局搜索 + 结构化面试评分卡 =====
    {
      name: "global_search",
      label: "全局搜索",
      description:
        "跨实体加权全文搜索(候选人/岗位/员工/面试/工单/申请单/知识文章/公告/报销单一并搜)," +
        "返回 {entity,title,sub,link}+facets;'找某条记录但不确定是哪类'时用," +
        "已知实体类型且只要少量结果时用 search_records 更省",
      parameters: Type.Object({
        q: Type.String({ description: "关键词" }),
        entity: Type.Optional(
          Type.String({ description: "限定实体:talents/jobs/employees/interviews/cases/requests/knowledge/announcements/expenses" }),
        ),
        limit: Type.Optional(Type.Number({ description: "默认 20" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ q: String(p.q ?? "") });
        if (p.entity) qs.set("entity", p.entity);
        if (p.limit) qs.set("limit", String(Number(p.limit) || 20));
        return t(`/api/search?${qs}`);
      },
    },
    {
      name: "list_scorecards",
      label: "查面试评分卡",
      description:
        "列出结构化面试评分卡(候选人详情/某岗位/状态过滤),返回聚合分与反馈数;查单个用 get_scorecard",
      parameters: Type.Object({
        candidate_id: Type.Optional(Type.String({ description: "候选人 id" })),
        job_id: Type.Optional(Type.String({ description: "岗位 id" })),
        status: Type.Optional(Type.String({ description: "open/closed" })),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: "20" });
        if (p.candidate_id) qs.set("candidate_id", p.candidate_id);
        if (p.job_id) qs.set("job_id", p.job_id);
        if (p.status) qs.set("status", p.status);
        return t(`/api/scorecards?${qs}`);
      },
    },
    {
      name: "get_scorecard",
      label: "查评分卡详情",
      description:
        "取一张评分卡全文:考察维度、指派面试官提交状态、逐维评分明细、聚合均分/分歧、AI 摘要",
      parameters: Type.Object({
        id: Type.String({ description: "评分卡 id" }),
      }),
      execute: async (_id, p: any) =>
        callApi(
          token,
          `/api/scorecards/${encodeURIComponent(p.id)}`,
        ).then((d) => toolResult(d, DOC_LIMIT)),
    },
    {
      name: "suggest_scorecard_criteria",
      label: "AI 建议考察维度",
      description:
        "按岗位技能/JD(+候选人技能)为结构化面试建议考察维度清单,只回草稿不落库",
      parameters: Type.Object({
        job_id: Type.Optional(Type.String({ description: "岗位 id" })),
        candidate_id: Type.Optional(Type.String({ description: "候选人 id" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/scorecards/ai-criteria", {
          method: "POST",
          body: { job_id: p.job_id ?? null, candidate_id: p.candidate_id ?? null },
        }),
    },
    {
      name: "create_scorecard",
      label: "创建面试评分卡",
      description:
        "为候选人创建结构化面试评分卡(criteria 考察维度 + interviewer_ids 指派的面试官 user.id);" +
        "可先调 suggest_scorecard_criteria 出草稿再与用户确认",
      parameters: Type.Object({
        candidate_id: Type.String({ description: "候选人 id" }),
        title: Type.String({ description: "评分卡标题,如 '张三 · 二面(技术)'" }),
        job_id: Type.Optional(Type.String({ description: "岗位 id" })),
        interview_id: Type.Optional(Type.String({ description: "关联的 AI 面试 id(可选)" })),
        criteria: Type.Optional(
          Type.Array(Type.Object({}, { additionalProperties: true }), {
            description: "[{name, dimension(tech/soft/general), weight(1-10)}]",
          }),
        ),
        interviewer_ids: Type.Optional(
          Type.Array(Type.String(), { description: "面试官 user.id 列表" }),
        ),
        scheduled_at: Type.Optional(Type.String({ description: "ISO 时间" })),
        location: Type.Optional(Type.String()),
        note: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) =>
        t("/api/scorecards", {
          method: "POST",
          body: {
            candidate_id: p.candidate_id,
            title: p.title,
            job_id: p.job_id ?? null,
            interview_id: p.interview_id ?? null,
            criteria: p.criteria ?? [],
            interviewer_ids: p.interviewer_ids ?? [],
            scheduled_at: p.scheduled_at ?? null,
            location: p.location ?? null,
            note: p.note ?? null,
          },
        }),
    },
    {
      name: "submit_scorecard_feedback",
      label: "提交面试反馈",
      description:
        "代当前用户提交其被指派评分卡的反馈(必须是该卡指派面试官):总评 rating 1-5、" +
        "verdict(recommend/hold/reject)、逐维评分 scores[{criterion=key,score,comment}]、notes",
      parameters: Type.Object({
        scorecard_id: Type.String({ description: "评分卡 id" }),
        rating: Type.Number({ description: "总评 1-5" }),
        verdict: Type.String({ description: "recommend / hold / reject" }),
        scores: Type.Optional(
          Type.Array(Type.Object({}, { additionalProperties: true }), {
            description: "[{criterion(维度key), score(1-5), comment?}]",
          }),
        ),
        notes: Type.Optional(Type.String({ description: "自由评语" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/scorecards/${encodeURIComponent(p.scorecard_id)}/feedback`, {
          method: "POST",
          body: {
            rating: p.rating,
            verdict: p.verdict,
            scores: p.scores ?? [],
            notes: p.notes ?? null,
          },
        }),
    },
    {
      name: "synthesize_scorecard",
      label: "AI 聚合评审摘要",
      description:
        "对一张已有面试官反馈的评分卡生成 AI 评审摘要(共识主题/分歧点/录用建议)," +
        "结果写入评分卡 ai_summary —— AI 建议仅供参考,录用决策以人工为准",
      parameters: Type.Object({
        id: Type.String({ description: "评分卡 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/scorecards/${encodeURIComponent(p.id)}/synthesize`, {
          method: "POST",
        }),
    },
  ];
}

/** 需要用户在界面上点"确认"才会真正执行的写操作 */
export const WRITE_TOOLS = new Set([
  "create_interview",
  "generate_questions",
  "dismiss_interview_suggestion",
  "create_request",
  "create_case",
  "create_watcher",
  "save_filter",
  "create_report_subscription",
  "delete_watcher",
  "delete_report_subscription",
  "publish_announcement",
  "update_case_status",
  "update_record",
  "create_record",
  "delete_record",
  "link_records",
  "unlink_records",
  "create_scorecard",
  "submit_scorecard_feedback",
  "synthesize_scorecard",
]);

export function hrTools(token: string): AgentTool[] {
  return makeTools(token);
}

const TOOL_LABELS = new Map(makeTools("").map((t) => [t.name, t.label]));

export function toolLabel(name: string): string {
  return TOOL_LABELS.get(name) ?? name;
}
