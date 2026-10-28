/** HR 业务工具 — 每个工具的 execute 就是对 HROnfire 公共 REST API 的一次调用。
 *
 * 鉴权:backend 在 /chat 请求里签发与当前用户等价的 JWT(tool_token),
 * sidecar 原样回带 — 租户隔离 / 组织可见域 / license feature / 权限位
 * 全部走既有中间件,sidecar 不做任何权限决策。写工具另外过一层
 * "用户在 UI 上确认"的门(见 agent.ts 的 beforeToolCall)。
 */
import { Type } from "@earendil-works/pi-ai";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { AsyncLocalStorage } from "node:async_hooks";

const BACKEND = process.env.BACKEND_URL || "http://backend:8000";
const BODY_LIMIT = 8000;
// 单文档读取(get_playbook 之类)的专用上限 —— 全文就是工具载荷,
// 通用 8000 会截掉 SOP 后半段且无法续读;30k 覆盖 article.content 的
// 实际体量(字段上限 50k,但真实 SOP 远小于此)同时保护模型上下文。
const DOC_LIMIT = 30000;

// T10:工具写调用的幂等键 = 该次 tool call 的 id(pi-agent 侧天然唯一)。
// execute 经 hrTools 包装进入 callCtx;LLM 重试 / 确认门重放 / MCP 重连
// 重发同一 call_id 的写调用时,backend 走幂等回放而非二次落库。
const callCtx = new AsyncLocalStorage<{ callId?: string }>();
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

async function callApi(
  token: string,
  path: string,
  opts: { method?: string; body?: unknown } = {},
): Promise<unknown> {
  const method = (opts.method ?? "GET").toUpperCase();
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "content-type": "application/json",
  };
  if (!SAFE_METHODS.has(method)) {
    const callId = callCtx.getStore()?.callId;
    if (callId) headers["Idempotency-Key"] = `tool:${callId}`;
  }
  const resp = await fetch(`${BACKEND}${path}`, {
    method,
    headers,
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

/** plan 明细常超 BODY_LIMIT:截断会让尾部字段(月/缺口/工作日)丢失、卡片 JSON 解析失败。压成前 15 条+总数。 */
function compactPlan(data: unknown): unknown {
  if (typeof data !== "object" || data === null || !("plan" in data)) return data;
  const d = data as Record<string, unknown>;
  const plan = Array.isArray(d.plan) ? d.plan : [];
  return { ...d, plan: plan.slice(0, 15), plan_total: plan.length };
}

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
      name: "query_metrics",
      label: "对话式 BI",
      description:
        "招聘经营指标问答:简历/候选人/岗位/面试 KPI、来源与漏斗分布、推荐率、岗位填充。问\"本月入职情况\"\"招聘漏斗转化\"\"哪个岗位人最多\"时用。",
      parameters: Type.Object({
        range: Type.Optional(
          Type.String({ description: "时间窗:7d / 30d / 90d / all,默认 30d" }),
        ),
      }),
      execute: async (_id, p: any) => {
        const range = ["7d", "30d", "90d", "all"].includes(p.range) ? p.range : "30d";
        return t(`/api/reports/overview?range=${range}`);
      },
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
        "在公司知识库(制度/FAQ/福利/IT 指引)里做混合检索;" +
        "回答政策类问题时先查这里,并引用命中自带的溯源字段" +
        "(title/section/paragraph — 形如「来源:《报销制度》§发票要求·第2段」)",
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
    // ===== P1:招聘 SOP 任务链 =====
    {
      name: "start_recruit_run",
      label: "发起招聘任务链",
      description:
        "当用户要求「帮我招一个 XX / 走完整个招聘流程」等编排型任务时先调用:" +
        "创建可追踪的 SOP 任务链,返回 run_id 和步骤 key 清单(jd/parse/source/score/shortlist/interview/report)。" +
        "之后每开始一步调 update_recruit_step 标 running,完成标 done(附 detail 摘要)——让用户在面板上实时看到推进。",
      parameters: Type.Object({
        title: Type.String({ description: "任务链标题,如「前端工程师招聘」" }),
        job_id: Type.Optional(
          Type.String({ description: "已有岗位 id;没有则先建岗位再补" }),
        ),
        kind: Type.Optional(
          Type.String({ description: "默认 full_pipeline" }),
        ),
      }),
      execute: async (_id, p: any) =>
        t("/api/recruit-runs/", {
          method: "POST",
          body: {
            title: p.title,
            job_id: p.job_id ?? null,
            kind: p.kind ?? "full_pipeline",
          },
        }),
    },
    {
      name: "update_recruit_step",
      label: "更新任务链步骤",
      description:
        "推进任务链:step_key 用 start_recruit_run 返回的 key;" +
        "status=running 开始 / done 完成(detail 附摘要如 {top_candidates:[...]}) / failed / skipped;" +
        "link 可附前端页面路径便于用户跳转。",
      parameters: Type.Object({
        run_id: Type.String({ description: "任务链 id" }),
        step_key: Type.String({ description: "步骤 key" }),
        status: Type.String({
          description: "running/done/failed/skipped/waiting_confirm",
        }),
        detail: Type.Optional(
          Type.Record(Type.String(), Type.Unknown(), {
            description: "步骤产出摘要(JSON 对象)",
          }),
        ),
        link: Type.Optional(Type.String({ description: "前端页面路径" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/recruit-runs/${encodeURIComponent(p.run_id)}/steps/${encodeURIComponent(p.step_key)}`,
          {
            method: "PATCH",
            body: {
              status: p.status,
              detail: p.detail ?? null,
              link: p.link ?? null,
            },
          },
        ),
    },
    {
      name: "get_recruit_run",
      label: "查任务链进度",
      description: "查看一条招聘任务链所有步骤的当前状态",
      parameters: Type.Object({
        run_id: Type.String({ description: "任务链 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/recruit-runs/${encodeURIComponent(p.run_id)}`),
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
    // ===== T16:对话式招聘(人才画像 + 人才库扫描 + 面试纪要) =====
    {
      name: "list_talent_profiles",
      label: "查人才画像",
      description:
        "查人才画像(「招什么样的人」可复用模板,按名称模糊搜)," +
        "返回 id/spec(技能/关键词/年限/学历)/上次使用时间",
      parameters: Type.Object({
        q: Type.Optional(Type.String({ description: "名称关键词" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/talent-profiles?q=${encodeURIComponent(p.q ?? "")}`),
    },
    {
      name: "create_talent_profile",
      label: "建人才画像",
      description:
        "创建人才画像(可复用招聘模板)。spec 结构:{skills[], keywords[]," +
        " nice_to_have[], min_years, education}。用户给出描述时先组装好 spec 再调",
      parameters: Type.Object({
        name: Type.String({ description: "画像名,如「AI 工程师画像」" }),
        spec: Type.Optional(
          Type.Object(
            {
              skills: Type.Optional(Type.Array(Type.String())),
              keywords: Type.Optional(Type.Array(Type.String())),
              nice_to_have: Type.Optional(Type.Array(Type.String())),
              min_years: Type.Optional(Type.Number()),
              education: Type.Optional(Type.String()),
            },
            { description: "画像规格" },
          ),
        ),
        source_job_id: Type.Optional(Type.String({ description: "来源岗位 id" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/talent-profiles", {
          method: "POST",
          body: {
            name: p.name,
            spec: p.spec ?? {},
            source_job_id: p.source_job_id,
          },
        }),
    },
    {
      name: "scan_talent_pool",
      label: "扫描人才库",
      description:
        "按人才画像/技能扫描内部人才库,返回排序短名单(每人含匹配分、" +
        "核心匹配点 highlights、缺口 gaps、技能覆盖 coverage)",
      parameters: Type.Object({
        profile_id: Type.Optional(Type.String({ description: "人才画像 id" })),
        spec: Type.Optional(
          Type.Object(
            {
              skills: Type.Optional(Type.Array(Type.String())),
              keywords: Type.Optional(Type.Array(Type.String())),
              min_years: Type.Optional(Type.Number()),
              education: Type.Optional(Type.String()),
            },
            { description: "不用画像时的临时 spec" },
          ),
        ),
        limit: Type.Optional(Type.Number({ description: "返回人数,默认 5" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/sourcing/scan", {
          method: "POST",
          body: {
            profile_id: p.profile_id,
            spec: p.spec,
            limit: p.limit ?? 5,
          },
        }),
    },
    {
      name: "get_interview_summary",
      label: "查面试纪要",
      description:
        "查某场面试的纪要聚合:AI 结论/维度分/逐题问答摘要/评分卡合成建议",
      parameters: Type.Object({
        interview_id: Type.String({ description: "面试 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/interviews/${encodeURIComponent(p.interview_id)}/summary-view`),
    },
    // ===== H1:人事自助(假期余额 + 考勤异常核对 + 人事异动报表) =====
    {
      name: "get_leave_balance",
      label: "查假期余额",
      description:
        "查当年假期余额(每类型:额度/已用/在批/剩余)。不传 employee_id 查本人;" +
        "HR 查他人先用人事查询类工具找到员工 id 再传",
      parameters: Type.Object({
        employee_id: Type.Optional(
          Type.String({ description: "员工 id,缺省查本人" }),
        ),
      }),
      execute: async (_id, p: any) =>
        p.employee_id
          ? t(`/api/leave/balances?employee_id=${encodeURIComponent(p.employee_id)}`)
          : t(`/api/me/leave-balances`),
    },
    {
      name: "list_attendance_flags",
      label: "查考勤异常",
      description:
        "查考勤异常(迟到/早退)列表。day=YYYY-MM-DD 查某天,month=YYYY-MM 查整月;" +
        "excused=false 只看未核对的。返回异常 id/员工/日期/类型/分钟数",
      parameters: Type.Object({
        day: Type.Optional(Type.String({ description: "YYYY-MM-DD" })),
        month: Type.Optional(Type.String({ description: "YYYY-MM" })),
        excused: Type.Optional(
          Type.Boolean({ description: "false=只看未核对,缺省全部" }),
        ),
        scope: Type.Optional(
          Type.String({ description: "all/team/mine,默认 all" }),
        ),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: "50" });
        if (p.day) qs.set("day", String(p.day));
        if (p.month) qs.set("month", String(p.month));
        if (p.excused !== undefined) qs.set("excused", String(p.excused));
        if (p.scope) qs.set("scope", String(p.scope));
        return t(`/api/attendance/flags?${qs}`);
      },
    },
    {
      name: "excuse_attendance_flags",
      label: "批量免计考勤异常",
      description:
        "批量把考勤异常标记为已核对(免计)。入参为 list_attendance_flags 返回的" +
        " flag id 数组;已免计的自动跳过,无权决定的计入 denied",
      parameters: Type.Object({
        flag_ids: Type.Array(Type.String(), { description: "异常 id 数组" }),
        note: Type.Optional(Type.String({ description: "免计备注" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/attendance/flags/bulk-excuse", {
          method: "POST",
          body: { flag_ids: p.flag_ids ?? [], note: p.note },
        }),
    },
    // ===== H2:组织调整(对话即方案:preview 规范化 → 确认 → apply) =====
    {
      name: "preview_org_changes",
      label: "预览组织调整",
      description:
        "把组织调整方案规范化+校验(不落库)。changes 数组,每条 type∈" +
        "{create_org_unit{name,parent?,kind?}, rename_org_unit{org,new_name}," +
        " move_org_unit{org,new_parent?}, transfer_employee{employee,to_org," +
        "new_manager?}, set_manager{employee,manager?}};org/employee 用名字或 id",
      parameters: Type.Object({
        changes: Type.Array(Type.Record(Type.String(), Type.Unknown()), {
          description: "变更条目数组",
        }),
      }),
      execute: async (_id, p: any) =>
        t("/api/org/preview-changes", {
          method: "POST",
          body: { changes: p.changes ?? [] },
        }),
    },
    {
      name: "apply_org_changes",
      label: "执行组织调整",
      description:
        "事务执行组织调整(需用户确认;changes 格式同 preview_org_changes。" +
        "必须先 preview 且 ok=true 再调,任一条非法整批拒绝)",
      parameters: Type.Object({
        changes: Type.Array(Type.Record(Type.String(), Type.Unknown()), {
          description: "与 preview 相同的变更数组",
        }),
      }),
      execute: async (_id, p: any) =>
        t("/api/org/apply-changes", {
          method: "POST",
          body: { changes: p.changes ?? [] },
        }),
    },
    // ===== H3:智能排班 =====
    {
      name: "list_shifts",
      label: "查班次定义",
      description: "查班次列表(名称/时间段/每班需人数),排班前先查有哪些班",
      parameters: Type.Object({}),
      execute: async () => t("/api/shifts"),
    },
    {
      name: "preview_shift_plan",
      label: "预览排班计划",
      description:
        "生成某月排班计划但**不落库**,用于先给用户过目。返回 plan 明细" +
        "(day/shift/employee)+ understaffed 排不满的槽位",
      parameters: Type.Object({
        month: Type.Optional(Type.String({ description: "YYYY-MM,默认当月" })),
        org_unit_id: Type.Optional(Type.String({ description: "只排某部门" })),
      }),
      execute: async (_id, p: any) =>
        callApi(token, "/api/shifts/generate", {
          method: "POST",
          body: { month: p.month, org_unit_id: p.org_unit_id, dry_run: true },
        }).then((d) => toolResult(compactPlan(d))),
    },
    {
      name: "generate_shifts",
      label: "生成排班",
      description:
        "正式生成某月排班并落库(需用户确认)。只填空缺:已有排班不动," +
        "已批请假的员工自动跳过。返回生成条数+排不满的槽位",
      parameters: Type.Object({
        month: Type.Optional(Type.String({ description: "YYYY-MM,默认当月" })),
        org_unit_id: Type.Optional(Type.String({ description: "只排某部门" })),
      }),
      execute: async (_id, p: any) =>
        callApi(token, "/api/shifts/generate", {
          method: "POST",
          body: { month: p.month, org_unit_id: p.org_unit_id, dry_run: false },
        }).then((d) => toolResult(compactPlan(d))),
    },
    {
      name: "list_shift_gaps",
      label: "查排班缺口",
      description:
        "查某月排班缺口:覆盖不足的槽位 + 「排了班但人在请假」的待替班项",
      parameters: Type.Object({
        month: Type.Optional(Type.String({ description: "YYYY-MM,默认当月" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/shifts/gaps?month=${encodeURIComponent(p.month ?? "")}`),
    },
    {
      name: "auto_cover_shifts",
      label: "自动找替班",
      description:
        "自动替班补缺(需用户确认):请假员工的班换给候选同事," +
        "覆盖不足的槽位自动补人。返回 covered/filled/remaining 明细",
      parameters: Type.Object({
        month: Type.Optional(Type.String({ description: "YYYY-MM,默认当月" })),
        dry_run: Type.Optional(
          Type.Boolean({ description: "true 只预览不落库,默认 false" }),
        ),
      }),
      execute: async (_id, p: any) =>
        t("/api/shifts/auto-cover", {
          method: "POST",
          body: { month: p.month, dry_run: p.dry_run ?? false },
        }),
    },
    // ===== AI 排班助手(时间智能体):解释排班 / 方案对比 / 微调 =====
    {
      name: "list_shift_employees",
      label: "查可排员工",
      description:
        "列出可排班员工(id/姓名/工号/部门)。用户给的是人名时,先用这个" +
        "或 global_search 把名字解析成 employee_id",
      parameters: Type.Object({}),
      execute: async () => t("/api/shifts/employees"),
    },
    {
      name: "get_schedule_board",
      label: "查当日排班看板",
      description:
        "某日各班次覆盖看板:每个班次谁上班/谁排了但在请假/缺几人。" +
        "用户问「今天谁上班/周三晚班几个人/明天排班怎么样」用这个",
      parameters: Type.Object({
        day: Type.String({ description: "YYYY-MM-DD" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/shifts/board?day=${encodeURIComponent(p.day)}`),
    },
    {
      name: "explain_schedule",
      label: "解释排班",
      description:
        "解释某员工某天的排班全貌:当天排了什么(来源)、没排的原因" +
        "(请假/登记不可排/空闲)、本月已排班数、连续出勤天数、进行中的换班。" +
        "带 shift_id 时额外返回「假设给此人排这个班」的合规预判 violations。" +
        "用户问「为什么把X排在周一/X那天能不能排晚班」用这个",
      parameters: Type.Object({
        employee_id: Type.String({ description: "员工 id" }),
        day: Type.String({ description: "YYYY-MM-DD" }),
        shift_id: Type.Optional(
          Type.String({ description: "假设要排的班次 id(可选)" })),
      }),
      execute: async (_id, p: any) => {
        let u = `/api/shifts/explain?employee_id=${encodeURIComponent(p.employee_id)}` +
          `&day=${encodeURIComponent(p.day)}`;
        if (p.shift_id) u += `&shift_id=${encodeURIComponent(p.shift_id)}`;
        return t(u);
      },
    },
    {
      name: "compare_shift_plans",
      label: "对比排班方案",
      description:
        "某槽位(日×班次)候选方案对比:对每个可排候选人给出合规预判、" +
        "当月已排班数、可用性偏好、同部门标记和确定性优劣项(pros/cons)," +
        "另列当天不可排的人及原因。用户要「换个有空的/找几个人选对比/" +
        "谁能顶这个班」时先调它再讲解优劣",
      parameters: Type.Object({
        day: Type.String({ description: "YYYY-MM-DD" }),
        shift_id: Type.String({ description: "班次 id" }),
        limit: Type.Optional(
          Type.Number({ description: "返回方案数,默认 5" })),
      }),
      execute: async (_id, p: any) =>
        t(
          `/api/shifts/plan-compare?day=${encodeURIComponent(p.day)}` +
          `&shift_id=${encodeURIComponent(p.shift_id)}` +
          `&limit=${Number(p.limit ?? 5) || 5}`,
        ),
    },
    {
      name: "check_shift_compliance",
      label: "排班合规体检",
      description:
        "全员排班合规体检:返回当月每条违规(员工/日期/规则/明细)。" +
        "规则含连班上限/周工时上限/班间休息间隔/员工可用性。问「排班有没有" +
        "违规/谁连班太多」用这个",
      parameters: Type.Object({
        month: Type.Optional(
          Type.String({ description: "YYYY-MM,默认当月" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/shifts/compliance?month=${encodeURIComponent(p.month ?? "")}`),
    },
    {
      name: "get_shift_rules",
      label: "查排班规则",
      description:
        "查当前排班合规规则口径:连班上限/周工时上限/班间最小休息分钟/" +
        "是否尊重员工可用性。解释违规含义前先查口径",
      parameters: Type.Object({}),
      execute: async () => t("/api/shifts/rules"),
    },
    {
      name: "list_shift_swaps",
      label: "查换班申请",
      description:
        "HR 换班审批队列:可按 status 过滤(pending_peer 待对方确认/" +
        "pending_approval 待 HR 审批/approved/rejected/withdrawn)。问" +
        "「有没有待审批的换班」用这个",
      parameters: Type.Object({
        status: Type.Optional(Type.String({ description: "状态过滤" })),
        month: Type.Optional(
          Type.String({ description: "YYYY-MM,按换班日过滤" })),
      }),
      execute: async (_id, p: any) => {
        let u = "/api/shifts/swaps?";
        if (p.status) u += `status=${encodeURIComponent(p.status)}&`;
        if (p.month) u += `month=${encodeURIComponent(p.month)}&`;
        return t(u);
      },
    },
    {
      name: "assign_shift",
      label: "给员工排班",
      description:
        "给某员工某天排某班次(需用户确认)。若返回 409 且 detail 含 " +
        "violations,先把违规明细解释给用户;只有用户明确说「仍然排/放行」" +
        "才带 force=true 重新调用",
      parameters: Type.Object({
        employee_id: Type.String({ description: "员工 id" }),
        shift_id: Type.String({ description: "班次 id" }),
        day: Type.String({ description: "YYYY-MM-DD" }),
        note: Type.Optional(Type.String({ description: "备注" })),
        force: Type.Optional(Type.Boolean({
          description: "true = 知晓违规仍强制排(需用户明确放行)" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/shifts/assignments", {
          method: "POST",
          body: {
            employee_id: p.employee_id, shift_id: p.shift_id, day: p.day,
            note: p.note ?? null, force: p.force ?? false,
          },
        }),
    },
    {
      name: "reassign_shift",
      label: "调整已排班次",
      description:
        "把一条已排班改给另一个员工(需用户确认),常用于「把张三四月三号的" +
        "晚班换成李四」。先用 compare_shift_plans 给候选人,用户选定后调用。" +
        "409 违规时同 assign_shift 的 force 规则",
      parameters: Type.Object({
        assignment_id: Type.String({ description: "排班记录 id" }),
        employee_id: Type.String({ description: "新替班员工 id" }),
        note: Type.Optional(Type.String({ description: "备注" })),
        force: Type.Optional(Type.Boolean({
          description: "true = 知晓违规仍强制(需用户明确放行)" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/shifts/assignments/${encodeURIComponent(p.assignment_id)}/reassign`, {
          method: "POST",
          body: {
            employee_id: p.employee_id, note: p.note ?? null,
            force: p.force ?? false,
          },
        }),
    },
    {
      name: "decide_shift_swap",
      label: "审批换班申请",
      description:
        "批准或驳回一条待审批换班申请(需用户确认)。批准后系统自动交换" +
        "双方班次并通知当事人;返回交换后的合规告警(不阻断)",
      parameters: Type.Object({
        swap_id: Type.String({ description: "换班申请 id" }),
        approve: Type.Boolean({ description: "true 批准 / false 驳回" }),
        note: Type.Optional(Type.String({ description: "审批意见" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/shifts/swaps/${encodeURIComponent(p.swap_id)}/decide`, {
          method: "POST",
          body: { approve: p.approve, note: p.note ?? null },
        }),
    },
    // ===== E1:员工自助(全部走 /me 端点,行级=本人绑定的员工档案) =====
    {
      name: "get_my_profile",
      label: "查我的档案",
      description:
        "查本人员工档案:工号/部门/职位/上级/入职日期/用工类型。员工问" +
        "「我的档案/我哪个部门/我上级是谁」用这个",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/profile"),
    },
    {
      name: "get_my_attendance",
      label: "查我的考勤",
      description:
        "查本人某月考勤明细(每天状态/首末打卡/工时)。month=YYYY-MM 默认当月",
      parameters: Type.Object({
        month: Type.Optional(Type.String({ description: "YYYY-MM" })),
      }),
      execute: async (_id, p: any) =>
        t(`/api/me/attendance?month=${encodeURIComponent(p.month ?? "")}`),
    },
    {
      name: "my_leave_applications",
      label: "查我的请假单",
      description: "查本人请假申请列表(类型/日期/天数/状态/审批意见)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/leave-applications"),
    },
    {
      name: "list_leave_types",
      label: "查假期类型",
      description:
        "查可申请假期类型列表(id/名称/是否带薪/是否调休)。提交请假/加班" +
        "申请需要 leave_type_id 时先查这个",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/leave-types"),
    },
    {
      name: "create_leave_application",
      label: "提交请假申请",
      description:
        "代本人提交请假申请(需用户确认)。leave_type_id 用 list_leave_types 查;" +
        "start_date/end_date=YYYY-MM-DD;days 不传由系统按工作日历算;reason 可选",
      parameters: Type.Object({
        leave_type_id: Type.String({ description: "假期类型 id" }),
        start_date: Type.String({ description: "YYYY-MM-DD" }),
        end_date: Type.String({ description: "YYYY-MM-DD" }),
        days: Type.Optional(
          Type.Number({ description: "天数,缺省=区间工作日数" }),
        ),
        reason: Type.Optional(Type.String({ description: "事由" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/leave-applications", {
          method: "POST",
          body: {
            leave_type_id: p.leave_type_id,
            start_date: p.start_date,
            end_date: p.end_date,
            days: p.days,
            reason: p.reason ?? "",
          },
        }),
    },
    {
      name: "my_overtime_requests",
      label: "查我的加班单",
      description: "查本人加班申请列表(加班日/时长/折算调休/状态)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/overtime-requests"),
    },
    {
      name: "create_overtime_request",
      label: "提交加班申请",
      description:
        "代本人提交加班申请(需用户确认;批准自动折算调休入账)。" +
        "leave_type_id 必须是调休类型(ist_comp),先 list_leave_types 查;" +
        "work_date 不能是未来;hours 以 0.5 为步长",
      parameters: Type.Object({
        leave_type_id: Type.String({ description: "调休类型 id" }),
        work_date: Type.String({ description: "加班日 YYYY-MM-DD" }),
        hours: Type.Number({ description: "时长(小时,0.5 步长)" }),
        reason: Type.Optional(Type.String({ description: "事由" })),
      }),
      execute: async (_id, p: any) =>
        t("/api/me/overtime-requests", {
          method: "POST",
          body: {
            leave_type_id: p.leave_type_id,
            work_date: p.work_date,
            hours: p.hours,
            reason: p.reason ?? "",
          },
        }),
    },
    {
      name: "my_requests",
      label: "查我的申请",
      description: "查本人提交的通用申请(类型/主题/状态/审批意见/留言数)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/requests"),
    },
    {
      name: "my_payslips",
      label: "查我的工资条",
      description: "查本人已发放工资条列表(月份/应发/实发/状态)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/salary-slips"),
    },
    {
      name: "my_goals",
      label: "查我的目标",
      description: "查本人绩效目标列表(标题/进度/状态/考核周期)",
      parameters: Type.Object({}),
      execute: async () => t("/api/me/goals"),
    },
    // ===== E2:入离职编排(模板铺 checklist,完成自动回写档案) =====
    {
      name: "list_lifecycle_templates",
      label: "查入转调离模板",
      description: "查生命周期任务模板(入职/调动/晋升/离职,各含任务清单)",
      parameters: Type.Object({}),
      execute: async () => t("/api/lifecycle/templates"),
    },
    {
      name: "list_lifecycle_events",
      label: "查入转调离事件",
      description:
        "查生命周期事件列表(进行中/已完成)。kind=onboarding|transfer|" +
        "promotion|separation,status=in_progress|completed|cancelled",
      parameters: Type.Object({
        kind: Type.Optional(Type.String()),
        status: Type.Optional(Type.String()),
        employee_id: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) => {
        const qs = new URLSearchParams({ limit: "50" });
        if (p.kind) qs.set("kind", p.kind);
        if (p.status) qs.set("status", p.status);
        if (p.employee_id) qs.set("employee_id", p.employee_id);
        return t(`/api/lifecycle/events?${qs}`);
      },
    },
    {
      name: "run_lifecycle_event",
      label: "发起入转调离",
      description:
        "为员工发起生命周期事件(需用户确认),按模板铺 checklist。" +
        "kind=onboarding 入职 | transfer 调动(payload.to_org_unit_id)|" +
        " promotion 晋升(payload.to_designation)| separation 离职;" +
        "任务全勾后自动回写员工档案。employee_id 先用人事查询工具找",
      parameters: Type.Object({
        employee_id: Type.String({ description: "员工 id" }),
        kind: Type.String({
          description: "onboarding|transfer|promotion|separation",
        }),
        effective_date: Type.Optional(
          Type.String({ description: "生效日 YYYY-MM-DD" }),
        ),
        template_id: Type.Optional(
          Type.String({ description: "模板 id,缺省用默认模板" }),
        ),
        payload: Type.Optional(
          Type.Record(Type.String(), Type.Unknown(), {
            description: "transfer:to_org_unit_id;promotion:to_designation",
          }),
        ),
        note: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) =>
        t("/api/lifecycle/events", {
          method: "POST",
          body: {
            employee_id: p.employee_id,
            kind: p.kind,
            effective_date: p.effective_date,
            template_id: p.template_id,
            payload: p.payload,
            note: p.note,
          },
        }),
    },
    {
      name: "toggle_lifecycle_task",
      label: "勾销入转调离任务",
      description:
        "勾选/取消某条清单任务(需用户确认);全部勾完事件自动完成并" +
        "回写员工档案。task_id 来自 list_lifecycle_events 返回的 tasks",
      parameters: Type.Object({
        task_id: Type.String({ description: "任务 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/lifecycle/tasks/${encodeURIComponent(p.task_id)}/toggle`, {
          method: "POST",
        }),
    },
    // ===== E3:审批中心(统一收件箱 + 多级链裁决 + 代理人) =====
    {
      name: "list_pending_approvals",
      label: "查我的待审批",
      description:
        "查当前用户可审批的待办:多级审批链当前步(step 模式)+" +
        "未配策略的一级审批(legacy 模式)。员工问「我有什么要批的」、" +
        "HR 问「待审批清单」用这个",
      parameters: Type.Object({}),
      execute: async () => t("/api/approvals/inbox"),
    },
    {
      name: "decide_approval",
      label: "审批通过/驳回",
      description:
        "对收件箱里的申请表态(需用户确认)。request_type ∈ leave|" +
        "overtime|attendance_request|expense_claim|employee_request;" +
        "action=approved|rejected;note 可选。多级链自动推进到下一级," +
        "末步批准才生效",
      parameters: Type.Object({
        request_type: Type.String(),
        request_id: Type.String(),
        action: Type.String({ description: "approved|rejected" }),
        note: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) =>
        t("/api/approvals/decide", {
          method: "POST",
          body: {
            request_type: p.request_type,
            request_id: p.request_id,
            action: p.action,
            note: p.note,
          },
        }),
    },
    {
      name: "list_approval_policies",
      label: "查审批策略",
      description:
        "查审批策略配置(request_type/条件/多级步骤)。管理员可配;" +
        "介绍「请假走几级审批」这类规则用这个",
      parameters: Type.Object({}),
      execute: async () => t("/api/approvals/policies"),
    },
    {
      name: "save_approval_policy",
      label: "配置审批策略",
      description:
        "新建/覆盖审批策略(需用户确认;管理员权限)。request_type ∈ " +
        "leave|overtime|attendance_request|expense_claim|employee_request;" +
        "conditions=[{field,op,value}] 字段白名单见 list_approval_policies " +
        "返回的 condition_fields(如 days gt 3);steps=[{resolver,key?,label?}]" +
        " 按顺序逐级审批:manager=申请人直属上级/role=该角色全员(key=角色名)" +
        "/user=指定人(key=user_id);auto_approve=[{field,op,value}] 非空时" +
        "命中策略且满足该条件组的单据提交即自动批准(如 days le 2)",
      parameters: Type.Object({
        name: Type.String(),
        request_type: Type.String(),
        priority: Type.Optional(Type.Number()),
        conditions: Type.Optional(
          Type.Array(Type.Record(Type.String(), Type.Unknown())),
        ),
        auto_approve: Type.Optional(
          Type.Array(Type.Record(Type.String(), Type.Unknown())),
        ),
        steps: Type.Array(Type.Record(Type.String(), Type.Unknown())),
      }),
      execute: async (_id, p: any) =>
        t("/api/approvals/policies", {
          method: "POST",
          body: {
            name: p.name,
            request_type: p.request_type,
            priority: p.priority ?? 0,
            conditions: p.conditions ?? [],
            auto_approve: p.auto_approve ?? null,
            steps: p.steps,
          },
        }),
    },
    {
      name: "list_automation_rules",
      label: "查自动化规则",
      description:
        "查租户的自动化规则列表(触发方式=业务事件/日期到达、实体、条件、" +
        "动作=站内通知/邮件/建工单、启停、上次触发时间)。「配置了哪些自动" +
        "化」「合同到期有没有提醒」用这个",
      parameters: Type.Object({}),
      execute: async () => t("/api/automations?limit=50"),
    },
    {
      name: "list_automation_runs",
      label: "查自动化执行记录",
      description:
        "查某条自动化规则的执行记录(状态 ok/skipped/error、收件人、" +
        "建单产出、执行时间)。rule_id 从 list_automation_rules 取",
      parameters: Type.Object({
        rule_id: Type.String({ description: "规则 id" }),
      }),
      execute: async (_id, p: any) =>
        t(`/api/automations/${p.rule_id}/runs?limit=20`),
    },
    {
      name: "create_automation_rule",
      label: "创建自动化规则",
      description:
        "建自动化规则(需用户确认;manage_settings 权限)。触发方式:" +
        "trigger_kind=audit 业务事件(entity_type 实体 + trigger=create/" +
        "update/delete/approve 等审计动作名,update 可加 watch_fields);" +
        "date_field 员工日期到达(entity_type 固定 employee," +
        "schedule={field:hire_date|birth_date|probation_end|contract_end|" +
        "cert_expire, offset_days:±N 负=提前, recurring?:每年重复})。" +
        "conditions=[{field,op,value}] 白名单见 GET /api/automations/schema。" +
        "action=notify|email 需要 recipients=[{kind:actor|role|user|" +
        "record_owner|record_manager,value?}];action=create_case 建 HR 工单," +
        "action_config={kind:it|admin|payroll|facility|leave_dispute|other," +
        "priority:low|normal|high|urgent, assignee_kind?:role|user," +
        "assignee_value?}。title_tpl/body_tpl 支持 {{字段}} 占位",
      parameters: Type.Object({
        name: Type.String(),
        entity_type: Type.String(),
        trigger_kind: Type.Optional(
          Type.String({ description: "audit|date_field,默认 audit" }),
        ),
        trigger: Type.Optional(
          Type.String({ description: "审计动作名:create/update/delete/approve 等" }),
        ),
        schedule: Type.Optional(
          Type.Record(Type.String(), Type.Unknown()),
        ),
        watch_fields: Type.Optional(Type.Array(Type.String())),
        conditions: Type.Optional(
          Type.Array(Type.Record(Type.String(), Type.Unknown())),
        ),
        action: Type.Optional(
          Type.String({ description: "notify|email|create_case" }),
        ),
        action_config: Type.Optional(
          Type.Record(Type.String(), Type.Unknown()),
        ),
        recipients: Type.Optional(
          Type.Array(Type.Record(Type.String(), Type.Unknown())),
        ),
        title_tpl: Type.String(),
        body_tpl: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) =>
        t("/api/automations", {
          method: "POST",
          body: {
            name: p.name,
            entity_type: p.entity_type,
            trigger_kind: p.trigger_kind ?? "audit",
            trigger: p.trigger,
            schedule: p.schedule,
            watch_fields: p.watch_fields,
            conditions: p.conditions ?? [],
            action: p.action ?? "notify",
            action_config: p.action_config,
            recipients: p.recipients ?? [],
            title_tpl: p.title_tpl,
            body_tpl: p.body_tpl,
          },
        }),
    },
    {
      name: "set_automation_rule_active",
      label: "启停自动化规则",
      description:
        "启用/停用一条自动化规则(需用户确认)。rule_id 从 " +
        "list_automation_rules 取;停用后不再触发但保留执行记录",
      parameters: Type.Object({
        rule_id: Type.String({ description: "规则 id" }),
        is_active: Type.Boolean(),
      }),
      execute: async (_id, p: any) =>
        t(`/api/automations/${p.rule_id}`, {
          method: "PATCH",
          body: { is_active: p.is_active },
        }),
    },
    {
      name: "set_my_delegate",
      label: "设置审批代理人",
      description:
        "把自己的审批权临时委托给另一个用户(需用户确认)。" +
        "delegate_user_id 传 null 取消代理",
      parameters: Type.Object({
        delegate_user_id: Type.Union([Type.String(), Type.Null()]),
      }),
      execute: async (_id, p: any) =>
        t("/api/approvals/delegate", {
          method: "PATCH",
          body: { delegate_user_id: p.delegate_user_id },
        }),
    },
    // ===== E5:白名单数据集查询(NL 报表,确定性执行) =====
    {
      name: "list_datasets",
      label: "查可统计的数据集",
      description:
        "查可自由统计的数据集清单及各自可用维度/指标/过滤字段。" +
        "用户要临时口径的统计表(如「各部门 90 后占比」「各月请假天数」)" +
        "先查这个选字段",
      parameters: Type.Object({}),
      execute: async () => t("/api/datasets"),
    },
    {
      name: "query_dataset",
      label: "自由统计查询",
      description:
        "在白名单数据集上做分组统计。dataset 见 list_datasets;" +
        "group_by=维度名数组;metrics=指标名数组(默认 count);" +
        "filters 支持 month=YYYY-MM / date_from / date_to / 数据集白名单字段" +
        "(eq 或数组 in)。结果行数封顶 200",
      parameters: Type.Object({
        dataset: Type.String(),
        group_by: Type.Optional(Type.Array(Type.String())),
        metrics: Type.Optional(Type.Array(Type.String())),
        filters: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
      }),
      execute: async (_id, p: any) =>
        t("/api/datasets/query", {
          method: "POST",
          body: {
            dataset: p.dataset,
            group_by: p.group_by ?? [],
            metrics: p.metrics ?? ["count"],
            filters: p.filters ?? {},
          },
        }),
    },
    // ===== P 批次:人才画像 + 内部人才匹配 =====
    {
      name: "get_talent_card",
      label: "看员工人才画像",
      description:
        "取员工 360 画像卡:基本信息+确认技能(带等级/来源)+待确认建议+" +
        "面试评价+项目经历+绩效+洞察时间线。employee_id 用 search_entities " +
        "先查到;本人问自己的画像也可以",
      parameters: Type.Object({
        employee_id: Type.String(),
      }),
      execute: async (_id, p: any) =>
        t(`/api/employees/${encodeURIComponent(p.employee_id)}/talent-card`),
    },
    {
      name: "search_talent",
      label: "按技能搜内部人才",
      description:
        "「新项目需要 XX 能力强的人」用这里确定性预筛:skills=[{name," +
        "min_level?}] name 是技能名片段(把用户需求拆成片段,如「AI coding " +
        "强」→ [{name:'AI'},{name:'Python',min_level:3}]),对每个片段在" +
        "员工已确认技能上匹配,按命中片段数+等级排序。include_suggested" +
        "=true 时连待确认建议也算。返回带技能证据与等级,你再综合选出" +
        "topK 并说明推荐理由",
      parameters: Type.Object({
        skills: Type.Array(Type.Record(Type.String(), Type.Unknown())),
        include_suggested: Type.Optional(Type.Boolean()),
        org_unit_id: Type.Optional(Type.String()),
        limit: Type.Optional(Type.Number()),
      }),
      execute: async (_id, p: any) =>
        t("/api/talent/search", {
          method: "POST",
          body: {
            skills: p.skills,
            include_suggested: p.include_suggested ?? false,
            org_unit_id: p.org_unit_id,
            limit: p.limit ?? 20,
          },
        }),
    },
    {
      name: "list_projects",
      label: "查项目列表",
      description:
        "查公司项目(名字/状态/所需技能/成员数)。要把人加进项目、" +
        "或问「XX 项目都有哪些人」先用它拿 project_id",
      parameters: Type.Object({
        status: Type.Optional(
          Type.String({ description: "draft|active|done|cancelled" }),
        ),
      }),
      execute: async (_id, p: any) =>
        t(`/api/projects${p.status ? `?status=${p.status}` : ""}`),
    },
    {
      name: "create_project",
      label: "新建项目",
      description:
        "登记新项目(需用户确认)。required_skills=[{name,min_level?}]" +
        " 供 AI 选人匹配;建完可接着用 search_talent 找人",
      parameters: Type.Object({
        name: Type.String(),
        description: Type.Optional(Type.String()),
        required_skills: Type.Optional(
          Type.Array(Type.Record(Type.String(), Type.Unknown())),
        ),
      }),
      execute: async (_id, p: any) =>
        t("/api/projects", {
          method: "POST",
          body: {
            name: p.name,
            description: p.description,
            status: "draft",
            required_skills: p.required_skills ?? [],
          },
        }),
    },
    {
      name: "add_project_member",
      label: "把员工加入项目",
      description:
        "把员工加进项目(需用户确认)。AI 推荐后用户拍板「就用 TA」时" +
        "调用,joined_via=ai_match 留痕;手动指定可传 false。role 可选" +
        "(如 TL/后端),allocation_pct 投入度 1-100",
      parameters: Type.Object({
        project_id: Type.String(),
        employee_id: Type.String(),
        role: Type.Optional(Type.String()),
        allocation_pct: Type.Optional(Type.Number()),
        ai_match: Type.Optional(Type.Boolean()),
      }),
      execute: async (_id, p: any) =>
        t(`/api/projects/${encodeURIComponent(p.project_id)}/members`, {
          method: "POST",
          body: {
            employee_id: p.employee_id,
            role: p.role,
            allocation_pct: p.allocation_pct ?? 100,
            joined_via: p.ai_match === false ? "manual" : "ai_match",
          },
        }),
    },
    {
      name: "add_insight_note",
      label: "给员工记一条洞察",
      description:
        "给员工画像加一条证据(需用户确认):会议纪要文本/项目复盘/" +
        "主管随手评。kind=meeting|project_review|manager_note|milestone;" +
        "body 是正文(纪要全文可直接粘贴,上限 8000 字);保存后自动" +
        "提炼技能建议进待确认队列。visibility 默认 manager_only(本人不可见)",
      parameters: Type.Object({
        employee_id: Type.String(),
        kind: Type.String(),
        body: Type.String(),
        title: Type.Optional(Type.String()),
        visibility: Type.Optional(Type.String()),
      }),
      execute: async (_id, p: any) =>
        t(`/api/employees/${encodeURIComponent(p.employee_id)}/insights`, {
          method: "POST",
          body: {
            kind: p.kind,
            body: p.body,
            title: p.title,
            visibility: p.visibility ?? "manager_only",
            extract_skills: true,
          },
        }),
    },
    // ===== P0:文档引擎 =====
    {
      name: "generate_document",
      label: "生成文档",
      description:
        "生成可下载的原生文档(PPT/Word/Excel),返回下载链接。" +
        "kind 取值:recruiting_pack=招聘进度汇报PPT(可指定 job_id 聚焦单岗位)、" +
        "candidate_report=候选人才分析报告Word(需 candidate_id)、" +
        "exam_paper=题集转笔试卷Word(需 question_set_id,with_answers 控制是否附答案)、" +
        "payroll_xlsx=薪酬底表Excel(可指定 period 如 2026-10)、" +
        "candidates_xlsx=候选人清单Excel(可指定 job_id)、" +
        "pipeline_xlsx=招聘管道明细Excel(可指定 job_id)。",
      parameters: Type.Object({
        kind: Type.String({ description: "文档类型(见上)" }),
        job_id: Type.Optional(Type.String({ description: "岗位 id" })),
        candidate_id: Type.Optional(
          Type.String({ description: "候选人 id" }),
        ),
        question_set_id: Type.Optional(
          Type.String({ description: "题集 id" }),
        ),
        period: Type.Optional(
          Type.String({ description: "薪酬月份 YYYY-MM" }),
        ),
        with_answers: Type.Optional(
          Type.Boolean({ description: "笔试卷是否附答案,默认 true" }),
        ),
      }),
      execute: async (_id, p: any) => {
        const kind = String(p.kind);
        let path = "";
        let body: Record<string, unknown> = {};
        switch (kind) {
          case "recruiting_pack":
            path = "/api/exports/recruiting-pack";
            body = { job_id: p.job_id ?? null };
            break;
          case "candidate_report":
            if (!p.candidate_id)
              return toolResult({ error: "candidate_report 需要 candidate_id" });
            path = `/api/exports/candidate-report/${encodeURIComponent(p.candidate_id)}`;
            break;
          case "exam_paper":
            if (!p.question_set_id)
              return toolResult({ error: "exam_paper 需要 question_set_id" });
            path = `/api/exports/exam-paper/${encodeURIComponent(p.question_set_id)}`;
            body = { with_answers: p.with_answers ?? true };
            break;
          case "payroll_xlsx":
            path = "/api/exports/payroll.xlsx";
            body = { period: p.period ?? null };
            break;
          case "candidates_xlsx":
            path = "/api/exports/candidates.xlsx";
            body = { job_id: p.job_id ?? null };
            break;
          case "pipeline_xlsx":
            path = "/api/exports/pipeline.xlsx";
            body = { job_id: p.job_id ?? null };
            break;
          default:
            return toolResult({
              error: `未知 kind:${kind};可选 recruiting_pack/candidate_report/exam_paper/payroll_xlsx/candidates_xlsx/pipeline_xlsx`,
            });
        }
        const doc = (await callApi(token, path, {
          method: "POST",
          body,
        })) as {
          id?: string;
          title?: string;
          file_name?: string;
          error?: string;
        };
        if (!doc || doc.error || !doc.id)
          return toolResult(doc ?? { error: "生成失败" });
        return toolResult({
          title: doc.title,
          file_name: doc.file_name,
          download_url: `/api/exports/${doc.id}/download`,
          hint: "把 download_url 作为可点击链接给用户(前端会自动带鉴权下载)",
        });
      },
    },
  ];
}

/** 需要用户在界面上点"确认"才会真正执行的写操作 */
export const WRITE_TOOLS = new Set([
  "generate_document",
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
  "create_talent_profile",
  "excuse_attendance_flags",
  "apply_org_changes",
  "generate_shifts",
  "auto_cover_shifts",
  "assign_shift",
  "reassign_shift",
  "decide_shift_swap",
  "create_leave_application",
  "create_overtime_request",
  "run_lifecycle_event",
  "toggle_lifecycle_task",
  "decide_approval",
  "save_approval_policy",
  "create_automation_rule",
  "set_automation_rule_active",
  "set_my_delegate",
  "create_project",
  "add_project_member",
  "add_insight_note",
]);

export function hrTools(token: string): AgentTool[] {
  // 包装 execute:把 tool_call_id 注入 AsyncLocalStorage,callApi 取作幂等键。
  return makeTools(token).map((tool) => ({
    ...tool,
    execute: (id: string, params: unknown) =>
      callCtx.run({ callId: id }, () => tool.execute(id, params)),
  })) as AgentTool[];
}

const TOOL_LABELS = new Map(makeTools("").map((t) => [t.name, t.label]));

export function toolLabel(name: string): string {
  return TOOL_LABELS.get(name) ?? name;
}
