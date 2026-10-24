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

function toolResult(data: unknown) {
  const json = JSON.stringify(data, null, 0);
  return {
    content: [
      {
        type: "text" as const,
        text: json.length > BODY_LIMIT ? json.slice(0, BODY_LIMIT) + "…(截断)" : json,
      },
    ],
    details: {},
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
    // ===== T3:通用实体元工具(MintHCM AI Tools search_records 思路)=====
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
  "create_report_subscription",
  "delete_watcher",
  "delete_report_subscription",
  "publish_announcement",
  "update_case_status",
]);

export function hrTools(token: string): AgentTool[] {
  return makeTools(token);
}

const TOOL_LABELS = new Map(makeTools("").map((t) => [t.name, t.label]));

export function toolLabel(name: string): string {
  return TOOL_LABELS.get(name) ?? name;
}
