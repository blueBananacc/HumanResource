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
        t(`/api/talents?q=${encodeURIComponent(p.q ?? "")}&limit=${p.limit ?? 10}`),
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
          `/api/matches/job/${p.job_id}?min_score=${p.min_score ?? 0}&limit=${p.limit ?? 20}`,
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
          `/api/interviews/?q=${encodeURIComponent(p.q ?? "")}&limit=${p.limit ?? 10}`,
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
      execute: async (_id, p: any) => t(`/api/interviews/${p.interview_id}/report`),
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
        t(`/api/matches/${p.match_id}/dismiss-suggestion`, { method: "POST" }),
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
  ];
}

/** 需要用户在界面上点"确认"才会真正执行的写操作 */
export const WRITE_TOOLS = new Set([
  "create_interview",
  "generate_questions",
  "dismiss_interview_suggestion",
]);

export function hrTools(token: string): AgentTool[] {
  return makeTools(token);
}

const TOOL_LABELS = new Map(makeTools("").map((t) => [t.name, t.label]));

export function toolLabel(name: string): string {
  return TOOL_LABELS.get(name) ?? name;
}
