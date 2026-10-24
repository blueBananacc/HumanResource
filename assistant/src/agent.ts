/** Agent 会话池 — 每个会话一个 pi-agent-core Agent 实例。
 *
 * - 会话级 Map 常驻内存(重启从 store 的 JSON 恢复 messages 重建)
 * - 写工具经 beforeToolCall 停放,等前端 POST /confirm 裁决
 * - subscribe 事件统一翻译成 SSE 帧写给当前 res
 */
import { randomUUID } from "node:crypto";
import { Agent } from "@earendil-works/pi-agent-core";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { MutableModels } from "@earendil-works/pi-ai";
import { buildModel, type LlmConfig } from "./llm.js";
import { hrTools, toolLabel, WRITE_TOOLS } from "./tools.js";
import {
  loadConversation,
  saveConversation,
  type StoredConversation,
} from "./store.js";

export type SseEmit = (event: string, data: unknown) => void;

type UserCtx = { tenant_id: string; id: string; role: string; name?: string };
/** 会话归属者(裁决确认/杀会话时校验,跨租户跨用户一律按不存在处理) */
export type SessionOwner = { tenantId: string; userId: string };

export type Session = {
  id: string;
  tenantId: string;
  userId: string;
  userName: string;
  agent: Agent;
  models: MutableModels;
  toolToken: string;
  llm: LlmConfig;
  busy: boolean;
  createdAt: string;
  /** 当前 run 的 SSE 写出函数;每次 /chat 时刷新 */
  emit: SseEmit;
};

const sessions = new Map<string, Session>();
const MAX_SESSIONS = 50;

type PendingConfirm = SessionOwner & {
  resolve: (approved: boolean) => void;
};

/** callId → 停放中的确认。裁决必须带会话归属,防止跨用户放行/拒绝写操作。 */
const pendingConfirms = new Map<string, PendingConfirm>();
const CONFIRM_TIMEOUT_MS = 120_000;

export function resolveConfirm(
  callId: string,
  approved: boolean,
  owner: SessionOwner,
): boolean {
  const pending = pendingConfirms.get(callId);
  if (
    !pending ||
    pending.tenantId !== owner.tenantId ||
    pending.userId !== owner.userId
  ) {
    return false;
  }
  pendingConfirms.delete(callId);
  pending.resolve(approved);
  return true;
}

export function dropSession(id: string, owner: SessionOwner): boolean {
  const s = sessions.get(id);
  if (!s || s.tenantId !== owner.tenantId || s.userId !== owner.userId) {
    return false;
  }
  s.agent.abort();
  sessions.delete(id);
  return true;
}

function evictIfNeeded(): void {
  if (sessions.size <= MAX_SESSIONS) return;
  // 最久没用的会话先出(只踢空闲的)
  let oldest: Session | null = null;
  for (const s of sessions.values()) {
    if (s.busy) continue;
    if (!oldest || s.createdAt < oldest.createdAt) oldest = s;
  }
  if (oldest) {
    oldest.agent.abort();
    sessions.delete(oldest.id);
  }
}

const SYSTEM_PROMPT = `你是 HROnfire 招聘平台的 AI 助手,服务对象是 HR 用户。

你可以调用以下工具获取真实业务数据,回答必须基于工具返回,不要编造数字或候选人:
- search_talents:查候选人(姓名/邮箱模糊搜,含简历数、面试数、最高匹配分)
- list_jobs:查岗位(id、名称、等级、状态)
- get_job_matches:查某岗位的 AI 匹配结果(分数/亮点/短板)
- list_interviews:查面试列表(状态、完成度、平均分)
- get_interview_suggestions:查 AI 推荐的待确认面试项(匹配>90)
- get_interview_report:查某场面试的完整报告(维度分/结论/证据)
- list_question_items:查题库题目
- dismiss_interview_suggestion:忽略某条推荐面试(需用户确认才执行)
- create_interview:发起远程面试生成邀请链接(需用户确认才执行)
- generate_questions:AI 生成题目存题库(需用户确认才执行)
- search_knowledge:知识库全文检索(制度/FAQ/福利,回答政策问题先查这里)
- search_product_help:产品使用手册检索(用户问"系统怎么用/功能在哪"时先查这里,引用章节名)
- save_filter:把自然语言筛选条件存成列表页筛选器(需用户确认;page 用 talents/cases/employees 等页标识)
- list_entities / search_records / count_records:通用实体检索与计数
  (不确定查什么实体时先 list_entities)
- aggregate_records:数值字段聚合(sum/avg/min/max/count,回答"总额/平均"类问题)
- update_record:更新记录白名单字段(需用户确认才执行;先 search_records 拿 id)
- list_relationships:查一条记录关联的其它实体(候选人→申请/面试/offer,员工→申请/工单等)
- list_report_kinds / run_report:执行内置报表(管道漏斗/周摘要/待办/编制)并解读
- list_playbooks / get_playbook:流程剧本(租户自定义 SOP)
- list_announcements / publish_announcement:公司公告(发布需确认)
- create_request / list_my_requests:员工申请(创建需确认)
- create_case / list_my_cases / list_cases / update_case_status:
  HR 工单(创建/改状态需确认)
- check_salary_band:offer 定薪是否落在薪酬带宽内
- create_watcher / list_watchers / delete_watcher:触发式提醒(写需确认)
- create_report_subscription / list_report_subscriptions /
  delete_report_subscription:每周定时邮件报表(写需确认)
- get_activity:租户近期动态流

行为准则:
1. 涉及"查/有哪些/多少"的问题,先调工具拿数据再回答;数据为空就如实说没有
2. 回答用简体中文,简洁,数字优先;列表类结果用 markdown 表格或短列表
3. 要执行写操作时直接调用对应工具 — 系统会自动让用户在界面上确认,不要重复向用户索要确认话术
4. 用户给的是名字而非 id 时,先列表/搜索工具找到 id 再调用需要 id 的工具
5. 候选人隐私信息(完整联系方式)不要主动罗列,除非用户明确要
6. 执行多步业务流程(发 offer/批量发面试/处理工单/发公告/订阅报表等)前,
   先调 list_playbooks 查租户有无对应 SOP 剧本;命中则 get_playbook 读后
   严格按剧本步骤执行,剧本缺失或不适配再按通用流程走`;

/** 工具结果 SSE 载荷裁剪:text 分块上限 12k,深拷贝避免改到 agent 内部状态。 */
function boundToolResult(result: unknown): unknown {
  if (result === null || result === undefined) return null;
  try {
    const MAX = 12_000;
    const clone = JSON.parse(JSON.stringify(result)) as {
      content?: Array<{ type?: string; text?: string }>;
      details?: unknown;
      isError?: boolean;
    };
    if (Array.isArray(clone.content)) {
      for (const c of clone.content) {
        if (typeof c?.text === "string" && c.text.length > MAX) {
          c.text = c.text.slice(0, MAX) + "…(已截断)";
        }
      }
    }
    return clone;
  } catch {
    return { isError: true, content: [{ type: "text", text: "结果不可序列化" }] };
  }
}

function waitForConfirm(
  callId: string,
  name: string,
  args: unknown,
  emit: SseEmit,
  tenantId: string,
  userId: string,
): Promise<boolean> {
  // callId 由 LLM 生成,可能在一轮里重复 — 撞号会覆盖 pending 注册,
  // 前一个调用被迫等满 120s 超时。重复时补随机后缀隔离。
  if (pendingConfirms.has(callId)) {
    callId = `${callId}#${randomUUID().slice(0, 8)}`;
  }
  emit("confirm_required", {
    call_id: callId,
    name,
    label: toolLabel(name),
    args,
  });
  return new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => {
      if (pendingConfirms.delete(callId)) {
        emit("confirm_timeout", { call_id: callId });
        resolve(false);
      }
    }, CONFIRM_TIMEOUT_MS);
    pendingConfirms.set(callId, {
      tenantId,
      userId,
      resolve: (approved) => {
        clearTimeout(timer);
        resolve(approved);
      },
    });
  });
}

export function getOrCreateSession(opts: {
  id: string;
  user: UserCtx;
  llm: LlmConfig;
  toolToken: string;
  emit: SseEmit;
}): Session | null {
  const { id, user, llm, toolToken, emit } = opts;
  const existing = sessions.get(id);
  if (existing) {
    // conversation_id 只对创建它的 (tenant,user) 有效 — 拿到他人 id 时按
    // 不存在处理,否则会话上下文(含他人业务数据)会泄漏进新请求。
    if (existing.tenantId !== user.tenant_id || existing.userId !== user.id) {
      return null;
    }
    // busy 会话不改写 emit/tools — 上一个 run 的帧要继续流给原请求,
    // 本次请求会在 server 层收到 busy 错误,不应把输出劫持到新 res。
    if (!existing.busy) {
      existing.toolToken = toolToken; // 每次请求刷新,避免拿到过期 JWT
      existing.llm = llm;
      existing.emit = emit;
      existing.agent.state.tools = hrTools(toolToken);
      // 租户可能在会话存续期间换/改 LLM 配置 — 原地 upsert provider + model
      existing.agent.state.model = buildModel(llm, existing.models).model;
    }
    return existing;
  }

  evictIfNeeded();

  const stored = loadConversation(user.tenant_id, user.id, id);
  const { models, model } = buildModel(llm);

  const agent = new Agent({
    initialState: {
      systemPrompt: SYSTEM_PROMPT,
      model,
      tools: hrTools(toolToken),
      messages: (stored?.messages ?? []) as AgentMessage[],
    },
    streamFn: models.streamSimple.bind(models),
    // 业务操作默认串行:写操作一个一个来,确认门也更清晰
    toolExecution: "sequential",
    sessionId: id,
  });

  const sess: Session = {
    id,
    tenantId: user.tenant_id,
    userId: user.id,
    userName: user.name ?? "",
    agent,
    models,
    toolToken,
    llm,
    busy: false,
    createdAt: new Date().toISOString(),
    emit,
  };

  // 写工具 → 停放等 UI 确认;批准放行,拒绝/超时 → block
  agent.beforeToolCall = async ({ toolCall, args }: { toolCall: { id?: string; name: string }; args: unknown }) => {
    if (!WRITE_TOOLS.has(toolCall.name)) return undefined;
    const callId = toolCall.id ?? `${toolCall.name}-${Date.now()}`;
    const ok = await waitForConfirm(
      callId,
      toolCall.name,
      args,
      sess.emit,
      sess.tenantId,
      sess.userId,
    );
    if (ok) return undefined;
    return {
      block: true,
      reason: "用户取消了这次操作或确认超时",
      terminate: false,
    };
  };

  agent.subscribe((event) => {
    const e = event as Record<string, unknown>;
    switch (event.type) {
      case "message_update": {
        const ame = (event as { assistantMessageEvent?: { type?: string; delta?: string } })
          .assistantMessageEvent;
        if (ame?.type === "text_delta" && ame.delta) {
          sess.emit("text_delta", { delta: ame.delta });
        }
        break;
      }
      case "tool_execution_start": {
        sess.emit("tool_start", {
          call_id: e.toolCallId,
          name: e.toolName,
          label: toolLabel(String(e.toolName)),
          args: e.args,
        });
        break;
      }
      case "tool_execution_end": {
        sess.emit("tool_end", {
          call_id: e.toolCallId,
          name: e.toolName,
          is_error: Boolean(e.isError),
        });
        // T6:工具结果透传给前端按工具名渲染富卡片。tools.ts 的 toolResult() 已截断到 8k/30k,
        // 这里再兜底裁剪每个 text 分块,保证 SSE 帧体积可控。
        sess.emit("tool_result", {
          call_id: e.toolCallId,
          name: e.toolName,
          is_error: Boolean(e.isError),
          duration_ms: e.durationMs ?? null,
          result: boundToolResult(e.result),
        });
        break;
      }
      case "message_end": {
        // prompt() 对 LLM 调用失败不 reject(错误被收敛进 stopReason="error"
        // 的 assistant 消息) — 这里翻译成 SSE error,否则用户只看到空回复。
        const msg = event.message as {
          stopReason?: string;
          errorMessage?: string;
        };
        if (msg?.stopReason === "error" && msg.errorMessage) {
          sess.emit("error", { message: `LLM 调用失败:${msg.errorMessage}` });
        }
        break;
      }
      case "agent_end":
        persist(sess);
        sess.emit("done", { conversation_id: id });
        break;
      default:
        break;
    }
  });

  sessions.set(id, sess);
  return sess;
}

function persist(sess: Session): void {
  try {
    const messages = sess.agent.state.messages as unknown[];
    const firstUser = messages.find(
      (m) => (m as { role?: string }).role === "user",
    ) as { content?: unknown } | undefined;
    const text =
      typeof firstUser?.content === "string"
        ? firstUser.content
        : Array.isArray(firstUser?.content)
          ? String(
              (firstUser.content as Array<{ type?: string; text?: string }>).find(
                (c) => c.type === "text",
              )?.text ?? "",
            )
          : "";
    const conv: StoredConversation = {
      id: sess.id,
      tenant_id: sess.tenantId,
      user_id: sess.userId,
      title: text.slice(0, 40) || "新对话",
      created_at: sess.createdAt,
      updated_at: new Date().toISOString(),
      messages,
    };
    saveConversation(conv);
  } catch (err) {
    console.error("[assistant] persist failed", err);
  }
}
