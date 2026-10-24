/** HROnfire AI 助手 sidecar HTTP 服务。
 *
 * 只在 docker 网内监听(不发布端口),调用方是 backend 的 /api/assistant/*。
 * 协议:
 *   POST /chat         → SSE 流(text_delta / tool_start / tool_end /
 *                        confirm_required / done / error / conversation)
 *   POST /confirm      → 裁决 beforeToolCall 停放的写操作
 *   GET  /conversations?tenant_id&user_id          → 会话列表
 *   GET  /conversations/:id?tenant_id&user_id      → 单会话(含消息)
 *   DELETE /conversations/:id?tenant_id&user_id    → 删除
 *   GET  /health
 */
import express from "express";
import { randomUUID } from "node:crypto";
import {
  dropSession,
  getOrCreateSession,
  resolveConfirm,
} from "./agent.js";
import {
  deleteConversation,
  listConversations,
  loadConversation,
} from "./store.js";
import { mcpHandler } from "./mcp.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT || 3001);

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// MCP(JSON-RPC):外部 AI 客户端协议端点,经 backend /api/mcp 代理进来。
// Streamable HTTP 约定:不支持 GET(SSE 流)时回 405 而不是 404。
app.post("/mcp", mcpHandler);
app.get("/mcp", (_req, res) => {
  res.status(405).json({ error: "Method Not Allowed:本服务不提供 SSE 流" });
});

type ChatBody = {
  conversation_id?: string;
  message?: string;
  llm?: { base_url?: string; api_key?: string; model?: string; mock?: boolean };
  tool_token?: string;
  user?: { tenant_id?: string; id?: string; role?: string; name?: string };
};

app.post("/chat", async (req, res) => {
  const body = req.body as ChatBody;
  const message = body.message?.trim();
  const user = body.user;
  if (!message || !user?.tenant_id || !user.id || !body.tool_token) {
    res.status(400).json({ error: "missing message/user/tool_token" });
    return;
  }

  const convId = body.conversation_id || randomUUID();
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  const emit = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  emit("conversation", { conversation_id: convId });

  // mock LLM:不建 Agent,回固定文案保持 demo 可走通
  if (!body.llm?.model || body.llm.mock) {
    emit("text_delta", {
      delta:
        "当前租户还没有配置 LLM Provider,AI 助手暂时无法工作。" +
        "请到「系统 → LLM 配置」添加并激活一个 Provider(例如 DeepSeek)后再试。",
    });
    emit("done", { conversation_id: convId });
    res.end();
    return;
  }

  let sess;
  try {
    sess = getOrCreateSession({
      id: convId,
      user: {
        tenant_id: user.tenant_id,
        id: user.id,
        role: user.role ?? "hr",
        name: user.name,
      },
      llm: {
        base_url: body.llm.base_url,
        api_key: body.llm.api_key,
        model: body.llm.model,
      },
      toolToken: body.tool_token,
      emit,
    });
  } catch (err) {
    emit("error", { message: `初始化 agent 失败: ${err}` });
    res.end();
    return;
  }

  if (!sess) {
    emit("error", { message: "会话不存在或无权访问" });
    res.end();
    return;
  }

  if (sess.busy) {
    emit("error", { message: "上一条消息还在处理中,请稍等" });
    res.end();
    return;
  }
  sess.busy = true;

  let finished = false;
  res.on("close", () => {
    if (!finished) sess.agent.abort();
  });

  try {
    await sess.agent.prompt(message);
  } catch (err) {
    emit("error", { message: `AI 处理失败: ${err}` });
  } finally {
    finished = true;
    sess.busy = false;
    res.end();
  }
});

app.post("/confirm", (req, res) => {
  const { call_id, approved, tenant_id, user_id } = req.body as {
    call_id?: string;
    approved?: boolean;
    tenant_id?: string;
    user_id?: string;
  };
  if (!call_id || !tenant_id || !user_id) {
    res.status(400).json({ error: "missing call_id/tenant_id/user_id" });
    return;
  }
  const found = resolveConfirm(call_id, Boolean(approved), {
    tenantId: tenant_id,
    userId: user_id,
  });
  res.status(found ? 200 : 404).json({ ok: found });
});

app.get("/conversations", (req, res) => {
  const { tenant_id, user_id } = req.query as Record<string, string>;
  if (!tenant_id || !user_id) {
    res.status(400).json({ error: "missing tenant_id/user_id" });
    return;
  }
  res.json({ items: listConversations(tenant_id, user_id) });
});

app.get("/conversations/:id", (req, res) => {
  const { tenant_id, user_id } = req.query as Record<string, string>;
  if (!tenant_id || !user_id) {
    res.status(400).json({ error: "missing tenant_id/user_id" });
    return;
  }
  const conv = loadConversation(tenant_id, user_id, req.params.id);
  if (!conv) {
    res.status(404).json({ error: "not found" });
    return;
  }
  res.json(conv);
});

app.delete("/conversations/:id", (req, res) => {
  const { tenant_id, user_id } = req.query as Record<string, string>;
  if (!tenant_id || !user_id) {
    res.status(400).json({ error: "missing tenant_id/user_id" });
    return;
  }
  dropSession(req.params.id, { tenantId: tenant_id, userId: user_id });
  const ok = deleteConversation(tenant_id, user_id, req.params.id);
  res.status(ok ? 204 : 404).end();
});

app.listen(PORT, () => {
  console.log(`[assistant] listening on :${PORT}, backend=${process.env.BACKEND_URL}`);
});
