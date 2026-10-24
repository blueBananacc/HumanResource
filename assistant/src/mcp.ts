/** MCP(Model Context Protocol)服务端 — 外部 AI 客户端接入。
 *
 * 协议:JSON-RPC 2.0 over Streamable HTTP(单端点 POST,JSON 响应)。
 * 方法:initialize / notifications.initialized / ping / tools/list / tools/call。
 *
 * 鉴权:客户端带 `Authorization: Bearer hrm_*`(用户在设置页签发的 MCP 令牌)。
 * 每个令牌经 backend `/api/mcp/token-exchange` 换成等价 tool_token ——
 * 该端点仅内部开放(`X-Internal-Token` 由 JWT_SECRET 派生),外部拿 PAT
 * 无法直接换全量 JWT;PAT 的有效范围被收敛在 MCP 工具面内。
 * 权限/租户/org 过滤照旧全部由 backend 判定,sidecar 不做权限决策。
 * 令牌缓存 10s(exchange 只是轻量查库;短缓存把吊销失效延迟压到 10s 内)。
 *
 * 与页内聊天的差异:没有 beforeToolCall 确认门 —— 持有令牌 = 用户授权
 * 该客户端以自己身份操作(与 MintHCM MCP 同一模型)。写操作照常写审计
 * (audit.via="mcp"),可追责。
 */
import { createHmac } from "node:crypto";
import type { Request, Response } from "express";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { validateToolArguments, type JsonObject } from "@earendil-works/pi-ai";
import { hrTools } from "./tools.js";

const BACKEND = process.env.BACKEND_URL || "http://backend:8000";
const CACHE_TTL_MS = 10_000;

// backend /api/mcp/token-exchange 的内部共享密钥 —— 与 backend _internal_key()
// 同一算法(HMAC-SHA256(JWT_SECRET, 固定消息串))。知道 JWT_SECRET 的攻击者
// 本来就能直接伪造任意 JWT,派生值不新增攻击面。
const INTERNAL_TOKEN = createHmac("sha256", process.env.JWT_SECRET ?? "")
  .update("hronfire/mcp/token-exchange")
  .digest("hex");
if (!process.env.JWT_SECRET) {
  console.warn(
    "[mcp] JWT_SECRET 未配置 — token-exchange 内部鉴权必失败,MCP 不可用(compose 里 assistant 与 backend 共用 env_file)",
  );
}

type McpSession = {
  toolToken: string;
  user: { tenant_id: string; id: string; role: string; name: string };
  expiresAt: number;
};

const sessions = new Map<string, McpSession>();

async function exchange(bearer: string): Promise<McpSession | null> {
  const cached = sessions.get(bearer);
  if (cached && cached.expiresAt > Date.now()) return cached;
  if (cached) sessions.delete(bearer);
  // 顺手清理过期项,防止 Map 随不同令牌数量无限增长
  const now = Date.now();
  for (const [k, v] of sessions) {
    if (v.expiresAt <= now) sessions.delete(k);
  }
  try {
    const resp = await fetch(`${BACKEND}/api/mcp/token-exchange`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-internal-token": INTERNAL_TOKEN,
      },
      body: JSON.stringify({ token: bearer }),
    });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      tool_token: string;
      user: McpSession["user"];
    };
    const sess: McpSession = {
      toolToken: data.tool_token,
      user: data.user,
      expiresAt: Date.now() + CACHE_TTL_MS,
    };
    sessions.set(bearer, sess);
    return sess;
  } catch {
    return null;
  }
}

type JsonRpc = {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
};

function result(id: JsonRpc["id"], value: unknown) {
  return { jsonrpc: "2.0", id: id ?? null, result: value };
}

function rpcError(id: JsonRpc["id"], code: number, message: string) {
  return { jsonrpc: "2.0", id: id ?? null, error: { code, message } };
}

const PROTOCOL_VERSION = "2025-03-26";
const SERVER_INFO = { name: "hronfire-assistant", version: "1.0.0" };

// inputSchema 给 MCP 客户端(pi-ai 的 Type.Object 产物与 MCP JSON Schema 兼容)
function toolSchemas(tools: AgentTool[]) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.parameters,
  }));
}

async function handleRpc(sess: McpSession, rpc: JsonRpc): Promise<unknown> {
  const tools = hrTools(sess.toolToken);
  switch (rpc.method) {
    case "initialize":
      return result(rpc.id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: SERVER_INFO,
      });
    case "notifications/initialized":
    case "notifications/cancelled":
      return undefined;
    case "ping":
      return result(rpc.id, {});
    case "tools/list":
      return result(rpc.id, { tools: toolSchemas(tools) });
    case "tools/call": {
      const name = rpc.params?.name as string | undefined;
      const args = (rpc.params?.arguments ?? {}) as JsonObject;
      if (args === null || typeof args !== "object" || Array.isArray(args)) {
        return rpcError(rpc.id, -32602, "arguments 必须是对象");
      }
      const tool = tools.find((t) => t.name === name);
      if (!tool) return rpcError(rpc.id, -32602, `未知工具: ${name}`);
      // 与 agent-loop 同一校验入口:arguments 先过工具声明的 TypeBox schema
      // (含值域/必填/required 检查),失败 -32602 而不是把坏参数丢给后端。
      let validated: Record<string, unknown>;
      try {
        validated = validateToolArguments(tool, {
          type: "toolCall",
          id: String(rpc.id ?? "mcp"),
          name: tool.name,
          arguments: args,
        }) as Record<string, unknown>;
      } catch (err) {
        return rpcError(rpc.id, -32602, `参数校验失败: ${err}`);
      }
      try {
        const out = (await tool.execute(String(rpc.id ?? "mcp"), validated)) as {
          content?: unknown;
          details?: unknown;
          isError?: boolean;
        };
        return result(rpc.id, {
          content: out.content ?? [{ type: "text", text: JSON.stringify(out) }],
          // 后端 HTTP 错误(403/402/...)在工具里被包装成 {error,…} 文本返回 —
          // 透传 isError,让 MCP 客户端知道这是失败结果而不是正常数据
          isError: Boolean(out.isError),
        });
      } catch (err) {
        return result(rpc.id, {
          content: [{ type: "text", text: `工具执行失败: ${err}` }],
          isError: true,
        });
      }
    }
    default:
      return rpcError(rpc.id, -32601, `不支持的方法: ${rpc.method}`);
  }
}

export async function mcpHandler(req: Request, res: Response): Promise<void> {
  const auth = req.headers.authorization ?? "";
  const bearer = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!bearer) {
    res.status(401).json({ error: "缺少 Authorization: Bearer <mcp_token>" });
    return;
  }
  const sess = await exchange(bearer);
  if (!sess) {
    res.status(401).json({ error: "令牌无效、已吊销或无权限" });
    return;
  }

  const body = req.body as JsonRpc | JsonRpc[];
  // JSON-RPC 批量请求(MCP 允许)
  const rpcs = Array.isArray(body) ? body : [body];
  const replies: unknown[] = [];
  for (const rpc of rpcs) {
    if (!rpc || typeof rpc.method !== "string") {
      replies.push(rpcError(rpc?.id ?? null, -32600, "非法 JSON-RPC 请求"));
      continue;
    }
    const out = await handleRpc(sess, rpc);
    if (out !== undefined) replies.push(out);
  }
  if (replies.length === 0) {
    // 全是 notifications —— 202 无响应体
    res.status(202).end();
    return;
  }
  res.status(200).json(Array.isArray(body) ? replies : replies[0]);
}
