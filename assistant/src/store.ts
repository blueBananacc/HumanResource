/** 会话持久化 — JSON 文件,按 DATA_DIR/{tenant}/{user}/{conv}.json 组织。
 *
 * 存的是 pi agent 的 AgentMessage[](含 user/assistant/toolResult,可直接
 * 回放进 initialState.messages);多副本部署时需要换共享存储,当前单
 * sidecar 场景文件即可。
 */
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

const DATA_DIR = process.env.DATA_DIR || "./data";

export type StoredConversation = {
  id: string;
  tenant_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: unknown[];
};

function dirOf(tenantId: string, userId: string): string {
  // 防路径穿越:id 只留安全字符
  const clean = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "_");
  return join(DATA_DIR, clean(tenantId), clean(userId));
}

function fileOf(tenantId: string, userId: string, id: string): string {
  const clean = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  return join(dirOf(tenantId, userId), `${clean}.json`);
}

export function saveConversation(conv: StoredConversation): void {
  const dir = dirOf(conv.tenant_id, conv.user_id);
  mkdirSync(dir, { recursive: true });
  writeFileSync(fileOf(conv.tenant_id, conv.user_id, conv.id), JSON.stringify(conv));
}

export function loadConversation(
  tenantId: string,
  userId: string,
  id: string,
): StoredConversation | null {
  const f = fileOf(tenantId, userId, id);
  if (!existsSync(f)) return null;
  try {
    const parsed = JSON.parse(readFileSync(f, "utf-8")) as StoredConversation;
    if (parsed.tenant_id !== tenantId || parsed.user_id !== userId) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function listConversations(
  tenantId: string,
  userId: string,
): Omit<StoredConversation, "messages">[] {
  const dir = dirOf(tenantId, userId);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      try {
        const c = JSON.parse(
          readFileSync(join(dir, f), "utf-8"),
        ) as StoredConversation;
        return {
          id: c.id,
          tenant_id: c.tenant_id,
          user_id: c.user_id,
          title: c.title,
          created_at: c.created_at,
          updated_at: c.updated_at,
          message_count: c.messages?.length ?? 0,
        };
      } catch {
        return null;
      }
    })
    .filter((x): x is NonNullable<typeof x> => x !== null)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export function deleteConversation(
  tenantId: string,
  userId: string,
  id: string,
): boolean {
  const f = fileOf(tenantId, userId, id);
  if (!existsSync(f)) return false;
  rmSync(f);
  return true;
}
