import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, KeyRound, Loader2, Plug, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

/**
 * MCP 接入令牌(个人访问令牌)— 让外部 AI 客户端连进 HROnfire。
 *
 * 任何登录用户都能给自己签令牌;令牌等价本人身份,调用的工具与页内
 * AI 助手同一套,权限/租户/org 照旧由后端判定,写操作走审计(via="mcp")。
 */

type McpToken = {
  id: string;
  name: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
};

type McpTokenCreated = McpToken & { token: string };

export function McpTokens() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const { data } = useQuery({
    queryKey: ["mcp-tokens"],
    queryFn: () => fetchJSON<McpToken[]>("/api/me/mcp-tokens"),
  });

  const [name, setName] = useState("");
  const [created, setCreated] = useState<McpTokenCreated | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      fetchJSON<McpTokenCreated>("/api/me/mcp-tokens", {
        method: "POST",
        body: JSON.stringify({ name: name.trim() }),
      }),
    onSuccess: (row) => {
      qc.invalidateQueries({ queryKey: ["mcp-tokens"] });
      setCreated(row);
      setName("");
      setCopied(false);
      setError(null);
    },
    onError: (e: unknown) =>
      setError(e instanceof ApiError ? e.message : "签发失败"),
  });

  const revoke = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/mcp-tokens/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["mcp-tokens"] }),
  });

  async function onRevoke(t: McpToken) {
    if (
      !(await confirm({
        title: `吊销令牌「${t.name}」?`,
        description: "使用该令牌的客户端将立即无法访问。",
        tone: "danger",
        confirmLabel: "吊销",
      }))
    )
      return;
    revoke.mutate(t.id);
  }

  async function copyToken() {
    if (!created) return;
    await navigator.clipboard.writeText(created.token);
    setCopied(true);
  }

  const mcpUrl = `${location.origin}/api/mcp`;

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <header className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)] flex items-center gap-2">
          <Plug className="w-6 h-6" />
          MCP 接入
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          签发个人访问令牌,让外部 AI 客户端(Claude Desktop、Cursor、
          自研 agent 等)通过 MCP 协议调用 HROnfire 的 HR 工具。
          令牌只能驱动与页内 AI 助手相同的工具集 —— 权限/租户/org
          判定一致,不能直接访问其它 API;所有写操作同样进审计日志,
          吊销立即生效。
        </p>
      </header>

      {/* 接入信息 */}
      <div className="bg-[var(--color-info-soft)] text-[var(--color-info)] px-4 py-3 rounded-lg text-[13px] font-body flex flex-col gap-1">
        <span>
          MCP 端点:<code className="font-mono">{mcpUrl}</code>
        </span>
        <span>
          鉴权:<code className="font-mono">Authorization: Bearer hrm_…</code>
          (客户端 config 里填 endpoint + token 即可,Streamable HTTP 传输)
        </span>
      </div>

      {/* 新令牌明文(仅显示一次) */}
      {created && (
        <div className="border border-[var(--color-warning)] bg-[var(--color-warning-soft)] rounded-xl p-4 flex flex-col gap-2">
          <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
            令牌「{created.name}」已签发 —— 明文只显示这一次,请立即复制保存
          </span>
          <div className="flex items-center gap-2">
            <code className="flex-1 font-mono text-[13px] bg-white border border-[var(--color-border-subtle)] rounded-md px-3 py-2 break-all select-all">
              {created.token}
            </code>
            <Button type="button" variant="secondary" onClick={copyToken}>
              <Copy className="w-4 h-4" />
              {copied ? "已复制" : "复制"}
            </Button>
          </div>
        </div>
      )}

      {/* 签发表单 */}
      <div className="flex items-center gap-2 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-4">
        <KeyRound className="w-4 h-4 text-[var(--color-text-tertiary)]" />
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="令牌备注名(如:Claude Desktop / 我的 Cursor)"
          className="flex-1"
          maxLength={64}
          onKeyDown={(e) => {
            if (e.key === "Enter" && name.trim()) create.mutate();
          }}
        />
        <Button
          onClick={() => create.mutate()}
          disabled={!name.trim() || create.isPending}
        >
          {create.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
          签发令牌
        </Button>
      </div>
      {error && (
        <div className="text-[13px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-3 py-2 rounded-md font-body">
          {error}
        </div>
      )}

      {/* 令牌列表 */}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        <table className="w-full text-[13px] font-body">
          <thead>
            <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
              <th className="px-4 py-2.5 font-medium">备注</th>
              <th className="px-4 py-2.5 font-medium">创建时间</th>
              <th className="px-4 py-2.5 font-medium">最近使用</th>
              <th className="px-4 py-2.5 font-medium">状态</th>
              <th className="px-4 py-2.5 font-medium w-20"></th>
            </tr>
          </thead>
          <tbody>
            {(data ?? []).map((t) => (
              <tr
                key={t.id}
                className="border-b border-[var(--color-border-row)] last:border-0"
              >
                <td className="px-4 py-2.5">{t.name}</td>
                <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">
                  {formatRelative(t.created_at)}
                </td>
                <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">
                  {t.last_used_at ? formatRelative(t.last_used_at) : "从未使用"}
                </td>
                <td className="px-4 py-2.5">
                  {t.revoked_at ? (
                    <span className="text-[var(--color-danger)]">已吊销</span>
                  ) : (
                    <span className="text-[var(--color-success)]">有效</span>
                  )}
                </td>
                <td className="px-4 py-2.5 text-right">
                  {!t.revoked_at && (
                    <button
                      type="button"
                      onClick={() => onRevoke(t)}
                      className="text-[var(--color-danger)] hover:opacity-70 p-1"
                      title="吊销"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {data && data.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="px-4 py-8 text-center text-[var(--color-text-tertiary)]"
                >
                  还没有令牌 —— 上方填个备注名签发第一个
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
