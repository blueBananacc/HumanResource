import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

type KeyRow = {
  id: string;
  scope: string;
  key: string;
  status: string;
  response_status: number | null;
  resource_type: string | null;
  resource_id: string | null;
  user_id: string | null;
  request_hash: string;
  expires_at: string;
  locked_at: string | null;
  created_at: string;
};

type Stats = { by_status: Record<string, number>; total: number; recent_24h: number };

const STATUS_LABEL: Record<string, string> = {
  in_progress: "执行中",
  completed: "已完成",
  failed: "已失败",
};

const STATUS_CLS: Record<string, string> = {
  completed: "bg-emerald-50 text-emerald-700",
  failed: "bg-red-50 text-red-700",
  in_progress: "bg-amber-50 text-amber-700",
};

/** 幂等键运维页(T10):看哪些写请求带了键、终态回放还是进行中;可手动释放。 */
export function IdempotencyKeys() {
  const qc = useQueryClient();
  const { data: keys, isLoading } = useQuery({
    queryKey: ["idempotency-keys"],
    queryFn: () => fetchJSON<KeyRow[]>("/api/idempotency-keys?limit=100"),
    refetchInterval: 15_000,
  });
  const { data: stats } = useQuery({
    queryKey: ["idempotency-stats"],
    queryFn: () => fetchJSON<Stats>("/api/idempotency-keys/stats"),
    refetchInterval: 15_000,
  });
  const release = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/idempotency-keys/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["idempotency-keys"] }),
  });

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-6">
      <div>
        <h1 className="font-heading text-xl text-[var(--color-text)]">幂等键</h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          带 Idempotency-Key 的写请求在此留痕:同键重试返回首次结果,不再重复落库。
          助手写工具用 call_id 作键;前端写请求自动生成 ui-* 键。
        </p>
      </div>
      {stats && (
        <div className="grid grid-cols-4 gap-3">
          <Stat label="总键数" value={stats.total} />
          <Stat label="24h 新建" value={stats.recent_24h} />
          <Stat label="已完成" value={stats.by_status.completed ?? 0} />
          <Stat label="执行中/失败" value={(stats.by_status.in_progress ?? 0) + (stats.by_status.failed ?? 0)} />
        </div>
      )}
      <div className="overflow-hidden rounded-2xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border-subtle)] text-left text-xs text-[var(--color-text-muted)]">
              <th className="px-4 py-3">键</th>
              <th className="px-4 py-3">作用域</th>
              <th className="px-4 py-3">状态</th>
              <th className="px-4 py-3">响应码</th>
              <th className="px-4 py-3">资源</th>
              <th className="px-4 py-3">创建</th>
              <th className="px-4 py-3">过期</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr><td colSpan={8} className="px-4 py-6 text-center text-[var(--color-text-muted)]">加载中…</td></tr>
            )}
            {!isLoading && !(keys ?? []).length && (
              <tr><td colSpan={8} className="px-4 py-6 text-center text-[var(--color-text-muted)]">暂无幂等键记录</td></tr>
            )}
            {(keys ?? []).map((k) => (
              <tr key={k.id} className="border-b border-[var(--color-border-subtle)] last:border-0">
                <td className="max-w-44 truncate px-4 py-3 font-mono text-xs text-[var(--color-text)]" title={k.key}>
                  {k.key}
                </td>
                <td className="px-4 py-3 text-xs text-[var(--color-text-muted)]">{k.scope}</td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-0.5 text-xs ${STATUS_CLS[k.status] ?? "bg-[var(--color-bg-muted)] text-[var(--color-text)]"}`}>
                    {STATUS_LABEL[k.status] ?? k.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-[var(--color-text-muted)]">{k.response_status ?? "—"}</td>
                <td className="max-w-40 truncate px-4 py-3 text-xs text-[var(--color-text-muted)]" title={k.resource_id ?? ""}>
                  {k.resource_type ? `${k.resource_type}` : "—"}
                </td>
                <td className="px-4 py-3 text-[var(--color-text)]">{formatRelative(k.created_at)}</td>
                <td className="px-4 py-3 text-[var(--color-text-muted)]">{formatRelative(k.expires_at)}</td>
                <td className="px-4 py-3 text-right">
                  <button
                    className="text-xs text-[var(--color-accent)] hover:underline disabled:opacity-40"
                    disabled={release.isPending}
                    onClick={() => release.mutate(k.id)}
                  >
                    释放
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white px-4 py-3 shadow-[var(--shadow-card)]">
      <p className="text-xs text-[var(--color-text-muted)]">{label}</p>
      <p className="mt-1 text-lg font-semibold text-[var(--color-text)]">{value}</p>
    </div>
  );
}
