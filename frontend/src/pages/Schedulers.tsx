import { useQuery } from "@tanstack/react-query";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

type Job = {
  name: string;
  label: string;
  source: string;
  last_status: string | null;
  last_started_at: string | null;
  last_duration_ms: number | null;
  last_message: string | null;
};

/** 后台任务运维页:loop/celery 任务最近一次运行状态。 */
export function Schedulers() {
  const { data: jobs, isLoading } = useQuery({
    queryKey: ["scheduler-jobs"],
    queryFn: () => fetchJSON<Job[]>("/api/scheduler/jobs"),
    refetchInterval: 30_000,
  });

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-6">
      <div>
        <h1 className="font-heading text-xl text-[var(--color-text)]">后台任务</h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          邮箱同步/触发器调度/简历解析/匹配评估等后台任务的最近运行状态(每 30s 自动刷新)。
        </p>
      </div>
      <div className="overflow-hidden rounded-2xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border-subtle)] text-left text-xs text-[var(--color-text-muted)]">
              <th className="px-4 py-3">任务</th>
              <th className="px-4 py-3">来源</th>
              <th className="px-4 py-3">上次运行</th>
              <th className="px-4 py-3">耗时</th>
              <th className="px-4 py-3">状态</th>
              <th className="px-4 py-3">消息</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr><td colSpan={6} className="px-4 py-6 text-center text-[var(--color-text-muted)]">加载中…</td></tr>
            )}
            {(jobs ?? []).map((j) => (
              <tr key={j.name} className="border-b border-[var(--color-border-subtle)] last:border-0">
                <td className="px-4 py-3">
                  <p className="font-medium text-[var(--color-text)]">{j.label}</p>
                  <p className="text-xs text-[var(--color-text-muted)]">{j.name}</p>
                </td>
                <td className="px-4 py-3 text-[var(--color-text-muted)]">{j.source}</td>
                <td className="px-4 py-3 text-[var(--color-text)]">
                  {j.last_started_at ? formatRelative(j.last_started_at) : "从未运行"}
                </td>
                <td className="px-4 py-3 text-[var(--color-text-muted)]">
                  {j.last_duration_ms != null ? `${j.last_duration_ms}ms` : "—"}
                </td>
                <td className="px-4 py-3">
                  {j.last_status ? (
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        j.last_status === "success"
                          ? "bg-[var(--color-bg-muted)] text-[var(--color-text)]"
                          : "bg-[var(--color-cappuccino)] text-[var(--color-accent)]"
                      }`}
                    >
                      {j.last_status === "success" ? "成功" : "失败"}
                    </span>
                  ) : (
                    <span className="text-xs text-[var(--color-text-muted)]">—</span>
                  )}
                </td>
                <td className="max-w-56 truncate px-4 py-3 text-xs text-[var(--color-text-muted)]" title={j.last_message ?? ""}>
                  {j.last_message ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
