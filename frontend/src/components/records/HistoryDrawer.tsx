import { useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { Drawer } from "@/components/ui/drawer";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

type AuditEvent = {
  id: string;
  actor_email: string | null;
  action: string;
  entity_type: string;
  entity_id: string;
  result: string;
  via: string | null;
  created_at: string;
  detail: Record<string, unknown> | null;
};

type PageOut<T> = { items: T[]; total: number };

/** 记录级变更历史抽屉 — 复用 /api/audit/events?entity_id=。 */
export function HistoryDrawer({
  entityId,
  open,
  onClose,
}: {
  entityId: string;
  open: boolean;
  onClose: () => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["record-history", entityId],
    enabled: open && !!entityId,
    queryFn: () =>
      fetchJSON<PageOut<AuditEvent>>(
        `/api/audit/events?entity_id=${encodeURIComponent(entityId)}&page_size=50`,
      ),
  });

  return (
    <Drawer open={open} onClose={onClose} title="变更记录" width={480}>
      {isLoading && <p className="text-sm text-[var(--color-text-muted)]">加载中…</p>}
      {!isLoading && (data?.items ?? []).length === 0 && (
        <Empty icon={History} title="暂无变更记录" description="该记录还没有留痕的变更" />
      )}
      <ul className="space-y-3">
        {(data?.items ?? []).map((e) => (
          <li
            key={e.id}
            className="rounded-xl border border-[var(--color-border-subtle)] bg-white p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-[var(--color-text)]">
                {e.action}
                {e.via && e.via !== "ui" && (
                  <span className="ml-1.5 rounded-full bg-[var(--color-cappuccino)] px-1.5 py-0.5 text-[10px] text-[var(--color-brand)]">
                    {e.via}
                  </span>
                )}
              </span>
              <span className="text-xs text-[var(--color-text-muted)]">
                {formatRelative(e.created_at)}
              </span>
            </div>
            <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
              {e.actor_email ?? "系统"} · {e.result}
            </p>
            {e.detail && Object.keys(e.detail).length > 0 && (
              <pre className="mt-1.5 max-h-32 overflow-auto rounded-lg bg-[var(--color-bg-muted)] p-2 text-[11px] text-[var(--color-text-muted)]">
                {JSON.stringify(e.detail, null, 2)}
              </pre>
            )}
          </li>
        ))}
      </ul>
    </Drawer>
  );
}
