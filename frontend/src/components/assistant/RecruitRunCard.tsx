import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  CheckCircle2,
  Circle,
  CircleMinus,
  Clock3,
  Loader2,
  Workflow,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { cn } from "@/lib/utils";

/** P1 招聘 SOP 任务链 — 对话内步骤链卡片(轮询实时推进)。 */

type RunStep = {
  id: string;
  seq: number;
  key: string;
  label: string;
  agent_label: string;
  status: string;
  detail: Record<string, unknown>;
  link: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type RecruitRun = {
  id: string;
  kind: string;
  title: string;
  status: string;
  context: Record<string, unknown>;
  created_at: string;
  finished_at: string | null;
  steps: RunStep[];
};

const RUN_ZH: Record<string, string> = {
  running: "进行中",
  done: "已完成",
  failed: "有步骤失败",
  cancelled: "已取消",
};

const STEP_ICON: Record<string, { icon: LucideIcon; cls: string; spin?: boolean }> = {
  pending: { icon: Circle, cls: "text-[var(--color-text-tertiary)]" },
  running: { icon: Loader2, cls: "text-[var(--color-accent)]", spin: true },
  done: { icon: CheckCircle2, cls: "text-[var(--color-success,#16a34a)]" },
  failed: { icon: XCircle, cls: "text-[var(--color-danger,#dc2626)]" },
  skipped: { icon: CircleMinus, cls: "text-[var(--color-text-tertiary)]" },
  waiting_confirm: { icon: Clock3, cls: "text-[var(--color-warning,#d97706)]" },
};

function detailSummary(d: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(d ?? {})) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) {
      parts.push(`${k} ×${v.length}`);
    } else {
      const s = String(v);
      parts.push(s.length > 40 ? s.slice(0, 40) + "…" : s);
    }
    if (parts.length >= 2) break;
  }
  return parts.join(" · ");
}

export function RecruitRunCard({ run: initial }: { run: RecruitRun }) {
  // 运行中 → 2.5s 轮询;终态 → 停
  const { data: run } = useQuery({
    queryKey: ["recruit-run", initial.id],
    queryFn: () => fetchJSON<RecruitRun>(`/api/recruit-runs/${initial.id}`),
    initialData: initial,
    refetchInterval: (q) =>
      q.state.data?.status === "running" ? 2500 : false,
    staleTime: 0,
  });

  const doneCount = run.steps.filter((s) => s.status === "done").length;
  const runMeta = RUN_ZH[run.status] ?? run.status;

  return (
    <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)] overflow-hidden">
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-[var(--color-border-row)] bg-[var(--color-bg-canvas)]">
        <Workflow className="w-3.5 h-3.5 text-[var(--color-accent)]" />
        <span className="text-[11px] font-bold text-[var(--color-text-secondary)] flex-1 truncate">
          {run.title} · 任务链
        </span>
        <span
          className={cn(
            "text-[10px] rounded-full px-1.5 py-0.5",
            run.status === "running"
              ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
              : run.status === "done"
                ? "bg-[var(--color-success-soft,#f0fdf4)] text-[var(--color-success,#16a34a)]"
                : "bg-[var(--color-danger-soft,#fef2f2)] text-[var(--color-danger,#dc2626)]",
          )}
        >
          {run.status === "running"
            ? `${doneCount}/${run.steps.length} 步`
            : runMeta}
        </span>
      </div>
      <div className="px-3 py-1.5">
        {run.steps.map((s, i) => {
          const meta = STEP_ICON[s.status] ?? STEP_ICON.pending;
          const Icon = meta.icon;
          const summary =
            s.status === "done" || s.status === "failed"
              ? detailSummary(s.detail)
              : "";
          const last = i === run.steps.length - 1;
          return (
            <div key={s.id} className="flex gap-2">
              {/* 时间线 rail */}
              <div className="flex flex-col items-center w-4 shrink-0">
                <Icon
                  className={cn(
                    "w-3.5 h-3.5 mt-1.5 shrink-0",
                    meta.cls,
                    meta.spin && "animate-spin",
                  )}
                />
                {!last && (
                  <div className="w-px flex-1 min-h-2 bg-[var(--color-border-row)]" />
                )}
              </div>
              <div className="flex-1 min-w-0 py-1.5">
                <div className="flex items-center gap-1.5 text-[12px]">
                  <span
                    className={cn(
                      "truncate",
                      s.status === "pending"
                        ? "text-[var(--color-text-tertiary)]"
                        : "text-[var(--color-text-primary)]",
                      s.status === "running" && "font-bold",
                    )}
                  >
                    {s.label}
                  </span>
                  {s.agent_label && (
                    <span className="shrink-0 rounded-full px-1.5 py-px text-[9px] bg-[var(--color-bg-muted)] text-[var(--color-text-tertiary)]">
                      {s.agent_label}
                    </span>
                  )}
                  {s.link && s.status === "done" && (
                    <Link
                      to={s.link}
                      className="ml-auto shrink-0 text-[10px] text-[var(--color-accent)] hover:underline"
                    >
                      查看 →
                    </Link>
                  )}
                </div>
                {summary && (
                  <div className="text-[10px] text-[var(--color-text-tertiary)] truncate">
                    {summary}
                  </div>
                )}
                {s.status === "waiting_confirm" && (
                  <div className="text-[10px] text-[var(--color-warning,#d97706)]">
                    等你确认下方操作…
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
