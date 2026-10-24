import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, RefreshCw, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchJSON } from "@/lib/api";
import { MarkdownContent } from "@/components/ui/markdown";

type SummaryResp = { entity: string; id: string; summary: string };

/**
 * 记录级 AI 摘要卡(T6-2):详情页顶部展示 LLM 生成的 360° 摘要。
 * 懒加载——首次展开才调 /api/ai/summarize,可手动刷新重算。
 */
export function SummaryCard({
  entity,
  id,
  title = "AI 摘要",
}: {
  entity: "candidates" | "employees" | "cases";
  id: string;
  title?: string;
}) {
  const [enabled, setEnabled] = useState(false);
  const { data, isFetching, isError, error, refetch } = useQuery({
    queryKey: ["ai-summary", entity, id],
    queryFn: () => fetchJSON<SummaryResp>(`/api/ai/summarize/${entity}/${id}`),
    enabled,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  return (
    <div className="rounded-2xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)] overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-[var(--color-border-row)] bg-[var(--color-bg-canvas)]">
        <span className="w-6 h-6 rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)] grid place-items-center">
          <Sparkles className="w-3.5 h-3.5" />
        </span>
        <span className="text-[13px] font-bold text-[var(--color-text-primary)] flex-1">
          {title}
        </span>
        {enabled && (
          <button
            type="button"
            onClick={() => void refetch()}
            disabled={isFetching}
            title="重新生成"
            className="w-7 h-7 grid place-items-center rounded-md text-[var(--color-text-tertiary)] hover:text-[var(--color-accent)] hover:bg-[var(--color-accent-soft)] transition-all disabled:opacity-40"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", isFetching && "animate-spin")} />
          </button>
        )}
      </div>

      <div className="px-4 py-3">
        {!enabled ? (
          <button
            type="button"
            onClick={() => setEnabled(true)}
            className="w-full h-9 rounded-lg border border-dashed border-[var(--color-accent)]/50 text-[12px] font-bold text-[var(--color-accent)] hover:bg-[var(--color-accent-soft)] active:scale-[0.99] transition-all flex items-center justify-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            生成 AI 摘要
          </button>
        ) : isFetching && !data ? (
          <div className="flex items-center gap-2 py-4 justify-center text-[12px] text-[var(--color-text-tertiary)]">
            <Loader2 className="w-4 h-4 animate-spin" />
            AI 正在汇总记录…
          </div>
        ) : isError ? (
          <div className="py-2 text-[12px] text-[var(--color-danger,#dc2626)]">
            摘要生成失败:
            {error instanceof Error ? error.message : "请稍后重试"}
          </div>
        ) : data ? (
          <div className="text-[13px] leading-relaxed text-[var(--color-text-primary)] [&_strong]:text-[var(--color-text-primary)]">
            <MarkdownContent text={data.summary} />
            <p className="mt-2 text-[10px] text-[var(--color-text-tertiary)]">
              由 AI 基于当前记录数据生成,仅供快速参考
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
