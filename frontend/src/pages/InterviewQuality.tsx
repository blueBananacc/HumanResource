import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Sparkles,
  BarChart3,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { Select } from "@/components/ui/select";
import { fetchJSON } from "@/lib/api";
import { askAssistant } from "@/components/layout/AiAssistant";
import { cn } from "@/lib/utils";

type Kpis = {
  interviews_total: number;
  interviews_delta: string | null;
  dimension_coverage: number;
  pass_rate: number;
  avg_duration_min: number | null;
  assessed_events: number;
  decided: number;
};
type DimRow = { name: string; coverage: number; scored_in: number; avg_score: number | null };
type JobRow = { job_id: string; title: string; interviews: number; pass_rate: number };
type Report = {
  kind: string;
  label: string;
  generated_at: string;
  data: {
    days: number;
    kpis: Kpis;
    dimensions: DimRow[];
    missed: DimRow[];
    by_job: JobRow[];
  };
};

const DAYS_OPTIONS = [
  { value: "7", label: "近 7 天" },
  { value: "30", label: "近 30 天" },
  { value: "90", label: "近 90 天" },
];

export function InterviewQuality() {
  const [params, setParams] = useSearchParams();
  const days = params.get("days") ?? "30";

  const q = useQuery<Report>({
    queryKey: ["interview-quality", days],
    queryFn: () =>
      fetchJSON(`/api/report-defs/interview_quality/run`, {
        method: "POST",
        body: JSON.stringify({ params: { days: Number(days) } }),
      }),
  });
  const d = q.data?.data;

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            面试质量洞察
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            组织面试质量报告:考察项覆盖、通过率与时长 — 让识人成为组织能力
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() =>
              askAssistant(`帮我解读近 ${days} 天的组织面试质量报告,哪些考察项经常被遗漏,有什么改进建议`)
            }
            className="h-8 px-3 rounded-md border border-[var(--color-border-subtle)] text-[12px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-canvas)] inline-flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-[var(--color-accent)]" />
            让 AI 解读
          </button>
          <Select
            value={days}
            onChange={(v) => setParams({ days: v })}
            options={DAYS_OPTIONS}
            className="w-[120px]"
          />
        </div>
      </div>

      {q.isLoading ? (
        <Skeleton h={360} />
      ) : !d || d.kpis.assessed_events === 0 ? (
        <EmptyHint text="所选范围内暂无已完成的面试评估数据" />
      ) : (
        <>
          {/* KPI 卡 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard
              label="面试总量"
              value={`${d.kpis.interviews_total}场`}
              delta={d.kpis.interviews_delta}
            />
            <KpiCard label="维度覆盖率" value={`${d.kpis.dimension_coverage}%`} />
            <KpiCard label="终面通过率" value={`${d.kpis.pass_rate}%`} />
            <KpiCard
              label="平均面试时长"
              value={d.kpis.avg_duration_min != null ? `${d.kpis.avg_duration_min}min` : "—"}
            />
          </div>

          {/* 维度覆盖情况 */}
          <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-[var(--color-bg-subtle)] flex items-center justify-center shrink-0">
                <BarChart3 className="w-4 h-4 text-[var(--color-accent)]" />
              </div>
              <div className="flex flex-col gap-0.5 flex-1">
                <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
                  维度覆盖情况
                </h3>
                <p className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  哪些考察项经常被遗漏(覆盖 &lt;50% 标红)
                </p>
              </div>
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                {d.kpis.assessed_events} 场评估
              </span>
            </div>
            <div className="flex flex-col gap-2.5">
              {d.dimensions.map((dim) => (
                <DimBar key={dim.name} dim={dim} />
              ))}
            </div>
          </section>

          {/* 经常被遗漏 */}
          {d.missed.length > 0 && (
            <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-[var(--color-warning,#d97706)]" />
                <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
                  经常被遗漏的考察项
                </h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {d.missed.map((m) => (
                  <span
                    key={m.name}
                    className="text-[12px] rounded-full px-3 py-1 bg-[var(--color-danger-soft,#fef2f2)] text-[var(--color-danger,#dc2626)] font-body"
                  >
                    {m.name} · {m.coverage}%
                  </span>
                ))}
              </div>
              <p className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                建议:在评分卡 criteria 或题库中补上这些考察维度,或让 AI
                助手帮你生成对应评分卡。
              </p>
            </section>
          )}

          {/* 岗位下钻 */}
          {d.by_job.length > 0 && (
            <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-[var(--color-accent)]" />
                <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
                  岗位下钻
                </h3>
              </div>
              <div className="flex flex-col">
                <div className="grid grid-cols-[1fr_80px_80px] gap-3 text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider px-1 py-2 border-b border-[var(--color-border-row)] font-body">
                  <span>岗位</span>
                  <span className="text-right">面试数</span>
                  <span className="text-right">通过率</span>
                </div>
                {d.by_job.map((j) => (
                  <div
                    key={j.job_id}
                    className="grid grid-cols-[1fr_80px_80px] gap-3 py-2 border-b border-[var(--color-border-row)] last:border-b-0 text-[12px] font-body"
                  >
                    <span className="text-[var(--color-text-primary)] truncate">
                      {j.title || "未关联岗位"}
                    </span>
                    <span className="text-right font-mono text-[var(--color-text-secondary)]">
                      {j.interviews}
                    </span>
                    <span className="text-right font-mono text-[var(--color-text-primary)]">
                      {j.pass_rate}%
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </main>
  );
}

function KpiCard({
  label,
  value,
  delta,
}: {
  label: string;
  value: string;
  delta?: string | null;
}) {
  const up = delta?.startsWith("+");
  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-4 flex flex-col gap-1">
      <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
        {label}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          {value}
        </span>
        {delta && (
          <span
            className={cn(
              "text-[11px] font-mono inline-flex items-center gap-0.5",
              up ? "text-[var(--color-success,#16a34a)]" : "text-[var(--color-danger,#dc2626)]",
            )}
          >
            {up ? (
              <TrendingUp className="w-3 h-3" />
            ) : (
              <TrendingDown className="w-3 h-3" />
            )}
            {delta}
          </span>
        )}
      </div>
    </div>
  );
}

function DimBar({ dim }: { dim: DimRow }) {
  const color =
    dim.coverage >= 80
      ? "var(--color-success,#16a34a)"
      : dim.coverage >= 50
        ? "var(--color-warning,#d97706)"
        : "var(--color-danger,#dc2626)";
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-[12px] text-[var(--color-text-primary)] font-body truncate">
        {dim.name}
      </span>
      <div className="flex-1 h-5 bg-[var(--color-bg-subtle)] rounded-md overflow-hidden">
        <div
          className="h-full rounded-md transition-all"
          style={{ width: `${dim.coverage}%`, background: color }}
        />
      </div>
      <span className="w-14 shrink-0 text-right text-[12px] font-mono text-[var(--color-text-secondary)]">
        {dim.coverage}%
      </span>
      <span className="w-16 shrink-0 text-right text-[11px] font-body text-[var(--color-text-tertiary)]">
        {dim.scored_in}场
        {dim.avg_score != null && ` · ${dim.avg_score}分`}
      </span>
    </div>
  );
}

function Skeleton({ h }: { h: number }) {
  return (
    <div
      className="bg-[var(--color-bg-subtle)] rounded-md animate-pulse"
      style={{ height: h }}
    />
  );
}

function EmptyHint({ text }: { text: string }) {
  return (
    <div className="text-center py-16 text-sm text-[var(--color-text-tertiary)] font-body">
      {text}
    </div>
  );
}
