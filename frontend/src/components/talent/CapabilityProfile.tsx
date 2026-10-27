/** 职业能力精准画像 — 人级聚合视图(对标 Workday Skills Cloud 的 skill level
 * + 来源分解):把候选人散落的评估数据(人岗匹配逐维 / 评分卡逐维 / 技能档案)
 * 归一成 0-5 的稳定能力画像,每个维度可点开看"分是哪几次评估贡献的"。
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, Radar, Sparkles } from "lucide-react";

import { RadarChart } from "@/components/charts/RadarChart";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";

type Src = {
  kind: "match" | "interview" | "declared";
  label: string;
  score5: number;
  detail?: string | null;
  at?: string | null;
};
type Dim = { name: string; avg: number; min: number; max: number; count: number; sources: Src[] };
type Profile = {
  evaluations_total: number;
  sources_count: Record<string, number>;
  dimensions: Dim[];
  skills: string[];
  strengths: string[];
  gaps: string[];
};

const KIND_LABEL: Record<Src["kind"], string> = {
  match: "匹配评估",
  interview: "面试反馈",
  declared: "技能档案",
};

function bar(score5: number) {
  const pct = Math.round((score5 / 5) * 100);
  const tone =
    score5 >= 4 ? "bg-emerald-500" : score5 >= 3 ? "bg-[var(--color-accent)]" : score5 >= 2 ? "bg-amber-500" : "bg-red-400";
  return (
    <span className="inline-block h-1.5 w-16 rounded-full bg-[var(--color-bg-muted)] overflow-hidden align-middle">
      <span className={cn("block h-full rounded-full", tone)} style={{ width: `${pct}%` }} />
    </span>
  );
}

export function CapabilityProfile({ candidateId }: { candidateId: string }) {
  const [open, setOpen] = useState<string | null>(null);
  const { data } = useQuery<Profile>({
    queryKey: ["capability-profile", candidateId],
    queryFn: () => fetchJSON(`/api/talents/${candidateId}/capability-profile`),
    staleTime: 60_000,
  });
  if (!data) return null;

  const dims = data.dimensions;
  const total = data.evaluations_total;

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)] p-5 flex flex-col gap-4">
      <header className="flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-[var(--color-text-primary)]">
          <Radar className="w-4 h-4 text-[var(--color-accent)]" />
          能力画像
        </h3>
        {total > 0 && (
          <span className="text-[11px] text-[var(--color-text-tertiary)]">
            {total} 次评估 · {Object.entries(data.sources_count).filter(([, v]) => v > 0).map(([k, v]) => `${KIND_LABEL[k as Src["kind"]]} ${v}`).join(" / ")}
          </span>
        )}
      </header>

      {dims.length === 0 ? (
        <Empty
          icon={Radar}
          title="画像数据不足"
          description="完成一次人岗匹配评估或面试评分卡后,这里会聚合出稳定的能力画像"
        />
      ) : (
        <>
          <div className="flex flex-wrap items-start gap-6">
            <RadarChart
              dims={dims.map((d) => ({ name: d.name, value: d.avg / 5 }))}
              size={230}
              maxLabel={8}
            />
            <div className="flex-1 min-w-[240px] flex flex-col gap-2.5">
              {(data.strengths.length > 0 || data.gaps.length > 0) && (
                <div className="flex flex-wrap gap-1.5">
                  {data.strengths.map((s) => (
                    <span key={s} className="px-2 py-0.5 rounded-full text-[11px] bg-emerald-500/10 text-emerald-600">
                      ▲ {s}
                    </span>
                  ))}
                  {data.gaps.map((g) => (
                    <span key={g} className="px-2 py-0.5 rounded-full text-[11px] bg-amber-500/10 text-amber-600">
                      ▽ {g}
                    </span>
                  ))}
                </div>
              )}
              <ul className="flex flex-col divide-y divide-[var(--color-border-row)]">
                {dims.slice(0, 8).map((d) => (
                  <li key={d.name}>
                    <button
                      type="button"
                      onClick={() => setOpen(open === d.name ? null : d.name)}
                      className="w-full flex items-center gap-3 py-2 text-left group"
                    >
                      <ChevronDown
                        className={cn("w-3.5 h-3.5 text-[var(--color-text-tertiary)] transition-transform", open === d.name && "rotate-180")}
                      />
                      <span className="flex-1 min-w-0 truncate text-sm text-[var(--color-text-primary)] group-hover:text-[var(--color-accent)]">
                        {d.name}
                      </span>
                      {bar(d.avg)}
                      <span className="w-9 text-right font-mono text-xs text-[var(--color-text-primary)]">{d.avg.toFixed(1)}</span>
                      <span className="w-12 text-right text-[11px] text-[var(--color-text-tertiary)]">{d.count} 次</span>
                    </button>
                    {open === d.name && (
                      <ul className="pb-2 pl-7 flex flex-col gap-1.5">
                        {d.sources.map((s, i) => (
                          <li key={i} className="text-[11px] text-[var(--color-text-secondary)]">
                            <span className="inline-block px-1.5 py-px rounded bg-[var(--color-bg-muted)] mr-1.5 text-[10px]">
                              {KIND_LABEL[s.kind]}
                            </span>
                            <span className="font-medium">{s.label}</span>
                            <span className="ml-1.5 font-mono">{s.score5.toFixed(1)}</span>
                            {s.at && <span className="ml-1.5 text-[var(--color-text-tertiary)]">{formatRelative(s.at)}</span>}
                            {s.detail && <p className="mt-0.5 pl-1 text-[var(--color-text-tertiary)] leading-relaxed">{s.detail}</p>}
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>
          {data.skills.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-[var(--color-border-row)]">
              <Sparkles className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />
              {data.skills.slice(0, 20).map((s) => (
                <span key={s} className="px-2 py-0.5 rounded-full bg-[var(--color-bg-muted)] text-[11px] text-[var(--color-text-secondary)]">
                  {s}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
