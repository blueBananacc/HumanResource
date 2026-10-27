/** 人才地图(对标 Moka Mapping 的库内版):
 * 组织级人才分布视图——阶段/来源/技能标签/岗位分布 + 30 天增长。
 * 纯 CSS 条形图,不引图表库。
 */
import { useQuery } from "@tanstack/react-query";
import { Map, TrendingUp, Users } from "lucide-react";
import { fetchJSON } from "@/lib/api";

type TalentMap = {
  candidates_total: number;
  growth_30d: number;
  by_stage: { stage: string; count: number }[];
  by_source: { source: string; count: number }[];
  top_tags: { tag: string; count: number }[];
  by_job: { title: string; count: number }[];
  avg_match_score: number | null;
};

const STAGE_LABEL: Record<string, string> = {
  applied: "已投递",
  screening: "筛选中",
  interview: "面试中",
  offer: "Offer",
  hired: "已录用",
  rejected: "已淘汰",
  withdrawn: "已撤回",
};
const STAGE_ORDER = ["applied", "screening", "interview", "offer", "hired", "rejected", "withdrawn"];

function Bar({ label, count, max, hint }: { label: string; count: number; max: number; hint?: string }) {
  const pct = max > 0 ? Math.max(3, (count / max) * 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 truncate text-[12px] text-[var(--color-text-secondary)] font-body" title={hint ?? label}>
        {label}
      </span>
      <div className="flex-1 h-4 rounded bg-[var(--color-bg-subtle)] overflow-hidden">
        <div className="h-full rounded bg-[var(--color-accent)]/70" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right font-mono text-[12px] text-[var(--color-text-primary)]">{count}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">{title}</h3>
      {children}
    </div>
  );
}

export function TalentMap() {
  const { data } = useQuery<TalentMap>({
    queryKey: ["talent-map"],
    queryFn: () => fetchJSON("/api/reports/talent-map"),
    staleTime: 60_000,
  });
  if (!data) return <main className="p-8 text-[var(--color-text-tertiary)] font-body">加载中…</main>;

  const stages = [...data.by_stage].sort(
    (a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage),
  );

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <header className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-full bg-[var(--color-accent-soft)] grid place-items-center">
          <Map className="w-4.5 h-4.5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">人才地图</h2>
          <p className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            库内人才结构一屏览 — 人堆在哪、来自哪里、技能覆盖如何
          </p>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <div className="text-right">
            <div className="font-mono font-semibold text-xl text-[var(--color-text-primary)]">{data.candidates_total}</div>
            <div className="text-[10px] text-[var(--color-text-tertiary)] font-body flex items-center gap-1">
              <Users className="w-3 h-3" /> 人才库总数
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono font-semibold text-xl text-emerald-600">+{data.growth_30d}</div>
            <div className="text-[10px] text-[var(--color-text-tertiary)] font-body flex items-center gap-1">
              <TrendingUp className="w-3 h-3" /> 30 天新增
            </div>
          </div>
          {data.avg_match_score != null && (
            <div className="text-right">
              <div className="font-mono font-semibold text-xl text-[var(--color-text-primary)]">{data.avg_match_score}</div>
              <div className="text-[10px] text-[var(--color-text-tertiary)] font-body">平均匹配分</div>
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-2 gap-4">
        <Section title="管道阶段分布">
          {stages.length === 0 && <Empty text="暂无申请记录" />}
          {stages.map((s) => (
            <Bar
              key={s.stage}
              label={STAGE_LABEL[s.stage] ?? s.stage}
              count={s.count}
              max={Math.max(...stages.map((x) => x.count))}
            />
          ))}
        </Section>
        <Section title="简历来源分布">
          {data.by_source.length === 0 && <Empty text="暂无简历" />}
          {data.by_source.map((s) => (
            <Bar
              key={s.source}
              label={s.source}
              count={s.count}
              max={Math.max(...data.by_source.map((x) => x.count))}
            />
          ))}
        </Section>
        <Section title="技能标签 Top15(人才库能力密度)">
          {data.top_tags.length === 0 && <Empty text="候选人暂无标签" />}
          <div className="flex flex-wrap gap-2">
            {data.top_tags.map((t, i) => (
              <span
                key={t.tag}
                className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)] px-2.5 py-1 text-[12px] font-body"
                style={{ fontSize: `${Math.max(11, 15 - i * 0.5)}px`, opacity: 1 - i * 0.04 }}
              >
                {t.tag}
                <span className="ml-1 font-mono text-[10px] text-[var(--color-text-tertiary)]">{t.count}</span>
              </span>
            ))}
          </div>
        </Section>
        <Section title="岗位候选分布 Top10">
          {data.by_job.length === 0 && <Empty text="暂无岗位申请" />}
          {data.by_job.map((j) => (
            <Bar
              key={j.title}
              label={j.title}
              count={j.count}
              max={Math.max(...data.by_job.map((x) => x.count))}
            />
          ))}
        </Section>
      </div>
    </main>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="text-[12px] text-[var(--color-text-tertiary)] font-body">{text}</p>;
}
