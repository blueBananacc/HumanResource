/** 招聘早报 — 主动陪伴入口(对标 Moka 主动推进 / HireEasy 早报)。
 * 「没有实质更新绝不打扰」:has_items=false 时只渲染一条静默提示。
 * 每条目可直接跳转到对应处理页。记忆中枢学习进度同卡呈现。
 */
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Brain, CalendarCheck, FilePlus2, HandCoins, Hourglass, Sparkles } from "lucide-react";
import { fetchJSON } from "@/lib/api";

type BriefSection = {
  key: string;
  title: string;
  count: number;
  items: { id?: string; name?: string; candidate?: string; title?: string; job?: string; position?: string; stage?: string; status?: string; scheduled_at?: string }[];
  link: string;
};

type Briefing = {
  date: string;
  sections: BriefSection[];
  memory: {
    decisions_total: number;
    positive: number;
    negative: number;
    confidence: "low" | "medium" | "high";
    top_dimensions: { name: string; current_weight: number; suggested_weight: number; gap: number }[];
  };
  has_items: boolean;
};

const ICONS: Record<string, typeof FilePlus2> = {
  new_resumes: FilePlus2,
  today_interviews: CalendarCheck,
  pending_offers: HandCoins,
  stale_apps: Hourglass,
};

const CONF_LABEL = { low: "低置信度", medium: "中置信度", high: "高置信度" };
const CONF_CLASS = {
  low: "bg-slate-100 text-slate-500",
  medium: "bg-amber-50 text-amber-600",
  high: "bg-emerald-50 text-emerald-600",
};

export function BriefingCard() {
  const { data } = useQuery<Briefing>({
    queryKey: ["briefing-today"],
    queryFn: () => fetchJSON("/api/briefing/today"),
    staleTime: 120_000,
  });
  if (!data) return null;

  // 跳过逻辑:无任何实质更新时不打扰,只留一行静默提示
  if (!data.has_items) {
    return (
      <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-[var(--color-bg-muted)] border border-[var(--color-border-subtle)]">
        <Sparkles className="w-4 h-4 text-[var(--color-text-tertiary)]" />
        <span className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          招聘早报 · {data.date} — 暂无实质更新,不打扰你。
        </span>
        {data.memory.decisions_total > 0 && <MemoryLine m={data.memory} />}
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-[var(--color-bg-card)] border border-[var(--color-border-subtle)] shadow-[var(--shadow-card)] overflow-hidden">
      <div className="flex items-center gap-2 px-5 pt-4 pb-3">
        <div className="w-8 h-8 rounded-full bg-[var(--color-orange-tint)] grid place-items-center">
          <Sparkles className="w-4 h-4 text-[var(--color-orange)]" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-heading font-semibold text-[15px]">招聘早报</div>
          <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
            {data.date} · AI 主动巡检,跟进不等人
          </div>
        </div>
        {data.memory.decisions_total > 0 && <MemoryLine m={data.memory} />}
      </div>
      <div className="grid grid-cols-4 gap-3 px-5 pb-4">
        {data.sections
          .filter((s) => s.count > 0)
          .map((s) => {
            const Icon = ICONS[s.key] ?? FilePlus2;
            return (
              <div
                key={s.key}
                className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)] p-3 flex flex-col gap-1.5"
              >
                <div className="flex items-center gap-1.5">
                  <Icon className="w-3.5 h-3.5 text-[var(--color-accent)]" />
                  <span className="text-[12px] font-medium font-body">{s.title}</span>
                  <span className="ml-auto rounded-full bg-[var(--color-accent-soft)] px-1.5 text-[11px] font-mono text-[var(--color-accent)]">
                    {s.count}
                  </span>
                </div>
                <ul className="flex flex-col gap-0.5">
                  {s.items.slice(0, 3).map((it, i) => (
                    <li key={i} className="text-[11px] text-[var(--color-text-secondary)] font-body truncate">
                      {it.name ?? it.candidate}
                      {it.job ? ` · ${it.job}` : ""}
                      {it.title ? ` · ${it.title}` : ""}
                      {it.position ? ` · ${it.position}` : ""}
                      {it.status ? `(${it.status === "draft" ? "待发出" : "已发未回"})` : ""}
                      {it.stage ? `(${it.stage})` : ""}
                    </li>
                  ))}
                </ul>
                <Link
                  to={s.link}
                  className="mt-auto text-[11px] text-[var(--color-accent)] hover:underline font-body"
                >
                  去处理 →
                </Link>
              </div>
            );
          })}
      </div>
    </div>
  );
}

/** 记忆中枢徽标:「已从 N 次决策中学习 · 置信度」— H 的记忆呈现层级 */
function MemoryLine({ m }: { m: Briefing["memory"] }) {
  return (
    <Link
      to="/scoring-rules"
      className="flex items-center gap-1.5 rounded-full bg-[var(--color-bg-muted)] px-3 py-1 hover:bg-[var(--color-cappuccino)] transition-colors"
      title={`正向 ${m.positive} / 负向 ${m.negative}`}
    >
      <Brain className="w-3.5 h-3.5 text-[var(--color-accent)]" />
      <span className="text-[11px] text-[var(--color-text-secondary)] font-body">
        已从 {m.decisions_total} 次决策中学习
      </span>
      <span className={`rounded px-1 py-px text-[10px] font-body ${CONF_CLASS[m.confidence]}`}>
        {CONF_LABEL[m.confidence]}
      </span>
      {m.top_dimensions[0] && (
        <span className="text-[10px] text-[var(--color-text-tertiary)] font-body hidden lg:inline">
          {m.top_dimensions[0].name} {m.top_dimensions[0].current_weight}→{m.top_dimensions[0].suggested_weight}
        </span>
      )}
    </Link>
  );
}
