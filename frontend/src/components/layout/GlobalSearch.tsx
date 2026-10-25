import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight, BookOpen, BriefcaseBusiness, CalendarCheck, CornerDownLeft,
  LifeBuoy, Megaphone, Receipt, Search, User, Users, type LucideIcon,
} from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useVisibleNav } from "./nav";

type SearchHit = {
  entity: string; entity_label: string; id: string;
  title: string; sub: string | null; link: string; score: number;
};
type SearchOut = { items: SearchHit[]; facets: { entity: string; label: string; count: number }[]; total: number };
type Item = { key: string; icon: LucideIcon; label: string; sub?: string; to: string; group: string };

/** 实体 key → 图标(T9 全局搜索) */
const ENTITY_ICON: Record<string, LucideIcon> = {
  talents: User,
  jobs: BriefcaseBusiness,
  employees: Users,
  interviews: CalendarCheck,
  cases: LifeBuoy,
  requests: Users,
  knowledge: BookOpen,
  announcements: Megaphone,
  expenses: Receipt,
};

/** 顶栏搜索:快速跳转功能页 + 跨实体全局搜索(/api/search)。Ctrl(⌘)+K 聚焦。 */
export function GlobalSearch() {
  const navigate = useNavigate();
  const { pages: navPages } = useVisibleNav();
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  const [entity, setEntity] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const isMac = navigator.platform.toLowerCase().includes("mac");

  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(q.trim()), 200);
    return () => window.clearTimeout(t);
  }, [q]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const { data: result, isFetching } = useQuery({
    queryKey: ["global-search", debounced, entity],
    queryFn: () =>
      fetchJSON<SearchOut>(
        `/api/search?q=${encodeURIComponent(debounced)}&limit=14${entity ? `&entity=${entity}` : ""}`,
      ),
    enabled: open && debounced.length > 0,
    staleTime: 10_000,
    retry: false,
  });

  // chip 的 facet 源:记住最近一次未过滤(多实体)结果,否则选中单实体后 chip 行消失无法取消
  const lastFacets = useRef<{ q: string; list: { entity: string; label: string; count: number }[] }>({ q: "", list: [] });
  if (!entity && (result?.facets.length ?? 0) > 1) {
    lastFacets.current = { q: debounced, list: result!.facets };
  }
  const facets = entity && lastFacets.current.q === debounced ? lastFacets.current.list : (result?.facets ?? []);

  const items = useMemo<Item[]>(() => {
    const kw = q.trim().toLowerCase();
    const pages = navPages
      .filter((n) => !kw || `${n.label} ${n.keywords ?? ""}`.toLowerCase().includes(kw))
      .slice(0, kw ? 6 : 8)
      .map<Item>((n) => ({
        key: `p:${n.to}`,
        icon: n.icon,
        label: n.label,
        sub: n.module !== n.label ? n.module : undefined,
        to: n.to,
        group: "功能",
      }));
    const records: Item[] = kw
      ? (result?.items ?? []).map<Item>((h) => ({
          key: `r:${h.entity}:${h.id}`,
          icon: ENTITY_ICON[h.entity] ?? Search,
          label: h.title,
          sub: h.sub ?? undefined,
          to: h.link,
          group: h.entity_label,
        }))
      : [];
    return [...pages, ...records];
  }, [q, navPages, result]);

  // 列表末尾的"在人才库中搜索"作为额外一项,参与键盘导航
  const showMore = q.trim().length > 0;
  const total = items.length + (showMore ? 1 : 0);

  useEffect(() => {
    setActive(0);
    if (!q.trim()) setEntity(null);
  }, [q]);

  const go = (to: string) => {
    navigate(to);
    setOpen(false);
    setQ("");
    inputRef.current?.blur();
  };
  const choose = (i: number) =>
    i < items.length ? go(items[i].to) : go(`/talents?q=${encodeURIComponent(q.trim())}`);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => (a + 1) % Math.max(total, 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => (a - 1 + total) % Math.max(total, 1));
    } else if (e.key === "Enter" && total > 0) {
      e.preventDefault();
      choose(active);
    } else if (e.key === "Escape") {
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  // 分组顺序:功能优先,实体组按后端 facets 顺序(命中数降序)
  const facetOrder = useMemo(
    () => new Map((result?.facets ?? []).map((f, i) => [f.label, i])),
    [result],
  );
  const facetCount = useMemo(
    () => new Map((result?.facets ?? []).map((f) => [f.label, f.count])),
    [result],
  );
  const groupNames = useMemo(() => {
    const names = [...new Set(items.map((it) => it.group))];
    return names.sort((a, b) =>
      a === "功能" ? -1 : b === "功能" ? 1
        : (facetOrder.get(a) ?? 99) - (facetOrder.get(b) ?? 99));
  }, [items, facetOrder]);
  const groups = groupNames
    .map((g) => ({ g, rows: items.map((it, i) => ({ it, i })).filter(({ it }) => it.group === g) }))
    .filter(({ rows }) => rows.length > 0);

  return (
    <div className="relative w-full max-w-[440px]">
      <div
        className={cn(
          "flex items-center gap-2 h-9 px-3.5 rounded-full border transition-all duration-200 ease-[var(--ease-bob)]",
          open
            ? "bg-white border-white"
            : "bg-white/10 border-white/15 hover:bg-white/15",
        )}
      >
        <Search className={cn("w-4 h-4 shrink-0", open ? "text-[var(--color-text-tertiary)]" : "text-white/70")} />
        <input
          ref={inputRef}
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder="全局搜索:候选人、岗位、员工、工单…"
          aria-label="全局搜索"
          className={cn(
            "flex-1 min-w-0 bg-transparent text-[13px] font-body outline-none focus-visible:outline-none",
            open
              ? "text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)]"
              : "text-white placeholder:text-white/60",
          )}
        />
        {!open && (
          <kbd className="hidden lg:block text-[10px] font-body font-bold text-white/60 border border-white/20 rounded px-1.5 py-0.5">
            {isMac ? "⌘" : "Ctrl"} K
          </kbd>
        )}
      </div>

      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} aria-hidden />
          <div className="absolute top-[calc(100%+8px)] left-0 right-0 z-40 bg-white rounded-xl border border-[var(--color-border-subtle)] shadow-[0_8px_24px_rgba(58,58,55,0.14)] overflow-hidden font-body">
            {facets.length > 1 && (
              <div className="flex flex-wrap gap-1.5 px-3 pt-2.5 pb-1 border-b border-[var(--color-border-row)]">
                <button
                  type="button"
                  onClick={() => setEntity(null)}
                  className={cn(
                    "px-2.5 py-1 rounded-full text-[11px] font-bold transition-colors",
                    entity === null
                      ? "bg-[var(--color-accent)] text-white"
                      : "bg-[var(--color-bg-muted)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-subtle)]",
                  )}
                >
                  全部
                </button>
                {facets.map((f) => (
                  <button
                    key={f.entity}
                    type="button"
                    onClick={() => setEntity(entity === f.entity ? null : f.entity)}
                    className={cn(
                      "px-2.5 py-1 rounded-full text-[11px] font-bold transition-colors",
                      entity === f.entity
                        ? "bg-[var(--color-accent)] text-white"
                        : "bg-[var(--color-bg-muted)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-subtle)]",
                    )}
                  >
                    {f.label} {f.count}
                  </button>
                ))}
              </div>
            )}
            <div className="max-h-[420px] overflow-y-auto py-1.5">
              {groups.map(({ g, rows }) => (
                <div key={g} className="py-1">
                  <div className="px-4 pt-1 pb-1.5 text-[11px] font-bold tracking-[0.12em] text-[var(--color-text-tertiary)] flex items-center gap-1.5">
                    {g === "功能" && !q.trim() ? "快速跳转" : g}
                    {facetCount.has(g) && (
                      <span className="text-[10px] font-normal text-[var(--color-text-tertiary)]/70">
                        {facetCount.get(g)} 条
                      </span>
                    )}
                  </div>
                  {rows.map(({ it, i }) => (
                    <button
                      key={it.key}
                      type="button"
                      onMouseEnter={() => setActive(i)}
                      onClick={() => choose(i)}
                      className={cn(
                        "w-full flex items-center gap-3 px-4 py-2 text-left transition-colors",
                        active === i && "bg-[var(--color-bg-muted)]",
                      )}
                    >
                      <span
                        className={cn(
                          "w-7 h-7 rounded-full grid place-items-center shrink-0",
                          it.group === "功能"
                            ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                            : "bg-[var(--color-orange-tint)] text-[var(--color-text-primary)]",
                        )}
                      >
                        <it.icon className="w-3.5 h-3.5" />
                      </span>
                      <span className="flex flex-col min-w-0 flex-1">
                        <span className="text-[13px] font-bold text-[var(--color-text-primary)] truncate">{it.label}</span>
                        {it.sub && <span className="text-[11px] text-[var(--color-text-tertiary)] truncate">{it.sub}</span>}
                      </span>
                      {active === i && <CornerDownLeft className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />}
                    </button>
                  ))}
                </div>
              ))}

              {q.trim() && isFetching && !result && (
                <div className="px-4 py-2 text-[12px] text-[var(--color-text-tertiary)]">正在全局搜索…</div>
              )}
              {q.trim() && groups.length === 0 && !isFetching && (
                <div className="px-4 py-6 text-center text-[13px] text-[var(--color-text-tertiary)]">
                  没找到「{q.trim()}」相关的功能或记录
                </div>
              )}
            </div>

            {showMore && (
              <button
                type="button"
                onMouseEnter={() => setActive(items.length)}
                onClick={() => choose(items.length)}
                className={cn(
                  "w-full flex items-center gap-2 px-4 py-2.5 border-t border-[var(--color-border-row)] text-[12px] font-bold text-[var(--color-accent)] transition-colors",
                  active === items.length && "bg-[var(--color-accent-soft)]",
                )}
              >
                {(result?.total ?? 0) > 0 ? `全部 ${result!.total} 条结果 · ` : ""}在人才库中搜索「{q.trim()}」
                <ArrowRight className="w-3.5 h-3.5 ml-auto" />
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
