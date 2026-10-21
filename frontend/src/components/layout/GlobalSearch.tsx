import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CornerDownLeft, Search, User, type LucideIcon } from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { cn } from "@/lib/utils";
import type { Page } from "@/lib/usePagedQuery";
import { useVisibleNav } from "./nav";

type TalentHit = { id: string; name: string; display_email: string | null; display_phone: string | null };
type Item = { key: string; icon: LucideIcon; label: string; sub?: string; to: string; group: "功能" | "候选人" };

/** 顶栏搜索:快速跳转功能页 / 搜候选人。Ctrl(⌘)+K 聚焦。 */
export function GlobalSearch() {
  const navigate = useNavigate();
  const { hasPermission } = useAuth();
  const canSeeTalents = hasPermission(PERM.VIEW_REPORTS);
  const { main, sys } = useVisibleNav();
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
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

  const { data: talents, isFetching } = useQuery({
    queryKey: ["global-search-talents", debounced],
    queryFn: () =>
      fetchJSON<Page<TalentHit>>(`/api/talents?limit=6&offset=0&q=${encodeURIComponent(debounced)}`),
    enabled: open && canSeeTalents && debounced.length > 0,
    staleTime: 10_000,
  });

  const items = useMemo<Item[]>(() => {
    const kw = q.trim().toLowerCase();
    const pages = [...main, ...sys]
      .filter((n) => !kw || `${n.label} ${n.keywords ?? ""}`.toLowerCase().includes(kw))
      .slice(0, kw ? 6 : 8)
      .map<Item>((n) => ({ key: `p:${n.to}`, icon: n.icon, label: n.label, to: n.to, group: "功能" }));
    const people =
      kw && canSeeTalents
        ? (talents?.items ?? []).map<Item>((t) => ({
            key: `t:${t.id}`,
            icon: User,
            label: t.name,
            sub: t.display_email ?? t.display_phone ?? undefined,
            to: `/talents/${t.id}`,
            group: "候选人",
          }))
        : [];
    return [...pages, ...people];
  }, [q, main, sys, talents, canSeeTalents]);

  // 列表末尾的"在人才库中搜索"作为额外一项,参与键盘导航
  const showMore = canSeeTalents && q.trim().length > 0;
  const total = items.length + (showMore ? 1 : 0);

  useEffect(() => setActive(0), [q]);

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

  const groups = (["功能", "候选人"] as const)
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
          placeholder="搜索候选人或功能…"
          aria-label="全局搜索"
          className={cn(
            "flex-1 min-w-0 bg-transparent text-[13px] font-body outline-none focus-visible:outline-none",
            open
              ? "text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)]"
              : "text-white placeholder:text-white/60",
          )}
        />
        {!open && (
          <kbd className="text-[10px] font-body font-bold text-white/60 border border-white/20 rounded px-1.5 py-0.5">
            {isMac ? "⌘" : "Ctrl"} K
          </kbd>
        )}
      </div>

      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} aria-hidden />
          <div className="absolute top-[calc(100%+8px)] left-0 right-0 z-40 bg-white rounded-xl border border-[var(--color-border-subtle)] shadow-[0_8px_24px_rgba(58,58,55,0.14)] overflow-hidden font-body">
            <div className="max-h-[420px] overflow-y-auto py-1.5">
              {groups.map(({ g, rows }) => (
                <div key={g} className="py-1">
                  <div className="px-4 pt-1 pb-1.5 text-[11px] font-bold tracking-[0.12em] text-[var(--color-text-tertiary)]">
                    {g === "功能" && !q.trim() ? "快速跳转" : g}
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
                          it.group === "候选人"
                            ? "bg-[var(--color-orange-tint)] text-[var(--color-text-primary)]"
                            : "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
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

              {q.trim() && canSeeTalents && isFetching && !talents && (
                <div className="px-4 py-2 text-[12px] text-[var(--color-text-tertiary)]">正在搜索候选人…</div>
              )}
              {q.trim() && groups.length === 0 && !isFetching && (
                <div className="px-4 py-6 text-center text-[13px] text-[var(--color-text-tertiary)]">
                  没找到「{q.trim()}」相关的功能或候选人
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
                在人才库中搜索「{q.trim()}」
                <ArrowRight className="w-3.5 h-3.5 ml-auto" />
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
