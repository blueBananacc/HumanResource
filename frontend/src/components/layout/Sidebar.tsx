import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ChevronRight, ChevronsLeft, ChevronsRight, PanelLeftClose } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  isItemActive,
  isModuleActive,
  moduleEntry,
  useVisibleNav,
  type NavItem,
  type NavModule,
} from "./nav";

const COLLAPSE_KEY = "hronfire.sidebar.collapsed";
const PANEL_KEY = "hronfire.sidebar.panel";

/**
 * 左侧导航 = 深色一级栏(模块) + 浅色二级面板(当前模块的分组与页面)。
 * 规则见 nav.ts 顶部注释与 docs/design/hibob-style-reference.md §7。
 * 帮助与支持入口在右侧工具栏底部(RightRail)。
 */
export function Sidebar() {
  const { main, bottom } = useVisibleNav();
  const loc = useLocation();
  const navigate = useNavigate();
  const search = new URLSearchParams(loc.search);

  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(COLLAPSE_KEY) === "1");
  const [panelOpen, setPanelOpenState] = useState(() => localStorage.getItem(PANEL_KEY) !== "0");
  const setPanelOpen = (v: boolean) => {
    localStorage.setItem(PANEL_KEY, v ? "1" : "0");
    setPanelOpenState(v);
  };
  const toggleCollapsed = () =>
    setCollapsed((v) => {
      localStorage.setItem(COLLAPSE_KEY, v ? "0" : "1");
      return !v;
    });

  const all = [...main, ...bottom];
  const active = all.find((m) => isModuleActive(m, loc.pathname, search));
  const panelModule = active?.groups ? active : undefined;
  const showPanel = !!panelModule && panelOpen;

  const onModuleClick = (m: NavModule) => {
    if (!m.groups) return;
    if (m === active) {
      setPanelOpen(!panelOpen);
      return;
    }
    const entry = moduleEntry(m);
    if (entry) navigate(entry);
    setPanelOpen(true);
  };

  const renderModule = (m: NavModule) => (
    <ModuleRow
      key={m.key}
      module={m}
      active={m === active}
      panelShown={m === active && showPanel}
      collapsed={collapsed}
      onClick={() => onModuleClick(m)}
    />
  );

  return (
    <div className="h-full flex shrink-0">
      <aside
        className={cn(
          "h-full bg-[var(--color-brand)] flex flex-col py-4 shrink-0 transition-[width] duration-300 ease-[var(--ease-bob)]",
          collapsed ? "w-[64px] px-2" : "w-[188px] px-3",
        )}
        aria-label="一级导航"
      >
        <nav className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col gap-0.5 -mx-1 px-1">
          {main.map(renderModule)}
        </nav>
        <div className="flex flex-col gap-0.5 pt-2 mt-2 border-t border-white/10">
          {bottom.map(renderModule)}
          <button
            type="button"
            onClick={toggleCollapsed}
            title={collapsed ? "展开导航" : "收起导航"}
            className={cn(
              "flex items-center gap-3 py-2 rounded-md text-[13px] font-body text-white/60 hover:text-white hover:bg-[var(--color-brand-hover)] transition-colors",
              collapsed ? "justify-center" : "px-3",
            )}
          >
            {collapsed ? (
              <ChevronsRight className="w-[18px] h-[18px]" />
            ) : (
              <>
                <ChevronsLeft className="w-[18px] h-[18px]" />
                收起导航
              </>
            )}
          </button>
        </div>
      </aside>

      {/* 二级面板:只在当前模块有分组时出现;宽度动画收放,内容固定宽度避免挤压换行 */}
      <div
        className={cn(
          "h-full overflow-hidden bg-[var(--color-bg-base)] border-[var(--color-border-subtle)] transition-[width] duration-300 ease-[var(--ease-bob)]",
          showPanel ? "w-[208px] border-r" : "w-0",
        )}
        aria-hidden={!showPanel}
      >
        {panelModule && (
          <SubPanel
            module={panelModule}
            pathname={loc.pathname}
            search={search}
            onClose={() => setPanelOpen(false)}
          />
        )}
      </div>
    </div>
  );
}

function ModuleRow({
  module: m,
  active,
  panelShown,
  collapsed,
  onClick,
}: {
  module: NavModule;
  active: boolean;
  panelShown: boolean;
  collapsed: boolean;
  onClick: () => void;
}) {
  const Icon = m.icon;
  const cls = cn(
    "relative flex items-center gap-3 py-2 rounded-md text-sm font-body transition-colors duration-200 w-full",
    collapsed ? "justify-center px-0" : "px-3",
    active
      ? "bg-[var(--color-brand-active)] text-white font-bold before:absolute before:left-[-8px] before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-full before:bg-[var(--color-orange)]"
      : "text-[var(--color-brand-text)] hover:bg-[var(--color-brand-hover)] hover:text-white",
  );
  const body = (
    <>
      <Icon className="w-[18px] h-[18px] shrink-0" />
      {!collapsed && <span className="truncate flex-1 text-left">{m.label}</span>}
      {!collapsed && m.groups && (
        <ChevronRight
          className={cn(
            "w-3.5 h-3.5 shrink-0 opacity-50 transition-transform duration-200",
            panelShown && "rotate-180 opacity-80",
          )}
        />
      )}
    </>
  );
  if (m.to) {
    return (
      <Link to={m.to} title={collapsed ? m.label : undefined} className={cls} aria-current={active ? "page" : undefined}>
        {body}
      </Link>
    );
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? m.label : active ? (panelShown ? "收起二级菜单" : "展开二级菜单") : undefined}
      aria-expanded={active ? panelShown : undefined}
      className={cls}
    >
      {body}
    </button>
  );
}

function SubPanel({
  module: m,
  pathname,
  search,
  onClose,
}: {
  module: NavModule;
  pathname: string;
  search: URLSearchParams;
  onClose: () => void;
}) {
  return (
    <div className="w-[208px] h-full flex flex-col" aria-label={`${m.label}二级导航`}>
      <div className="flex items-center justify-between pl-5 pr-2 pt-5 pb-3">
        <h2 className="font-heading font-semibold text-[17px] text-[var(--color-text-primary)] truncate">{m.label}</h2>
        <button
          type="button"
          onClick={onClose}
          title="收起二级菜单"
          className="p-1.5 rounded-md text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-muted)] transition-colors"
        >
          <PanelLeftClose className="w-4 h-4" />
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto px-3 pb-4 flex flex-col gap-4">
        {m.groups!.map((g, gi) => (
          <div key={g.label ?? gi} className="flex flex-col gap-0.5">
            {g.label && (
              <div className="px-2 pb-1 text-[11px] font-bold text-[var(--color-text-tertiary)] tracking-[0.08em] font-body">
                {g.label}
              </div>
            )}
            {g.items.map((it) => (
              <SubItem key={it.to} item={it} active={isItemActive(it, pathname, search)} />
            ))}
          </div>
        ))}
      </nav>
    </div>
  );
}

function SubItem({ item: it, active }: { item: NavItem; active: boolean }) {
  const Icon = it.icon;
  const inner = (
    <>
      {Icon && <Icon className="w-4 h-4 shrink-0" />}
      <span className="truncate flex-1">{it.label}</span>
      {it.soon && (
        <span className="shrink-0 rounded-full bg-[var(--color-bg-subtle)] px-1.5 py-px text-[10px] font-bold text-[var(--color-text-tertiary)]">
          规划
        </span>
      )}
    </>
  );
  const base = "relative flex items-center gap-2.5 px-2 py-[7px] rounded-md text-[13px] font-body transition-colors duration-150";
  if (it.soon) {
    return (
      <span className={cn(base, "text-[var(--color-text-tertiary)] cursor-not-allowed")} aria-disabled="true" title="规划中,尚未开放">
        {inner}
      </span>
    );
  }
  return (
    <Link
      to={it.to}
      aria-current={active ? "page" : undefined}
      className={cn(
        base,
        active
          ? "bg-[var(--color-bg-muted)] text-[var(--color-text-primary)] font-bold before:absolute before:left-0 before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-full before:bg-[var(--color-accent)]"
          : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)]",
      )}
    >
      {inner}
    </Link>
  );
}
