import { useState } from "react";
import { NavLink } from "react-router-dom";
import { ChevronsLeft, ChevronsRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useVisibleNav, type NavDef } from "./nav";

function NavRow({ to, label, icon: Icon, collapsed }: NavDef & { collapsed: boolean }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      title={collapsed ? label : undefined}
      className={({ isActive }) =>
        cn(
          "relative flex items-center gap-3 py-2 rounded-md text-sm font-body transition-colors duration-200",
          collapsed ? "justify-center px-0" : "px-3",
          isActive
            ? "bg-[var(--color-brand-active)] text-white font-bold before:absolute before:left-[-8px] before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-full before:bg-[var(--color-orange)]"
            : "text-[var(--color-brand-text)] hover:bg-[var(--color-brand-hover)] hover:text-white",
        )
      }
    >
      <Icon className="w-[18px] h-[18px] shrink-0" />
      {!collapsed && <span className="truncate">{label}</span>}
    </NavLink>
  );
}

const COLLAPSE_KEY = "hronfire.sidebar.collapsed";

// 帮助与支持入口已移到右侧工具栏底部(RightRail)。
export function Sidebar() {
  const { main, sys } = useVisibleNav();
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSE_KEY) === "1",
  );
  const toggleCollapsed = () =>
    setCollapsed((v) => {
      localStorage.setItem(COLLAPSE_KEY, v ? "0" : "1");
      return !v;
    });

  return (
    <aside
      className={cn(
        "h-full bg-[var(--color-brand)] flex flex-col py-4 shrink-0 transition-[width] duration-300 ease-[var(--ease-bob)]",
        collapsed ? "w-[64px] px-2" : "w-[220px] px-3",
      )}
    >
      <nav className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col gap-0.5 -mx-1 px-1">
        <SectionLabel collapsed={collapsed}>工作台</SectionLabel>
        {main.map((it) => (
          <NavRow key={it.to} {...it} collapsed={collapsed} />
        ))}
        <div className="h-4" />
        <SectionLabel collapsed={collapsed}>系统</SectionLabel>
        {sys.map((it) => (
          <NavRow key={it.to} {...it} collapsed={collapsed} />
        ))}
      </nav>

      <button
        type="button"
        onClick={toggleCollapsed}
        title={collapsed ? "展开导航" : "收起导航"}
        className={cn(
          "flex items-center gap-3 py-2 mt-2 pt-3 border-t border-white/10 rounded-md text-[13px] font-body text-white/60 hover:text-white hover:bg-[var(--color-brand-hover)] transition-colors",
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
    </aside>
  );
}

function SectionLabel({ collapsed, children }: { collapsed: boolean; children: string }) {
  return collapsed ? (
    <div className="mx-3 my-2 h-px bg-white/15" />
  ) : (
    <div className="px-3 py-1 text-[11px] font-bold text-white/50 tracking-[0.12em] uppercase font-body">
      {children}
    </div>
  );
}
