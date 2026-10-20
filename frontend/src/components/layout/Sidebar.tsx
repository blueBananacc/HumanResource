import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  FileText,
  Briefcase,
  Video,
  ChartBar,
  KeyRound,
  Mail,
  Mic,
  Plug,
  Send,
  ShieldCheck,
  Shuffle,
  Sparkles,
  BookOpen,
  TrendingUp,
  Users,
  ChevronsLeft,
  ChevronsRight,
  CircleHelp,
  Headset,
  Rocket,
  ExternalLink,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";
import { PERM, type AppPermission } from "@/lib/roles";

// 帮助菜单链接。私有化部署时可替换为客户自己的客服 / 文档地址。
const REPO = "https://github.com/blueBananacc/HumanResource";
const helpLinks: { label: string; desc: string; href: string; icon: LucideIcon }[] = [
  {
    label: "使用手册",
    desc: "功能说明与常见操作",
    href: `${REPO}/blob/main/docs/ops/customer-guide.md`,
    icon: BookOpen,
  },
  {
    label: "快速入门",
    desc: "部署与首次配置",
    href: `${REPO}/blob/main/docs/ops/quickstart.md`,
    icon: Rocket,
  },
  {
    label: "联系客服",
    desc: "提交问题或建议",
    href: `${REPO}/issues`,
    icon: Headset,
  },
];

type NavDef = {
  to: string;
  label: string;
  icon: LucideIcon;
  /** 仅 admin 可见(快捷:相当于 require_role('admin'))。 */
  adminOnly?: boolean;
  /** 需要至少一个该权限位才显示。优先级低于 adminOnly。 */
  permission?: AppPermission;
};

const mainNav: NavDef[] = [
  { to: "/", label: "概览", icon: LayoutDashboard },
  { to: "/resumes", label: "简历库", icon: FileText },
  { to: "/talents", label: "人才库", icon: Users },
  { to: "/jobs", label: "岗位", icon: Briefcase },
  { to: "/question-sets", label: "面试题集", icon: Sparkles },
  { to: "/question-library", label: "题库", icon: BookOpen },
  { to: "/interviews", label: "面试", icon: Video, permission: PERM.VIEW_REPORTS },
  { to: "/reports", label: "报告", icon: ChartBar, permission: PERM.VIEW_REPORTS },
  { to: "/analytics", label: "数据分析", icon: TrendingUp, permission: PERM.VIEW_REPORTS },
];

// 系统设置类菜单 —— 默认需要 manage_settings(只有 admin 持有)。
// adminOnly 单独标的项是更严格的"硬限 admin"(SSO / 审计)。
const sysNav: NavDef[] = [
  { to: "/settings/llm", label: "LLM 配置", icon: Plug, permission: PERM.MANAGE_SETTINGS },
  { to: "/settings/email", label: "邮箱拉取", icon: Mail, permission: PERM.MANAGE_SETTINGS },
  { to: "/settings/smtp", label: "SMTP 发件", icon: Send, permission: PERM.MANAGE_SETTINGS },
  { to: "/settings/voice", label: "语音 STT/TTS", icon: Mic, permission: PERM.MANAGE_SETTINGS },
  { to: "/settings/license", label: "License 设置", icon: KeyRound, permission: PERM.MANAGE_SETTINGS },
  { to: "/settings/team", label: "团队", icon: Users, permission: PERM.MANAGE_TEAM },
  { to: "/settings/sso", label: "SSO 接入", icon: Shuffle, adminOnly: true },
  { to: "/settings/audit", label: "审计中心", icon: ShieldCheck, adminOnly: true },
];

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

export function Sidebar() {
  const { user, hasPermission } = useAuth();
  const [openMenu, setOpenMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSE_KEY) === "1",
  );
  const toggleCollapsed = () =>
    setCollapsed((v) => {
      localStorage.setItem(COLLAPSE_KEY, v ? "0" : "1");
      return !v;
    });
  const isAdmin = user?.role === "admin";

  const filterNav = (items: NavDef[]) =>
    items.filter((it) => {
      if (it.adminOnly) return isAdmin;
      if (it.permission) return hasPermission(it.permission);
      return true;
    });

  const visibleMainNav = filterNav(mainNav);
  const visibleSysNav = filterNav(sysNav);

  return (
    <aside
      className={cn(
        "h-full bg-[var(--color-brand)] flex flex-col py-4 shrink-0 transition-[width] duration-300 ease-[var(--ease-bob)]",
        collapsed ? "w-[64px] px-2" : "w-[220px] px-3",
      )}
    >
      <nav className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col gap-0.5 -mx-1 px-1">
        <SectionLabel collapsed={collapsed}>工作台</SectionLabel>
        {visibleMainNav.map((it) => (
          <NavRow key={it.to} {...it} collapsed={collapsed} />
        ))}
        <div className="h-4" />
        <SectionLabel collapsed={collapsed}>系统</SectionLabel>
        {visibleSysNav.map((it) => (
          <NavRow key={it.to} {...it} collapsed={collapsed} />
        ))}
      </nav>

      <button
        type="button"
        onClick={toggleCollapsed}
        title={collapsed ? "展开导航" : "收起导航"}
        className={cn(
          "flex items-center gap-3 py-2 mt-2 rounded-md text-[13px] font-body text-white/60 hover:text-white hover:bg-[var(--color-brand-hover)] transition-colors",
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

      {/* 帮助入口 — 账号信息已在顶栏右上角,这里只放帮助与支持 */}
      <div className="relative shrink-0 mt-1 pt-2 border-t border-white/10">
        <button
          type="button"
          onClick={() => setOpenMenu((v) => !v)}
          title={collapsed ? "帮助与支持" : undefined}
          aria-label="帮助与支持"
          aria-expanded={openMenu}
          className={cn(
            "flex items-center gap-3 py-2 rounded-md w-full text-[13px] font-body text-white/80 hover:text-white hover:bg-[var(--color-brand-hover)] transition-colors",
            collapsed ? "justify-center px-0" : "px-3",
            openMenu && "bg-[var(--color-brand-active)] text-white",
          )}
        >
          <CircleHelp className="w-[18px] h-[18px] shrink-0" />
          {!collapsed && "帮助与支持"}
        </button>
        {openMenu && (
          <div
            className="absolute bottom-[calc(100%+6px)] left-0 w-[240px] z-40 bg-white rounded-xl border border-[var(--color-border-subtle)] shadow-[0_8px_24px_rgba(58,58,55,0.14)] overflow-hidden py-1.5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-4 pt-1.5 pb-2 text-[11px] font-bold tracking-[0.12em] text-[var(--color-text-tertiary)] font-body">
              帮助与支持
            </div>
            {helpLinks.map(({ label, desc, href, icon: Icon }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setOpenMenu(false)}
                className="group flex items-center gap-3 px-4 py-2.5 hover:bg-[var(--color-bg-muted)] transition-colors"
              >
                <span className="w-8 h-8 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] grid place-items-center shrink-0">
                  <Icon className="w-4 h-4" />
                </span>
                <span className="flex flex-col flex-1 min-w-0">
                  <span className="text-[13px] font-bold text-[var(--color-text-primary)] font-body">
                    {label}
                  </span>
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    {desc}
                  </span>
                </span>
                <ExternalLink className="w-3.5 h-3.5 text-[var(--color-text-tertiary)] opacity-0 group-hover:opacity-100 transition-opacity" />
              </a>
            ))}
          </div>
        )}
      </div>

      {openMenu && (
        <div
          className="fixed inset-0 z-30"
          onClick={() => setOpenMenu(false)}
          aria-hidden
        />
      )}
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
