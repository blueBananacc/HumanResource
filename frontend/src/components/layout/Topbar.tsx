import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { formatRoleLabel } from "@/lib/roles";

/**
 * 顶栏:页标题 + 通知 + 用户菜单。
 *
 * 搜索已下放到各个有列表的模块自身 (简历库 / 岗位 / 面试 / 题集),
 * 顶栏不再承担搜索入口,避免"看着是全局搜索、行为只搜简历库"的预期错位。
 *
 * - 通知:占位下拉,真实通知后端待 M6 实装,这里先反馈"暂无新通知"避免控件像坏的
 * - 用户菜单:展示邮箱 + 角色 + 登出。Sidebar 底部用同一交互逻辑
 */
export function Topbar({ title }: { title: string }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [openMenu, setOpenMenu] = useState<"none" | "bell" | "user">("none");

  const initials = user?.email.slice(0, 2).toUpperCase() ?? "—";

  return (
    <header className="h-14 bg-[var(--color-brand-deep)] flex items-center justify-between px-4 gap-4 shrink-0 relative z-20">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-8 h-8 rounded-[12px_12px_12px_3px] bg-[var(--color-accent)] text-white font-body font-bold text-sm flex items-center justify-center shrink-0">
          Hi
        </div>
        <span className="font-body font-bold text-base text-white shrink-0 tracking-tight">HROnfire</span>
        <span className="w-px h-5 bg-white/20 shrink-0" />
        <h1 className="font-heading text-[15px] text-white/90 truncate">
          {title}
        </h1>
      </div>
      <div className="flex items-center gap-2">
        {/* 铃铛 */}
        <div className="relative">
          <button
            type="button"
            onClick={() =>
              setOpenMenu((s) => (s === "bell" ? "none" : "bell"))
            }
            className="w-9 h-9 grid place-items-center rounded-full hover:bg-[var(--color-brand-hover)] transition-colors"
            aria-label="通知"
          >
            <Bell className="w-[18px] h-[18px] text-white" />
          </button>
          {openMenu === "bell" && (
            <Dropdown align="right">
              <div className="px-4 py-3 flex flex-col gap-2 min-w-[240px]">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                  通知
                </span>
                <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                  暂无新通知
                </span>
              </div>
            </Dropdown>
          )}
        </div>

        {/* 用户头像 */}
        <div className="relative">
          <button
            type="button"
            onClick={() =>
              setOpenMenu((s) => (s === "user" ? "none" : "user"))
            }
            className="h-8 pl-1 pr-3 rounded-full bg-white hover:bg-[var(--color-bg-subtle)] flex items-center gap-2 transition-colors"
            aria-label="个人信息"
          >
            <span className="w-6 h-6 rounded-full bg-[var(--color-accent)] text-white flex items-center justify-center text-[10px] font-heading font-semibold">
              {initials}
            </span>
            <span className="text-[13px] font-semibold text-[var(--color-accent)] font-body max-w-[140px] truncate">
              {user?.email.split("@")[0] ?? "—"}
            </span>
          </button>
          {openMenu === "user" && (
            <Dropdown align="right">
              <div className="px-4 py-3 flex flex-col gap-1 border-b border-[var(--color-border-subtle)] min-w-[220px]">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body truncate">
                  {user?.email ?? "未登录"}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body capitalize">
                  {formatRoleLabel(user?.role)}
                </span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setOpenMenu("none");
                  logout();
                  navigate("/login", { replace: true });
                }}
                className="flex items-center gap-2 w-full px-4 py-2.5 text-[13px] text-[var(--color-text-primary)] hover:bg-[var(--color-bg-muted)] font-body transition-colors"
              >
                <LogOut className="w-4 h-4" />
                登出
              </button>
            </Dropdown>
          )}
        </div>
      </div>

      {/* 点空白处关菜单 */}
      {openMenu !== "none" && (
        <div
          className="fixed inset-0 z-30"
          onClick={() => setOpenMenu("none")}
          aria-hidden
        />
      )}
    </header>
  );
}

function Dropdown({
  children,
  align = "right",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <div
      className={`absolute top-[calc(100%+6px)] ${
        align === "right" ? "right-0" : "left-0"
      } z-40 bg-white rounded-xl border border-[var(--color-border-subtle)] shadow-[0_8px_24px_rgba(15,17,21,0.08)] overflow-hidden`}
      onClick={(e) => e.stopPropagation()}
    >
      {children}
    </div>
  );
}
