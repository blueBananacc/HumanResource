import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bell, LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { formatRoleLabel } from "@/lib/roles";
import { GlobalSearch } from "./GlobalSearch";

type AppNotification = {
  id: string;
  kind: string;
  title: string;
  body: string | null;
  link: string | null;
  created_at: string;
};

/**
 * 顶栏:页标题 + 全局搜索 + 通知 + 用户菜单。
 *
 * 全局搜索只做"快速跳转":功能页 + 候选人;列表内的筛选仍由各模块自身的搜索框负责。
 *
 * - 通知:/api/me/notifications 真实数据 —— 未读徽标(60s 轮询)+ 下拉,
 *   点击跳 link 并标记已读;工单/申请/watcher 的通知都会汇聚到这里
 * - 用户菜单:展示邮箱 + 角色 + 登出。Sidebar 底部用同一交互逻辑
 */
export function Topbar({ title }: { title: string }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [openMenu, setOpenMenu] = useState<"none" | "bell" | "user">("none");

  const { data: unread } = useQuery<{ count: number }>({
    queryKey: ["me-notif-unread-count"],
    queryFn: () => fetchJSON("/api/me/notifications/unread-count"),
    refetchInterval: 60_000,
  });
  const { data: notes } = useQuery<AppNotification[]>({
    queryKey: ["me-notif-dropdown"],
    queryFn: () => fetchJSON("/api/me/notifications?unread_only=true"),
    enabled: openMenu === "bell",
  });

  const invalidateNotifs = () => {
    qc.invalidateQueries({ queryKey: ["me-notif-unread-count"] });
    qc.invalidateQueries({ queryKey: ["me-notif-dropdown"] });
    qc.invalidateQueries({ queryKey: ["me-notifications-unread"] });
  };
  const markRead = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/notifications/${id}/read`, { method: "POST" }),
    onSuccess: invalidateNotifs,
  });
  const markAll = useMutation({
    mutationFn: () =>
      fetchJSON("/api/me/notifications/read-all", { method: "POST" }),
    onSuccess: invalidateNotifs,
  });

  function openNotification(n: AppNotification) {
    markRead.mutate(n.id);
    setOpenMenu("none");
    if (n.link) {
      const path = n.link.split("#")[0];
      navigate(path || "/");
    }
  }

  const initials = user?.email.slice(0, 2).toUpperCase() ?? "—";

  return (
    <header className="h-14 bg-[var(--color-brand-deep)] flex items-center justify-between px-4 gap-4 shrink-0 relative z-40">
      <div className="absolute left-1/2 -translate-x-1/2 w-[min(440px,40vw)] z-10">
        <GlobalSearch />
      </div>
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
        {/* 铃铛 —— 未读徽标 + 未读通知下拉 */}
        <div className="relative">
          <button
            type="button"
            onClick={() =>
              setOpenMenu((s) => (s === "bell" ? "none" : "bell"))
            }
            className="relative w-9 h-9 grid place-items-center rounded-full hover:bg-[var(--color-brand-hover)] transition-colors"
            aria-label="通知"
          >
            <Bell className="w-[18px] h-[18px] text-white" />
            {(unread?.count ?? 0) > 0 && (
              <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-[var(--color-danger)] text-white text-[10px] font-body font-semibold grid place-items-center">
                {unread!.count > 9 ? "9+" : unread!.count}
              </span>
            )}
          </button>
          {openMenu === "bell" && (
            <Dropdown align="right">
              <div className="px-4 py-3 flex flex-col gap-1 w-[320px]">
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                    通知
                  </span>
                  {(notes?.length ?? 0) > 0 && (
                    <button
                      type="button"
                      onClick={() => markAll.mutate()}
                      className="text-[11px] text-[var(--color-accent)] hover:underline font-body"
                    >
                      全部已读
                    </button>
                  )}
                </div>
                {(notes ?? []).length === 0 ? (
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body py-1">
                    暂无新通知
                  </span>
                ) : (
                  <div className="max-h-[360px] overflow-auto -mx-1">
                    {(notes ?? []).slice(0, 15).map((n) => (
                      <button
                        key={n.id}
                        type="button"
                        onClick={() => openNotification(n)}
                        className="w-full text-left px-1 py-2 rounded-md hover:bg-[var(--color-bg-muted)] transition-colors"
                      >
                        <div className="text-[12px] font-medium text-[var(--color-text-primary)] font-body truncate">
                          {n.title}
                        </div>
                        {n.body && (
                          <div className="text-[11px] text-[var(--color-text-secondary)] font-body truncate">
                            {n.body}
                          </div>
                        )}
                        <div className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                          {formatRelative(n.created_at)}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
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
