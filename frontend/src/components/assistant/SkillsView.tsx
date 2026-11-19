/** 助手内嵌「AI 技能中心」视图 — 技能卡就地预填到输入框;
 * 含 MCP 生态就绪度(紧凑版)与技能图谱,与原 /skills 页同一数据源。 */
import { useQuery } from "@tanstack/react-query";
import { Mail, Plug } from "lucide-react";
import { Link } from "react-router-dom";
import { fetchJSON } from "@/lib/api";
import { SKILLS } from "@/pages/SkillCenter";
import { SkillGraphPanel } from "@/components/skills/SkillGraphPanel";

export function SkillsView({ onPick }: { onPick: (prompt: string) => void }) {
  const { data: tokens } = useQuery<
    { id: string; revoked_at: string | null; last_used_at: string | null }[]
  >({
    queryKey: ["mcp-tokens"],
    queryFn: () => fetchJSON("/api/me/mcp-tokens"),
    staleTime: 60_000,
  });
  const activeTokens = tokens?.filter((t) => !t.revoked_at) ?? [];
  const usedTokens = activeTokens.filter((t) => t.last_used_at);
  // 就绪度:端点可达(40)+ 已签发令牌(30)+ 被外部调用过(30),与页面版同口径
  const ecoScore =
    40 + (activeTokens.length > 0 ? 30 : 0) + (usedTokens.length > 0 ? 30 : 0);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
      <p className="text-[12px] text-[var(--color-text-secondary)] font-body">
        点选技能即用 — 预填到下方输入框;写操作仍需你在对话内确认。
      </p>

      <div className="grid grid-cols-2 gap-2">
        {SKILLS.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => onPick(s.prompt)}
            className="group text-left rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-card)] p-3 hover:border-[var(--color-accent)] hover:shadow-[var(--shadow-card)] transition-all flex flex-col gap-1.5"
          >
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-[var(--color-accent-soft)] grid place-items-center shrink-0 group-hover:bg-[var(--color-orange-tint)] transition-colors">
                <s.icon className="w-3.5 h-3.5 text-[var(--color-accent)]" />
              </div>
              <span className="font-heading font-medium text-[12px] truncate">
                {s.title}
              </span>
              <span className="ml-auto rounded bg-[var(--color-bg-subtle)] px-1 py-px text-[10px] text-[var(--color-text-tertiary)] font-body shrink-0">
                {s.tag}
              </span>
            </div>
            <p className="text-[11px] text-[var(--color-text-secondary)] font-body leading-snug">
              {s.desc}
            </p>
          </button>
        ))}
      </div>

      <section className="rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-card)] px-4 py-3 flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <Plug className="w-4 h-4 text-[var(--color-accent)]" />
          <span className="font-heading font-semibold text-[13px] text-[var(--color-text-primary)]">
            开放生态 · MCP
          </span>
          <span className="ml-auto font-mono text-[13px] font-semibold text-[var(--color-text-primary)]">
            {ecoScore}
          </span>
        </div>
        <div className="flex flex-col gap-0.5 text-[11px] text-[var(--color-text-tertiary)] font-body">
          <span className="font-mono truncate">
            {location.origin}/api/mcp 已就绪
          </span>
          <span>
            {activeTokens.length} 个令牌生效
            {usedTokens.length > 0 ? ` · ${usedTokens.length} 个有外部调用` : " · 等待首次调用"}
          </span>
          <Link
            to="/settings/mcp"
            className="text-[var(--color-accent)] hover:underline"
          >
            设置 → MCP 令牌
          </Link>
          <span className="inline-flex items-center gap-1">
            <Mail className="w-3 h-3" />
            邮箱拉取见
            <Link to="/settings/email" className="text-[var(--color-accent)] hover:underline">
              设置 → 邮箱
            </Link>
          </span>
        </div>
      </section>

      <SkillGraphPanel />
    </div>
  );
}
