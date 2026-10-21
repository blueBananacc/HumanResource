import { useState } from "react";
import {
  BookOpen,
  CircleHelp,
  ExternalLink,
  Headset,
  Rocket,
  Settings,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SettingsModal } from "./SettingsModal";
import { AiAvatar } from "./AiAssistant";

// 帮助菜单链接。私有化部署时可替换为客户自己的客服 / 文档地址。
const REPO = "https://github.com/blueBananacc/HumanResource";
const helpLinks: { label: string; desc: string; href: string; icon: LucideIcon }[] = [
  { label: "使用手册", desc: "功能说明与常见操作", href: `${REPO}/blob/main/docs/ops/customer-guide.md`, icon: BookOpen },
  { label: "快速入门", desc: "部署与首次配置", href: `${REPO}/blob/main/docs/ops/quickstart.md`, icon: Rocket },
  { label: "联系客服", desc: "提交问题或建议", href: `${REPO}/issues`, icon: Headset },
];

/** 页面最右侧的窄工具栏:顶部 AI 助手,底部帮助与支持。 */
export function RightRail({ aiOpen, onToggleAi }: { aiOpen: boolean; onToggleAi: () => void }) {
  const [openHelp, setOpenHelp] = useState(false);
  const [openSettings, setOpenSettings] = useState(false);

  return (
    <aside className="w-12 shrink-0 h-full bg-white border-l border-[var(--color-border-subtle)] flex flex-col items-center py-3 gap-2 relative z-30">
      <button
        type="button"
        onClick={onToggleAi}
        title="AI 助手"
        aria-label="AI 助手"
        aria-pressed={aiOpen}
        className={cn(
          "rounded-full p-0.5 transition-all duration-200 ease-[var(--ease-bob)] active:scale-[0.94]",
          aiOpen ? "ring-2 ring-[var(--color-accent)] ring-offset-1" : "hover:scale-105",
        )}
      >
        <AiAvatar size={32} />
      </button>

      <div className="flex-1" />

      <div className="relative">
        <RailButton
          icon={CircleHelp}
          label="帮助与支持"
          active={openHelp}
          onClick={() => setOpenHelp((v) => !v)}
        />
        {openHelp && (
          <div className="absolute bottom-0 right-[calc(100%+10px)] w-[240px] z-40 bg-white rounded-xl border border-[var(--color-border-subtle)] shadow-[0_8px_24px_rgba(58,58,55,0.14)] overflow-hidden py-1.5">
            <div className="px-4 pt-1.5 pb-2 text-[11px] font-bold tracking-[0.12em] text-[var(--color-text-tertiary)] font-body">
              帮助与支持
            </div>
            {helpLinks.map(({ label, desc, href, icon: Icon }) => (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setOpenHelp(false)}
                className="group flex items-center gap-3 px-4 py-2.5 hover:bg-[var(--color-bg-muted)] transition-colors"
              >
                <MenuIcon icon={Icon} tone="accent" />
                <MenuText label={label} desc={desc} />
                <ExternalLink className="w-3.5 h-3.5 text-[var(--color-text-tertiary)] opacity-0 group-hover:opacity-100 transition-opacity" />
              </a>
            ))}
            <div className="my-1.5 h-px bg-[var(--color-border-row)]" />
            <button
              type="button"
              onClick={() => {
                setOpenHelp(false);
                setOpenSettings(true);
              }}
              className="flex items-center gap-3 px-4 py-2.5 w-full text-left hover:bg-[var(--color-bg-muted)] transition-colors"
            >
              <MenuIcon icon={Settings} tone="muted" />
              <MenuText label="设置" desc="页面字体大小等偏好" />
            </button>
          </div>
        )}
      </div>

      {openHelp && (
        <div className="fixed inset-0 -z-10" onClick={() => setOpenHelp(false)} aria-hidden />
      )}
      <SettingsModal open={openSettings} onClose={() => setOpenSettings(false)} />
    </aside>
  );
}

function RailButton({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: LucideIcon;
  label: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-expanded={active}
      className={cn(
        "w-9 h-9 grid place-items-center rounded-full transition-colors",
        active
          ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
          : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)]",
      )}
    >
      <Icon className="w-[18px] h-[18px]" />
    </button>
  );
}

function MenuIcon({ icon: Icon, tone }: { icon: LucideIcon; tone: "accent" | "muted" }) {
  return (
    <span
      className={cn(
        "w-8 h-8 rounded-full grid place-items-center shrink-0",
        tone === "accent"
          ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
          : "bg-[var(--color-bg-muted)] text-[var(--color-text-secondary)]",
      )}
    >
      <Icon className="w-4 h-4" />
    </span>
  );
}

function MenuText({ label, desc }: { label: string; desc: string }) {
  return (
    <span className="flex flex-col flex-1 min-w-0">
      <span className="text-[13px] font-bold text-[var(--color-text-primary)] font-body">{label}</span>
      <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">{desc}</span>
    </span>
  );
}
