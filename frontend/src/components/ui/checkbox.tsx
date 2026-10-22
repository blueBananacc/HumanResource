import { Check, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * 复选框(bob 风格,圆形):
 * - 未选中:白底 + 暖灰细环,hover 描 Orange(规范里橙色的职责就是"选中指示")
 * - 选中 / 半选:Cherry 底 + 白色对勾 / 横杠,对勾带轻微弹入动画
 * - 热区比可视框大一圈(p-1.5 -m-1.5),行内更容易点中
 * 在 <Link>/<a> 里用时,外层包一个 onClick={e => e.preventDefault()} 的 span 阻止跳转。
 */
export function Checkbox({
  checked = false,
  indeterminate = false,
  onToggle,
  label,
  className,
}: {
  checked?: boolean;
  indeterminate?: boolean;
  onToggle: () => void;
  label: string;
  className?: string;
}) {
  const on = checked || indeterminate;
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={indeterminate ? "mixed" : checked}
      aria-label={label}
      onClick={onToggle}
      className={cn(
        "grid place-items-center shrink-0 cursor-pointer -m-1.5 p-1.5 rounded-full",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/30",
        className,
      )}
    >
      <span
        className={cn(
          "grid place-items-center w-[18px] h-[18px] rounded-full border-[1.5px]",
          "transition-all duration-150 ease-[var(--ease-bob)]",
          on
            ? "bg-[var(--color-accent)] border-[var(--color-accent)] shadow-[0_1px_3px_rgba(238,22,79,0.35)]"
            : "bg-white border-[var(--color-text-tertiary)]/45 hover:border-[var(--color-orange)] hover:scale-105",
        )}
      >
        {indeterminate ? (
          <Minus className="w-3 h-3 text-white animate-[check-pop_.18s_var(--ease-bob)]" strokeWidth={3.5} />
        ) : checked ? (
          <Check className="w-3 h-3 text-white animate-[check-pop_.18s_var(--ease-bob)]" strokeWidth={3.5} />
        ) : null}
      </span>
    </button>
  );
}
