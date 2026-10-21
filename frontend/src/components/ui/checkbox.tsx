import { Check, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * 复选框(bob 风格):未选中白底灰边 → hover 描 Cherry;
 * 选中 Cherry 底 + 白勾;部分选中(indeterminate)显示横杠。
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
        "w-4 h-4 rounded-[5px] border-[1.5px] grid place-items-center shrink-0 cursor-pointer",
        "transition-all duration-150 ease-[var(--ease-bob)] active:scale-90",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/30 focus-visible:ring-offset-1",
        on
          ? "bg-[var(--color-accent)] border-[var(--color-accent)]"
          : "bg-white border-[var(--color-text-tertiary)]/50 hover:border-[var(--color-accent)]",
        className,
      )}
    >
      {indeterminate ? (
        <Minus className="w-3 h-3 text-white" strokeWidth={3.5} />
      ) : checked ? (
        <Check className="w-3 h-3 text-white" strokeWidth={3.5} />
      ) : null}
    </button>
  );
}
