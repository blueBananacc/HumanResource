import { Check, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * 复选框(对齐 bob 产品):小方框,白底暖灰边;
 * 选中 / 半选时 Cherry 底 + 白色粗对勾(或横杠),对勾带轻微弹入;
 * hover 描边变 Cherry;热区比可视框大一圈更好点。
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
        "grid place-items-center shrink-0 cursor-pointer -m-1.5 p-1.5",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/30 rounded",
        className,
      )}
    >
      <span
        className={cn(
          "grid place-items-center w-4 h-4 rounded-[4px] border-[1.5px]",
          "transition-all duration-150 ease-[var(--ease-bob)]",
          on
            ? "bg-[var(--color-accent)] border-[var(--color-accent)] shadow-[0_1px_2px_rgba(238,22,79,0.3)]"
            : "bg-white border-[#c9c4b8] hover:border-[var(--color-accent)]",
        )}
      >
        {indeterminate ? (
          <Minus className="w-3 h-3 text-white animate-[check-pop_.15s_var(--ease-bob)]" strokeWidth={4} />
        ) : checked ? (
          <Check className="w-3 h-3 text-white animate-[check-pop_.15s_var(--ease-bob)]" strokeWidth={4} />
        ) : null}
      </span>
    </button>
  );
}
