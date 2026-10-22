import { type ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "tertiary" | "ghost";
  /** sm = 列表行内操作;lg = ESS 打卡等大动作 */
  size?: "sm" | "md" | "lg";
  fullWidth?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { variant = "primary", size = "md", fullWidth, className, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md font-bold font-body transition-all duration-200 ease-[var(--ease-bob)] active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none",
        size === "md" && "px-5 py-2.5 text-sm",
        size === "sm" && "px-3 py-1.5 text-[13px]",
        size === "lg" && "px-6 py-3 text-[15px]",
        variant === "primary" &&
          "bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] shadow-[0_1px_2px_rgba(187,17,65,0.25)]",
        variant === "secondary" &&
          "bg-white text-[var(--color-text-primary)] border border-[var(--color-border-subtle)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]",
        variant === "tertiary" &&
          "bg-transparent text-[var(--color-accent)] px-3 hover:bg-[var(--color-accent-soft)]",
        variant === "ghost" &&
          "bg-transparent text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-subtle)] hover:text-[var(--color-text-primary)]",
        fullWidth && "w-full",
        className,
      )}
      {...rest}
    />
  );
});
