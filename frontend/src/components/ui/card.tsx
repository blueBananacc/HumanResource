import { type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Card({
  className,
  ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]",
        className,
      )}
      {...rest}
    />
  );
}
