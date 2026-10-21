import { useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { FONT_SIZES, getFontSize, setFontSize, type FontSize } from "@/lib/fontSize";

export function SettingsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [size, setSize] = useState<FontSize>(getFontSize);

  const choose = (v: FontSize) => {
    setSize(v);
    setFontSize(v);
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="设置"
      description="偏好只保存在当前浏览器"
      width={460}
      footer={<Button onClick={onClose}>完成</Button>}
    >
      <section className="flex flex-col gap-3 pb-4">
        <div className="flex flex-col gap-0.5">
          <h4 className="text-sm font-bold text-[var(--color-text-primary)] font-body">页面字体大小</h4>
          <p className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            整体等比缩放，文字、图标和间距一起变化
          </p>
        </div>

        <div role="radiogroup" aria-label="页面字体大小" className="grid grid-cols-3 gap-2">
          {FONT_SIZES.map((o) => {
            const active = size === o.value;
            return (
              <button
                key={o.value}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => choose(o.value)}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-xl border px-2 py-3 transition-all duration-200 ease-[var(--ease-bob)] active:scale-[0.98]",
                  active
                    ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                    : "border-[var(--color-border-subtle)] bg-white text-[var(--color-text-primary)] hover:border-[var(--color-accent)]",
                )}
              >
                <span className="font-heading font-semibold leading-none" style={{ fontSize: o.value + 4 }}>
                  Aa
                </span>
                <span className="text-[13px] font-bold font-body">{o.label}</span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {o.value}px · {o.hint}
                </span>
              </button>
            );
          })}
        </div>
      </section>
    </Modal>
  );
}
