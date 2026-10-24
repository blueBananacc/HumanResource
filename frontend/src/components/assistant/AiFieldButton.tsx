import { useEffect, useRef, useState } from "react";
import { Check, Loader2, Sparkles, Wand2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchJSON } from "@/lib/api";

type Intent = "draft" | "polish" | "expand" | "shorten";

const INTENTS: { key: Intent; label: string; needsText: boolean }[] = [
  { key: "draft", label: "起草", needsText: false },
  { key: "polish", label: "润色", needsText: true },
  { key: "expand", label: "扩写", needsText: true },
  { key: "shorten", label: "精简", needsText: true },
];

/**
 * 字段级 AI 助手(T6-1):表单字段旁的 ✨ 按钮。
 * 弹出小卡片选意图 → 调 /api/ai/field-assist 拿草稿 → "应用"回填字段。
 * AI 只产出草稿,实际写入由用户决定。
 */
export function AiFieldButton({
  target,
  fields,
  currentText,
  onApply,
  className,
}: {
  /** "jobs.description" / "announcements.content" / "cases.resolution" / "generic" */
  target: string;
  /** 传给后端的上下文字段(表单当前相关值) */
  fields?: Record<string, string>;
  /** 当前字段文本(polish/expand/shorten 需要) */
  currentText?: string;
  /** 草稿确认后回调,由调用方 set 到表单 state */
  onApply: (text: string) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [intent, setIntent] = useState<Intent>("draft");
  const [instruction, setInstruction] = useState("");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [applied, setApplied] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const generate = async () => {
    if (INTENTS.find((i) => i.key === intent)?.needsText && !currentText?.trim()) {
      setError("该操作需要字段里已有内容");
      return;
    }
    setLoading(true);
    setError("");
    setDraft("");
    setApplied(false);
    try {
      const res = await fetchJSON<{ text: string }>("/api/ai/field-assist", {
        method: "POST",
        body: JSON.stringify({
          target,
          intent,
          fields: fields ?? {},
          instruction: instruction.trim(),
          current_text: currentText ?? "",
        }),
      });
      setDraft(res.text);
    } catch (e) {
      setError(e instanceof Error ? e.message : "生成失败,请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div ref={rootRef} className={cn("relative inline-block", className)}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="AI 助手"
        aria-label="AI 助手"
        className="w-7 h-7 grid place-items-center rounded-lg text-[var(--color-accent)] bg-[var(--color-accent-soft)] hover:bg-[var(--color-accent)] hover:text-white active:scale-[0.94] transition-all"
      >
        <Sparkles className="w-3.5 h-3.5" />
      </button>

      {open && (
        <div className="absolute right-0 z-50 mt-1.5 w-[300px] rounded-xl border border-[var(--color-border-subtle)] bg-white p-3 shadow-[var(--shadow-card)]">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[12px] font-bold text-[var(--color-text-primary)] flex items-center gap-1">
              <Wand2 className="w-3.5 h-3.5 text-[var(--color-accent)]" />
              AI 写作助手
            </span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] p-0.5"
              aria-label="关闭"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex gap-1 mb-2">
            {INTENTS.map((i) => (
              <button
                key={i.key}
                type="button"
                onClick={() => setIntent(i.key)}
                className={cn(
                  "flex-1 h-7 rounded-md text-[11px] font-bold transition-all",
                  intent === i.key
                    ? "bg-[var(--color-accent)] text-white"
                    : "bg-[var(--color-bg-muted)] text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-accent)]",
                )}
              >
                {i.label}
              </button>
            ))}
          </div>

          <input
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="补充要求(可选),如:语气活泼一点"
            className="w-full h-8 rounded-lg border border-[var(--color-border-subtle)] px-2.5 text-[12px] mb-2 focus:outline-none focus:border-[var(--color-accent)]"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void generate();
              }
            }}
          />

          <button
            type="button"
            onClick={() => void generate()}
            disabled={loading}
            className="w-full h-8 rounded-lg bg-[var(--color-accent)] text-white text-[12px] font-bold hover:bg-[var(--color-accent-hover)] active:scale-[0.98] transition-all disabled:opacity-50 flex items-center justify-center gap-1.5"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            {loading ? "生成中…" : draft ? "重新生成" : "生成草稿"}
          </button>

          {error && (
            <p className="mt-2 text-[11px] text-[var(--color-danger,#dc2626)]">{error}</p>
          )}

          {draft && (
            <div className="mt-2">
              <div className="max-h-44 overflow-y-auto rounded-lg bg-[var(--color-bg-muted)] px-2.5 py-2 text-[12px] text-[var(--color-text-primary)] whitespace-pre-wrap leading-relaxed">
                {draft}
              </div>
              <button
                type="button"
                onClick={() => {
                  onApply(draft);
                  setApplied(true);
                }}
                className={cn(
                  "mt-2 w-full h-8 rounded-lg border text-[12px] font-bold transition-all flex items-center justify-center gap-1",
                  applied
                    ? "border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                    : "border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[var(--color-accent-soft)] active:scale-[0.98]",
                )}
              >
                {applied ? (
                  <>
                    <Check className="w-3.5 h-3.5" /> 已应用,可继续修改
                  </>
                ) : (
                  "应用到字段"
                )}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
