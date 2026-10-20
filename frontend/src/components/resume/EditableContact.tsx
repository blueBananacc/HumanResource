import { useEffect, useState } from "react";
import { Check, Loader2, Mail, Pencil, X } from "lucide-react";
import { ApiError } from "@/lib/api";

/**
 * 联系信息卡片 — 默认展示态,点铅笔切到编辑态。
 *
 * 设计:
 * - 不传 ``onSave`` 就是只读 ContactCard(目前所有调用都传了,但保留以兼容)
 * - validate 同步返回 error message 或 null;为空字符串 + null 表示"允许清空"
 * - 保存成功通过 onSave(promise) 返回后自动退出编辑态;失败把 error 显在内联
 */
export function EditableContact({
  icon: Icon,
  label,
  value,
  placeholder,
  className,
  onSave,
  validate,
  inputType,
}: {
  icon: typeof Mail;
  label: string;
  value: string | null;
  placeholder?: string;
  className?: string;
  onSave?: (next: string) => Promise<unknown>;
  validate?: (next: string) => string | null;
  inputType?: "text" | "email" | "tel";
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value ?? "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // 外部 value 变化(刷新/同步)时,只有不在编辑态才同步,避免覆盖用户输入
  useEffect(() => {
    if (!editing) setDraft(value ?? "");
  }, [value, editing]);

  function startEdit() {
    setDraft(value ?? "");
    setErr(null);
    setEditing(true);
  }

  function cancel() {
    setDraft(value ?? "");
    setErr(null);
    setEditing(false);
  }

  async function save() {
    const next = draft.trim();
    const v = validate?.(next);
    if (v) {
      setErr(v);
      return;
    }
    if (!onSave) {
      setEditing(false);
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      await onSave(next);
      setEditing(false);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className={`flex items-start gap-3 p-3 rounded-lg bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)] group ${className ?? ""}`}
    >
      <div className="w-7 h-7 rounded-md bg-white border border-[var(--color-border-subtle)] flex items-center justify-center shrink-0">
        <Icon className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
      </div>
      <div className="flex flex-col gap-1 min-w-0 flex-1">
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          {label}
        </span>
        {editing ? (
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <input
                autoFocus
                type={inputType ?? "text"}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") save();
                  else if (e.key === "Escape") cancel();
                }}
                disabled={saving}
                placeholder={placeholder}
                className="flex-1 min-w-0 text-[13px] font-mono text-[var(--color-text-primary)] bg-white border border-[var(--color-border-subtle)] rounded-md px-2 py-1 focus:outline-none focus:border-[var(--color-text-primary)]"
              />
              <button
                onClick={save}
                disabled={saving}
                className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] disabled:opacity-50 p-1"
                aria-label="保存"
                title="保存(回车)"
              >
                {saving ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Check className="w-3.5 h-3.5" />
                )}
              </button>
              <button
                onClick={cancel}
                disabled={saving}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] disabled:opacity-50 p-1"
                aria-label="取消"
                title="取消(Esc)"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            {err && (
              <span className="text-[11px] text-[var(--color-danger)] font-body">
                {err}
              </span>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span
              className={`text-[13px] font-mono truncate ${
                value
                  ? "text-[var(--color-text-primary)]"
                  : "text-[var(--color-text-tertiary)] italic"
              }`}
            >
              {value || placeholder || "—"}
            </span>
            {onSave && (
              <button
                onClick={startEdit}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] p-0.5"
                aria-label={`编辑${label}`}
                title={`编辑${label}`}
              >
                <Pencil className="w-3 h-3" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}