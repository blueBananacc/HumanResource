import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Copy, Download, FileText, Loader2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchJSON } from "@/lib/api";
import { Drawer } from "@/components/ui/drawer";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";

type TemplateOpt = { key: string; name: string; entity: string };

/**
 * AI 文档起草抽屉(T6-4):
 * 选模板 → 后端按记录预填 {{var}} → LLM 补全 → 可编辑预览 → 复制/下载。
 * 只产草稿,不自动发送。
 */
export function DocDraftDrawer({
  open,
  onClose,
  entity,
  recordId,
}: {
  open: boolean;
  onClose: () => void;
  entity: "candidates" | "employees";
  recordId: string;
}) {
  const [tpl, setTpl] = useState("");
  const [instruction, setInstruction] = useState("");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const { data: templates } = useQuery({
    queryKey: ["doc-template-options"],
    queryFn: () => fetchJSON<TemplateOpt[]>("/api/ai/doc-templates"),
    enabled: open,
  });

  const usable = useMemo(
    () =>
      (templates ?? []).filter((t) => t.entity === "any" || t.entity === entity),
    [templates, entity],
  );

  const generate = async () => {
    if (!tpl) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetchJSON<{ draft: string }>("/api/ai/doc-draft", {
        method: "POST",
        body: JSON.stringify({
          key: tpl,
          entity,
          record_id: recordId,
          instruction: instruction.trim(),
        }),
      });
      setDraft(res.draft);
      setCopied(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "生成失败,请重试");
    } finally {
      setLoading(false);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  const download = () => {
    const name = usable.find((t) => t.key === tpl)?.name ?? "document";
    const blob = new Blob([draft], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="AI 文档起草"
      description="选择模板,AI 按当前记录自动填充并补全文书"
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <label className="text-[12px] font-bold text-[var(--color-text-secondary)]">
            文档模板
          </label>
          <Select
            value={tpl}
            onChange={setTpl}
            options={usable.map((t) => ({ value: t.key, label: t.name }))}
            placeholder="选择文档类型"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-[12px] font-bold text-[var(--color-text-secondary)]">
            附加要求(可选)
          </label>
          <input
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="如:语气亲切一点,补一段欢迎语"
            className="h-9 rounded-lg border border-[var(--color-border-subtle)] px-3 text-[13px] focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>

        <Button
          onClick={() => void generate()}
          disabled={!tpl || loading}
          className="h-9"
        >
          {loading ? (
            <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="mr-1.5 h-4 w-4" />
          )}
          {loading ? "AI 起草中…" : draft ? "重新生成" : "生成草稿"}
        </Button>

        {error && (
          <p className="text-[12px] text-[var(--color-danger,#dc2626)]">{error}</p>
        )}

        {draft && (
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-[12px] font-bold text-[var(--color-text-secondary)] flex items-center gap-1">
                <FileText className="w-3.5 h-3.5" />
                草稿(可直接编辑)
              </span>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => void copy()}
                  className="h-7 px-2 rounded-md text-[11px] font-bold text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] flex items-center gap-1 transition-all"
                >
                  <Copy className="w-3 h-3" />
                  {copied ? "已复制" : "复制"}
                </button>
                <button
                  type="button"
                  onClick={download}
                  className="h-7 px-2 rounded-md text-[11px] font-bold text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)] flex items-center gap-1 transition-all"
                >
                  <Download className="w-3 h-3" />
                  下载 .md
                </button>
              </div>
            </div>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={18}
              className={cn(
                "w-full resize-y rounded-lg border border-[var(--color-border-subtle)] px-3 py-2.5",
                "text-[13px] leading-relaxed text-[var(--color-text-primary)]",
                "focus:outline-none focus:border-[var(--color-accent)]",
              )}
            />
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              草稿由 AI 生成,请核对后使用;本操作已记录审计日志
            </p>
          </div>
        )}
      </div>
    </Drawer>
  );
}
