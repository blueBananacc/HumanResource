import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bookmark, ChevronDown, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { fetchJSON } from "@/lib/api";

type SavedFilter = { id: string; page: string; name: string; params: Record<string, unknown> };

/**
 * 列表页"保存筛选"下拉。
 * `currentParams` 返回当前筛选状态快照;`onApply` 负责把快照写回页面 state。
 */
export function SavedFilters({
  page,
  currentParams,
  onApply,
}: {
  page: string;
  currentParams: () => Record<string, unknown>;
  onApply: (params: Record<string, unknown>) => void;
}) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");

  const { data: filters } = useQuery({
    queryKey: ["saved-filters", page],
    queryFn: () => fetchJSON<SavedFilter[]>(`/api/saved-filters?page=${encodeURIComponent(page)}`),
  });

  const save = useMutation({
    mutationFn: (n: string) =>
      fetchJSON<SavedFilter>("/api/saved-filters", {
        method: "POST",
        body: JSON.stringify({ page, name: n, params: currentParams() }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["saved-filters", page] });
      setSaving(false);
      setName("");
    },
  });

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/saved-filters/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["saved-filters", page] }),
  });

  return (
    <div className="relative">
      <Button variant="secondary" onClick={() => setOpen((v) => !v)}>
        <Bookmark className="mr-1.5 h-3.5 w-3.5" />
        筛选
        <ChevronDown className="ml-1 h-3.5 w-3.5" />
      </Button>
      {open && (
        <div
          className="absolute right-0 z-40 mt-1 w-64 rounded-2xl border border-[var(--color-border-subtle)] bg-white p-2 shadow-[var(--shadow-card)]"
          onMouseLeave={() => setOpen(false)}
        >
          {(filters ?? []).length === 0 && !saving && (
            <p className="px-2 py-3 text-xs text-[var(--color-text-muted)]">暂无保存的筛选</p>
          )}
          {(filters ?? []).map((f) => (
            <div
              key={f.id}
              className="group flex items-center justify-between rounded-lg px-2 py-1.5 text-sm hover:bg-[var(--color-bg-muted)]"
            >
              <button
                type="button"
                className="flex-1 text-left"
                onClick={() => {
                  onApply(f.params);
                  setOpen(false);
                }}
              >
                {f.name}
              </button>
              <button
                type="button"
                className="invisible text-[var(--color-text-muted)] hover:text-[var(--color-accent)] group-hover:visible"
                onClick={() => del.mutate(f.id)}
                aria-label="删除筛选"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          <div className="mt-1 border-t border-[var(--color-border-subtle)] pt-2">
            {saving ? (
              <form
                className="flex gap-1.5"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (name.trim()) save.mutate(name.trim());
                }}
              >
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="筛选名称"
                  className="h-8 flex-1 rounded-lg border border-[var(--color-border-subtle)] bg-white px-2 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
                />
                <Button type="submit" disabled={save.isPending || !name.trim()}>
                  存
                </Button>
              </form>
            ) : (
              <button
                type="button"
                className="w-full rounded-lg px-2 py-1.5 text-left text-sm text-[var(--color-accent)] hover:bg-[var(--color-bg-muted)]"
                onClick={() => setSaving(true)}
              >
                + 保存当前筛选
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
