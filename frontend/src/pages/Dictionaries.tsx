import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ApiError, fetchJSON } from "@/lib/api";

type DictKey = { key: string; builtin: string[] };
type DictItem = {
  id: string;
  dict_key: string;
  value: string;
  label: string;
  sort: number;
  is_active: boolean;
};

const KEY_LABEL: Record<string, string> = {
  request_kind: "员工申请类型",
  case_kind: "工单类型",
  skill_category: "技能分类",
};

/** 数据字典管理:租户可配置下拉枚举。 */
export function Dictionaries() {
  const qc = useQueryClient();
  const [key, setKey] = useState("request_kind");
  const [value, setValue] = useState("");
  const [label, setLabel] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { data: keys } = useQuery({
    queryKey: ["dict-keys"],
    queryFn: () => fetchJSON<DictKey[]>("/api/dictionaries/keys"),
  });
  const { data: items } = useQuery({
    queryKey: ["dict-items", key],
    queryFn: () => fetchJSON<DictItem[]>(`/api/dictionaries?dict_key=${encodeURIComponent(key)}`),
  });
  const builtin = (keys ?? []).find((k) => k.key === key)?.builtin ?? [];

  const add = useMutation({
    mutationFn: () =>
      fetchJSON<DictItem>("/api/dictionaries", {
        method: "POST",
        body: JSON.stringify({ dict_key: key, value: value.trim(), label: label.trim(), sort: (items ?? []).length }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dict-items", key] });
      setValue("");
      setLabel("");
      setError(null);
    },
    onError: (e) => setError(e instanceof ApiError ? String(e.message) : "新增失败"),
  });
  const toggle = useMutation({
    mutationFn: (d: DictItem) =>
      fetchJSON(`/api/dictionaries/${d.id}`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: !d.is_active }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dict-items", key] }),
  });
  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/dictionaries/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dict-items", key] }),
  });

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-6">
      <div>
        <h1 className="font-heading text-xl text-[var(--color-text)]">数据字典</h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          自定义下拉枚举值;新增项即时生效于对应表单,内置值不可删除。
        </p>
      </div>
      <div className="w-64">
        <Select
          value={key}
          onChange={setKey}
          options={(keys ?? []).map((k) => ({
            value: k.key,
            label: KEY_LABEL[k.key] ?? k.key,
          }))}
        />
      </div>
      <div className="rounded-2xl border border-[var(--color-border-subtle)] bg-white p-4 shadow-[var(--shadow-card)]">
        <h2 className="mb-2 text-sm font-medium text-[var(--color-text)]">内置值</h2>
        <div className="flex flex-wrap gap-1.5">
          {builtin.map((v) => (
            <span
              key={v}
              className="rounded-full bg-[var(--color-cappuccino)] px-2.5 py-1 text-xs text-[var(--color-brand)]"
            >
              {v}
            </span>
          ))}
        </div>
        <h2 className="mb-2 mt-5 text-sm font-medium text-[var(--color-text)]">自定义值</h2>
        {(items ?? []).length === 0 && (
          <p className="text-sm text-[var(--color-text-muted)]">暂无自定义值</p>
        )}
        <ul className="divide-y divide-[var(--color-border-subtle)]">
          {(items ?? []).map((d) => (
            <li key={d.id} className="flex items-center justify-between py-2">
              <div>
                <span className="text-sm text-[var(--color-text)]">{d.label}</span>
                <span className="ml-2 text-xs text-[var(--color-text-muted)]">{d.value}</span>
                {!d.is_active && (
                  <span className="ml-2 rounded-full bg-[var(--color-bg-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-text-muted)]">
                    已停用
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <Button variant="secondary" onClick={() => toggle.mutate(d)}>
                  {d.is_active ? "停用" : "启用"}
                </Button>
                <button
                  type="button"
                  className="rounded-lg p-1.5 text-[var(--color-text-muted)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-accent)]"
                  onClick={() => del.mutate(d.id)}
                  aria-label="删除"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </li>
          ))}
        </ul>
        <form
          className="mt-4 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (value.trim() && label.trim()) add.mutate();
          }}
        >
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="值(英文,如 relocation)"
            className="h-9 flex-1 rounded-lg border border-[var(--color-border-subtle)] bg-white px-3 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
          />
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="显示名(如 异地调动)"
            className="h-9 flex-1 rounded-lg border border-[var(--color-border-subtle)] bg-white px-3 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
          />
          <Button type="submit" disabled={add.isPending || !value.trim() || !label.trim()}>
            <Plus className="mr-1 h-4 w-4" />
            新增
          </Button>
        </form>
        {error && <p className="mt-2 text-sm text-[var(--color-accent)]">{error}</p>}
      </div>
    </div>
  );
}
