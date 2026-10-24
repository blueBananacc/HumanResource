import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GitMerge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { ApiError, fetchJSON } from "@/lib/api";

type Dup = {
  id: string;
  name: string;
  display_email: string | null;
  display_phone: string | null;
  status: string;
  match: "email" | "phone" | "name";
};

const FIELDS = [
  { key: "name", label: "姓名" },
  { key: "display_email", label: "邮箱" },
  { key: "display_phone", label: "电话" },
  { key: "tags", label: "标签" },
  { key: "status", label: "状态" },
] as const;

/** 候选人合并向导(MintHCM MergeRecords 三步的紧凑版):查重→取舍→合并。 */
export function MergeTalentModal({
  keepId,
  keepName,
  keep,
  open,
  onClose,
}: {
  keepId: string;
  keepName: string;
  keep: Record<string, unknown> | undefined;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [dropId, setDropId] = useState<string | null>(null);
  const [choices, setChoices] = useState<Record<string, "keep" | "drop">>({});
  const [error, setError] = useState<string | null>(null);

  const { data: dups, isLoading } = useQuery({
    queryKey: ["talent-duplicates", keepId],
    enabled: open,
    queryFn: () => fetchJSON<Dup[]>(`/api/talents/${encodeURIComponent(keepId)}/duplicates`),
  });
  const drop = (dups ?? []).find((d) => d.id === dropId);

  const merge = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/talents/${encodeURIComponent(keepId)}/merge`, {
        method: "POST",
        body: JSON.stringify({ drop_id: dropId, field_choices: choices }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["talents"] });
      qc.invalidateQueries({ queryKey: ["talent", keepId] });
      onClose();
      setDropId(null);
      setChoices({});
    },
    onError: (e) => setError(e instanceof ApiError ? String(e.message) : "合并失败"),
  });

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="合并重复候选人"
      description={`保留「${keepName}」,把另一条的关联数据(简历/申请/面试/offer/技能/备注/分组)合并过来`}
      width={640}
      footer={
        dropId ? (
          <>
            <Button variant="secondary" onClick={() => setDropId(null)}>返回</Button>
            <Button onClick={() => merge.mutate()} disabled={merge.isPending}>
              {merge.isPending ? "合并中…" : "确认合并"}
            </Button>
          </>
        ) : undefined
      }
    >
      {error && <p className="mb-3 text-sm text-[var(--color-accent)]">{error}</p>}
      {!dropId && (
        <div>
          {isLoading && <p className="text-sm text-[var(--color-text-muted)]">查找重复中…</p>}
          {!isLoading && (dups ?? []).length === 0 && (
            <p className="text-sm text-[var(--color-text-muted)]">没有发现疑似重复的候选人</p>
          )}
          <ul className="space-y-2">
            {(dups ?? []).map((d) => (
              <li key={d.id}>
                <button
                  type="button"
                  className="flex w-full items-center justify-between rounded-xl border border-[var(--color-border-subtle)] bg-white p-3 text-left hover:border-[var(--color-accent)]"
                  onClick={() => setDropId(d.id)}
                >
                  <div>
                    <p className="text-sm font-medium text-[var(--color-text)]">{d.name}</p>
                    <p className="text-xs text-[var(--color-text-muted)]">
                      {d.display_email ?? "—"} · {d.display_phone ?? "—"} · {d.status}
                    </p>
                  </div>
                  <span className="rounded-full bg-[var(--color-cappuccino)] px-2 py-0.5 text-xs text-[var(--color-brand)]">
                    {d.match === "email" ? "同邮箱" : d.match === "phone" ? "同电话" : "同名"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {dropId && drop && (
        <div>
          <p className="mb-3 text-sm text-[var(--color-text-muted)]">
            逐字段选择保留谁的值(默认保留当前候选人):
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-[var(--color-text-muted)]">
                <th className="pb-2">字段</th>
                <th className="pb-2">{keepName}(保留)</th>
                <th className="pb-2">{drop.name}(被合并)</th>
              </tr>
            </thead>
            <tbody>
              {FIELDS.map((f) => {
                const keepVal = keep?.[f.key];
                const dropVal = (drop as unknown as Record<string, unknown>)[f.key];
                const fmt = (v: unknown) =>
                  Array.isArray(v) ? v.join(", ") : v == null || v === "" ? "—" : String(v);
                return (
                  <tr key={f.key} className="border-t border-[var(--color-border-subtle)]">
                    <td className="py-2 text-[var(--color-text-muted)]">{f.label}</td>
                    {(["keep", "drop"] as const).map((side) => (
                      <td key={side} className="py-2">
                        <label className="flex items-center gap-1.5">
                          <input
                            type="radio"
                            name={`f-${f.key}`}
                            checked={(choices[f.key] ?? "keep") === side}
                            onChange={() => setChoices((c) => ({ ...c, [f.key]: side }))}
                          />
                          <span className="text-[var(--color-text)]">
                            {fmt(side === "keep" ? keepVal : dropVal)}
                          </span>
                        </label>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-3 flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
            <GitMerge className="h-3.5 w-3.5" />
            合并后「{drop.name}」会被删除,操作会写入审计日志
          </p>
        </div>
      )}
    </Modal>
  );
}
