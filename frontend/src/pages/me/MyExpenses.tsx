import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Receipt, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  fmtMoney,
  type ExpenseType,
  type MeClaim,
} from "@/lib/hris";
import { Field } from "../hris/Employees";

export function MyExpenses() {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const { data, isLoading } = useQuery<MeClaim[]>({
    queryKey: ["me-claims"],
    queryFn: () => fetchJSON("/api/me/expense-claims"),
  });
  const items = data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            我的报销
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            一张报销单可含多条明细,提交后进入审批。
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="w-4 h-4" />
          新建报销
        </Button>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Receipt}
            title="还没有报销单"
            action={
              <Button onClick={() => setOpen(true)}>新建报销</Button>
            }
          />
        ) : (
          items.map((c, i) => (
            <div
              key={c.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={() =>
                    setExpanded((p) => ({ ...p, [c.id]: !p[c.id] }))
                  }
                  className="font-medium text-sm text-[var(--color-text-primary)] font-body hover:text-[var(--color-accent)] text-left"
                >
                  {c.title}
                </button>
                <StatusPill status={c.status} />
                <div className="flex-1" />
                <span className="font-mono font-semibold text-sm text-[var(--color-text-primary)]">
                  {fmtMoney(c.total)}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {formatRelative(c.created_at)}
                </span>
              </div>
              {expanded[c.id] && (
                <div className="mt-2 ml-1 flex flex-col gap-1">
                  {c.items.map((it) => (
                    <div
                      key={it.id}
                      className="flex items-center gap-3 text-[12px] font-body text-[var(--color-text-secondary)]"
                    >
                      <span className="font-mono">{it.occurred_on}</span>
                      <span>{it.type_name ?? "未分类"}</span>
                      <span className="font-mono">{fmtMoney(it.amount)}</span>
                      {it.note && (
                        <span className="text-[var(--color-text-tertiary)]">
                          {it.note}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
              {c.decision_note && (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  审批意见: {c.decision_note}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <ClaimCreateModal open={open} onClose={() => setOpen(false)} />
    </main>
  );
}

type ItemRow = {
  type_id: string;
  occurred_on: string;
  amount: string;
  note: string;
};

function ClaimCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: types } = useQuery<ExpenseType[]>({
    queryKey: ["me-expense-types"],
    queryFn: () => fetchJSON("/api/me/expense-types"),
  });
  const [rows, setRows] = useState<ItemRow[]>([
    { type_id: "", occurred_on: "", amount: "", note: "" },
  ]);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/expense-claims", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-claims"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const items = rows
      .filter((r) => r.occurred_on && Number(r.amount) > 0)
      .map((r) => ({
        type_id: r.type_id || null,
        occurred_on: r.occurred_on,
        amount: Number(r.amount),
        note: r.note.trim() || null,
      }));
    if (items.length === 0) {
      setErr("至少填一条明细(日期+金额)");
      return;
    }
    mut.mutate({
      title: String(fd.get("title") || "").trim(),
      items,
    });
  }

  const typeOptions = (types ?? []).map((t) => ({
    value: t.id,
    label: t.name,
  }));

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建报销"
      width={640}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="me-claim" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="me-claim" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="报销单标题" required>
          <Input name="title" required placeholder="10 月出差上海" />
        </Field>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
              明细
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() =>
                setRows((p) => [
                  ...p,
                  { type_id: "", occurred_on: "", amount: "", note: "" },
                ])
              }
            >
              <Plus className="w-3.5 h-3.5" />
              加一行
            </Button>
          </div>
          {rows.map((r, i) => (
            <div key={i} className="flex items-center gap-2">
              <Input
                type="date"
                value={r.occurred_on}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, occurred_on: e.target.value } : x,
                    ),
                  )
                }
                className="w-[140px]"
              />
              <Select
                value={r.type_id}
                onChange={(v) =>
                  setRows((p) =>
                    p.map((x, j) => (j === i ? { ...x, type_id: v } : x)),
                  )
                }
                options={typeOptions}
                placeholder="类型"
                className="w-[120px]"
              />
              <Input
                type="number"
                min="0"
                step="0.01"
                value={r.amount}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, amount: e.target.value } : x,
                    ),
                  )
                }
                placeholder="金额"
                className="w-[110px]"
              />
              <Input
                value={r.note}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, note: e.target.value } : x,
                    ),
                  )
                }
                placeholder="备注"
                className="flex-1"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setRows((p) => p.filter((_, j) => j !== i))}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          ))}
        </div>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
