import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Tags, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { fetchJSON, ApiError } from "@/lib/api";
import { type ExpenseType } from "@/lib/hris";
import { Field } from "./Employees";

export function ExpenseTypes() {
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();
  const { data, isLoading } = useQuery<ExpenseType[]>({
    queryKey: ["expense-types"],
    queryFn: () => fetchJSON("/api/expenses/types"),
  });
  const items = data ?? [];

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/expenses/types/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["expense-types"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[800px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            报销类型
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            差旅 / 餐费 / 交通等 — 员工提交报销明细时选用。
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="w-4 h-4" />
          新建类型
        </Button>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Tags}
            title="还没有报销类型"
            action={<Button onClick={() => setOpen(true)}>新建类型</Button>}
          />
        ) : (
          items.map((t, i) => (
            <div
              key={t.id}
              className={`flex items-center gap-4 px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                {t.name}
              </span>
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => del.mutate(t.id)}
                disabled={del.isPending}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          ))
        )}
      </div>

      <TypeCreateModal open={open} onClose={() => setOpen(false)} />
    </main>
  );
}

function TypeCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/expenses/types", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["expense-types"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({ name: String(fd.get("name") || "").trim() });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建报销类型"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="et-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="et-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="名称" required>
          <Input name="name" required placeholder="差旅" />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
