import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Tag, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { fetchJSON, ApiError } from "@/lib/api";
import { type LeaveType } from "@/lib/hris";
import { Field } from "./Employees";

export function LeaveTypes() {
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();
  const { data, isLoading } = useQuery<LeaveType[]>({
    queryKey: ["leave-types"],
    queryFn: () => fetchJSON("/api/leave/types"),
  });
  const items = data ?? [];

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/leave/types/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["leave-types"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            假期类型
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            年假 / 病假 / 事假等 — 员工按类型申请,HR 按类型分配额度。
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
            icon={Tag}
            title="还没有假期类型"
            description="先建「年假」「病假」等类型,再分配额度、员工就能请假了。"
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
              <span
                className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                  t.is_paid
                    ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                    : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                }`}
              >
                {t.is_paid ? "带薪" : "无薪"}
              </span>
              {t.is_comp && (
                <span className="px-2 py-0.5 rounded-md text-[11px] font-body bg-[var(--color-info-soft)] text-[var(--color-info)]">
                  调休
                </span>
              )}
              <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                年度默认 {t.annual_days} 天
              </span>
              <div className="flex-1" />
              <span className="text-[12px] font-mono text-[var(--color-text-tertiary)]">
                已分配 {t.allocated_days} · 已用 {t.used_days}
              </span>
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
      fetchJSON("/api/leave/types", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-types"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      annual_days: Number(fd.get("annual_days") || 0),
      is_paid: fd.get("is_paid") === "on",
      is_comp: fd.get("is_comp") === "on",
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建假期类型"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="lt-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="lt-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="名称" required>
          <Input name="name" required placeholder="年假" />
        </Field>
        <Field label="年度默认天数">
          <Input
            name="annual_days"
            type="number"
            step="0.5"
            min="0"
            defaultValue={10}
          />
        </Field>
        <label className="flex items-center gap-2 text-[13px] font-body text-[var(--color-text-primary)]">
          <input type="checkbox" name="is_paid" defaultChecked />
          带薪假
        </label>
        <label className="flex items-center gap-2 text-[13px] font-body text-[var(--color-text-primary)]">
          <input type="checkbox" name="is_comp" />
          调休类型(加班申请批准后把额度结算到这类类型)
        </label>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
