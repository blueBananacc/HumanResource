import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Banknote,
  Eye,
  Plus,
  Send,
  Trash2,
  Undo2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  fmtMoney,
  type SalarySlip,
} from "@/lib/hris";
import { Field, useEmployeeOptions } from "./Employees";

export function Payroll() {
  const [sp] = useSearchParams();
  const period = sp.get("period") ?? "";
  const employeeId = sp.get("employee_id") ?? "";
  const [createOpen, setCreateOpen] = useState(false);
  const [viewing, setViewing] = useState<SalarySlip | null>(null);
  const qc = useQueryClient();

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<SalarySlip>({
      key: ["salary-slips"],
      url: "/api/payroll/slips",
      params: { period, employee_id: employeeId },
    });
  const items = data?.items ?? [];

  const act = useMutation({
    mutationFn: ({ id, action }: { id: string; action: string }) =>
      fetchJSON(`/api/payroll/slips/${id}/${action}`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["salary-slips"] }),
  });
  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/payroll/slips/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["salary-slips"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            工资条
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            轻量薪酬登记 — 录入/导入工资条,发放后员工在自助端可见。
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4" />
          录入工资条
        </Button>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Banknote}
            title="还没有工资条"
            description="录入后先「发放」,员工才在自助端可见;发错可收回再删。"
            action={
              <Button onClick={() => setCreateOpen(true)}>录入工资条</Button>
            }
          />
        ) : (
          items.map((s, i) => (
            <div
              key={s.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {s.employee_name}
                </span>
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {s.org_unit_name ?? ""}
                </span>
                <span className="font-mono text-[12px] text-[var(--color-text-secondary)]">
                  {s.period}
                </span>
                <StatusPill status={s.status} />
                <div className="flex-1" />
                <span className="text-[12px] font-mono text-[var(--color-text-secondary)]">
                  应发 {fmtMoney(s.gross, s.currency)}
                </span>
                <span className="font-mono font-semibold text-sm text-[var(--color-text-primary)]">
                  实发 {fmtMoney(s.net, s.currency)}
                </span>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setViewing(s)}
                >
                  <Eye className="w-3.5 h-3.5" />
                  明细
                </Button>
                {s.status === "draft" && (
                  <>
                    <Button
                      size="sm"
                      onClick={() =>
                        act.mutate({ id: s.id, action: "issue" })
                      }
                      disabled={act.isPending}
                    >
                      <Send className="w-3.5 h-3.5" />
                      发放
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => del.mutate(s.id)}
                      disabled={del.isPending}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </>
                )}
                {s.status === "issued" && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() =>
                      act.mutate({ id: s.id, action: "retract" })
                    }
                    disabled={act.isPending}
                  >
                    <Undo2 className="w-3.5 h-3.5" />
                    收回
                  </Button>
                )}
                {s.issued_at && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    发放于 {formatRelative(s.issued_at)}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />

      <SlipCreateModal open={createOpen} onClose={() => setCreateOpen(false)} />
      {viewing && <SlipViewModal slip={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

type SlipRow = { label: string; kind: string; amount: string };

function SlipCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const empOptions = useEmployeeOptions();
  const [empId, setEmpId] = useState("");
  const [rows, setRows] = useState<SlipRow[]>([
    { label: "基本工资", kind: "earning", amount: "" },
  ]);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/payroll/slips", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["salary-slips"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!empId) {
      setErr("请选择员工");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    const items = rows
      .filter((r) => r.label.trim() && Number(r.amount) > 0)
      .map((r) => ({
        label: r.label.trim(),
        kind: r.kind,
        amount: Number(r.amount),
      }));
    mut.mutate({
      employee_id: empId,
      period: String(fd.get("period") || ""),
      currency: "CNY",
      items,
      note: String(fd.get("note") || "").trim() || null,
    });
  }

  const kindOptions = [
    { value: "earning", label: "应发项" },
    { value: "deduction", label: "扣减项" },
  ];

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="录入工资条"
      description="明细行自动汇总应发/扣减/实发;创建后为草稿,点「发放」员工才可见。"
      width={620}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="slip-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="slip-create" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="员工" required>
            <Select
              value={empId}
              onChange={setEmpId}
              options={empOptions}
              placeholder="选择员工"
            />
          </Field>
          <Field label="期间" required>
            <Input name="period" type="month" required />
          </Field>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
              明细行
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() =>
                setRows((p) => [...p, { label: "", kind: "earning", amount: "" }])
              }
            >
              <Plus className="w-3.5 h-3.5" />
              加一行
            </Button>
          </div>
          {rows.map((r, i) => (
            <div key={i} className="flex items-center gap-2">
              <Input
                value={r.label}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, label: e.target.value } : x,
                    ),
                  )
                }
                placeholder="项目名(基本工资/社保…)"
                className="flex-1"
              />
              <Select
                value={r.kind}
                onChange={(v) =>
                  setRows((p) =>
                    p.map((x, j) => (j === i ? { ...x, kind: v } : x)),
                  )
                }
                options={kindOptions}
                className="w-[110px]"
              />
              <Input
                value={r.amount}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, amount: e.target.value } : x,
                    ),
                  )
                }
                type="number"
                min="0"
                step="0.01"
                placeholder="金额"
                className="w-[120px]"
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
        <Field label="备注">
          <Input name="note" />
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

function SlipViewModal({
  slip,
  onClose,
}: {
  slip: SalarySlip;
  onClose: () => void;
}) {
  return (
    <Modal
      open
      onClose={onClose}
      title={`${slip.employee_name} · ${slip.period}`}
      footer={<Button onClick={onClose}>关闭</Button>}
    >
      <div className="flex flex-col gap-3 text-[13px] font-body">
        {(slip.items ?? []).map((it, i) => (
          <div key={i} className="flex items-center justify-between">
            <span className="text-[var(--color-text-secondary)]">
              {it.label}
              <span className="ml-2 text-[11px] text-[var(--color-text-tertiary)]">
                {it.kind === "earning" ? "应发" : "扣减"}
              </span>
            </span>
            <span className="font-mono">{fmtMoney(it.amount, slip.currency)}</span>
          </div>
        ))}
        <div className="border-t border-[var(--color-border-row)] pt-3 flex flex-col gap-1.5">
          <Row label="应发合计" value={slip.gross} currency={slip.currency} />
          <Row
            label="扣减合计"
            value={slip.deductions}
            currency={slip.currency}
          />
          <div className="flex items-center justify-between">
            <span className="font-medium text-[var(--color-text-primary)]">
              实发
            </span>
            <span className="font-mono font-semibold text-[15px] text-[var(--color-text-primary)]">
              {fmtMoney(slip.net, slip.currency)}
            </span>
          </div>
        </div>
        {slip.note && (
          <div className="text-[12px] text-[var(--color-text-tertiary)]">
            {slip.note}
          </div>
        )}
      </div>
    </Modal>
  );
}

function Row({
  label,
  value,
  currency,
}: {
  label: string;
  value: number;
  currency: string;
}) {
  return (
    <div className="flex items-center justify-between text-[var(--color-text-secondary)]">
      <span>{label}</span>
      <span className="font-mono">{fmtMoney(value, currency)}</span>
    </div>
  );
}
