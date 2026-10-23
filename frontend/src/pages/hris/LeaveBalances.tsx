import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, Trash2, Wallet } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { Pagination } from "@/components/ui/pagination";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import {
  type LeaveBalance,
  type LeaveAllocation,
  type LeaveType,
} from "@/lib/hris";
import { Field, useEmployeeOptions } from "./Employees";

const REF_TYPE_LABEL: Record<string, string> = {
  allocation: "额度发放",
  allocation_adjust: "额度调整",
  allocation_revoke: "额度收回",
  application: "请假扣减",
  application_cancel: "撤销返还",
};

export function LeaveBalances() {
  const [sp] = useSearchParams();
  const employeeFilter = sp.get("employee_id") ?? "";
  const [allocOpen, setAllocOpen] = useState(false);
  const [adjusting, setAdjusting] = useState<LeaveBalance | null>(null);
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_LEAVE);
  const confirm = useConfirm();
  const qc = useQueryClient();
  const year = new Date().getFullYear();

  const { data: balances, isLoading } = useQuery<LeaveBalance[]>({
    queryKey: ["leave-balances", year, employeeFilter],
    queryFn: () =>
      fetchJSON(
        `/api/leave/balances?year=${year}${employeeFilter ? `&employee_id=${employeeFilter}` : ""}`,
      ),
  });
  const { data: ledger, page, pageCount, total, goto } =
    usePagedQuery<import("@/lib/hris").LeaveLedgerEntry>({
      key: ["leave-ledger"],
      url: "/api/leave/ledger",
      params: { year },
    });

  const revoke = useMutation({
    mutationFn: (allocationId: string) =>
      fetchJSON(`/api/leave/allocations/${allocationId}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-balances"] });
      qc.invalidateQueries({ queryKey: ["leave-ledger"] });
    },
  });

  async function onRevoke(b: LeaveBalance) {
    const ok = await confirm({
      title: "收回该额度?",
      description: `将收回 ${b.employee_name} 的${b.leave_type_name} ${b.allocated} 天,台账记一笔负向流水。已扣减的请假不受影响。`,
      confirmLabel: "收回",
      tone: "danger",
    });
    if (ok) revoke.mutate(b.allocation_id);
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            假期额度
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            {year} 年 — 分配 / 已用 / 在途 / 余额,下方为台账流水。
          </p>
        </div>
        {canManage && (
          <Button onClick={() => setAllocOpen(true)}>
            <Plus className="w-4 h-4" />
            分配额度
          </Button>
        )}
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : !balances || balances.length === 0 ? (
          <Empty
            icon={Wallet}
            title="还没有额度记录"
            description="先创建假期类型,再为员工分配年度额度。"
            action={
              canManage ? (
                <Button onClick={() => setAllocOpen(true)}>分配额度</Button>
              ) : undefined
            }
          />
        ) : (
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-6 py-3 font-medium">员工</th>
                <th className="px-4 py-3 font-medium">假期类型</th>
                <th className="px-4 py-3 font-medium text-right">分配</th>
                <th className="px-4 py-3 font-medium text-right">已用</th>
                <th className="px-4 py-3 font-medium text-right">审批中</th>
                <th className="px-6 py-3 font-medium text-right">余额</th>
                {canManage && <th className="px-6 py-3 font-medium text-right" />}
              </tr>
            </thead>
            <tbody>
              {balances.map((b, i) => (
                <tr
                  key={b.allocation_id}
                  className={
                    i < balances.length - 1
                      ? "border-b border-[var(--color-border-row)]"
                      : ""
                  }
                >
                  <td className="px-6 py-3 text-[var(--color-text-primary)] font-medium">
                    {b.employee_name}
                  </td>
                  <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                    {b.leave_type_name}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {b.allocated}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">{b.used}</td>
                  <td className="px-4 py-3 text-right font-mono text-[var(--color-warning-text)]">
                    {b.pending}
                  </td>
                  <td className="px-6 py-3 text-right font-mono font-semibold text-[var(--color-text-primary)]">
                    {b.balance}
                  </td>
                  {canManage && (
                    <td className="px-6 py-3 text-right whitespace-nowrap">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setAdjusting(b)}
                        title="调整额度"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onRevoke(b)}
                        disabled={revoke.isPending}
                        title="收回额度"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <section className="flex flex-col gap-3">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
          台账流水
        </h3>
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          {(ledger?.items ?? []).length === 0 ? (
            <div className="p-8 text-center text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无流水 — 分配和请假审批都会写入台账。
            </div>
          ) : (
            (ledger?.items ?? []).map((l, i, arr) => (
              <div
                key={l.id}
                className={`flex items-center gap-4 px-6 py-3 text-[13px] font-body ${
                  i < arr.length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <span
                  className={`font-mono font-semibold w-16 text-right ${
                    l.delta_days >= 0
                      ? "text-[var(--color-success)]"
                      : "text-[var(--color-danger)]"
                  }`}
                >
                  {l.delta_days >= 0 ? "+" : ""}
                  {l.delta_days}
                </span>
                <span className="text-[var(--color-text-primary)] font-medium">
                  {l.employee_name}
                </span>
                <span className="text-[var(--color-text-secondary)]">
                  {l.leave_type_name} · {l.year}
                </span>
                <span className="text-[var(--color-text-tertiary)]">
                  {REF_TYPE_LABEL[l.ref_type] ?? l.ref_type}
                  {l.note ? ` · ${l.note}` : ""}
                </span>
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)]">
                  {l.created_at.slice(0, 10)}
                </span>
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
      </section>

      <AllocationModal open={allocOpen} onClose={() => setAllocOpen(false)} />
      {adjusting && (
        <AdjustModal balance={adjusting} onClose={() => setAdjusting(null)} />
      )}
    </main>
  );
}

function AdjustModal({
  balance,
  onClose,
}: {
  balance: LeaveBalance;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON<LeaveAllocation>(
        `/api/leave/allocations/${balance.allocation_id}`,
        { method: "PUT", body: JSON.stringify(body) },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-balances"] });
      qc.invalidateQueries({ queryKey: ["leave-ledger"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "调整失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      days: Number(fd.get("days")),
      note: String(fd.get("note") || "").trim() || null,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`调整额度 — ${balance.employee_name} ${balance.leave_type_name}`}
      description="差额会记入台账流水,余额实时重算;已用/在途的请假不受影响。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="alloc-adjust" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="alloc-adjust" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="额度天数" required>
          <Input
            name="days"
            type="number"
            step="0.5"
            min="0"
            required
            defaultValue={balance.allocated}
          />
        </Field>
        <Field label="备注">
          <Input name="note" placeholder="例如:入职不满一年折算" />
        </Field>
        {balance.used + balance.pending > 0 && (
          <div className="text-[13px] text-[var(--color-warning-text)] font-body">
            已用 {balance.used} 天 · 审批中 {balance.pending}{" "}
            天,调低额度可能出现负余额。
          </div>
        )}
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}

function AllocationModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const empOptions = useEmployeeOptions();
  const { data: types } = useQuery<LeaveType[]>({
    queryKey: ["leave-types"],
    queryFn: () => fetchJSON("/api/leave/types"),
  });
  const [empId, setEmpId] = useState("");
  const [typeId, setTypeId] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON<LeaveAllocation>("/api/leave/allocations", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-balances"] });
      qc.invalidateQueries({ queryKey: ["leave-ledger"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "分配失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!empId || !typeId) {
      setErr("请选择员工和假期类型");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      employee_id: empId,
      leave_type_id: typeId,
      year: Number(fd.get("year")),
      days: Number(fd.get("days")),
      note: String(fd.get("note") || "").trim() || null,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="分配假期额度"
      description="同一员工/类型/年份仅一笔额度,重复发放会提示已存在 — 可在列表中调整或收回。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="alloc" disabled={mut.isPending}>
            {mut.isPending ? "分配中…" : "分配"}
          </Button>
        </>
      }
    >
      <form id="alloc" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="员工" required>
          <Select
            value={empId}
            onChange={setEmpId}
            options={empOptions}
            placeholder="选择员工"
          />
        </Field>
        <Field label="假期类型" required>
          <Select
            value={typeId}
            onChange={setTypeId}
            options={(types ?? []).map((t) => ({
              value: t.id,
              label: `${t.name}(${t.is_paid ? "带薪" : "无薪"})`,
            }))}
            placeholder="选择类型"
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="年份" required>
            <Input
              name="year"
              type="number"
              required
              defaultValue={new Date().getFullYear()}
            />
          </Field>
          <Field label="天数" required>
            <Input
              name="days"
              type="number"
              step="0.5"
              min="0"
              required
              placeholder="10"
            />
          </Field>
        </div>
        <Field label="备注">
          <Input name="note" placeholder="例如:年度额度" />
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
