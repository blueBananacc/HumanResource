import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Receipt, Wallet, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  fmtMoney,
  type ExpenseClaim,
} from "@/lib/hris";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";

const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "pending", label: "待审批" },
  { value: "approved", label: "已批准" },
  { value: "rejected", label: "已驳回" },
  { value: "paid", label: "已支付" },
];

export function Expenses() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [err, setErr] = useState("");
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  const canPay = hasPermission(PERM.MANAGE_EXPENSES);
  // admin/hr/用人经理可批;后端再按汇报线限制到直属下属、并拒绝自批
  const canDecide = hasPermission(PERM.APPROVE_EXPENSES);

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<ExpenseClaim>({
      key: ["expense-claims"],
      url: "/api/expenses/claims",
      params: { status },
    });
  const items = data?.items ?? [];

  const act = useMutation({
    mutationFn: ({ id, action }: { id: string; action: string }) =>
      fetchJSON(`/api/expenses/claims/${id}/${action}`, {
        method: "POST",
        body: JSON.stringify({ note: notes[id] || null }),
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["expense-claims"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          报销审批
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          员工在自助端提交报销单 — 审批通过后由财务标记支付。
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Select
          value={status}
          onChange={(v) =>
            setSp((prev) => {
              const next = new URLSearchParams(prev);
              if (v) next.set("status", v);
              else next.delete("status");
              next.delete("page");
              return next;
            })
          }
          options={STATUS_OPTIONS}
          className="w-[140px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          单
        </span>
        {err && (
          <span className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </span>
        )}
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Receipt}
            title="没有报销单"
            description="员工在自助端「报销」页提交后出现在这里。"
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
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {c.employee_name} · {c.org_unit_name ?? ""}
                </span>
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
              {c.status === "pending" && c.own_request && (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  本人提交的报销,需其他审批人处理。
                </div>
              )}
              {c.status === "pending" && !c.own_request && canDecide && (
                <div className="mt-3 flex items-center gap-2">
                  <Input
                    value={notes[c.id] ?? ""}
                    onChange={(e) =>
                      setNotes((p) => ({ ...p, [c.id]: e.target.value }))
                    }
                    placeholder="审批意见(可选)"
                    className="w-[300px] h-8 text-[13px]"
                  />
                  <Button
                    size="sm"
                    onClick={() =>
                      act.mutate({ id: c.id, action: "approve" })
                    }
                    disabled={act.isPending}
                  >
                    <Check className="w-3.5 h-3.5" />
                    批准
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => act.mutate({ id: c.id, action: "reject" })}
                    disabled={act.isPending}
                  >
                    <X className="w-3.5 h-3.5" />
                    驳回
                  </Button>
                </div>
              )}
              {c.status === "approved" && canPay && (
                <div className="mt-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => act.mutate({ id: c.id, action: "pay" })}
                    disabled={act.isPending}
                  >
                    <Wallet className="w-3.5 h-3.5" />
                    标记支付
                  </Button>
                </div>
              )}
              {c.decision_note && c.status !== "pending" && (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  审批意见: {c.decision_note}
                </div>
              )}
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
    </main>
  );
}
