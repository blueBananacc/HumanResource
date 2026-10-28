import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCheck, Wallet } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import { fmtMoney, fmtTime, StatusPill, type MeSlip } from "@/lib/hris";

export function MyPayslips() {
  const [viewing, setViewing] = useState<MeSlip | null>(null);
  const { data, isLoading } = useQuery<MeSlip[]>({
    queryKey: ["me-slips"],
    queryFn: () => fetchJSON("/api/me/salary-slips"),
  });
  const items = data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[800px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          我的工资条
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          已发放的工资条;对明细有疑问请联系 HR。
        </p>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Wallet}
            title="还没有工资条"
            description="HR 发放后会出现在这里。"
          />
        ) : (
          items.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setViewing(s)}
              className={`w-full flex items-center gap-4 px-6 py-4 text-left transition-colors hover:bg-[var(--color-bg-muted)] ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <span className="font-mono text-sm text-[var(--color-text-primary)] font-semibold">
                {s.period}
              </span>
              <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                发放 {fmtTime(s.issued_at)}
              </span>
              <StatusPill
                status={s.confirmed_at ? "approved" : "pending"}
                label={s.confirmed_at ? "已签收" : "待签收"}
              />
              <div className="flex-1" />
              <span className="font-mono font-semibold text-sm text-[var(--color-text-primary)]">
                {fmtMoney(s.net, s.currency)}
              </span>
            </button>
          ))
        )}
      </div>

      {viewing && (
        <SlipModal slip={viewing} onClose={() => setViewing(null)} />
      )}
    </main>
  );
}

function SlipModal({
  slip,
  onClose,
}: {
  slip: MeSlip;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: full } = useQuery<MeSlip>({
    queryKey: ["me-slip", slip.id],
    queryFn: () => fetchJSON(`/api/me/salary-slips/${slip.id}`),
    initialData: slip,
  });
  const s = full ?? slip;

  const confirm = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/me/salary-slips/${s.id}/confirm`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-slips"] });
      qc.invalidateQueries({ queryKey: ["me-slip", s.id] });
    },
  });

  return (
    <Modal
      open
      onClose={onClose}
      title={`工资条 · ${s.period}`}
      footer={
        <div className="flex items-center gap-2">
          {s.confirmed_at ? (
            <span className="text-[12px] text-[var(--color-success)] font-body flex items-center gap-1">
              <CheckCheck className="w-3.5 h-3.5" />
              已于 {fmtTime(s.confirmed_at)} 签收
            </span>
          ) : (
            <Button
              onClick={() => confirm.mutate()}
              disabled={confirm.isPending}
            >
              <CheckCheck className="w-4 h-4" />
              确认无误,签收
            </Button>
          )}
          <Button variant="secondary" onClick={onClose}>
            关闭
          </Button>
        </div>
      }
    >
      <div className="flex flex-col gap-3 text-[13px] font-body">
        {(s.items ?? []).length === 0 ? (
          <div className="text-[var(--color-text-tertiary)]">无明细</div>
        ) : (
          (s.items ?? []).map((it, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className="text-[var(--color-text-secondary)]">
                {it.label}
                <span className="ml-2 text-[11px] text-[var(--color-text-tertiary)]">
                  {it.kind === "earning" ? "应发" : "扣减"}
                </span>
              </span>
              <span className="font-mono">
                {fmtMoney(it.amount, s.currency)}
              </span>
            </div>
          ))
        )}
        <div className="border-t border-[var(--color-border-row)] pt-3 flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[var(--color-text-secondary)]">
            <span>应发合计</span>
            <span className="font-mono">{fmtMoney(s.gross, s.currency)}</span>
          </div>
          <div className="flex items-center justify-between text-[var(--color-text-secondary)]">
            <span>扣减合计</span>
            <span className="font-mono">
              {fmtMoney(s.deductions, s.currency)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="font-medium text-[var(--color-text-primary)]">
              实发
            </span>
            <span className="font-mono font-semibold text-[15px] text-[var(--color-text-primary)]">
              {fmtMoney(s.net, s.currency)}
            </span>
          </div>
        </div>
        {s.note && (
          <div className="text-[12px] text-[var(--color-text-tertiary)]">
            {s.note}
          </div>
        )}
      </div>
    </Modal>
  );
}
