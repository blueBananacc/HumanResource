import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CalendarDays, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { fetchJSON } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { StatusPill, type LeaveApplication } from "@/lib/hris";

const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "pending", label: "待审批" },
  { value: "approved", label: "已批准" },
  { value: "rejected", label: "已驳回" },
  { value: "cancelled", label: "已取消" },
];
const SCOPE_OPTIONS = [
  { value: "all", label: "全部" },
  { value: "team", label: "我管理的" },
  { value: "mine", label: "我自己的" },
];

export function LeaveAdmin() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const scope = sp.get("scope") ?? "all";
  const qc = useQueryClient();
  const [notes, setNotes] = useState<Record<string, string>>({});

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<LeaveApplication>({
      key: ["leave-apps"],
      url: "/api/leave/applications",
      params: { status, scope },
    });
  const items = data?.items ?? [];

  function setParam(k: string, v: string) {
    setSp((prev) => {
      const next = new URLSearchParams(prev);
      if (v) next.set(k, v);
      else next.delete(k);
      next.delete("page");
      return next;
    });
  }

  const decide = useMutation({
    mutationFn: ({
      id,
      action,
    }: {
      id: string;
      action: "approve" | "reject";
    }) =>
      fetchJSON(`/api/leave/applications/${id}/${action}`, {
        method: "POST",
        body: JSON.stringify({ note: notes[id] || null }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["leave-apps"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          请假审批
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          员工请假申请 — 批准后自动扣减额度并写台账。
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Select
          value={status}
          onChange={(v) => setParam("status", v)}
          options={STATUS_OPTIONS}
          className="w-[140px]"
        />
        <Select
          value={scope}
          onChange={(v) => setParam("scope", v)}
          options={SCOPE_OPTIONS}
          className="w-[140px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          条
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={CalendarDays}
            title="没有请假申请"
            description="员工在自助端提交请假后会出现在这里。"
          />
        ) : (
          items.map((a, i) => (
            <div
              key={a.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {a.employee_name}
                </span>
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {a.org_unit_name ?? ""}
                </span>
                <span className="px-2 py-0.5 rounded-md bg-[var(--color-bg-subtle)] text-[11px] font-body text-[var(--color-text-secondary)]">
                  {a.leave_type_name}
                </span>
                <StatusPill status={a.status} />
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {formatRelative(a.created_at)}
                </span>
              </div>
              <div className="mt-1.5 text-[13px] text-[var(--color-text-primary)] font-body">
                {a.start_date} → {a.end_date} ·{" "}
                <span className="font-mono">{a.days}</span> 天
                {a.reason && (
                  <span className="text-[var(--color-text-secondary)]">
                    {" "}
                    · {a.reason}
                  </span>
                )}
              </div>
              {a.status === "pending" ? (
                <div className="mt-3 flex items-center gap-2">
                  <Input
                    value={notes[a.id] ?? ""}
                    onChange={(e) =>
                      setNotes((p) => ({ ...p, [a.id]: e.target.value }))
                    }
                    placeholder="审批意见(可选)"
                    className="w-[320px] h-8 text-[13px]"
                  />
                  <Button
                    size="sm"
                    onClick={() =>
                      decide.mutate({ id: a.id, action: "approve" })
                    }
                    disabled={decide.isPending}
                  >
                    <Check className="w-3.5 h-3.5" />
                    批准
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() =>
                      decide.mutate({ id: a.id, action: "reject" })
                    }
                    disabled={decide.isPending}
                  >
                    <X className="w-3.5 h-3.5" />
                    驳回
                  </Button>
                </div>
              ) : (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  {a.approver_email && `审批人 ${a.approver_email}`}
                  {a.decision_note && ` · ${a.decision_note}`}
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
