import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ClipboardList, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { StatusPill, fmtTime, type AttendanceRequest } from "@/lib/hris";
import { PageError } from "./Employees";

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

export function AttendanceRequests() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const { user, hasPermission } = useAuth();
  const canApprove = hasPermission(PERM.APPROVE_LEAVE);
  const isManagerOnly = user?.role !== "admin" && user?.role !== "hr";
  const scope = sp.get("scope") ?? (isManagerOnly ? "team" : "all");
  const qc = useQueryClient();
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [err, setErr] = useState("");

  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<AttendanceRequest>({
      key: ["att-requests"],
      url: "/api/attendance/requests",
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
      fetchJSON(`/api/attendance/requests/${id}/${action}`, {
        method: "POST",
        body: JSON.stringify({ note: notes[id] || null }),
      }),
    onSuccess: () => {
      setErr("");
      setNotes({});
      qc.invalidateQueries({ queryKey: ["att-requests"] });
      // 批准会补写 checkin 并重算考勤 — 考勤页下次拉新数据
      qc.invalidateQueries({ queryKey: ["attendance"] });
      qc.invalidateQueries({ queryKey: ["checkins"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          补卡审批
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          员工漏打卡的补录申请 — 批准后自动补写打卡并重算当日考勤。
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
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={ClipboardList}
            title="没有补卡申请"
            description="员工在自助端「我的考勤」里提交补卡后会出现在这里。"
          />
        ) : (
          items.map((r, i) => (
            <div
              key={r.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {r.employee_name}
                </span>
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {r.org_unit_name ?? ""}
                </span>
                <StatusPill status={r.status} />
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {formatRelative(r.created_at)}
                </span>
              </div>
              <div className="mt-1.5 text-[13px] text-[var(--color-text-primary)] font-body">
                <span className="font-mono">{r.ts.slice(0, 10)}</span>{" "}
                {fmtTime(r.ts)}{" "}
                <span
                  className={`px-2 py-0.5 rounded-md text-[11px] ${
                    r.kind === "in"
                      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                      : "bg-[var(--color-info-soft)] text-[var(--color-info)]"
                  }`}
                >
                  {r.kind === "in" ? "补上班卡" : "补下班卡"}
                </span>
                {r.reason && (
                  <span className="text-[var(--color-text-secondary)]">
                    {" "}
                    · {r.reason}
                  </span>
                )}
              </div>
              {r.status === "pending" && r.own_request ? (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  本人提交的申请,需其他审批人处理。
                </div>
              ) : r.status === "pending" && canApprove ? (
                <div className="mt-3 flex items-center gap-2">
                  <Input
                    value={notes[r.id] ?? ""}
                    onChange={(e) =>
                      setNotes((p) => ({ ...p, [r.id]: e.target.value }))
                    }
                    placeholder="审批意见(可选)"
                    className="w-[320px] h-8 text-[13px]"
                  />
                  <Button
                    size="sm"
                    onClick={() =>
                      decide.mutate({ id: r.id, action: "approve" })
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
                      decide.mutate({ id: r.id, action: "reject" })
                    }
                    disabled={decide.isPending}
                  >
                    <X className="w-3.5 h-3.5" />
                    驳回
                  </Button>
                </div>
              ) : (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  {r.approver_email && `审批人 ${r.approver_email}`}
                  {r.decision_note && ` · ${r.decision_note}`}
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
