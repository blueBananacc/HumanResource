import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheck, MessageSquareWarning } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { ApiError, fetchJSON } from "@/lib/api";

type Row = {
  employee_id: string;
  employee_name: string;
  org_unit_name: string | null;
  /** null=未确认 | confirmed | disputed | resolved(HR 已处理待员工再确认) */
  status: string | null;
  note: string | null;
  resolved_note: string | null;
  updated_at: string | null;
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  disputed: {
    label: "异议中",
    cls: "bg-[var(--color-danger-soft)] text-[var(--color-danger)]",
  },
  resolved: {
    label: "已处理待再确认",
    cls: "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]",
  },
  confirmed: {
    label: "已确认",
    cls: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
  },
};

export function AttendanceConfirmations() {
  const qc = useQueryClient();
  const [month, setMonth] = useState(new Date().toISOString().slice(0, 7));
  const [status, setStatus] = useState("");
  const [resolveRow, setResolveRow] = useState<Row | null>(null);
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");

  const { data, isLoading } = useQuery<Row[]>({
    queryKey: ["att-confirmations", month],
    queryFn: () => fetchJSON(`/api/attendance/confirmations?month=${month}`),
  });

  const resolve = useMutation({
    mutationFn: ({ employeeId, note }: { employeeId: string; note: string }) =>
      fetchJSON(`/api/attendance/confirmations/${employeeId}/resolve`, {
        method: "POST",
        body: JSON.stringify({ month, note: note || null }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["att-confirmations"] });
      setResolveRow(null);
      setNote("");
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "处理失败"),
  });

  const all = data ?? [];
  const rows = status
    ? all.filter((r) => (r.status ?? "none") === status)
    : all;
  const stats = {
    total: all.length,
    confirmed: all.filter((r) => r.status === "confirmed").length,
    disputed: all.filter((r) => r.status === "disputed").length,
    resolved: all.filter((r) => r.status === "resolved").length,
    pending: all.filter((r) => !r.status).length,
  };

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          考勤在线确认
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          员工在「我的考勤」中确认或提出异议;异议处理后员工可再次确认。
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {(
          [
            ["应确认", stats.total, "text-[var(--color-text-primary)]"],
            ["已确认", stats.confirmed, "text-[var(--color-success)]"],
            ["异议中", stats.disputed, "text-[var(--color-danger)]"],
            ["未确认", stats.pending, "text-[var(--color-text-tertiary)]"],
          ] as const
        ).map(([label, v, cls]) => (
          <div
            key={label}
            className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] px-5 py-4 flex flex-col gap-1"
          >
            <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
              {label}
            </span>
            <span className={`font-heading font-semibold text-2xl ${cls}`}>
              {v}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <Input
          type="month"
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="w-[160px]"
        />
        <Select
          value={status}
          onChange={setStatus}
          options={[
            { value: "", label: "全部状态" },
            { value: "disputed", label: "异议中" },
            { value: "resolved", label: "已处理待再确认" },
            { value: "confirmed", label: "已确认" },
            { value: "none", label: "未确认" },
          ]}
          className="w-[160px]"
        />
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
        ) : rows.length === 0 ? (
          <Empty
            icon={ClipboardCheck}
            title="没有符合条件的记录"
            description="员工在「我的考勤」点击确认或提出异议后会出现在这里。"
          />
        ) : (
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-6 py-3 font-medium">员工</th>
                <th className="px-4 py-3 font-medium">部门</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">说明</th>
                <th className="px-4 py-3 font-medium">更新时间</th>
                <th className="px-6 py-3 font-medium text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const st = r.status
                  ? STATUS_LABEL[r.status]
                  : {
                      label: "未确认",
                      cls: "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]",
                    };
                return (
                  <tr
                    key={r.employee_id}
                    className="border-b border-[var(--color-border-row)] last:border-0"
                  >
                    <td className="px-6 py-3">
                      <Link
                        to={`/employees/${r.employee_id}`}
                        className="font-medium text-[var(--color-text-primary)] hover:underline"
                      >
                        {r.employee_name}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                      {r.org_unit_name ?? "—"}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[11px] font-medium ${st?.cls ?? ""}`}
                      >
                        {st?.label ?? r.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-secondary)] max-w-[280px]">
                      {r.note ? (
                        <span className="flex items-start gap-1">
                          <MessageSquareWarning className="w-3.5 h-3.5 mt-0.5 shrink-0 text-[var(--color-warning-text,var(--color-text-tertiary))]" />
                          <span>
                            {r.note}
                            {r.resolved_note && (
                              <span className="block text-[11px] text-[var(--color-text-tertiary)] mt-0.5">
                                处理:{r.resolved_note}
                              </span>
                            )}
                          </span>
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-tertiary)] whitespace-nowrap">
                      {r.updated_at
                        ? r.updated_at.slice(0, 16).replace("T", " ")
                        : "—"}
                    </td>
                    <td className="px-6 py-3 text-right">
                      {r.status === "disputed" && (
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => {
                            setResolveRow(r);
                            setNote("");
                          }}
                        >
                          标记已处理
                        </Button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <Modal
        open={!!resolveRow}
        onClose={() => setResolveRow(null)}
        title={`处理异议 — ${resolveRow?.employee_name ?? ""}`}
        description="填写处理结果(如「已修正 10/12 下班卡」),员工在「我的考勤」可再次确认。"
        footer={
          <>
            <Button variant="secondary" onClick={() => setResolveRow(null)}>
              取消
            </Button>
            <Button
              onClick={() =>
                resolveRow &&
                resolve.mutate({ employeeId: resolveRow.employee_id, note: note.trim() })
              }
              disabled={resolve.isPending}
            >
              {resolve.isPending ? "提交中…" : "标记已处理"}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          {resolveRow?.note && (
            <div className="text-[13px] text-[var(--color-text-secondary)] bg-[var(--color-bg-subtle)] rounded-lg px-3 py-2 font-body">
              员工异议:{resolveRow.note}
            </div>
          )}
          <Input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="处理结果说明(可选)"
          />
        </div>
      </Modal>
    </main>
  );
}
