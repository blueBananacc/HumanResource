import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, Pencil, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import {
  StatusPill,
  fmtTime,
  type Attendance,
} from "@/lib/hris";
import { Field } from "./Employees";

const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "present", label: "出勤" },
  { value: "absent", label: "缺勤" },
  { value: "leave", label: "请假" },
  { value: "remote", label: "远程" },
  { value: "holiday", label: "节假日" },
];
const EDIT_STATUS = STATUS_OPTIONS.slice(1);

/** "YYYY-MM" → 该月最后一天 "YYYY-MM-DD";非法输入返回 null(禁用重算)。 */
function monthEnd(month: string): string | null {
  const m = /^(\d{4})-(\d{2})$/.exec(month);
  if (!m) return null;
  const y = Number(m[1]);
  const mo = Number(m[2]);
  if (mo < 1 || mo > 12) return null;
  // day 0 of month (mo+1) = 当月最后一天,本地时区手拼避免 toISOString 跨日
  const last = new Date(y, mo, 0).getDate();
  return `${m[1]}-${m[2]}-${String(last).padStart(2, "0")}`;
}

export function AttendanceAdmin() {
  const [sp, setSp] = useSearchParams();
  const month =
    sp.get("month") ?? new Date().toISOString().slice(0, 7);
  const status = sp.get("status") ?? "";
  const scope = sp.get("scope") ?? "all";
  const [edit, setEdit] = useState<Attendance | null>(null);
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_ATTENDANCE);

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<Attendance>({
      key: ["attendance"],
      url: "/api/attendance",
      params: { month, status, scope },
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

  const summarize = useMutation({
    mutationFn: () => {
      const end = monthEnd(month);
      if (!end) return Promise.reject(new ApiError(400, "月份格式无效"));
      return fetchJSON("/api/attendance/summarize", {
        method: "POST",
        body: JSON.stringify({
          start: `${month}-01`,
          end,
        }),
      });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["attendance"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            考勤记录
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            按天汇总员工的上下班打卡 — 支持手工修正与整月重算。
          </p>
        </div>
        {canManage && (
          <Button
            variant="secondary"
            onClick={() => summarize.mutate()}
            disabled={summarize.isPending || !monthEnd(month)}
          >
            <RefreshCw className="w-4 h-4" />
            {summarize.isPending ? "汇总中…" : "重算本月"}
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Input
          type="month"
          value={month}
          onChange={(e) => setParam("month", e.target.value)}
          className="w-[160px]"
        />
        <Select
          value={status}
          onChange={(v) => setParam("status", v)}
          options={STATUS_OPTIONS}
          className="w-[140px]"
        />
        <Select
          value={scope}
          onChange={(v) => setParam("scope", v)}
          options={[
            { value: "all", label: "全部" },
            { value: "team", label: "我管理的" },
          ]}
          className="w-[140px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          天
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={CalendarClock}
            title="本月还没有考勤记录"
            description="员工在自助端打卡后自动汇总;也可以点右上角整月重算。"
          />
        ) : (
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-6 py-3 font-medium">日期</th>
                <th className="px-4 py-3 font-medium">员工</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">上班</th>
                <th className="px-4 py-3 font-medium">下班</th>
                <th className="px-4 py-3 font-medium text-right">工时</th>
                <th className="px-6 py-3 font-medium text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((a) => (
                <tr
                  key={a.id}
                  className="border-b border-[var(--color-border-row)] last:border-0"
                >
                  <td className="px-6 py-3 font-mono">{a.day}</td>
                  <td className="px-4 py-3 text-[var(--color-text-primary)] font-medium">
                    {a.employee_name}
                    <span className="ml-2 text-[11px] font-normal text-[var(--color-text-tertiary)]">
                      {a.org_unit_name ?? ""}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <StatusPill status={a.status} />
                    {a.source === "manual" && (
                      <span className="ml-1.5 text-[11px] text-[var(--color-text-tertiary)]">
                        手工
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.first_in)}</td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.last_out)}</td>
                  <td className="px-4 py-3 text-right font-mono">
                    {a.work_hours != null ? `${a.work_hours}h` : "—"}
                  </td>
                  <td className="px-6 py-3 text-right">
                    {canManage && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setEdit(a)}
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />

      {edit && (
        <AttendanceEditModal att={edit} onClose={() => setEdit(null)} />
      )}
    </main>
  );
}

function AttendanceEditModal({
  att,
  onClose,
}: {
  att: Attendance;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [status, setStatus] = useState(att.status);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON(`/api/attendance/${att.id}`, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["attendance"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    const toIso = (v: string) =>
      v ? `${att.day}T${v}:00` : null;
    mut.mutate({
      status,
      first_in: toIso(String(fd.get("first_in") || "")),
      last_out: toIso(String(fd.get("last_out") || "")),
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`修正考勤 — ${att.employee_name} ${att.day}`}
      description="手工修正会标记 source=manual,重算时不覆盖手工记录。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="att-edit" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="att-edit" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="状态" required>
          <Select
            value={status}
            onChange={setStatus}
            options={EDIT_STATUS}
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="上班时间">
            <Input
              name="first_in"
              type="time"
              defaultValue={att.first_in?.slice(11, 16) ?? ""}
            />
          </Field>
          <Field label="下班时间">
            <Input
              name="last_out"
              type="time"
              defaultValue={att.last_out?.slice(11, 16) ?? ""}
            />
          </Field>
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
