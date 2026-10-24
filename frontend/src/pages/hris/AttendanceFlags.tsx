import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, RotateCcw, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import {
  StatusPill,
  type AttendanceFlag,
  type AttendancePolicy,
  type MonthlySummaryRow,
  type WorkCalendar,
} from "@/lib/hris";
import { Field, PageError } from "./Employees";

const KIND_OPTIONS = [
  { value: "", label: "全部类型" },
  { value: "late_in", label: "迟到" },
  { value: "early_out", label: "早退" },
];
const EXCUSED_OPTIONS = [
  { value: "", label: "全部" },
  { value: "false", label: "未免计" },
  { value: "true", label: "已免计" },
];
const SCOPE_OPTIONS = [
  { value: "all", label: "全部" },
  { value: "team", label: "我管理的" },
  { value: "mine", label: "我自己的" },
];

export function AttendanceFlagsPage() {
  const [sp, setSp] = useSearchParams();
  const tab = sp.get("tab") ?? "flags";
  const { user, hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_ATTENDANCE);
  const isManagerOnly = user?.role !== "admin" && user?.role !== "hr";
  const scope = sp.get("scope") ?? (isManagerOnly ? "team" : "all");
  const kind = sp.get("kind") ?? "";
  const excused = sp.get("excused") ?? "";
  const month = sp.get("month") ?? new Date().toISOString().slice(0, 7);
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [notes, setNotes] = useState<Record<string, string>>({});

  function setParam(k: string, v: string) {
    setSp((prev) => {
      const next = new URLSearchParams(prev);
      if (v) next.set(k, v);
      else next.delete(k);
      if (k !== "tab") next.delete("page");
      return next;
    });
  }

  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<AttendanceFlag>({
      key: ["att-flags"],
      url: "/api/attendance/flags",
      params: { month, kind, excused, scope },
      enabled: tab === "flags",
    });
  const items = data?.items ?? [];

  const { data: summary, isError: sumErr, error: sumError } =
    useQuery<MonthlySummaryRow[]>({
      queryKey: ["att-monthly", month, scope],
      queryFn: () =>
        fetchJSON(
          `/api/attendance/monthly-summary?month=${month}&scope=${scope === "mine" ? "team" : scope}`,
        ),
      enabled: tab === "summary",
    });

  const excuse = useMutation({
    mutationFn: ({ id, un }: { id: string; un?: boolean }) =>
      fetchJSON(`/api/attendance/flags/${id}/${un ? "unexcuse" : "excuse"}`, {
        method: "POST",
        body: JSON.stringify({ note: notes[id] || null }),
      }),
    onSuccess: () => {
      setErr("");
      setNotes({});
      qc.invalidateQueries({ queryKey: ["att-flags"] });
      qc.invalidateQueries({ queryKey: ["att-monthly"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          考勤异常
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          迟到/早退台账、月度工时汇总与考勤规则 — 异常由系统按考勤策略自动判定,可免计。
        </p>
      </div>

      <div className="flex items-center gap-2">
        {[
          { v: "flags", label: "异常记录" },
          { v: "summary", label: "月度汇总" },
          ...(canManage ? [{ v: "policy", label: "考勤规则" }] : []),
        ].map((t) => (
          <button
            key={t.v}
            type="button"
            onClick={() => setParam("tab", t.v)}
            className={`px-4 h-9 rounded-lg text-[13px] font-body border transition-colors ${
              tab === t.v
                ? "bg-[var(--color-text-primary)] text-white border-transparent"
                : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border-subtle)] hover:border-[var(--color-text-tertiary)]"
            }`}
          >
            {t.label}
          </button>
        ))}
        <div className="flex-1" />
        <Input
          type="month"
          value={month}
          onChange={(e) => setParam("month", e.target.value)}
          className="w-[150px] h-9 text-[13px]"
        />
      </div>

      {tab === "flags" && (
        <>
          <div className="flex items-center gap-3">
            <Select
              value={kind}
              onChange={(v) => setParam("kind", v)}
              options={KIND_OPTIONS}
              className="w-[130px]"
            />
            <Select
              value={excused}
              onChange={(v) => setParam("excused", v)}
              options={EXCUSED_OPTIONS}
              className="w-[130px]"
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
                icon={AlertTriangle}
                title="没有考勤异常"
                description="打卡超出考勤规则宽限后会自动出现在这里。"
              />
            ) : (
              items.map((f, i) => (
                <div
                  key={f.id}
                  className={`px-6 py-4 ${
                    i < items.length - 1
                      ? "border-b border-[var(--color-border-row)]"
                      : ""
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                      {f.employee_name}
                    </span>
                    <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                      {f.org_unit_name ?? ""}
                    </span>
                    <StatusPill status={f.kind} />
                    <span className="font-mono text-[13px] text-[var(--color-text-primary)]">
                      {f.day}
                    </span>
                    <span className="text-[13px] font-semibold text-[var(--color-warning-text)] font-body">
                      {f.minutes} 分钟
                    </span>
                    {f.excused && (
                      <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-success-soft)] text-[var(--color-success)] font-body">
                        已免计{f.excused_note ? `:${f.excused_note}` : ""}
                      </span>
                    )}
                    <div className="flex-1" />
                    {!f.excused ? (
                      <div className="flex items-center gap-2">
                        <Input
                          value={notes[f.id] ?? ""}
                          onChange={(e) =>
                            setNotes((p) => ({ ...p, [f.id]: e.target.value }))
                          }
                          placeholder="免计原因(可选)"
                          className="w-[200px] h-8 text-[13px]"
                        />
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => excuse.mutate({ id: f.id })}
                          disabled={excuse.isPending}
                        >
                          <Check className="w-3.5 h-3.5" />
                          免计
                        </Button>
                      </div>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => excuse.mutate({ id: f.id, un: true })}
                        disabled={excuse.isPending}
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        撤销免计
                      </Button>
                    )}
                  </div>
                  {f.detail?.expected && (
                    <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                      期望 {f.detail.expected} · 实际 {f.detail.actual}
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
        </>
      )}

      {tab === "summary" && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          {sumErr ? (
            <PageError error={sumError} />
          ) : !summary ? (
            <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
              加载中…
            </div>
          ) : summary.length === 0 ? (
            <Empty
              icon={AlertTriangle}
              title="没有汇总数据"
              description="先在「出勤记录」里跑一次按段汇总,或等有打卡流水。"
            />
          ) : (
            <table className="w-full text-[13px] font-body">
              <thead>
                <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                  <th className="px-6 py-3 font-medium">员工</th>
                  <th className="px-4 py-3 font-medium text-right">出勤</th>
                  <th className="px-4 py-3 font-medium text-right">缺勤</th>
                  <th className="px-4 py-3 font-medium text-right">请假</th>
                  <th className="px-4 py-3 font-medium text-right">工时</th>
                  <th className="px-4 py-3 font-medium text-right">迟到</th>
                  <th className="px-4 py-3 font-medium text-right">早退</th>
                  <th className="px-6 py-3 font-medium text-right">批准加班</th>
                </tr>
              </thead>
              <tbody>
                {summary.map((r) => (
                  <tr
                    key={r.employee_id}
                    className="border-b border-[var(--color-border-row)] last:border-0"
                  >
                    <td className="px-6 py-3">
                      <div className="font-medium text-[var(--color-text-primary)]">
                        {r.employee_name}
                      </div>
                      <div className="text-[11px] text-[var(--color-text-tertiary)]">
                        {r.org_unit_name ?? ""}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{r.present}</td>
                    <td className="px-4 py-3 text-right font-mono text-[var(--color-danger)]">
                      {r.absent || "—"}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{r.leave || "—"}</td>
                    <td className="px-4 py-3 text-right font-mono">
                      {r.work_hours}h
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {r.late_count > 0 ? (
                        <span className="text-[var(--color-warning-text)]">
                          {r.late_count} 次 / {r.late_minutes}min
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {r.early_count || "—"}
                    </td>
                    <td className="px-6 py-3 text-right font-mono">
                      {r.overtime_hours > 0 ? `${r.overtime_hours}h` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "policy" && canManage && <PolicyTab />}
    </main>
  );
}

function PolicyTab() {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const { data: policy } = useQuery<AttendancePolicy>({
    queryKey: ["att-policy"],
    queryFn: () => fetchJSON("/api/attendance/policy"),
  });
  const { data: cal } = useQuery<WorkCalendar>({
    queryKey: ["work-calendar"],
    queryFn: () => fetchJSON("/api/work-calendar"),
  });

  return (
    <div className="flex flex-col gap-6">
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4">
        <div>
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            迟到 / 早退规则
          </h3>
          <p className="text-[12px] text-[var(--color-text-secondary)] font-body mt-1">
            期望作息在「节假日历」里配置
            {cal ? `(当前 ${cal.work_start.slice(0, 5)} ~ ${cal.work_end.slice(0, 5)})` : ""}
            ;宽限分钟内不算异常。
          </p>
        </div>
        {policy ? (
          <PolicyForm
            key={JSON.stringify(policy)}
            policy={policy}
            onSaved={() => {
              qc.invalidateQueries({ queryKey: ["att-policy"] });
              setErr("");
            }}
            onError={setErr}
          />
        ) : (
          <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        )}
        {err && (
          <span className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </span>
        )}
      </section>
    </div>
  );
}

function PolicyForm({
  policy,
  onSaved,
  onError,
}: {
  policy: AttendancePolicy;
  onSaved: () => void;
  onError: (s: string) => void;
}) {
  const [graceIn, setGraceIn] = useState(String(policy.grace_in_minutes));
  const [graceOut, setGraceOut] = useState(String(policy.grace_out_minutes));
  const [track, setTrack] = useState(policy.track_late_early);
  const [notify, setNotify] = useState(policy.notify_on_flag);
  const [ipEnabled, setIpEnabled] = useState(policy.ip_check_enabled);
  const [ips, setIps] = useState((policy.allowed_ips ?? []).join("\n"));

  const mut = useMutation({
    mutationFn: () =>
      fetchJSON("/api/attendance/policy", {
        method: "PUT",
        body: JSON.stringify({
          grace_in_minutes: Number(graceIn) || 0,
          grace_out_minutes: Number(graceOut) || 0,
          track_late_early: track,
          notify_on_flag: notify,
          ip_check_enabled: ipEnabled,
          allowed_ips: ips
            .split(/[\n,;]+/)
            .map((s) => s.trim())
            .filter(Boolean),
        }),
      }),
    onSuccess: () => {
      onError("");
      onSaved();
    },
    onError: (e) =>
      onError(e instanceof ApiError ? e.message : "保存失败"),
  });

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-4 max-w-[560px]">
        <Field label="上班宽限(分钟)">
          <Input
            type="number"
            min="0"
            value={graceIn}
            onChange={(e) => setGraceIn(e.target.value)}
          />
        </Field>
        <Field label="下班宽限(分钟)">
          <Input
            type="number"
            min="0"
            value={graceOut}
            onChange={(e) => setGraceOut(e.target.value)}
          />
        </Field>
      </div>
      <div className="flex items-center gap-6">
        <Toggle
          checked={track}
          onChange={setTrack}
          label="记录迟到/早退异常"
        />
        <Toggle
          checked={notify}
          onChange={setNotify}
          label="异常时通知员工本人"
        />
      </div>
      <div className="flex flex-col gap-2 max-w-[560px]">
        <Toggle
          checked={ipEnabled}
          onChange={setIpEnabled}
          label="打卡 IP 白名单"
        />
        {ipEnabled && (
          <>
            <textarea
              value={ips}
              onChange={(e) => setIps(e.target.value)}
              rows={3}
              placeholder={"每行一个 IP 或 CIDR,例如:\n203.0.113.8\n10.0.0.0/8"}
              className="w-full rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-[13px] font-mono focus:outline-none focus:ring-2 focus:ring-[var(--color-focus-ring,#2563EB)]"
            />
            <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
              留空等于未启用(不会拦截打卡);开启后名单外 IP 打卡会被拒绝。
            </span>
          </>
        )}
      </div>
      <div>
        <Button onClick={() => mut.mutate()} disabled={mut.isPending}>
          <Save className="w-4 h-4" />
          {mut.isPending ? "保存中…" : "保存规则"}
        </Button>
        {mut.isSuccess && (
          <span className="ml-3 text-[13px] text-[var(--color-success)] font-body">
            已保存 — 存量考勤重算后按新规则出账
          </span>
        )}
      </div>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-[13px] font-body text-[var(--color-text-primary)]"
    >
      <span
        className={`w-9 h-5 rounded-full relative transition-colors ${
          checked ? "bg-[var(--color-text-primary)]" : "bg-[var(--color-border-subtle)]"
        }`}
      >
        <span
          className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
            checked ? "left-[18px]" : "left-0.5"
          }`}
        />
      </span>
      {label}
    </button>
  );
}
