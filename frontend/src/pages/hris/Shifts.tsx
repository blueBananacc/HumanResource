import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeftRight,
  CalendarDays,
  Check,
  LayoutTemplate,
  Plus,
  ShieldCheck,
  Sparkles,
  UserCheck,
  UserPlus,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { cn } from "@/lib/utils";
import { Field } from "./Employees";

type Shift = {
  id: string;
  name: string;
  time_range: string;
  required: number;
  color: string | null;
  active: boolean;
  start_min: number;
  end_min: number;
};
type Assignment = {
  id: string;
  employee_id: string;
  employee_name: string;
  shift_id: string;
  shift_name: string;
  day: string;
  source: string;
  note: string | null;
};
type Gap = {
  day: string;
  shift_id: string;
  shift_name: string;
  required: number;
  covered: number;
  on_leave: { assignment_id: string; employee_id: string; employee_name: string }[];
  missing: number;
};
type SwapReq = {
  id: string;
  day: string;
  requester_employee_id: string;
  requester_name: string;
  target_employee_id: string;
  target_name: string;
  mode: string;
  status: string;
  requester_note: string | null;
  peer_note: string | null;
  decision_note: string | null;
  created_at: string;
};
type Violation = {
  employee_id: string;
  employee_name: string;
  rule: string;
  severity: string;
  day: string;
  detail: string;
};
type ShiftRules = {
  max_consecutive_days: number;
  max_weekly_hours: number;
  min_rest_minutes: number;
  respect_availability: boolean;
};
type ScheduleTemplate = {
  id: string;
  name: string;
  period_days: number;
  pattern: { offset: number; shift_id: string | null }[];
  active: boolean;
};
type PlannableEmployee = {
  id: string;
  name: string;
  employee_no: string;
  org_unit_id: string | null;
};

const RULE_LABEL: Record<string, string> = {
  consecutive_days: "连班超限",
  weekly_hours: "周工时超限",
  rest_gap: "休息间隔不足",
  availability: "可用性冲突",
};
const SWAP_STATUS_LABEL: Record<string, string> = {
  pending_peer: "待对方确认",
  pending_approval: "待审批",
  approved: "已批准",
  peer_declined: "对方拒绝",
  rejected: "已驳回",
  cancelled: "已撤回",
};

function daysOfMonth(month: string): string[] {
  const [y, m] = month.split("-").map(Number);
  const n = new Date(y, m, 0).getDate();
  return Array.from({ length: n }, (_, i) =>
    `${month}-${String(i + 1).padStart(2, "0")}`);
}

const WEEK = ["日", "一", "二", "三", "四", "五", "六"];

type PageTab = "board" | "swaps" | "compliance" | "templates";

export function ShiftsPage() {
  const [sp, setSp] = useSearchParams();
  const month = sp.get("month") ?? new Date().toISOString().slice(0, 7);
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_ATTENDANCE);
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [showDefs, setShowDefs] = useState(false);
  const [tab, setTab] = useState<PageTab>("board");
  const [assignOpen, setAssignOpen] = useState(false);

  const { data: shifts } = useQuery<Shift[]>({
    queryKey: ["shifts"],
    queryFn: () => fetchJSON("/api/shifts"),
  });
  const { data: assigns } = useQuery<{ items: Assignment[] }>({
    queryKey: ["shift-assigns", month],
    queryFn: () =>
      fetchJSON(`/api/shifts/assignments?month=${month}&limit=200`),
  });
  const { data: gaps } = useQuery<{ gaps: Gap[] }>({
    queryKey: ["shift-gaps", month],
    queryFn: () => fetchJSON(`/api/shifts/gaps?month=${month}`),
  });

  const bySlot = useMemo(() => {
    const m = new Map<string, Assignment[]>();
    for (const a of assigns?.items ?? []) {
      const k = `${a.day}|${a.shift_id}`;
      m.set(k, [...(m.get(k) ?? []), a]);
    }
    return m;
  }, [assigns]);
  const gapMap = useMemo(
    () => new Map((gaps?.gaps ?? []).map((g) => [`${g.day}|${g.shift_id}`, g])),
    [gaps],
  );

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["shift-assigns"] });
    qc.invalidateQueries({ queryKey: ["shift-gaps"] });
  };
  const generate = useMutation({
    mutationFn: (): Promise<{ planned: number; understaffed: unknown[] }> =>
      fetchJSON("/api/shifts/generate", {
        method: "POST",
        body: JSON.stringify({ month, dry_run: false }),
      }),
    onSuccess: (d: { planned: number; understaffed: unknown[] }) => {
      setErr("");
      setMsg(`已生成 ${d.planned} 条排班` + (d.understaffed.length ? `,${d.understaffed.length} 个槽位人手不足` : ""));
      invalidate();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "生成失败"),
  });
  const cover = useMutation({
    mutationFn: (): Promise<{ covered: unknown[]; filled: unknown[]; remaining: unknown[] }> =>
      fetchJSON("/api/shifts/auto-cover", {
        method: "POST",
        body: JSON.stringify({ month }),
      }),
    onSuccess: (d: { covered: unknown[]; filled: unknown[]; remaining: unknown[] }) => {
      setErr("");
      setMsg(`替班 ${d.covered.length} 人次 · 补缺 ${d.filled.length} 人` +
        (d.remaining.length ? ` · ${d.remaining.length} 项仍缺` : ""));
      invalidate();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "替班失败"),
  });

  const days = daysOfMonth(month);
  const activeShifts = (shifts ?? []).filter((s) => s.active);

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1400px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          排班
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          按工作日历 × 班次需求自动生成;已批请假自动跳过,缺口可一键找替班。
        </p>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Input
          type="month"
          value={month}
          onChange={(e) =>
            setSp((p) => {
              const n = new URLSearchParams(p);
              n.set("month", e.target.value);
              return n;
            })
          }
          className="w-[150px] h-9 text-[13px]"
        />
        <div className="flex rounded-lg border border-[var(--color-border-subtle)] overflow-hidden">
          {(
            [
              ["board", "排班看板"],
              ["swaps", "换班审批"],
              ["compliance", "合规体检"],
              ["templates", "模板与循环"],
            ] as [PageTab, string][]
          ).map(([k, label]) => (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={cn(
                "px-3 py-1.5 text-[12px] font-body transition-colors",
                tab === k
                  ? "bg-[var(--color-accent)] text-white"
                  : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-canvas)]",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        {msg && <span className="text-[13px] text-[var(--color-success)] font-body">{msg}</span>}
        {err && <span className="text-[13px] text-[var(--color-danger)] font-body">{err}</span>}
      </div>

      {tab === "board" && canManage && (
        <div className="flex items-center gap-2 flex-wrap">
          <Button size="sm" variant="secondary" onClick={() => setShowDefs((v) => !v)}>
            <CalendarDays className="w-3.5 h-3.5" />
            班次管理
          </Button>
          <Button
            size="sm"
            onClick={() => generate.mutate()}
            disabled={generate.isPending || activeShifts.length === 0}
          >
            <Sparkles className="w-3.5 h-3.5" />
            生成排班
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => cover.mutate()}
            disabled={cover.isPending}
          >
            <UserCheck className="w-3.5 h-3.5" />
            自动找替班
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setAssignOpen(true)}
            disabled={activeShifts.length === 0}
          >
            <UserPlus className="w-3.5 h-3.5" />
            手工排班
          </Button>
        </div>
      )}

      {tab === "board" && showDefs && canManage && (
        <ShiftDefs onChanged={() => qc.invalidateQueries({ queryKey: ["shifts"] })} />
      )}

      {tab === "swaps" && <SwapsPanel canManage={canManage} />}
      {tab === "compliance" && <CompliancePanel month={month} canManage={canManage} />}
      {tab === "templates" && (
        <TemplatesPanel canManage={canManage} onApplied={() => invalidate()} />
      )}

      {tab === "board" && (activeShifts.length === 0 ? (
        <Empty
          icon={CalendarDays}
          title="还没有班次"
          description="先在「班次管理」里建早班/晚班等班次定义,再生成排班。"
        />
      ) : (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-auto">
          <table className="w-full text-[12px] font-body border-collapse min-w-[720px]">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-4 py-2.5 font-medium w-[110px] sticky left-0 bg-white">日期</th>
                {activeShifts.map((s) => (
                  <th key={s.id} className="px-3 py-2.5 font-medium">
                    {s.name}
                    <span className="ml-1.5 text-[10px] font-mono">{s.time_range}</span>
                    <span className="ml-1.5 text-[10px]">需{s.required}人</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {days.map((d) => {
                const dow = WEEK[new Date(d + "T00:00:00").getDay()];
                const weekend = dow === "六" || dow === "日";
                return (
                  <tr
                    key={d}
                    className={cn(
                      "border-b border-[var(--color-border-row)] last:border-0",
                      weekend && "bg-[var(--color-bg-canvas)]/60",
                    )}
                  >
                    <td className="px-4 py-2 sticky left-0 bg-inherit font-mono whitespace-nowrap">
                      {d.slice(5)} 周{dow}
                    </td>
                    {activeShifts.map((s) => {
                      const cell = bySlot.get(`${d}|${s.id}`) ?? [];
                      const g = gapMap.get(`${d}|${s.id}`);
                      return (
                        <td key={s.id} className={cn("px-3 py-1.5 align-top", g && "bg-[var(--color-danger-soft,#fef2f2)]/50")}>
                          <div className="flex flex-wrap gap-1">
                            {cell.map((a) => (
                              <span
                                key={a.id}
                                title={a.note ?? a.source}
                                className={cn(
                                  "px-1.5 py-0.5 rounded text-[11px]",
                                  a.source === "cover"
                                    ? "bg-[var(--color-warning-soft,#fef3c7)] text-[var(--color-warning,#d97706)]"
                                    : "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
                                )}
                              >
                                {a.employee_name}
                              </span>
                            ))}
                            {g?.on_leave.map((o) => (
                              <span key={o.assignment_id} className="px-1.5 py-0.5 rounded text-[11px] bg-[var(--color-danger-soft,#fef2f2)] text-[var(--color-danger,#dc2626)] line-through">
                                {o.employee_name}(假)
                              </span>
                            ))}
                            {g && g.missing > 0 && (
                              <span className="px-1.5 py-0.5 rounded text-[11px] border border-dashed border-[var(--color-danger,#dc2626)] text-[var(--color-danger,#dc2626)]">
                                缺{g.missing}
                              </span>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      {assignOpen && (
        <ManualAssignModal
          shifts={activeShifts}
          onClose={() => setAssignOpen(false)}
          onDone={() => {
            setAssignOpen(false);
            invalidate();
          }}
        />
      )}
    </main>
  );
}

// ===== 手工排班(合规 409 → 确认放行) =====

function ManualAssignModal({
  shifts, onClose, onDone,
}: {
  shifts: Shift[];
  onClose: () => void;
  onDone: () => void;
}) {
  const [empId, setEmpId] = useState("");
  const [shiftId, setShiftId] = useState(shifts[0]?.id ?? "");
  const [day, setDay] = useState(new Date().toISOString().slice(0, 10));
  const [err, setErr] = useState("");
  const [violations, setViolations] = useState<Violation[] | null>(null);
  const { data: emps } = useQuery<PlannableEmployee[]>({
    queryKey: ["plannable-emps"],
    queryFn: () => fetchJSON("/api/shifts/employees"),
  });

  const submit = useMutation({
    mutationFn: (force: boolean) =>
      fetchJSON("/api/shifts/assignments", {
        method: "POST",
        body: JSON.stringify({
          employee_id: empId, shift_id: shiftId, day, force,
        }),
      }),
    onSuccess: onDone,
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409) {
        const vs = (e.detail as { detail?: { violations?: Violation[] } })
          ?.detail?.violations;
        if (vs?.length) {
          setViolations(vs);
          setErr("");
          return;
        }
      }
      setErr(e instanceof ApiError ? e.message : "保存失败");
    },
  });

  return (
    <Modal open onClose={onClose} title="手工排班"
      description="排班前先做合规校验(连班/周工时/休息间隔/可用性);违反硬规则需确认放行。">
      <div className="flex flex-col gap-3">
        <Field label="员工" required>
          <Select
            value={empId}
            onChange={setEmpId}
            options={[
              { value: "", label: "选择员工" },
              ...(emps ?? []).map((e) => ({
                value: e.id,
                label: `${e.name}(${e.employee_no})`,
              })),
            ]}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="班次" required>
            <Select
              value={shiftId}
              onChange={setShiftId}
              options={shifts.map((s) => ({
                value: s.id,
                label: `${s.name} ${s.time_range}`,
              }))}
            />
          </Field>
          <Field label="日期" required>
            <Input type="date" value={day} onChange={(e) => setDay(e.target.value)} />
          </Field>
        </div>
        {violations && (
          <div className="rounded-lg border border-[var(--color-danger)]/40 bg-[var(--color-danger-soft,#fef2f2)] p-3 flex flex-col gap-1">
            <p className="text-[12px] font-medium text-[var(--color-danger)]">
              违反 {violations.length} 条合规规则:
            </p>
            {violations.map((v, i) => (
              <p key={i} className="text-[12px] text-[var(--color-danger)]">
                · {RULE_LABEL[v.rule] ?? v.rule}({v.day}):{v.detail}
              </p>
            ))}
          </div>
        )}
        {err && <p className="text-[12px] text-[var(--color-danger)]">{err}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="secondary" size="sm" onClick={onClose}>取消</Button>
          {violations ? (
            <Button size="sm" onClick={() => submit.mutate(true)} disabled={submit.isPending}>
              <ShieldCheck className="w-3.5 h-3.5" /> 仍要排班(确认放行)
            </Button>
          ) : (
            <Button size="sm" onClick={() => submit.mutate(false)}
              disabled={!empId || !shiftId || submit.isPending}>
              <Plus className="w-3.5 h-3.5" /> 排班
            </Button>
          )}
        </div>
      </div>
    </Modal>
  );
}

// ===== 换班审批 =====

function SwapsPanel({ canManage }: { canManage: boolean }) {
  const qc = useQueryClient();
  const [status, setStatus] = useState("");
  const [err, setErr] = useState("");
  const [warn, setWarn] = useState<Violation[]>([]);
  const { data: swaps } = useQuery<SwapReq[]>({
    queryKey: ["shift-swaps", status],
    queryFn: () =>
      fetchJSON(`/api/shifts/swaps${status ? `?status=${status}` : ""}`),
  });
  const decide = useMutation({
    mutationFn: ({ id, approve }: { id: string; approve: boolean }) =>
      fetchJSON<{ warnings: Violation[] }>(`/api/shifts/swaps/${id}/decide`, {
        method: "POST",
        body: JSON.stringify({ approve }),
      }),
    onSuccess: (d) => {
      setWarn(d.warnings ?? []);
      setErr("");
      qc.invalidateQueries({ queryKey: ["shift-swaps"] });
      qc.invalidateQueries({ queryKey: ["shift-assigns"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });
  const pending = (swaps ?? []).filter((s) => s.status === "pending_approval");
  const rest = (swaps ?? []).filter((s) => s.status !== "pending_approval");
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Select
          value={status}
          onChange={setStatus}
          options={[
            { value: "", label: "全部状态" },
            ...Object.entries(SWAP_STATUS_LABEL).map(([v, l]) => ({ value: v, label: l })),
          ]}
        />
        {err && <span className="text-[12px] text-[var(--color-danger)]">{err}</span>}
      </div>
      {warn.length > 0 && (
        <div className="rounded-lg border border-[var(--color-warning,#d97706)]/40 bg-[var(--color-warning-soft,#fef3c7)] p-3">
          <p className="text-[12px] font-medium text-[var(--color-warning,#b45309)]">
            已批准,但交换后存在 {warn.length} 条合规告警:
          </p>
          {warn.map((v, i) => (
            <p key={i} className="text-[12px] text-[var(--color-warning,#b45309)]">
              · {v.employee_name} {RULE_LABEL[v.rule] ?? v.rule}({v.day}):{v.detail}
            </p>
          ))}
        </div>
      )}
      {(swaps ?? []).length === 0 ? (
        <Empty icon={ArrowLeftRight} title="暂无换班申请" description="员工在「我的班次」发起后出现在这里。" />
      ) : (
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-5 py-2.5 font-medium">日期</th>
                <th className="px-3 py-2.5 font-medium">申请人 → 对方</th>
                <th className="px-3 py-2.5 font-medium">方式</th>
                <th className="px-3 py-2.5 font-medium">状态</th>
                <th className="px-3 py-2.5 font-medium">备注</th>
                <th className="px-5 py-2.5 font-medium text-right">操作</th>
              </tr>
            </thead>
            <tbody>
              {[...pending, ...rest].map((s) => (
                <tr key={s.id} className="border-b border-[var(--color-border-row)] last:border-0">
                  <td className="px-5 py-2.5 font-mono">{s.day}</td>
                  <td className="px-3 py-2.5">{s.requester_name} → {s.target_name}</td>
                  <td className="px-3 py-2.5 text-[12px] text-[var(--color-text-secondary)]">
                    {s.mode === "exchange" ? "互换班次" : "转出班次"}
                  </td>
                  <td className="px-3 py-2.5">
                    <span className={cn(
                      "text-[12px]",
                      s.status === "approved" && "text-[var(--color-success)]",
                      (s.status === "rejected" || s.status === "peer_declined") && "text-[var(--color-danger)]",
                      s.status === "pending_approval" && "text-[var(--color-accent)]",
                    )}>
                      {SWAP_STATUS_LABEL[s.status] ?? s.status}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-[12px] text-[var(--color-text-tertiary)] max-w-[220px] truncate">
                    {s.requester_note ?? s.peer_note ?? s.decision_note ?? "—"}
                  </td>
                  <td className="px-5 py-2.5 text-right">
                    {s.status === "pending_approval" && canManage && (
                      <span className="inline-flex gap-1.5">
                        <Button size="sm" onClick={() => decide.mutate({ id: s.id, approve: true })} disabled={decide.isPending}>
                          <Check className="w-3.5 h-3.5" /> 批准
                        </Button>
                        <Button size="sm" variant="secondary" onClick={() => decide.mutate({ id: s.id, approve: false })} disabled={decide.isPending}>
                          <X className="w-3.5 h-3.5" /> 驳回
                        </Button>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

// ===== 合规体检 =====

function CompliancePanel({ month, canManage }: { month: string; canManage: boolean }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const { data: rules } = useQuery<ShiftRules>({
    queryKey: ["shift-rules"],
    queryFn: () => fetchJSON("/api/shifts/rules"),
  });
  const { data: report } = useQuery<{ violations: Violation[] }>({
    queryKey: ["shift-compliance", month],
    queryFn: () => fetchJSON(`/api/shifts/compliance?month=${month}`),
  });
  const [form, setForm] = useState<ShiftRules | null>(null);
  const cur = form ?? rules;
  const saveRules = useMutation({
    mutationFn: () =>
      fetchJSON("/api/shifts/rules", {
        method: "PUT",
        body: JSON.stringify(cur),
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["shift-rules"] });
      qc.invalidateQueries({ queryKey: ["shift-compliance"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });
  const violations = report?.violations ?? [];
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6 items-start">
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)] flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-[var(--color-accent)]" /> 合规规则
        </h3>
        {cur && (
          <>
            <Field label="连续出勤上限(天)">
              <Input type="number" min="1" max="14" value={cur.max_consecutive_days}
                onChange={(e) => setForm({ ...cur, max_consecutive_days: Number(e.target.value) })} />
            </Field>
            <Field label="周工时上限(小时)">
              <Input type="number" min="1" max="80" step="0.5" value={cur.max_weekly_hours}
                onChange={(e) => setForm({ ...cur, max_weekly_hours: Number(e.target.value) })} />
            </Field>
            <Field label="班间最小休息(小时)">
              <Input type="number" min="0" max="24" value={Math.round(cur.min_rest_minutes / 60)}
                onChange={(e) => setForm({ ...cur, min_rest_minutes: Number(e.target.value) * 60 })} />
            </Field>
            <label className="flex items-center gap-2 text-[13px] text-[var(--color-text-secondary)]">
              <input
                type="checkbox"
                checked={cur.respect_availability}
                onChange={(e) => setForm({ ...cur, respect_availability: e.target.checked })}
              />
              尊重员工可用性偏好(不可排日不排)
            </label>
            {canManage && (
              <Button size="sm" onClick={() => saveRules.mutate()} disabled={!form || saveRules.isPending}>
                保存规则
              </Button>
            )}
          </>
        )}
        {err && <p className="text-[12px] text-[var(--color-danger)]">{err}</p>}
      </section>
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        <header className="px-5 py-3 border-b border-[var(--color-border-row)]">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            {month} 违规清单
            <span className="ml-2 text-[12px] font-normal text-[var(--color-text-tertiary)]">
              {violations.length} 条
            </span>
          </h3>
        </header>
        {violations.length === 0 ? (
          <Empty icon={ShieldCheck} title="本月无违规" description="排班符合全部合规规则。" />
        ) : (
          <table className="w-full text-[13px] font-body">
            <tbody>
              {violations.map((v, i) => (
                <tr key={i} className="border-b border-[var(--color-border-row)] last:border-0">
                  <td className="px-5 py-2.5 font-mono w-[110px]">{v.day}</td>
                  <td className="px-3 py-2.5 font-medium">{v.employee_name}</td>
                  <td className="px-3 py-2.5">
                    <span className={cn(
                      "px-2 py-0.5 rounded text-[11px]",
                      v.severity === "block"
                        ? "bg-[var(--color-danger-soft,#fef2f2)] text-[var(--color-danger)]"
                        : "bg-[var(--color-warning-soft,#fef3c7)] text-[var(--color-warning,#b45309)]",
                    )}>
                      {RULE_LABEL[v.rule] ?? v.rule}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-[12px] text-[var(--color-text-secondary)]">{v.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

// ===== 排班模板 / 循环 =====

function TemplatesPanel({
  canManage, onApplied,
}: { canManage: boolean; onApplied: () => void }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const { data: templates } = useQuery<ScheduleTemplate[]>({
    queryKey: ["shift-templates"],
    queryFn: () => fetchJSON("/api/shifts/templates"),
  });
  const { data: shifts } = useQuery<Shift[]>({
    queryKey: ["shifts"],
    queryFn: () => fetchJSON("/api/shifts"),
  });
  const { data: emps } = useQuery<PlannableEmployee[]>({
    queryKey: ["plannable-emps"],
    queryFn: () => fetchJSON("/api/shifts/employees"),
  });
  const shiftName = useMemo(
    () => new Map((shifts ?? []).map((s) => [s.id, s.name])), [shifts]);

  const [name, setName] = useState("");
  const [period, setPeriod] = useState("7");
  const [pattern, setPattern] = useState<Record<number, string>>({});
  const create = useMutation({
    mutationFn: () =>
      fetchJSON("/api/shifts/templates", {
        method: "POST",
        body: JSON.stringify({
          name,
          period_days: Number(period),
          pattern: Array.from({ length: Number(period) }, (_, i) => ({
            offset: i,
            shift_id: pattern[i] || null,
          })),
        }),
      }),
    onSuccess: () => {
      setErr("");
      setName("");
      setPattern({});
      qc.invalidateQueries({ queryKey: ["shift-templates"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  const remove = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/shifts/templates/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["shift-templates"] }),
  });

  return (
    <div className="flex flex-col gap-4">
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)] flex items-center gap-2">
          <LayoutTemplate className="w-4 h-4 text-[var(--color-accent)]" /> 新建循环模板
        </h3>
        <div className="flex items-end gap-2 flex-wrap">
          <Field label="模板名">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="如:早早晚休" className="w-[140px] h-9 text-[13px]" />
          </Field>
          <Field label="周期(天)">
            <Input type="number" min="1" max="28" value={period}
              onChange={(e) => setPeriod(e.target.value)} className="w-[90px] h-9 text-[13px]" />
          </Field>
          {Array.from({ length: Math.min(Number(period) || 0, 14) }, (_, i) => (
            <Field key={i} label={`第${i + 1}天`}>
              <Select
                value={pattern[i] ?? ""}
                onChange={(v) => setPattern((p) => ({ ...p, [i]: v }))}
                options={[
                  { value: "", label: "休" },
                  ...(shifts ?? []).filter((s) => s.active).map((s) => ({
                    value: s.id, label: s.name,
                  })),
                ]}
              />
            </Field>
          ))}
          {canManage && (
            <Button size="sm" onClick={() => create.mutate()}
              disabled={!name.trim() || create.isPending}>
              <Plus className="w-3.5 h-3.5" /> 保存模板
            </Button>
          )}
        </div>
        {err && <p className="text-[12px] text-[var(--color-danger)]">{err}</p>}
        {msg && <p className="text-[12px] text-[var(--color-success)]">{msg}</p>}
      </section>

      {(templates ?? []).length === 0 ? (
        <Empty icon={LayoutTemplate} title="还没有模板"
          description="建一个「早早晚休」式循环模板,批量套给员工自动生成排班。" />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {(templates ?? []).map((t) => (
            <TemplateCard
              key={t.id} t={t} shiftName={shiftName}
              emps={emps ?? []} canManage={canManage}
              onDeleted={() => remove.mutate(t.id)}
              onApplied={(m) => { setMsg(m); onApplied(); }}
              onError={setErr}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function TemplateCard({
  t, shiftName, emps, canManage, onDeleted, onApplied, onError,
}: {
  t: ScheduleTemplate;
  shiftName: Map<string, string>;
  emps: PlannableEmployee[];
  canManage: boolean;
  onDeleted: () => void;
  onApplied: (msg: string) => void;
  onError: (m: string) => void;
}) {
  const [start, setStart] = useState(new Date().toISOString().slice(0, 10));
  const [cycles, setCycles] = useState("4");
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState(false);
  const [warn, setWarn] = useState<Violation[]>([]);
  const apply = useMutation({
    mutationFn: (dryRun: boolean) =>
      fetchJSON<{
        planned: number; skipped: { reason: string }[];
        warnings: Violation[];
      }>(`/api/shifts/templates/${t.id}/apply`, {
        method: "POST",
        body: JSON.stringify({
          employee_ids: [...picked],
          start, cycles: Number(cycles), dry_run: dryRun,
        }),
      }),
    onSuccess: (d, dryRun) => {
      setWarn(d.warnings ?? []);
      if (!dryRun) {
        onApplied(`已套用生成 ${d.planned} 条排班,跳过 ${d.skipped.length} 天`);
      }
    },
    onError: (e) => onError(e instanceof ApiError ? e.message : "套用失败"),
  });
  const pat = Array.from({ length: t.period_days }, (_, i) => {
    const p = (t.pattern ?? []).find((x) => x.offset === i);
    return p?.shift_id ? shiftName.get(p.shift_id) ?? "?" : "休";
  });
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <h4 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">{t.name}</h4>
        <span className="text-[11px] text-[var(--color-text-tertiary)]">{t.period_days} 天循环</span>
        {canManage && (
          <button className="ml-auto text-[12px] text-[var(--color-danger)] hover:underline" onClick={onDeleted}>
            删除
          </button>
        )}
      </div>
      <div className="flex gap-1 flex-wrap">
        {pat.map((label, i) => (
          <span key={i} className={cn(
            "px-2 py-0.5 rounded text-[11px]",
            label === "休"
              ? "bg-[var(--color-bg-canvas)] text-[var(--color-text-tertiary)]"
              : "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
          )}>
            {i + 1}:{label}
          </span>
        ))}
      </div>
      {canManage && (
        <>
          <button
            className="text-[12px] text-[var(--color-accent)] hover:underline self-start"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "收起套用" : `套用给员工(${picked.size} 已选)`}
          </button>
          {open && (
            <div className="flex flex-col gap-2 border-t border-[var(--color-border-row)] pt-3">
              <div className="flex flex-wrap gap-1.5 max-h-[120px] overflow-auto">
                {emps.map((e) => (
                  <button
                    key={e.id}
                    onClick={() => setPicked((p) => {
                      const n = new Set(p);
                      if (n.has(e.id)) n.delete(e.id); else n.add(e.id);
                      return n;
                    })}
                    className={cn(
                      "px-2 py-0.5 rounded text-[12px] border",
                      picked.has(e.id)
                        ? "bg-[var(--color-accent-soft)] border-[var(--color-accent)]/50 text-[var(--color-accent)]"
                        : "border-[var(--color-border-subtle)] text-[var(--color-text-secondary)]",
                    )}
                  >
                    {e.name}
                  </button>
                ))}
              </div>
              <div className="flex items-end gap-2">
                <Field label="起始日">
                  <Input type="date" value={start} onChange={(e) => setStart(e.target.value)} className="h-9 text-[13px]" />
                </Field>
                <Field label="循环次数">
                  <Input type="number" min="1" max="12" value={cycles} onChange={(e) => setCycles(e.target.value)} className="w-[80px] h-9 text-[13px]" />
                </Field>
                <Button size="sm" variant="secondary" onClick={() => apply.mutate(true)}
                  disabled={picked.size === 0 || apply.isPending}>
                  预览
                </Button>
                <Button size="sm" onClick={() => apply.mutate(false)}
                  disabled={picked.size === 0 || apply.isPending}>
                  套用生成
                </Button>
              </div>
              {warn.length > 0 && (
                <div className="rounded-lg border border-[var(--color-warning,#d97706)]/40 bg-[var(--color-warning-soft,#fef3c7)] p-2.5 max-h-[120px] overflow-auto">
                  {warn.slice(0, 8).map((v, i) => (
                    <p key={i} className="text-[11px] text-[var(--color-warning,#b45309)]">
                      · {v.employee_name} {RULE_LABEL[v.rule] ?? v.rule}({v.day})
                    </p>
                  ))}
                  {warn.length > 8 && (
                    <p className="text-[11px] text-[var(--color-warning,#b45309)]">…共 {warn.length} 条合规告警</p>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function ShiftDefs({ onChanged }: { onChanged: () => void }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [name, setName] = useState("");
  const [start, setStart] = useState("09:00");
  const [end, setEnd] = useState("18:00");
  const [required, setRequired] = useState("1");
  const { data: shifts } = useQuery<Shift[]>({
    queryKey: ["shifts"],
    queryFn: () => fetchJSON("/api/shifts"),
  });
  const toMin = (hhmm: string) => {
    const [h, m] = hhmm.split(":").map(Number);
    return h * 60 + m;
  };
  const create = useMutation({
    mutationFn: () =>
      fetchJSON("/api/shifts", {
        method: "POST",
        body: JSON.stringify({
          name, start_min: toMin(start), end_min: toMin(end),
          required: Number(required) || 1,
        }),
      }),
    onSuccess: () => {
      setErr("");
      setName("");
      qc.invalidateQueries({ queryKey: ["shifts"] });
      onChanged();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  const remove = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/shifts/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["shifts"] }),
  });

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">班次定义</h3>
      {(shifts ?? []).map((s) => (
        <div key={s.id} className="flex items-center gap-3 text-[13px] font-body">
          <span className="font-medium text-[var(--color-text-primary)] w-24">{s.name}</span>
          <span className="font-mono text-[var(--color-text-secondary)]">{s.time_range}</span>
          <span className="text-[var(--color-text-tertiary)]">每班需 {s.required} 人</span>
          {!s.active && <span className="text-[11px] text-[var(--color-text-tertiary)]">(已停用)</span>}
          <button
            className="ml-auto text-[12px] text-[var(--color-danger)] hover:underline"
            onClick={() => remove.mutate(s.id)}
          >
            {s.active ? "删除/停用" : "彻底删除"}
          </button>
        </div>
      ))}
      <div className="flex items-end gap-2 pt-2 border-t border-[var(--color-border-row)]">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="班次名(如早班)" className="w-[120px] h-9 text-[13px]" />
        <Input type="time" value={start} onChange={(e) => setStart(e.target.value)} className="w-[110px] h-9 text-[13px]" />
        <span className="pb-2 text-[var(--color-text-tertiary)]">-</span>
        <Input type="time" value={end} onChange={(e) => setEnd(e.target.value)} className="w-[110px] h-9 text-[13px]" />
        <Input type="number" min="1" value={required} onChange={(e) => setRequired(e.target.value)} placeholder="需人数" className="w-[80px] h-9 text-[13px]" />
        <Button size="sm" onClick={() => create.mutate()} disabled={!name.trim() || create.isPending}>
          <Plus className="w-3.5 h-3.5" /> 新增班次
        </Button>
        {err && <span className="text-[12px] text-[var(--color-danger)]">{err}</span>}
      </div>
    </section>
  );
}
