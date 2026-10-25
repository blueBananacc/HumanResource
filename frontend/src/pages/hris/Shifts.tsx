import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CalendarDays,
  Plus,
  Sparkles,
  UserCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { cn } from "@/lib/utils";

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

function daysOfMonth(month: string): string[] {
  const [y, m] = month.split("-").map(Number);
  const n = new Date(y, m, 0).getDate();
  return Array.from({ length: n }, (_, i) =>
    `${month}-${String(i + 1).padStart(2, "0")}`);
}

const WEEK = ["日", "一", "二", "三", "四", "五", "六"];

export function ShiftsPage() {
  const [sp, setSp] = useSearchParams();
  const month = sp.get("month") ?? new Date().toISOString().slice(0, 7);
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_ATTENDANCE);
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [showDefs, setShowDefs] = useState(false);

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

      <div className="flex items-center gap-2">
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
        {canManage && (
          <>
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
          </>
        )}
        <div className="flex-1" />
        {msg && <span className="text-[13px] text-[var(--color-success)] font-body">{msg}</span>}
        {err && <span className="text-[13px] text-[var(--color-danger)] font-body">{err}</span>}
      </div>

      {showDefs && canManage && <ShiftDefs onChanged={() => qc.invalidateQueries({ queryKey: ["shifts"] })} />}

      {activeShifts.length === 0 ? (
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
      )}
    </main>
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
