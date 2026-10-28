import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeftRight,
  CalendarDays,
  Check,
  Heart,
  Send,
  Undo2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Field } from "../hris/Employees";

type MyShift = {
  id: string;
  day: string;
  shift_id: string;
  shift_name: string;
  time_range: string;
  color: string | null;
  source: string;
  note: string | null;
};
type AvailItem = { weekday: number; preference: string };
type SwapReq = {
  id: string;
  day: string;
  mode: string;
  status: string;
  direction: string;
  other_employee_id: string;
  other_name: string;
  my_shift_name: string | null;
  other_shift_name: string | null;
  requester_note: string | null;
  peer_note: string | null;
  decision_note: string | null;
  created_at: string;
};
type Target = {
  id: string;
  name: string;
  same_org: boolean;
};

const WEEK_LABEL = ["一", "二", "三", "四", "五", "六", "日"];
const PREF_LABEL: Record<string, string> = {
  available: "可排班",
  preferred: "偏好",
  unavailable: "不可排",
};
const SWAP_STATUS: Record<string, { label: string; cls: string }> = {
  pending_peer: { label: "待对方确认", cls: "text-[var(--color-warning)]" },
  pending_approval: { label: "待主管审批", cls: "text-[var(--color-accent)]" },
  approved: { label: "已换班", cls: "text-[var(--color-success)]" },
  peer_declined: { label: "对方已拒绝", cls: "text-[var(--color-danger)]" },
  rejected: { label: "审批驳回", cls: "text-[var(--color-danger)]" },
  cancelled: { label: "已撤回", cls: "text-[var(--color-text-tertiary)]" },
};
const SOURCE_LABEL: Record<string, string> = {
  manual: "手工", auto: "生成", cover: "替班",
  template: "模板", swap: "换班",
};

export function MyShifts() {
  const [month, setMonth] = useState(new Date().toISOString().slice(0, 7));
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [swapFor, setSwapFor] = useState<MyShift | null>(null);
  const qc = useQueryClient();

  const { data: shifts, isLoading } = useQuery<MyShift[]>({
    queryKey: ["me-shifts", month],
    queryFn: () => fetchJSON(`/api/me/shifts?month=${month}`),
  });
  const { data: avail } = useQuery<{ items: AvailItem[] }>({
    queryKey: ["me-avail"],
    queryFn: () => fetchJSON("/api/me/shift-availability"),
  });
  const { data: swaps } = useQuery<SwapReq[]>({
    queryKey: ["me-swaps"],
    queryFn: () => fetchJSON("/api/me/shift-swaps"),
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["me-swaps"] });
    qc.invalidateQueries({ queryKey: ["me-shifts"] });
  };

  const inbox = (swaps ?? []).filter(
    (s) => s.direction === "in" && s.status === "pending_peer");
  const outbox = (swaps ?? []).filter((s) => s.direction === "out");

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          我的班次
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          查看排班、登记每周可用时间,与同事发起换班(双方确认 + 主管审批)。
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Input
          type="month" value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="w-[150px] h-9 text-[13px]"
        />
        <div className="flex-1" />
        {msg && <span className="text-[13px] text-[var(--color-success)]">{msg}</span>}
        {err && <span className="text-[13px] text-[var(--color-danger)]">{err}</span>}
      </div>

      {/* 本月班次 */}
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        <header className="px-5 py-3 border-b border-[var(--color-border-row)] flex items-center gap-2">
          <CalendarDays className="w-4 h-4 text-[var(--color-accent)]" />
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            {month} 排班
          </h3>
          <span className="text-[12px] text-[var(--color-text-tertiary)]">
            共 {(shifts ?? []).length} 天有班
          </span>
        </header>
        {isLoading ? (
          <div className="px-5 py-8 text-center text-[13px] text-[var(--color-text-tertiary)]">加载中…</div>
        ) : (shifts ?? []).length === 0 ? (
          <Empty icon={CalendarDays} title="本月暂无排班" description="HR 排班后会显示在这里。" />
        ) : (
          <table className="w-full text-[13px] font-body">
            <tbody>
              {(shifts ?? []).map((s) => {
                const today = s.day === new Date().toISOString().slice(0, 10);
                const canSwap = s.day >= new Date().toISOString().slice(0, 10);
                return (
                  <tr key={s.id} className={cn(
                    "border-b border-[var(--color-border-row)] last:border-0",
                    today && "bg-[var(--color-accent-soft)]/40",
                  )}>
                    <td className="px-5 py-2.5 font-mono whitespace-nowrap w-[130px]">
                      {s.day} 周{WEEK_LABEL[(new Date(s.day + "T00:00:00").getDay() + 6) % 7]}
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="px-2 py-0.5 rounded-md text-[12px] bg-[var(--color-accent-soft)] text-[var(--color-accent)] font-medium">
                        {s.shift_name}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-[var(--color-text-secondary)]">
                      {s.time_range}
                    </td>
                    <td className="px-3 py-2.5 text-[12px] text-[var(--color-text-tertiary)]">
                      {SOURCE_LABEL[s.source] ?? s.source}
                      {s.note ? ` · ${s.note}` : ""}
                    </td>
                    <td className="px-5 py-2.5 text-right">
                      {canSwap && (
                        <button
                          className="text-[12px] text-[var(--color-accent)] hover:underline inline-flex items-center gap-1"
                          onClick={() => setSwapFor(s)}
                        >
                          <ArrowLeftRight className="w-3 h-3" /> 换班
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 可用性偏好 */}
        <AvailabilityCard
          items={avail?.items ?? []}
          onSaved={() => {
            setMsg("可用性已保存");
            qc.invalidateQueries({ queryKey: ["me-avail"] });
          }}
          onError={setErr}
        />

        {/* 换班申请 */}
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden flex flex-col">
          <header className="px-5 py-3 border-b border-[var(--color-border-row)] flex items-center gap-2">
            <ArrowLeftRight className="w-4 h-4 text-[var(--color-accent)]" />
            <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">换班申请</h3>
            {inbox.length > 0 && (
              <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-[var(--color-danger)] text-white">
                {inbox.length} 待处理
              </span>
            )}
          </header>
          <div className="flex-1 overflow-auto max-h-[420px]">
            {inbox.length === 0 && outbox.length === 0 ? (
              <Empty icon={ArrowLeftRight} title="暂无换班申请" description="在上方班次行点「换班」即可发起。" />
            ) : (
              <div className="divide-y divide-[var(--color-border-row)]">
                {inbox.map((s) => (
                  <InboxRow key={s.id} s={s} onDone={invalidate} onError={setErr} />
                ))}
                {outbox.map((s) => (
                  <OutboxRow key={s.id} s={s} onDone={invalidate} onError={setErr} />
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      {swapFor && (
        <SwapModal
          shift={swapFor}
          onClose={() => setSwapFor(null)}
          onDone={(m) => {
            setSwapFor(null);
            setMsg(m);
            invalidate();
          }}
        />
      )}
    </main>
  );
}

function AvailabilityCard({
  items, onSaved, onError,
}: {
  items: AvailItem[];
  onSaved: () => void;
  onError: (m: string) => void;
}) {
  const [prefs, setPrefs] = useState<Record<number, string>>({});
  const merged = items.map((i) => ({
    ...i,
    preference: prefs[i.weekday] ?? i.preference,
  }));
  const dirty = Object.keys(prefs).length > 0;
  const save = useMutation({
    mutationFn: () =>
      fetchJSON("/api/me/shift-availability", {
        method: "PUT",
        body: JSON.stringify({
          items: merged.filter(
            (i) => i.preference !== "available"),
        }),
      }),
    onSuccess: () => {
      setPrefs({});
      onSaved();
    },
    onError: (e) => onError(e instanceof ApiError ? e.message : "保存失败"),
  });
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden flex flex-col">
      <header className="px-5 py-3 border-b border-[var(--color-border-row)] flex items-center gap-2">
        <Heart className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">每周可用性</h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)]">排班与换班时参考</span>
        {dirty && (
          <Button size="sm" className="ml-auto" onClick={() => save.mutate()} disabled={save.isPending}>
            <Check className="w-3.5 h-3.5" /> 保存
          </Button>
        )}
      </header>
      <div className="p-4 grid grid-cols-1 gap-1.5">
        {merged.map((i) => (
          <div key={i.weekday} className="flex items-center gap-3">
            <span className="w-14 text-[13px] text-[var(--color-text-secondary)]">
              周{WEEK_LABEL[i.weekday]}
            </span>
            <div className="flex gap-1">
              {(["available", "preferred", "unavailable"] as const).map((p) => (
                <button
                  key={p}
                  onClick={() =>
                    setPrefs((prev) => ({ ...prev, [i.weekday]: p }))}
                  className={cn(
                    "px-2.5 py-1 rounded-md text-[12px] border transition-colors",
                    i.preference === p
                      ? p === "unavailable"
                        ? "bg-[var(--color-danger-soft,#fef2f2)] border-[var(--color-danger)]/40 text-[var(--color-danger)]"
                        : p === "preferred"
                          ? "bg-[var(--color-warning-soft,#fef3c7)] border-[var(--color-warning,#d97706)]/40 text-[var(--color-warning,#b45309)]"
                          : "bg-[var(--color-accent-soft)] border-[var(--color-accent)]/40 text-[var(--color-accent)]"
                      : "border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)] hover:border-[var(--color-accent)]/40",
                  )}
                >
                  {PREF_LABEL[p]}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SwapModal({
  shift, onClose, onDone,
}: {
  shift: MyShift;
  onClose: () => void;
  onDone: (msg: string) => void;
}) {
  const [targetId, setTargetId] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const { data: targets } = useQuery<Target[]>({
    queryKey: ["me-swap-targets"],
    queryFn: () => fetchJSON("/api/me/shift-swap-targets"),
  });
  const sorted = useMemo(
    () => [...(targets ?? [])].sort((a, b) => Number(b.same_org) - Number(a.same_org)),
    [targets]);
  const create = useMutation({
    mutationFn: () =>
      fetchJSON<SwapReq>("/api/me/shift-swaps", {
        method: "POST",
        body: JSON.stringify({
          target_employee_id: targetId, day: shift.day, note: note || null,
        }),
      }),
    onSuccess: (d: SwapReq) =>
      onDone(d.mode === "exchange" ? "换班请求已发出,等待对方确认" : "转班请求已发出,等待对方确认"),
    onError: (e) => setErr(e instanceof ApiError ? e.message : "发起失败"),
  });
  return (
    <Modal open onClose={onClose} title={`换班 · ${shift.day} ${shift.shift_name}`}
      description="双方都有班则互换班次;对方当天无班则把你的班转给他。对方确认后还需主管审批生效。">
      <div className="flex flex-col gap-3">
        <Field label="选择同事" required>
          <Select
            value={targetId}
            onChange={setTargetId}
            options={[
              { value: "", label: "选择换班同事" },
              ...sorted.map((t) => ({
                value: t.id,
                label: `${t.name}${t.same_org ? "(同部门)" : ""}`,
              })),
            ]}
          />
        </Field>
        <Field label="备注">
          <Input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="换班原因(可选)"
          />
        </Field>
        {err && <p className="text-[12px] text-[var(--color-danger)]">{err}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="secondary" size="sm" onClick={onClose}>取消</Button>
          <Button size="sm" onClick={() => create.mutate()}
            disabled={!targetId || create.isPending}>
            <Send className="w-3.5 h-3.5" /> 发起换班
          </Button>
        </div>
      </div>
    </Modal>
  );
}

function InboxRow({
  s, onDone, onError,
}: { s: SwapReq; onDone: () => void; onError: (m: string) => void }) {
  const peer = useMutation({
    mutationFn: (accept: boolean) =>
      fetchJSON(`/api/me/shift-swaps/${s.id}/peer`, {
        method: "POST",
        body: JSON.stringify({ accept }),
      }),
    onSuccess: onDone,
    onError: (e) => onError(e instanceof ApiError ? e.message : "操作失败"),
  });
  return (
    <div className="px-5 py-3 flex items-center gap-3 text-[13px]">
      <div className="flex-1 min-w-0">
        <div className="font-medium text-[var(--color-text-primary)]">
          {s.other_name} · {s.day}
        </div>
        <div className="text-[12px] text-[var(--color-text-tertiary)]">
          {s.mode === "exchange"
            ? `互换:${s.other_shift_name ?? "?"} ⇄ 你的 ${s.my_shift_name ?? "?"}`
            : `把 TA 的 ${s.other_shift_name ?? s.my_shift_name ?? "?"} 转给你`}
          {s.requester_note ? ` · 「${s.requester_note}」` : ""}
        </div>
      </div>
      <Button size="sm" onClick={() => peer.mutate(true)} disabled={peer.isPending}>
        <Check className="w-3.5 h-3.5" /> 同意
      </Button>
      <Button size="sm" variant="secondary" onClick={() => peer.mutate(false)} disabled={peer.isPending}>
        <X className="w-3.5 h-3.5" /> 拒绝
      </Button>
    </div>
  );
}

function OutboxRow({
  s, onDone, onError,
}: { s: SwapReq; onDone: () => void; onError: (m: string) => void }) {
  const cancel = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/me/shift-swaps/${s.id}/cancel`, { method: "POST" }),
    onSuccess: onDone,
    onError: (e) => onError(e instanceof ApiError ? e.message : "撤回失败"),
  });
  const st = SWAP_STATUS[s.status] ?? { label: s.status, cls: "" };
  const pending = s.status === "pending_peer" || s.status === "pending_approval";
  return (
    <div className="px-5 py-3 flex items-center gap-3 text-[13px]">
      <div className="flex-1 min-w-0">
        <div className="font-medium text-[var(--color-text-primary)]">
          → {s.other_name} · {s.day}
        </div>
        <div className="text-[12px] text-[var(--color-text-tertiary)]">
          {s.mode === "exchange" ? "互换班次" : "转出班次"}
          {s.peer_note ? ` · 对方:${s.peer_note}` : ""}
          {s.decision_note ? ` · 审批:${s.decision_note}` : ""}
        </div>
      </div>
      <span className={cn("text-[12px]", st.cls)}>{st.label}</span>
      {pending && (
        <button
          className="text-[12px] text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] inline-flex items-center gap-1"
          onClick={() => cancel.mutate()}
          disabled={cancel.isPending}
        >
          <Undo2 className="w-3 h-3" /> 撤回
        </button>
      )}
    </div>
  );
}
