import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Clock, ClipboardPlus, LogIn, LogOut, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  StatusPill,
  fmtTime,
  type MeAttendance,
  type MeAttRequest,
  type MeCheckin,
} from "@/lib/hris";
import { Field } from "../hris/Employees";

export function MyAttendance() {
  const [month, setMonth] = useState(new Date().toISOString().slice(0, 7));
  const [reqOpen, setReqOpen] = useState(false);
  const [err, setErr] = useState("");
  const qc = useQueryClient();
  const confirm = useConfirm();

  const { data: attendance, isLoading } = useQuery<MeAttendance[]>({
    queryKey: ["me-attendance", month],
    queryFn: () => fetchJSON(`/api/me/attendance?month=${month}`),
  });
  const { data: checkins } = useQuery<MeCheckin[]>({
    queryKey: ["me-checkins"],
    queryFn: () => fetchJSON("/api/me/checkins"),
  });
  const { data: attReqs } = useQuery<MeAttRequest[]>({
    queryKey: ["me-att-reqs"],
    queryFn: () => fetchJSON("/api/me/attendance-requests"),
  });

  const todayStr = new Date().toISOString().slice(0, 10);
  const todayCheckins = (checkins ?? []).filter((c) =>
    c.ts.startsWith(todayStr),
  );
  const lastKind = todayCheckins[todayCheckins.length - 1]?.kind;
  const nextKind = lastKind === "in" ? "out" : "in";

  const punch = useMutation({
    mutationFn: (kind: "in" | "out") =>
      fetchJSON("/api/me/checkins", {
        method: "POST",
        body: JSON.stringify({ kind }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-checkins"] });
      qc.invalidateQueries({ queryKey: ["me-attendance"] });
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "打卡失败"),
  });

  const cancelReq = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/attendance-requests/${id}/cancel`, {
        method: "POST",
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["me-att-reqs"] });
      qc.invalidateQueries({ queryKey: ["me-checkins"] });
      qc.invalidateQueries({ queryKey: ["me-attendance"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "撤销失败"),
  });

  async function onCancelReq(r: MeAttRequest) {
    if (r.status === "approved") {
      const ok = await confirm({
        title: "撤销已批准的补卡?",
        description: `${r.ts.slice(0, 10)} 的${r.kind === "in" ? "上班" : "下班"}打卡将被删除,当日考勤重新计算。`,
        confirmLabel: "撤销",
        tone: "danger",
      });
      if (!ok) return;
    }
    cancelReq.mutate(r.id);
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          我的考勤
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          上下班打卡 + 按天汇总;异常请联系 HR 修正。
        </p>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex items-center gap-6">
        <div className="flex flex-col gap-1">
          <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
            今日 {todayStr}
          </div>
          <div className="font-mono font-semibold text-2xl text-[var(--color-text-primary)]">
            {new Date().toLocaleTimeString("zh-CN", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
          <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
            今日已打卡 {todayCheckins.length} 次
            {todayCheckins.length > 0 &&
              ` · 最近 ${fmtTime(todayCheckins[todayCheckins.length - 1].ts)}(${
                todayCheckins[todayCheckins.length - 1].kind === "in"
                  ? "上班"
                  : "下班"
              })`}
          </div>
        </div>
        <div className="flex-1" />
        <Button variant="secondary" onClick={() => setReqOpen(true)}>
          <ClipboardPlus className="w-4 h-4" />
          申请补卡
        </Button>
        {nextKind === "in" ? (
          <Button
            size="lg"
            onClick={() => punch.mutate("in")}
            disabled={punch.isPending}
          >
            <LogIn className="w-4 h-4" />
            上班打卡
          </Button>
        ) : (
          <Button
            size="lg"
            onClick={() => punch.mutate("out")}
            disabled={punch.isPending}
          >
            <LogOut className="w-4 h-4" />
            下班打卡
          </Button>
        )}
        {err && (
          <span className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Input
          type="month"
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="w-[160px]"
        />
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (attendance ?? []).length === 0 ? (
          <Empty
            icon={Clock}
            title="本月还没有考勤记录"
            description="打卡后第二天会自动生成考勤记录。"
          />
        ) : (
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-6 py-3 font-medium">日期</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">上班</th>
                <th className="px-4 py-3 font-medium">下班</th>
                <th className="px-6 py-3 font-medium text-right">工时</th>
              </tr>
            </thead>
            <tbody>
              {(attendance ?? []).map((a) => (
                <tr
                  key={a.day}
                  className="border-b border-[var(--color-border-row)] last:border-0"
                >
                  <td className="px-6 py-3 font-mono">{a.day}</td>
                  <td className="px-4 py-3">
                    <StatusPill status={a.status} />
                  </td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.first_in)}</td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.last_out)}</td>
                  <td className="px-6 py-3 text-right font-mono">
                    {a.work_hours != null ? `${a.work_hours}h` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {(attReqs ?? []).length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            补卡申请
          </h3>
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            {(attReqs ?? []).map((r, i) => (
              <div
                key={r.id}
                className={`flex items-center gap-3 px-6 py-3.5 ${
                  i < (attReqs ?? []).length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <span className="font-mono text-[13px] text-[var(--color-text-primary)]">
                  {r.ts.slice(0, 10)} {fmtTime(r.ts)}
                </span>
                <span className="px-2 py-0.5 rounded-md text-[11px] font-body bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]">
                  {r.kind === "in" ? "上班卡" : "下班卡"}
                </span>
                <StatusPill status={r.status} />
                {r.decision_note && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    {r.decision_note}
                  </span>
                )}
                <div className="flex-1" />
                {(r.status === "pending" || r.status === "approved") && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onCancelReq(r)}
                  >
                    <Undo2 className="w-3.5 h-3.5" />
                    {r.status === "pending" ? "撤回" : "撤销"}
                  </Button>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      <AttRequestModal open={reqOpen} onClose={() => setReqOpen(false)} />
    </main>
  );
}

function AttRequestModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [kind, setKind] = useState("in");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/attendance-requests", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-att-reqs"] });
      onClose();
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const ts = String(fd.get("ts"));
    const parsed = new Date(ts);
    if (!ts || Number.isNaN(parsed.getTime())) {
      setErr("请选择要补的时刻");
      return;
    }
    // datetime-local 是本地墙钟 naive 串;打卡记录按 UTC 存,
    // 转 ISO(带 Z)交给后端统一换算,口径与实时打卡一致
    mut.mutate({
      ts: parsed.toISOString(),
      kind,
      reason: String(fd.get("reason") || "").trim(),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="申请补卡"
      description="漏打卡时补录 — 主管批准后自动写入打卡记录并重算当日考勤。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="att-req" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="att-req" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="补卡时刻" required>
            <Input name="ts" type="datetime-local" required />
          </Field>
          <Field label="方向" required>
            <Select
              value={kind}
              onChange={setKind}
              options={[
                { value: "in", label: "上班卡" },
                { value: "out", label: "下班卡" },
              ]}
            />
          </Field>
        </div>
        <Field label="事由">
          <Input name="reason" placeholder="例如:早会直接进会议室,忘打卡" />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
