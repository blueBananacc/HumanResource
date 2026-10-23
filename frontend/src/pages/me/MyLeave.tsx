import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarDays, Plus, Timer, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useConfirm } from "@/components/ui/confirm";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  type MeApplication,
  type MeBalance,
  type MeLeaveType,
  type MeOvertimeRequest,
} from "@/lib/hris";
import { Field } from "../hris/Employees";

export function MyLeave() {
  const [open, setOpen] = useState(false);
  const [otOpen, setOtOpen] = useState(false);
  const [err, setErr] = useState("");
  const qc = useQueryClient();
  const confirm = useConfirm();
  const { data: balances } = useQuery<MeBalance[]>({
    queryKey: ["me-balances"],
    queryFn: () => fetchJSON("/api/me/leave-balances"),
  });
  const { data: apps, isLoading } = useQuery<MeApplication[]>({
    queryKey: ["me-leave-apps"],
    queryFn: () => fetchJSON("/api/me/leave-applications"),
  });
  const { data: otReqs } = useQuery<MeOvertimeRequest[]>({
    queryKey: ["me-ot-reqs"],
    queryFn: () => fetchJSON("/api/me/overtime-requests"),
  });

  const cancel = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/leave-applications/${id}/cancel`, {
        method: "POST",
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["me-leave-apps"] });
      qc.invalidateQueries({ queryKey: ["me-balances"] });
      // 撤销已批申请会恢复考勤/台账,让相关页面下次拉新数据
      qc.invalidateQueries({ queryKey: ["me-attendance"] });
      qc.invalidateQueries({ queryKey: ["me-attendance-today"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "撤销失败"),
  });

  const cancelOt = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/overtime-requests/${id}/cancel`, {
        method: "POST",
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["me-ot-reqs"] });
      qc.invalidateQueries({ queryKey: ["me-balances"] });
      qc.invalidateQueries({ queryKey: ["me-leave-apps"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "撤销失败"),
  });

  async function onCancelOt(r: MeOvertimeRequest) {
    if (r.status === "approved") {
      const ok = await confirm({
        title: "撤销已批准的加班?",
        description: `${r.work_date} 加班 ${r.hours}h 折算的 ${r.comp_days} 天调休将被回冲。若调休已使用则无法撤销。`,
        confirmLabel: "撤销",
        tone: "danger",
      });
      if (!ok) return;
    }
    cancelOt.mutate(r.id);
  }

  async function onCancel(a: MeApplication) {
    if (a.status === "approved") {
      const ok = await confirm({
        title: "撤销已批准的请假?",
        description: `${a.leave_type_name} ${a.start_date} → ${a.end_date} · ${a.days} 天,撤销后返还额度并恢复考勤。`,
        confirmLabel: "撤销",
        tone: "danger",
      });
      if (!ok) return;
    }
    cancel.mutate(a.id);
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            我的请假
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            提交请假申请,主管批准后自动扣减余额。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => setOtOpen(true)}>
            <Timer className="w-4 h-4" />
            申请加班
          </Button>
          <Button onClick={() => setOpen(true)}>
            <Plus className="w-4 h-4" />
            申请请假
          </Button>
        </div>
      </div>

      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </div>
      )}

      {(balances ?? []).length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {(balances ?? []).map((b) => (
            <div
              key={b.leave_type_id}
              className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-4"
            >
              <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
                {b.leave_type_name}
              </div>
              <div className="mt-1 font-mono font-semibold text-xl text-[var(--color-text-primary)]">
                {b.balance}
                <span className="text-[12px] font-normal text-[var(--color-text-tertiary)]">
                  {" "}
                  / {b.allocated} 天
                </span>
              </div>
              {(b.used > 0 || b.pending > 0) && (
                <div className="mt-1 text-[11px] text-[var(--color-text-tertiary)] font-body">
                  已用 {b.used}
                  {b.pending > 0 && ` · 审批中 ${b.pending}`}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (apps ?? []).length === 0 ? (
          <Empty
            icon={CalendarDays}
            title="还没有请假记录"
            description="点右上角「申请请假」提交第一条申请。"
            action={<Button onClick={() => setOpen(true)}>申请请假</Button>}
          />
        ) : (
          (apps ?? []).map((a, i) => (
            <div
              key={a.id}
              className={`flex items-center gap-4 px-6 py-4 ${
                i < (apps ?? []).length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex flex-col gap-1 flex-1 min-w-0">
                <div className="flex items-center gap-2.5">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                    {a.leave_type_name}
                  </span>
                  <StatusPill status={a.status} />
                </div>
                <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {a.start_date} → {a.end_date} ·{" "}
                  <span className="font-mono">{a.days}</span> 天
                  {a.reason && ` · ${a.reason}`}
                </div>
                {a.decision_note && (
                  <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    审批意见: {a.decision_note}
                  </div>
                )}
              </div>
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                {formatRelative(a.created_at)}
              </span>
              {(a.status === "pending" || a.status === "approved") && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCancel(a)}
                  disabled={cancel.isPending}
                >
                  <Undo2 className="w-3.5 h-3.5" />
                  {a.status === "pending" ? "撤回" : "撤销"}
                </Button>
              )}
            </div>
          ))
        )}
      </div>

      {(otReqs ?? []).length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            加班调休
          </h3>
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            {(otReqs ?? []).map((r, i) => (
              <div
                key={r.id}
                className={`flex items-center gap-3 px-6 py-3.5 ${
                  i < (otReqs ?? []).length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-[13px] text-[var(--color-text-primary)]">
                      {r.work_date}
                    </span>
                    <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
                      加班 {r.hours}h → {r.leave_type_name}{" "}
                      <span className="font-mono">{r.comp_days}</span> 天
                    </span>
                    <StatusPill status={r.status} />
                  </div>
                  {r.reason && (
                    <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                      {r.reason}
                    </span>
                  )}
                  {r.decision_note && (
                    <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                      审批意见: {r.decision_note}
                    </span>
                  )}
                </div>
                {(r.status === "pending" || r.status === "approved") && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onCancelOt(r)}
                    disabled={cancelOt.isPending}
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

      <ApplyModal open={open} onClose={() => setOpen(false)} />
      <OvertimeModal open={otOpen} onClose={() => setOtOpen(false)} />
    </main>
  );
}

function ApplyModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: types } = useQuery<MeLeaveType[]>({
    queryKey: ["me-leave-types"],
    queryFn: () => fetchJSON("/api/me/leave-types"),
  });
  const [typeId, setTypeId] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [err, setErr] = useState("");
  // 工作日数提示 — 服务端按节假日历算,days 留空即按它提交
  const { data: wd } = useQuery<{ workdays: number }>({
    queryKey: ["workday-count", start, end],
    queryFn: () =>
      fetchJSON(`/api/me/workday-count?start=${start}&end=${end}`),
    enabled: !!start && !!end && end >= start,
  });
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/leave-applications", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-leave-apps"] });
      qc.invalidateQueries({ queryKey: ["me-balances"] });
      onClose();
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!typeId) {
      setErr("请选择假期类型");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    const daysStr = String(fd.get("days") || "").trim();
    // days 留空 → 不传,服务端按工作日历算(周末/节假日不算请假)
    mut.mutate({
      leave_type_id: typeId,
      start_date: String(fd.get("start_date")),
      end_date: String(fd.get("end_date")),
      ...(daysStr ? { days: Number(daysStr) } : {}),
      reason: String(fd.get("reason") || "").trim(),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="申请请假"
      description="提交后进入主管审批;撤回只适用于待审批的申请。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="me-leave" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="me-leave" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="假期类型" required>
          <Select
            value={typeId}
            onChange={setTypeId}
            options={(types ?? []).map((t) => ({
              value: t.id,
              label: `${t.name}${t.is_paid ? "" : "(无薪)"}`,
            }))}
            placeholder="选择类型"
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="开始日期" required>
            <Input
              name="start_date"
              type="date"
              required
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </Field>
          <Field label="结束日期" required>
            <Input
              name="end_date"
              type="date"
              required
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </Field>
        </div>
        <Field
          label={
            wd != null
              ? `天数(区间内 ${wd.workdays} 个工作日)`
              : "天数"
          }
        >
          <Input
            name="days"
            type="number"
            step="0.5"
            min="0.5"
            placeholder={
              wd != null ? `留空按 ${wd.workdays} 天计` : "留空按工作日自动算"
            }
          />
        </Field>
        {wd != null && wd.workdays === 0 && (
          <div className="text-[13px] text-[var(--color-warning-text)] font-body">
            所选区间内没有工作日(周末/节假日),请调整日期。
          </div>
        )}
        <Field label="事由">
          <Input name="reason" placeholder="例如:家中事务" />
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

function OvertimeModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: types } = useQuery<MeLeaveType[]>({
    queryKey: ["me-leave-types"],
    queryFn: () => fetchJSON("/api/me/leave-types"),
  });
  const compTypes = (types ?? []).filter((t) => t.is_comp);
  const [typeId, setTypeId] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/overtime-requests", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-ot-reqs"] });
      onClose();
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!typeId) {
      setErr("请选择结算到的调休类型");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      leave_type_id: typeId,
      work_date: String(fd.get("work_date")),
      hours: Number(fd.get("hours")),
      reason: String(fd.get("reason") || "").trim(),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="申请加班"
      description="主管批准后按日标准工时折算调休额度(0.5 天粒度),自动计入假期余额。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="me-ot" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="me-ot" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="结算类型" required>
          <Select
            value={typeId}
            onChange={setTypeId}
            options={compTypes.map((t) => ({ value: t.id, label: t.name }))}
            placeholder={
              compTypes.length === 0
                ? "还没有调休类型 — 请联系 HR 创建"
                : "选择调休类型"
            }
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="加班日期" required>
            <Input name="work_date" type="date" required />
          </Field>
          <Field label="加班时长(小时)" required>
            <Input
              name="hours"
              type="number"
              step="0.5"
              min="0.5"
              max="24"
              required
              placeholder="8"
            />
          </Field>
        </div>
        <Field label="事由">
          <Input name="reason" placeholder="例如:版本发布、客户紧急支持" />
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
