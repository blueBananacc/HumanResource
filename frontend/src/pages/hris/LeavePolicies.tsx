import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers, Play, Plus, Trash2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  type Employee,
  type LeavePolicy,
  type LeaveType,
} from "@/lib/hris";
import { Field, PageError } from "./Employees";

export function LeavePolicies() {
  const [open, setOpen] = useState(false);
  const [assigning, setAssigning] = useState<LeavePolicy | null>(null);
  const [running, setRunning] = useState<LeavePolicy | null>(null);
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [err, setErr] = useState("");

  const { data, isLoading, isError, error } = useQuery<LeavePolicy[]>({
    queryKey: ["leave-policies"],
    queryFn: () => fetchJSON("/api/leave/policies"),
  });
  const items = data ?? [];

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/leave/policies/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["leave-policies"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "删除失败"),
  });

  async function onDelete(p: LeavePolicy) {
    const ok = await confirm({
      title: `删除政策「${p.name}」?`,
      description:
        p.assigned_count > 0
          ? `仍有 ${p.assigned_count} 名员工绑定该政策,需先解除绑定。`
          : "删除不影响已发放的额度。",
      confirmLabel: "删除",
      tone: "danger",
    });
    if (ok) del.mutate(p.id);
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            假期政策
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            一组「类型 × 天数」+ 结转上限 — 绑定员工后按年自动发放额度,台账可查。
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="w-4 h-4" />
          新建政策
        </Button>
      </div>

      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </div>
      )}

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={Layers}
            title="还没有假期政策"
            description="建一个政策(如「正式员工包:年假10 + 病假5」),批量绑定员工就能免手工发额度。"
            action={<Button onClick={() => setOpen(true)}>新建政策</Button>}
          />
        ) : (
          items.map((p, i) => (
            <div
              key={p.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {p.name}
                </span>
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {p.assigned_count} 人绑定
                </span>
                {p.carry_forward != null && (
                  <span className="px-2 py-0.5 rounded-md bg-[var(--color-info-soft)] text-[11px] font-body text-[var(--color-info)]">
                    结转上限 {p.carry_forward} 天
                  </span>
                )}
                <div className="flex-1" />
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setAssigning(p)}
                >
                  <UserPlus className="w-3.5 h-3.5" />
                  绑定员工
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setRunning(p)}
                  disabled={p.assigned_count === 0}
                  title={p.assigned_count === 0 ? "先绑定员工" : "年度发放"}
                >
                  <Play className="w-3.5 h-3.5" />
                  年度发放
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDelete(p)}
                  disabled={del.isPending}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
              <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                {p.items.map((it) => (
                  <span key={it.leave_type_id}>
                    {it.leave_type_name}{" "}
                    <span className="font-mono">{it.days}</span> 天
                  </span>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      <PolicyModal open={open} onClose={() => setOpen(false)} />
      {assigning && (
        <AssignModal policy={assigning} onClose={() => setAssigning(null)} />
      )}
      {running && (
        <RunModal policy={running} onClose={() => setRunning(null)} />
      )}
    </main>
  );
}

function PolicyModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: types } = useQuery<LeaveType[]>({
    queryKey: ["leave-types"],
    queryFn: () => fetchJSON("/api/leave/types"),
  });
  const [rows, setRows] = useState<{ leave_type_id: string; days: string }[]>([
    { leave_type_id: "", days: "10" },
  ]);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/leave/policies", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-policies"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const items = rows
      .filter((r) => r.leave_type_id)
      .map((r) => ({ leave_type_id: r.leave_type_id, days: Number(r.days) }));
    if (items.length === 0) {
      setErr("至少配置一个假期类型");
      return;
    }
    const carry = String(fd.get("carry_forward") || "").trim();
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      items,
      carry_forward: carry === "" ? null : Number(carry),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建假期政策"
      description="绑定员工后按年自动发放各类型额度;结转上限控制上年余额带入(留空不结转)。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="policy" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="policy" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="政策名称" required>
          <Input name="name" required placeholder="正式员工假勤包" />
        </Field>
        <div className="flex flex-col gap-2">
          <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
            发放项
          </span>
          {rows.map((r, i) => (
            <div key={i} className="flex items-center gap-2">
              <Select
                value={r.leave_type_id}
                onChange={(v) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, leave_type_id: v } : x,
                    ),
                  )
                }
                options={(types ?? []).map((t) => ({
                  value: t.id,
                  label: t.name,
                  // 调休额度只由加班结算,政策包不收
                  disabled: t.is_comp,
                  hint: t.is_comp ? "调休类型" : undefined,
                }))}
                placeholder="假期类型"
                className="flex-1"
              />
              <Input
                type="number"
                step="0.5"
                min="0"
                value={r.days}
                onChange={(e) =>
                  setRows((p) =>
                    p.map((x, j) =>
                      j === i ? { ...x, days: e.target.value } : x,
                    ),
                  )
                }
                className="w-[90px]"
              />
              <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                天/年
              </span>
              {rows.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setRows((p) => p.filter((_, j) => j !== i))
                  }
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              )}
            </div>
          ))}
          <Button
            variant="secondary"
            size="sm"
            className="self-start"
            onClick={() =>
              setRows((p) => [...p, { leave_type_id: "", days: "5" }])
            }
          >
            <Plus className="w-3.5 h-3.5" />
            添加类型
          </Button>
        </div>
        <Field label="结转上限(天)">
          <Input
            name="carry_forward"
            type="number"
            step="0.5"
            min="0"
            placeholder="留空 = 不结转"
          />
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

function AssignModal({
  policy,
  onClose,
}: {
  policy: LeavePolicy;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: emps } = useQuery<{ items: Employee[] }>({
    queryKey: ["employee-options"],
    queryFn: () => fetchJSON("/api/employees?limit=200"),
    staleTime: 30_000,
  });
  // 勾选状态 = 期望的绑定态:toBind 是待新绑定的,toUnbind 是待解绑的
  const [toBind, setToBind] = useState<Set<string>>(new Set());
  const [toUnbind, setToUnbind] = useState<Set<string>>(new Set());
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: async (body: { bind: string[]; unbind: string[] }) => {
      // 两个端点各管一段:绑定会按当年发放额度,解绑只脱钩不回冲
      if (body.bind.length) {
        await fetchJSON(`/api/leave/policies/${policy.id}/assign`, {
          method: "POST",
          body: JSON.stringify({ employee_ids: body.bind }),
        });
      }
      if (body.unbind.length) {
        await fetchJSON(`/api/leave/policies/${policy.id}/unassign`, {
          method: "POST",
          body: JSON.stringify({ employee_ids: body.unbind }),
        });
      }
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-policies"] });
      qc.invalidateQueries({ queryKey: ["employee-options"] });
      qc.invalidateQueries({ queryKey: ["leave-balances"] });
      qc.invalidateQueries({ queryKey: ["leave-ledger"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  const bound = new Set(
    (emps?.items ?? [])
      .filter((e) => e.leave_policy_id === policy.id)
      .map((e) => e.id),
  );
  const changes = toBind.size + toUnbind.size;

  function toggle(id: string) {
    const flip = (p: Set<string>) => {
      const n = new Set(p);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    };
    if (bound.has(id)) setToUnbind(flip);
    else setToBind(flip);
  }

  function submit() {
    if (changes === 0) {
      setErr("没有变更 — 勾选绑定新员工,或取消勾选解除绑定");
      return;
    }
    mut.mutate({ bind: [...toBind], unbind: [...toUnbind] });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`管理绑定 — ${policy.name}`}
      description="勾选保存后生效:新绑定的员工立即按今年发放政策额度(差额写台账);取消勾选解除绑定(已发放的额度保留)。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button onClick={submit} disabled={mut.isPending}>
            {mut.isPending
              ? "保存中…"
              : `保存${changes > 0 ? `(${changes} 项变更)` : ""}`}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-2 max-h-[320px] overflow-y-auto">
        {(emps?.items ?? []).map((e) => {
          const isBound = bound.has(e.id);
          const unbinding = toUnbind.has(e.id);
          const checked = isBound ? !unbinding : toBind.has(e.id);
          return (
            <label
              key={e.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg border text-[13px] font-body ${
                unbinding
                  ? "border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)]"
                  : checked
                    ? "border-[var(--color-text-primary)] bg-[var(--color-bg-subtle)]"
                    : "border-[var(--color-border-subtle)] hover:border-[var(--color-text-tertiary)]"
              }`}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(e.id)}
              />
              <span className="text-[var(--color-text-primary)]">
                {e.name}
              </span>
              <span className="font-mono text-[11px] text-[var(--color-text-tertiary)]">
                {e.employee_no}
              </span>
              <div className="flex-1" />
              {unbinding ? (
                <span className="text-[11px] text-[var(--color-danger)]">
                  将解绑
                </span>
              ) : isBound ? (
                <span className="text-[11px] text-[var(--color-text-tertiary)]">
                  已绑定
                </span>
              ) : null}
            </label>
          );
        })}
      </div>
      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </div>
      )}
    </Modal>
  );
}

function RunModal({
  policy,
  onClose,
}: {
  policy: LeavePolicy;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const year = new Date().getFullYear();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: { year: number }) =>
      fetchJSON(`/api/leave/policies/${policy.id}/run`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["leave-policies"] });
      qc.invalidateQueries({ queryKey: ["leave-balances"] });
      qc.invalidateQueries({ queryKey: ["leave-ledger"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "发放失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({ year: Number(fd.get("year")) || year });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`年度发放 — ${policy.name}`}
      description={`对全部 ${policy.assigned_count} 名绑定员工按政策对齐额度:目标 = 政策天数 + 上年结转(上限 ${policy.carry_forward ?? "不结转"}),差额写台账,重复执行幂等。`}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="policy-run" disabled={mut.isPending}>
            {mut.isPending ? "发放中…" : "发放"}
          </Button>
        </>
      }
    >
      <form id="policy-run" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="发放年份" required>
          <Input
            name="year"
            type="number"
            min="2000"
            max="2100"
            defaultValue={year}
            required
          />
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
