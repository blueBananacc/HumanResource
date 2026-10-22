import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check, ClipboardList, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { StatusPill, type Requisition } from "@/lib/hris";
import { Field, useOrgOptions } from "./Employees";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";

const LEVEL_OPTIONS = [
  { value: "junior", label: "初级" },
  { value: "intermediate", label: "中级" },
  { value: "senior", label: "高级" },
  { value: "expert", label: "专家" },
];
const LEVEL_LABEL = Object.fromEntries(
  LEVEL_OPTIONS.map((o) => [o.value, o.label]),
);
const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "pending", label: "待审批" },
  { value: "approved", label: "已批准" },
  { value: "rejected", label: "已驳回" },
];

export function Requisitions() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const [open, setOpen] = useState(false);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  const canApprove = hasPermission(PERM.APPROVE_REQUISITIONS);

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<Requisition>({
      key: ["requisitions"],
      url: "/api/requisitions",
      params: { status },
    });
  const items = data?.items ?? [];

  const act = useMutation({
    mutationFn: ({
      id,
      action,
    }: {
      id: string;
      action: "approve" | "reject" | "to-job";
    }) =>
      fetchJSON(`/api/requisitions/${id}/${action}`, {
        method: "POST",
        body: JSON.stringify({ note: notes[id] || null }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["requisitions"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            招聘需求审批
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            用人部门提编制需求,HR 审批后可一键转成招聘岗位。
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="w-4 h-4" />
          提需求
        </Button>
      </div>

      <div className="flex items-center gap-3">
        <Select
          value={status}
          onChange={(v) =>
            setSp((prev) => {
              const next = new URLSearchParams(prev);
              if (v) next.set("status", v);
              else next.delete("status");
              next.delete("page");
              return next;
            })
          }
          options={STATUS_OPTIONS}
          className="w-[140px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          条
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={ClipboardList}
            title="没有招聘需求"
            description="点右上角「提需求」创建编制申请,审批通过后可转为招聘岗位。"
            action={<Button onClick={() => setOpen(true)}>提需求</Button>}
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
                  {r.title}
                </span>
                <span className="text-[12px] font-mono text-[var(--color-text-secondary)]">
                  {r.headcount} 人 · {LEVEL_LABEL[r.level] ?? r.level}
                </span>
                <StatusPill status={r.status} label={r.job_id ? "已转岗位" : undefined} />
                {r.job_title && (
                  <Link
                    to={`/jobs/${r.job_id}`}
                    className="text-[12px] text-[var(--color-info)] font-body hover:underline"
                  >
                    → {r.job_title}
                  </Link>
                )}
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {formatRelative(r.created_at)}
                </span>
              </div>
              <div className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                {r.org_unit_name ?? "未指定部门"}
                {r.reason && ` · ${r.reason}`}
                {r.requested_by_email && ` · 提交人 ${r.requested_by_email}`}
              </div>
              {r.status === "pending" && canApprove && (
                <div className="mt-3 flex items-center gap-2">
                  <Input
                    value={notes[r.id] ?? ""}
                    onChange={(e) =>
                      setNotes((p) => ({ ...p, [r.id]: e.target.value }))
                    }
                    placeholder="审批意见(可选)"
                    className="w-[300px] h-8 text-[13px]"
                  />
                  <Button
                    size="sm"
                    onClick={() => act.mutate({ id: r.id, action: "approve" })}
                    disabled={act.isPending}
                  >
                    <Check className="w-3.5 h-3.5" />
                    批准
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => act.mutate({ id: r.id, action: "reject" })}
                    disabled={act.isPending}
                  >
                    <X className="w-3.5 h-3.5" />
                    驳回
                  </Button>
                </div>
              )}
              {r.status === "approved" && !r.job_id && canApprove && (
                <div className="mt-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => act.mutate({ id: r.id, action: "to-job" })}
                    disabled={act.isPending}
                  >
                    <ArrowRight className="w-3.5 h-3.5" />
                    转成招聘岗位
                  </Button>
                </div>
              )}
              {r.decision_note && r.status !== "pending" && (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                  {r.approver_email && `审批人 ${r.approver_email} · `}
                  {r.decision_note}
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

      <ReqCreateModal open={open} onClose={() => setOpen(false)} />
    </main>
  );
}

function ReqCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const orgOptions = useOrgOptions();
  const [orgId, setOrgId] = useState("");
  const [level, setLevel] = useState("intermediate");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/requisitions", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["requisitions"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      title: String(fd.get("title") || "").trim(),
      org_unit_id: orgId || null,
      headcount: Number(fd.get("headcount") || 1),
      level,
      reason: String(fd.get("reason") || "").trim(),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="提招聘需求"
      description="提交后进入待审批;HR 批准后可转成招聘岗位进入常规招聘流程。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="req-create" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="req-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="需求标题" required>
          <Input name="title" required placeholder="研发部-高级前端工程师" />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="部门">
            <Select
              value={orgId}
              onChange={setOrgId}
              options={orgOptions}
              placeholder="选择部门"
            />
          </Field>
          <Field label="级别">
            <Select value={level} onChange={setLevel} options={LEVEL_OPTIONS} />
          </Field>
          <Field label="人数" required>
            <Input
              name="headcount"
              type="number"
              min="1"
              required
              defaultValue={1}
            />
          </Field>
        </div>
        <Field label="理由">
          <Input name="reason" placeholder="业务扩张 / 离职补位…" />
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
