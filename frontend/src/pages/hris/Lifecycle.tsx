import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Plus, Shuffle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  LIFECYCLE_KIND_LABEL,
  type LifecycleEvent,
  type LifecycleTemplate,
} from "@/lib/hris";
import { Field, PageError, useEmployeeOptions, useOrgOptions } from "./Employees";

const KIND_OPTIONS = Object.entries(LIFECYCLE_KIND_LABEL).map(
  ([value, label]) => ({ value, label }),
);

export function Lifecycle() {
  const [createOpen, setCreateOpen] = useState(false);
  const qc = useQueryClient();
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<LifecycleEvent>({
      key: ["lifecycle-events"],
      url: "/api/lifecycle/events",
    });
  const { data: templates } = useQuery<LifecycleTemplate[]>({
    queryKey: ["lifecycle-templates"],
    queryFn: () => fetchJSON("/api/lifecycle/templates"),
  });
  const items = data?.items ?? [];

  const toggleTask = useMutation({
    mutationFn: (taskId: string) =>
      fetchJSON(`/api/lifecycle/tasks/${taskId}/toggle`, { method: "POST" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["lifecycle-events"] }),
  });
  const cancelEvent = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/lifecycle/events/${id}/cancel`, { method: "POST" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["lifecycle-events"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            生命周期
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            入职 / 调动 / 晋升 / 离职清单 — 任务全部勾完自动标记完成。
          </p>
        </div>
        {!isError && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4" />
            发起事件
          </Button>
        )}
      </div>

      {templates && templates.length > 0 && (
        <div className="flex flex-wrap gap-2 text-[12px] font-body text-[var(--color-text-secondary)]">
          <span>模板:</span>
          {templates.map((t) => (
            <span
              key={t.id}
              className="px-2 py-0.5 rounded-md bg-[var(--color-bg-subtle)]"
            >
              {t.name}({LIFECYCLE_KIND_LABEL[t.kind] ?? t.kind} ·{" "}
              {t.tasks.length} 项{t.is_default ? " · 默认" : ""})
            </span>
          ))}
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
            icon={Shuffle}
            title="还没有生命周期事件"
            description="入职通常在 Offer 接受时自动创建;调动/晋升/离职在这里手动发起。"
            action={<Button onClick={() => setCreateOpen(true)}>发起事件</Button>}
          />
        ) : (
          items.map((ev, i) => {
            const doneCount = ev.tasks.filter((t) => t.done).length;
            return (
              <div
                key={ev.id}
                className={`px-6 py-5 ${
                  i < items.length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="px-2 py-0.5 rounded-md bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-[11px] font-medium font-body">
                    {LIFECYCLE_KIND_LABEL[ev.kind] ?? ev.kind}
                  </span>
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                    {ev.employee_name}
                  </span>
                  <StatusPill status={ev.status} />
                  {ev.effective_date && (
                    <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                      生效 {ev.effective_date}
                    </span>
                  )}
                  <div className="flex-1" />
                  <span className="text-[12px] font-mono text-[var(--color-text-tertiary)]">
                    {doneCount}/{ev.tasks.length}
                  </span>
                  {ev.status !== "completed" && ev.status !== "cancelled" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => cancelEvent.mutate(ev.id)}
                    >
                      取消
                    </Button>
                  )}
                </div>
                {ev.note && (
                  <div className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                    {ev.note}
                  </div>
                )}
                <div className="mt-3 grid grid-cols-2 gap-x-8 gap-y-1.5">
                  {ev.tasks.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => toggleTask.mutate(t.id)}
                      disabled={ev.status === "completed" || ev.status === "cancelled"}
                      className="flex items-center gap-2 text-left text-[13px] font-body text-[var(--color-text-primary)] disabled:opacity-60 group"
                    >
                      <span
                        className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors ${
                          t.done
                            ? "bg-[var(--color-accent)] border-[var(--color-accent)]"
                            : "border-[var(--color-border-subtle)] group-hover:border-[var(--color-accent)]"
                        }`}
                      >
                        {t.done && <Check className="w-3 h-3 text-white" />}
                      </span>
                      <span className={t.done ? "line-through text-[var(--color-text-tertiary)]" : ""}>
                        {t.title}
                      </span>
                    </button>
                  ))}
                </div>
                <div className="mt-2 text-[11px] text-[var(--color-text-tertiary)] font-body">
                  创建于 {formatRelative(ev.created_at)}
                  {ev.completed_at && ` · 完成于 ${formatRelative(ev.completed_at)}`}
                </div>
              </div>
            );
          })
        )}
      </div>

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />

      <EventCreateModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        templates={templates ?? []}
      />
    </main>
  );
}

function EventCreateModal({
  open,
  onClose,
  templates,
}: {
  open: boolean;
  onClose: () => void;
  templates: LifecycleTemplate[];
}) {
  const qc = useQueryClient();
  const empOptions = useEmployeeOptions(open);
  const orgOptions = useOrgOptions();
  const [empId, setEmpId] = useState("");
  const [kind, setKind] = useState("onboarding");
  const [tplId, setTplId] = useState("");
  const [toOrgId, setToOrgId] = useState("");
  const [err, setErr] = useState("");
  const tplOptions = templates
    .filter((t) => t.kind === kind)
    .map((t) => ({ value: t.id, label: `${t.name}(${t.tasks.length} 项)` }));

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/lifecycle/events", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lifecycle-events"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!empId) {
      setErr("请选择员工");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    // transfer/promotion 的目标值放 payload,任务全部完成时后端回写员工档案
    const payload: Record<string, string> = {};
    if (kind === "transfer") {
      if (!toOrgId) {
        setErr("调动事件必须选择目标部门");
        return;
      }
      payload.to_org_unit_id = toOrgId;
    }
    if (kind === "promotion") {
      const toDesig = String(fd.get("to_designation") || "").trim();
      if (!toDesig) {
        setErr("晋升事件必须填写新职位");
        return;
      }
      payload.to_designation = toDesig;
    }
    mut.mutate({
      employee_id: empId,
      kind,
      effective_date: String(fd.get("effective_date") || "") || null,
      template_id: tplId || null,
      payload: Object.keys(payload).length ? payload : null,
      note: String(fd.get("note") || "").trim() || null,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="发起生命周期事件"
      description="任务清单来自所选模板;不选模板则按事件类型生成默认清单。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="lc-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="lc-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="员工" required>
          <Select
            value={empId}
            onChange={setEmpId}
            options={empOptions}
            placeholder="选择员工"
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="事件类型" required>
            <Select value={kind} onChange={setKind} options={KIND_OPTIONS} />
          </Field>
          <Field label="生效日期">
            <Input name="effective_date" type="date" />
          </Field>
          {kind === "transfer" && (
            <Field label="目标部门" required>
              <Select
                value={toOrgId}
                onChange={setToOrgId}
                options={orgOptions}
                placeholder="选择部门"
              />
            </Field>
          )}
          {kind === "promotion" && (
            <Field label="新职位" required>
              <Input name="to_designation" placeholder="高级前端工程师" />
            </Field>
          )}
        </div>
        {tplOptions.length > 0 && (
          <Field label="任务模板">
            <Select
              value={tplId}
              onChange={setTplId}
              options={tplOptions}
              placeholder="默认清单"
            />
          </Field>
        )}
        <Field label="备注">
          <Input name="note" placeholder="例如:从研发一部调至研发二部" />
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
