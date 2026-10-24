import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LifeBuoy, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Field } from "@/pages/hris/Employees";

type CaseUpdate = {
  id: string;
  author_email: string | null;
  body: string;
  is_internal: boolean;
  created_at: string;
};

type MyCase = {
  id: string;
  case_no: number;
  subject: string;
  kind: string;
  priority: string;
  status: string;
  description: string | null;
  resolution: string | null;
  updates: CaseUpdate[];
  created_at: string;
};

const KIND_LABEL: Record<string, string> = {
  payroll: "薪酬",
  it: "IT",
  admin: "行政",
  facility: "设施",
  leave_dispute: "假勤争议",
  other: "其他",
};

const PRIO_LABEL: Record<string, string> = { low: "低", normal: "普通", high: "高", urgent: "紧急" };

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  new: { label: "已提交", cls: "bg-blue-50 text-blue-700" },
  assigned: { label: "已受理", cls: "bg-violet-50 text-violet-700" },
  in_progress: { label: "处理中", cls: "bg-amber-50 text-amber-700" },
  resolved: { label: "已解决", cls: "bg-green-50 text-green-700" },
  closed: { label: "已关闭", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
};

export function MyCases() {
  const [createOpen, setCreateOpen] = useState(false);
  const [viewing, setViewing] = useState<MyCase | null>(null);
  const { data, isLoading, isError } = useQuery<MyCase[]>({
    queryKey: ["me-cases"],
    queryFn: () => fetchJSON("/api/me/cases"),
  });
  const items = data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">我的工单</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            薪酬疑问、IT 故障、设施报修…提给 HR 服务台,进展可见。
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4" />提工单
        </Button>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <div className="p-12 text-center text-sm text-[var(--color-danger)] font-body">加载失败,请刷新重试。</div>
        ) : items.length === 0 ? (
          <Empty
            icon={LifeBuoy}
            title="还没有工单"
            description="遇到问题找 HR?在这里提单,处理进展会有通知。"
            action={<Button onClick={() => setCreateOpen(true)}>提工单</Button>}
          />
        ) : (
          items.map((c, i) => {
            const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.new;
            return (
              <div
                key={c.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(c)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[12px] text-[var(--color-text-tertiary)]">HR-{c.case_no}</span>
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{c.subject}</span>
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">
                    {KIND_LABEL[c.kind] ?? c.kind} · {PRIO_LABEL[c.priority] ?? c.priority}
                  </span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <div className="flex-1" />
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(c.created_at)}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
      {createOpen && <CaseCreate onClose={() => setCreateOpen(false)} />}
      {viewing && <CaseView c={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

function CaseCreate({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [kind, setKind] = useState("other");
  const [priority, setPriority] = useState("normal");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/cases", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-cases"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      kind,
      priority,
      subject: String(fd.get("subject") || "").trim(),
      description: String(fd.get("description") || "").trim() || null,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title="提工单"
      width={520}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="case-form" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="case-form" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="类型">
            <Select value={kind} onChange={setKind} options={Object.entries(KIND_LABEL).map(([v, l]) => ({ value: v, label: l }))} />
          </Field>
          <Field label="优先级">
            <Select value={priority} onChange={setPriority} options={Object.entries(PRIO_LABEL).map(([v, l]) => ({ value: v, label: l }))} />
          </Field>
        </div>
        <Field label="主题" required>
          <Input name="subject" maxLength={128} required placeholder="如:工资条明细疑问" />
        </Field>
        <Field label="描述">
          <textarea
            name="description"
            rows={5}
            maxLength={10000}
            className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 text-sm font-body outline-none focus:border-[var(--color-accent)]"
          />
        </Field>
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}

function CaseView({ c, onClose }: { c: MyCase; onClose: () => void }) {
  const qc = useQueryClient();
  const comment = useMutation({
    mutationFn: (body: string) =>
      fetchJSON(`/api/me/cases/${c.id}/updates`, { method: "POST", body: JSON.stringify({ body, is_internal: false }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-cases"] }),
  });
  const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.new;

  return (
    <Modal open onClose={onClose} title={`HR-${c.case_no} ${c.subject}`} width={560} footer={<Button onClick={onClose}>关闭</Button>}>
      <div className="flex flex-col gap-3 text-[13px] font-body">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
          <span className="text-[var(--color-text-secondary)]">{KIND_LABEL[c.kind] ?? c.kind}</span>
        </div>
        {c.description && <p className="whitespace-pre-wrap text-[var(--color-text-primary)]">{c.description}</p>}
        {c.resolution && (
          <div className="rounded-lg bg-green-50 px-3 py-2 text-green-800">解决方案:{c.resolution}</div>
        )}
        {(c.updates ?? []).length > 0 && (
          <div className="flex flex-col gap-2 border-t border-[var(--color-border-row)] pt-3">
            {c.updates.map((u) => (
              <div key={u.id} className="text-[12px]">
                <span className="text-[var(--color-text-tertiary)]">{u.author_email ?? "?"} · {formatRelative(u.created_at)}</span>
                <p className="text-[var(--color-text-primary)]">{u.body}</p>
              </div>
            ))}
          </div>
        )}
        {c.status !== "closed" && (
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              const fd = new FormData(e.currentTarget);
              const body = String(fd.get("ubody") || "").trim();
              if (body) comment.mutate(body);
              e.currentTarget.reset();
            }}
          >
            <Input name="ubody" placeholder="补充信息…" maxLength={3000} />
            <Button type="submit" variant="secondary" size="sm" disabled={comment.isPending}>发送</Button>
          </form>
        )}
      </div>
    </Modal>
  );
}
