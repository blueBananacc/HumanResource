import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardList, Plus, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Field } from "@/pages/hris/Employees";

type ReqComment = {
  id: string;
  author_email: string | null;
  body: string;
  created_at: string;
};

type MyRequest = {
  id: string;
  kind: string;
  subject: string;
  detail: string | null;
  status: string;
  decision_note: string | null;
  comments: ReqComment[];
  created_at: string;
};

const KIND_LABEL: Record<string, string> = {
  general: "通用",
  training: "培训",
  benefit: "福利",
  equipment: "设备申领",
  wellness: "健康关怀",
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  pending: { label: "待审批", cls: "bg-amber-50 text-amber-700" },
  approved: { label: "已批准", cls: "bg-green-50 text-green-700" },
  rejected: { label: "已驳回", cls: "bg-red-50 text-red-700" },
  cancelled: { label: "已取消", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
};

export function MyRequests() {
  const [createOpen, setCreateOpen] = useState(false);
  const [viewing, setViewing] = useState<MyRequest | null>(null);
  const qc = useQueryClient();
  const { data, isLoading, isError } = useQuery<MyRequest[]>({
    queryKey: ["me-requests"],
    queryFn: () => fetchJSON("/api/me/requests"),
  });
  const items = data ?? [];

  const cancel = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/me/requests/${id}/cancel`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-requests"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">我的申请</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            培训 / 福利 / IT / 行政等诉求,统一在这里提交和跟踪。
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="w-4 h-4" />提申请
        </Button>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <div className="p-12 text-center text-sm text-[var(--color-danger)] font-body">加载失败,请刷新重试。</div>
        ) : items.length === 0 ? (
          <Empty
            icon={ClipboardList}
            title="还没有申请"
            description="有事找 HR?在这里提申请,进展会有通知。"
            action={<Button onClick={() => setCreateOpen(true)}>提申请</Button>}
          />
        ) : (
          items.map((r, i) => {
            const st = STATUS_LABEL[r.status] ?? STATUS_LABEL.pending;
            return (
              <div
                key={r.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(r)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{r.subject}</span>
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">
                    {KIND_LABEL[r.kind] ?? r.kind}
                  </span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <div className="flex-1" />
                  {r.status === "pending" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); cancel.mutate(r.id); }}
                      disabled={cancel.isPending}
                    >
                      <Undo2 className="w-3.5 h-3.5" />取消
                    </Button>
                  )}
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(r.created_at)}</span>
                </div>
                {r.decision_note && (
                  <p className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">批注:{r.decision_note}</p>
                )}
              </div>
            );
          })
        )}
      </div>
      {createOpen && <RequestCreate onClose={() => setCreateOpen(false)} />}
      {viewing && <RequestView req={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

function RequestCreate({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [kind, setKind] = useState("general");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/requests", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-requests"] });
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
      subject: String(fd.get("subject") || "").trim(),
      detail: String(fd.get("detail") || "").trim() || null,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title="提申请"
      width={520}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="req-form" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form id="req-form" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="类型">
          <Select
            value={kind}
            onChange={setKind}
            options={Object.entries(KIND_LABEL).map(([v, l]) => ({ value: v, label: l }))}
          />
        </Field>
        <Field label="主题" required>
          <Input name="subject" maxLength={128} required placeholder="如:申请参加外部培训" />
        </Field>
        <Field label="说明">
          <textarea
            name="detail"
            rows={5}
            maxLength={5000}
            className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 text-sm font-body outline-none focus:border-[var(--color-accent)]"
          />
        </Field>
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}

function RequestView({ req, onClose }: { req: MyRequest; onClose: () => void }) {
  const qc = useQueryClient();
  const comment = useMutation({
    mutationFn: (body: string) =>
      fetchJSON(`/api/me/requests/${req.id}/comments`, { method: "POST", body: JSON.stringify({ body }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-requests"] }),
  });
  const st = STATUS_LABEL[req.status] ?? STATUS_LABEL.pending;

  return (
    <Modal open onClose={onClose} title={req.subject} width={520} footer={<Button onClick={onClose}>关闭</Button>}>
      <div className="flex flex-col gap-3 text-[13px] font-body">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
          <span className="text-[var(--color-text-secondary)]">{KIND_LABEL[req.kind] ?? req.kind}</span>
        </div>
        {req.detail && <p className="whitespace-pre-wrap text-[var(--color-text-primary)]">{req.detail}</p>}
        {req.decision_note && (
          <div className="rounded-lg bg-[var(--color-surface-1)] px-3 py-2">批注:{req.decision_note}</div>
        )}
        {(req.comments ?? []).map((cm) => (
          <div key={cm.id} className="text-[12px]">
            <span className="text-[var(--color-text-tertiary)]">{cm.author_email ?? "?"} · {formatRelative(cm.created_at)}</span>
            <p className="text-[var(--color-text-primary)]">{cm.body}</p>
          </div>
        ))}
        {req.status !== "cancelled" && (
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              const fd = new FormData(e.currentTarget);
              const body = String(fd.get("cbody") || "").trim();
              if (body) comment.mutate(body);
              e.currentTarget.reset();
            }}
          >
            <Input name="cbody" placeholder="补充说明…" maxLength={2000} />
            <Button type="submit" variant="secondary" size="sm" disabled={comment.isPending}>发送</Button>
          </form>
        )}
      </div>
    </Modal>
  );
}
