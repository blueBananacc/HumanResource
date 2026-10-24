import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, ClipboardList, MessageSquare, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { Field, PageError } from "./Employees";

type ReqComment = {
  id: string;
  author_user_id: string | null;
  author_email: string | null;
  body: string;
  created_at: string;
};

type EmpRequest = {
  id: string;
  employee_id: string;
  employee_name: string | null;
  kind: string;
  subject: string;
  detail: string | null;
  status: string;
  decided_at: string | null;
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

export function EmployeeRequests() {
  const [statusFilter, setStatusFilter] = useState("pending");
  const [viewing, setViewing] = useState<string | null>(null);
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<EmpRequest>({
      key: ["emp-requests"],
      url: "/api/requests",
      params: { status_filter: statusFilter },
    });
  const items = data?.items ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">员工申请</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            员工统一申请入口 — 培训/福利/IT/行政等诉求的受理与审批。
          </p>
        </div>
        <div className="w-[160px]">
          <Select
            value={statusFilter}
            onChange={setStatusFilter}
            options={[
              { value: "pending", label: "待审批" },
              { value: "approved", label: "已批准" },
              { value: "rejected", label: "已驳回" },
              { value: "cancelled", label: "已取消" },
              { value: "", label: "全部" },
            ]}
          />
        </div>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty icon={ClipboardList} title="没有申请" description="当前筛选下没有员工申请。" />
        ) : (
          items.map((r, i) => {
            const st = STATUS_LABEL[r.status] ?? STATUS_LABEL.pending;
            return (
              <div
                key={r.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(r.id)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{r.subject}</span>
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">
                    {KIND_LABEL[r.kind] ?? r.kind}
                  </span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <div className="flex-1" />
                  <span className="text-[12px] text-[var(--color-text-secondary)] font-body">{r.employee_name}</span>
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(r.created_at)}</span>
                </div>
                {r.detail && <p className="mt-1 text-[13px] text-[var(--color-text-secondary)] font-body line-clamp-1">{r.detail}</p>}
              </div>
            );
          })
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {viewing && <RequestDetail rid={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

function RequestDetail({ rid, onClose }: { rid: string; onClose: () => void }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");

  // 列表接口不带 comments — 详情单独拉,留言线程才可见
  const { data: req } = useQuery<EmpRequest>({
    queryKey: ["emp-request", rid],
    queryFn: () => fetchJSON(`/api/requests/${rid}`),
  });

  const decide = useMutation({
    mutationFn: ({ action, n }: { action: string; n?: string }) =>
      fetchJSON(`/api/requests/${rid}/decide`, {
        method: "POST",
        body: JSON.stringify({ action, note: n || null }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["emp-requests"] });
      qc.invalidateQueries({ queryKey: ["emp-request", rid] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });
  const comment = useMutation({
    mutationFn: (body: string) =>
      fetchJSON(`/api/requests/${rid}/comments`, { method: "POST", body: JSON.stringify({ body }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["emp-requests"] });
      qc.invalidateQueries({ queryKey: ["emp-request", rid] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "留言失败"),
  });

  if (!req) return null;

  const st = STATUS_LABEL[req.status] ?? STATUS_LABEL.pending;

  return (
    <Modal
      open
      onClose={onClose}
      title={req.subject}
      description={`${req.employee_name ?? ""} · ${KIND_LABEL[req.kind] ?? req.kind} · ${formatRelative(req.created_at)}`}
      width={560}
      footer={
        req.status === "pending" ? (
          <>
            <Button variant="secondary" onClick={onClose}>关闭</Button>
            <Button
              variant="secondary"
              onClick={() => decide.mutate({ action: "rejected", n: note })}
              disabled={decide.isPending}
            >
              <XCircle className="w-4 h-4" />驳回
            </Button>
            <Button onClick={() => decide.mutate({ action: "approved", n: note })} disabled={decide.isPending}>
              <CheckCircle2 className="w-4 h-4" />批准
            </Button>
          </>
        ) : (
          <Button onClick={onClose}>关闭</Button>
        )
      }
    >
      <div className="flex flex-col gap-4 text-[13px] font-body">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
          {req.decision_note && <span className="text-[var(--color-text-secondary)]">批注:{req.decision_note}</span>}
        </div>
        {req.detail && <p className="text-[var(--color-text-primary)] whitespace-pre-wrap">{req.detail}</p>}
        {(req.comments ?? []).length > 0 && (
          <div className="flex flex-col gap-2 border-t border-[var(--color-border-row)] pt-3">
            {req.comments.map((cm) => (
              <div key={cm.id} className="text-[12px]">
                <span className="text-[var(--color-text-tertiary)]">{cm.author_email ?? "?"} · {formatRelative(cm.created_at)}</span>
                <p className="text-[var(--color-text-primary)]">{cm.body}</p>
              </div>
            ))}
          </div>
        )}
        {req.status === "pending" && (
          <Field label="审批意见(可选)">
            <Input value={note} onChange={(e) => setNote(e.target.value)} maxLength={500} placeholder="随决定一起通知员工" />
          </Field>
        )}
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
            <Input name="cbody" placeholder="留言…" maxLength={2000} />
            <Button type="submit" variant="secondary" size="sm" disabled={comment.isPending}>
              <MessageSquare className="w-3.5 h-3.5" />
            </Button>
          </form>
        )}
        {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
      </div>
    </Modal>
  );
}
