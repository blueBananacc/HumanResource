import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { History, LifeBuoy, Lock, MessageSquare, UserCheck } from "lucide-react";
import { FavoriteStar } from "@/components/records/FavoriteStar";
import { HistoryDrawer } from "@/components/records/HistoryDrawer";
import { SavedFilters } from "@/components/records/SavedFilters";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { PageError } from "./Employees";

type CaseUpdate = {
  id: string;
  author_user_id: string | null;
  author_email: string | null;
  body: string;
  is_internal: boolean;
  created_at: string;
};

type HrCase = {
  id: string;
  case_no: number;
  subject: string;
  kind: string;
  priority: string;
  status: string;
  description: string | null;
  requester_name: string | null;
  assignee_user_id: string | null;
  assignee_email: string | null;
  resolution: string | null;
  resolved_at: string | null;
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

const PRIO_LABEL: Record<string, { label: string; cls: string }> = {
  low: { label: "低", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
  normal: { label: "普通", cls: "bg-blue-50 text-blue-700" },
  high: { label: "高", cls: "bg-amber-50 text-amber-700" },
  urgent: { label: "紧急", cls: "bg-red-50 text-red-700" },
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  new: { label: "新建", cls: "bg-blue-50 text-blue-700" },
  assigned: { label: "已分配", cls: "bg-violet-50 text-violet-700" },
  in_progress: { label: "处理中", cls: "bg-amber-50 text-amber-700" },
  resolved: { label: "已解决", cls: "bg-green-50 text-green-700" },
  closed: { label: "已关闭", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
};

export function HrCases() {
  const [statusFilter, setStatusFilter] = useState("new");
  const [mineOnly, setMineOnly] = useState(false);
  const [viewing, setViewing] = useState<string | null>(null);
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<HrCase>({
      key: ["hr-cases"],
      url: "/api/cases",
      params: { status_filter: statusFilter, mine: mineOnly ? "true" : undefined },
    });
  const items = data?.items ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">HR 工单</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            员工诉求通道 — 分类/优先级/指派/处理记录,员工端自助跟踪。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[13px] text-[var(--color-text-secondary)] font-body whitespace-nowrap">
            <input type="checkbox" checked={mineOnly} onChange={(e) => setMineOnly(e.target.checked)} />
            只看我的
          </label>
          <div className="w-[140px]">
            <Select
              value={statusFilter}
              onChange={setStatusFilter}
              options={[
                { value: "new", label: "新建" },
                { value: "assigned", label: "已分配" },
                { value: "in_progress", label: "处理中" },
                { value: "resolved", label: "已解决" },
                { value: "closed", label: "已关闭" },
                { value: "", label: "全部" },
              ]}
            />
          </div>
          <SavedFilters
            page="cases"
            currentParams={() => ({
              status_filter: statusFilter || undefined,
              mine: mineOnly || undefined,
            })}
            onApply={(p) => {
              setStatusFilter(typeof p.status_filter === "string" ? p.status_filter : "");
              setMineOnly(Boolean(p.mine));
            }}
          />
        </div>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty icon={LifeBuoy} title="没有工单" description="当前筛选下没有 HR 工单。" />
        ) : (
          items.map((c, i) => {
            const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.new;
            const pr = PRIO_LABEL[c.priority] ?? PRIO_LABEL.normal;
            return (
              <div
                key={c.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(c.id)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[12px] text-[var(--color-text-tertiary)]">HR-{c.case_no}</span>
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{c.subject}</span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${pr.cls}`}>{pr.label}</span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <div className="flex-1" />
                  <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                    {c.requester_name} {c.assignee_email ? `→ ${c.assignee_email}` : ""}
                  </span>
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(c.created_at)}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {viewing && <CaseDetail cid={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

function CaseDetail({ cid, onClose }: { cid: string; onClose: () => void }) {
  const qc = useQueryClient();
  const [resolution, setResolution] = useState("");
  const [internal, setInternal] = useState(false);
  const [err, setErr] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);

  const { data: c } = useQuery<HrCase>({
    queryKey: ["hr-case", cid],
    queryFn: () => fetchJSON(`/api/cases/${cid}`),
  });

  const act = useMutation({
    mutationFn: ({ path, body }: { path: string; body?: unknown }) =>
      fetchJSON(`/api/cases/${cid}/${path}`, { method: "POST", body: JSON.stringify(body ?? {}) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hr-cases"] });
      qc.invalidateQueries({ queryKey: ["hr-case", cid] });
      setErr("");
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  const { data: me } = useQuery<{ id: string }>({
    queryKey: ["me-profile-mini"],
    queryFn: () => fetchJSON("/api/auth/me"),
  });

  if (!c) return null;
  const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.new;

  return (
    <Modal
      open
      onClose={onClose}
      title={`HR-${c.case_no} ${c.subject}`}
      description={`${c.requester_name ?? ""} · ${KIND_LABEL[c.kind] ?? c.kind} · ${formatRelative(c.created_at)}`}
      width={600}
      footer={<Button onClick={onClose}>关闭</Button>}
    >
      <div className="flex flex-col gap-4 text-[13px] font-body">
        <HistoryDrawer entityId={c.id} open={historyOpen} onClose={() => setHistoryOpen(false)} />
        <div className="flex items-center gap-2">
          <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
          <FavoriteStar entity="cases" entityId={c.id} />
          <Button size="sm" variant="secondary" onClick={() => setHistoryOpen(true)}>
            <History className="w-3.5 h-3.5" />
            变更记录
          </Button>
          <span className="text-[var(--color-text-secondary)]">
            受理人:{c.assignee_email ?? "未分配"}
          </span>
          {!c.assignee_user_id && me && (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => act.mutate({ path: "assign", body: { assignee_user_id: me.id } })}
            >
              <UserCheck className="w-3.5 h-3.5" />我来受理
            </Button>
          )}
        </div>
        {c.description && <p className="text-[var(--color-text-primary)] whitespace-pre-wrap">{c.description}</p>}
        {c.resolution && (
          <div className="rounded-lg bg-green-50 px-3 py-2 text-green-800">解决方案:{c.resolution}</div>
        )}

        {(c.updates ?? []).length > 0 && (
          <div className="flex flex-col gap-2 border-t border-[var(--color-border-row)] pt-3">
            {c.updates.map((u) => (
              <div key={u.id} className="text-[12px]">
                <span className="text-[var(--color-text-tertiary)]">
                  {u.author_email ?? "?"} · {formatRelative(u.created_at)}
                  {u.is_internal && (
                    <span className="ml-1 inline-flex items-center gap-0.5 text-amber-600">
                      <Lock className="w-3 h-3" />内部
                    </span>
                  )}
                </span>
                <p className="text-[var(--color-text-primary)]">{u.body}</p>
              </div>
            ))}
          </div>
        )}

        {c.status !== "closed" && (
          <>
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                const fd = new FormData(e.currentTarget);
                const body = String(fd.get("ubody") || "").trim();
                if (body) act.mutate({ path: "updates", body: { body, is_internal: internal } });
                e.currentTarget.reset();
              }}
            >
              <Input name="ubody" placeholder={internal ? "内部备注(员工不可见)…" : "回复(员工可见)…"} maxLength={3000} />
              <label className="flex items-center gap-1 text-[12px] text-[var(--color-text-secondary)] whitespace-nowrap">
                <input type="checkbox" checked={internal} onChange={(e) => setInternal(e.target.checked)} />
                内部
              </label>
              <Button type="submit" variant="secondary" size="sm" disabled={act.isPending}>
                <MessageSquare className="w-3.5 h-3.5" />
              </Button>
            </form>
            {c.status !== "resolved" && (
              <div className="flex items-center gap-2">
                <Input
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  placeholder="解决方案(解决工单时必填)"
                  maxLength={3000}
                />
                <Button
                  size="sm"
                  disabled={act.isPending || !resolution.trim()}
                  onClick={() =>
                    act.mutate({ path: "status", body: { status: "resolved", resolution: resolution.trim() } })
                  }
                >
                  标记解决
                </Button>
              </div>
            )}
            {c.status === "resolved" && (
              <Button
                variant="secondary"
                size="sm"
                className="self-start"
                onClick={() => act.mutate({ path: "status", body: { status: "closed" } })}
                disabled={act.isPending}
              >
                关闭工单
              </Button>
            )}
          </>
        )}
        {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
      </div>
    </Modal>
  );
}
