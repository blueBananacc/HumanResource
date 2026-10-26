import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ClipboardCheck,
  Plus,
  Settings2,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { formatRelative } from "@/lib/format";

type InboxItem = {
  mode: "step" | "legacy";
  request_type: string;
  request_id: string;
  instance_id?: string;
  step_idx?: number;
  step_count?: number;
  step_label?: string;
  employee_name: string;
  org_name?: string | null;
  title: string;
  created_at?: string | null;
};

type HistoryItem = {
  step_id: string;
  idx: number;
  status: string;
  note?: string | null;
  decided_at?: string | null;
  request_type?: string;
  request_id?: string;
  title: string;
};

type Policy = {
  id: string;
  name: string;
  request_type: string;
  priority: number;
  conditions: { field: string; op: string; value: unknown }[];
  steps: { resolver: string; key?: string | null; label?: string | null }[];
  active: boolean;
  condition_fields: string[];
};

const TYPE_LABEL: Record<string, string> = {
  leave: "请假",
  overtime: "加班",
  attendance_request: "补卡",
  expense_claim: "报销",
  employee_request: "申请",
};

const TYPE_OPTIONS = Object.entries(TYPE_LABEL).map(([value, label]) => ({
  value,
  label,
}));
const OP_OPTIONS = ["eq", "ne", "gt", "lt", "ge", "le", "in"].map((v) => ({
  value: v,
  label: v,
}));


export function Approvals() {
  const { user } = useAuth();
  const [tab, setTab] = useState<"inbox" | "policies">("inbox");
  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          审批中心
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          统一收件箱:多级审批链与一级审批都在这处理;策略决定单据走几级。
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant={tab === "inbox" ? "primary" : "secondary"}
          size="sm"
          onClick={() => setTab("inbox")}
        >
          <ClipboardCheck className="w-3.5 h-3.5" />
          待审批
        </Button>
        <Button
          variant={tab === "policies" ? "primary" : "secondary"}
          size="sm"
          onClick={() => setTab("policies")}
        >
          <Settings2 className="w-3.5 h-3.5" />
          审批策略
        </Button>
        <div className="flex-1" />
        <DelegateWidget />
      </div>
      {tab === "inbox" ? <InboxTab /> : <PoliciesTab isAdmin={user?.role === "admin"} />}
    </main>
  );
}

function DelegateWidget() {
  const qc = useQueryClient();
  const [email, setEmail] = useState("");
  const [err, setErr] = useState("");
  const { data } = useQuery<{ delegate_user_id: string | null; delegate_email: string | null }>({
    queryKey: ["my-delegate"],
    queryFn: () => fetchJSON("/api/approvals/delegate"),
  });
  const save = useMutation({
    mutationFn: (v: { delegate_email?: string | null; delegate_user_id?: string | null }) =>
      fetchJSON("/api/approvals/delegate", {
        method: "PATCH",
        body: JSON.stringify(v),
      }),
    onSuccess: () => {
      setErr("");
      setEmail("");
      qc.invalidateQueries({ queryKey: ["my-delegate"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "设置失败"),
  });
  return (
    <div className="flex items-center gap-2">
      <UserRound className="w-4 h-4 text-[var(--color-text-tertiary)]" />
      <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
        代理人:
        {data?.delegate_email ? (
          <span className="text-[var(--color-text-primary)] font-medium">
            {data.delegate_email}
          </span>
        ) : (
          "未设置"
        )}
      </span>
      <Input
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="代理人邮箱"
        className="w-[180px] h-8 text-[13px]"
      />
      <Button
        size="sm"
        variant="secondary"
        disabled={save.isPending || !email.trim()}
        onClick={() => save.mutate({ delegate_email: email.trim() })}
      >
        设置
      </Button>
      {data?.delegate_user_id && (
        <Button
          size="sm"
          variant="ghost"
          disabled={save.isPending}
          onClick={() => save.mutate({ delegate_user_id: null })}
        >
          取消
        </Button>
      )}
      {err && (
        <span className="text-[12px] text-[var(--color-danger)] font-body">{err}</span>
      )}
    </div>
  );
}

function InboxTab() {
  const qc = useQueryClient();
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [err, setErr] = useState("");

  const { data, isLoading, isError, error } = useQuery<{ items: InboxItem[] }>({
    queryKey: ["approval-inbox"],
    queryFn: () => fetchJSON("/api/approvals/inbox"),
    refetchInterval: 30_000,
  });
  const items = data?.items ?? [];

  const { data: hist } = useQuery<{ items: HistoryItem[] }>({
    queryKey: ["approval-history"],
    queryFn: () => fetchJSON("/api/approvals/history"),
  });

  const decide = useMutation({
    mutationFn: (v: { item: InboxItem; action: "approved" | "rejected" }) =>
      fetchJSON("/api/approvals/decide", {
        method: "POST",
        body: JSON.stringify({
          request_type: v.item.request_type,
          request_id: v.item.request_id,
          action: v.action,
          note: notes[v.item.request_id] || null,
        }),
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["approval-inbox"] });
      qc.invalidateQueries({ queryKey: ["approval-history"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  return (
    <>
      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>
      )}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : isError ? (
          <div className="p-12 text-center text-sm text-[var(--color-danger)] font-body">
            {(error as Error).message}
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={ClipboardCheck}
            title="没有待审批"
            description="多级链当前步和未配策略的一级审批都会出现在这里。"
          />
        ) : (
          items.map((it, i) => (
            <div
              key={`${it.request_type}:${it.request_id}`}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-info-soft)] text-[var(--color-info)] font-body">
                  {TYPE_LABEL[it.request_type] ?? it.request_type}
                </span>
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {it.employee_name}
                </span>
                {it.org_name && (
                  <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                    {it.org_name}
                  </span>
                )}
                {it.mode === "step" ? (
                  <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-warning-soft)] text-[var(--color-warning)] font-body">
                    第 {(it.step_idx ?? 0) + 1}/{it.step_count} 级 · {it.step_label}
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] font-body">
                    一级审批
                  </span>
                )}
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {it.created_at ? formatRelative(it.created_at) : ""}
                </span>
              </div>
              <div className="mt-1.5 text-[13px] text-[var(--color-text-primary)] font-body">
                {it.title}
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Input
                  value={notes[it.request_id] ?? ""}
                  onChange={(e) =>
                    setNotes((p) => ({ ...p, [it.request_id]: e.target.value }))
                  }
                  placeholder="审批意见(可选)"
                  className="w-[320px] h-8 text-[13px]"
                />
                <Button
                  size="sm"
                  disabled={decide.isPending}
                  onClick={() => decide.mutate({ item: it, action: "approved" })}
                >
                  <Check className="w-3.5 h-3.5" />
                  批准
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={decide.isPending}
                  onClick={() => decide.mutate({ item: it, action: "rejected" })}
                >
                  <X className="w-3.5 h-3.5" />
                  驳回
                </Button>
              </div>
            </div>
          ))
        )}
      </div>

      {hist && hist.items.length > 0 && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <div className="px-6 py-3 border-b border-[var(--color-border-row)] text-sm font-medium text-[var(--color-text-primary)] font-body">
            我最近处理的
          </div>
          {hist.items.slice(0, 20).map((h) => (
            <div
              key={h.step_id}
              className="px-6 py-3 flex items-center gap-3 border-b border-[var(--color-border-row)] last:border-b-0"
            >
              <span
                className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                  h.status === "approved"
                    ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                    : h.status === "rejected"
                      ? "bg-[var(--color-danger-soft)] text-[var(--color-danger)]"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                }`}
              >
                {h.status === "approved"
                  ? "已批准"
                  : h.status === "rejected"
                    ? "已驳回"
                    : h.status}
              </span>
              <span className="text-[13px] text-[var(--color-text-primary)] font-body">
                {h.title}
              </span>
              {h.note && (
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {h.note}
                </span>
              )}
              <div className="flex-1" />
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                {h.decided_at ? formatRelative(h.decided_at) : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function PoliciesTab({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [form, setForm] = useState({
    name: "",
    request_type: "leave",
    priority: 0,
    condField: "",
    condOp: "ge",
    condValue: "",
    stepsText: "manager|直属上级\nrole:hr|HR复核",
  });
  const { data } = useQuery<{
    items: Policy[];
    request_types: string[];
    condition_fields: Record<string, string[]>;
  }>({
    queryKey: ["approval-policies"],
    queryFn: () => fetchJSON("/api/approvals/policies"),
  });
  const policies = data?.items ?? [];

  const create = useMutation({
    mutationFn: () => {
      const conds = form.condField
        ? [{ field: form.condField, op: form.condOp, value: _parseVal(form.condValue) }]
        : [];
      const steps = form.stepsText
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => {
          const [spec, label] = l.split("|").map((s) => s.trim());
          const [resolver, key] = spec.split(":").map((s) => s.trim());
          return { resolver, key: key || undefined, label: label || undefined };
        });
      return fetchJSON("/api/approvals/policies", {
        method: "POST",
        body: JSON.stringify({
          name: form.name.trim(),
          request_type: form.request_type,
          priority: form.priority,
          conditions: conds,
          steps,
        }),
      });
    },
    onSuccess: () => {
      setErr("");
      setForm((f) => ({ ...f, name: "" }));
      qc.invalidateQueries({ queryKey: ["approval-policies"] });
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/approvals/policies/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["approval-policies"] }),
    onError: (e) => setErr(e instanceof ApiError ? e.message : "删除失败"),
  });

  const fields = data?.condition_fields?.[form.request_type] ?? [];

  return (
    <>
      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>
      )}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {policies.length === 0 ? (
          <Empty
            icon={Settings2}
            title="还没有审批策略"
            description="配置后,命中的单据走多级审批链;不配置则保持一级审批。"
          />
        ) : (
          policies.map((p, i) => (
            <div
              key={p.id}
              className={`px-6 py-4 ${
                i < policies.length - 1 ? "border-b border-[var(--color-border-row)]" : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {p.name}
                </span>
                <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-info-soft)] text-[var(--color-info)] font-body">
                  {TYPE_LABEL[p.request_type] ?? p.request_type}
                </span>
                <span
                  className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                    p.active
                      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                  }`}
                >
                  {p.active ? "启用" : "停用"}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-mono">
                  优先级 {p.priority}
                </span>
                <div className="flex-1" />
                {isAdmin && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => del.mutate(p.id)}
                    disabled={del.isPending}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                )}
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[12px] font-body">
                {p.conditions.length > 0 ? (
                  p.conditions.map((c, j) => (
                    <span
                      key={j}
                      className="px-2 py-0.5 rounded bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] font-mono"
                    >
                      {c.field} {c.op} {String(c.value)}
                    </span>
                  ))
                ) : (
                  <span className="text-[var(--color-text-tertiary)]">无条件(默认)</span>
                )}
                <span className="text-[var(--color-text-tertiary)]">→</span>
                {p.steps.map((s, j) => (
                  <span key={j} className="flex items-center gap-1.5">
                    {j > 0 && <span className="text-[var(--color-text-tertiary)]">→</span>}
                    <span className="px-2 py-0.5 rounded bg-[var(--color-warning-soft)] text-[var(--color-warning)]">
                      {s.label ?? s.resolver}
                      {s.key ? `:${s.key}` : ""}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {isAdmin && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
          <div className="text-sm font-medium text-[var(--color-text-primary)] font-body">
            新建策略
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="策略名(如:请假超3天两级审批)"
              className="w-[240px] h-8 text-[13px]"
            />
            <Select
              value={form.request_type}
              onChange={(v) => setForm((f) => ({ ...f, request_type: v, condField: "" }))}
              options={TYPE_OPTIONS}
              className="w-[120px]"
            />
            <Input
              type="number"
              value={String(form.priority)}
              onChange={(e) =>
                setForm((f) => ({ ...f, priority: Number(e.target.value) || 0 }))
              }
              placeholder="优先级"
              className="w-[90px] h-8 text-[13px]"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
              条件(可空=全部命中):
            </span>
            <Select
              value={form.condField}
              onChange={(v) => setForm((f) => ({ ...f, condField: v }))}
              options={[
                { value: "", label: "无条件" },
                ...fields.map((x) => ({ value: x, label: x })),
              ]}
              className="w-[150px]"
            />
            <Select
              value={form.condOp}
              onChange={(v) => setForm((f) => ({ ...f, condOp: v }))}
              options={OP_OPTIONS}
              className="w-[90px]"
            />
            <Input
              value={form.condValue}
              onChange={(e) =>
                setForm((f) => ({ ...f, condValue: e.target.value }))
              }
              placeholder="值(如 3)"
              className="w-[110px] h-8 text-[13px]"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
              步骤(每行一条:manager | role:角色 | user:用户ID,| 后接显示名):
            </span>
            <textarea
              value={form.stepsText}
              onChange={(e) =>
                setForm((f) => ({ ...f, stepsText: e.target.value }))
              }
              rows={3}
              className="w-full max-w-[480px] rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-[13px] font-mono text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-primary)]"
            />
          </div>
          <div>
            <Button
              size="sm"
              disabled={create.isPending || !form.name.trim() || !form.stepsText.trim()}
              onClick={() => create.mutate()}
            >
              <Plus className="w-3.5 h-3.5" />
              保存策略
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

function _parseVal(v: string): unknown {
  const t = v.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : t;
}
