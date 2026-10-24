import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { History, Plus, Trash2, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { PageError, Field } from "./hris/Employees";

type Cond = { field: string; op: string; value: string };
type Recip = { kind: string; value: string };

type Rule = {
  id: string;
  name: string;
  entity_type: string;
  entity_label: string;
  trigger: string;
  watch_fields: string[] | null;
  conditions: Cond[];
  action: "notify" | "email";
  recipients: { kind: string; value?: string }[];
  title_tpl: string;
  body_tpl: string | null;
  is_active: boolean;
  last_fired_at: string | null;
  created_at: string;
};

type Run = {
  id: string;
  trigger: string;
  action: string;
  status: string;
  entity_type: string;
  entity_id: string;
  payload: { title?: string; body?: string; to?: string[] } | null;
  error: string | null;
  created_at: string;
};

type Schema = {
  entities: Record<
    string,
    { label: string; fields: string[]; recipient_kinds: string[] }
  >;
  ops: string[];
  actions: string[];
  triggers: string[];
};

const OP_LABEL: Record<string, string> = {
  eq: "=",
  ne: "≠",
  gt: ">",
  lt: "<",
  ge: "≥",
  le: "≤",
  contains: "包含",
  in: "属于",
  exists: "存在",
};
const RECIP_LABEL: Record<string, string> = {
  actor: "操作者本人",
  role: "角色全员",
  user: "指定用户",
  record_owner: "记录负责人",
  record_manager: "记录员工的主管",
};
const ROLE_OPTIONS = [
  { value: "admin", label: "admin" },
  { value: "hr", label: "hr" },
  { value: "hiring_manager", label: "hiring_manager" },
  { value: "interviewer", label: "interviewer" },
  { value: "viewer", label: "viewer" },
];

const RUN_STATUS: Record<string, string> = {
  ok: "已通知",
  skipped_no_recipients: "无收件人跳过",
  queued_email: "邮件排队中",
  sent: "邮件已发",
  email_failed: "邮件失败",
  error: "执行失败",
};

export function Automations() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [err, setErr] = useState("");
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState<Rule | null>(null);
  const [runsOf, setRunsOf] = useState<Rule | null>(null);

  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<Rule>({ key: ["automations"], url: "/api/automations" });
  const { data: schema } = useQuery<Schema>({
    queryKey: ["automation-schema"],
    queryFn: () => fetchJSON("/api/automations/schema"),
  });
  const items = data?.items ?? [];

  const toggle = useMutation({
    mutationFn: (r: Rule) =>
      fetchJSON(`/api/automations/${r.id}`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: !r.is_active }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["automations"] }),
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });
  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/automations/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["automations"] }),
    onError: (e) => setErr(e instanceof ApiError ? e.message : "删除失败"),
  });

  async function onDelete(r: Rule) {
    const ok = await confirm({
      title: "删除自动化规则?",
      description: `「${r.name}」删除后不再触发,历史执行记录一并删除。`,
      confirmLabel: "删除",
      tone: "danger",
    });
    if (ok) del.mutate(r.id);
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            自动化规则
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            业务事件发生瞬间触发站内通知或邮件 — 例如「请假被批准后通知员工本人」。
          </p>
        </div>
        <Button
          onClick={() => {
            setEditing(null);
            setEditOpen(true);
          }}
        >
          <Plus className="w-4 h-4" />
          新建规则
        </Button>
      </div>

      {err && (
        <span className="text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </span>
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
            icon={Zap}
            title="还没有自动化规则"
            description="新建一条规则,业务事件发生时就自动通知相关人。"
            action={<Button onClick={() => setEditOpen(true)}>新建规则</Button>}
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
                <button
                  type="button"
                  onClick={() => toggle.mutate(r)}
                  title={r.is_active ? "点击停用" : "点击启用"}
                  className={`w-9 h-5 rounded-full relative transition-colors shrink-0 ${
                    r.is_active
                      ? "bg-[var(--color-success)]"
                      : "bg-[var(--color-border-subtle)]"
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
                      r.is_active ? "left-[18px]" : "left-0.5"
                    }`}
                  />
                </button>
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {r.name}
                </span>
                <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-info-soft)] text-[var(--color-info)] font-body">
                  {r.entity_label} · {r.trigger}
                </span>
                <span className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] font-body">
                  {r.action === "email" ? "邮件" : "站内通知"}
                </span>
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {r.last_fired_at
                    ? `上次触发 ${formatRelative(r.last_fired_at)}`
                    : "未触发过"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setRunsOf(r)}
                  title="执行记录"
                >
                  <History className="w-4 h-4" />
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setEditing(r);
                    setEditOpen(true);
                  }}
                >
                  编辑
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDelete(r)}
                  disabled={del.isPending}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
              <div className="mt-1.5 text-[12px] text-[var(--color-text-secondary)] font-body pl-12">
                {r.conditions.length > 0 &&
                  `条件 ${r.conditions
                    .map((c) => `${c.field} ${OP_LABEL[c.op] ?? c.op} ${c.value ?? ""}`)
                    .join(" 且 ")} · `}
                收件人{" "}
                {r.recipients
                  .map((x) =>
                    x.kind === "role"
                      ? `角色:${x.value}`
                      : x.kind === "user"
                        ? `用户:${x.value}`
                        : RECIP_LABEL[x.kind] ?? x.kind,
                  )
                  .join("、") || "—"}
              </div>
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

      <RuleModal
        open={editOpen}
        rule={editing}
        schema={schema}
        onClose={() => setEditOpen(false)}
      />
      <RunsModal rule={runsOf} onClose={() => setRunsOf(null)} />
    </main>
  );
}

function RuleModal({
  open,
  rule,
  schema,
  onClose,
}: {
  open: boolean;
  rule: Rule | null;
  schema: Schema | undefined;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [conds, setConds] = useState<Cond[]>([]);
  const [recips, setRecips] = useState<Recip[]>([{ kind: "role", value: "hr" }]);
  const [entity, setEntity] = useState("leave_application");
  const [trigger, setTrigger] = useState("approve");
  const [action, setAction] = useState("notify");

  // 打开时初始化(用 key 重置整个 Modal 也行,这里显式 init)
  const [inited, setInited] = useState<string>("");
  const initKey = `${open}|${rule?.id ?? "new"}`;
  if (open && initKey !== inited) {
    setInited(initKey);
    setErr("");
    setEntity(rule?.entity_type ?? "leave_application");
    setTrigger(rule?.trigger ?? "approve");
    setAction(rule?.action ?? "notify");
    setConds(
      (rule?.conditions ?? []).map((c) => ({
        field: c.field,
        op: c.op,
        value: c.value == null ? "" : String(c.value),
      })),
    );
    setRecips(
      (rule?.recipients ?? []).length > 0
        ? (rule!.recipients as Recip[])
        : [{ kind: "role", value: "hr" }],
    );
  }

  const entSpec = schema?.entities[entity];
  const fieldOptions = (entSpec?.fields ?? []).map((f) => ({
    value: f,
    label: f,
  }));
  const recipKinds = entSpec?.recipient_kinds ?? Object.keys(RECIP_LABEL);

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      rule
        ? fetchJSON(`/api/automations/${rule.id}`, {
            method: "PATCH",
            body: JSON.stringify(body),
          })
        : fetchJSON("/api/automations", {
            method: "POST",
            body: JSON.stringify(body),
          }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["automations"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: React.FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const name = String(fd.get("name") || "").trim();
    const title = String(fd.get("title_tpl") || "").trim();
    const watch = String(fd.get("watch_fields") || "").trim();
    if (!name || !title) {
      setErr("规则名与通知标题必填");
      return;
    }
    mut.mutate({
      name,
      ...(rule ? {} : { entity_type: entity }),
      trigger,
      watch_fields: watch
        ? watch.split(/[,\s]+/).filter(Boolean)
        : null,
      conditions: conds
        .filter((c) => c.field && c.op)
        .map((c) => ({ field: c.field, op: c.op, value: c.value })),
      action,
      recipients: recips.filter((r) => r.kind),
      title_tpl: title,
      body_tpl: String(fd.get("body_tpl") || "").trim() || null,
      ...(rule ? {} : { is_active: true }),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={rule ? `编辑规则:${rule.name}` : "新建自动化规则"}
      description="事件 → 条件 → 收件人 → 动作。条件留空 = 事件一发生就触发。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="rule-form" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存规则"}
          </Button>
        </>
      }
    >
      <form
        id="rule-form"
        onSubmit={submit}
        className="flex flex-col gap-4 max-h-[62vh] overflow-y-auto pr-1"
      >
        <div className="grid grid-cols-2 gap-4">
          <Field label="规则名" required>
            <Input name="name" defaultValue={rule?.name} placeholder="例如:请假批准通知员工" />
          </Field>
          <Field label="动作">
            <Select
              value={action}
              onChange={setAction}
              options={[
                { value: "notify", label: "站内通知" },
                { value: "email", label: "邮件" },
              ]}
            />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="实体" required>
            <Select
              value={entity}
              onChange={setEntity}
              disabled={!!rule}
              options={Object.entries(schema?.entities ?? {}).map(
                ([k, v]) => ({ value: k, label: v.label }),
              )}
            />
          </Field>
          <Field label="触发动作" required>
            <Select
              value={trigger}
              onChange={setTrigger}
              options={(schema?.triggers ?? ["create", "update", "delete"]).map(
                (t: string) => ({
                  value: t,
                  label: { create: "创建", update: "更新", delete: "删除" }[t] ?? t,
                }),
              )}
            />
          </Field>
        </div>
        {trigger === "update" && (
          <Field label="仅当这些字段变化(逗号分隔,留空=任意变更)">
            <Input
              name="watch_fields"
              defaultValue={(rule?.watch_fields ?? []).join(",")}
              placeholder="status, stage"
            />
          </Field>
        )}

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
              触发条件(全部满足)
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() =>
                setConds((p) => [...p, { field: "status", op: "eq", value: "" }])
              }
            >
              <Plus className="w-3.5 h-3.5" />
              加条件
            </Button>
          </div>
          {conds.map((c, i) => (
            <div key={i} className="flex items-center gap-2">
              <Select
                value={c.field}
                onChange={(v) =>
                  setConds((p) => p.map((x, j) => (j === i ? { ...x, field: v } : x)))
                }
                options={fieldOptions}
                className="w-[170px]"
              />
              <Select
                value={c.op}
                onChange={(v) =>
                  setConds((p) => p.map((x, j) => (j === i ? { ...x, op: v } : x)))
                }
                options={(schema?.ops ?? []).map((o) => ({
                  value: o,
                  label: OP_LABEL[o] ?? o,
                }))}
                className="w-[110px]"
              />
              <Input
                value={c.value}
                onChange={(e) =>
                  setConds((p) =>
                    p.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)),
                  )
                }
                placeholder="值(in 算子可逗号分隔)"
                className="flex-1 h-9 text-[13px]"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setConds((p) => p.filter((_, j) => j !== i))}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
              收件人
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setRecips((p) => [...p, { kind: "actor", value: "" }])}
            >
              <Plus className="w-3.5 h-3.5" />
              加收件人
            </Button>
          </div>
          {recips.map((r, i) => (
            <div key={i} className="flex items-center gap-2">
              <Select
                value={r.kind}
                onChange={(v) =>
                  setRecips((p) => p.map((x, j) => (j === i ? { ...x, kind: v, value: "" } : x)))
                }
                options={recipKinds.map((k) => ({
                  value: k,
                  label: RECIP_LABEL[k] ?? k,
                }))}
                className="w-[170px]"
              />
              {r.kind === "role" && (
                <Select
                  value={r.value || "hr"}
                  onChange={(v) =>
                    setRecips((p) => p.map((x, j) => (j === i ? { ...x, value: v } : x)))
                  }
                  options={ROLE_OPTIONS}
                  className="flex-1"
                />
              )}
              {r.kind === "user" && (
                <Input
                  value={r.value}
                  onChange={(e) =>
                    setRecips((p) =>
                      p.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)),
                    )
                  }
                  placeholder="用户 ID"
                  className="flex-1 h-9 text-[13px] font-mono"
                />
              )}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setRecips((p) => p.filter((_, j) => j !== i))}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          ))}
        </div>

        <Field label="标题模板(可用 {{字段}} 占位)" required>
          <Input
            name="title_tpl"
            defaultValue={rule?.title_tpl}
            placeholder="例如:你的请假已{{status}}"
          />
        </Field>
        <Field label="正文模板">
          <Input
            name="body_tpl"
            defaultValue={rule?.body_tpl ?? ""}
            placeholder="例如:{{start_date}} ~ {{end_date}} 共 {{days}} 天"
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

function RunsModal({ rule, onClose }: { rule: Rule | null; onClose: () => void }) {
  const { data, isLoading } = useQuery<{ items: Run[]; total: number }>({
    queryKey: ["automation-runs", rule?.id],
    queryFn: () => fetchJSON(`/api/automations/${rule!.id}/runs?limit=30`),
    enabled: !!rule,
  });
  return (
    <Modal
      open={!!rule}
      onClose={onClose}
      title={rule ? `执行记录:${rule.name}` : ""}
      description="每次触发写一条 — ok=通知已投递;sent=邮件已发;其余为跳过/失败原因。"
      footer={<Button variant="secondary" onClick={onClose}>关闭</Button>}
    >
      <div className="flex flex-col gap-2 max-h-[60vh] overflow-y-auto">
        {isLoading ? (
          <div className="py-8 text-center text-[13px] text-[var(--color-text-tertiary)]">
            加载中…
          </div>
        ) : (data?.items ?? []).length === 0 ? (
          <div className="py-8 text-center text-[13px] text-[var(--color-text-tertiary)]">
            还没有执行记录
          </div>
        ) : (
          data!.items.map((r) => (
            <div
              key={r.id}
              className="px-4 py-3 rounded-lg border border-[var(--color-border-subtle)]"
            >
              <div className="flex items-center gap-2">
                <span className="text-[12px] font-mono text-[var(--color-text-secondary)]">
                  {r.entity_type}:{r.trigger}
                </span>
                <span
                  className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                    r.status === "ok" || r.status === "sent"
                      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                      : r.status === "queued_email"
                        ? "bg-[var(--color-info-soft)] text-[var(--color-info)]"
                        : "bg-[var(--color-warning-soft)] text-[var(--color-warning-text)]"
                  }`}
                >
                  {RUN_STATUS[r.status] ?? r.status}
                </span>
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)]">
                  {formatRelative(r.created_at)}
                </span>
              </div>
              {r.payload?.title && (
                <div className="mt-1 text-[13px] text-[var(--color-text-primary)] font-body">
                  {r.payload.title}
                </div>
              )}
              {r.error && (
                <div className="mt-1 text-[12px] text-[var(--color-danger)] font-body">
                  {r.error}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </Modal>
  );
}
