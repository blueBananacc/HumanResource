import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FileText,
  Mail,
  Plus,
  ToggleLeft,
  ToggleRight,
  Trash2,
  UserPlus,
  Users,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON, ApiError } from "@/lib/api";
import { Field } from "@/pages/hris/Employees";

/**
 * Teamtailor 阶段触发器 —— 岗位详情页上的自动化挂载面。
 *
 * 底层复用 AutomationRule 引擎:每条触发器 = entity_type=job_application、
 * conditions=[stage eq X, job_id eq 本岗位] 的规则;applied 阶段用 create
 * 触发(建卡即进入),其余阶段用 transition。规则同时出现在全局
 * /settings/automations 里(审计/启停/执行记录)。
 */

const STAGE_ORDER = [
  "applied",
  "screening",
  "interview",
  "offer",
  "hired",
  "rejected",
] as const;
const STAGE_LABEL: Record<string, string> = {
  applied: "已投递",
  screening: "筛选中",
  interview: "面试中",
  offer: "Offer",
  hired: "已录用",
  rejected: "已淘汰",
};
const STAGE_DOT: Record<string, string> = {
  applied: "bg-[var(--color-info)]",
  screening: "bg-[var(--color-warning)]",
  interview: "bg-[var(--color-primary,#6366f1)]",
  offer: "bg-[#a855f7]",
  hired: "bg-[var(--color-success)]",
  rejected: "bg-[var(--color-danger)]",
};

type ActionKind =
  | "email_candidate"
  | "send_survey"
  | "add_to_group"
  | "notify"
  | "create_case";

const ACTION_META: Record<
  ActionKind,
  { label: string; hint: string; icon: typeof Mail }
> = {
  email_candidate: {
    label: "给候选人发邮件",
    hint: "进入阶段时自动发信(面试邀请/感谢信)",
    icon: Mail,
  },
  send_survey: {
    label: "发送问卷",
    hint: "自动建一次性填写链接并邮件候选人(体验调查/NPS)",
    icon: FileText,
  },
  add_to_group: {
    label: "加入人才分组",
    hint: "候选人自动进入指定人才池分组",
    icon: Users,
  },
  notify: {
    label: "站内通知",
    hint: "通知操作者/角色/申请负责人",
    icon: Zap,
  },
  create_case: {
    label: "创建 HR 工单",
    hint: "转入人工处理通道(背调/Offer 审批等)",
    icon: UserPlus,
  },
};
const ACTION_ORDER: ActionKind[] = [
  "email_candidate",
  "send_survey",
  "add_to_group",
  "notify",
  "create_case",
];

const RECIP_OPTIONS = [
  { value: "actor", label: "操作者(执行迁移的人)" },
  { value: "role:hr", label: "全部 HR" },
  { value: "role:admin", label: "全部管理员" },
  { value: "record_owner", label: "申请负责人" },
];
const PRIORITY_OPTIONS = [
  { value: "normal", label: "普通" },
  { value: "low", label: "低" },
  { value: "high", label: "高" },
  { value: "urgent", label: "紧急" },
];
const ASSIGNEE_OPTIONS = [
  { value: "", label: "不指派(进入待分配)" },
  { value: "hr", label: "首个 HR 用户" },
  { value: "admin", label: "首个管理员" },
];

type Cond = { field: string; op: string; value: string };
type Rule = {
  id: string;
  name: string;
  trigger: string;
  conditions: Cond[];
  action: ActionKind;
  action_config: Record<string, unknown> | null;
  recipients: { kind: string; value?: string }[];
  title_tpl: string;
  body_tpl: string | null;
  is_active: boolean;
};
type Group = { id: string; name: string };
type Survey = { id: string; title: string; status: string };

function stageOf(r: Rule): string | null {
  const c = r.conditions.find((x) => x.field === "stage");
  return c ? String(c.value) : null;
}
function jobOf(r: Rule): string | null {
  const c = r.conditions.find((x) => x.field === "job_id");
  return c ? String(c.value) : null;
}
function recipKey(r: Rule): string {
  const x = r.recipients?.[0];
  if (!x) return "actor";
  return x.kind === "role" ? `role:${x.value}` : x.kind;
}

export function JobStageTriggersCard({
  jobId,
  jobTitle,
}: {
  jobId: string;
  jobTitle: string;
}) {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [editing, setEditing] = useState<
    { stage: string; rule: Rule | null } | null
  >(null);
  const [err, setErr] = useState("");

  const { data } = useQuery({
    queryKey: ["job-stage-triggers", jobId],
    queryFn: () =>
      fetchJSON<{ items: Rule[] }>(
        "/api/automations?entity_type=job_application&limit=200",
      ),
  });

  const byStage = useMemo(() => {
    const m = new Map<string, Rule[]>();
    for (const r of data?.items ?? []) {
      const s = stageOf(r);
      if (s && jobOf(r) === jobId) {
        m.set(s, [...(m.get(s) ?? []), r]);
      }
    }
    return m;
  }, [data, jobId]);

  const toggle = useMutation({
    mutationFn: (r: Rule) =>
      fetchJSON(`/api/automations/${r.id}`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: !r.is_active }),
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["job-stage-triggers", jobId] }),
  });
  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/automations/${id}`, { method: "DELETE" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["job-stage-triggers", jobId] }),
  });

  const total = [...byStage.values()].reduce((n, xs) => n + xs.length, 0);

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-1">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-[var(--color-text-tertiary)]" />
          <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
            阶段触发器
          </h3>
          {total > 0 && (
            <span className="px-2 py-0.5 rounded-md text-[11px] font-medium font-body bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]">
              {total} 个生效中
            </span>
          )}
        </div>
        <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
          候选人进入阶段时自动执行 · 全局管理见设置 → 自动化
        </span>
      </div>

      {STAGE_ORDER.map((stage) => {
        const rules = byStage.get(stage) ?? [];
        return (
          <div
            key={stage}
            className="border-t border-[var(--color-border-subtle)] first:border-t-0 py-3 flex flex-col gap-2"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${STAGE_DOT[stage]}`}
                  aria-hidden
                />
                <span className="text-[13px] font-medium font-body text-[var(--color-text-primary)]">
                  {STAGE_LABEL[stage]}
                </span>
                {rules.length > 0 && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    ⚡ {rules.length}
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => {
                  setErr("");
                  setEditing({ stage, rule: null });
                }}
                className="inline-flex items-center gap-1 text-[12px] font-body text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                添加触发器
              </button>
            </div>
            {rules.map((r) => {
              const Icon = ACTION_META[r.action]?.icon ?? Zap;
              return (
                <div
                  key={r.id}
                  className="ml-4 flex items-center gap-3 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-subtle)]/40 px-3 py-2"
                >
                  <Icon className="w-3.5 h-3.5 text-[var(--color-text-tertiary)] shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-body text-[var(--color-text-primary)] truncate">
                      {r.name}
                    </div>
                    <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                      {ACTION_META[r.action]?.label ?? r.action}
                    </div>
                  </div>
                  <button
                    type="button"
                    title={r.is_active ? "点击停用" : "点击启用"}
                    onClick={() => toggle.mutate(r)}
                    className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors"
                  >
                    {r.is_active ? (
                      <ToggleRight className="w-5 h-5 text-[var(--color-success)]" />
                    ) : (
                      <ToggleLeft className="w-5 h-5" />
                    )}
                  </button>
                  <button
                    type="button"
                    title="编辑"
                    onClick={() => {
                      setErr("");
                      setEditing({ stage, rule: r });
                    }}
                    className="text-[12px] font-body text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
                  >
                    编辑
                  </button>
                  <button
                    type="button"
                    title="删除触发器"
                    onClick={async () => {
                      if (
                        await confirm({
                          title: "删除触发器?",
                          description: `「${r.name}」删除后不再自动执行。`,
                          tone: "danger",
                          confirmLabel: "删除",
                        })
                      ) {
                        try {
                          await del.mutateAsync(r.id);
                        } catch (e) {
                          void (e instanceof ApiError ? e.message : e);
                        }
                      }
                    }}
                    className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        );
      })}

      <TriggerModal
        key={
          editing
            ? `${editing.stage}-${editing.rule?.id ?? "new"}`
            : "closed"
        }
        editing={editing}
        jobId={jobId}
        jobTitle={jobTitle}
        err={err}
        setErr={setErr}
        onClose={() => setEditing(null)}
      />
    </div>
  );
}

function TriggerModal({
  editing,
  jobId,
  jobTitle,
  err,
  setErr,
  onClose,
}: {
  editing: { stage: string; rule: Rule | null } | null;
  jobId: string;
  jobTitle: string;
  err: string;
  setErr: (s: string) => void;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const rule = editing?.rule ?? null;
  const stage = editing?.stage ?? "interview";
  const stageLabel = STAGE_LABEL[stage] ?? stage;

  const [name, setName] = useState(rule?.name ?? "");
  const [action, setAction] = useState<ActionKind>(
    rule?.action ?? "email_candidate",
  );
  const [title, setTitle] = useState(rule?.title_tpl ?? "");
  const [body, setBody] = useState(rule?.body_tpl ?? "");
  const [groupId, setGroupId] = useState(
    String(rule?.action_config?.group_id ?? ""),
  );
  const [surveyId, setSurveyId] = useState(
    String(rule?.action_config?.survey_id ?? ""),
  );
  const [recip, setRecip] = useState(rule ? recipKey(rule) : "actor");
  const [priority, setPriority] = useState(
    String(rule?.action_config?.priority ?? "normal"),
  );
  const [assignee, setAssignee] = useState(
    String(rule?.action_config?.assignee_value ?? ""),
  );

  const { data: groups } = useQuery({
    queryKey: ["talent-groups"],
    queryFn: () => fetchJSON<Group[]>("/api/talent-groups"),
    enabled: !!editing && action === "add_to_group",
  });
  const { data: surveys } = useQuery({
    queryKey: ["surveys"],
    queryFn: () => fetchJSON<Survey[]>("/api/surveys"),
    enabled: !!editing && action === "send_survey",
  });
  const published = (surveys ?? []).filter((s) => s.status === "published");

  const save = useMutation({
    mutationFn: () => {
      const actionConfig =
        action === "add_to_group"
          ? { group_id: groupId }
          : action === "send_survey"
            ? { survey_id: surveyId }
            : action === "create_case"
              ? {
                  kind: "other",
                  priority,
                  subject_tpl: title || name,
                  ...(assignee
                    ? { assignee_kind: "role", assignee_value: assignee }
                    : {}),
                }
              : undefined;
      const recipients =
        action === "notify"
          ? recip === "record_owner"
            ? [{ kind: "record_owner" }]
            : recip === "actor"
              ? [{ kind: "actor" }]
              : [{ kind: "role", value: recip.slice(5) }]
          : [];
      const payload = {
        name: name.trim(),
        entity_type: "job_application",
        trigger: stage === "applied" ? "create" : "transition",
        conditions: [
          { field: "stage", op: "eq", value: stage },
          { field: "job_id", op: "eq", value: jobId },
        ],
        action,
        action_config: actionConfig,
        recipients,
        title_tpl: title.trim() || name.trim(),
        body_tpl: body.trim() || undefined,
      };
      return rule
        ? fetchJSON(`/api/automations/${rule.id}`, {
            method: "PATCH",
            body: JSON.stringify({
              name: payload.name,
              trigger: payload.trigger,
              conditions: payload.conditions,
              action: payload.action,
              action_config: payload.action_config ?? null,
              recipients: payload.recipients,
              title_tpl: payload.title_tpl,
              body_tpl: payload.body_tpl ?? null,
            }),
          })
        : fetchJSON("/api/automations", {
            method: "POST",
            body: JSON.stringify(payload),
          });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["job-stage-triggers", jobId] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  const needCfg =
    (action === "add_to_group" && !groupId) ||
    (action === "send_survey" && !surveyId);
  const invalid = !name.trim() || !title.trim() || needCfg;

  return (
    <Modal
      open={!!editing}
      onClose={onClose}
      title={`${rule ? "编辑" : "添加"}阶段触发器 · ${stageLabel}`}
      description={`候选人进入「${stageLabel}」阶段时自动执行 · ${jobTitle}`}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button
            onClick={() => save.mutate()}
            disabled={invalid || save.isPending}
          >
            {save.isPending ? "保存中…" : "保存触发器"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4 pb-2">
        {err && (
          <div className="text-[13px] font-body text-[var(--color-danger)] bg-[var(--color-danger-soft,var(--color-bg-subtle))] rounded-lg px-3 py-2">
            {err}
          </div>
        )}
        <div className="grid grid-cols-2 gap-4">
          <Field label="触发器名称" required>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={`${stageLabel} · ${ACTION_META[action].label}`}
            />
          </Field>
          <Field label="动作" required>
            <Select
              value={action}
              onChange={(v) => setAction(v as ActionKind)}
              options={ACTION_ORDER.map((a) => ({
                value: a,
                label: ACTION_META[a].label,
                hint: ACTION_META[a].hint,
              }))}
            />
          </Field>
        </div>

        {action === "add_to_group" && (
          <Field label="人才分组" required>
            <Select
              value={groupId}
              onChange={setGroupId}
              placeholder="— 选择分组 —"
              options={(groups ?? []).map((g) => ({
                value: g.id,
                label: g.name,
              }))}
            />
          </Field>
        )}
        {action === "send_survey" && (
          <Field label="问卷(仅已发布)" required>
            <Select
              value={surveyId}
              onChange={setSurveyId}
              placeholder="— 选择问卷 —"
              options={published.map((s) => ({ value: s.id, label: s.title }))}
            />
          </Field>
        )}
        {action === "notify" && (
          <Field label="收件人" required>
            <Select
              value={recip}
              onChange={setRecip}
              options={RECIP_OPTIONS}
            />
          </Field>
        )}
        {action === "create_case" && (
          <div className="grid grid-cols-2 gap-4">
            <Field label="优先级">
              <Select
                value={priority}
                onChange={setPriority}
                options={PRIORITY_OPTIONS}
              />
            </Field>
            <Field label="受理人">
              <Select
                value={assignee}
                onChange={setAssignee}
                options={ASSIGNEE_OPTIONS}
              />
            </Field>
          </div>
        )}

        <Field
          label={
            action === "create_case"
              ? "工单主题"
              : action === "notify"
                ? "通知标题"
                : "邮件标题"
          }
          required
        >
          <Input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder={
              action === "email_candidate" || action === "send_survey"
                ? `如:{{job_title}} 面试邀请`
                : `如:{{candidate_name}} 进入${stageLabel}`
            }
          />
        </Field>
        {action !== "add_to_group" && (
          <Field label={action === "create_case" ? "工单描述(可选)" : "正文(可选)"}>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows={3}
              className="w-full rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-sm font-body text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-text-primary)] resize-y"
              placeholder="支持 {{candidate_name}} {{job_title}} {{stage}} 占位符"
            />
          </Field>
        )}
        <p className="text-[12px] text-[var(--color-text-tertiary)] font-body leading-relaxed">
          {action === "email_candidate" &&
            "邮件将发送到候选人档案中的邮箱;标题/正文支持 {{candidate_name}} {{job_title}} 占位符。"}
          {action === "send_survey" &&
            "自动为候选人生成一次性填写链接,附在邮件正文末尾发送;未配邮箱时仅创建链接。"}
          {action === "add_to_group" &&
            "候选人自动加入所选分组(已入组则跳过,不重复添加)。"}
          {action === "notify" && "向所选收件人发送站内通知,含候选人上下文。"}
          {action === "create_case" &&
            "创建 HR 工单进入人工处理通道,受理人可选角色自动指派。"}
        </p>
      </div>
    </Modal>
  );
}
