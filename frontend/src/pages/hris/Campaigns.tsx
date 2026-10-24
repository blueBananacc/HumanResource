import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MailCheck, Plus, Send, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { fetchJSON, ApiError } from "@/lib/api";
import { AiFieldButton } from "@/components/assistant/AiFieldButton";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { Field, PageError } from "./Employees";

type Send = {
  id: string;
  candidate_id: string;
  candidate_name: string | null;
  to_email: string;
  status: string;
  error: string | null;
  sent_at: string | null;
};

type Campaign = {
  id: string;
  name: string;
  filter: { group_id?: string | null; skill_ids?: string[]; query?: string | null };
  subject: string;
  body_text: string;
  status: string;
  total: number;
  sent_count: number;
  failed_count: number;
  started_at: string | null;
  done_at: string | null;
  created_at: string;
  sends?: Send[];
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  draft: { label: "草稿", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
  sending: { label: "发送中", cls: "bg-blue-50 text-blue-700" },
  done: { label: "已完成", cls: "bg-green-50 text-green-700" },
  failed: { label: "发送失败", cls: "bg-red-50 text-red-700" },
  cancelled: { label: "已取消", cls: "bg-amber-50 text-amber-700" },
};

const SEND_LABEL: Record<string, string> = {
  queued: "待投递",
  sent: "已发送",
  failed: "失败",
  cancelled: "已取消",
  skipped_no_email: "无邮箱跳过",
};

export function Campaigns() {
  const [createOpen, setCreateOpen] = useState(false);
  const [viewing, setViewing] = useState<string | null>(null);
  const { hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.WRITE_RESUMES);
  const qc = useQueryClient();
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<Campaign>({ key: ["campaigns"], url: "/api/campaigns" });
  const items = data?.items ?? [];

  const act = useMutation({
    mutationFn: ({ id, action }: { id: string; action: string }) =>
      fetchJSON(`/api/campaigns/${id}/${action}`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["campaigns"] });
      qc.invalidateQueries({ queryKey: ["campaign"] });
    },
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">人才再激活</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            按条件圈选人才库 → 批量个性化外联;支持 {"{{candidate_name}}"} 变量。
          </p>
        </div>
        {!isError && canWrite && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4" />新建外联
          </Button>
        )}
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={MailCheck}
            title="还没有外联活动"
            description="把沉睡的人才重新激活:圈选条件 + 个性化邮件一键触达。"
            action={canWrite ? <Button onClick={() => setCreateOpen(true)}>新建外联</Button> : undefined}
          />
        ) : (
          items.map((c, i) => {
            const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.draft;
            return (
              <div
                key={c.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(c.id)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{c.name}</span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <div className="flex-1" />
                  <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                    {c.total} 人 · 已发 {c.sent_count} · 失败 {c.failed_count}
                  </span>
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(c.created_at)}</span>
                </div>
                {canWrite && (
                  <div className="mt-2 flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                    {(c.status === "draft" || c.status === "failed") && (
                      <Button size="sm" onClick={() => act.mutate({ id: c.id, action: "send" })} disabled={act.isPending}>
                        <Send className="w-3.5 h-3.5" />{c.status === "failed" ? "重试发送" : "发送"}
                      </Button>
                    )}
                    {(c.status === "draft" || c.status === "sending") && (
                      <Button variant="ghost" size="sm" onClick={() => act.mutate({ id: c.id, action: "cancel" })} disabled={act.isPending}>
                        <XCircle className="w-3.5 h-3.5" />取消
                      </Button>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {createOpen && <CampaignModal onClose={() => setCreateOpen(false)} />}
      {viewing && <CampaignDetail cid={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}

function CampaignDetail({ cid, onClose }: { cid: string; onClose: () => void }) {
  const { data: c } = useQuery<Campaign>({
    queryKey: ["campaign", cid],
    queryFn: () => fetchJSON(`/api/campaigns/${cid}`),
    refetchInterval: (q) => (q.state.data?.status === "sending" ? 3000 : false),
  });
  if (!c) return null;
  const st = STATUS_LABEL[c.status] ?? STATUS_LABEL.draft;
  return (
    <Modal
      open
      onClose={onClose}
      title={c.name}
      description={`${c.total} 人 · 已发 ${c.sent_count} · 失败 ${c.failed_count}`}
      width={640}
      footer={<Button onClick={onClose}>关闭</Button>}
    >
      <div className="flex flex-col gap-3 text-[13px] font-body">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
          {c.status === "sending" && <span className="text-[var(--color-text-tertiary)]">发送中,自动刷新…</span>}
        </div>
        <div className="rounded-lg bg-[var(--color-surface-1)] px-3 py-2">
          <div className="text-[12px] text-[var(--color-text-tertiary)]">主题</div>
          <div>{c.subject}</div>
          <div className="mt-2 text-[12px] text-[var(--color-text-tertiary)]">正文</div>
          <div className="whitespace-pre-wrap">{c.body_text}</div>
        </div>
        <div className="max-h-[280px] overflow-auto border-t border-[var(--color-border-row)] pt-2">
          {(c.sends ?? []).map((s) => (
            <div key={s.id} className="flex items-center gap-3 py-1 text-[12px]">
              <span className="w-32 truncate">{s.candidate_name}</span>
              <span className="flex-1 truncate text-[var(--color-text-secondary)]">{s.to_email || "—"}</span>
              <span className={s.status === "failed" ? "text-red-600" : "text-[var(--color-text-tertiary)]"}>
                {SEND_LABEL[s.status] ?? s.status}{s.error ? `(${s.error.slice(0, 40)})` : ""}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}

function CampaignModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const { data: groups } = useQuery<{ id: string; name: string }[]>({
    queryKey: ["talent-groups"],
    queryFn: () => fetchJSON("/api/talent-groups"),
  });
  const { data: skills } = useQuery<{ id: string; name: string }[]>({
    queryKey: ["skills-all"],
    queryFn: () => fetchJSON("/api/skills?limit=200"),
  });
  const [groupId, setGroupId] = useState("");
  const [skillIds, setSkillIds] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [subject, setSubject] = useState("");
  const [bodyText, setBodyText] = useState("");

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/campaigns", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["campaigns"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      name: name.trim(),
      subject: subject.trim(),
      body_text: bodyText.trim(),
      filter: {
        group_id: groupId || null,
        skill_ids: skillIds,
        query: String(fd.get("query") || "").trim() || null,
      },
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title="新建外联活动"
      description="创建时把命中的候选人快照为收件清单;发送走后台任务,逐个标记成功/失败。"
      width={560}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="camp-form" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="camp-form" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="活动名" required>
          <Input name="name" value={name} onChange={(e) => setName(e.target.value)} maxLength={128} required placeholder="如 Q4 后端候选人激活" />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="人才分组(可选)">
            <select
              className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 text-sm font-body"
              value={groupId}
              onChange={(e) => setGroupId(e.target.value)}
            >
              <option value="">不限</option>
              {(groups ?? []).map((g) => (
                <option key={g.id} value={g.id}>{g.name}</option>
              ))}
            </select>
          </Field>
          <Field label="姓名/邮箱包含(可选)">
            <Input name="query" maxLength={128} placeholder="模糊匹配" />
          </Field>
        </div>
        {(skills ?? []).length > 0 && (
          <Field label={`技能圈选(可选,已选 ${skillIds.length})`}>
            <div className="max-h-[120px] overflow-auto rounded-lg border border-[var(--color-border-default)] px-3 py-2 flex flex-wrap gap-x-4 gap-y-1.5">
              {(skills ?? []).map((s) => (
                <label key={s.id} className="flex items-center gap-1.5 text-[13px] font-body text-[var(--color-text-primary)]">
                  <input
                    type="checkbox"
                    checked={skillIds.includes(s.id)}
                    onChange={(e) =>
                      setSkillIds((prev) =>
                        e.target.checked ? [...prev, s.id] : prev.filter((x) => x !== s.id),
                      )
                    }
                  />
                  {s.name}
                </label>
              ))}
            </div>
          </Field>
        )}
        <Field label="邮件主题" required>
          <div className="relative">
            <Input name="subject" value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={128} required placeholder="好久不见 {{candidate_name}}" />
            <AiFieldButton
              target="campaigns.subject"
              fields={{ name }}
              onApply={setSubject}
              className="absolute top-1 right-1"
            />
          </div>
        </Field>
        <Field label="邮件正文" required>
          <div className="relative">
            <textarea
              name="body"
              value={bodyText}
              onChange={(e) => setBodyText(e.target.value)}
              required
              rows={7}
              placeholder={"Hi {{candidate_name}},\n我们有一个新机会想跟你聊聊…"}
              className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 pr-11 text-sm font-body outline-none focus:border-[var(--color-accent)]"
            />
            <AiFieldButton
              target="campaigns.body"
              fields={{ name }}
              currentText={bodyText}
              onApply={setBodyText}
              className="absolute top-1.5 right-1.5"
            />
          </div>
        </Field>
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}
