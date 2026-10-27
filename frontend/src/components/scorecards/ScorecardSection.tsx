import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2, ChevronDown, ChevronUp, ClipboardCheck, Lock, Plus,
  Sparkles, Trash2, Users, X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Empty } from "@/components/ui/empty";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";
import { CopilotPanel } from "./CopilotPanel";

// ---------- 类型 ----------

type Criterion = { key: string; name: string; dimension: string; weight: number };

type Feedback = {
  id: string;
  interviewer_user_id: string;
  interviewer_name: string;
  rating: number;
  verdict: string;
  scores: { criterion: string; score: number; comment: string | null }[];
  notes: string | null;
  submitted_at: string;
};

type Aggregate = {
  feedback_count: number;
  avg_rating: number | null;
  verdict_tally: Record<string, number>;
  criteria: { key: string; name: string; dimension: string; weight: number; avg: number | null; min: number | null; max: number | null; count: number }[];
  disagreements: { kind: string; name: string; spread: number }[];
};

export type Scorecard = {
  id: string;
  candidate_id: string;
  candidate_name: string;
  job_id: string | null;
  job_title: string | null;
  interview_id: string | null;
  title: string;
  criteria: Criterion[];
  status: string;
  scheduled_at: string | null;
  location: string | null;
  note: string | null;
  interviewers: { user_id: string; name: string; submitted: boolean }[];
  feedbacks: Feedback[];
  aggregate: Aggregate;
  ai_summary: {
    summary?: string; themes?: string[]; disagreements?: string[];
    recommendation?: string; feedback_count?: number; generated_at?: string;
  } | null;
  created_at: string;
  updated_at: string;
};

type Page<T> = { items: T[]; total: number };

type InterviewerOption = { user_id: string; name: string; role: string; busy_count: number };

// crypto.randomUUID 仅在 secure context(https/localhost)可用,HTTP 部署兜底
const uid8 = () =>
  (typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 14)}`
  ).slice(0, 8);

const DIM_LABEL: Record<string, string> = { tech: "技术", soft: "软素质", general: "通用" };
const DIM_OPTS = [
  { value: "tech", label: "技术能力" },
  { value: "soft", label: "软素质" },
  { value: "general", label: "通用" },
];
const VERDICT_LABEL: Record<string, { label: string; cls: string }> = {
  recommend: { label: "推荐", cls: "text-[var(--color-success)] bg-[var(--color-success)]/10" },
  hold: { label: "待定", cls: "text-[var(--color-warning,#b45309)] bg-amber-500/10" },
  reject: { label: "不推荐", cls: "text-[var(--color-danger)] bg-[var(--color-danger)]/10" },
};

function Stars({ n }: { n: number }) {
  return (
    <span className="font-mono text-[12px] tracking-tight text-[var(--color-text-primary)]">
      {"●".repeat(n)}{"○".repeat(Math.max(0, 5 - n))}
    </span>
  );
}

// ---------- 创建/编辑弹窗 ----------

function ScorecardFormModal({
  open,
  onClose,
  candidateId,
  jobId,
  jobTitle,
  existing,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  candidateId: string;
  jobId?: string | null;
  jobTitle?: string | null;
  existing?: Scorecard | null;
  onSaved: () => void;
}) {
  const [title, setTitle] = useState(existing?.title ?? "");
  const [location, setLocation] = useState(existing?.location ?? "");
  const [note, setNote] = useState(existing?.note ?? "");
  const [scheduledAt, setScheduledAt] = useState(
    existing?.scheduled_at ? existing.scheduled_at.slice(0, 16) : "",
  );
  const [criteria, setCriteria] = useState<Criterion[]>(
    existing?.criteria?.length
      ? existing.criteria.map((c) => ({ ...c }))
      : [{ key: uid8(), name: "", dimension: "tech", weight: 1 }],
  );
  const [interviewers, setInterviewers] = useState<string[]>(
    existing?.interviewers.map((i) => i.user_id) ?? [],
  );
  const [err, setErr] = useState("");

  // 面试官候选:复用 availability 枚举租户用户(只取名单,忽略忙碌度)
  const { data: options } = useQuery({
    queryKey: ["interviewer-options"],
    queryFn: () =>
      fetchJSON<InterviewerOption[]>(
        `/api/interviewers/availability?start=${encodeURIComponent(new Date().toISOString())}&end=${encodeURIComponent(new Date(Date.now() + 3600_000).toISOString())}`,
      ),
    enabled: open,
    staleTime: 60_000,
  });

  const suggestMut = useMutation({
    mutationFn: () =>
      fetchJSON<{ criteria: { name: string; dimension: string }[] }>(
        "/api/scorecards/ai-criteria",
        { method: "POST", body: JSON.stringify({ job_id: jobId ?? null, candidate_id: candidateId }) },
      ),
    onSuccess: (d) => {
      const items = (d.criteria ?? []).map((c) => ({
        key: uid8(),
        name: c.name,
        dimension: c.dimension ?? "general",
        weight: 1,
      }));
      if (items.length) setCriteria(items);
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? `AI 建议不可用:${e.message}` : "AI 建议不可用"),
  });

  const saveMut = useMutation({
    mutationFn: () => {
      const payload = {
        title: title.trim(),
        criteria: criteria
          .filter((c) => c.name.trim())
          .map((c) => ({ key: c.key, name: c.name.trim(), dimension: c.dimension, weight: c.weight })),
        interviewer_ids: interviewers,
        scheduled_at: scheduledAt ? new Date(scheduledAt).toISOString() : null,
        location: location.trim() || null,
        note: note.trim() || null,
      };
      return existing
        ? fetchJSON<Scorecard>(`/api/scorecards/${existing.id}`, {
            method: "PATCH",
            body: JSON.stringify({ ...payload, base_updated_at: existing.updated_at }),
          })
        : fetchJSON<Scorecard>("/api/scorecards", {
            method: "POST",
            body: JSON.stringify({ ...payload, candidate_id: candidateId, job_id: jobId ?? null }),
          });
    },
    onSuccess: () => {
      onSaved();
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  const frozen = !!existing && existing.feedbacks.length > 0;

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={existing ? "编辑评分卡" : "新建评分卡"}
      description={jobTitle ? `岗位:${jobTitle}` : "定义考察维度并指派面试官"}
      width={620}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>取消</Button>
          <Button
            onClick={() => saveMut.mutate()}
            disabled={!title.trim() || !criteria.some((c) => c.name.trim()) || saveMut.isPending}
          >
            {saveMut.isPending ? "保存中…" : existing ? "保存" : "创建"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4 pb-4 font-body">
        {err && (
          <div className="text-[12px] text-[var(--color-danger)] bg-[var(--color-danger)]/8 rounded-lg px-3 py-2">{err}</div>
        )}
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">标题 *</span>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="例:二面 · 技术面"
            className="h-10 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm outline-none focus:border-[var(--color-text-primary)]"
          />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">面试时间</span>
            <input
              type="datetime-local"
              value={scheduledAt}
              onChange={(e) => setScheduledAt(e.target.value)}
              className="h-10 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm outline-none focus:border-[var(--color-text-primary)]"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">地点/方式</span>
            <input
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="例:线上 / 3F-会议室A"
              className="h-10 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm outline-none focus:border-[var(--color-text-primary)]"
            />
          </label>
        </div>

        {/* 考察维度 */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">
              考察维度 *{frozen && <span className="ml-1 font-normal text-[var(--color-text-tertiary)]">(已有反馈,维度冻结)</span>}
            </span>
            {!frozen && (
              <Button
                size="sm" variant="tertiary"
                onClick={() => suggestMut.mutate()}
                disabled={suggestMut.isPending}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {suggestMut.isPending ? "AI 生成中…" : "AI 建议维度"}
              </Button>
            )}
          </div>
          <div className="flex flex-col gap-2">
            {criteria.map((c, i) => (
              <div key={c.key} className="flex items-center gap-2">
                <input
                  value={c.name}
                  disabled={frozen}
                  onChange={(e) =>
                    setCriteria((cs) => cs.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))
                  }
                  placeholder="维度名,如:系统设计能力"
                  className="flex-1 h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-[13px] outline-none focus:border-[var(--color-text-primary)] disabled:opacity-60"
                />
                <Select
                  value={c.dimension}
                  onChange={(v) =>
                    !frozen && setCriteria((cs) => cs.map((x, j) => (j === i ? { ...x, dimension: v } : x)))
                  }
                  options={DIM_OPTS}
                  className="w-28"
                  disabled={frozen}
                />
                <input
                  type="number" min={1} max={10} value={c.weight}
                  disabled={frozen}
                  onChange={(e) =>
                    setCriteria((cs) => cs.map((x, j) => (j === i ? { ...x, weight: Number(e.target.value) || 1 } : x)))
                  }
                  title="权重 1-10"
                  className="w-16 h-9 px-2 rounded-lg border border-[var(--color-border-subtle)] text-[13px] text-center outline-none focus:border-[var(--color-text-primary)] disabled:opacity-60"
                />
                {!frozen && (
                  <button
                    type="button"
                    onClick={() => setCriteria((cs) => cs.filter((_, j) => j !== i))}
                    className="p-1.5 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                    aria-label="删除维度"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            ))}
          </div>
          {!frozen && (
            <button
              type="button"
              onClick={() =>
                setCriteria((cs) => [...cs, { key: uid8(), name: "", dimension: "general", weight: 1 }])
              }
              className="self-start flex items-center gap-1 text-[12px] font-bold text-[var(--color-accent)] hover:underline"
            >
              <Plus className="w-3.5 h-3.5" /> 加维度
            </button>
          )}
        </div>

        {/* 面试官 */}
        <div className="flex flex-col gap-2">
          <span className="text-[12px] font-bold text-[var(--color-text-secondary)] flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5" /> 指派面试官(登录用户)
          </span>
          <div className="max-h-44 overflow-y-auto rounded-lg border border-[var(--color-border-subtle)] divide-y divide-[var(--color-border-row)]">
            {(options ?? []).map((o) => (
              <label key={o.user_id} className="flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-[var(--color-bg-muted)]">
                <Checkbox
                  checked={interviewers.includes(o.user_id)}
                  onToggle={() =>
                    setInterviewers((ids) =>
                      ids.includes(o.user_id) ? ids.filter((x) => x !== o.user_id) : [...ids, o.user_id],
                    )
                  }
                  label={o.name}
                />
                <span className="text-[13px] text-[var(--color-text-primary)] truncate">{o.name}</span>
                {o.busy_count > 0 && (
                  <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)] shrink-0">
                    接下来 1h 有 {o.busy_count} 场
                  </span>
                )}
              </label>
            ))}
            {options && options.length === 0 && (
              <div className="px-3 py-4 text-center text-[12px] text-[var(--color-text-tertiary)]">租户内暂无可指派用户</div>
            )}
          </div>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">备注</span>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            placeholder="给面试官的指引、考察重点等"
            className="px-3 py-2 rounded-lg border border-[var(--color-border-subtle)] text-sm outline-none focus:border-[var(--color-text-primary)] resize-none"
          />
        </label>
      </div>
    </Modal>
  );
}

// ---------- 面试官反馈弹窗 ----------

function FeedbackModal({
  open,
  onClose,
  scorecard,
  mine,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  scorecard: Scorecard;
  mine: Feedback | null;
  onSaved: () => void;
}) {
  const [rating, setRating] = useState(mine?.rating ?? 3);
  const [verdict, setVerdict] = useState(mine?.verdict ?? "hold");
  const [notes, setNotes] = useState(mine?.notes ?? "");
  const [scores, setScores] = useState<Record<string, { score: number; comment: string }>>(() => {
    const m: Record<string, { score: number; comment: string }> = {};
    for (const c of scorecard.criteria) {
      const prev = mine?.scores.find((s) => s.criterion === c.key);
      m[c.key] = { score: prev?.score ?? 3, comment: prev?.comment ?? "" };
    }
    return m;
  });
  const [err, setErr] = useState("");

  const saveMut = useMutation({
    mutationFn: () =>
      fetchJSON<Scorecard>(`/api/scorecards/${scorecard.id}/feedback`, {
        method: "POST",
        body: JSON.stringify({
          rating,
          verdict,
          notes: notes.trim() || null,
          scores: scorecard.criteria.map((c) => ({
            criterion: c.key,
            score: scores[c.key]?.score ?? 3,
            comment: scores[c.key]?.comment?.trim() || null,
          })),
        }),
      }),
    onSuccess: () => {
      onSaved();
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={mine ? "修改我的反馈" : "填写面试反馈"}
      description={scorecard.title}
      width={600}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>取消</Button>
          <Button onClick={() => saveMut.mutate()} disabled={saveMut.isPending}>
            {saveMut.isPending ? "提交中…" : "提交反馈"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4 pb-4 font-body">
        {err && (
          <div className="text-[12px] text-[var(--color-danger)] bg-[var(--color-danger)]/8 rounded-lg px-3 py-2">{err}</div>
        )}

        {/* 逐维评分 */}
        <div className="flex flex-col gap-3">
          {scorecard.criteria.map((c) => (
            <div key={c.key} className="rounded-lg border border-[var(--color-border-subtle)] p-3 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-bold text-[var(--color-text-primary)] flex-1">{c.name}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]">
                  {DIM_LABEL[c.dimension] ?? c.dimension}
                </span>
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setScores((s) => ({ ...s, [c.key]: { ...s[c.key], score: n } }))}
                      className={cn(
                        "w-7 h-7 rounded-md text-[12px] font-mono font-bold transition-colors",
                        scores[c.key]?.score === n
                          ? "bg-[var(--color-accent)] text-white"
                          : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-soft)]",
                      )}
                      aria-label={`${c.name} ${n} 分`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <input
                value={scores[c.key]?.comment ?? ""}
                onChange={(e) => setScores((s) => ({ ...s, [c.key]: { ...s[c.key], comment: e.target.value } }))}
                placeholder="评语/证据(可选)"
                className="h-8 px-2.5 rounded-md border border-[var(--color-border-subtle)] text-[12px] outline-none focus:border-[var(--color-text-primary)]"
              />
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1.5">
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">总评 *</span>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => setRating(n)}
                  className={cn(
                    "flex-1 h-9 rounded-md text-[13px] font-mono font-bold transition-colors",
                    rating === n
                      ? "bg-[var(--color-accent)] text-white"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-accent-soft)]",
                  )}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-1.5">
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">录用表态 *</span>
            <Select
              value={verdict}
              onChange={setVerdict}
              options={[
                { value: "recommend", label: "推荐进入下一轮" },
                { value: "hold", label: "待定" },
                { value: "reject", label: "不推荐" },
              ]}
            />
          </div>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">总体评语</span>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="候选人的整体印象、关键证据、风险点…"
            className="px-3 py-2 rounded-lg border border-[var(--color-border-subtle)] text-sm outline-none focus:border-[var(--color-text-primary)] resize-none"
          />
        </label>
      </div>
    </Modal>
  );
}

// ---------- 单卡展开 ----------

function ScorecardRow({
  sc,
  canWrite,
  onEdit,
  onFeedback,
  onChanged,
}: {
  sc: Scorecard;
  canWrite: boolean;
  onEdit: () => void;
  onFeedback: () => void;
  onChanged: () => void;
}) {
  const { user } = useAuth();
  const confirm = useConfirm();
  const [expanded, setExpanded] = useState(false);
  const closed = sc.status === "closed";
  const myFb = sc.feedbacks.find((f) => f.interviewer_user_id === user?.id) ?? null;
  const amInterviewer = sc.interviewers.some((i) => i.user_id === user?.id);
  const submitted = sc.interviewers.filter((i) => i.submitted).length;

  const synthMut = useMutation({
    mutationFn: () => fetchJSON<Scorecard>(`/api/scorecards/${sc.id}/synthesize`, { method: "POST" }),
    onSuccess: onChanged,
  });
  const closeMut = useMutation({
    mutationFn: () =>
      fetchJSON<Scorecard>(`/api/scorecards/${sc.id}`, {
        method: "PATCH",
        body: JSON.stringify({ status: closed ? "open" : "closed" }),
      }),
    onSuccess: onChanged,
  });
  const delMut = useMutation({
    mutationFn: () => fetchJSON(`/api/scorecards/${sc.id}`, { method: "DELETE" }),
    onSuccess: onChanged,
  });

  const agg = sc.aggregate;
  const ai = sc.ai_summary;

  return (
    <li className="py-3 border-b border-[var(--color-border-row)] last:border-b-0">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm text-[var(--color-text-primary)]">{sc.title}</span>
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded font-bold",
                closed
                  ? "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                  : "bg-[var(--color-success)]/10 text-[var(--color-success)]",
              )}
            >
              {closed ? "已关闭" : "进行中"}
            </span>
            {sc.job_title && (
              <span className="text-[11px] text-[var(--color-text-tertiary)]">{sc.job_title}</span>
            )}
          </div>
          <span className="text-[11px] text-[var(--color-text-tertiary)]">
            {sc.criteria.length} 个维度 · {sc.interviewers.length} 位面试官
            {sc.interviewers.length > 0 && ` · ${submitted}/${sc.interviewers.length} 已提交`}
            {sc.scheduled_at && ` · ${formatRelative(sc.scheduled_at)}`}
            {sc.location && ` · ${sc.location}`}
          </span>
          {agg.avg_rating != null && (
            <span className="text-[12px] font-mono text-[var(--color-text-primary)]">
              总均分 {agg.avg_rating}
              {agg.verdict_tally.recommend > 0 && ` · 推荐×${agg.verdict_tally.recommend}`}
              {agg.verdict_tally.reject > 0 && ` · 不推荐×${agg.verdict_tally.reject}`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {amInterviewer && !closed && (
            <Button size="sm" variant={myFb ? "secondary" : "primary"} onClick={onFeedback}>
              <ClipboardCheck className="w-3.5 h-3.5" />
              {myFb ? "改反馈" : "填反馈"}
            </Button>
          )}
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="p-1.5 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
            aria-label={expanded ? "收起" : "展开"}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 flex flex-col gap-3">
          {/* 维度 */}
          <div className="flex flex-wrap gap-1.5">
            {sc.criteria.map((c) => {
              const a = agg.criteria.find((x) => x.key === c.key);
              return (
                <span
                  key={c.key}
                  className="px-2 py-1 rounded-md bg-[var(--color-bg-subtle)] text-[11px] text-[var(--color-text-secondary)]"
                  title={`${DIM_LABEL[c.dimension] ?? c.dimension} · 权重 ${c.weight}`}
                >
                  {c.name}
                  {a?.avg != null && <b className="ml-1 font-mono">{a.avg}</b>}
                  {a && a.max != null && a.min != null && a.max - a.min >= 2 && (
                    <span className="ml-1 text-[var(--color-danger)]" title="评分分歧">⚠</span>
                  )}
                </span>
              );
            })}
          </div>

          {/* 面试官提交状态 */}
          {sc.interviewers.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {sc.interviewers.map((i) => (
                <span
                  key={i.user_id}
                  className={cn(
                    "inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full",
                    i.submitted
                      ? "bg-[var(--color-success)]/10 text-[var(--color-success)]"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]",
                  )}
                >
                  {i.submitted && <CheckCircle2 className="w-3 h-3" />}
                  {i.name}
                </span>
              ))}
            </div>
          )}

          {/* AI 面试副驾:open 评分卡才有追问建议的意义 */}
          {sc.status === "open" && (
            <CopilotPanel
              scorecardId={sc.id}
              coveredDims={sc.feedbacks.flatMap((f) =>
                f.scores.map((s) => sc.criteria.find((x) => x.key === s.criterion)?.name ?? s.criterion)
              )}
            />
          )}

          {/* 分歧提示 */}
          {agg.disagreements.length > 0 && (
            <div className="rounded-lg bg-amber-500/8 border border-amber-500/20 px-3 py-2 text-[12px] text-amber-800">
              存在分歧:{agg.disagreements.map((d) => d.name).join("、")}
            </div>
          )}

          {/* 反馈明细 */}
          {sc.feedbacks.map((f) => (
            <div key={f.id} className="rounded-lg border border-[var(--color-border-subtle)] p-3 flex flex-col gap-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[13px] font-bold text-[var(--color-text-primary)]">{f.interviewer_name}</span>
                <Stars n={f.rating} />
                <span className={cn("text-[10px] px-1.5 py-0.5 rounded font-bold", VERDICT_LABEL[f.verdict]?.cls)}>
                  {VERDICT_LABEL[f.verdict]?.label ?? f.verdict}
                </span>
                <span className="text-[10px] text-[var(--color-text-tertiary)] ml-auto">
                  {formatRelative(f.submitted_at)}
                </span>
              </div>
              {f.scores.length > 0 && (
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {f.scores.map((s) => {
                    const c = sc.criteria.find((x) => x.key === s.criterion);
                    return (
                      <span key={s.criterion} className="text-[11px] text-[var(--color-text-secondary)]">
                        {c?.name ?? s.criterion} <b className="font-mono">{s.score}</b>
                        {s.comment && <span className="text-[var(--color-text-tertiary)]"> · {s.comment}</span>}
                      </span>
                    );
                  })}
                </div>
              )}
              {f.notes && (
                <p className="text-[12px] text-[var(--color-text-secondary)] whitespace-pre-wrap">{f.notes}</p>
              )}
            </div>
          ))}

          {/* AI 摘要 */}
          {ai && (
            <div className="rounded-lg bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent)]/15 p-3 flex flex-col gap-1.5">
              <span className="text-[11px] font-bold text-[var(--color-accent)] flex items-center gap-1">
                <Sparkles className="w-3 h-3" /> AI 评审摘要
                {ai.generated_at && <span className="font-normal text-[var(--color-text-tertiary)]">{formatRelative(ai.generated_at)}</span>}
              </span>
              {ai.summary && <p className="text-[12px] text-[var(--color-text-primary)]">{ai.summary}</p>}
              {(ai.themes?.length ?? 0) > 0 && (
                <div className="text-[11px] text-[var(--color-text-secondary)]">
                  共识:{ai.themes!.join(";")}
                </div>
              )}
              {(ai.disagreements?.length ?? 0) > 0 && (
                <div className="text-[11px] text-amber-800">分歧:{ai.disagreements!.join(";")}</div>
              )}
              {ai.recommendation && (
                <div className="text-[12px] font-bold text-[var(--color-text-primary)]">
                  建议:{ai.recommendation}
                </div>
              )}
              <div className="text-[10px] text-[var(--color-text-tertiary)]">AI 建议仅供参考,录用决策以人工为准</div>
            </div>
          )}

          {/* 管理操作 */}
          {canWrite && (
            <div className="flex items-center gap-2 pt-1">
              <Button size="sm" variant="secondary" onClick={onEdit}>编辑</Button>
              {!closed && sc.feedbacks.length > 0 && (
                <Button
                  size="sm" variant="secondary"
                  onClick={() => synthMut.mutate()}
                  disabled={synthMut.isPending}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  {synthMut.isPending ? "聚合中…" : ai ? "重新聚合" : "AI 聚合"}
                </Button>
              )}
              <Button
                size="sm" variant="secondary"
                onClick={() => closeMut.mutate()}
                disabled={closeMut.isPending}
              >
                <Lock className="w-3.5 h-3.5" />
                {closed ? "重开" : "关闭"}
              </Button>
              <button
                type="button"
                onClick={async () => {
                  if (
                    await confirm({
                      title: "删除评分卡",
                      description: `删除「${sc.title}」及其全部反馈?`,
                      tone: "danger",
                    })
                  )
                    delMut.mutate();
                }}
                className="p-1.5 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                aria-label="删除评分卡"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

// ---------- Section ----------

export function ScorecardSection({
  candidateId,
  jobId,
  jobTitle,
}: {
  candidateId: string;
  jobId?: string | null;
  jobTitle?: string | null;
}) {
  const { user, hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.WRITE_INTERVIEW);
  const qc = useQueryClient();
  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<Scorecard | null>(null);
  const [feedbackFor, setFeedbackFor] = useState<Scorecard | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["scorecards", candidateId],
    queryFn: () =>
      fetchJSON<Page<Scorecard>>(`/api/scorecards?candidate_id=${encodeURIComponent(candidateId)}&limit=50`),
  });
  const items = data?.items ?? [];

  const refresh = () => qc.invalidateQueries({ queryKey: ["scorecards", candidateId] });

  return (
    <div className="flex flex-col">
      {isLoading ? (
        <div className="text-[12px] text-[var(--color-text-tertiary)] font-body py-4">加载中…</div>
      ) : items.length === 0 ? (
        <Empty
          icon={ClipboardCheck}
          title="还没有评分卡"
          description={canWrite ? "创建结构化评分卡,让面试官按统一维度打分" : "面试安排后会出现在这里"}
        />
      ) : (
        <ul className="flex flex-col">
          {items.map((sc) => (
            <ScorecardRow
              key={sc.id}
              sc={sc}
              canWrite={canWrite}
              onEdit={() => {
                setEditing(sc);
                setFormOpen(true);
              }}
              onFeedback={() => setFeedbackFor(sc)}
              onChanged={refresh}
            />
          ))}
        </ul>
      )}
      {canWrite && (
        <button
          type="button"
          onClick={() => {
            setEditing(null);
            setFormOpen(true);
          }}
          className="self-start mt-2 flex items-center gap-1 text-[12px] font-bold text-[var(--color-accent)] hover:underline"
        >
          <Plus className="w-3.5 h-3.5" /> 新建评分卡
        </button>
      )}

      <ScorecardFormModal
        open={formOpen}
        onClose={() => setFormOpen(false)}
        candidateId={candidateId}
        jobId={editing?.job_id ?? jobId}
        jobTitle={editing?.job_title ?? jobTitle}
        existing={editing}
        onSaved={refresh}
      />
      {feedbackFor && (
        <FeedbackModal
          open
          onClose={() => setFeedbackFor(null)}
          scorecard={feedbackFor}
          mine={
            feedbackFor.feedbacks.find((f) => f.interviewer_user_id === user?.id) ?? null
          }
          onSaved={refresh}
        />
      )}
    </div>
  );
}
