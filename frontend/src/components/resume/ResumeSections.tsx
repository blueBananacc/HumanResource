import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ApiError, fetchJSON } from "@/lib/api";
import { type Page } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { MatchBadge } from "@/components/match/MatchBadge";

// ---------- 抽屉里的「面试题集」section----------

type QSItem = {
  id: string;
  level: string;
  count: number;
  status: "pending" | "generating" | "done" | "failed";
  created_at: string;
  finished_at: string | null;
  job_title: string | null;
};

const QS_KINDS = ["技术深度", "项目复盘", "场景排查", "系统设计", "软技能"];

export function QuestionSetsForResume({ resumeId }: { resumeId: string }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [showConfig, setShowConfig] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["question-sets-by-resume", resumeId],
    queryFn: () =>
      // 简历详情抽屉里子查询:不分页直接抓最多 200 条,通常一份简历对
      // 应的题集 < 10 条,翻页没意义。下面取 .items 适配统一分页响应。
      fetchJSON<Page<QSItem>>(
        `/api/question-sets/?resume_id=${resumeId}&limit=200`,
      ),
    refetchInterval: (q) => {
      const d = q.state.data as Page<QSItem> | undefined;
      return d &&
        d.items.some((x) => x.status === "pending" || x.status === "generating")
        ? 2000
        : false;
    },
  });

  const items = data?.items ?? [];

  return (
    <div className="flex flex-col gap-2.5">
      {!isLoading && items.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {items.map((q) => (
            <Link
              key={q.id}
              to={`/question-sets/${q.id}`}
              className="flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-muted)] transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <Sparkles className="w-3.5 h-3.5 text-[var(--color-accent)] shrink-0" />
                <span className="text-[12px] font-body text-[var(--color-text-primary)] truncate">
                  {q.count} 题 · {q.job_title ?? "未指定岗位"}
                </span>
                {q.status !== "done" && (
                  <span className="text-[10px] font-body text-[var(--color-text-tertiary)] shrink-0">
                    ({q.status === "failed" ? "失败" : "生成中"})
                  </span>
                )}
              </div>
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-mono shrink-0 ml-3">
                {formatRelative(q.finished_at ?? q.created_at)}
              </span>
            </Link>
          ))}
        </div>
      )}

      {showConfig ? (
        <GenerateQSForm
          resumeId={resumeId}
          onCancel={() => setShowConfig(false)}
          onCreated={(id) => {
            qc.invalidateQueries({
              queryKey: ["question-sets-by-resume", resumeId],
            });
            navigate(`/question-sets/${id}`);
          }}
        />
      ) : (
        <Button variant="secondary" onClick={() => setShowConfig(true)}>
          <Sparkles className="w-3.5 h-3.5" />
          生成面试题
        </Button>
      )}
    </div>
  );
}

function GenerateQSForm({
  resumeId,
  onCancel,
  onCreated,
}: {
  resumeId: string;
  onCancel: () => void;
  onCreated: (id: string) => void;
}) {
  const [level, setLevel] = useState<
    "initial" | "intermediate" | "advanced" | "expert"
  >("intermediate");
  const [count, setCount] = useState(5);
  const [kinds, setKinds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      fetchJSON<{ id: string }>("/api/question-sets/", {
        method: "POST",
        body: JSON.stringify({
          resume_id: resumeId,
          level,
          count,
          kinds,
        }),
      }),
    onSuccess: (out) => onCreated(out.id),
    onError: (e: unknown) =>
      setError(e instanceof ApiError ? e.message : "生成失败"),
  });

  return (
    <div className="flex flex-col gap-3 p-3 rounded-lg bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)]">
      <div className="grid grid-cols-2 gap-3">
        <Field label="难度">
          <Select
            value={level}
            onChange={(v) => setLevel(v as typeof level)}
            options={[
              { value: "initial", label: "初级" },
              { value: "intermediate", label: "中级" },
              { value: "advanced", label: "高级" },
              { value: "expert", label: "专家" },
            ]}
          />
        </Field>
        <Field label="题量">
          <Select
            value={String(count)}
            onChange={(v) => setCount(Number(v))}
            options={[
              { value: "5", label: "5 题" },
              { value: "10", label: "10 题" },
              { value: "15", label: "15 题" },
            ]}
          />
        </Field>
      </div>
      <Field label="题目类型(留空交给 AI 自由分配)">
        <div className="flex flex-wrap gap-1.5">
          {QS_KINDS.map((k) => (
            <button
              key={k}
              type="button"
              onClick={() =>
                setKinds((prev) =>
                  prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k],
                )
              }
              className={`px-2.5 py-1 rounded-md text-[11px] font-body transition-colors ${
                kinds.includes(k)
                  ? "bg-[var(--color-text-primary)] text-white"
                  : "bg-white border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-muted)]"
              }`}
            >
              {k}
            </button>
          ))}
        </div>
      </Field>
      {error && (
        <div className="text-[12px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-2 py-1.5 rounded-md font-body">
          {error}
        </div>
      )}
      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onCancel}>
          取消
        </Button>
        <Button onClick={() => create.mutate()} disabled={create.isPending}>
          <Sparkles className="w-3.5 h-3.5" />
          {create.isPending ? "生成中…" : "开始生成"}
        </Button>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
        {label}
      </span>
      {children}
    </div>
  );
}

// ----------------------------- 岗位匹配 -----------------------------

type MatchItem = {
  id: string;
  resume_id: string;
  job_id: string;
  status: "pending" | "matching" | "done" | "failed";
  score: number | null;
  comment: string | null;
  error: string | null;
  finished_at: string | null;
  job_title: string | null;
};

type TriggerOut = {
  enqueued: boolean;
  target_total: number;
  queued: number;
};

const MATCH_POLL_MS = 2000;
const isMatchInflight = (s: MatchItem["status"]) =>
  s === "pending" || s === "matching";

export function MatchesForResume({ resumeId }: { resumeId: string }) {
  const qc = useQueryClient();
  const [feedback, setFeedback] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["matches-by-resume", resumeId],
    queryFn: () =>
      // 按简历维度的子查询,抓满 200 够用(一份简历对应的 active 岗位
      // 通常 < 50),取 .items 适配统一分页响应。
      fetchJSON<Page<MatchItem>>(`/api/matches/resume/${resumeId}?limit=200`),
    refetchInterval: (q) => {
      const d = q.state.data as Page<MatchItem> | undefined;
      return d && d.items.some((x) => isMatchInflight(x.status))
        ? MATCH_POLL_MS
        : false;
    },
  });

  const matchItems = data?.items ?? [];

  const trigger = useMutation({
    mutationFn: () =>
      fetchJSON<TriggerOut>(`/api/matches/resume/${resumeId}/evaluate-all`, {
        method: "POST",
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["matches-by-resume", resumeId] });
      setFeedback(
        r.queued > 0
          ? `已对 ${r.queued} 个岗位入队评估`
          : `${r.target_total} 个岗位都已评估,如需重评请单条点重评`,
      );
    },
    onError: (e) =>
      setFeedback(e instanceof ApiError ? e.message : "触发失败"),
  });

  const regen = useMutation({
    mutationFn: (matchId: string) =>
      fetchJSON<MatchItem>(`/api/matches/${matchId}/regen`, { method: "POST" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["matches-by-resume", resumeId] }),
  });

  return (
    <div className="flex flex-col gap-2.5">
      {!isLoading && matchItems.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {matchItems.map((m) => (
            <div
              key={m.id}
              className="flex items-start gap-3 px-3 py-2 rounded-lg bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)]"
            >
              <MatchBadge
                status={m.status}
                score={m.score}
                error={m.error}
                size="sm"
              />
              <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                <span className="text-[12px] font-medium text-[var(--color-text-primary)] font-body truncate">
                  {m.job_title ?? "(岗位已删除)"}
                </span>
                {m.comment && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body line-clamp-2">
                    {m.comment}
                  </span>
                )}
              </div>
              {m.status === "done" && (
                <button
                  type="button"
                  onClick={() => regen.mutate(m.id)}
                  className="text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] font-body shrink-0 self-center"
                  title="重新评估"
                >
                  重评
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {!isLoading && matchItems.length === 0 && (
        <div className="text-[11px] text-[var(--color-text-tertiary)] font-body py-2">
          还没有岗位匹配评估结果。
        </div>
      )}

      {feedback && (
        <div className="text-[11px] bg-[var(--color-info-soft)] text-[var(--color-info)] px-3 py-1.5 rounded-md font-body">
          {feedback}
        </div>
      )}

      <Button
        variant="secondary"
        onClick={() => {
          setFeedback(null);
          trigger.mutate();
        }}
        disabled={trigger.isPending}
      >
        <Sparkles className="w-3.5 h-3.5" />
        {trigger.isPending ? "触发中…" : "评估所有 active 岗位"}
      </Button>
    </div>
  );
}