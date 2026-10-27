import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft, Ban, ChevronDown, ChevronRight, KanbanSquare,
  RefreshCw, Sparkles, Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { ApiError, fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { RadarChart } from "@/components/charts/RadarChart";
import { formatRelative, levelLabel } from "@/lib/format";
import { MatchBadge } from "@/components/match/MatchBadge";

/**
 * 岗位「匹配候选人」视图。
 *
 * 核心数据来自 ``GET /api/matches/job/{id}?min_score&limit``,只返回 status='done' 的对。
 * 顶部「重新评估」走 ``POST /api/matches/job/{id}/evaluate-all`` 把还没评的简历入队。
 *
 * HR 直接从这里挑人发面试 — 点「发起面试」跳到 `/interviews?prefill=...`(由
 * Interviews 页解析 query 自动选中候选人 + 岗位)。简化:这次只做 query 参数
 * 跳转;Interviews 是否解析 prefill 在下一步收尾里加。
 */

type Job = {
  id: string;
  title: string;
  level: string;
  status: string;
  skills: string[];
};

type GateAnswer = {
  id: string;
  question: string;
  verdict: "PASS" | "FAIL" | "UNCERTAIN";
  evidence: string;
};

type Disqualifier = {
  id: string;
  question: string;
  severity: number;
  evidence: string;
};

type Gate = {
  passed: boolean;
  answers: GateAnswer[];
  disqualifiers: Disqualifier[];
  skipped?: string;
};

type Criterion = {
  key: string;
  name: string;
  weight: number;
  level: number;
  evidence: string;
  method?: string; // T13 规则评估器方法;无 = T12 五维 AI 评分
};

type Match = {
  id: string;
  resume_id: string;
  job_id: string;
  status: "pending" | "matching" | "done" | "failed" | "gated";
  score: number | null;
  strengths: string[];
  gaps: string[];
  comment: string | null;
  gate: Gate | null;
  criteria: Criterion[] | null;
  red_flags: { major: string[]; minor: string[] } | null;
  rule_snapshot?: { rule_id: string; name: string; version: number } | null;
  coverage?: { score: number; hit: string[]; missing: string[]; total: number } | null;
  error: string | null;
  finished_at: string | null;
  resume_file_name: string | null;
  candidate_id: string | null;
  candidate_name: string | null;
};

type TriggerResponse = {
  enqueued: boolean;
  target_total: number;
  queued: number;
};

const MIN_SCORE_OPTS = [
  { value: 0, label: "全部" },
  { value: 50, label: "≥ 50" },
  { value: 65, label: "≥ 65" },
  { value: 80, label: "≥ 80" },
];

export function JobMatches() {
  const { id = "" } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [minScore, setMinScore] = useState(0);
  const [gatedOpen, setGatedOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.WRITE_JOBS);

  const jobQuery = useQuery({
    queryKey: ["job", id],
    queryFn: () => fetchJSON<Job>(`/api/jobs/${id}`),
  });

  const matchesQuery = useQuery({
    queryKey: ["matches-by-job", id, minScore],
    queryFn: () =>
      fetchJSON<Match[]>(
        `/api/matches/job/${id}?min_score=${minScore}&limit=50`,
      ),
  });

  // T12:硬性门槛淘汰名单(独立于分数池,score=NULL 不进排序)
  const gatedQuery = useQuery({
    queryKey: ["matches-gated", id],
    queryFn: () =>
      fetchJSON<Match[]>(`/api/matches/job/${id}?status=gated&limit=50`),
  });

  const addToPipeline = useMutation({
    mutationFn: (candidateId: string) =>
      fetchJSON(`/api/applications`, {
        method: "POST",
        body: JSON.stringify({ candidate_id: candidateId, job_id: id }),
      }),
    onSuccess: () => setFeedback("已加入招聘管道(新投递)"),
    onError: (e) =>
      setFeedback(
        e instanceof ApiError && e.status === 409
          ? "该候选人已在管道中"
          : e instanceof ApiError
            ? e.message
            : "加入失败",
      ),
  });

  const trigger = useMutation({
    mutationFn: () =>
      fetchJSON<TriggerResponse>(`/api/matches/job/${id}/evaluate-all`, {
        method: "POST",
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["matches-by-job", id] });
      setFeedback(
        r.queued > 0
          ? `已对 ${r.queued} 份简历入队评估,稍后刷新查看结果`
          : `${r.target_total} 份简历都已评估过,如需重评请在单条上点"重新评估"`,
      );
    },
    onError: (e) =>
      setFeedback(e instanceof ApiError ? e.message : "触发失败"),
  });

  const job = jobQuery.data;
  const matches = matchesQuery.data ?? [];
  const gated = gatedQuery.data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <Link
        to="/jobs"
        className="inline-flex items-center gap-1.5 text-[13px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-body w-fit"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        返回岗位列表
      </Link>

      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5 min-w-0">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)] truncate">
            {job?.title ?? "(加载中…)"}
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            按 AI 匹配度排序的候选人池 · 点「发起面试」直接邀请
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            onClick={() => navigate(`/pipeline?job=${id}`)}
          >
            <KanbanSquare className="w-4 h-4" />
            招聘管道
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              setFeedback(null);
              trigger.mutate();
            }}
            disabled={trigger.isPending}
          >
            <RefreshCw
              className={`w-4 h-4 ${trigger.isPending ? "animate-spin" : ""}`}
            />
            {trigger.isPending ? "触发中…" : "重新评估"}
          </Button>
        </div>
      </div>

      {feedback && (
        <div className="text-[13px] bg-[var(--color-info-soft)] text-[var(--color-info)] px-3 py-2 rounded-md font-body">
          {feedback}
        </div>
      )}

      <div className="flex items-center gap-2 text-[12px] font-body">
        <span className="text-[var(--color-text-tertiary)]">最低分</span>
        {MIN_SCORE_OPTS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setMinScore(opt.value)}
            className={`px-2.5 py-1 rounded-md transition-colors ${
              minScore === opt.value
                ? "bg-[var(--color-text-primary)] text-white"
                : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-subtle)]"
            }`}
          >
            {opt.label}
          </button>
        ))}
        <span className="ml-auto text-[var(--color-text-tertiary)]">
          {matches.length} 位候选人
          {gated.length > 0 && ` · ${gated.length} 人门槛淘汰`}
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {matchesQuery.isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : matches.length === 0 ? (
          <Empty
            icon={Users}
            title="暂无匹配结果"
            description={
              minScore > 0
                ? "调低最低分阈值看看,或点上方「重新评估」对未评估的简历入队。"
                : "AI 还没评估完成,或者还没简历。点「重新评估」对最近简历入队评分。"
            }
          />
        ) : (
          matches.map((m, i) => (
            <MatchRow
              key={m.id}
              match={m}
              level={job?.level ?? ""}
              isLast={i === matches.length - 1}
              onStartInterview={() => {
                if (m.candidate_id) {
                  navigate(
                    `/interviews?candidate=${m.candidate_id}&job=${m.job_id}`,
                  );
                }
              }}
              onAddPipeline={
                canWrite
                  ? () => {
                      if (m.candidate_id) {
                        setFeedback(null);
                        addToPipeline.mutate(m.candidate_id);
                      }
                    }
                  : undefined
              }
            />
          ))
        )}
      </div>

      {gated.length > 0 && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <button
            type="button"
            onClick={() => setGatedOpen((v) => !v)}
            className="w-full flex items-center gap-2 px-6 py-3 text-left text-[13px] font-medium text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-subtle)]"
          >
            <Ban className="w-3.5 h-3.5 text-red-500" />
            门槛淘汰({gated.length})
            <span className="text-[11px] text-[var(--color-text-tertiary)]">
              硬性约束不满足,未进入评分
            </span>
            {gatedOpen ? (
              <ChevronDown className="w-3.5 h-3.5 ml-auto" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 ml-auto" />
            )}
          </button>
          {gatedOpen &&
            gated.map((m, i) => (
              <div
                key={m.id}
                className={`flex items-start gap-4 px-6 py-3.5 ${
                  i === gated.length - 1
                    ? ""
                    : "border-b border-[var(--color-border-row)]"
                } border-t`}
              >
                <MatchBadge status="gated" score={null} size="md" />
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  <span className="font-medium text-sm text-[var(--color-text-primary)]">
                    {m.candidate_name ?? "(候选人不存在)"}
                  </span>
                  {(m.gate?.disqualifiers ?? []).map((d) => (
                    <div key={d.id} className="text-[12px] font-body">
                      <span className="text-red-600 font-medium">
                        ✗ {d.question}
                      </span>
                      <span className="ml-2 text-[var(--color-text-tertiary)]">
                        严重度 {d.severity}/10 · 证据: {d.evidence}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
        </div>
      )}
    </main>
  );
}

function MatchRow({
  match,
  level,
  isLast,
  onStartInterview,
  onAddPipeline,
}: {
  match: Match;
  level: string;
  isLast: boolean;
  onStartInterview: () => void;
  onAddPipeline?: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={`flex items-start gap-4 px-6 py-4 ${
        isLast ? "" : "border-b border-[var(--color-border-row)]"
      }`}
    >
      <MatchBadge
        status={match.status}
        score={match.score}
        error={match.error}
        size="md"
      />
      <div className="flex flex-col gap-1.5 flex-1 min-w-0">
        <div className="flex items-center gap-2.5">
          <span className="font-medium text-sm text-[var(--color-text-primary)] font-body truncate">
            {match.candidate_name ?? "(候选人不存在)"}
          </span>
          {level && (
            <span className="px-2 py-0.5 rounded-md bg-[var(--color-bg-subtle)] text-[11px] text-[var(--color-text-secondary)] font-body">
              对 {levelLabel(level)} 岗位
            </span>
          )}
          <span className="text-[11px] text-[var(--color-text-tertiary)] font-mono ml-auto">
            {match.finished_at ? formatRelative(match.finished_at) : ""}
          </span>
        </div>
        {match.comment && (
          <p className="text-[13px] text-[var(--color-text-secondary)] font-body">
            {match.comment}
          </p>
        )}
        {(match.strengths.length > 0 || match.gaps.length > 0) && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {match.strengths.slice(0, 3).map((s, idx) => (
              <span
                key={`s-${idx}`}
                className="px-2 py-0.5 rounded-md bg-[var(--color-success-soft)] text-[var(--color-success)] text-[11px] font-body"
              >
                ✓ {s}
              </span>
            ))}
            {match.gaps.slice(0, 2).map((g, idx) => (
              <span
                key={`g-${idx}`}
                className="px-2 py-0.5 rounded-md bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)] text-[11px] font-body"
              >
                ✗ {g}
              </span>
            ))}
          </div>
        )}
        {(match.red_flags?.major?.length ?? 0) > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {match.red_flags!.major.map((f) => (
              <span
                key={f}
                className="px-2 py-0.5 rounded-md bg-red-50 text-red-700 text-[11px] font-body"
              >
                🚩 {f}
              </span>
            ))}
            {(match.red_flags?.minor ?? []).map((f) => (
              <span
                key={f}
                className="px-2 py-0.5 rounded-md bg-amber-50 text-amber-700 text-[11px] font-body"
              >
                ⚑ {f}
              </span>
            ))}
          </div>
        )}
        {match.criteria && match.criteria.length > 0 && (
          <div className="pt-1">
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              className="text-[11px] text-[var(--color-accent)] hover:underline font-body"
            >
              {open ? "收起评分明细" : "评分明细(逐维证据)"}
            </button>
            {match.rule_snapshot && (
              <span className="ml-2 rounded-full bg-[var(--color-cappuccino)] px-2 py-0.5 text-[10px] text-[var(--color-brand)] font-body">
                规则「{match.rule_snapshot.name}」v{match.rule_snapshot.version}
              </span>
            )}
            {open && (
              <div className="mt-1.5 flex flex-wrap items-start gap-4">
                {/* 雷达图:逐维 level(0-4)归一化后可视化,不替代下方证据链 */}
                {match.criteria.length >= 3 && (
                  <RadarChart
                    dims={match.criteria.map((c) => ({
                      name: c.name,
                      value: c.level / 4,
                    }))}
                    size={200}
                  />
                )}
                <div className="min-w-[280px] flex-1 rounded-lg border border-[var(--color-border-subtle)] divide-y divide-[var(--color-border-row)]">
                {match.coverage && (
                  <div className="px-3 py-2 bg-[var(--color-bg-canvas)]">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] font-medium text-[var(--color-text-primary)] font-body">
                        关键词覆盖
                      </span>
                      <span
                        className={`rounded px-1.5 py-px text-[10px] font-mono ${
                          match.coverage.score >= 80
                            ? "bg-emerald-50 text-emerald-600"
                            : match.coverage.score >= 50
                              ? "bg-amber-50 text-amber-600"
                              : "bg-red-50 text-red-500"
                        }`}
                      >
                        {match.coverage.score}%
                      </span>
                      <span className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
                        命中 {match.coverage.hit.length}/{match.coverage.total}
                      </span>
                    </div>
                    {match.coverage.missing.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">缺失:</span>
                        {match.coverage.missing.map((k) => (
                          <span
                            key={k}
                            className="rounded bg-[var(--color-bg-subtle)] px-1.5 py-px text-[10px] text-[var(--color-text-secondary)] font-mono"
                          >
                            {k}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {match.criteria.map((c) => (
                  <div key={c.key} className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] font-medium text-[var(--color-text-primary)] font-body">
                        {c.name}
                      </span>
                      <span className="flex gap-0.5">
                        {[0, 1, 2, 3, 4].map((i) => (
                          <span
                            key={i}
                            className={`w-3 h-1.5 rounded-full ${
                              i <= c.level
                                ? c.level >= 3
                                  ? "bg-emerald-500"
                                  : c.level >= 2
                                    ? "bg-amber-400"
                                    : "bg-red-400"
                                : "bg-[var(--color-bg-subtle)]"
                            }`}
                          />
                        ))}
                      </span>
                      <span className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
                        L{c.level} · 权重{c.weight}
                      </span>
                      {c.method && (
                        <span
                          className={`rounded px-1 py-px text-[9px] font-body ${
                            c.method === "ai"
                              ? "bg-violet-50 text-violet-600"
                              : "bg-sky-50 text-sky-600"
                          }`}
                        >
                          {{ keyword: "关键词", coverage: "覆盖", duration: "年限", boolean: "有无", range: "区间", ai: "AI" }[
                            c.method
                          ] ?? c.method}
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 text-[11px] text-[var(--color-text-secondary)] font-body italic">
                      “{c.evidence}”
                    </p>
                  </div>
                ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex flex-col gap-1.5 shrink-0">
        <Button
          variant="secondary"
          onClick={onStartInterview}
          className="!px-3 !py-1.5 !text-[12px]"
        >
          <Sparkles className="w-3.5 h-3.5" />
          发起面试
        </Button>
        {match.candidate_id && onAddPipeline && (
          <Button
            variant="secondary"
            onClick={onAddPipeline}
            className="!px-3 !py-1.5 !text-[12px]"
          >
            <KanbanSquare className="w-3.5 h-3.5" />
            加入管道
          </Button>
        )}
      </div>
    </div>
  );
}
