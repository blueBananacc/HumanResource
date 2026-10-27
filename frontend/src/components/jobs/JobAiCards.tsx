/** 岗位 AI 卡片组(对标 Moka 动态画像 + AI 审核 JD):
 * - JobAiProfileCard:动态职位画像(规则权重×决策学习×匹配统计,一句话画像)
 * - JobJdReview:JD 六维合规审查(按钮触发,结果就地展示)
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Brain, ShieldCheck, Sparkles, TrendingUp } from "lucide-react";
import { fetchJSON, ApiError } from "@/lib/api";

/* ---------------- AI 画像 ---------------- */

type AiProfile = {
  job_id: string;
  title: string;
  rule: { name: string; version: number; dimensions: { key: string; name: string; weight: number }[] } | null;
  learned: {
    decisions_total: number;
    positive: number;
    negative: number;
    confidence: "low" | "medium" | "high";
    dimensions: {
      name: string;
      current_weight: number | null;
      suggested_weight: number | null;
      gap: number | null;
      pos_avg_level: number | null;
      neg_avg_level: number | null;
    }[];
  };
  match_stats: { evaluated: number; avg_score: number | null; top_score: number | null };
  one_liner: string;
};

const CONF = {
  low: { label: "低置信", cls: "bg-slate-100 text-slate-500" },
  medium: { label: "中置信", cls: "bg-amber-50 text-amber-600" },
  high: { label: "高置信", cls: "bg-emerald-50 text-emerald-600" },
};

export function JobAiProfileCard({ jobId }: { jobId: string }) {
  const { data } = useQuery<AiProfile>({
    queryKey: ["job-ai-profile", jobId],
    queryFn: () => fetchJSON(`/api/jobs/${jobId}/ai-profile`),
    staleTime: 60_000,
  });
  if (!data) return null;
  const conf = CONF[data.learned.confidence];
  const learnedDims = data.learned.dimensions
    .filter((d) => d.suggested_weight != null && Math.abs(d.gap ?? 0) >= 0.5)
    .slice(0, 4);

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
          AI 动态画像
        </h3>
        <span className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-body ${conf.cls}`}>
          {conf.label} · {data.learned.decisions_total} 次决策
        </span>
      </div>
      <p className="text-[13px] text-[var(--color-text-secondary)] font-body leading-relaxed">
        {data.one_liner}
      </p>
      {learnedDims.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
            决策学习发现的偏好偏移:
          </span>
          {learnedDims.map((d) => (
            <div key={d.name} className="flex items-center gap-2 text-[12px] font-body">
              <span className="w-24 truncate text-[var(--color-text-primary)]">{d.name}</span>
              <span className="font-mono text-[var(--color-text-tertiary)]">
                {d.current_weight ?? "?"} → {d.suggested_weight}
              </span>
              <span
                className={`rounded px-1 text-[10px] ${
                  (d.gap ?? 0) > 0 ? "bg-emerald-50 text-emerald-600" : "bg-red-50 text-red-500"
                }`}
              >
                {(d.gap ?? 0) > 0 ? "更看中" : "放宽"}
              </span>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center gap-3 text-[11px] text-[var(--color-text-tertiary)] font-body">
        <span className="flex items-center gap-1">
          <TrendingUp className="w-3 h-3" />
          已评估 {data.match_stats.evaluated} 人
        </span>
        {data.match_stats.avg_score != null && <span>平均 {data.match_stats.avg_score} 分</span>}
        {data.rule && (
          <span>
            规则「{data.rule.name}」v{data.rule.version}
          </span>
        )}
      </div>
    </div>
  );
}

/* ---------------- JD 审查 ---------------- */

type JdIssue = { dimension: string; severity: "high" | "medium" | "low"; quote: string; suggestion: string };
type JdReview = { score: number; issues: JdIssue[]; summary: string; degraded: boolean };

const SEV = {
  high: { label: "高", cls: "bg-red-50 text-red-600 border-red-200" },
  medium: { label: "中", cls: "bg-amber-50 text-amber-600 border-amber-200" },
  low: { label: "低", cls: "bg-slate-50 text-slate-500 border-slate-200" },
};

export function JobJdReview({ jobId, hasJd }: { jobId: string; hasJd: boolean }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<JdReview | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetchJSON<JdReview>("/api/ai/jd-review", {
        method: "POST",
        body: JSON.stringify({ job_id: jobId }),
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "审查失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={loading || !hasJd}
          className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--color-border-subtle)] px-3 py-1.5 text-[12px] font-body text-[var(--color-text-primary)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] disabled:opacity-40 transition-colors"
          title={hasJd ? "AI 六维合规审查" : "先填写岗位描述"}
        >
          <ShieldCheck className="w-3.5 h-3.5" />
          {loading ? "审查中…" : "AI 审查 JD"}
        </button>
        {result && !result.degraded && (
          <span
            className={`rounded-full px-2.5 py-0.5 text-[12px] font-mono font-medium ${
              result.score >= 90
                ? "bg-emerald-50 text-emerald-600"
                : result.score >= 70
                  ? "bg-amber-50 text-amber-600"
                  : "bg-red-50 text-red-600"
            }`}
          >
            {result.score} 分
          </span>
        )}
      </div>
      {error && <p className="text-[12px] text-[var(--color-danger)] font-body">{error}</p>}
      {result && (
        <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)] p-3 flex flex-col gap-2">
          <p className="text-[12px] text-[var(--color-text-secondary)] font-body">{result.summary}</p>
          {result.issues.length === 0 && !result.degraded && (
            <p className="text-[12px] text-emerald-600 font-body flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> 未发现合规问题
            </p>
          )}
          {result.issues.map((it, i) => (
            <div key={i} className={`rounded-md border p-2 ${SEV[it.severity]?.cls ?? SEV.low.cls}`}>
              <div className="flex items-center gap-2 text-[11px] font-medium">
                <span className="rounded bg-white/60 px-1.5 py-px">{SEV[it.severity]?.label}危</span>
                <span>{it.dimension}</span>
              </div>
              {it.quote && (
                <p className="mt-1 text-[11px] italic opacity-80">“{it.quote}”</p>
              )}
              <p className="mt-0.5 text-[11px]">建议:{it.suggestion}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
