import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

type LogRow = {
  id: string;
  call_type: string;
  provider: string;
  model: string;
  prompt_key: string | null;
  input_digest: string;
  output: Record<string, unknown> | null;
  raw_output: string | null;
  error_kind: string;
  error_message: string | null;
  latency_ms: number;
  user_id: string | null;
  created_at: string;
};

type Page<T> = { items: T[]; total: number; limit: number; offset: number };

type Stats = {
  window_hours: number;
  total: number;
  errors: number;
  error_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  by_type: { call_type: string; count: number; errors: number; avg_latency_ms: number }[];
};

const ERR_LABEL: Record<string, string> = {
  llm_error: "调用失败",
  parse_error: "解析失败",
  schema_error: "结构不符",
  timeout: "超时",
};

const CALL_TYPES = [
  "assistant.completion", "match.evaluate", "interview.score", "interview.opening",
  "interview.report", "question.generate", "question_bank.generate",
  "question_library.generate", "scorecard.suggest", "scorecard.synthesize",
  "import.column_map", "import.check_note", "job_governance.competency",
  "job_governance.optimize_jd", "ai.field_assist", "ai.summarize", "ai.doc_draft",
  "llm.test",
];

/** 模型调用留痕(T11):每次 LLM 调用的输入指纹/输出/耗时/错误,AI 可观测性。 */
export function ModelCalls() {
  const [callType, setCallType] = useState("");
  const [errOnly, setErrOnly] = useState(false);
  const [hours, setHours] = useState(168);
  const [detail, setDetail] = useState<LogRow | null>(null);

  const params = new URLSearchParams({ limit: "100", hours: String(hours) });
  if (callType) params.set("call_type", callType);
  if (errOnly) params.set("has_error", "true");

  const { data: page, isLoading } = useQuery({
    queryKey: ["model-calls", callType, errOnly, hours],
    queryFn: () => fetchJSON<Page<LogRow>>(`/api/model-call-logs?${params}`),
    refetchInterval: 15_000,
  });
  const { data: stats } = useQuery({
    queryKey: ["model-call-stats", hours],
    queryFn: () => fetchJSON<Stats>(`/api/model-call-logs/stats?hours=${hours}`),
    refetchInterval: 15_000,
  });

  const rows = page?.items ?? [];

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-6">
      <div>
        <h1 className="font-heading text-xl text-[var(--color-text)]">模型调用</h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          每次 LLM 调用的留痕:调用类型、模型、输入指纹、输出、耗时与错误分类——
          AI 行为可观测、可回溯。
        </p>
      </div>

      {stats && (
        <div className="grid grid-cols-5 gap-3">
          <Stat label="调用总量" value={stats.total} />
          <Stat label="错误数" value={stats.errors} tone={stats.errors ? "bad" : ""} />
          <Stat label="错误率" value={`${(stats.error_rate * 100).toFixed(1)}%`} />
          <Stat label="平均延迟" value={`${stats.avg_latency_ms}ms`} />
          <Stat label="P95 延迟" value={`${stats.p95_latency_ms}ms`} />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <select
          className="rounded-lg border border-[var(--color-border-subtle)] bg-white px-3 py-1.5"
          value={callType}
          onChange={(e) => setCallType(e.target.value)}
        >
          <option value="">全部类型</option>
          {CALL_TYPES.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select
          className="rounded-lg border border-[var(--color-border-subtle)] bg-white px-3 py-1.5"
          value={hours}
          onChange={(e) => setHours(Number(e.target.value))}
        >
          <option value={24}>近 24 小时</option>
          <option value={168}>近 7 天</option>
          <option value={720}>近 30 天</option>
        </select>
        <label className="flex items-center gap-1.5 text-[var(--color-text-muted)]">
          <input
            type="checkbox"
            checked={errOnly}
            onChange={(e) => setErrOnly(e.target.checked)}
          />
          只看错误
        </label>
        <span className="ml-auto text-xs text-[var(--color-text-muted)]">
          共 {page?.total ?? 0} 条
        </span>
      </div>

      <div className="overflow-hidden rounded-2xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border-subtle)] text-left text-xs text-[var(--color-text-muted)]">
              <th className="px-4 py-3">调用类型</th>
              <th className="px-4 py-3">模型</th>
              <th className="px-4 py-3">输入指纹</th>
              <th className="px-4 py-3">耗时</th>
              <th className="px-4 py-3">状态</th>
              <th className="px-4 py-3">时间</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr><td colSpan={7} className="px-4 py-6 text-center text-[var(--color-text-muted)]">加载中…</td></tr>
            )}
            {!isLoading && !rows.length && (
              <tr><td colSpan={7} className="px-4 py-6 text-center text-[var(--color-text-muted)]">暂无调用记录</td></tr>
            )}
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-[var(--color-border-subtle)] last:border-0">
                <td className="px-4 py-3 font-mono text-xs text-[var(--color-text)]">{r.call_type}</td>
                <td className="max-w-40 truncate px-4 py-3 text-xs text-[var(--color-text-muted)]" title={r.model}>
                  {r.model || "—"}
                </td>
                <td className="px-4 py-3 font-mono text-xs text-[var(--color-text-muted)]">{r.input_digest}</td>
                <td className="px-4 py-3 text-[var(--color-text)]">{r.latency_ms}ms</td>
                <td className="px-4 py-3">
                  {r.error_kind ? (
                    <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs text-red-700">
                      {ERR_LABEL[r.error_kind] ?? r.error_kind}
                    </span>
                  ) : (
                    <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">成功</span>
                  )}
                </td>
                <td className="px-4 py-3 text-[var(--color-text-muted)]">{formatRelative(r.created_at)}</td>
                <td className="px-4 py-3 text-right">
                  <button
                    className="text-xs text-[var(--color-accent)] hover:underline"
                    onClick={() => setDetail(r)}
                  >
                    详情
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {detail && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4"
          onClick={() => setDetail(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-white p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-mono text-sm text-[var(--color-text)]">{detail.call_type}</h3>
              <button className="text-sm text-[var(--color-text-muted)]" onClick={() => setDetail(null)}>关闭</button>
            </div>
            <dl className="mb-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <Meta k="模型" v={`${detail.provider} / ${detail.model}`} />
              <Meta k="耗时" v={`${detail.latency_ms}ms`} />
              <Meta k="prompt" v={detail.prompt_key ?? "—"} />
              <Meta k="输入指纹" v={detail.input_digest} />
            </dl>
            {detail.error_kind && (
              <div className="mb-3 rounded-lg bg-red-50 p-3 text-xs text-red-700">
                <b>{ERR_LABEL[detail.error_kind] ?? detail.error_kind}</b>
                <p className="mt-1 whitespace-pre-wrap">{detail.error_message}</p>
              </div>
            )}
            {detail.output != null && (
              <>
                <p className="mb-1 text-xs font-medium text-[var(--color-text-muted)]">归一化输出</p>
                <pre className="mb-3 max-h-48 overflow-auto rounded-lg bg-[var(--color-bg-muted)] p-3 text-xs">
                  {JSON.stringify(detail.output, null, 2)}
                </pre>
              </>
            )}
            {detail.raw_output && (
              <>
                <p className="mb-1 text-xs font-medium text-[var(--color-text-muted)]">原始输出</p>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-[var(--color-bg-muted)] p-3 text-xs">
                  {detail.raw_output}
                </pre>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: number | string; tone?: string }) {
  return (
    <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white px-4 py-3 shadow-[var(--shadow-card)]">
      <p className="text-xs text-[var(--color-text-muted)]">{label}</p>
      <p className={`mt-1 text-lg font-semibold ${tone === "bad" ? "text-red-600" : "text-[var(--color-text)]"}`}>
        {value}
      </p>
    </div>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-[var(--color-text-muted)]">{k}</dt>
      <dd className="font-mono text-[var(--color-text)]">{v}</dd>
    </>
  );
}
