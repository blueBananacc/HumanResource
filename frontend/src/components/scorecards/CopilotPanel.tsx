/** AI 面试副驾(对标 Moka 面试教练·实时护航的轻量版):
 * 面试官把候选人刚说的回答贴进来 → 返回未覆盖考察维度 + 行为化追问建议。
 * LLM 不可用时后端自动降级为模板追问(degraded 标记)。
 */
import { useState } from "react";
import { Compass, Lightbulb, Sparkles } from "lucide-react";
import { fetchJSON, ApiError } from "@/lib/api";

type Followup = { dimension: string; question: string; intent: string };
type CopilotOut = {
  covered: string[];
  missing: string[];
  followups: Followup[];
  note: string;
  degraded: boolean;
};

export function CopilotPanel({
  scorecardId,
  coveredDims,
}: {
  scorecardId: string;
  /** 已被评分/提及的维度名,传给后端做覆盖判断 */
  coveredDims: string[];
}) {
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState<CopilotOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!answer.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await fetchJSON<CopilotOut>("/api/ai/interview-copilot", {
        method: "POST",
        body: JSON.stringify({
          scorecard_id: scorecardId,
          answer_text: answer.trim(),
          asked_dimensions: coveredDims,
        }),
      });
      setOut(r);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "副驾调用失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-accent-soft)]/30 p-3 flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-[11px] font-bold text-[var(--color-accent)]">
        <Compass className="w-3.5 h-3.5" /> AI 面试副驾
        <span className="font-normal text-[var(--color-text-tertiary)]">粘贴候选人回答 → 缺维提醒+追问建议</span>
      </div>
      <div className="flex gap-2">
        <textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          rows={2}
          placeholder="候选人刚才的回答(粘贴或速记要点)…"
          className="flex-1 rounded-md border border-[var(--color-border-subtle)] bg-white px-2.5 py-1.5 text-[12px] font-body text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)]"
        />
        <button
          type="button"
          onClick={run}
          disabled={loading || !answer.trim()}
          className="self-end rounded-lg bg-[var(--color-accent)] px-3 py-1.5 text-[12px] font-body text-white hover:opacity-90 disabled:opacity-40 transition-opacity"
        >
          {loading ? "分析中…" : "分析"}
        </button>
      </div>
      {error && <p className="text-[11px] text-[var(--color-danger)]">{error}</p>}
      {out && (
        <div className="flex flex-col gap-2">
          {out.missing.length > 0 && (
            <div className="text-[11px] text-[var(--color-text-secondary)]">
              未覆盖维度:
              {out.missing.map((d) => (
                <span key={d} className="ml-1 rounded bg-amber-50 text-amber-700 px-1.5 py-px">{d}</span>
              ))}
            </div>
          )}
          {out.followups.map((f, i) => (
            <div key={i} className="rounded-md bg-white border border-[var(--color-border-subtle)] p-2 flex flex-col gap-0.5">
              <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-accent)]">
                <Lightbulb className="w-3 h-3" />
                <span className="font-medium">{f.dimension}</span>
              </div>
              <p className="text-[12px] text-[var(--color-text-primary)] font-body">“{f.question}”</p>
              {f.intent && <p className="text-[10px] text-[var(--color-text-tertiary)]">{f.intent}</p>}
            </div>
          ))}
          {out.note && (
            <p className="text-[11px] text-[var(--color-text-tertiary)] flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> {out.note}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
