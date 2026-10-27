import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Brain,
  Flame,
  Lightbulb,
  MessageSquareText,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError, fetchJSON } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * AI 绩效助手配套组件(对标北森 AI 绩效助手 + Moka 离职预警):
 *
 * - FlightRiskCard   员工详情页:单人流失风险信号明细(仅 manage_employees 可见)
 * - FlightRiskBoard  绩效页顶部:组织范围风险榜
 * - FeedbackPanel    员工详情页:InsightNote 过程反馈 AI 主题/趋势分析
 * - SurveyInsights   问卷结果抽屉:答卷主题/情绪/风险/行动建议
 *
 * 403(无权限)一律静默不渲染 — 预警类信息按 Moka 实践只对 HR 可见。
 */

export type RiskSignal = {
  key: string;
  layer: string;
  label: string;
  detail: string;
  weight: number;
};

export type FlightRisk = {
  employee_id: string;
  employee_name: string;
  designation: string | null;
  org_unit: string | null;
  score: number;
  level: "low" | "medium" | "high";
  signals: RiskSignal[];
};

export type FeedbackTheme = {
  theme: string;
  support: string;
  quote: string;
  count: number | null;
  polarity: string;
};

export type FeedbackAnalysis = {
  target: string;
  sentiment: string;
  themes: FeedbackTheme[];
  risks: string[];
  actions: string[];
  trajectory: string;
  summary: string;
  response_count: number | null;
  degraded: boolean;
};

const LEVEL_STYLE: Record<string, { label: string; cls: string }> = {
  high: {
    label: "高风险",
    cls: "bg-[var(--color-danger-soft)] text-[var(--color-danger)]",
  },
  medium: {
    label: "需关注",
    cls: "bg-[var(--color-warning-soft)] text-[var(--color-warning-text)]",
  },
  low: {
    label: "低风险",
    cls: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
  },
};

const LAYER_LABEL: Record<string, string> = {
  behavior: "敬业度",
  performance: "绩效发展",
  attendance: "考勤",
  relation: "组织关系",
};

const SENTIMENT_LABEL: Record<string, { label: string; cls: string }> = {
  positive: { label: "偏正向", cls: "text-[var(--color-success)]" },
  mixed: { label: "喜忧参半", cls: "text-[var(--color-warning-text)]" },
  negative: { label: "偏负向", cls: "text-[var(--color-danger)]" },
};

const TRAJ = {
  up: { icon: TrendingUp, label: "反馈趋势向好" },
  flat: { icon: MessageSquareText, label: "反馈趋势平稳" },
  down: { icon: TrendingDown, label: "反馈趋势走弱" },
} as const;

export function FlightRiskCard({ employeeId }: { employeeId: string }) {
  const { data, error } = useQuery<FlightRisk>({
    queryKey: ["flight-risk", employeeId],
    queryFn: () => fetchJSON(`/api/flight-risk/${employeeId}`),
    retry: false,
  });
  if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
    return null;
  }
  if (!data || data.level === "low") {
    if (data?.level === "low") {
      return (
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
          <div className="flex items-center gap-2">
            <Flame className="w-4 h-4 text-[var(--color-text-tertiary)]" />
            <span className="text-sm font-medium text-[var(--color-text-primary)] font-body">
              流失风险
            </span>
            <span className={cn("px-2 py-0.5 rounded-md text-[11px] font-body", LEVEL_STYLE.low.cls)}>
              {LEVEL_STYLE.low.label}
            </span>
          </div>
        </section>
      );
    }
    return null;
  }
  const lv = LEVEL_STYLE[data.level] ?? LEVEL_STYLE.low;
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Flame className="w-4 h-4 text-[var(--color-danger)]" />
        <span className="text-sm font-medium text-[var(--color-text-primary)] font-body">
          流失风险
        </span>
        <span className={cn("px-2 py-0.5 rounded-md text-[11px] font-body", lv.cls)}>
          {lv.label} · {data.score}
        </span>
        <span className="text-[10px] text-[var(--color-text-tertiary)] font-body ml-auto">
          规则引擎 · 仅 HR 可见
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {data.signals.map((s) => (
          <div
            key={s.key}
            className="flex items-start gap-2 text-[12px] font-body border border-[var(--color-border-row)] rounded-md px-3 py-1.5"
          >
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-[var(--color-warning-text)]" />
            <div className="min-w-0">
              <span className="text-[var(--color-text-primary)]">{s.label}</span>
              <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]">
                {LAYER_LABEL[s.layer] ?? s.layer}
              </span>
              <div className="text-[11px] text-[var(--color-text-secondary)]">
                {s.detail}
              </div>
            </div>
            <span className="ml-auto font-mono text-[11px] text-[var(--color-text-tertiary)]">
              +{s.weight}
            </span>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-[var(--color-text-tertiary)] font-body">
        预警是关怀对话的触发点,请结合实际情况使用,勿作为管控名单。
      </p>
    </section>
  );
}

export function FlightRiskBoard() {
  const { data, error } = useQuery<FlightRisk[]>({
    queryKey: ["flight-risk-board"],
    queryFn: () => fetchJSON("/api/flight-risk?limit=8"),
    retry: false,
  });
  if (error instanceof ApiError && error.status === 403) return null;
  const rows = (data ?? []).filter((r) => r.level !== "low");
  if (rows.length === 0) return null;
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Flame className="w-4 h-4 text-[var(--color-danger)]" />
        <span className="text-sm font-medium text-[var(--color-text-primary)] font-body">
          流失风险预警
        </span>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          考勤 / 绩效 / 敬业度 / 组织关系四层信号 · 仅 HR 可见
        </span>
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        {rows.map((r) => {
          const lv = LEVEL_STYLE[r.level];
          return (
            <div
              key={r.employee_id}
              className="border border-[var(--color-border-row)] rounded-lg px-3 py-2 flex items-center gap-2"
            >
              <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                {r.employee_name}
              </span>
              <span className="text-[11px] text-[var(--color-text-secondary)] font-body">
                {r.designation ?? ""}
              </span>
              <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-body", lv.cls)}>
                {lv.label} {r.score}
              </span>
              <span className="ml-auto text-[10px] text-[var(--color-text-tertiary)] font-body truncate max-w-[160px]">
                {r.signals[0]?.label ?? ""}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function FeedbackPanel({ employeeId }: { employeeId: string }) {
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<FeedbackAnalysis | null>(null);
  const [err, setErr] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    try {
      const res = await fetchJSON<FeedbackAnalysis>("/api/ai/feedback-analyze", {
        method: "POST",
        body: JSON.stringify({ target: "employee", employee_id: employeeId }),
      });
      setData(res);
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) {
        setErr("没有员工反馈分析权限");
      } else {
        setErr(e instanceof ApiError ? e.message : "分析失败");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-[var(--color-accent)]" />
        <span className="text-sm font-medium text-[var(--color-text-primary)] font-body">
          反馈分析
        </span>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          聚合便签 / 复盘 / 里程碑的过程反馈
        </span>
        <Button
          variant="secondary"
          size="sm"
          className="ml-auto"
          onClick={run}
          disabled={busy}
        >
          <Sparkles className="w-3.5 h-3.5" />
          {busy ? "分析中…" : data ? "重新分析" : "AI 分析"}
        </Button>
      </div>
      {err && (
        <div className="text-[12px] text-[var(--color-danger)] font-body">{err}</div>
      )}
      {data && <FeedbackBody data={data} />}
    </section>
  );
}

function FeedbackBody({ data }: { data: FeedbackAnalysis }) {
  const sent = SENTIMENT_LABEL[data.sentiment];
  const traj = TRAJ[data.trajectory as keyof typeof TRAJ];
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3 text-[12px] font-body">
        {sent && <span className={sent.cls}>情绪:{sent.label}</span>}
        {traj && (
          <span className="flex items-center gap-1 text-[var(--color-text-secondary)]">
            <traj.icon className="w-3.5 h-3.5" />
            {traj.label}
          </span>
        )}
        {data.degraded && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-warning-soft)] text-[var(--color-warning-text)]">
            降级结果
          </span>
        )}
      </div>
      {data.summary && (
        <p className="text-[13px] text-[var(--color-text-secondary)] font-body leading-relaxed">
          {data.summary}
        </p>
      )}
      {data.themes.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {data.themes.map((t, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-[12px] font-body border border-[var(--color-border-row)] rounded-md px-3 py-1.5"
            >
              <span
                className={cn(
                  "w-2 h-2 rounded-full shrink-0",
                  t.polarity === "neg"
                    ? "bg-[var(--color-danger)]"
                    : t.polarity === "pos"
                      ? "bg-[var(--color-success)]"
                      : "bg-[var(--color-text-tertiary)]",
                )}
              />
              <span className="text-[var(--color-text-primary)]">{t.theme}</span>
              {t.count != null && (
                <span className="text-[10px] text-[var(--color-text-tertiary)]">
                  ×{t.count}
                </span>
              )}
              {t.support && (
                <span className="text-[10px] text-[var(--color-text-tertiary)]">
                  {t.support}
                </span>
              )}
              {t.quote && (
                <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)] italic truncate max-w-[220px]">
                  “{t.quote}”
                </span>
              )}
            </div>
          ))}
        </div>
      )}
      {data.risks.length > 0 && (
        <div className="flex flex-col gap-1">
          {data.risks.map((r, i) => (
            <div
              key={i}
              className="flex items-start gap-1.5 text-[12px] font-body text-[var(--color-danger)]"
            >
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              {r}
            </div>
          ))}
        </div>
      )}
      {data.actions.length > 0 && (
        <div className="flex flex-col gap-1">
          {data.actions.map((a, i) => (
            <div
              key={i}
              className="flex items-start gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]"
            >
              <Lightbulb className="w-3.5 h-3.5 mt-0.5 shrink-0 text-[var(--color-accent)]" />
              {a}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SurveyInsights({ surveyId }: { surveyId: string }) {
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<FeedbackAnalysis | null>(null);
  const [err, setErr] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    try {
      const res = await fetchJSON<FeedbackAnalysis>("/api/ai/feedback-analyze", {
        method: "POST",
        body: JSON.stringify({ target: "survey", survey_id: surveyId }),
      });
      setData(res);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "分析失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border border-[var(--color-border-subtle)] rounded-lg p-4 flex flex-col gap-2.5 bg-[var(--color-bg-subtle)]">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-[var(--color-accent)]" />
        <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
          AI 反馈分析
        </span>
        <Button
          variant="secondary"
          size="sm"
          className="ml-auto"
          onClick={run}
          disabled={busy}
        >
          <Sparkles className="w-3.5 h-3.5" />
          {busy ? "分析中…" : data ? "重新分析" : "开始分析"}
        </Button>
      </div>
      {err && (
        <div className="text-[12px] text-[var(--color-danger)] font-body">{err}</div>
      )}
      {data && <FeedbackBody data={data} />}
      {!data && !err && (
        <p className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          聚合全部答卷:提炼主题、情绪倾向、风险信号与行动建议。
        </p>
      )}
    </div>
  );
}
