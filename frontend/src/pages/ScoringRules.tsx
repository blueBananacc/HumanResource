import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Sparkles, Trash2, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { ApiError, fetchJSON } from "@/lib/api";

type Evaluator = {
  method: "keyword" | "duration" | "ai" | "boolean" | "range" | "coverage";
  keywords?: string[];
  matchMode?: "any" | "all";
  minYears?: number;
  preferredYears?: number;
  aiPrompt?: string;
  booleanField?: string;
  field?: string;
  minValue?: number;
  maxValue?: number;
};
type Dimension = {
  id: string;
  name: string;
  weight: number;
  description?: string;
  evaluator: Evaluator;
};
type Rule = {
  id: string;
  name: string;
  description: string | null;
  dimensions: Dimension[];
  version: number;
  weight_total: number;
  job_count: number;
};
type Job = { id: string; title: string; status: string };
type Analysis = {
  summary: string;
  coverageGaps: string[];
  weightAdjustments: string[];
  industrySignals: string[];
  recommendedChanges: string[];
  proposedRule: { name: string; description: string; dimensions: Dimension[] };
};

const METHOD_LABEL: Record<string, string> = {
  keyword: "关键词",
  coverage: "关键词覆盖",
  duration: "年限",
  ai: "AI 评估",
  boolean: "有无",
  range: "数值区间",
};
const BOOL_FIELDS = [
  { value: "education", label: "教育背景" },
  { value: "experience", label: "工作经历" },
  { value: "projects", label: "项目经验" },
  { value: "skills", label: "技能列表" },
];

let dimSeq = 0;
const newDim = (): Dimension => ({
  id: `dim-${Date.now()}-${dimSeq++}`,
  name: "",
  weight: 25,
  description: "",
  evaluator: { method: "keyword", keywords: [], matchMode: "any" },
});

/** T13 声明式评分规则管理:维度×评估器,确定性方法零成本,AI 维批量评估。 */
export function ScoringRules() {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<Rule | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: rules } = useQuery({
    queryKey: ["scoring-rules"],
    queryFn: () => fetchJSON<Rule[]>("/api/scoring-rules"),
  });
  const { data: jobsPage } = useQuery({
    queryKey: ["jobs-for-rules"],
    queryFn: () => fetchJSON<{ items: Job[] }>("/api/jobs/?limit=100"),
  });
  const jobs = jobsPage?.items ?? [];

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/scoring-rules/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scoring-rules"] }),
  });

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-heading text-xl text-[var(--color-text)]">评分规则</h1>
          <p className="mt-1 text-sm text-[var(--color-text-muted)]">
            声明式维度×评估器(关键词/年限/有无/区间零成本,AI 维按证据定级);挂到岗位后匹配评估走该规则。
          </p>
        </div>
        <Button
          onClick={() => {
            setError(null);
            setEditing({
              id: "",
              name: "",
              description: "",
              dimensions: [newDim()],
              version: 1,
              weight_total: 0,
              job_count: 0,
            });
          }}
        >
          <Plus className="mr-1 h-4 w-4" /> 新建规则
        </Button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {(rules ?? []).map((r) => (
          <li
            key={r.id}
            className="flex items-center justify-between rounded-2xl border border-[var(--color-border-subtle)] bg-white px-4 py-3 shadow-[var(--shadow-card)]"
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-[var(--color-text)]">{r.name}</span>
                <span className="rounded-full bg-[var(--color-bg-muted)] px-2 py-0.5 text-[10px] text-[var(--color-text-muted)]">
                  v{r.version}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] ${
                    r.weight_total === 100
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-amber-50 text-amber-700"
                  }`}
                >
                  权重 {r.weight_total}
                </span>
              </div>
              <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
                {r.dimensions.length} 维 ·{" "}
                {r.dimensions.map((d) => METHOD_LABEL[d.evaluator.method]).join("/")} · 挂{" "}
                {r.job_count} 岗位
              </p>
            </div>
            <div className="flex items-center gap-1">
              <Button variant="secondary" onClick={() => { setError(null); setEditing(structuredClone(r)); }}>
                编辑
              </Button>
              <button
                type="button"
                className="p-1.5 text-[var(--color-text-muted)] hover:text-red-600"
                onClick={() => del.mutate(r.id)}
                aria-label="删除规则"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </li>
        ))}
        {(rules ?? []).length === 0 && (
          <p className="rounded-2xl border border-dashed border-[var(--color-border-subtle)] p-8 text-center text-sm text-[var(--color-text-muted)]">
            还没有评分规则 — 点击「新建规则」,或让 AI 按岗位生成
          </p>
        )}
      </ul>

      {editing && (
        <RuleEditor
          rule={editing}
          jobs={jobs}
          onClose={() => setEditing(null)}
          onError={setError}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 规则编辑器(右下浮层,非模态以保留岗位参照)
// ---------------------------------------------------------------------------

function RuleEditor({
  rule,
  jobs,
  onClose,
  onError,
}: {
  rule: Rule;
  jobs: Job[];
  onClose: () => void;
  onError: (e: string | null) => void;
}) {
  const qc = useQueryClient();
  const isNew = !rule.id;
  const [name, setName] = useState(rule.name);
  const [desc, setDesc] = useState(rule.description ?? "");
  const [dims, setDims] = useState<Dimension[]>(rule.dimensions);
  const [genJobId, setGenJobId] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);

  const weightTotal = useMemo(() => dims.reduce((s, d) => s + (d.weight || 0), 0), [dims]);

  const save = useMutation({
    mutationFn: () =>
      fetchJSON<Rule>(isNew ? "/api/scoring-rules" : `/api/scoring-rules/${rule.id}`, {
        method: isNew ? "POST" : "PUT",
        body: JSON.stringify({ name, description: desc, dimensions: dims }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["scoring-rules"] });
      onClose();
    },
    onError: (e) => onError(e instanceof ApiError ? e.message : "保存失败"),
  });

  const generate = useMutation({
    mutationFn: () =>
      fetchJSON<{ name: string; description: string; dimensions: Dimension[] }>(
        "/api/scoring-rules/generate",
        { method: "POST", body: JSON.stringify({ job_id: genJobId }) },
      ),
    onSuccess: (d) => {
      if (!name) setName(d.name);
      if (!desc) setDesc(d.description);
      setDims(d.dimensions);
    },
    onError: (e) => onError(e instanceof ApiError ? e.message : "AI 生成失败"),
  });

  const analyze = useMutation({
    mutationFn: () => fetchJSON<Analysis>(`/api/scoring-rules/${rule.id}/analyze`, { method: "POST" }),
    onSuccess: setAnalysis,
    onError: (e) => onError(e instanceof ApiError ? e.message : "分析失败(规则需先挂岗位)"),
  });

  const apply = useMutation({
    mutationFn: () =>
      fetchJSON<Rule>(`/api/scoring-rules/${rule.id}/apply-analysis`, {
        method: "POST",
        body: JSON.stringify({ analysis }),
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["scoring-rules"] });
      setName(r.name);
      setDesc(r.description ?? "");
      setDims(r.dimensions);
      setAnalysis(null);
    },
    onError: (e) => onError(e instanceof ApiError ? e.message : "应用失败"),
  });

  const patchDim = (id: string, patch: Partial<Dimension>) =>
    setDims((ds) => ds.map((d) => (d.id === id ? { ...d, ...patch } : d)));
  const patchEv = (id: string, patch: Partial<Evaluator>) =>
    setDims((ds) =>
      ds.map((d) => (d.id === id ? { ...d, evaluator: { ...d.evaluator, ...patch } } : d)),
    );

  return (
    <div className="fixed inset-x-0 bottom-0 top-16 z-40 overflow-auto bg-[var(--color-bg)]/95 p-6 backdrop-blur-sm">
      <div className="mx-auto max-w-3xl space-y-4 rounded-2xl border border-[var(--color-border-subtle)] bg-white p-5 shadow-[var(--shadow-card)]">
        <div className="flex items-center justify-between">
          <h2 className="font-heading text-lg text-[var(--color-text)]">
            {isNew ? "新建评分规则" : `编辑「${rule.name}」`}
          </h2>
          <button className="text-sm text-[var(--color-text-muted)]" onClick={onClose}>
            关闭
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">规则名</label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="如:后端工程师评分规则" />
          </div>
          <div>
            <label className="mb-1 block text-xs text-[var(--color-text-muted)]">说明</label>
            <Input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="适用岗位/场景" />
          </div>
        </div>

        {/* AI 生成 */}
        <div className="flex items-center gap-2 rounded-xl bg-[var(--color-bg-muted)] p-3">
          <div className="w-64">
            <Select
              value={genJobId}
              onChange={setGenJobId}
              options={[
                { value: "", label: "选择岗位…" },
                ...jobs.map((j) => ({ value: j.id, label: j.title })),
              ]}
            />
          </div>
          <Button
            variant="secondary"
            disabled={!genJobId || generate.isPending}
            onClick={() => generate.mutate()}
          >
            <Sparkles className="mr-1 h-4 w-4" />
            {generate.isPending ? "生成中…" : "按岗位 AI 生成维度"}
          </Button>
          {!isNew && (
            <Button
              variant="secondary"
              disabled={analyze.isPending}
              onClick={() => analyze.mutate()}
            >
              <Wand2 className="mr-1 h-4 w-4" />
              {analyze.isPending ? "分析中…" : "AI 审查此规则"}
            </Button>
          )}
        </div>

        {/* 分析结果 */}
        {analysis && (
          <div className="space-y-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm">
            <p className="font-medium text-amber-900">{analysis.summary}</p>
            {[
              ["覆盖缺口", analysis.coverageGaps],
              ["权重建议", analysis.weightAdjustments],
              ["建议动作", analysis.recommendedChanges],
            ].map(([label, arr]) =>
              (arr as string[]).length ? (
                <div key={label as string}>
                  <span className="text-xs font-medium text-amber-800">{label}:</span>
                  <ul className="ml-4 list-disc text-xs text-amber-800">
                    {(arr as string[]).map((x, i) => (
                      <li key={i}>{x}</li>
                    ))}
                  </ul>
                </div>
              ) : null,
            )}
            <div className="pt-1">
              <Button
                variant="secondary"
                disabled={apply.isPending}
                onClick={() => apply.mutate()}
              >
                {apply.isPending ? "应用中…" : `应用 AI 草案(${analysis.proposedRule.dimensions.length} 维)`}
              </Button>
              <button className="ml-3 text-xs text-amber-800 underline" onClick={() => setAnalysis(null)}>
                忽略
              </button>
            </div>
          </div>
        )}

        {/* 维度编辑器 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--color-text)]">
              维度({dims.length})
              <span
                className={`ml-2 rounded-full px-2 py-0.5 text-[10px] ${
                  weightTotal === 100 ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                }`}
              >
                权重合计 {weightTotal}
              </span>
            </h3>
            <Button variant="secondary" onClick={() => setDims((ds) => [...ds, newDim()])}>
              <Plus className="mr-1 h-3.5 w-3.5" /> 加维度
            </Button>
          </div>

          {dims.map((d) => (
            <div key={d.id} className="rounded-xl border border-[var(--color-border-subtle)] p-3">
              <div className="flex items-center gap-2">
                <Input
                  className="w-44"
                  value={d.name}
                  onChange={(e) => patchDim(d.id, { name: e.target.value })}
                  placeholder="维度名"
                />
                <div className="flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
                  权重
                  <Input
                    className="w-16"
                    type="number"
                    value={d.weight}
                    onChange={(e) => patchDim(d.id, { weight: Number(e.target.value) || 0 })}
                  />
                </div>
                <div className="w-32">
                  <Select
                    value={d.evaluator.method}
                    onChange={(m) =>
                      patchEv(d.id, { method: m as Evaluator["method"] })
                    }
                    options={Object.entries(METHOD_LABEL).map(([value, label]) => ({ value, label }))}
                  />
                </div>
                <button
                  className="ml-auto p-1 text-[var(--color-text-muted)] hover:text-red-600"
                  onClick={() => setDims((ds) => ds.filter((x) => x.id !== d.id))}
                  aria-label="删除维度"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
              <div className="mt-2">
                <DimParams d={d} onEv={(p) => patchEv(d.id, p)} />
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button
            disabled={save.isPending || !name.trim() || dims.length === 0}
            onClick={() => save.mutate()}
          >
            {save.isPending ? "保存中…" : "保存规则"}
          </Button>
        </div>
      </div>
    </div>
  );
}

function DimParams({ d, onEv }: { d: Dimension; onEv: (p: Partial<Evaluator>) => void }) {
  const m = d.evaluator.method;
  if (m === "keyword") {
    return (
      <div className="flex items-center gap-2">
        <Input
          className="flex-1"
          defaultValue={(d.evaluator.keywords ?? []).join(", ")}
          placeholder="关键词,逗号分隔(命中比例计分)"
          onBlur={(e) =>
            onEv({ keywords: e.target.value.split(/[,,\s]+/).map((s) => s.trim()).filter(Boolean) })
          }
        />
        <div className="w-32">
          <Select
            value={d.evaluator.matchMode ?? "any"}
            onChange={(v) => onEv({ matchMode: v as "any" | "all" })}
            options={[
              { value: "any", label: "任一命中" },
              { value: "all", label: "全部命中" },
            ]}
          />
        </div>
      </div>
    );
  }
  if (m === "duration") {
    return (
      <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
        最低
        <Input className="w-16" type="number" value={d.evaluator.minYears ?? 0}
          onChange={(e) => onEv({ minYears: Number(e.target.value) || 0 })} />
        年 → 理想
        <Input className="w-16" type="number" value={d.evaluator.preferredYears ?? 0}
          onChange={(e) => onEv({ preferredYears: Number(e.target.value) || 0 })} />
        年(线性插值)
      </div>
    );
  }
  if (m === "boolean") {
    return (
      <div className="w-48">
        <Select
          value={d.evaluator.booleanField ?? "skills"}
          onChange={(v) => onEv({ booleanField: v })}
          options={BOOL_FIELDS}
        />
      </div>
    );
  }
  if (m === "range") {
    return (
      <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)]">
        经验年限
        <Input className="w-16" type="number" value={d.evaluator.minValue ?? 0}
          onChange={(e) => onEv({ minValue: Number(e.target.value) || 0 })} />
        ~
        <Input className="w-16" type="number" value={d.evaluator.maxValue ?? 100}
          onChange={(e) => onEv({ maxValue: Number(e.target.value) || 0 })} />
        年映射 0-100
      </div>
    );
  }
  return (
    <Input
      value={d.evaluator.aiPrompt ?? ""}
      placeholder="AI 评估要点(如:评估项目深度、领域契合)"
      onChange={(e) => onEv({ aiPrompt: e.target.value })}
    />
  );
}
