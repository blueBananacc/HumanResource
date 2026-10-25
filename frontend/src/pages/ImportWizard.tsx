import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  FileSpreadsheet,
  History,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, downloadApiFile, fetchJSON, getToken } from "@/lib/api";
import { formatRelative } from "@/lib/format";

// ---------- 类型 ----------

type ImportField = {
  key: string;
  label: string;
  required: boolean;
  kind: string;
  aliases: string[];
  choices: string[];
  sample: string;
  options: string[];
};

type Dataset = {
  key: string;
  label: string;
  subtitle: string;
  fields: ImportField[];
  allowed: boolean;
};

type MappingMeta = Record<
  string,
  { confidence: number; source: string; reason?: string }
>;

type CheckResult = {
  valid: number;
  error_count: number;
  errors: {
    row: number;
    data: Record<string, string | null>;
    messages: string[];
    by_field: Record<string, string>;
  }[];
  will_create: { orgs?: string[] };
  ai_note?: string;
};

type RunSummary = {
  created: number;
  skipped: number;
  errors: number;
  will_create?: { orgs?: string[] };
};

type Job = {
  id: string;
  dataset: string;
  dataset_label?: string;
  file_name: string;
  columns: string[];
  row_count: number;
  mapping: Record<string, string> | null;
  mapping_meta: MappingMeta | null;
  status: string;
  check_result: CheckResult | null;
  run_summary: RunSummary | null;
  created_at: string | null;
  preview?: Record<string, string | null>[];
  suggested_mapping?: Record<string, string>;
};

const STEPS = ["上传文件", "列映射", "校验预览", "执行结果"];

const SOURCE_LABEL: Record<string, string> = {
  ai: "AI",
  heuristic: "启发式",
  user: "手动",
};

const STATUS_LABEL: Record<string, string> = {
  uploaded: "已上传",
  mapped: "已映射",
  checked: "已校验",
  done: "已完成",
  failed: "失败",
};

const KIND_LABEL: Record<string, string> = {
  text: "文本",
  email: "邮箱",
  tel: "电话",
  date: "日期",
  number: "数字",
  choice: "枚举",
  suggest: "引用(部门)",
};

async function uploadFile(dataset: string, file: File): Promise<Job> {
  const fd = new FormData();
  fd.append("dataset", dataset);
  fd.append("file", file);
  const resp = await fetch("/api/imports", {
    method: "POST",
    headers: { authorization: `Bearer ${getToken() ?? ""}` },
    body: fd,
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new ApiError(resp.status, body?.detail ?? "上传失败");
  }
  return body as Job;
}

export function ImportWizard() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const fileRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState(0);
  const [datasetKey, setDatasetKey] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<MappingMeta>({});
  const [warnings, setWarnings] = useState<string[]>([]);
  const [checkResult, setCheckResult] = useState<CheckResult | null>(null);
  const [runSummary, setRunSummary] = useState<RunSummary | null>(null);
  const [err, setErr] = useState("");

  const datasetsQ = useQuery({
    queryKey: ["import-datasets"],
    queryFn: () =>
      fetchJSON<{ datasets: Dataset[]; ai_enabled: boolean; max_rows: number }>(
        "/api/imports/datasets",
      ),
  });
  const jobsQ = useQuery({
    queryKey: ["import-jobs"],
    queryFn: () => fetchJSON<Job[]>("/api/imports"),
  });

  const dataset = useMemo(
    () => datasetsQ.data?.datasets.find((d) => d.key === datasetKey) ?? null,
    [datasetsQ.data, datasetKey],
  );
  const aiEnabled = datasetsQ.data?.ai_enabled ?? false;

  const refreshJobs = () => qc.invalidateQueries({ queryKey: ["import-jobs"] });

  const uploadMut = useMutation({
    mutationFn: (file: File) => uploadFile(datasetKey, file),
    onSuccess: (j) => {
      setJob(j);
      setMapping(j.suggested_mapping ?? {});
      setMeta(
        Object.fromEntries(
          Object.keys(j.suggested_mapping ?? {}).map((k) => [
            k,
            { confidence: 0.55, source: "heuristic" },
          ]),
        ),
      );
      setWarnings([]);
      setStep(1);
      refreshJobs();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "上传失败"),
  });

  const suggestMut = useMutation({
    mutationFn: () =>
      fetchJSON<{
        mapping: Record<string, string>;
        meta: MappingMeta;
        warnings: string[];
      }>(`/api/imports/${job!.id}/suggest`, { method: "POST" }),
    onSuccess: (d) => {
      setMapping(d.mapping);
      setMeta(d.meta ?? {});
      setWarnings(d.warnings);
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "AI 映射失败"),
  });

  const checkMut = useMutation({
    mutationFn: async () => {
      await fetchJSON(`/api/imports/${job!.id}/mapping`, {
        method: "PUT",
        body: JSON.stringify({ mapping }),
      });
      return fetchJSON<CheckResult>(`/api/imports/${job!.id}/check`, {
        method: "POST",
      });
    },
    onSuccess: (r) => {
      setCheckResult(r);
      setStep(2);
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "校验失败"),
  });

  const runMut = useMutation({
    mutationFn: () =>
      fetchJSON<RunSummary>(`/api/imports/${job!.id}/run`, { method: "POST" }),
    onSuccess: (r) => {
      setRunSummary(r);
      setStep(3);
      refreshJobs();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "导入失败"),
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/imports/${id}`, { method: "DELETE" }),
    onSuccess: refreshJobs,
  });

  const requiredMissing = useMemo(
    () => (dataset?.fields ?? []).filter((f) => f.required && !mapping[f.key]),
    [dataset, mapping],
  );

  const reset = () => {
    setStep(0);
    setJob(null);
    setDatasetKey("");
    setMapping({});
    setMeta({});
    setWarnings([]);
    setCheckResult(null);
    setRunSummary(null);
    setErr("");
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-heading font-bold text-[20px] text-[var(--color-text-primary)]">
            数据导入
          </h1>
          <p className="mt-1 text-[13px] text-[var(--color-text-secondary)] font-body">
            上传 CSV / Excel,AI 自动识别列并映射到字段;校验后再写入,错误行可下载修正
          </p>
        </div>
        {step > 0 && (
          <Button variant="secondary" size="sm" onClick={reset}>
            重新开始
          </Button>
        )}
      </div>

      {/* 步骤条 */}
      <div className="flex items-center gap-2">
        {STEPS.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            {i > 0 && (
              <div className="w-8 h-px bg-[var(--color-border-subtle)]" />
            )}
            <div className="flex items-center gap-2">
              <span
                className={`w-6 h-6 rounded-full flex items-center justify-center text-[12px] font-body ${
                  i < step
                    ? "bg-[var(--color-success)] text-white"
                    : i === step
                      ? "bg-[var(--color-text-primary)] text-white"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                }`}
              >
                {i < step ? "✓" : i + 1}
              </span>
              <span
                className={`text-[13px] font-body ${
                  i === step
                    ? "font-semibold text-[var(--color-text-primary)]"
                    : "text-[var(--color-text-tertiary)]"
                }`}
              >
                {s}
              </span>
            </div>
          </div>
        ))}
      </div>

      {err && (
        <div className="rounded-lg border border-[var(--color-danger)]/30 bg-[var(--color-danger-soft,var(--color-bg-subtle))] px-4 py-2.5 text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </div>
      )}

      {/* Step 0: 选数据集 + 上传 */}
      {step === 0 && (
        <>
          <div className="grid grid-cols-2 gap-3">
            {(datasetsQ.data?.datasets ?? []).map((d) => (
              <button
                key={d.key}
                disabled={!d.allowed}
                onClick={() => setDatasetKey(d.key)}
                className={`flex items-start gap-3 rounded-xl border p-4 text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  datasetKey === d.key
                    ? "border-[var(--color-text-primary)] bg-white shadow-[var(--shadow-card)]"
                    : "border-[var(--color-border-subtle)] bg-white hover:border-[var(--color-text-tertiary)]"
                }`}
              >
                <FileSpreadsheet className="w-5 h-5 mt-0.5 text-[var(--color-brand)]" />
                <div>
                  <div className="text-[14px] font-semibold text-[var(--color-text-primary)] font-body">
                    {d.label}
                    {!d.allowed && (
                      <span className="ml-2 text-[11px] text-[var(--color-text-tertiary)]">
                        无权限
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 text-[12px] text-[var(--color-text-tertiary)] font-body">
                    {d.subtitle}
                  </div>
                </div>
              </button>
            ))}
          </div>

          {dataset && (
            <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--color-border-row)]">
                <h3 className="text-[14px] font-semibold text-[var(--color-text-primary)] font-body">
                  字段要求(带 * 必填)
                </h3>
                <button
                  className="flex items-center gap-1.5 text-[13px] text-[var(--color-brand)] font-body hover:underline"
                  onClick={() =>
                    downloadApiFile(
                      `/api/imports/datasets/${dataset.key}/template`,
                      `${dataset.key}_template.csv`,
                    )
                  }
                >
                  <Download className="w-3.5 h-3.5" /> 下载模板
                </button>
              </div>
              <table className="w-full text-[13px] font-body">
                <thead>
                  <tr className="text-left text-[12px] text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                    <th className="px-5 py-2.5 font-medium">字段</th>
                    <th className="px-5 py-2.5 font-medium">类型</th>
                    <th className="px-5 py-2.5 font-medium">可识别的表头别名</th>
                  </tr>
                </thead>
                <tbody>
                  {dataset.fields.map((f) => (
                    <tr
                      key={f.key}
                      className="border-b border-[var(--color-border-row)] last:border-0"
                    >
                      <td className="px-5 py-2.5 text-[var(--color-text-primary)]">
                        {f.label}
                        {f.required && (
                          <span className="text-[var(--color-danger)]"> *</span>
                        )}
                      </td>
                      <td className="px-5 py-2.5 text-[var(--color-text-secondary)]">
                        {KIND_LABEL[f.kind] ?? f.kind}
                      </td>
                      <td className="px-5 py-2.5 text-[var(--color-text-tertiary)]">
                        {[f.key, ...f.aliases.slice(0, 5)].join(" / ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button
                className="w-full flex items-center justify-center gap-2 px-5 py-6 text-[13px] text-[var(--color-text-secondary)] font-body border-t border-dashed border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-subtle)] transition-colors"
                onClick={() => fileRef.current?.click()}
                disabled={uploadMut.isPending}
              >
                <Upload className="w-4 h-4" />
                {uploadMut.isPending
                  ? "解析中…"
                  : "点击选择 CSV / Excel 文件(≤10MB,≤2000 行)"}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setErr("");
                      uploadMut.mutate(f);
                    }
                    e.target.value = "";
                  }}
                />
              </button>
            </div>
          )}

          {/* 历史任务 */}
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            <div className="flex items-center gap-2 px-5 py-3.5 border-b border-[var(--color-border-row)]">
              <History className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              <h3 className="text-[14px] font-semibold text-[var(--color-text-primary)] font-body">
                最近导入
              </h3>
            </div>
            {(jobsQ.data ?? []).length === 0 ? (
              <Empty
                icon={FileSpreadsheet}
                title="还没有导入任务"
                description="上传一个文件开始。"
              />
            ) : (
              (jobsQ.data ?? []).slice(0, 10).map((j) => (
                <div
                  key={j.id}
                  className="flex items-center gap-3 px-5 py-3 border-b border-[var(--color-border-row)] last:border-0"
                >
                  <div className="flex-1 min-w-0">
                    <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                      {j.file_name}
                    </span>
                    <span className="ml-2 text-[12px] text-[var(--color-text-tertiary)] font-body">
                      {j.dataset_label ?? j.dataset} · {j.row_count} 行 ·{" "}
                      {STATUS_LABEL[j.status] ?? j.status}
                      {j.run_summary ? ` · 新建 ${j.run_summary.created}` : ""}
                    </span>
                  </div>
                  {j.created_at && (
                    <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                      {formatRelative(j.created_at)}
                    </span>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      if (
                        await confirm({
                          title: "删除这条导入记录?",
                          description: j.file_name,
                          tone: "danger",
                          confirmLabel: "删除",
                        })
                      ) {
                        deleteMut.mutate(j.id);
                      }
                    }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </>
      )}

      {/* Step 1: 映射 */}
      {step === 1 && job && dataset && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--color-border-row)]">
            <h3 className="text-[14px] font-semibold text-[var(--color-text-primary)] font-body">
              列映射 — {job.file_name}({job.row_count} 行)
            </h3>
            {aiEnabled && (
              <Button
                size="sm"
                onClick={() => suggestMut.mutate()}
                disabled={suggestMut.isPending}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {suggestMut.isPending ? "AI 识别中…" : "AI 智能映射"}
              </Button>
            )}
          </div>
          {warnings.length > 0 && (
            <div className="px-5 py-3 bg-[var(--color-warning-soft,var(--color-bg-subtle))] border-b border-[var(--color-border-row)] text-[12px] text-[var(--color-warning-text)] font-body space-y-1">
              {warnings.map((w, i) => (
                <div key={i}>⚠ {w}</div>
              ))}
            </div>
          )}
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[12px] text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-5 py-2.5 font-medium w-[160px]">目标字段</th>
                <th className="px-5 py-2.5 font-medium w-[220px]">文件列</th>
                <th className="px-5 py-2.5 font-medium">样本值</th>
                <th className="px-5 py-2.5 font-medium w-[110px]">来源</th>
              </tr>
            </thead>
            <tbody>
              {dataset.fields.map((f) => {
                const col = mapping[f.key] || "";
                const m = meta?.[f.key];
                const samples = col
                  ? (job.preview ?? [])
                      .map((r) => r[col])
                      .filter(Boolean)
                      .slice(0, 2)
                  : [];
                return (
                  <tr
                    key={f.key}
                    className="border-b border-[var(--color-border-row)] last:border-0"
                  >
                    <td className="px-5 py-2.5 text-[var(--color-text-primary)]">
                      {f.label}
                      {f.required && (
                        <span className="text-[var(--color-danger)]"> *</span>
                      )}
                    </td>
                    <td className="px-5 py-1.5">
                      <Select
                        value={col}
                        onChange={(v) => {
                          setMapping((prev) => ({ ...prev, [f.key]: v }));
                          setMeta((prev) => ({
                            ...prev,
                            [f.key]: { confidence: 1, source: "user" },
                          }));
                        }}
                        options={[
                          { value: "", label: "(不映射)" },
                          ...(job.columns ?? [])
                            .filter(
                              (c) =>
                                c === col ||
                                !Object.values(mapping).includes(c),
                            )
                            .map((c) => ({ value: c, label: c })),
                        ]}
                        placeholder="(不映射)"
                      />
                    </td>
                    <td className="px-5 py-2.5 text-[var(--color-text-tertiary)]">
                      {samples.join(", ")}
                    </td>
                    <td className="px-5 py-2.5">
                      {m && col && (
                        <span
                          title={m.reason ?? ""}
                          className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                            m.source === "ai"
                              ? m.confidence >= 0.8
                                ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                                : "bg-[var(--color-warning-soft,var(--color-bg-subtle))] text-[var(--color-warning-text)]"
                              : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                          }`}
                        >
                          {SOURCE_LABEL[m.source] ?? m.source}
                          {m.source === "ai"
                            ? ` ${Math.round(m.confidence * 100)}%`
                            : ""}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {requiredMissing.length > 0 && (
            <div className="px-5 py-3 border-t border-[var(--color-border-row)] text-[12px] text-[var(--color-danger)] font-body">
              必填字段未映射:
              {requiredMissing.map((f) => f.label).join("、")}
            </div>
          )}
          <div className="flex items-center justify-between px-5 py-3.5 border-t border-[var(--color-border-row)]">
            <Button variant="secondary" size="sm" onClick={() => setStep(0)}>
              <ChevronLeft className="w-3.5 h-3.5" /> 上一步
            </Button>
            <Button
              size="sm"
              onClick={() => checkMut.mutate()}
              disabled={checkMut.isPending || requiredMissing.length > 0}
            >
              {checkMut.isPending ? "校验中…" : "下一步:校验"}
              <ChevronRight className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 2: 校验 */}
      {step === 2 && job && checkResult && dataset && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--color-border-row)]">
            <h3 className="text-[14px] font-semibold text-[var(--color-text-primary)] font-body">
              校验结果
            </h3>
            <span className="text-[13px] font-body text-[var(--color-text-secondary)]">
              可导入{" "}
              <b className="text-[var(--color-success)]">
                {checkResult.valid}
              </b>{" "}
              行 · 错误{" "}
              <b
                className={
                  checkResult.error_count
                    ? "text-[var(--color-danger)]"
                    : undefined
                }
              >
                {checkResult.error_count}
              </b>{" "}
              行
            </span>
          </div>
          {checkResult.ai_note && (
            <div className="flex items-start gap-2 px-5 py-3 border-b border-[var(--color-border-row)] text-[13px] text-[var(--color-text-secondary)] font-body">
              <Sparkles className="w-4 h-4 mt-0.5 shrink-0 text-[var(--color-brand)]" />
              {checkResult.ai_note}
            </div>
          )}
          {(checkResult.will_create?.orgs ?? []).length > 0 && (
            <div className="px-5 py-3 border-b border-[var(--color-border-row)] text-[12px] text-[var(--color-warning-text)] font-body bg-[var(--color-warning-soft,var(--color-bg-subtle))]">
              将自动新建部门:{checkResult.will_create!.orgs!.join("、")}
            </div>
          )}
          {checkResult.errors.length === 0 ? (
            <div className="px-5 py-8 text-center text-[13px] text-[var(--color-success)] font-body">
              全部 {checkResult.valid} 行校验通过,可以执行导入
            </div>
          ) : (
            <table className="w-full text-[13px] font-body">
              <thead>
                <tr className="text-left text-[12px] text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                  <th className="px-5 py-2.5 font-medium w-[70px]">行</th>
                  <th className="px-5 py-2.5 font-medium w-[220px]">数据</th>
                  <th className="px-5 py-2.5 font-medium">问题</th>
                </tr>
              </thead>
              <tbody>
                {checkResult.errors.slice(0, 50).map((e) => (
                  <tr
                    key={e.row}
                    className="border-b border-[var(--color-border-row)] last:border-0 align-top"
                  >
                    <td className="px-5 py-2.5 font-mono text-[12px] text-[var(--color-text-secondary)]">
                      {e.row}
                    </td>
                    <td className="px-5 py-2.5 text-[var(--color-text-tertiary)]">
                      {Object.values(e.data)
                        .filter(Boolean)
                        .slice(0, 2)
                        .join(" / ")}
                    </td>
                    <td className="px-5 py-2.5">
                      {e.messages.map((m, i) => (
                        <div
                          key={i}
                          className="text-[12px] text-[var(--color-danger)]"
                        >
                          {m}
                        </div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {checkResult.errors.length > 50 && (
            <div className="px-5 py-2.5 text-[12px] text-[var(--color-text-tertiary)] font-body border-t border-[var(--color-border-row)]">
              仅显示前 50 条,完整列表请下载错误 CSV
            </div>
          )}
          <div className="flex items-center justify-between px-5 py-3.5 border-t border-[var(--color-border-row)]">
            <div className="flex items-center gap-3">
              <Button variant="secondary" size="sm" onClick={() => setStep(1)}>
                <ChevronLeft className="w-3.5 h-3.5" /> 返回改映射
              </Button>
              {checkResult.error_count > 0 && (
                <button
                  className="flex items-center gap-1.5 text-[13px] text-[var(--color-brand)] font-body hover:underline"
                  onClick={() =>
                    downloadApiFile(
                      `/api/imports/${job.id}/errors.csv`,
                      "import_errors.csv",
                    )
                  }
                >
                  <Download className="w-3.5 h-3.5" /> 下载错误行
                </button>
              )}
            </div>
            <Button
              size="sm"
              onClick={() => runMut.mutate()}
              disabled={runMut.isPending || checkResult.valid === 0}
            >
              {runMut.isPending
                ? "导入中…"
                : `执行导入(${checkResult.valid} 行)`}
            </Button>
          </div>
        </div>
      )}

      {/* Step 3: 结果 */}
      {step === 3 && job && runSummary && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          <div className="flex flex-col items-center gap-2 px-5 py-10">
            <CheckCircle2 className="w-10 h-10 text-[var(--color-success)]" />
            <h2 className="text-[17px] font-semibold text-[var(--color-text-primary)] font-heading">
              导入完成
            </h2>
            <p className="text-[13px] text-[var(--color-text-secondary)] font-body">
              {job.file_name}:新建 <b>{runSummary.created}</b> 条
              {runSummary.errors > 0 && (
                <>
                  ,
                  <b className="text-[var(--color-danger)]">
                    {runSummary.errors}
                  </b>{" "}
                  行错误已跳过
                </>
              )}
            </p>
            {(runSummary.will_create?.orgs ?? []).length > 0 && (
              <p className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                已自动新建部门:{runSummary.will_create!.orgs!.join("、")}
              </p>
            )}
            <div className="flex items-center gap-3 mt-4">
              {runSummary.errors > 0 && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() =>
                    downloadApiFile(
                      `/api/imports/${job.id}/errors.csv`,
                      "import_errors.csv",
                    )
                  }
                >
                  <Download className="w-3.5 h-3.5" /> 下载错误行 CSV
                </Button>
              )}
              <Button size="sm" onClick={reset}>
                再导一个文件
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
