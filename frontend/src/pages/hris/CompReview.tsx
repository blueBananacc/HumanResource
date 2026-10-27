import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchJSON, ApiError } from "@/lib/api";

/* 调薪建议矩阵(Workday merit matrix 建议侧):
   绩效带(行,高→低) × compa-ratio 带(列,低→高)→ 建议调幅;
   点员工开决策抽屉 → 调整 final_pct → 批准后锁定。不碰算薪。 */

type Emp = {
  id: string;
  name: string;
  designation: string | null;
  org_name: string | null;
  perf_band: number;
  ratio_band: string;
  gross: number | null;
  band_mid: number | null;
  compa_ratio: number | null;
  suggested_pct: number;
  estimated: boolean;
  decision_id: string | null;
  decision_status: string | null;
  final_pct: number | null;
  final_amount: number | null;
};

type MatrixOut = {
  cycle: string;
  matrix: Record<string, number>;
  stats: {
    headcount: number;
    covered: number;
    decided: number;
    avg_suggested_pct: number;
    est_cost: number;
  };
  employees: Emp[];
};

const PERF_LABEL = ["", "低绩效", "中绩效", "高绩效"];
const RATIO_LABEL: Record<string, string> = {
  low: "低于带宽",
  mid: "带宽内",
  high: "高于带宽",
};
const RATIO_HINT: Record<string, string> = {
  low: "compa<0.8",
  mid: "0.8-1.2",
  high: ">1.2",
};
const RATIOS = ["low", "mid", "high"];
const CELL_TONE: Record<number, string> = {
  3: "border-emerald-300 bg-emerald-50/60",
  2: "border-[var(--color-border-subtle)] bg-[var(--color-surface)]",
  1: "border-amber-200 bg-amber-50/40",
};

export function CompReview() {
  const qc = useQueryClient();
  const [sel, setSel] = useState<Emp | null>(null);
  const [err, setErr] = useState("");

  const q = useQuery<MatrixOut>({
    queryKey: ["comp-matrix"],
    queryFn: () => fetchJSON("/api/comp-review/matrix"),
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["comp-matrix"] });
  const cells = useMemo(() => {
    const map: Record<string, Emp[]> = {};
    for (const e of q.data?.employees ?? []) {
      (map[`${e.perf_band}_${e.ratio_band}`] ??= []).push(e);
    }
    return map;
  }, [q.data]);

  const st = q.data?.stats;
  return (
    <div className="flex flex-col gap-5">
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            调薪建议矩阵
          </h2>
          <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            周期 {q.data?.cycle ?? "…"}
          </span>
        </div>
        <div className="text-[12px] text-[var(--color-text-tertiary)] font-body mb-3">
          绩效评级 × compa-ratio(现薪/带宽中点)→ 建议调幅;点员工录入决策
        </div>
        <div className="flex gap-6 text-[13px] font-body" data-testid="comp-stats">
          <span>覆盖 <b>{st?.covered ?? 0}/{st?.headcount ?? 0}</b> 人</span>
          <span>已决策 <b>{st?.decided ?? 0}</b></span>
          <span>平均建议 <b>{st?.avg_suggested_pct ?? 0}%</b></span>
          <span className="ml-auto">预估调薪成本 <b>¥{(st?.est_cost ?? 0).toLocaleString()}</b></span>
        </div>
      </section>

      {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}

      {/* ===== 3×3 矩阵 ===== */}
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
        <div className="flex">
          <div className="w-20 shrink-0" />
          {RATIOS.map((r) => (
            <div key={r} className="flex-1 text-center pb-2">
              <div className="text-[12px] font-medium font-body text-[var(--color-text-primary)]">
                {RATIO_LABEL[r]}
              </div>
              <div className="text-[10px] text-[var(--color-text-tertiary)]">{RATIO_HINT[r]}</div>
            </div>
          ))}
        </div>
        {[3, 2, 1].map((perf) => (
          <div key={perf} className="flex mb-2">
            <div className="w-20 shrink-0 flex items-center">
              <span className="text-[12px] font-body text-[var(--color-text-secondary)]">
                {PERF_LABEL[perf]}
              </span>
            </div>
            {RATIOS.map((ratio) => {
              const key = `${perf}_${ratio}`;
              const emps = cells[key] ?? [];
              const pct = q.data?.matrix[key];
              return (
                <div
                  key={key}
                  data-cell={key}
                  className={`flex-1 min-h-[110px] rounded-lg border-2 border-dashed p-2 mr-2 last:mr-0 ${CELL_TONE[perf]}`}
                >
                  <div className="flex items-baseline justify-between">
                    <span className="text-[15px] font-semibold font-heading text-[var(--color-text-primary)]">
                      {pct != null ? `+${pct}%` : "—"}
                    </span>
                    <span className="text-[11px] text-[var(--color-text-tertiary)]">
                      {emps.length}人
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {emps.map((e) => (
                      <button
                        key={e.id}
                        onClick={() => setSel(e)}
                        data-emp={e.id}
                        className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[12px] font-body bg-white ${
                          e.decision_status === "approved"
                            ? "border-emerald-400 text-emerald-700"
                            : e.decision_id
                              ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                              : "border-[var(--color-border-subtle)] text-[var(--color-text-primary)]"
                        }`}
                      >
                        {e.name}
                        {e.estimated && <span className="opacity-50">·估</span>}
                        {e.decision_status === "approved" && <Check className="w-3 h-3 ml-0.5" />}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </section>

      {/* ===== 决策抽屉 ===== */}
      {sel && (
        <DecisionDrawer
          emp={sel}
          onClose={() => setSel(null)}
          onDone={() => { invalidate(); setSel(null); }}
          onErr={setErr}
        />
      )}
    </div>
  );
}

function DecisionDrawer({
  emp,
  onClose,
  onDone,
  onErr,
}: {
  emp: Emp;
  onClose: () => void;
  onDone: () => void;
  onErr: (s: string) => void;
}) {
  const [pct, setPct] = useState(emp.final_pct ?? emp.suggested_pct);
  const [note, setNote] = useState("");
  const decided = emp.decision_status === "approved";

  const decide = useMutation({
    mutationFn: () =>
      fetchJSON("/api/comp-review/decisions", {
        method: "POST",
        body: JSON.stringify({ employee_id: emp.id, final_pct: pct, note: note || undefined }),
      }),
    onSuccess: onDone,
    onError: (e) => onErr(e instanceof ApiError ? e.message : "决策失败"),
  });
  const approve = useMutation({
    mutationFn: (did: string) =>
      fetchJSON(`/api/comp-review/decisions/${did}/approve`, { method: "POST" }),
    onSuccess: onDone,
    onError: (e) => onErr(e instanceof ApiError ? e.message : "批准失败"),
  });

  const fmt = (v: number | null) => (v == null ? "—" : `¥${v.toLocaleString()}`);
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/20" onClick={onClose}>
      <div
        className="w-[400px] bg-white h-full p-5 flex flex-col gap-3 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="font-heading font-semibold text-[15px]">{emp.name}</h3>
          <button onClick={onClose}>
            <X className="w-4 h-4 text-[var(--color-text-tertiary)]" />
          </button>
        </div>
        <div className="text-[12px] font-body text-[var(--color-text-secondary)]">
          {emp.designation ?? "—"} · {emp.org_name ?? "未分配组织"}
        </div>
        <div className="rounded-lg border border-[var(--color-border-subtle)] p-3 text-[12px] font-body flex flex-col gap-1">
          <div>现薪 {fmt(emp.gross)} · 带宽中点 {fmt(emp.band_mid)}</div>
          <div>
            compa-ratio {emp.compa_ratio ?? "—"}({RATIO_LABEL[emp.ratio_band]})·{" "}
            {PERF_LABEL[emp.perf_band]}
          </div>
          {emp.estimated && (
            <div className="text-amber-700">⚠ 缺薪资或带宽数据,按中档估算</div>
          )}
          <div className="mt-1 text-[14px] font-semibold text-[var(--color-accent)]">
            矩阵建议 +{emp.suggested_pct}%{emp.gross ? ` ≈ +¥${Math.round(emp.gross * emp.suggested_pct / 100).toLocaleString()}` : ""}
          </div>
        </div>
        {!decided ? (
          <>
            <label className="text-[12px] font-body text-[var(--color-text-secondary)]">
              决策调幅(%)
              <Input
                type="number"
                min={0}
                max={100}
                step={0.5}
                value={pct}
                onChange={(e) => setPct(Number(e.target.value))}
                className="mt-1"
              />
            </label>
            <Input
              placeholder="备注(可选,如破格理由)"
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
            <div className="flex gap-2">
              <Button
                onClick={() => decide.mutate()}
                disabled={decide.isPending}
                data-testid="decide-btn"
              >
                {emp.decision_id ? "更新决策" : "记录决策"}
              </Button>
              {emp.decision_id && (
                <Button
                  variant="secondary"
                  onClick={() => approve.mutate(emp.decision_id!)}
                  disabled={approve.isPending}
                  data-testid="approve-btn"
                >
                  <Check className="w-4 h-4" /> 批准
                </Button>
              )}
            </div>
          </>
        ) : (
          <div className="rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-[13px] font-body text-emerald-800">
            已批准:最终调幅 +{emp.final_pct}%{emp.final_amount != null ? ` · ¥${emp.final_amount.toLocaleString()}` : ""}
          </div>
        )}
      </div>
    </div>
  );
}
