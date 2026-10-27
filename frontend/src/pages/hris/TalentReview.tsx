import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GripVertical, Plus, RotateCcw, Sparkles, Trash2, Users, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useOrgOptions } from "@/pages/hris/Employees";

/* 人才盘点九宫格 + 人才池(对标北森继任云)。
   初盘=系统按绩效×潜力自动落位;校准=拖拽换格覆盖;
   一键入池=点选员工→右侧池面板加入;池内分阶段梯队。 */

type CellEmp = {
  id: string;
  name: string;
  designation: string | null;
  org_name: string | null;
  perf_score: number | null;
  potential_score: number | null;
  perf_band: number;
  potential_band: number;
  auto_perf_band: number;
  auto_potential_band: number;
  calibrated: boolean;
  estimated: boolean;
  pools: string[];
};

type Pool = {
  id: string;
  name: string;
  kind: string;
  target_designation: string | null;
  description: string | null;
  member_count: number;
  saturation: Record<string, number>;
};

type PoolDetail = Pool & {
  members: {
    employee_id: string;
    name: string;
    designation: string | null;
    stage: string;
    readiness: number | null;
    perf_band: number | null;
    potential_band: number | null;
  }[];
};

/* band 约定:X=perf 1低→3高(列,左→右);Y=potential 1低→3高(行,下→上) */
const CELL_LABEL: Record<string, { name: string; hint: string; tone: string }> = {
  "3_3": { name: "超级明星", hint: "优先任用/继任首选", tone: "border-emerald-300 bg-emerald-50/60" },
  "2_3": { name: "高潜新星", hint: "给挑战任务验证", tone: "border-emerald-200 bg-emerald-50/40" },
  "1_3": { name: "待观察高潜", hint: "查绩效瓶颈", tone: "border-amber-200 bg-amber-50/40" },
  "3_2": { name: "高绩效者", hint: "深耕专家线", tone: "border-emerald-200 bg-emerald-50/40" },
  "2_2": { name: "核心中坚", hint: "稳定输出/定向培养", tone: "border-[var(--color-border-subtle)] bg-[var(--color-surface)]" },
  "1_2": { name: "待发展", hint: "短板辅导", tone: "border-amber-200 bg-amber-50/40" },
  "3_1": { name: "绩效专家", hint: "保持产出/激励保留", tone: "border-sky-200 bg-sky-50/40" },
  "2_1": { name: "稳定贡献者", hint: "岗位适配观察", tone: "border-[var(--color-border-subtle)] bg-[var(--color-surface)]" },
  "1_1": { name: "待改进", hint: "改进计划/岗位调整", tone: "border-rose-200 bg-rose-50/40" },
};

const STAGE_LABEL: Record<string, string> = {
  in: "已入池",
  developing: "培养中",
  ready: "就绪",
  exited: "已出池",
};
const POOL_KIND: Record<string, string> = {
  succession: "继任池",
  high_potential: "高潜池",
  custom: "自定义",
};

function chipTitle(e: CellEmp) {
  const s = (v: number | null) => (v == null ? "—" : v.toFixed(2));
  return `绩效 ${s(e.perf_score)} · 潜力 ${s(e.potential_score)}${e.estimated ? " · 数据不足按中档估位" : ""}`;
}

export function TalentReview() {
  const qc = useQueryClient();
  const [orgUnitId, setOrgUnitId] = useState("");
  const [sel, setSel] = useState<CellEmp | null>(null);
  const [expandedPool, setExpandedPool] = useState<string | null>(null);
  const [err, setErr] = useState("");

  const grid = useQuery<{ cycle: string; employees: CellEmp[] }>({
    queryKey: ["nine-box", orgUnitId],
    queryFn: () =>
      fetchJSON(`/api/talent-review/nine-box${orgUnitId ? `?org_unit_id=${orgUnitId}` : ""}`),
  });
  const pools = useQuery<Pool[]>({
    queryKey: ["talent-pools"],
    queryFn: () => fetchJSON("/api/talent-pools"),
  });
  const poolDetail = useQuery<PoolDetail>({
    queryKey: ["talent-pool", expandedPool],
    queryFn: () => fetchJSON(`/api/talent-pools/${expandedPool}`),
    enabled: !!expandedPool,
  });
  const orgs = useOrgOptions();

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["nine-box"] });
    qc.invalidateQueries({ queryKey: ["talent-pools"] });
    qc.invalidateQueries({ queryKey: ["talent-pool"] });
  };

  const calibrate = useMutation({
    mutationFn: (b: { emp: string; perf_band?: number; potential_band?: number }) =>
      fetchJSON(`/api/talent-review/nine-box/${b.emp}/calibrate`, {
        method: "POST",
        body: JSON.stringify({ perf_band: b.perf_band, potential_band: b.potential_band }),
      }),
    onSuccess: () => {
      invalidate();
      setSel(null);
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "校准失败"),
  });
  const reset = useMutation({
    mutationFn: (emp: string) =>
      fetchJSON(`/api/talent-review/nine-box/${emp}/calibrate`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: invalidate,
  });
  const addToPool = useMutation({
    mutationFn: (b: { pool: string; emp: string }) =>
      fetchJSON(`/api/talent-pools/${b.pool}/members`, {
        method: "POST",
        body: JSON.stringify({ employee_ids: [b.emp] }),
      }),
    onSuccess: () => {
      invalidate();
      setSel(null);
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "入池失败"),
  });
  const stageMut = useMutation({
    mutationFn: (b: { pool: string; emp: string; stage: string }) =>
      fetchJSON(`/api/talent-pools/${b.pool}/members/${b.emp}`, {
        method: "PATCH",
        body: JSON.stringify({ stage: b.stage }),
      }),
    onSuccess: invalidate,
  });
  const removeMember = useMutation({
    mutationFn: (b: { pool: string; emp: string }) =>
      fetchJSON(`/api/talent-pools/${b.pool}/members/${b.emp}`, { method: "DELETE" }),
    onSuccess: invalidate,
  });

  const cells = useMemo(() => {
    const map: Record<string, CellEmp[]> = {};
    for (const e of grid.data?.employees ?? []) {
      const k = `${e.perf_band}_${e.potential_band}`;
      (map[k] ??= []).push(e);
    }
    return map;
  }, [grid.data]);

  const onDrop = (perf: number, pot: number) => (ev: React.DragEvent) => {
    ev.preventDefault();
    const emp = ev.dataTransfer.getData("text/plain");
    if (emp) calibrate.mutate({ emp, perf_band: perf, potential_band: pot });
  };

  return (
    <div className="flex gap-5 items-start">
      {/* ===== 九宫格 ===== */}
      <section className="flex-1 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
              人才盘点九宫格
            </h2>
            <div className="text-[12px] text-[var(--color-text-tertiary)] font-body mt-0.5">
              周期 {grid.data?.cycle ?? "…"} · 横轴绩效(低→高)· 纵轴潜力(低→高)·
              拖拽头像即校准落位
            </div>
          </div>
          <div className="w-44">
            <Select
              value={orgUnitId}
              onChange={setOrgUnitId}
              options={[
                { value: "", label: "全部组织" },
                ...orgs.map((o) => ({ value: o.value, label: o.label })),
              ]}
            />
          </div>
        </div>

        {err && <div className="text-[13px] text-[var(--color-danger)] mb-2">{err}</div>}

        {/* 潜力轴标签 + 3x3 */}
        <div className="flex">
          <div className="w-8 shrink-0 flex flex-col justify-between py-2 text-[11px] text-[var(--color-text-tertiary)] font-body">
            <span className="rotate-0">高</span>
            <span>中</span>
            <span>低</span>
          </div>
          <div className="flex-1 grid grid-cols-3 gap-2">
            {[3, 2, 1].map((pot) =>
              [1, 2, 3].map((perf) => {
                const key = `${perf}_${pot}`;
                const cell = CELL_LABEL[key];
                const emps = cells[key] ?? [];
                return (
                  <div
                    key={key}
                    data-cell={key}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDrop(perf, pot)}
                    className={`min-h-[120px] rounded-lg border-2 border-dashed p-2 flex flex-col gap-1.5 ${cell.tone}`}
                  >
                    <div className="flex items-baseline justify-between">
                      <span className="text-[12px] font-medium font-body text-[var(--color-text-primary)]">
                        {cell.name}
                      </span>
                      <span className="text-[11px] text-[var(--color-text-tertiary)]">
                        {emps.length}人
                      </span>
                    </div>
                    <div className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                      {cell.hint}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {emps.map((e) => (
                        <button
                          key={e.id}
                          draggable
                          onDragStart={(ev) =>
                            ev.dataTransfer.setData("text/plain", e.id)
                          }
                          onClick={() => setSel(e)}
                          title={chipTitle(e)}
                          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[12px] font-body bg-white cursor-grab active:cursor-grabbing ${
                            e.calibrated
                              ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                              : "border-[var(--color-border-subtle)] text-[var(--color-text-primary)]"
                          }`}
                        >
                          <GripVertical className="w-3 h-3 opacity-40" />
                          {e.name}
                          {e.estimated && <span className="opacity-50">·估</span>}
                          {e.calibrated && <span className="opacity-60">·校</span>}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
        <div className="flex mt-2 ml-8 justify-between text-[11px] text-[var(--color-text-tertiary)] font-body px-1">
          <span>绩效 低</span>
          <span>中</span>
          <span className="mr-6">高</span>
        </div>

        {/* 选中员工抽屉 */}
        {sel && (
          <div className="mt-4 rounded-lg border border-[var(--color-border-subtle)] p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="font-body text-[14px] font-medium">
                {sel.name}
                <span className="ml-2 text-[12px] text-[var(--color-text-tertiary)]">
                  {sel.designation ?? ""} {sel.org_name ? `· ${sel.org_name}` : ""}
                </span>
              </div>
              <button onClick={() => setSel(null)}>
                <X className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              </button>
            </div>
            <div className="text-[12px] font-body text-[var(--color-text-secondary)]">
              绩效 {sel.perf_score?.toFixed(2) ?? "—"} · 潜力{" "}
              {sel.potential_score?.toFixed(2) ?? "—"} · 落位{" "}
              {CELL_LABEL[`${sel.perf_band}_${sel.potential_band}`].name}
              {sel.calibrated && "(校准后)"}
              {sel.pools.length > 0 && ` · 池:${sel.pools.join("、")}`}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {(pools.data ?? []).map((p) => (
                <Button
                  key={p.id}
                  size="sm"
                  variant="secondary"
                  onClick={() => addToPool.mutate({ pool: p.id, emp: sel.id })}
                  disabled={addToPool.isPending}
                >
                  <Users className="w-3.5 h-3.5" />
                  入池·{p.name}
                </Button>
              ))}
              {sel.calibrated && (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => reset.mutate(sel.id)}
                  disabled={reset.isPending}
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  重置校准
                </Button>
              )}
            </div>
          </div>
        )}
      </section>

      {/* ===== 人才池面板 ===== */}
      <section className="w-[320px] shrink-0 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            人才池
          </h3>
          <NewPoolForm onDone={invalidate} />
        </div>
        {(pools.data ?? []).length === 0 && (
          <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
            还没有人才池 — 建一个继任池/高潜池,再从九宫格一键入池。
          </div>
        )}
        {(pools.data ?? []).map((p) => (
          <div
            key={p.id}
            className="border border-[var(--color-border-subtle)] rounded-lg p-3 flex flex-col gap-1.5"
          >
            <button
              className="flex items-center justify-between text-left"
              onClick={() => setExpandedPool(expandedPool === p.id ? null : p.id)}
            >
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium font-body">{p.name}</span>
                <span className="rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] px-2 py-0.5 text-[11px] font-body">
                  {POOL_KIND[p.kind] ?? p.kind}
                </span>
              </div>
              <span className="text-[11px] text-[var(--color-text-tertiary)]">
                {p.member_count}人
              </span>
            </button>
            {p.target_designation && (
              <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                目标岗位:{p.target_designation}
              </div>
            )}
            <div className="flex gap-2 text-[11px] font-body text-[var(--color-text-secondary)]">
              <span>在池 {p.saturation.in ?? 0}</span>
              <span>培养 {p.saturation.developing ?? 0}</span>
              <span>就绪 {p.saturation.ready ?? 0}</span>
            </div>
            {expandedPool === p.id && poolDetail.data && (
              <div className="mt-1 flex flex-col gap-1.5 border-t border-[var(--color-border-subtle)] pt-2">
                {poolDetail.data.members.length === 0 && (
                  <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                    空池 — 从九宫格点员工入池
                  </div>
                )}
                {poolDetail.data.members.map((m) => (
                  <div
                    key={m.employee_id}
                    className="flex items-center justify-between gap-2 text-[12px] font-body"
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="font-medium">{m.name}</span>
                      {m.perf_band != null && (
                        <span className="text-[10px] text-[var(--color-text-tertiary)]">
                          {m.perf_band}×{m.potential_band}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      <select
                        value={m.stage}
                        onChange={(e) =>
                          stageMut.mutate({
                            pool: p.id,
                            emp: m.employee_id,
                            stage: e.target.value,
                          })
                        }
                        className="h-7 rounded border border-[var(--color-border-subtle)] px-1.5 text-[11px] font-body bg-white"
                      >
                        {Object.entries(STAGE_LABEL).map(([v, l]) => (
                          <option key={v} value={v}>
                            {l}
                          </option>
                        ))}
                      </select>
                      <button
                        className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                        onClick={() =>
                          removeMember.mutate({ pool: p.id, emp: m.employee_id })
                        }
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </section>
    </div>
  );
}

function NewPoolForm({ onDone }: { onDone: () => void }) {
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (b: Record<string, unknown>) =>
      fetchJSON("/api/talent-pools", { method: "POST", body: JSON.stringify(b) }),
    onSuccess: () => {
      setOpen(false);
      onDone();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  if (!open)
    return (
      <Button size="sm" variant="secondary" onClick={() => setOpen(true)}>
        <Plus className="w-3.5 h-3.5" /> 建池
      </Button>
    );
  return (
    <form
      className="flex items-center gap-1.5"
      onSubmit={(e) => {
        e.preventDefault();
        const fd = new FormData(e.currentTarget);
        mut.mutate({
          name: String(fd.get("name") || "").trim(),
          kind: String(fd.get("kind") || "custom"),
          target_designation: String(fd.get("desig") || "").trim() || null,
        });
      }}
    >
      <Input name="name" required placeholder="池名" className="h-8 w-28 text-[12px]" />
      <select
        name="kind"
        className="h-8 rounded border border-[var(--color-border-subtle)] px-1 text-[12px] font-body bg-white"
      >
        <option value="succession">继任</option>
        <option value="high_potential">高潜</option>
        <option value="custom">自定义</option>
      </select>
      <Input name="desig" placeholder="目标岗位" className="h-8 w-24 text-[12px]" />
      <Button size="sm" type="submit" disabled={mut.isPending}>
        <Sparkles className="w-3.5 h-3.5" />
      </Button>
      {err && <span className="text-[11px] text-[var(--color-danger)]">{err}</span>}
    </form>
  );
}
