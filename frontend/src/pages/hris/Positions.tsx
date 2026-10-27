import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Briefcase, Plus, UserMinus, UserPlus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useOrgOptions, useEmployeeOptions } from "@/pages/hris/Employees";

/* 职位管理(Workday staffing model):职位=独立于任职人的"座位"。
   open→filled→(frozen/closed);空缺率=open/非closed;
   by_org 即北森"编制达成"的等价口径。 */

type Position = {
  id: string;
  title: string;
  designation: string | null;
  org_name: string | null;
  band_name: string | null;
  status: "open" | "filled" | "frozen" | "closed";
  incumbent_id: string | null;
  incumbent_name: string | null;
  filled_at: string | null;
  created_at: string;
};

type ListOut = {
  items: Position[];
  stats: {
    total: number;
    open: number;
    filled: number;
    frozen: number;
    closed: number;
    vacancy_rate: number;
    by_org: Record<string, { open?: number; filled?: number; frozen?: number }>;
  };
};

const STATUS_LABEL: Record<string, { l: string; cls: string }> = {
  open: { l: "空缺", cls: "bg-amber-50 text-amber-700 border-amber-200" },
  filled: { l: "在编", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  frozen: { l: "冻结", cls: "bg-sky-50 text-sky-700 border-sky-200" },
  closed: { l: "撤销", cls: "bg-gray-50 text-gray-500 border-gray-200" },
};

export function Positions() {
  const qc = useQueryClient();
  const [orgUnitId, setOrgUnitId] = useState("");
  const [status, setStatus] = useState("");
  const [fillFor, setFillFor] = useState<Position | null>(null);
  const [empId, setEmpId] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [err, setErr] = useState("");

  const q = useQuery<ListOut>({
    queryKey: ["positions", orgUnitId, status],
    queryFn: () => {
      const p = new URLSearchParams();
      if (orgUnitId) p.set("org_unit_id", orgUnitId);
      if (status) p.set("status", status);
      return fetchJSON(`/api/positions${p.size ? `?${p}` : ""}`);
    },
  });
  const orgs = useOrgOptions();
  const emps = useEmployeeOptions(!!fillFor);

  const invalidate = () => qc.invalidateQueries({ queryKey: ["positions"] });
  const onErr = (e: unknown) =>
    setErr(e instanceof ApiError ? e.message : "操作失败");

  const fill = useMutation({
    mutationFn: (b: { pid: string; employee_id: string }) =>
      fetchJSON(`/api/positions/${b.pid}/fill`, {
        method: "POST",
        body: JSON.stringify({ employee_id: b.employee_id }),
      }),
    onSuccess: () => {
      invalidate();
      setFillFor(null);
      setEmpId("");
    },
    onError: onErr,
  });
  const vacate = useMutation({
    mutationFn: (pid: string) =>
      fetchJSON(`/api/positions/${pid}/vacate`, { method: "POST" }),
    onSuccess: invalidate,
    onError: onErr,
  });
  const patch = useMutation({
    mutationFn: (b: { pid: string; status: string }) =>
      fetchJSON(`/api/positions/${b.pid}`, {
        method: "PATCH",
        body: JSON.stringify({ status: b.status }),
      }),
    onSuccess: invalidate,
    onError: onErr,
  });

  const stats = q.data?.stats;
  const rows = useMemo(() => q.data?.items ?? [], [q.data]);

  return (
    <div className="flex flex-col gap-5">
      {/* ===== 统计条 ===== */}
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
              职位管理
            </h2>
            <div className="text-[12px] text-[var(--color-text-tertiary)] font-body mt-0.5">
              职位独立于任职人 — 离职释放回空缺,空缺率反映编制缺口
            </div>
          </div>
          <Button size="sm" variant="secondary" onClick={() => setShowNew(!showNew)}>
            <Plus className="w-3.5 h-3.5" /> 新建职位
          </Button>
        </div>
        <div className="flex gap-6 text-[13px] font-body" data-testid="position-stats">
          <span>
            总编制 <b>{stats?.total ?? "…"}</b>
          </span>
          <span className="text-amber-700">
            空缺 <b>{stats?.open ?? 0}</b>
          </span>
          <span className="text-emerald-700">
            在编 <b>{stats?.filled ?? 0}</b>
          </span>
          <span className="text-sky-700">
            冻结 <b>{stats?.frozen ?? 0}</b>
          </span>
          <span className="ml-auto">
            空缺率 <b>{stats?.vacancy_rate ?? 0}%</b>
          </span>
        </div>
        {/* 编制对账(按组织) */}
        {stats && Object.keys(stats.by_org).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2" data-testid="by-org">
            {Object.entries(stats.by_org).map(([org, s]) => (
              <span
                key={org}
                className="rounded-full border border-[var(--color-border-subtle)] px-2.5 py-1 text-[11px] font-body text-[var(--color-text-secondary)]"
              >
                {org}:空缺 {s.open ?? 0} · 在编 {s.filled ?? 0}
                {(s.frozen ?? 0) > 0 ? ` · 冻结 ${s.frozen}` : ""}
              </span>
            ))}
          </div>
        )}
      </section>

      {/* ===== 新建 ===== */}
      {showNew && <NewPosition orgs={orgs} onDone={() => { setShowNew(false); invalidate(); }} />}

      {/* ===== 列表 ===== */}
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-44">
            <Select
              value={orgUnitId}
              onChange={setOrgUnitId}
              options={[{ value: "", label: "全部组织" }, ...orgs]}
            />
          </div>
          <div className="w-36">
            <Select
              value={status}
              onChange={setStatus}
              options={[
                { value: "", label: "全部状态" },
                { value: "open", label: "空缺" },
                { value: "filled", label: "在编" },
                { value: "frozen", label: "冻结" },
                { value: "closed", label: "撤销" },
              ]}
            />
          </div>
          {err && <span className="text-[12px] text-[var(--color-danger)]">{err}</span>}
        </div>

        <div className="flex flex-col gap-2" data-testid="position-list">
          {rows.length === 0 && (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body py-6 text-center">
              暂无职位 — 点「新建职位」创建编制
            </div>
          )}
          {rows.map((p) => (
            <div
              key={p.id}
              className="flex items-center gap-3 rounded-lg border border-[var(--color-border-subtle)] px-3 py-2.5"
              data-position={p.id}
            >
              <Briefcase className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium font-body">{p.title}</span>
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[11px] font-body ${STATUS_LABEL[p.status].cls}`}
                    data-status={p.status}
                  >
                    {STATUS_LABEL[p.status].l}
                  </span>
                </div>
                <div className="text-[11px] text-[var(--color-text-tertiary)] font-body mt-0.5">
                  {p.designation ?? "—"} · {p.org_name ?? "未分配组织"}
                  {p.band_name ? ` · ${p.band_name}` : ""}
                  {p.incumbent_name ? ` · 任职 ${p.incumbent_name}` : ""}
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {p.status === "open" && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => { setFillFor(p); setEmpId(""); setErr(""); }}
                  >
                    <UserPlus className="w-3.5 h-3.5" /> 分配
                  </Button>
                )}
                {p.status === "filled" && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => vacate.mutate(p.id)}
                    disabled={vacate.isPending}
                  >
                    <UserMinus className="w-3.5 h-3.5" /> 释放
                  </Button>
                )}
                {p.status === "open" && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => patch.mutate({ pid: p.id, status: "frozen" })}
                  >
                    冻结
                  </Button>
                )}
                {p.status === "frozen" && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => patch.mutate({ pid: p.id, status: "open" })}
                  >
                    解冻
                  </Button>
                )}
                {(p.status === "open" || p.status === "frozen") && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => patch.mutate({ pid: p.id, status: "closed" })}
                  >
                    <X className="w-3.5 h-3.5" /> 撤销
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ===== 分配抽屉 ===== */}
      {fillFor && (
        <div className="fixed inset-0 z-40 flex justify-end bg-black/20" onClick={() => setFillFor(null)}>
          <div
            className="w-[360px] bg-white h-full p-5 flex flex-col gap-3 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-heading font-semibold text-[15px]">分配任职人</h3>
              <button onClick={() => setFillFor(null)}>
                <X className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              </button>
            </div>
            <div className="text-[12px] font-body text-[var(--color-text-secondary)]">
              {fillFor.title} · {fillFor.org_name ?? "未分配组织"}
            </div>
            <Select
              value={empId}
              onChange={setEmpId}
              options={[{ value: "", label: "选择员工…" }, ...emps]}
            />
            <Button
              onClick={() => fill.mutate({ pid: fillFor.id, employee_id: empId })}
              disabled={!empId || fill.isPending}
            >
              <UserPlus className="w-4 h-4" /> 确认上岗
            </Button>
            {err && <div className="text-[12px] text-[var(--color-danger)]">{err}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

function NewPosition({
  orgs,
  onDone,
}: {
  orgs: { value: string; label: string }[];
  onDone: () => void;
}) {
  const [err, setErr] = useState("");
  const [orgId, setOrgId] = useState("");
  const mut = useMutation({
    mutationFn: (b: Record<string, unknown>) =>
      fetchJSON("/api/positions", { method: "POST", body: JSON.stringify(b) }),
    onSuccess: onDone,
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5">
      <form
        className="flex items-end gap-2 flex-wrap"
        onSubmit={(e) => {
          e.preventDefault();
          const fd = new FormData(e.currentTarget);
          mut.mutate({
            title: String(fd.get("title") || "").trim(),
            designation: String(fd.get("desig") || "").trim() || null,
            org_unit_id: orgId || null,
          });
        }}
      >
        <Input name="title" required placeholder="职位名称 *" className="h-9 w-44" />
        <Input name="desig" placeholder="职级/头衔" className="h-9 w-36" />
        <div className="w-44">
          <Select value={orgId} onChange={setOrgId}
            options={[{ value: "", label: "所属组织(可选)" }, ...orgs]} />
        </div>
        <Button size="sm" type="submit" disabled={mut.isPending}>创建</Button>
        {err && <span className="text-[12px] text-[var(--color-danger)]">{err}</span>}
      </form>
    </section>
  );
}
