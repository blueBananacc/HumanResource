import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, Scale, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { fmtMoney } from "@/lib/hris";
import { Field, PageError, useOrgOptions as useOrgOptionsBase } from "./Employees";

type Band = {
  id: string;
  name: string;
  designation: string | null;
  org_unit_id: string | null;
  org_unit_name: string | null;
  currency: string;
  gross_min: number;
  gross_max: number;
  employer_cost_min: number | null;
  employer_cost_max: number | null;
  effective_from: string | null;
  effective_to: string | null;
  is_active: boolean;
};

export function SalaryBands() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Band | null>(null);
  const [checkDesig, setCheckDesig] = useState("");
  const [checkAmount, setCheckAmount] = useState("");
  const qc = useQueryClient();
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<Band>({ key: ["salary-bands"], url: "/api/salary-bands" });
  const items = data?.items ?? [];

  const { data: check } = useQuery<{
    band: Band | null;
    within: boolean | null;
    amount: number;
  }>({
    queryKey: ["band-check", checkDesig, checkAmount],
    queryFn: () =>
      fetchJSON(
        `/api/salary-bands/check?amount=${checkAmount}${checkDesig ? `&designation=${encodeURIComponent(checkDesig)}` : ""}`,
      ),
    enabled: checkAmount !== "" && Number(checkAmount) > 0,
  });

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/salary-bands/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["salary-bands"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">薪酬带宽</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            职级/岗位定薪区间 — offer 详情自动校验建议薪资是否落在带宽内。
          </p>
        </div>
        {!isError && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4" />新增带宽
          </Button>
        )}
      </div>

      {/* 定薪参考检查器 */}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] px-6 py-4">
        <div className="flex items-end gap-3">
          <Field label="职级/岗位名">
            <Input value={checkDesig} onChange={(e) => setCheckDesig(e.target.value)} placeholder="如 后端工程师" className="w-[200px]" />
          </Field>
          <Field label="月薪(元)">
            <Input value={checkAmount} onChange={(e) => setCheckAmount(e.target.value)} type="number" min="0" placeholder="25000" className="w-[160px]" />
          </Field>
          {check && checkAmount && (
            <div className="pb-1.5 text-[13px] font-body">
              {check.band ? (
                check.within ? (
                  <span className="text-green-700">
                    ✓ 落在带宽「{check.band.name}」内({fmtMoney(check.band.gross_min, check.band.currency)} ~ {fmtMoney(check.band.gross_max, check.band.currency)})
                  </span>
                ) : (
                  <span className="text-amber-700">
                    ⚠ 超出带宽「{check.band.name}」({fmtMoney(check.band.gross_min, check.band.currency)} ~ {fmtMoney(check.band.gross_max, check.band.currency)})— 建议走特批
                  </span>
                )
              ) : (
                <span className="text-[var(--color-text-tertiary)]">无匹配带宽,按通用区间管理</span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={Scale}
            title="还没有薪酬带宽"
            description="按职级/岗位定义薪资参考区间,offer 定薪时给出带内/超带提示。"
            action={<Button onClick={() => setCreateOpen(true)}>新增带宽</Button>}
          />
        ) : (
          items.map((b, i) => (
            <div key={b.id} className={`px-6 py-4 ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}>
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{b.name}</span>
                {!b.is_active && <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">停用</span>}
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {b.designation ?? "通用"}{b.org_unit_name ? ` · ${b.org_unit_name}` : ""}
                </span>
                <div className="flex-1" />
                <span className="font-mono text-sm text-[var(--color-text-primary)]">
                  {fmtMoney(b.gross_min, b.currency)} ~ {fmtMoney(b.gross_max, b.currency)}
                </span>
                <Button variant="ghost" size="sm" onClick={() => setEditing(b)}>
                  <Pencil className="w-3.5 h-3.5" />
                </Button>
                <Button variant="ghost" size="sm" onClick={() => del.mutate(b.id)} disabled={del.isPending}>
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>
          ))
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {/* 条件挂载重初始化受控状态(isActive/orgId),避免沿用上次编辑残留 */}
      {(createOpen || editing) && (
        <BandModal editing={editing} onClose={() => { setCreateOpen(false); setEditing(null); }} />
      )}
    </main>
  );
}

function BandModal({ editing, onClose }: { editing: Band | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const [isActive, setIsActive] = useState(editing?.is_active ?? true);
  const [orgId, setOrgId] = useState(editing?.org_unit_id ?? "");
  const orgOptions = [{ value: "", label: "不限组织" }].concat(useOrgOptionsBase());

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      editing
        ? fetchJSON(`/api/salary-bands/${editing.id}`, { method: "PUT", body: JSON.stringify(body) })
        : fetchJSON("/api/salary-bands", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["salary-bands"] });
      qc.invalidateQueries({ queryKey: ["band-check"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const num = (k: string) => {
      const v = String(fd.get(k) || "").trim();
      return v === "" ? null : Number(v);
    };
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      designation: String(fd.get("designation") || "").trim() || null,
      org_unit_id: orgId || null,
      currency: String(fd.get("currency") || "CNY"),
      gross_min: num("gross_min") ?? 0,
      gross_max: num("gross_max") ?? 0,
      employer_cost_min: num("employer_cost_min"),
      employer_cost_max: num("employer_cost_max"),
      effective_from: String(fd.get("effective_from") || "") || null,
      effective_to: String(fd.get("effective_to") || "") || null,
      is_active: isActive,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={editing ? "编辑带宽" : "新增带宽"}
      description="匹配优先级:组织专属 > 职级精确 > 通用;同名唯一。"
      width={560}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="band-form" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="band-form" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="名称" required>
            <Input name="name" defaultValue={editing?.name} maxLength={64} required placeholder="如 后端P5" />
          </Field>
          <Field label="职级/岗位名(匹配 designation)">
            <Input name="designation" defaultValue={editing?.designation ?? ""} maxLength={128} placeholder="留空 = 通用带宽" />
          </Field>
          <Field label="所属组织">
            <Select value={orgId} onChange={setOrgId} options={orgOptions} />
          </Field>
          <Field label="币种">
            <Input name="currency" defaultValue={editing?.currency ?? "CNY"} maxLength={8} />
          </Field>
          <Field label="月薪下限" required>
            <Input name="gross_min" type="number" min="0" defaultValue={editing?.gross_min} required />
          </Field>
          <Field label="月薪上限" required>
            <Input name="gross_max" type="number" min="0" defaultValue={editing?.gross_max} required />
          </Field>
          <Field label="雇主成本下限(可选)">
            <Input name="employer_cost_min" type="number" min="0" defaultValue={editing?.employer_cost_min ?? ""} />
          </Field>
          <Field label="雇主成本上限(可选)">
            <Input name="employer_cost_max" type="number" min="0" defaultValue={editing?.employer_cost_max ?? ""} />
          </Field>
          <Field label="生效日期">
            <Input name="effective_from" type="date" defaultValue={editing?.effective_from ?? ""} />
          </Field>
          <Field label="失效日期">
            <Input name="effective_to" type="date" defaultValue={editing?.effective_to ?? ""} />
          </Field>
        </div>
        <label className="flex items-center gap-2 text-[13px] text-[var(--color-text-secondary)] font-body">
          <input type="checkbox" checked={isActive} onChange={(e) => setIsActive(e.target.checked)} />
          启用
        </label>
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}
