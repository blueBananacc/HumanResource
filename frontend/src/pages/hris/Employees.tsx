import { useState, type FormEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Lock, Network, Plus, Sparkles, Upload, Users } from "lucide-react";
import { ImportDrawer } from "@/components/records/ImportDrawer";
import { SavedFilters } from "@/components/records/SavedFilters";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { SearchInput } from "@/components/ui/search-input";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import {
  StatusPill,
  EMPLOYMENT_TYPE_LABEL,
  type Employee,
  type OrgTreeNode,
} from "@/lib/hris";

export function flattenOrgTree(
  nodes: OrgTreeNode[],
  depth = 0,
): { id: string; name: string; depth: number }[] {
  return nodes.flatMap((n) => [
    { id: n.id, name: n.name, depth },
    ...flattenOrgTree(n.children ?? [], depth + 1),
  ]);
}

export function useOrgOptions() {
  const { data } = useQuery<OrgTreeNode[]>({
    queryKey: ["org-tree"],
    queryFn: () => fetchJSON("/api/org/tree"),
    staleTime: 60_000,
  });
  return flattenOrgTree(data ?? []).map((o) => ({
    value: o.id,
    label: `${"　".repeat(o.depth)}${o.name}`,
  }));
}

/** 员工下拉选项(分配额度/定目标/生命周期事件等共用)。enabled=false 时不发请求。 */
export function useEmployeeOptions(enabled = true) {
  const { data } = useQuery<{ items: Employee[] }>({
    queryKey: ["employee-options"],
    queryFn: () => fetchJSON("/api/employees?limit=200"),
    staleTime: 30_000,
    enabled,
  });
  return (data?.items ?? []).map((e) => ({
    value: e.id,
    label: `${e.name} (${e.employee_no})`,
  }));
}

/** 列表/详情接口失败态 — 403 提示无权限,其余显示错误信息。 */
export function PageError({ error }: { error: unknown }) {
  const denied = error instanceof ApiError && error.status === 403;
  return (
    <Empty
      icon={Lock}
      title={denied ? "没有权限查看" : "加载失败"}
      description={
        denied
          ? "当前账号无权访问该模块,请联系管理员开通权限。"
          : error instanceof ApiError
            ? error.message
            : "请稍后重试。"
      }
    />
  );
}

export function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
        {label}
        {required && <span className="text-[var(--color-danger)]"> *</span>}
      </span>
      {children}
    </label>
  );
}

export function Employees() {
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const { data, isLoading, isError, error, page, pageCount, total, goto, q, setQ } =
    usePagedQuery<Employee>({ key: ["employees"], url: "/api/employees" });
  const items = data?.items ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            员工列表
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            全员花名册 — 员工档案、部门、汇报线与账号绑定。
          </p>
        </div>
        {!isError && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4" />
            新建员工
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3">
        <SearchInput
          value={q}
          onChange={setQ}
          placeholder="搜姓名 / 工号 / 职位…"
          className="w-[320px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          人
        </span>
        <div className="flex-1" />
        <Button variant="secondary" size="sm" onClick={() => setImportOpen(true)}>
          <Upload className="w-3.5 h-3.5" />
          批量导入
        </Button>
        <Link to="/settings/imports">
          <Button variant="ghost" size="sm">
            <Sparkles className="w-3.5 h-3.5" />
            智能导入
          </Button>
        </Link>
        <SavedFilters
          page="employees"
          currentParams={() => ({ q: q || undefined })}
          onApply={(p) => setQ(typeof p.q === "string" ? p.q : "")}
        />
        <Link to="/org-chart">
          <Button variant="secondary" size="sm">
            <Network className="w-3.5 h-3.5" />
            组织架构图
          </Button>
        </Link>
      </div>
      <ImportDrawer
        endpoint="/api/employees/import"
        open={importOpen}
        onClose={() => setImportOpen(false)}
        columnsHint="name(必填),employee_no,work_email,phone,designation,org_name,employment_type,hire_date"
        onDone={() => qc.invalidateQueries({ queryKey: ["employees"] })}
      />

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={Users}
            title={q ? "没有匹配的员工" : "还没有员工"}
            description={
              q
                ? "试试别的关键词,或清除搜索看全部。"
                : "创建第一位员工后,可以绑定登录账号并纳入组织架构。"
            }
            action={
              q ? undefined : (
                <Button onClick={() => setCreateOpen(true)}>新建员工</Button>
              )
            }
          />
        ) : (
          items.map((e, i) => (
            <div
              key={e.id}
              className={`flex items-center gap-4 px-6 py-4 transition-colors hover:bg-[var(--color-bg-muted)] ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <Link
                to={`/employees/${e.id}`}
                className="flex items-center gap-4 flex-1 min-w-0"
              >
                <div className="w-9 h-9 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] flex items-center justify-center font-heading font-semibold text-sm shrink-0">
                  {e.name[0]}
                </div>
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  <div className="flex items-center gap-2.5">
                    <span className="font-medium text-sm text-[var(--color-text-primary)] font-body truncate">
                      {e.name}
                    </span>
                    <span className="text-[11px] font-mono text-[var(--color-text-tertiary)]">
                      {e.employee_no}
                    </span>
                    <StatusPill status={e.status} />
                    {!e.has_account && (
                      <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                        未开账号
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[12px] text-[var(--color-text-secondary)] font-body">
                    <span>{e.designation || "—"}</span>
                    <span className="text-[var(--color-text-tertiary)]">·</span>
                    <span>{e.org_unit_name || "未分配部门"}</span>
                    {e.reports_to_name && (
                      <>
                        <span className="text-[var(--color-text-tertiary)]">
                          ·
                        </span>
                        <span>汇报: {e.reports_to_name}</span>
                      </>
                    )}
                    <span className="text-[var(--color-text-tertiary)]">·</span>
                    <span>
                      {EMPLOYMENT_TYPE_LABEL[e.employment_type] ??
                        e.employment_type}
                    </span>
                  </div>
                </div>
                {e.hire_date && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body shrink-0">
                    入职 {e.hire_date}
                  </span>
                )}
              </Link>
            </div>
          ))
        )}
      </div>

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />

      <EmployeeCreateModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
      />
    </main>
  );
}

const EMPLOYMENT_OPTIONS = Object.entries(EMPLOYMENT_TYPE_LABEL).map(
  ([value, label]) => ({ value, label }),
);

function EmployeeCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const orgOptions = useOrgOptions();
  const [orgId, setOrgId] = useState("");
  const [empType, setEmpType] = useState("full_time");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON<Employee>("/api/employees", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["employees"] });
      qc.invalidateQueries({ queryKey: ["employee-options"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      employee_no: String(fd.get("employee_no") || "").trim() || null,
      designation: String(fd.get("designation") || "").trim() || null,
      work_email: String(fd.get("work_email") || "").trim() || null,
      phone: String(fd.get("phone") || "").trim() || null,
      hire_date: String(fd.get("hire_date") || "") || null,
      org_unit_id: orgId || null,
      employment_type: empType,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建员工"
      description="工号留空自动编号;创建后可在详情页绑定登录账号。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="emp-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="emp-create" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="姓名" required>
            <Input name="name" required placeholder="张三" />
          </Field>
          <Field label="工号">
            <Input name="employee_no" placeholder="留空自动编号" />
          </Field>
          <Field label="职位">
            <Input name="designation" placeholder="前端工程师" />
          </Field>
          <Field label="用工类型">
            <Select
              value={empType}
              onChange={setEmpType}
              options={EMPLOYMENT_OPTIONS}
            />
          </Field>
          <Field label="部门">
            <Select
              value={orgId}
              onChange={setOrgId}
              options={orgOptions}
              placeholder="选择部门"
            />
          </Field>
          <Field label="入职日期">
            <Input name="hire_date" type="date" />
          </Field>
          <Field label="工作邮箱">
            <Input name="work_email" type="email" />
          </Field>
          <Field label="手机号">
            <Input name="phone" />
          </Field>
        </div>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
