import { useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  KeyRound,
  Pencil,
  Trash2,
  UserRound,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import {
  StatusPill,
  EMPLOYMENT_TYPE_LABEL,
  type Employee,
} from "@/lib/hris";
import { Field, useEmployeeOptions, useOrgOptions } from "./Employees";

const EMPLOYMENT_OPTIONS = Object.entries(EMPLOYMENT_TYPE_LABEL).map(
  ([value, label]) => ({ value, label }),
);
const STATUS_OPTIONS = [
  { value: "active", label: "在职" },
  { value: "leave", label: "休假中" },
  { value: "suspended", label: "停职" },
  { value: "terminated", label: "已离职" },
];

export function EmployeeDetail() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const [editOpen, setEditOpen] = useState(false);
  const [bindOpen, setBindOpen] = useState(false);
  const { data: e, isLoading } = useQuery<Employee>({
    queryKey: ["employee", id],
    queryFn: () => fetchJSON(`/api/employees/${id}`),
  });

  const unbind = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/employees/${id}/account`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["employee", id] }),
  });

  if (isLoading || !e) {
    return (
      <main className="p-8 max-w-[1000px] mx-auto w-full">
        <div className="text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      </main>
    );
  }

  const personal = (e.personal ?? {}) as Record<string, string>;

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <Link
        to="/employees"
        className="flex items-center gap-1.5 text-[13px] text-[var(--color-text-secondary)] font-body hover:text-[var(--color-text-primary)] w-fit"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        员工列表
      </Link>

      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] flex items-center justify-center font-heading font-semibold text-xl shrink-0">
            {e.name[0]}
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-3">
              <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
                {e.name}
              </h2>
              <StatusPill status={e.status} />
            </div>
            <div className="text-sm text-[var(--color-text-secondary)] font-body">
              {e.designation || "—"} · {e.org_unit_name || "未分配部门"} ·{" "}
              {EMPLOYMENT_TYPE_LABEL[e.employment_type] ?? e.employment_type}
            </div>
          </div>
        </div>
        <Button variant="secondary" onClick={() => setEditOpen(true)}>
          <Pencil className="w-3.5 h-3.5" />
          编辑
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            基本信息
          </h3>
          <InfoRow label="工号" value={e.employee_no} mono />
          <InfoRow label="工作邮箱" value={e.work_email} />
          <InfoRow label="手机号" value={e.phone} />
          <InfoRow label="入职日期" value={e.hire_date} />
          <InfoRow label="汇报对象" value={e.reports_to_name} />
          {Object.entries(personal).map(([k, v]) => (
            <InfoRow key={k} label={k} value={String(v)} />
          ))}
        </section>

        <div className="flex flex-col gap-4">
          <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4">
            <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
              登录账号
            </h3>
            {e.has_account ? (
              <>
                <div className="flex items-center gap-2 text-[13px] font-body text-[var(--color-text-secondary)]">
                  <UserRound className="w-4 h-4 text-[var(--color-success)]" />
                  已绑定员工自助账号 — 员工可登录查看自己的请假/考勤/工资条。
                </div>
                <div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => unbind.mutate()}
                    disabled={unbind.isPending}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    解绑账号
                  </Button>
                </div>
              </>
            ) : (
              <>
                <div className="text-[13px] font-body text-[var(--color-text-secondary)]">
                  尚未绑定登录账号。绑定后该员工可以用邮箱登录员工自助端。
                </div>
                <div>
                  <Button size="sm" onClick={() => setBindOpen(true)}>
                    <KeyRound className="w-3.5 h-3.5" />
                    绑定账号
                  </Button>
                </div>
              </>
            )}
          </section>

          <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
            <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
              快捷入口
            </h3>
            <div className="flex flex-wrap gap-2">
              <Link to={`/leave/balances?employee_id=${e.id}`}>
                <Button variant="secondary" size="sm">
                  假期额度
                </Button>
              </Link>
              <Link to={`/attendance?employee_id=${e.id}`}>
                <Button variant="secondary" size="sm">
                  考勤记录
                </Button>
              </Link>
              <Link to={`/payroll?employee_id=${e.id}`}>
                <Button variant="secondary" size="sm">
                  工资条
                </Button>
              </Link>
              {e.source_candidate_id && (
                <Link to={`/talents/${e.source_candidate_id}`}>
                  <Button variant="secondary" size="sm">
                    来源候选人
                  </Button>
                </Link>
              )}
            </div>
          </section>

          <section className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            创建于 {formatRelative(e.created_at)} · 更新于{" "}
            {formatRelative(e.updated_at)}
          </section>
        </div>
      </div>

      <EmployeeEditModal
        emp={e}
        open={editOpen}
        onClose={() => setEditOpen(false)}
      />
      <BindAccountModal
        emp={e}
        open={bindOpen}
        onClose={() => setBindOpen(false)}
      />
    </main>
  );
}

function InfoRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-3 text-[13px] font-body">
      <span className="w-20 shrink-0 text-[var(--color-text-tertiary)]">
        {label}
      </span>
      <span
        className={`text-[var(--color-text-primary)] ${mono ? "font-mono" : ""}`}
      >
        {value || "—"}
      </span>
    </div>
  );
}

function EmployeeEditModal({
  emp,
  open,
  onClose,
}: {
  emp: Employee;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const orgOptions = useOrgOptions();
  const empOptions = useEmployeeOptions().filter((o) => o.value !== emp.id);
  const [orgId, setOrgId] = useState(emp.org_unit_id ?? "");
  const [reportsTo, setReportsTo] = useState(emp.reports_to_id ?? "");
  const [empType, setEmpType] = useState(emp.employment_type);
  const [status, setStatus] = useState(emp.status);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON<Employee>(`/api/employees/${emp.id}`, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["employee", emp.id] });
      qc.invalidateQueries({ queryKey: ["employees"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    // personal 保留员工自助端填的键(可能还有 HR 手工加的),只覆盖表单里的两项
    const personal: Record<string, unknown> = { ...(emp.personal ?? {}) };
    const addr = String(fd.get("address") || "").trim();
    const emergency = String(fd.get("emergency_contact") || "").trim();
    if (addr) personal["住址"] = addr;
    else delete personal["住址"];
    if (emergency) personal["紧急联系人"] = emergency;
    else delete personal["紧急联系人"];
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      designation: String(fd.get("designation") || "").trim() || null,
      work_email: String(fd.get("work_email") || "").trim() || null,
      phone: String(fd.get("phone") || "").trim() || null,
      hire_date: String(fd.get("hire_date") || "") || null,
      org_unit_id: orgId || null,
      reports_to_id: reportsTo || null,
      employment_type: empType,
      status,
      personal,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="编辑员工"
      width={560}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="emp-edit" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="emp-edit" onSubmit={submit} className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="姓名" required>
            <Input name="name" required defaultValue={emp.name} />
          </Field>
          <Field label="状态">
            <Select
              value={status}
              onChange={setStatus}
              options={STATUS_OPTIONS}
            />
          </Field>
          <Field label="职位">
            <Input name="designation" defaultValue={emp.designation ?? ""} />
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
          <Field label="汇报对象">
            <Select
              value={reportsTo}
              onChange={setReportsTo}
              options={empOptions}
              placeholder="选择上级"
            />
          </Field>
          <Field label="入职日期">
            <Input
              name="hire_date"
              type="date"
              defaultValue={emp.hire_date ?? ""}
            />
          </Field>
          <Field label="手机号">
            <Input name="phone" defaultValue={emp.phone ?? ""} />
          </Field>
        </div>
        <Field label="工作邮箱">
          <Input
            name="work_email"
            type="email"
            defaultValue={emp.work_email ?? ""}
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="住址">
            <Input
              name="address"
              defaultValue={String((emp.personal ?? {})["住址"] ?? "")}
            />
          </Field>
          <Field label="紧急联系人">
            <Input
              name="emergency_contact"
              defaultValue={String(
                (emp.personal ?? {})["紧急联系人"] ?? "",
              )}
            />
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

function BindAccountModal({
  emp,
  open,
  onClose,
}: {
  emp: Employee;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [created, setCreated] = useState<{ email: string; password: string | null } | null>(null);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: { email: string; password?: string }) =>
      fetchJSON<{ user_id: string; email: string; password: string | null }>(
        `/api/employees/${emp.id}/account`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["employee", emp.id] });
      qc.invalidateQueries({ queryKey: ["employees"] });
      setCreated({ email: r.email, password: r.password });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "绑定失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const pwd = String(fd.get("password") || "").trim();
    mut.mutate({
      email: String(fd.get("email") || "").trim(),
      ...(pwd ? { password: pwd } : {}),
    });
  }

  function close() {
    setCreated(null);
    setErr("");
    onClose();
  }

  return (
    <Modal
      open={open}
      onClose={close}
      title="绑定登录账号"
      description="为该员工创建员工自助(employee)账号,用邮箱密码登录。已存在的邮箱会直接绑定。"
      footer={
        created ? (
          <Button onClick={close}>完成</Button>
        ) : (
          <>
            <Button variant="secondary" onClick={close}>
              取消
            </Button>
            <Button type="submit" form="emp-bind" disabled={mut.isPending}>
              {mut.isPending ? "绑定中…" : "绑定"}
            </Button>
          </>
        )
      }
    >
      {created ? (
        <div className="flex flex-col gap-3 text-[13px] font-body">
          <div className="text-[var(--color-success)]">绑定成功</div>
          <div>
            登录邮箱:
            <span className="font-mono text-[var(--color-text-primary)]">
              {" "}
              {created.email}
            </span>
          </div>
          {created.password && (
            <div className="bg-[var(--color-warning-soft)] border border-[var(--color-warning-stroke)] rounded-lg p-3">
              初始密码(仅显示一次):
              <span className="font-mono font-semibold text-[var(--color-text-primary)]">
                {" "}
                {created.password}
              </span>
            </div>
          )}
        </div>
      ) : (
        <form id="emp-bind" onSubmit={submit} className="flex flex-col gap-4">
          <Field label="登录邮箱" required>
            <Input
              name="email"
              type="email"
              required
              defaultValue={emp.work_email ?? ""}
              placeholder="zhangsan@company.com"
            />
          </Field>
          <Field label="初始密码">
            <Input
              name="password"
              type="text"
              placeholder="留空自动生成"
            />
          </Field>
          {err && (
            <div className="text-[13px] text-[var(--color-danger)] font-body">
              {err}
            </div>
          )}
        </form>
      )}
    </Modal>
  );
}
