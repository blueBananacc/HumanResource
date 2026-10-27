import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Goal as GoalIcon, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { StatusPill, type Goal } from "@/lib/hris";
import { Field, useEmployeeOptions } from "./Employees";
import { FlightRiskBoard } from "@/components/people/PeopleInsights";
import { Sparkles } from "lucide-react";

const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "active", label: "进行中" },
  { value: "done", label: "已完成" },
  { value: "dropped", label: "已放弃" },
];
const GOAL_STATUS_LABEL: Record<string, string> = {
  active: "进行中",
  done: "已完成",
  dropped: "已放弃",
};

export function Performance() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_PERFORMANCE);

  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<Goal>({
      key: ["goals"],
      url: "/api/performance/goals",
      params: { status },
    });
  const items = data?.items ?? [];

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/performance/goals/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["goals"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            目标管理
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            员工个人目标 — 进度由员工自助端更新,HR 在这里统览。
          </p>
        </div>
        {canManage && (
          <Button onClick={() => setOpen(true)}>
            <Plus className="w-4 h-4" />
            定目标
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Select
          value={status}
          onChange={(v) =>
            setSp((prev) => {
              const next = new URLSearchParams(prev);
              if (v) next.set("status", v);
              else next.delete("status");
              next.delete("page");
              return next;
            })
          }
          options={STATUS_OPTIONS}
          className="w-[140px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          个
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={GoalIcon}
            title="还没有目标"
            description="为员工创建目标,员工在自助端更新进度。"
            action={
              canManage ? (
                <Button onClick={() => setOpen(true)}>定目标</Button>
              ) : undefined
            }
          />
        ) : (
          items.map((g, i) => (
            <div
              key={g.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {g.title}
                </span>
                <StatusPill status={g.status} label={GOAL_STATUS_LABEL[g.status]} />
                <div className="flex-1" />
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {g.employee_name}
                </span>
                {g.due_date && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    截止 {g.due_date}
                  </span>
                )}
                {canManage && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => del.mutate(g.id)}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                )}
              </div>
              {g.description && (
                <div className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                  {g.description}
                </div>
              )}
              <div className="mt-2 flex items-center gap-3">
                <div className="flex-1 h-2 rounded-full bg-[var(--color-bg-subtle)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--color-accent)] transition-all"
                    style={{ width: `${g.progress}%` }}
                  />
                </div>
                <span className="text-[12px] font-mono text-[var(--color-text-secondary)] w-10 text-right">
                  {g.progress}%
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      <FlightRiskBoard />

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />

      <GoalCreateModal open={open} onClose={() => setOpen(false)} />
    </main>
  );
}

type GoalSuggestion = { title: string; description: string };

function GoalCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  // 打开弹窗时才拉员工列表 — 该页面 view_reports 可看,无 manage 权限的用户不应触发 /api/employees
  const empOptions = useEmployeeOptions(open);
  const [empId, setEmpId] = useState("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [suggestions, setSuggestions] = useState<GoalSuggestion[] | null>(null);
  const [suggestErr, setSuggestErr] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/performance/goals", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["goals"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  const suggest = useMutation({
    mutationFn: () =>
      fetchJSON<{ goals: GoalSuggestion[]; degraded: boolean }>(
        "/api/ai/goal-suggest",
        { method: "POST", body: JSON.stringify({ employee_id: empId }) },
      ),
    onSuccess: (d) => {
      setSuggestions(d.goals);
      setSuggestErr(d.degraded ? "AI 暂不可用,已给出通用骨架建议" : "");
    },
    onError: (e) =>
      setSuggestErr(e instanceof ApiError ? e.message : "建议生成失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    if (!empId) {
      setErr("请选择员工");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      employee_id: empId,
      title: title.trim(),
      description: description.trim(),
      due_date: String(fd.get("due_date") || "") || null,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="定目标"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="goal-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="goal-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="员工" required>
          <Select
            value={empId}
            onChange={(v) => {
              setEmpId(v);
              setSuggestions(null);
            }}
            options={empOptions}
            placeholder="选择员工"
          />
        </Field>
        {empId && (
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={() => suggest.mutate()}
              disabled={suggest.isPending}
              className="self-start flex items-center gap-1.5 text-[12px] text-[var(--color-accent)] font-body disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5" />
              {suggest.isPending ? "生成建议中…" : "AI 建议目标(SMART)"}
            </button>
            {suggestErr && (
              <div className="text-[11px] text-[var(--color-warning-text)] font-body">
                {suggestErr}
              </div>
            )}
            {suggestions && (
              <div className="flex flex-col gap-1.5">
                {suggestions.map((g, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => {
                      setTitle(g.title);
                      setDescription(g.description);
                    }}
                    className="text-left border border-[var(--color-border-subtle)] rounded-md px-3 py-2 hover:border-[var(--color-accent)] transition-colors"
                  >
                    <div className="text-[12px] font-medium text-[var(--color-text-primary)] font-body">
                      {g.title}
                    </div>
                    <div className="text-[11px] text-[var(--color-text-secondary)] font-body">
                      {g.description}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        <Field label="目标" required>
          <Input
            name="title"
            required
            placeholder="Q3 完成新版官网上线"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </Field>
        <Field label="描述">
          <Input
            name="description"
            placeholder="衡量标准 / 里程碑"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </Field>
        <Field label="截止日期">
          <Input name="due_date" type="date" />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
