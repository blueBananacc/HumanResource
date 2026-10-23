import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Mail,
  Plus,
  Send,
  Undo2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { StatusPill, type Offer } from "@/lib/hris";
import { Field, PageError } from "./Employees";

const STATUS_OPTIONS = [
  { value: "", label: "全部状态" },
  { value: "draft", label: "草稿" },
  { value: "sent", label: "已发出" },
  { value: "accepted", label: "已接受" },
  { value: "declined", label: "已谢绝" },
  { value: "withdrawn", label: "已撤回" },
];

type ResumeItem = {
  id: string;
  candidate: { id: string; name: string };
};
type JobItem = { id: string; title: string };

export function Offers() {
  const [sp, setSp] = useSearchParams();
  const status = sp.get("status") ?? "";
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();

  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<Offer>({
      key: ["offers"],
      url: "/api/offers",
      params: { status },
    });
  const items = data?.items ?? [];

  const act = useMutation({
    mutationFn: ({
      id,
      action,
      body,
    }: {
      id: string;
      action: string;
      body?: Record<string, unknown>;
    }) =>
      fetchJSON(`/api/offers/${id}/${action}`, {
        method: "POST",
        body: JSON.stringify(body ?? {}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["offers"] });
      qc.invalidateQueries({ queryKey: ["employees"] });
      qc.invalidateQueries({ queryKey: ["lifecycle-events"] });
    },
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            Offer 管理
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            录用通知书 — 候选人接受后自动创建员工档案并生成入职清单。
          </p>
        </div>
        {!isError && (
          <Button onClick={() => setOpen(true)}>
            <Plus className="w-4 h-4" />
            新建 Offer
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
          条
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={Mail}
            title="还没有 Offer"
            description="从人才库的候选人创建 Offer,发出后记录候选人回应。"
            action={
              <Button onClick={() => setOpen(true)}>新建 Offer</Button>
            }
          />
        ) : (
          items.map((o, i) => (
            <div
              key={o.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {o.candidate_name}
                </span>
                <StatusPill status={o.status} />
                {o.employee_id && (
                  <Link
                    to={`/employees/${o.employee_id}`}
                    className="text-[12px] text-[var(--color-info)] font-body hover:underline"
                  >
                    → 已建档员工
                  </Link>
                )}
                <div className="flex-1" />
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {formatRelative(o.created_at)}
                </span>
              </div>
              <div className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                {o.position || o.job_title || "—"}
                {o.salary?.amount != null &&
                  ` · ${o.salary.amount} ${o.salary.currency ?? "CNY"}`}
                {o.start_date && ` · 入职 ${o.start_date}`}
              </div>
              <div className="mt-3 flex items-center gap-2">
                {o.status === "draft" && (
                  <Button
                    size="sm"
                    onClick={() => act.mutate({ id: o.id, action: "send" })}
                    disabled={act.isPending}
                  >
                    <Send className="w-3.5 h-3.5" />
                    发出
                  </Button>
                )}
                {o.status === "sent" && (
                  <>
                    <Button
                      size="sm"
                      onClick={() =>
                        act.mutate({
                          id: o.id,
                          action: "respond",
                          body: { accepted: true },
                        })
                      }
                      disabled={act.isPending}
                    >
                      <Check className="w-3.5 h-3.5" />
                      候选人接受
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() =>
                        act.mutate({
                          id: o.id,
                          action: "respond",
                          body: { accepted: false },
                        })
                      }
                      disabled={act.isPending}
                    >
                      <X className="w-3.5 h-3.5" />
                      候选人谢绝
                    </Button>
                  </>
                )}
                {(o.status === "draft" || o.status === "sent") && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      act.mutate({ id: o.id, action: "withdraw" })
                    }
                    disabled={act.isPending}
                  >
                    <Undo2 className="w-3.5 h-3.5" />
                    撤回
                  </Button>
                )}
              </div>
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

      <OfferCreateModal open={open} onClose={() => setOpen(false)} />
    </main>
  );
}

function OfferCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: resumes } = useQuery<{ items: ResumeItem[] }>({
    queryKey: ["offer-candidates"],
    queryFn: () => fetchJSON("/api/resumes/?limit=200"),
    staleTime: 30_000,
  });
  const { data: jobs } = useQuery<{ items: JobItem[] }>({
    queryKey: ["offer-jobs"],
    queryFn: () => fetchJSON("/api/jobs/?limit=200"),
    staleTime: 30_000,
  });
  const [candId, setCandId] = useState("");
  const [jobId, setJobId] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/offers", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["offers"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  const candOptions = (resumes?.items ?? []).map((r) => ({
    value: r.candidate.id,
    label: r.candidate.name,
  }));

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    if (!candId) {
      setErr("请选择候选人");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    const amount = String(fd.get("salary_amount") || "").trim();
    mut.mutate({
      candidate_id: candId,
      job_id: jobId || null,
      position: String(fd.get("position") || "").trim() || null,
      salary: amount
        ? { amount: Number(amount), currency: "CNY", period: "monthly" }
        : null,
      start_date: String(fd.get("start_date") || "") || null,
      letter: String(fd.get("letter") || "").trim() || null,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建 Offer"
      description="Offer 发出并记录「候选人接受」后,系统自动创建员工档案 + 入职清单。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="offer-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="offer-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="候选人" required>
          <Select
            value={candId}
            onChange={setCandId}
            options={candOptions}
            placeholder="选择候选人"
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="关联岗位">
            <Select
              value={jobId}
              onChange={setJobId}
              options={(jobs?.items ?? []).map((j) => ({
                value: j.id,
                label: j.title,
              }))}
              placeholder="选择岗位"
            />
          </Field>
          <Field label="录用职位">
            <Input name="position" placeholder="高级前端工程师" />
          </Field>
          <Field label="月薪 (CNY)">
            <Input name="salary_amount" type="number" min="0" placeholder="25000" />
          </Field>
          <Field label="入职日期">
            <Input name="start_date" type="date" />
          </Field>
        </div>
        <Field label="通知书正文">
          <Input name="letter" placeholder="可留空,后续补发正式 offer letter" />
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
