import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Brain,
  ClipboardCheck,
  Lightbulb,
  Lock,
  Play,
  Plus,
  Star,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { Pagination } from "@/components/ui/pagination";
import {
  StatusPill,
  type Appraisal,
  type AppraisalCycle,
} from "@/lib/hris";
import { Field, useEmployeeOptions } from "./Employees";

const CYCLE_STATUS_LABEL: Record<string, string> = {
  draft: "草稿",
  open: "进行中",
  closed: "已关闭",
};
const APPRAISAL_STATUS_LABEL: Record<string, string> = {
  draft: "待自评",
  self_submitted: "待上级评",
  reviewed: "已完成",
};

export function AppraisalCycles() {
  const qc = useQueryClient();
  const [cycleOpen, setCycleOpen] = useState(false);
  const [assignOpen, setAssignOpen] = useState<AppraisalCycle | null>(null);
  const [reviewing, setReviewing] = useState<Appraisal | null>(null);
  const [cycleFilter, setCycleFilter] = useState("");

  const { data: cycles, isLoading: cyclesLoading } = useQuery<
    AppraisalCycle[]
  >({
    queryKey: ["appraisal-cycles"],
    queryFn: () => fetchJSON("/api/performance/cycles"),
  });
  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<Appraisal>({
      key: ["appraisals"],
      url: "/api/performance/appraisals",
      params: { cycle_id: cycleFilter },
    });
  const items = data?.items ?? [];

  const cycleAct = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "open" | "close" }) =>
      fetchJSON(`/api/performance/cycles/${id}/${action}`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["appraisal-cycles"] }),
  });

  const cycleOptions = (cycles ?? []).map((c) => ({
    value: c.id,
    label: c.name,
  }));

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            绩效考核
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            考核周期 → 发起考核 → 员工自评 → 上级评分。
          </p>
        </div>
        <Button onClick={() => setCycleOpen(true)}>
          <Plus className="w-4 h-4" />
          新建周期
        </Button>
      </div>

      <div className="flex flex-wrap gap-3">
        {cyclesLoading ? (
          <div className="text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (cycles ?? []).length === 0 ? (
          <div className="text-sm text-[var(--color-text-tertiary)] font-body">
            还没有考核周期 — 先建一个(如「2025 年度考核」)。
          </div>
        ) : (
          (cycles ?? []).map((c) => (
            <div
              key={c.id}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl border bg-white shadow-[var(--shadow-card)] ${
                cycleFilter === c.id
                  ? "border-[var(--color-accent)]"
                  : "border-[var(--color-border-subtle)]"
              }`}
            >
              <button
                onClick={() =>
                  setCycleFilter(cycleFilter === c.id ? "" : c.id)
                }
                className="text-left"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                    {c.name}
                  </span>
                  <StatusPill status={c.status} label={CYCLE_STATUS_LABEL[c.status]} />
                </div>
                <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {c.period_start} ~ {c.period_end} · {c.appraisal_count} 人
                </div>
              </button>
              {c.status === "draft" && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() =>
                    cycleAct.mutate({ id: c.id, action: "open" })
                  }
                >
                  <Play className="w-3.5 h-3.5" />
                  开启
                </Button>
              )}
              {c.status === "open" && (
                <>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setAssignOpen(c)}
                  >
                    <Plus className="w-3.5 h-3.5" />
                    发起考核
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      cycleAct.mutate({ id: c.id, action: "close" })
                    }
                  >
                    <Lock className="w-3.5 h-3.5" />
                    关闭
                  </Button>
                </>
              )}
            </div>
          ))
        )}
      </div>

      <div className="flex items-center gap-3">
        <Select
          value={cycleFilter}
          onChange={setCycleFilter}
          options={[{ value: "", label: "全部周期" }, ...cycleOptions]}
          className="w-[200px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          条考核
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={ClipboardCheck}
            title="还没有考核记录"
            description="在开启的周期上点「发起考核」把员工纳入考核。"
          />
        ) : (
          items.map((a, i) => (
            <div
              key={a.id}
              className={`px-6 py-4 ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {a.employee_name}
                </span>
                <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                  {a.cycle_name} · {a.org_unit_name ?? ""}
                </span>
                <StatusPill status={a.status} label={APPRAISAL_STATUS_LABEL[a.status]} />
                <div className="flex-1" />
                <Rating value={a.self_rating} label="自评" />
                <Rating value={a.manager_rating} label="上级" />
                {a.status === "self_submitted" && (
                  <Button size="sm" onClick={() => setReviewing(a)}>
                    <Star className="w-3.5 h-3.5" />
                    评分
                  </Button>
                )}
              </div>
              {a.self_review && (
                <div className="mt-1.5 text-[12px] text-[var(--color-text-secondary)] font-body">
                  自评: {a.self_review}
                </div>
              )}
              {a.manager_review && (
                <div className="mt-1 text-[12px] text-[var(--color-text-secondary)] font-body">
                  上级评语: {a.manager_review}
                </div>
              )}
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

      <CycleCreateModal open={cycleOpen} onClose={() => setCycleOpen(false)} />
      {assignOpen && (
        <AssignModal cycle={assignOpen} onClose={() => setAssignOpen(null)} />
      )}
      {reviewing && (
        <ReviewModal a={reviewing} onClose={() => setReviewing(null)} />
      )}
    </main>
  );
}

function Rating({ value, label }: { value: number | null; label: string }) {
  return (
    <span className="flex items-center gap-1 text-[12px] font-body text-[var(--color-text-secondary)]">
      {label}
      <span className="font-mono font-semibold text-[var(--color-text-primary)]">
        {value != null ? `${value}/5` : "—"}
      </span>
    </span>
  );
}

function CycleCreateModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/performance/cycles", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["appraisal-cycles"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      period_start: String(fd.get("period_start") || ""),
      period_end: String(fd.get("period_end") || ""),
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建考核周期"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="cy-create" disabled={mut.isPending}>
            {mut.isPending ? "创建中…" : "创建"}
          </Button>
        </>
      }
    >
      <form id="cy-create" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="名称" required>
          <Input name="name" required placeholder="2025 年度考核" />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="开始" required>
            <Input name="period_start" type="date" required />
          </Field>
          <Field label="结束" required>
            <Input name="period_end" type="date" required />
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

function AssignModal({
  cycle,
  onClose,
}: {
  cycle: AppraisalCycle;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const empOptions = useEmployeeOptions();
  const [empId, setEmpId] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: () =>
      fetchJSON("/api/performance/appraisals", {
        method: "POST",
        body: JSON.stringify({
          cycle_id: cycle.id,
          employee_ids: [empId],
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["appraisals"] });
      qc.invalidateQueries({ queryKey: ["appraisal-cycles"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "发起失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title={`发起考核 — ${cycle.name}`}
      description="员工在自助端完成自评后,由上级在这里评分。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button
            onClick={() => {
              if (!empId) {
                setErr("请选择员工");
                return;
              }
              mut.mutate();
            }}
            disabled={mut.isPending}
          >
            {mut.isPending ? "发起中…" : "发起"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Field label="员工" required>
          <Select
            value={empId}
            onChange={setEmpId}
            options={empOptions}
            placeholder="选择员工"
          />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </div>
    </Modal>
  );
}

type AssistOut = {
  suggested_rating: number | null;
  review_draft: string;
  strengths: string[];
  improvements: string[];
  calibration_note: string;
  talking_points: string[];
  evidence: Record<string, number | null>;
  degraded: boolean;
};

function ReviewModal({
  a,
  onClose,
}: {
  a: Appraisal;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [rating, setRating] = useState("4");
  const [review, setReview] = useState("");
  const [assist, setAssist] = useState<AssistOut | null>(null);
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON(`/api/performance/appraisals/${a.id}/review`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["appraisals"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });
  const assistMut = useMutation({
    mutationFn: () =>
      fetchJSON<AssistOut>("/api/ai/appraisal-assist", {
        method: "POST",
        body: JSON.stringify({ appraisal_id: a.id }),
      }),
    onSuccess: (d) => {
      setAssist(d);
      if (d.review_draft) setReview(d.review_draft);
      if (d.suggested_rating != null) setRating(String(d.suggested_rating));
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "AI 起草失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title={`上级评分 — ${a.employee_name}`}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="review" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form
        id="review"
        className="flex flex-col gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          mut.mutate({
            manager_review: review.trim(),
            manager_rating: Number(rating),
          });
        }}
      >
        {a.self_review && (
          <div className="bg-[var(--color-bg-subtle)] rounded-lg p-3 text-[13px] font-body text-[var(--color-text-secondary)]">
            员工自评({a.self_rating}/5): {a.self_review}
          </div>
        )}

        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={() => assistMut.mutate()}
            disabled={assistMut.isPending}
            className="self-start flex items-center gap-1.5 text-[12px] text-[var(--color-accent)] font-body disabled:opacity-50"
          >
            <Brain className="w-3.5 h-3.5" />
            {assistMut.isPending
              ? "AI 分析目标与过程记录中…"
              : "AI 起草评语(目标达成+过程记录)"}
          </button>
          {assist && (
            <div className="border border-[var(--color-border-subtle)] rounded-lg p-3 flex flex-col gap-2 bg-[var(--color-bg-subtle)]">
              {assist.degraded && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-warning-soft)] text-[var(--color-warning-text)] self-start">
                  AI 暂不可用 · 模板草稿
                </span>
              )}
              {assist.calibration_note && (
                <div className="text-[12px] font-body text-[var(--color-warning-text)]">
                  校准: {assist.calibration_note}
                </div>
              )}
              {(assist.strengths.length > 0 || assist.improvements.length > 0) && (
                <div className="flex flex-wrap gap-1.5">
                  {assist.strengths.filter(Boolean).map((s, i) => (
                    <span
                      key={`s${i}`}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-success-soft)] text-[var(--color-success)] font-body"
                    >
                      ▲ {s}
                    </span>
                  ))}
                  {assist.improvements.filter(Boolean).map((s, i) => (
                    <span
                      key={`i${i}`}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-warning-soft)] text-[var(--color-warning-text)] font-body"
                    >
                      ▽ {s}
                    </span>
                  ))}
                </div>
              )}
              {assist.talking_points.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] font-body">
                    面谈切入点
                  </span>
                  {assist.talking_points.map((t, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]"
                    >
                      <Lightbulb className="w-3.5 h-3.5 mt-0.5 shrink-0 text-[var(--color-accent)]" />
                      {t}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <Field label="评分" required>
          <Select
            value={rating}
            onChange={setRating}
            options={[1, 2, 3, 4, 5].map((n) => ({
              value: String(n),
              label: `${n} / 5`,
            }))}
          />
        </Field>
        <Field label="评语" required>
          <textarea
            name="manager_review"
            required
            placeholder="表现点评"
            rows={3}
            value={review}
            onChange={(e) => setReview(e.target.value)}
            className="rounded-lg border border-[var(--color-border-subtle)] p-3 text-sm font-body focus:outline-none focus:border-[var(--color-accent)] resize-y"
          />
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
