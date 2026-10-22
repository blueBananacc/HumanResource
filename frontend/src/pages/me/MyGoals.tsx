import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheck, Goal as GoalIcon, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  StatusPill,
  type MeAppraisal,
  type MeGoal,
} from "@/lib/hris";
import { Field } from "../hris/Employees";

const GOAL_STATUS_LABEL: Record<string, string> = {
  active: "进行中",
  done: "已完成",
  dropped: "已放弃",
};
const APPRAISAL_STATUS_LABEL: Record<string, string> = {
  draft: "待自评",
  self_submitted: "待上级评",
  reviewed: "已完成",
};

export function MyGoals() {
  const qc = useQueryClient();
  const [reviewing, setReviewing] = useState<MeAppraisal | null>(null);
  const { data: goals, isLoading } = useQuery<MeGoal[]>({
    queryKey: ["me-goals"],
    queryFn: () => fetchJSON("/api/me/goals"),
  });
  const { data: appraisals } = useQuery<MeAppraisal[]>({
    queryKey: ["me-appraisals"],
    queryFn: () => fetchJSON("/api/me/appraisals"),
  });

  const setProgress = useMutation({
    mutationFn: ({ id, progress }: { id: string; progress: number }) =>
      fetchJSON(`/api/me/goals/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ progress }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-goals"] }),
  });

  const pendingReviews = (appraisals ?? []).filter(
    (a) => a.status === "draft",
  );

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          我的目标
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          个人目标进度 + 绩效考核自评。
        </p>
      </div>

      {pendingReviews.length > 0 && (
        <section className="bg-[var(--color-warning-soft)] border border-[var(--color-warning-stroke)] rounded-xl p-5 flex flex-col gap-3">
          <h3 className="font-heading font-semibold text-[14px] text-[var(--color-warning-text)]">
            待自评考核({pendingReviews.length})
          </h3>
          {pendingReviews.map((a) => (
            <div key={a.id} className="flex items-center gap-3">
              <span className="text-[13px] font-body text-[var(--color-text-primary)]">
                {a.cycle_name}
              </span>
              <div className="flex-1" />
              <Button size="sm" onClick={() => setReviewing(a)}>
                <Star className="w-3.5 h-3.5" />
                去自评
              </Button>
            </div>
          ))}
        </section>
      )}

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (goals ?? []).length === 0 ? (
          <Empty
            icon={GoalIcon}
            title="还没有目标"
            description="主管/HR 给你定目标后会出现在这里。"
          />
        ) : (
          (goals ?? []).map((g, i) => (
            <div
              key={g.id}
              className={`px-6 py-4 ${
                i < (goals ?? []).length - 1
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
                {g.due_date && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    截止 {g.due_date}
                  </span>
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
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={g.progress}
                  onChange={(e) =>
                    setProgress.mutate({
                      id: g.id,
                      progress: Number(e.target.value),
                    })
                  }
                  className="w-[140px] accent-[var(--color-accent)]"
                  disabled={g.status !== "active"}
                />
                <span className="text-[12px] font-mono text-[var(--color-text-secondary)] w-10 text-right">
                  {g.progress}%
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      {(appraisals ?? []).length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            历史考核
          </h3>
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            {(appraisals ?? []).map((a, i) => (
              <div
                key={a.id}
                className={`px-6 py-4 ${
                  i < (appraisals ?? []).length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <div className="flex items-center gap-3">
                  <ClipboardCheck className="w-4 h-4 text-[var(--color-accent)]" />
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                    {a.cycle_name}
                  </span>
                  <StatusPill status={a.status} label={APPRAISAL_STATUS_LABEL[a.status]} />
                  <div className="flex-1" />
                  {a.self_rating != null && (
                    <span className="text-[12px] font-mono text-[var(--color-text-secondary)]">
                      自评 {a.self_rating}/5
                    </span>
                  )}
                  {a.manager_rating != null && (
                    <span className="text-[12px] font-mono font-semibold text-[var(--color-text-primary)]">
                      上级 {a.manager_rating}/5
                    </span>
                  )}
                </div>
                {a.manager_review && (
                  <div className="mt-1.5 text-[12px] text-[var(--color-text-secondary)] font-body">
                    上级评语: {a.manager_review}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {reviewing && (
        <SelfReviewModal a={reviewing} onClose={() => setReviewing(null)} />
      )}
    </main>
  );
}

function SelfReviewModal({
  a,
  onClose,
}: {
  a: MeAppraisal;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [rating, setRating] = useState("4");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON(`/api/me/appraisals/${a.id}/submit-self`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-appraisals"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title={`自评 — ${a.cycle_name}`}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="self-review" disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交"}
          </Button>
        </>
      }
    >
      <form
        id="self-review"
        className="flex flex-col gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          const fd = new FormData(e.currentTarget);
          mut.mutate({
            self_review: String(fd.get("self_review") || "").trim(),
            self_rating: Number(rating),
          });
        }}
      >
        <Field label="自评分" required>
          <Select
            value={rating}
            onChange={setRating}
            options={[1, 2, 3, 4, 5].map((n) => ({
              value: String(n),
              label: `${n} / 5`,
            }))}
          />
        </Field>
        <Field label="自评内容" required>
          <Input
            name="self_review"
            required
            placeholder="本周期主要工作与成果"
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
