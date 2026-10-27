import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlarmClock,
  CalendarClock,
  CheckCircle2,
  ClipboardCheck,
  MapPin,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  FeedbackModal,
  type Scorecard,
} from "@/components/scorecards/ScorecardSection";

type WorkbenchItem = {
  id: string;
  title: string;
  candidate_id: string;
  candidate_name: string;
  job_id: string | null;
  job_title: string | null;
  scheduled_at: string | null;
  location: string | null;
  status: string;
  criteria: { key: string; name: string; dimension: string; weight: number }[];
  my_submitted: boolean;
  submitted_count: number;
  interviewer_count: number;
  overdue: boolean;
};

type Workbench = {
  todo: WorkbenchItem[];
  upcoming: WorkbenchItem[];
  done: WorkbenchItem[];
};

type Tab = "todo" | "upcoming" | "done";

const TAB_LABEL: Record<Tab, string> = {
  todo: "待我评价",
  upcoming: "即将到来",
  done: "已提交",
};

export function InterviewerWorkbench() {
  const { user } = useAuth();
  const qc = useQueryClient();
  const [tab, setTab] = useState<Tab>("todo");
  const [feedbackFor, setFeedbackFor] = useState<Scorecard | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [err, setErr] = useState("");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["interviewer-workbench"],
    queryFn: () => fetchJSON<Workbench>("/api/scorecards/workbench"),
    refetchInterval: 60_000,
  });

  const refresh = () =>
    qc.invalidateQueries({ queryKey: ["interviewer-workbench"] });

  // 工作台条目是瘦身版 — 填反馈需要完整卡(含 feedbacks),按需拉详情
  const openFeedback = async (item: WorkbenchItem) => {
    setLoadingId(item.id);
    setErr("");
    try {
      const full = await fetchJSON<Scorecard>(`/api/scorecards/${item.id}`);
      setFeedbackFor(full);
    } catch {
      setErr("加载评分卡失败,请稍后重试");
    } finally {
      setLoadingId(null);
    }
  };

  const list = data?.[tab] ?? [];
  const overdueCount = data?.todo.filter((i) => i.overdue).length ?? 0;

  return (
    <main className="p-4 md:p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            面试官工作台
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            你被指派的所有面试评价任务。面试结束超过 24 小时未提交会被标记逾期并收到提醒。
          </p>
        </div>
      </div>

      {/* 统计条 */}
      {data && (
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white p-4 flex flex-col gap-1">
            <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
              待我评价
            </span>
            <span className="text-2xl font-heading font-semibold text-[var(--color-text-primary)]">
              {data.todo.length}
            </span>
            {overdueCount > 0 && (
              <span className="text-[11px] font-bold text-[var(--color-danger)]">
                {overdueCount} 条逾期
              </span>
            )}
          </div>
          <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white p-4 flex flex-col gap-1">
            <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
              即将到来
            </span>
            <span className="text-2xl font-heading font-semibold text-[var(--color-text-primary)]">
              {data.upcoming.length}
            </span>
          </div>
          <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white p-4 flex flex-col gap-1">
            <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
              已提交
            </span>
            <span className="text-2xl font-heading font-semibold text-[var(--color-text-primary)]">
              {data.done.length}
            </span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-[var(--color-border-subtle)]">
        {(Object.keys(TAB_LABEL) as Tab[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setTab(k)}
            className={cn(
              "px-4 py-2.5 text-sm font-body font-medium border-b-2 -mb-px transition-colors",
              tab === k
                ? "border-[var(--color-accent)] text-[var(--color-text-primary)]"
                : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]",
            )}
          >
            {TAB_LABEL[k]}
            {k === "todo" && overdueCount > 0 && (
              <span className="ml-1.5 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-[var(--color-danger)] text-white text-[10px] font-bold">
                {overdueCount}
              </span>
            )}
          </button>
        ))}
      </div>

      {err && (
        <div className="text-[12px] text-[var(--color-danger)] bg-[var(--color-danger)]/8 rounded-lg px-3 py-2">
          {err}
        </div>
      )}

      {isLoading ? (
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : isError ? (
        <div className="p-12 text-center text-sm text-[var(--color-danger)] font-body">
          加载失败,请刷新重试
        </div>
      ) : list.length === 0 ? (
        <Empty
          icon={ClipboardCheck}
          title={
            tab === "todo"
              ? "没有待评价的任务"
              : tab === "upcoming"
                ? "近期没有安排给你的面试"
                : "还没有提交过反馈"
          }
          description={
            tab === "todo"
              ? "HR 指派你当面试官后,任务会出现在这里"
              : undefined
          }
        />
      ) : (
        <ul className="flex flex-col gap-3">
          {list.map((item) => (
            <li
              key={item.id}
              className={cn(
                "rounded-xl border bg-white p-4 md:p-5 flex flex-col gap-3 shadow-[var(--shadow-card)]",
                item.overdue
                  ? "border-[var(--color-danger)]/40"
                  : "border-[var(--color-border-subtle)]",
              )}
            >
              <div className="flex items-start gap-3">
                <div className="flex-1 min-w-0 flex flex-col gap-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm text-[var(--color-text-primary)]">
                      {item.title}
                    </span>
                    {item.overdue && (
                      <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-bold bg-[var(--color-danger)]/10 text-[var(--color-danger)]">
                        <AlarmClock className="w-3 h-3" /> 逾期未评
                      </span>
                    )}
                    {item.my_submitted && (
                      <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded font-bold bg-[var(--color-success)]/10 text-[var(--color-success)]">
                        <CheckCircle2 className="w-3 h-3" /> 已提交
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 flex-wrap text-[12px] text-[var(--color-text-secondary)] font-body">
                    <Link
                      to={`/talents/${item.candidate_id}`}
                      className="font-medium text-[var(--color-accent)] hover:underline"
                    >
                      {item.candidate_name}
                    </Link>
                    {item.job_title && <span>{item.job_title}</span>}
                    {item.scheduled_at && (
                      <span className="inline-flex items-center gap-1">
                        <CalendarClock className="w-3.5 h-3.5" />
                        {formatRelative(item.scheduled_at)}
                      </span>
                    )}
                    {item.location && (
                      <span className="inline-flex items-center gap-1">
                        <MapPin className="w-3.5 h-3.5" />
                        {item.location}
                      </span>
                    )}
                    <span className="inline-flex items-center gap-1 text-[var(--color-text-tertiary)]">
                      <Users className="w-3.5 h-3.5" />
                      {item.submitted_count}/{item.interviewer_count} 已交
                    </span>
                  </div>
                  {item.criteria.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-0.5">
                      {item.criteria.map((c) => (
                        <span
                          key={c.key}
                          className="px-2 py-0.5 rounded-md bg-[var(--color-bg-subtle)] text-[11px] text-[var(--color-text-secondary)]"
                        >
                          {c.name}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {item.status === "open" && (
                  <Button
                    size="sm"
                    variant={item.my_submitted ? "secondary" : "primary"}
                    disabled={loadingId === item.id}
                    onClick={() => openFeedback(item)}
                    className="shrink-0"
                  >
                    <ClipboardCheck className="w-3.5 h-3.5" />
                    {loadingId === item.id
                      ? "加载中…"
                      : item.my_submitted
                        ? "改反馈"
                        : "填反馈"}
                  </Button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {feedbackFor && (
        <FeedbackModal
          open
          onClose={() => setFeedbackFor(null)}
          scorecard={feedbackFor}
          mine={
            feedbackFor.feedbacks.find(
              (f) => f.interviewer_user_id === user?.id,
            ) ?? null
          }
          onSaved={refresh}
        />
      )}
    </main>
  );
}
