import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Briefcase,
  GripVertical,
  KanbanSquare,
  Plus,
  Search,
  UserPlus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { Modal } from "@/components/ui/modal";
import { ApiError, fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";

/**
 * 招聘管道看板(T1-1,MintHCM Candidatures Kanban 移植)。
 *
 * - 列 = job_applications.stage;卡片可拖到目标列 → POST transition
 * - 顶部岗位选择器(job 写进 URL query,可深链)
 * - 「加入候选人」弹窗走人才库搜索,默认落在 applied
 */

type Job = { id: string; title: string; status: string };

type Card = {
  id: string;
  candidate_id: string;
  candidate_name: string;
  stage: string;
  match_score: number | null;
  owner_email: string | null;
  days_in_stage: number;
};

type PipelineColumn = { stage: string; count: number; cards: Card[] };

type Pipeline = {
  job_id: string;
  job_title: string;
  columns: PipelineColumn[];
  total: number;
};

type Talent = { id: string; name: string; display_email: string | null };

const STAGE_META: Record<string, { label: string; tone: string; chip: string }> = {
  applied: {
    label: "新投递",
    tone: "text-[var(--color-info)]",
    chip: "bg-[var(--color-info-soft)] text-[var(--color-info)]",
  },
  screening: {
    label: "筛选中",
    tone: "text-[var(--color-text-secondary)]",
    chip: "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]",
  },
  interview: {
    label: "面试中",
    tone: "text-[var(--color-accent)]",
    chip: "bg-[var(--color-accent-soft,rgba(238,22,79,0.08))] text-[var(--color-accent)]",
  },
  offer: {
    label: "Offer",
    tone: "text-[var(--color-orange)]",
    chip: "bg-[var(--color-orange-soft,rgba(250,163,43,0.14))] text-[var(--color-orange)]",
  },
  hired: {
    label: "已录用",
    tone: "text-[var(--color-success)]",
    chip: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
  },
  rejected: {
    label: "已淘汰",
    tone: "text-[var(--color-error)]",
    chip: "bg-[var(--color-error-soft,rgba(220,38,38,0.08))] text-[var(--color-error)]",
  },
  withdrawn: {
    label: "已放弃",
    tone: "text-[var(--color-text-tertiary)]",
    chip: "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]",
  },
};

const MAIN_STAGES = ["applied", "screening", "interview", "offer", "hired"];
const SIDE_STAGES = ["rejected", "withdrawn"];

export function Pipeline() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const jobId = params.get("job") ?? "";
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.WRITE_JOBS);

  const jobsQuery = useQuery({
    queryKey: ["jobs-for-pipeline"],
    queryFn: () =>
      fetchJSON<{ items: Job[] }>("/api/jobs/?limit=200&status=open"),
  });
  const jobs = jobsQuery.data?.items ?? [];
  const effectiveJobId = jobId || jobs[0]?.id || "";

  const pipelineQuery = useQuery({
    queryKey: ["pipeline", effectiveJobId],
    queryFn: () => fetchJSON<Pipeline>(`/api/jobs/${effectiveJobId}/pipeline`),
    enabled: !!effectiveJobId,
  });
  const pipeline = pipelineQuery.data;

  const transition = useMutation({
    mutationFn: (vars: { appId: string; to: string }) =>
      fetchJSON(`/api/applications/${vars.appId}/transition`, {
        method: "POST",
        body: JSON.stringify({ to_stage: vars.to }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pipeline", effectiveJobId] });
      setFeedback(null);
    },
    onError: (e) =>
      setFeedback(e instanceof ApiError ? e.message : "移动失败"),
  });

  function onDrop(e: React.DragEvent, stage: string) {
    e.preventDefault();
    setDragOver(null);
    const appId = e.dataTransfer.getData("text/plain");
    const from = e.dataTransfer.getData("application/x-stage");
    if (!appId || from === stage) return;
    transition.mutate({ appId, to: stage });
  }

  const cols = useMemo(
    () => pipeline?.columns ?? [],
    [pipeline],
  );

  return (
    <main className="p-8 flex flex-col gap-6 w-full">
      <Link
        to="/jobs"
        className="inline-flex items-center gap-1.5 text-[13px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-body w-fit"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        返回岗位列表
      </Link>

      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            招聘管道
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            拖动卡片推进阶段 · Offer / 面试操作会自动联动
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={effectiveJobId}
            onChange={(e) =>
              setParams(e.target.value ? { job: e.target.value } : {})
            }
            className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] bg-white text-sm font-body text-[var(--color-text-primary)] min-w-[220px]"
          >
            {jobs.map((j) => (
              <option key={j.id} value={j.id}>
                {j.title}
              </option>
            ))}
          </select>
          {canWrite && (
            <Button
              variant="secondary"
              onClick={() => setAddOpen(true)}
              disabled={!effectiveJobId}
            >
              <UserPlus className="w-4 h-4" />
              加入候选人
            </Button>
          )}
        </div>
      </div>

      {feedback && (
        <div className="text-[13px] bg-[var(--color-error-soft,rgba(220,38,38,0.08))] text-[var(--color-error)] px-3 py-2 rounded-md font-body">
          {feedback}
        </div>
      )}

      {!effectiveJobId ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={Briefcase}
            title="暂无进行中的岗位"
            description="先在「岗位」里创建一个 open 状态的岗位,再到这里管理候选管道。"
          />
        </div>
      ) : pipelineQuery.isLoading ? (
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : pipelineQuery.isError ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={KanbanSquare}
            title="管道加载失败"
            description={
              pipelineQuery.error instanceof ApiError
                ? pipelineQuery.error.message
                : "请稍后重试"
            }
          />
        </div>
      ) : !pipeline || pipeline.total === 0 ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={KanbanSquare}
            title="管道还是空的"
            description="把候选人加进管道开始跟踪,或在「匹配候选人」页直接点「加入管道」。发起面试 / 发 Offer 也会自动建卡。"
            action={
              canWrite ? (
                <Button onClick={() => setAddOpen(true)}>
                  <Plus className="w-4 h-4" />
                  加入第一位候选人
                </Button>
              ) : undefined
            }
          />
        </div>
      ) : (
        <div className="flex gap-4 overflow-x-auto pb-4 items-start">
          {cols.map((col) => (
            <PipelineCol
              key={col.stage}
              col={col}
              isMain={MAIN_STAGES.includes(col.stage)}
              dimmed={SIDE_STAGES.includes(col.stage)}
              canWrite={canWrite}
              dragOver={dragOver === col.stage}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(col.stage);
              }}
              onDragLeave={() => setDragOver((d) => (d === col.stage ? null : d))}
              onDrop={(e) => onDrop(e, col.stage)}
              onCardClick={(c) => navigate(`/talents/${c.candidate_id}`)}
            />
          ))}
        </div>
      )}

      <AddCandidateModal
        open={addOpen}
        jobId={effectiveJobId}
        onClose={() => setAddOpen(false)}
        onAdded={() => {
          setAddOpen(false);
          qc.invalidateQueries({ queryKey: ["pipeline", effectiveJobId] });
        }}
      />
    </main>
  );
}

function PipelineCol({
  col,
  isMain,
  dimmed,
  canWrite,
  dragOver,
  onDragOver,
  onDragLeave,
  onDrop,
  onCardClick,
}: {
  col: PipelineColumn;
  isMain: boolean;
  dimmed: boolean;
  canWrite: boolean;
  dragOver: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
  onCardClick: (c: Card) => void;
}) {
  const meta = STAGE_META[col.stage] ?? STAGE_META.applied;
  return (
    <div
      className={`shrink-0 w-[240px] rounded-xl border transition-colors flex flex-col max-h-[70vh] ${
        dragOver
          ? "border-[var(--color-accent)] bg-[rgba(238,22,79,0.04)]"
          : "border-[var(--color-border-subtle)] bg-[var(--color-bg-subtle)]"
      }`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div className="flex items-center justify-between px-3.5 py-3">
        <span
          className={`text-[13px] font-semibold font-body ${meta.tone} ${
            dimmed ? "opacity-80" : ""
          }`}
        >
          {meta.label}
        </span>
        <span className="text-[12px] font-mono text-[var(--color-text-tertiary)]">
          {col.count}
        </span>
      </div>
      <div className="flex flex-col gap-2 px-2.5 pb-3 overflow-y-auto">
        {col.cards.map((c) => (
          <div
            key={c.id}
            draggable={canWrite}
            onDragStart={(e) => {
              e.dataTransfer.setData("text/plain", c.id);
              e.dataTransfer.setData("application/x-stage", c.stage);
              e.dataTransfer.effectAllowed = "move";
            }}
            onClick={() => onCardClick(c)}
            className="bg-white rounded-lg border border-[var(--color-border-subtle)] px-3 py-2.5 cursor-grab active:cursor-grabbing hover:shadow-[var(--shadow-card)] transition-shadow flex flex-col gap-1.5"
          >
            <div className="flex items-center gap-1.5">
              <GripVertical className="w-3 h-3 text-[var(--color-text-tertiary)] shrink-0" />
              <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body truncate">
                {c.candidate_name}
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] font-body">
              {c.match_score != null && (
                <span
                  className={`px-1.5 py-0.5 rounded ${
                    c.match_score >= 80
                      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                  }`}
                >
                  {c.match_score}分
                </span>
              )}
              {c.days_in_stage > 0 && (
                <span className="text-[var(--color-text-tertiary)]">
                  {c.days_in_stage}天
                </span>
              )}
              {c.owner_email && (
                <span className="text-[var(--color-text-tertiary)] truncate ml-auto">
                  {c.owner_email.split("@")[0]}
                </span>
              )}
            </div>
          </div>
        ))}
        {col.count === 0 && isMain && canWrite && (
          <div className="text-[11px] text-[var(--color-text-tertiary)] text-center py-4 border border-dashed border-[var(--color-border-subtle)] rounded-lg">
            拖卡片到这里
          </div>
        )}
      </div>
    </div>
  );
}

function AddCandidateModal({
  open,
  jobId,
  onClose,
  onAdded,
}: {
  open: boolean;
  jobId: string;
  onClose: () => void;
  onAdded: () => void;
}) {
  const [q, setQ] = useState("");
  const [error, setError] = useState<string | null>(null);
  const talentsQuery = useQuery({
    queryKey: ["talent-pick", q],
    queryFn: () =>
      fetchJSON<{ items: Talent[] }>(
        `/api/talents?limit=8${q ? `&q=${encodeURIComponent(q)}` : ""}`,
      ),
    enabled: open,
  });
  const add = useMutation({
    mutationFn: (candId: string) =>
      fetchJSON("/api/applications", {
        method: "POST",
        body: JSON.stringify({ candidate_id: candId, job_id: jobId }),
      }),
    onSuccess: onAdded,
    onError: (e) =>
      setError(
        e instanceof ApiError ? e.message : "加入失败(可能已在管道中)",
      ),
  });

  const items = talentsQuery.data?.items ?? [];
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="加入候选人"
      description="从人才库选人进入当前岗位的管道(默认落在「新投递」)"
    >
      <div className="flex flex-col gap-3 pb-2">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)]" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="搜索候选人姓名 / 邮箱…"
            className="w-full h-9 pl-9 pr-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}
        <div className="flex flex-col divide-y divide-[var(--color-border-row)] min-h-[200px]">
          {items.map((t) => (
            <button
              key={t.id}
              type="button"
              disabled={add.isPending}
              onClick={() => {
                setError(null);
                add.mutate(t.id);
              }}
              className="flex items-center justify-between px-2 py-2.5 text-left hover:bg-[var(--color-bg-subtle)] rounded-md transition-colors"
            >
              <div className="flex flex-col min-w-0">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                  {t.name}
                </span>
                {t.display_email && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    {t.display_email}
                  </span>
                )}
              </div>
              <Plus className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
            </button>
          ))}
          {items.length === 0 && (
            <div className="py-8 text-center text-[12px] text-[var(--color-text-tertiary)] font-body">
              {talentsQuery.isLoading ? "搜索中…" : "没有匹配的候选人"}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}
