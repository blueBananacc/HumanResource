import { useState } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Briefcase, Copy, Layers, Loader2, RefreshCw, Trash2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON } from "@/lib/api";
import { formatRelative, levelLabel } from "@/lib/format";
import {
  ItemDrawer,
  QuestionRow,
  RegenerateDialog,
  SelectionBar,
  StatusChip,
  deleteLibraryItems,
  diffOptions,
  invalidateQuestions,
  isInflight,
  kindOptions,
  questionText,
  useSelection,
  type LibraryItem,
  type QuestionSetSummary,
} from "@/components/questions/shared";

type QuestionSetDetail = QuestionSetSummary & {
  questions: LibraryItem[] | null;
  started_at: string | null;
};

const POLL_MS = 1500;

/** 旧链接 /question-sets/:id → 新地址。 */
export function LegacyQuestionSetRedirect() {
  const { id = "" } = useParams<{ id: string }>();
  return <Navigate to={`/question-library/sets/${id}`} replace />;
}

export function QuestionSetDetail() {
  const { id = "" } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [copiedAll, setCopiedAll] = useState(false);
  const [filterKind, setFilterKind] = useState("");
  const [filterDifficulty, setFilterDifficulty] = useState("");
  const [editing, setEditing] = useState<LibraryItem | null>(null);
  const [regenItem, setRegenItem] = useState<LibraryItem | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["question-set", id],
    queryFn: () => fetchJSON<QuestionSetDetail>(`/api/question-sets/${id}`),
    refetchInterval: (q) => {
      const d = q.state.data as QuestionSetDetail | undefined;
      return d && isInflight(d.status) ? POLL_MS : false;
    },
  });

  const regen = useMutation({
    mutationFn: () => fetchJSON<QuestionSetDetail>(`/api/question-sets/${id}/regen`, { method: "POST" }),
    onSuccess: () => invalidateQuestions(qc),
  });

  const del = useMutation({
    mutationFn: () => fetchJSON(`/api/question-sets/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      invalidateQuestions(qc);
      navigate("/question-library?view=sets");
    },
  });

  const delItem = useMutation({
    mutationFn: (itemId: string) => fetchJSON(`/api/question-library/${itemId}`, { method: "DELETE" }),
    onSuccess: () => invalidateQuestions(qc),
  });

  const allIds = (data?.questions ?? []).map((q) => q.id);
  const sel = useSelection(allIds);
  const [batchDeleting, setBatchDeleting] = useState(false);
  const delMany = async () => {
    const ids = [...sel.selected];
    if (
      await confirm({
        title: `批量删除 ${ids.length} 道题?`,
        description: "选中的题目会从这个题集和题库中删除，操作不可恢复。",
        tone: "danger",
        confirmLabel: "删除",
      })
    ) {
      setBatchDeleting(true);
      await deleteLibraryItems(ids);
      setBatchDeleting(false);
      sel.clear();
      invalidateQuestions(qc);
    }
  };

  if (isLoading || !data) {
    return (
      <main className="h-full grid place-items-center text-sm text-[var(--color-text-tertiary)] font-body">
        加载中…
      </main>
    );
  }

  const all = data.questions ?? [];
  // 题号按题集内原始顺序,筛选后仍保留
  const shown = all
    .map((q, i) => ({ q, idx: i + 1 }))
    .filter(({ q }) => (!filterKind || q.kind === filterKind) && (!filterDifficulty || q.difficulty === filterDifficulty));

  function copyAll() {
    const text = shown.map(({ q, idx }) => `Q${idx}. ${questionText(q)}`).join("\n");
    navigator.clipboard.writeText(text).then(() => {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1500);
    });
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <Link
          to="/question-library?view=sets"
          className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-body"
        >
          <ArrowLeft className="w-4 h-4" />
          返回题集
        </Link>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            onClick={async () => {
              if (
                await confirm({
                  title: "整个题集重新生成?",
                  description: "题集里的所有题目会被新生成的题替换（包括你手动改过的），旧内容无法恢复。",
                  tone: "danger",
                  confirmLabel: "重新生成",
                })
              )
                regen.mutate();
            }}
            disabled={regen.isPending || isInflight(data.status)}
          >
            <RefreshCw className={`w-4 h-4 ${regen.isPending ? "animate-spin" : ""}`} />
            重新生成
          </Button>
          <Button
            variant="secondary"
            onClick={async () => {
              if (
                await confirm({
                  title: "删除题集?",
                  description: `题集及其中 ${data.question_count} 道题会从题库中删除，操作不可恢复。`,
                  tone: "danger",
                  confirmLabel: "删除",
                })
              )
                del.mutate();
            }}
          >
            <Trash2 className="w-4 h-4" />
            删除
          </Button>
        </div>
      </div>

      {/* 元信息 */}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
        <div className="flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-[var(--color-accent)]" />
          <h2 className="font-heading font-semibold text-xl">{data.candidate_name} · 题集</h2>
          <StatusChip status={data.status} />
        </div>
        <div className="grid grid-cols-2 gap-3 text-[12px] text-[var(--color-text-secondary)] font-body">
          <Meta label="简历" value={data.resume_file_name} />
          <Meta label="目标岗位" value={data.job_title ?? "未指定(纯按简历出题)"} icon={Briefcase} />
          <Meta label="难度" value={levelLabel(data.level)} />
          <Meta
            label="题量"
            value={
              data.status === "done" && data.question_count !== data.count
                ? `${data.question_count} 题(生成时 ${data.count} 题)`
                : `${data.count} 题`
            }
          />
          {data.kinds.length > 0 && <Meta label="题目类型" value={data.kinds.join(" / ")} />}
          <Meta
            label="生成于"
            value={
              data.finished_at
                ? formatRelative(data.finished_at)
                : data.started_at
                  ? `开始于 ${formatRelative(data.started_at)}`
                  : "等待中"
            }
          />
        </div>
      </div>

      {data.status === "failed" && (
        <div className="bg-[var(--color-danger-soft)] text-[var(--color-danger)] p-4 rounded-xl font-body flex items-start gap-3">
          <XCircle className="w-5 h-5 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <div className="font-medium">出题失败</div>
            <div className="text-[12px] opacity-80">{data.error || "LLM 调用异常,可点「重新生成」重试。"}</div>
          </div>
        </div>
      )}

      {isInflight(data.status) && (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-4 font-body flex items-center gap-3 text-[var(--color-text-secondary)]">
          <Loader2 className="w-4 h-4 animate-spin" />
          {data.status === "pending" ? "已入队,等 worker 拉取…" : "AI 正在出题,大约 5-15 秒…"}
        </div>
      )}

      {all.length > 0 && (
        <>
          <div className="flex items-center gap-3 flex-wrap">
            <Select value={filterKind} onChange={setFilterKind} options={kindOptions("全部题型")} className="w-[160px]" />
            <Select
              value={filterDifficulty}
              onChange={setFilterDifficulty}
              options={diffOptions("全部难度")}
              className="w-[160px]"
            />
            <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
              {shown.length === all.length ? `共 ${all.length} 道题` : `筛选出 ${shown.length} / ${all.length} 道题`}
            </span>
            <Button variant="secondary" onClick={copyAll} disabled={shown.length === 0} className="ml-auto">
              <Copy className="w-3.5 h-3.5" />
              {copiedAll ? "已复制" : shown.length === all.length ? "复制全部" : "复制筛选结果"}
            </Button>
          </div>
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            <SelectionBar
              allChecked={sel.allChecked}
              onToggleAll={sel.toggleAll}
              count={sel.selected.size}
              unit="道题"
              deleting={batchDeleting}
              onDelete={delMany}
              onClear={sel.clear}
            />
            {shown.length === 0 ? (
              <div className="p-10 text-center text-sm text-[var(--color-text-tertiary)] font-body">没有符合筛选条件的题目</div>
            ) : (
              shown.map(({ q, idx }, i) => (
                <QuestionRow
                  key={q.id}
                  item={q}
                  index={idx}
                  showSet={false}
                  expanded
                  isLast={i === shown.length - 1}
                  selected={sel.selected.has(q.id)}
                  onSelect={(v) => sel.toggle(q.id, v)}
                  onEdit={() => setEditing(q)}
                  onRegenerate={() => setRegenItem(q)}
                  onDelete={async () => {
                    if (
                      await confirm({
                        title: `删除 Q${idx}?`,
                        description: "这道题会从题集和题库中删除，操作不可恢复。",
                        tone: "danger",
                        confirmLabel: "删除",
                      })
                    )
                      delItem.mutate(q.id);
                  }}
                />
              ))
            )}
          </div>
        </>
      )}

      {editing && <ItemDrawer open onClose={() => setEditing(null)} item={editing} />}
      {regenItem && <RegenerateDialog item={regenItem} onClose={() => setRegenItem(null)} />}
    </main>
  );
}

function Meta({ label, value, icon: Icon }: { label: string; value: string; icon?: typeof Briefcase }) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="text-[var(--color-text-tertiary)] shrink-0">{label}:</span>
      {Icon && <Icon className="w-3 h-3 text-[var(--color-text-tertiary)] shrink-0" />}
      <span className="text-[var(--color-text-primary)] font-medium truncate">{value}</span>
    </div>
  );
}
