import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  ClipboardList,
  Copy,
  Lock,
  Plus,
  Send,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { Empty } from "@/components/ui/empty";
import { ApiError, fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";

/**
 * 问卷管理(T2-3,MintHCM Surveys 移植)。
 *
 * 列表 → 编辑抽屉(题目构建器)→ 发布(得公开链接 /s/:id)
 * → 结果聚合面板。问卷为候选人体验/面试反馈场景。
 */

type Question = {
  id: string;
  type: string;
  text: string;
  options: string[];
  required: boolean;
  order: number;
};

type Survey = {
  id: string;
  title: string;
  description: string | null;
  status: string;
  question_count: number;
  response_count: number;
  created_at: string;
};

type SurveyDetail = Survey & { questions: Question[] };

type QuestionResult = {
  question_id: string;
  type: string;
  text: string;
  answered: number;
  option_counts: Record<string, number> | null;
  avg_rating: number | null;
  samples: string[];
};

type Results = {
  survey_id: string;
  response_count: number;
  avg_happiness: number | null;
  questions: QuestionResult[];
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  draft: { label: "草稿", cls: "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]" },
  published: { label: "已发布", cls: "bg-[var(--color-success-soft)] text-[var(--color-success)]" },
  closed: { label: "已关闭", cls: "bg-[var(--color-warning-soft)] text-[var(--color-warning-text)]" },
};

const QTYPE_LABEL: Record<string, string> = {
  short_text: "单行文本",
  long_text: "多行文本",
  single_choice: "单选",
  multi_choice: "多选",
  rating: "评分 1-5",
};

export function Surveys() {
  const qc = useQueryClient();
  const { user } = useAuth();
  const canWrite =
    user?.role === "admin" || user?.role === "hr" || user?.role === "hiring_manager";
  const [editing, setEditing] = useState<SurveyDetail | null>(null);
  const [creating, setCreating] = useState(false);
  const [resultsFor, setResultsFor] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  const listQuery = useQuery({
    queryKey: ["surveys"],
    queryFn: () => fetchJSON<Survey[]>("/api/surveys"),
  });
  const items = listQuery.data ?? [];

  const setStatus = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      fetchJSON<Survey>(`/api/surveys/${id}/status`, {
        method: "POST",
        body: JSON.stringify({ status }),
      }),
    onSuccess: (_s, v) => {
      setFeedback(v.status === "published" ? "已发布,可复制填写链接" : "已关闭");
      qc.invalidateQueries({ queryKey: ["surveys"] });
    },
    onError: (e) =>
      setFeedback(e instanceof ApiError ? e.message : "操作失败"),
  });

  const openDetail = useMutation({
    mutationFn: (id: string) =>
      fetchJSON<SurveyDetail>(`/api/surveys/${id}`),
    onSuccess: setEditing,
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            问卷调查
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            候选人体验 / 面试反馈 · 发布后生成公开填写链接
          </p>
        </div>
        {canWrite && (
          <Button onClick={() => setCreating(true)}>
            <Plus className="w-4 h-4" />
            新建问卷
          </Button>
        )}
      </div>

      {feedback && (
        <div className="text-[13px] bg-[var(--color-info-soft)] text-[var(--color-info)] px-3 py-2 rounded-md font-body flex items-center justify-between">
          {feedback}
          <button type="button" onClick={() => setFeedback(null)}>
            <Trash2 className="w-3.5 h-3.5 opacity-60" />
          </button>
        </div>
      )}

      {listQuery.isLoading ? (
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : items.length === 0 ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={ClipboardList}
            title="还没有问卷"
            description="新建一份候选人体验问卷,发布后把链接发给候选人"
          />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {items.map((s) => {
            const st = STATUS_LABEL[s.status] ?? STATUS_LABEL.draft;
            return (
              <div
                key={s.id}
                className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="font-medium text-sm text-[var(--color-text-primary)] font-body truncate">
                      {s.title}
                    </span>
                    <span className="text-[12px] text-[var(--color-text-secondary)] font-body line-clamp-1">
                      {s.description || "—"}
                    </span>
                  </div>
                  <span className={`px-2 py-0.5 rounded-md text-[11px] font-body shrink-0 ${st.cls}`}>
                    {st.label}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[11px] text-[var(--color-text-tertiary)] font-body">
                  <span>{s.question_count} 题</span>
                  <span>{s.response_count} 份答卷</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  {s.status === "draft" && canWrite && (
                    <>
                      <Button
                        variant="secondary"
                        className="!px-2.5 !py-1 !text-[12px]"
                        onClick={() => openDetail.mutate(s.id)}
                      >
                        编辑
                      </Button>
                      <Button
                        className="!px-2.5 !py-1 !text-[12px]"
                        onClick={() =>
                          setStatus.mutate({ id: s.id, status: "published" })
                        }
                      >
                        <Send className="w-3.5 h-3.5" />
                        发布
                      </Button>
                    </>
                  )}
                  {s.status === "published" && (
                    <>
                      <Button
                        variant="secondary"
                        className="!px-2.5 !py-1 !text-[12px]"
                        onClick={() => {
                          navigator.clipboard.writeText(
                            `${location.origin}/s/${s.id}`,
                          );
                          setFeedback("填写链接已复制");
                        }}
                      >
                        <Copy className="w-3.5 h-3.5" />
                        复制链接
                      </Button>
                      {canWrite && (
                        <Button
                          variant="secondary"
                          className="!px-2.5 !py-1 !text-[12px]"
                          onClick={() =>
                            setStatus.mutate({ id: s.id, status: "closed" })
                          }
                        >
                          <Lock className="w-3.5 h-3.5" />
                          关闭
                        </Button>
                      )}
                    </>
                  )}
                  <Button
                    variant="ghost"
                    className="!px-2.5 !py-1 !text-[12px] ml-auto"
                    onClick={() => setResultsFor(s.id)}
                  >
                    <BarChart3 className="w-3.5 h-3.5" />
                    结果
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {(creating || editing) && (
        <SurveyEditor
          initial={editing}
          onClose={() => {
            setCreating(false);
            setEditing(null);
          }}
          onSaved={() => {
            setCreating(false);
            setEditing(null);
            setFeedback("已保存");
            qc.invalidateQueries({ queryKey: ["surveys"] });
          }}
        />
      )}

      {resultsFor && (
        <ResultsDrawer surveyId={resultsFor} onClose={() => setResultsFor(null)} />
      )}
    </main>
  );
}

type DraftQ = {
  type: string;
  text: string;
  optionsText: string;
  required: boolean;
};

function SurveyEditor({
  initial,
  onClose,
  onSaved,
}: {
  initial: SurveyDetail | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [title, setTitle] = useState(initial?.title ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [questions, setQuestions] = useState<DraftQ[]>(
    initial?.questions.map((q) => ({
      type: q.type,
      text: q.text,
      optionsText: q.options.join("\n"),
      required: q.required,
    })) ?? [
      { type: "rating", text: "", optionsText: "", required: true },
    ],
  );
  const [error, setError] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: () => {
      const body = {
        title,
        description: description || null,
        questions: questions.map((q, i) => ({
          type: q.type,
          text: q.text,
          options:
            q.type === "single_choice" || q.type === "multi_choice"
              ? q.optionsText.split("\n").map((s) => s.trim()).filter(Boolean)
              : [],
          required: q.required,
          order: i,
        })),
      };
      return initial
        ? fetchJSON<SurveyDetail>(`/api/surveys/${initial.id}`, {
            method: "PUT",
            body: JSON.stringify(body),
          })
        : fetchJSON<SurveyDetail>("/api/surveys", {
            method: "POST",
            body: JSON.stringify(body),
          });
    },
    onSuccess: onSaved,
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  function setQ(i: number, patch: Partial<DraftQ>) {
    setQuestions((qs) => qs.map((q, j) => (j === i ? { ...q, ...patch } : q)));
  }

  return (
    <Drawer
      open
      onClose={onClose}
      title={initial ? "编辑问卷" : "新建问卷"}
      description="发布后题目锁定,先保存草稿再发布"
      width={640}
    >
      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            标题
          </span>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="如:面试体验问卷"
            className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            说明(可选)
          </span>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={2}
            className="rounded-lg border border-[var(--color-border-subtle)] p-3 text-sm font-body focus:outline-none focus:border-[var(--color-accent)] resize-y"
          />
        </label>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              题目({questions.length})
            </span>
            <button
              type="button"
              onClick={() =>
                setQuestions((qs) => [
                  ...qs,
                  { type: "short_text", text: "", optionsText: "", required: false },
                ])
              }
              className="flex items-center gap-1 text-[12px] text-[var(--color-accent)] font-body"
            >
              <Plus className="w-3.5 h-3.5" />
              加题
            </button>
          </div>
          {questions.map((q, i) => (
            <div
              key={i}
              className="border border-[var(--color-border-subtle)] rounded-lg p-3 flex flex-col gap-2.5"
            >
              <div className="flex items-center gap-2">
                <select
                  value={q.type}
                  onChange={(e) => setQ(i, { type: e.target.value })}
                  className="h-8 px-2 rounded-md border border-[var(--color-border-subtle)] text-[12px] font-body"
                >
                  {Object.entries(QTYPE_LABEL).map(([v, l]) => (
                    <option key={v} value={v}>
                      {l}
                    </option>
                  ))}
                </select>
                <label className="flex items-center gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]">
                  <input
                    type="checkbox"
                    checked={q.required}
                    onChange={(e) => setQ(i, { required: e.target.checked })}
                    className="accent-[var(--color-accent)]"
                  />
                  必填
                </label>
                <button
                  type="button"
                  onClick={() =>
                    setQuestions((qs) => qs.filter((_, j) => j !== i))
                  }
                  className="ml-auto text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                  aria-label="删除题目"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <input
                value={q.text}
                onChange={(e) => setQ(i, { text: e.target.value })}
                placeholder={`第 ${i + 1} 题题干`}
                className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
              />
              {(q.type === "single_choice" || q.type === "multi_choice") && (
                <textarea
                  value={q.optionsText}
                  onChange={(e) => setQ(i, { optionsText: e.target.value })}
                  rows={3}
                  placeholder={"选项,每行一个\n如:\n官网\n内推\n猎头"}
                  className="rounded-lg border border-[var(--color-border-subtle)] p-2.5 text-[13px] font-body focus:outline-none focus:border-[var(--color-accent)] resize-y"
                />
              )}
            </div>
          ))}
        </div>

        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button
            onClick={() => save.mutate()}
            disabled={
              save.isPending ||
              !title.trim() ||
              questions.some((q) => !q.text.trim())
            }
          >
            {save.isPending ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </Drawer>
  );
}

function ResultsDrawer({
  surveyId,
  onClose,
}: {
  surveyId: string;
  onClose: () => void;
}) {
  const query = useQuery({
    queryKey: ["survey-results", surveyId],
    queryFn: () => fetchJSON<Results>(`/api/surveys/${surveyId}/results`),
  });
  const data = query.data;

  return (
    <Drawer open onClose={onClose} title="问卷结果" width={600}>
      {!data ? (
        <div className="p-8 text-center text-sm text-[var(--color-text-tertiary)]">
          加载中…
        </div>
      ) : (
        <div className="flex flex-col gap-5">
          <div className="flex items-center gap-6 text-sm font-body">
            <span>
              答卷 <b className="font-heading">{data.response_count}</b> 份
            </span>
            {data.avg_happiness != null && (
              <span>
                满意度均值{" "}
                <b className="font-heading text-[var(--color-accent)]">
                  {data.avg_happiness}
                </b>
                /5
              </span>
            )}
          </div>
          {data.questions.map((q, i) => (
            <div
              key={q.question_id}
              className="border border-[var(--color-border-subtle)] rounded-lg p-4 flex flex-col gap-2.5"
            >
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                  {i + 1}. {q.text}
                </span>
                <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                  {QTYPE_LABEL[q.type]} · {q.answered} 答
                </span>
              </div>
              {q.option_counts && (
                <div className="flex flex-col gap-1.5">
                  {Object.entries(q.option_counts).map(([opt, n]) => {
                    const pct = q.answered ? Math.round((n / q.answered) * 100) : 0;
                    return (
                      <div key={opt} className="flex items-center gap-2 text-[12px] font-body">
                        <span className="w-20 truncate text-[var(--color-text-secondary)]">
                          {opt}
                        </span>
                        <div className="flex-1 h-2 rounded-full bg-[var(--color-bg-subtle)] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[var(--color-accent)]"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-10 text-right text-[var(--color-text-tertiary)]">
                          {n}·{pct}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
              {q.avg_rating != null && (
                <span className="text-[13px] font-body text-[var(--color-text-primary)]">
                  平均 <b className="text-[var(--color-accent)]">{q.avg_rating}</b>/5
                </span>
              )}
              {q.samples.length > 0 && (
                <div className="flex flex-col gap-1">
                  {q.samples.map((s, j) => (
                    <span
                      key={j}
                      className="text-[12px] text-[var(--color-text-secondary)] font-body bg-[var(--color-bg-subtle)] rounded px-2 py-1"
                    >
                      “{s}”
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Drawer>
  );
}
