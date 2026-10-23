import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, ClipboardList, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError, fetchJSON } from "@/lib/api";

/**
 * 公开问卷填写页(/s/:id,免登录,T2-3)。
 *
 * - 加载 /api/public/surveys/{id} 的题目
 * - respondent_key 存 localStorage → 服务端 UNIQUE 防重复
 * - 提交前可选满意度(happiness 1-5)
 */

type Question = {
  id: string;
  type: string;
  text: string;
  options: string[];
  required: boolean;
  order: number;
};

type PublicSurvey = {
  title: string;
  description: string | null;
  questions: Question[];
};

function respondentKey(sid: string) {
  const k = `survey-key-${sid}`;
  let v = localStorage.getItem(k);
  if (!v) {
    // crypto.randomUUID 仅在 secure context(https/localhost)可用,HTTP 部署兜底
    const rand =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
    v = `rk-${rand}`;
    localStorage.setItem(k, v);
  }
  return v;
}

export function SurveyFill() {
  const { id = "" } = useParams<{ id: string }>();
  const [answers, setAnswers] = useState<Record<string, string | string[] | number>>({});
  const [name, setName] = useState("");
  const [happiness, setHappiness] = useState<number | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useQuery({
    queryKey: ["public-survey", id],
    queryFn: () => fetchJSON<PublicSurvey>(`/api/public/surveys/${id}`),
    retry: false,
  });

  const key = useMemo(() => (id ? respondentKey(id) : ""), [id]);
  useEffect(() => {
    document.title = query.data ? `${query.data.title} - 问卷` : "问卷";
  }, [query.data]);

  const submit = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/public/surveys/${id}/responses`, {
        method: "POST",
        body: JSON.stringify({
          respondent_key: key,
          respondent_name: name.trim() || null,
          happiness,
          answers,
        }),
      }),
    onSuccess: () => setSubmitted(true),
    onError: (e) => {
      if (e instanceof ApiError && e.message.includes("提交过")) {
        setSubmitted(true);
      } else {
        setError(e instanceof ApiError ? e.message : "提交失败,请重试");
      }
    },
  });

  if (query.isLoading) {
    return (
      <Shell>
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载问卷…
        </div>
      </Shell>
    );
  }
  if (query.isError || !query.data) {
    return (
      <Shell>
        <div className="p-12 text-center flex flex-col items-center gap-2">
          <ClipboardList className="w-8 h-8 text-[var(--color-text-tertiary)]" />
          <span className="text-sm text-[var(--color-text-secondary)] font-body">
            问卷不存在或已关闭
          </span>
        </div>
      </Shell>
    );
  }
  const s = query.data;

  if (submitted) {
    return (
      <Shell>
        <div className="p-12 text-center flex flex-col items-center gap-3">
          <CheckCircle2 className="w-10 h-10 text-[var(--color-success)]" />
          <h2 className="font-heading font-semibold text-lg text-[var(--color-text-primary)]">
            感谢你的反馈
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            你的回答已提交,我们会认真阅读。
          </p>
        </div>
      </Shell>
    );
  }

  const missingRequired = s.questions.some((q) => {
    if (!q.required) return false;
    const v = answers[q.id];
    return v === undefined || v === "" || (Array.isArray(v) && v.length === 0);
  });

  return (
    <Shell>
      <div className="flex flex-col gap-1.5 px-6 pt-6 pb-4 border-b border-[var(--color-border-subtle)]">
        <h2 className="font-heading font-semibold text-xl text-[var(--color-text-primary)]">
          {s.title}
        </h2>
        {s.description && (
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            {s.description}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-5 px-6 py-6">
        {s.questions.map((q, i) => (
          <div key={q.id} className="flex flex-col gap-2">
            <span className="text-[14px] font-medium text-[var(--color-text-primary)] font-body">
              {i + 1}. {q.text}
              {q.required && <span className="text-[var(--color-accent)]"> *</span>}
            </span>

            {q.type === "short_text" && (
              <input
                value={(answers[q.id] as string) ?? ""}
                onChange={(e) =>
                  setAnswers((a) => ({ ...a, [q.id]: e.target.value }))
                }
                className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
              />
            )}
            {q.type === "long_text" && (
              <textarea
                value={(answers[q.id] as string) ?? ""}
                onChange={(e) =>
                  setAnswers((a) => ({ ...a, [q.id]: e.target.value }))
                }
                rows={3}
                className="rounded-lg border border-[var(--color-border-subtle)] p-3 text-sm font-body focus:outline-none focus:border-[var(--color-accent)] resize-y"
              />
            )}
            {q.type === "single_choice" && (
              <div className="flex flex-col gap-1.5">
                {q.options.map((opt) => (
                  <label
                    key={opt}
                    className="flex items-center gap-2 text-[14px] font-body text-[var(--color-text-primary)] cursor-pointer"
                  >
                    <input
                      type="radio"
                      name={q.id}
                      checked={answers[q.id] === opt}
                      onChange={() =>
                        setAnswers((a) => ({ ...a, [q.id]: opt }))
                      }
                      className="accent-[var(--color-accent)]"
                    />
                    {opt}
                  </label>
                ))}
              </div>
            )}
            {q.type === "multi_choice" && (
              <div className="flex flex-col gap-1.5">
                {q.options.map((opt) => {
                  const cur = (answers[q.id] as string[]) ?? [];
                  return (
                    <label
                      key={opt}
                      className="flex items-center gap-2 text-[14px] font-body text-[var(--color-text-primary)] cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={cur.includes(opt)}
                        onChange={(e) =>
                          setAnswers((a) => ({
                            ...a,
                            [q.id]: e.target.checked
                              ? [...cur, opt]
                              : cur.filter((x) => x !== opt),
                          }))
                        }
                        className="accent-[var(--color-accent)]"
                      />
                      {opt}
                    </label>
                  );
                })}
              </div>
            )}
            {q.type === "rating" && (
              <div className="flex items-center gap-1.5">
                {[1, 2, 3, 4, 5].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setAnswers((a) => ({ ...a, [q.id]: n }))}
                    className="p-1"
                    aria-label={`${n} 分`}
                  >
                    <Star
                      className={`w-6 h-6 transition-colors ${
                        (answers[q.id] as number) >= n
                          ? "text-[var(--color-warning)] fill-current"
                          : "text-[var(--color-text-tertiary)]"
                      }`}
                    />
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        <div className="border-t border-[var(--color-border-row)] pt-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              怎么称呼你(可选)
            </span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
            />
          </label>
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              这次体验总体满意度(可选)
            </span>
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setHappiness((h) => (h === n ? null : n))}
                aria-label={`满意度 ${n}`}
              >
                <Star
                  className={`w-5 h-5 transition-colors ${
                    (happiness ?? 0) >= n
                      ? "text-[var(--color-warning)] fill-current"
                      : "text-[var(--color-text-tertiary)]"
                  }`}
                />
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}

        <Button
          onClick={() => {
            setError(null);
            submit.mutate();
          }}
          disabled={submit.isPending || missingRequired}
          className="self-center !px-8"
        >
          {submit.isPending ? "提交中…" : "提交问卷"}
        </Button>
        {missingRequired && (
          <span className="text-center text-[11px] text-[var(--color-text-tertiary)] font-body">
            请先回答标 * 的必答题
          </span>
        )}
      </div>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] flex items-start justify-center py-10 px-4">
      <div className="w-full max-w-[560px] bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {children}
      </div>
    </div>
  );
}
