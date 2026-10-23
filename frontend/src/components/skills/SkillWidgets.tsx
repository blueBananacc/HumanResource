import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileDown, Plus, Search, Star, Tags, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError, fetchJSON } from "@/lib/api";

/**
 * 技能/胜任力 UI(T2-1,MintHCM Competencies 移植)。
 *
 * - ``SkillSearch``:字典搜索 + 回车新建(无命中项时)
 * - ``JobSkillsCard``:岗位技能要求(must/nice)+ 候选池覆盖度
 * - ``CandidateSkillsCard``:候选人技能(level 1-5)+ 简历导入
 *
 * 两处都走「整组 PUT」语义:本地改完立即整体提交。
 */

type Skill = { id: string; name: string; category: string };

type JobSkill = {
  skill_id: string;
  name: string;
  category: string;
  importance: "must" | "nice";
};

type CandidateSkill = {
  skill_id: string;
  name: string;
  category: string;
  level: number;
  source: string;
};

type Coverage = {
  skill_id: string;
  name: string;
  importance: string;
  pool_count: number;
  pipeline_count: number;
};

const CATEGORY_LABEL: Record<string, string> = {
  tech: "技术",
  language: "语言",
  soft: "软技能",
  cert: "证书",
  other: "其他",
};

export function SkillSearch({
  onPick,
  placeholder = "搜索技能,回车选中;无结果时回车新建…",
}: {
  onPick: (v: { skill_id?: string; name: string }) => void;
  placeholder?: string;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const query = useQuery({
    queryKey: ["skill-search", q],
    queryFn: () =>
      fetchJSON<Skill[]>(`/api/skills?limit=8${q ? `&q=${encodeURIComponent(q)}` : ""}`),
    enabled: open,
  });
  const items = query.data ?? [];

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  function pick(v: { skill_id?: string; name: string }) {
    onPick(v);
    setQ("");
    setOpen(false);
  }

  const exact = items.some((s) => s.name === q.trim());
  return (
    <div ref={wrapRef} className="relative">
      <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)]" />
      <input
        value={q}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && q.trim()) {
            e.preventDefault();
            const hit = items.find((s) => s.name === q.trim());
            pick(hit ? { skill_id: hit.id, name: hit.name } : { name: q.trim() });
          }
          if (e.key === "Escape") setOpen(false);
        }}
        placeholder={placeholder}
        className="w-full h-9 pl-8 pr-3 rounded-lg border border-[var(--color-border-subtle)] text-[13px] font-body focus:outline-none focus:border-[var(--color-accent)]"
      />
      {open && (
        <div className="absolute z-20 mt-1 w-full bg-white border border-[var(--color-border-subtle)] rounded-lg shadow-[var(--shadow-card)] max-h-[240px] overflow-y-auto">
          {items.map((s) => (
            <button
              key={s.id}
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick({ skill_id: s.id, name: s.name })}
              className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-[var(--color-bg-subtle)] text-[13px] font-body"
            >
              <span className="text-[var(--color-text-primary)]">{s.name}</span>
              <span className="text-[10px] text-[var(--color-text-tertiary)]">
                {CATEGORY_LABEL[s.category] ?? s.category}
              </span>
            </button>
          ))}
          {q.trim() && !exact && (
            <button
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick({ name: q.trim() })}
              className="w-full flex items-center gap-1.5 px-3 py-2 text-left hover:bg-[var(--color-bg-subtle)] text-[13px] font-body text-[var(--color-accent)] border-t border-[var(--color-border-row)]"
            >
              <Plus className="w-3.5 h-3.5" />
              新建「{q.trim()}」
            </button>
          )}
          {items.length === 0 && !q.trim() && (
            <div className="px-3 py-3 text-[12px] text-[var(--color-text-tertiary)] font-body">
              输入以搜索技能字典
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function JobSkillsCard({
  jobId,
  canWrite,
}: {
  jobId: string;
  canWrite: boolean;
}) {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const skillsQuery = useQuery({
    queryKey: ["job-skills", jobId],
    queryFn: () => fetchJSON<JobSkill[]>(`/api/jobs/${jobId}/skills`),
  });
  const coverageQuery = useQuery({
    queryKey: ["job-skill-coverage", jobId],
    queryFn: () => fetchJSON<Coverage[]>(`/api/jobs/${jobId}/skill-coverage`),
  });
  const items = skillsQuery.data ?? [];
  const coverage = new Map(
    (coverageQuery.data ?? []).map((c) => [c.skill_id, c]),
  );

  const save = useMutation({
    mutationFn: (next: JobSkill[]) =>
      fetchJSON<JobSkill[]>(`/api/jobs/${jobId}/skills`, {
        method: "PUT",
        body: JSON.stringify({
          items: next.map((s) => ({
            skill_id: s.skill_id || undefined,
            name: s.skill_id ? undefined : s.name,
            importance: s.importance,
          })),
        }),
      }),
    onSuccess: () => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["job-skills", jobId] });
      qc.invalidateQueries({ queryKey: ["job-skill-coverage", jobId] });
    },
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  function commit(next: JobSkill[]) {
    save.mutate(next);
  }

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Tags className="w-4 h-4 text-[var(--color-text-tertiary)]" />
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
          技能模型
        </h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          结构化要求 · 候选池覆盖
        </span>
      </div>

      {items.length === 0 ? (
        <span className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          {canWrite ? "用下方搜索框添加必备 / 加分技能" : "未配置"}
        </span>
      ) : (
        <div className="flex flex-col divide-y divide-[var(--color-border-row)]">
          {items.map((s) => {
            const cov = coverage.get(s.skill_id);
            return (
              <div key={s.skill_id} className="flex items-center gap-3 py-2">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                  {s.name}
                </span>
                <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                  {CATEGORY_LABEL[s.category] ?? s.category}
                </span>
                {canWrite ? (
                  <button
                    type="button"
                    onClick={() =>
                      commit(
                        items.map((x) =>
                          x.skill_id === s.skill_id
                            ? { ...x, importance: x.importance === "must" ? "nice" : "must" }
                            : x,
                        ),
                      )
                    }
                    className={`px-2 py-0.5 rounded-md text-[11px] font-body transition-colors ${
                      s.importance === "must"
                        ? "bg-[var(--color-accent-soft,rgba(238,22,79,0.08))] text-[var(--color-accent)]"
                        : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                    }`}
                  >
                    {s.importance === "must" ? "必备" : "加分"}
                  </button>
                ) : (
                  <span
                    className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                      s.importance === "must"
                        ? "bg-[var(--color-accent-soft,rgba(238,22,79,0.08))] text-[var(--color-accent)]"
                        : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                    }`}
                  >
                    {s.importance === "must" ? "必备" : "加分"}
                  </span>
                )}
                {cov && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body ml-auto">
                    池 {cov.pool_count} · 管道 {cov.pipeline_count}
                  </span>
                )}
                {canWrite && (
                  <button
                    type="button"
                    onClick={() =>
                      commit(items.filter((x) => x.skill_id !== s.skill_id))
                    }
                    className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-0.5"
                    aria-label="移除"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {error && (
        <span className="text-[12px] text-[var(--color-error)] font-body">
          {error}
        </span>
      )}
      {canWrite && (
        <SkillSearch
          onPick={(v) =>
            commit([
              ...items,
              {
                skill_id: v.skill_id ?? "",
                name: v.name,
                category: "other",
                importance: "must",
              } satisfies JobSkill,
            ])
          }
        />
      )}
    </div>
  );
}

export function CandidateSkillsCard({
  candidateId,
  canWrite,
}: {
  candidateId: string;
  canWrite: boolean;
}) {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const skillsQuery = useQuery({
    queryKey: ["candidate-skills", candidateId],
    queryFn: () =>
      fetchJSON<CandidateSkill[]>(`/api/candidates/${candidateId}/skills`),
  });
  const items = skillsQuery.data ?? [];

  const save = useMutation({
    mutationFn: (next: CandidateSkill[]) =>
      fetchJSON<CandidateSkill[]>(`/api/candidates/${candidateId}/skills`, {
        method: "PUT",
        body: JSON.stringify({
          items: next.map((s) => ({
            skill_id: s.skill_id || undefined,
            name: s.skill_id ? undefined : s.name,
            level: s.level,
          })),
        }),
      }),
    onSuccess: () => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["candidate-skills", candidateId] });
    },
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  const importResume = useMutation({
    mutationFn: () =>
      fetchJSON<{ imported: number; total: number }>(
        `/api/candidates/${candidateId}/skills/import-resume`,
        { method: "POST" },
      ),
    onSuccess: (r) => {
      setNotice(
        r.imported > 0
          ? `从简历导入 ${r.imported} 项技能`
          : `简历共识别 ${r.total} 项,均已存在`,
      );
      qc.invalidateQueries({ queryKey: ["candidate-skills", candidateId] });
    },
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "导入失败"),
  });

  function commit(next: CandidateSkill[]) {
    save.mutate(next);
  }

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Star className="w-4 h-4 text-[var(--color-text-tertiary)]" />
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
          技能档案
        </h3>
        {canWrite && (
          <Button
            variant="secondary"
            className="!px-2.5 !py-1 !text-[11px] ml-auto"
            onClick={() => {
              setNotice(null);
              importResume.mutate();
            }}
            disabled={importResume.isPending}
          >
            <FileDown className="w-3.5 h-3.5" />
            {importResume.isPending ? "导入中…" : "从简历导入"}
          </Button>
        )}
      </div>

      {notice && (
        <span className="text-[12px] text-[var(--color-info)] font-body">
          {notice}
        </span>
      )}

      {items.length === 0 ? (
        <span className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          {canWrite ? "搜索添加,或点「从简历导入」自动抽取" : "未记录"}
        </span>
      ) : (
        <div className="flex flex-col divide-y divide-[var(--color-border-row)]">
          {items.map((s) => (
            <div key={s.skill_id} className="flex items-center gap-3 py-2">
              <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                {s.name}
              </span>
              <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                {CATEGORY_LABEL[s.category] ?? s.category}
              </span>
              {s.source === "resume" && (
                <span className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--color-info-soft)] text-[var(--color-info)] font-body">
                  简历
                </span>
              )}
              {canWrite ? (
                <select
                  value={s.level}
                  onChange={(e) =>
                    commit(
                      items.map((x) =>
                        x.skill_id === s.skill_id
                          ? { ...x, level: Number(e.target.value) }
                          : x,
                      ),
                    )
                  }
                  className="ml-auto h-7 px-1.5 rounded-md border border-[var(--color-border-subtle)] text-[12px] font-body text-[var(--color-text-secondary)]"
                  aria-label="熟练度"
                >
                  {[1, 2, 3, 4, 5].map((n) => (
                    <option key={n} value={n}>
                      Lv{n}
                    </option>
                  ))}
                </select>
              ) : (
                <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)] font-body">
                  Lv{s.level}
                </span>
              )}
              {canWrite && (
                <button
                  type="button"
                  onClick={() =>
                    commit(items.filter((x) => x.skill_id !== s.skill_id))
                  }
                  className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-0.5"
                  aria-label="移除"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {error && (
        <span className="text-[12px] text-[var(--color-error)] font-body">
          {error}
        </span>
      )}
      {canWrite && (
        <SkillSearch
          onPick={(v) =>
            commit([
              ...items,
              {
                skill_id: v.skill_id ?? "",
                name: v.name,
                category: "other",
                level: 3,
                source: "manual",
              } satisfies CandidateSkill,
            ])
          }
        />
      )}
    </div>
  );
}
