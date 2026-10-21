import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, ChevronRight, FileText, Layers, List, Loader2, Plus, Sparkles, Tag, Trash2 } from "lucide-react";
import { Empty } from "@/components/ui/empty";
import { useConfirm } from "@/components/ui/confirm";
import { SearchInput } from "@/components/ui/search-input";
import { Pagination } from "@/components/ui/pagination";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Modal } from "@/components/ui/modal";
import { fetchJSON } from "@/lib/api";
import { usePagedQuery, PAGE_SIZE, type Page } from "@/lib/usePagedQuery";
import { formatRelative, levelLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  ErrorBox,
  Field,
  ItemDrawer,
  QuestionRow,
  RegenerateDialog,
  KIND_LABEL,
  StatusChip,
  categorySetPath,
  diffOptions,
  invalidateQuestions,
  isInflight,
  kindOptions,
  type CategorySet,
  type LibraryItem,
  type QuestionSetSummary,
} from "@/components/questions/shared";

type View = "questions" | "sets";

/**
 * 题库(原「面试题集」+「题库」合并)。
 * 题目是唯一实体:可独立存在,也可隶属某个题集。题集两种:
 * - 简历题集:从候选人简历 AI 生成
 * - 分类题集:同分类标签的独立题目 >= 2 道自动合并
 * 题目视图展示全部题目并标出所属题集;题集视图只看题集,不成集的独立题目不出现。
 */
export function QuestionLibrary() {
  const navigate = useNavigate();
  const [sp, setSp] = useSearchParams();
  const view: View = sp.get("view") === "sets" ? "sets" : "questions";
  const [creating, setCreating] = useState(false);
  const [genOpen, setGenOpen] = useState(false);

  // 切视图时清掉另一视图的搜索 / 页码
  const switchView = (v: View) => setSp(v === "sets" ? { view: "sets" } : {}, { replace: true });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">题库</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            所有面试题都在这里：手动录入、AI 批量生成，或从候选人简历一键生成题集
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => navigate("/talents")}>
            <FileText className="w-4 h-4" />
            去人才库出题集
          </Button>
          <Button variant="secondary" onClick={() => setGenOpen(true)}>
            <Sparkles className="w-4 h-4" />
            AI 生成
          </Button>
          <Button onClick={() => setCreating(true)}>
            <Plus className="w-4 h-4" />
            手动新增
          </Button>
        </div>
      </div>

      <div role="tablist" aria-label="展示方式" className="flex items-center gap-1 self-start rounded-full bg-[var(--color-bg-muted)] p-1">
        {([
          ["questions", "题目", List],
          ["sets", "题集", Layers],
        ] as const).map(([v, label, Icon]) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            onClick={() => switchView(v)}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-4 py-1.5 text-[13px] font-bold font-body transition-all duration-200 ease-[var(--ease-bob)]",
              view === v
                ? "bg-white text-[var(--color-accent)] shadow-[var(--shadow-card)]"
                : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]",
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {view === "questions" ? <QuestionsView /> : <SetsView />}

      {creating && <ItemDrawer open onClose={() => setCreating(false)} item={null} />}
      {genOpen && <GenerateDialog onClose={() => setGenOpen(false)} />}
    </main>
  );
}

function QuestionsView() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [error, setError] = useState<string | null>(null);
  const [filterKind, setFilterKind] = useState("");
  const [filterDifficulty, setFilterDifficulty] = useState("");
  const [editing, setEditing] = useState<LibraryItem | null>(null);
  const [regenItem, setRegenItem] = useState<LibraryItem | null>(null);

  const params: Record<string, string | undefined> = {
    kind: filterKind || undefined,
    difficulty: filterDifficulty || undefined,
  };
  const { data, isLoading, page, pageCount, total, goto, q, setQ } = usePagedQuery<LibraryItem>({
    key: ["question-library", filterKind, filterDifficulty],
    url: "/api/question-library/",
    params,
  });
  const [inputQ, setInputQ] = useState(q);
  useEffect(() => setInputQ(q), [q]);
  const items = data?.items ?? [];
  const filtered = !!(q || filterKind || filterDifficulty);

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/question-library/${id}`, { method: "DELETE" }),
    onSuccess: () => invalidateQuestions(qc),
    onError: (e: unknown) => setError(e instanceof Error ? e.message : "删除失败"),
  });

  return (
    <>
      <div className="flex items-center gap-3 flex-wrap">
        <SearchInput
          value={inputQ}
          onChange={(v) => {
            setInputQ(v);
            setQ(v);
          }}
          placeholder="搜索题干 / 分类"
          className="w-[280px]"
        />
        <Select value={filterKind} onChange={setFilterKind} options={kindOptions("全部题型")} className="w-[160px]" />
        <Select value={filterDifficulty} onChange={setFilterDifficulty} options={diffOptions("全部难度")} className="w-[160px]" />
        <span className="ml-auto text-[13px] text-[var(--color-text-secondary)] font-body">
          共 <span className="font-mono font-semibold text-[var(--color-text-primary)]">{total}</span> 道题
        </span>
      </div>

      {error && <ErrorBox>{error}</ErrorBox>}

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <Loading />
        ) : items.length === 0 ? (
          <Empty
            icon={BookOpen}
            title={filtered ? "没有匹配的题目" : "题库还是空的"}
            description={
              filtered
                ? "试试别的筛选条件"
                : "点击「手动新增」创建第一道题，「AI 生成」批量生成，或去人才库为候选人生成题集"
            }
          />
        ) : (
          <>
            {items.map((it, i) => (
              <QuestionRow
                key={it.id}
                item={it}
                isLast={i === items.length - 1}
                onEdit={() => setEditing(it)}
                onRegenerate={() => setRegenItem(it)}
                onDelete={async () => {
                  if (
                    await confirm({
                      title: "删除题目?",
                      description: it.question_set_label
                        ? `这道题也会从题集「${it.question_set_label}」中移除，操作不可恢复。`
                        : "操作不可恢复",
                      tone: "danger",
                      confirmLabel: "删除",
                    })
                  )
                    del.mutate(it.id);
                }}
              />
            ))}
            <Pagination
              page={page}
              pageCount={pageCount}
              total={total}
              pageSize={PAGE_SIZE}
              onChange={goto}
              className="border-t border-[var(--color-border-row)]"
            />
          </>
        )}
      </div>

      {editing && <ItemDrawer open onClose={() => setEditing(null)} item={editing} />}
      {regenItem && <RegenerateDialog item={regenItem} onClose={() => setRegenItem(null)} />}
    </>
  );
}

function SetsView() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [error, setError] = useState<string | null>(null);

  // 服务端分页 + 搜索 — q 命中候选人 / 简历名 / 岗位名
  const { data, isLoading, page, pageCount, total, goto, q, setQ } = usePagedQuery<QuestionSetSummary>({
    key: ["question-sets"],
    url: "/api/question-sets/",
    refetchInterval: (d: Page<QuestionSetSummary> | undefined) =>
      d && d.items.some((x) => isInflight(x.status)) ? 2000 : false,
  });
  const [inputQ, setInputQ] = useState(q);
  useEffect(() => setInputQ(q), [q]);
  const items = data?.items ?? [];

  const { data: categories = [] } = useQuery({
    queryKey: ["question-categories", "sets", q],
    queryFn: () =>
      fetchJSON<CategorySet[]>(`/api/question-library/categories${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  });

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/question-sets/${id}`, { method: "DELETE" }),
    onSuccess: () => invalidateQuestions(qc),
    onError: (e: unknown) => setError(e instanceof Error ? e.message : "删除失败"),
  });
  const delCategory = useMutation({
    mutationFn: (category: string) =>
      fetchJSON("/api/question-library/categories/delete", {
        method: "POST",
        body: JSON.stringify({ category }),
      }),
    onSuccess: () => invalidateQuestions(qc),
    onError: (e: unknown) => setError(e instanceof Error ? e.message : "删除失败"),
  });

  return (
    <>
      <div className="flex items-center gap-3">
        <SearchInput
          value={inputQ}
          onChange={(v) => {
            setInputQ(v);
            setQ(v);
          }}
          placeholder="搜索分类 / 候选人 / 简历名 / 岗位名"
          className="w-[320px]"
        />
        <span className="ml-auto text-[13px] text-[var(--color-text-secondary)] font-body">
          共 <span className="font-mono font-semibold text-[var(--color-text-primary)]">{total + categories.length}</span> 个题集
        </span>
      </div>

      {error && <ErrorBox>{error}</ErrorBox>}

      <SectionTitle
        title="分类题集"
        hint="同一分类标签的独立题目满 2 道自动合并"
        count={categories.length}
      />
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden -mt-3">
        {categories.length === 0 ? (
          <div className="px-6 py-5 text-[13px] text-[var(--color-text-tertiary)] font-body">
            {q ? "没有匹配的分类题集" : "新增题目时填写相同的分类标签，满 2 道就会出现在这里"}
          </div>
        ) : (
          categories.map((c, i) => (
            <CategoryRow
              key={c.category}
              cat={c}
              isLast={i === categories.length - 1}
              onDelete={async () => {
                if (
                  await confirm({
                    title: "删除分类题集?",
                    description: `分类「${c.category}」下的 ${c.question_count} 道题会全部删除，操作不可恢复。`,
                    tone: "danger",
                    confirmLabel: "删除",
                  })
                )
                  delCategory.mutate(c.category);
              }}
            />
          ))
        )}
      </div>

      <SectionTitle title="简历题集" hint="从候选人简历一键生成" count={total} />
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden -mt-3">
        {isLoading ? (
          <Loading />
        ) : items.length === 0 ? (
          <Empty
            icon={Layers}
            title={q ? "没有匹配的简历题集" : "还没有简历题集"}
            description={
              q ? "试试别的关键词,或清除搜索看全部。" : "去人才库打开候选人的简历，点「生成面试题」即可生成题集"
            }
          />
        ) : (
          <>
            {items.map((qs, i) => (
              <SetRow
                key={qs.id}
                qs={qs}
                isLast={i === items.length - 1}
                onDelete={async () => {
                  if (
                    await confirm({
                      title: "删除题集?",
                      description: `将永久删除 ${qs.candidate_name} 的题集及其中 ${qs.question_count} 道题，操作不可恢复。`,
                      tone: "danger",
                      confirmLabel: "删除",
                    })
                  )
                    del.mutate(qs.id);
                }}
              />
            ))}
            <Pagination
              page={page}
              pageCount={pageCount}
              total={total}
              pageSize={PAGE_SIZE}
              onChange={goto}
              className="border-t border-[var(--color-border-row)]"
            />
          </>
        )}
      </div>
    </>
  );
}

function SetRow({ qs, isLast, onDelete }: { qs: QuestionSetSummary; isLast: boolean; onDelete: () => void }) {
  return (
    <Link
      to={`/question-library/sets/${qs.id}`}
      className={cn(
        "flex items-center gap-4 px-6 py-4 group hover:bg-[var(--color-bg-muted)] transition-colors",
        !isLast && "border-b border-[var(--color-border-row)]",
      )}
    >
      <div className="w-9 h-9 rounded-full bg-[var(--color-accent-soft)] flex items-center justify-center shrink-0">
        <Layers className="w-4 h-4 text-[var(--color-accent)]" />
      </div>
      <div className="flex flex-col flex-1 min-w-0 gap-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-bold text-sm text-[var(--color-text-primary)] font-body">{qs.candidate_name}</span>
          <span className="text-xs text-[var(--color-text-tertiary)] font-body">·</span>
          <span className="text-xs text-[var(--color-text-secondary)] font-body truncate">
            {qs.job_title ?? "未指定岗位"}
          </span>
          <StatusChip status={qs.status} />
        </div>
        <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          {levelLabel(qs.level)} · {qs.status === "done" ? qs.question_count : qs.count} 题
          {qs.kinds.length > 0 && ` · ${qs.kinds.join(" / ")}`}
          {" · "}
          {formatRelative(qs.finished_at ?? qs.created_at)}
        </div>
      </div>
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onDelete();
        }}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-1"
        aria-label="删除题集"
      >
        <Trash2 className="w-4 h-4" />
      </button>
      <ChevronRight className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
    </Link>
  );
}

function SectionTitle({ title, hint, count }: { title: string; hint: string; count: number }) {
  return (
    <div className="flex items-baseline gap-2">
      <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">{title}</h3>
      <span className="text-[12px] font-mono text-[var(--color-text-tertiary)]">{count}</span>
      <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">· {hint}</span>
    </div>
  );
}

function CategoryRow({ cat, isLast, onDelete }: { cat: CategorySet; isLast: boolean; onDelete: () => void }) {
  return (
    <Link
      to={categorySetPath(cat.category)}
      className={cn(
        "flex items-center gap-4 px-6 py-4 group hover:bg-[var(--color-bg-muted)] transition-colors",
        !isLast && "border-b border-[var(--color-border-row)]",
      )}
    >
      <div className="w-9 h-9 rounded-full bg-[var(--color-orange-tint)]/60 flex items-center justify-center shrink-0">
        <Tag className="w-4 h-4 text-[var(--color-text-primary)]" />
      </div>
      <div className="flex flex-col flex-1 min-w-0 gap-1">
        <span className="font-bold text-sm text-[var(--color-text-primary)] font-body truncate">{cat.category}</span>
        <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          {cat.question_count} 题 · {cat.kinds.map((k) => KIND_LABEL[k] ?? k).join(" / ")} ·{" "}
          {formatRelative(cat.last_updated_at)}
        </div>
      </div>
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onDelete();
        }}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-1"
        aria-label="删除分类题集"
      >
        <Trash2 className="w-4 h-4" />
      </button>
      <ChevronRight className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
    </Link>
  );
}

function Loading() {
  return <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>;
}

type Job = { id: string; title: string };

function GenerateDialog({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState("");
  const [category, setCategory] = useState("");
  const [kind, setKind] = useState("tech");
  const [difficulty, setDifficulty] = useState("intermediate");
  const [count, setCount] = useState(5);
  const [err, setErr] = useState<string | null>(null);

  const jobsQ = useQuery<{ items: Job[]; total: number }>({
    queryKey: ["jobs-for-library"],
    queryFn: () => fetchJSON(`/api/jobs/?limit=100&offset=0`),
  });

  const gen = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/question-library/generate`, {
        method: "POST",
        body: JSON.stringify({ job_id: jobId || null, category: category.trim(), kind, difficulty, count }),
      }),
    onSuccess: () => {
      invalidateQuestions(qc);
      onClose();
    },
    onError: (e: unknown) => setErr(e instanceof Error ? e.message : "AI 生成失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title="AI 批量生成"
      description="按岗位 / 题型 / 难度生成多道独立题目，自动入库"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button disabled={gen.isPending} onClick={() => gen.mutate()}>
            {gen.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
            开始生成
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Field label="岗位（可选 — 选岗位后按岗位需求出题）">
          <Select
            value={jobId}
            onChange={setJobId}
            options={[
              { value: "", label: "不指定岗位" },
              ...(jobsQ.data?.items ?? []).map((j) => ({ value: j.id, label: j.title })),
            ]}
          />
        </Field>
        <Field label="分类标签（不选岗位时建议填）">
          <Input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="后端工程师 / 数据分析 …" />
        </Field>
        <div className="grid grid-cols-3 gap-3">
          <Field label="题型">
            <Select value={kind} onChange={setKind} options={kindOptions()} />
          </Field>
          <Field label="难度">
            <Select value={difficulty} onChange={setDifficulty} options={diffOptions()} />
          </Field>
          <Field label="题数">
            <Select
              value={String(count)}
              onChange={(v) => setCount(Number(v))}
              options={[3, 5, 10, 15, 20].map((n) => ({ value: String(n), label: String(n) }))}
            />
          </Field>
        </div>
        {err && <ErrorBox>{err}</ErrorBox>}
      </div>
    </Modal>
  );
}
