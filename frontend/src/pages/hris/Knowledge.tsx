import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, Pencil, Plus, SearchX, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { Field, PageError } from "./Employees";

type Article = {
  id: string;
  title: string;
  category: string;
  content: string;
  tags: string[];
  status: string;
  is_playbook: boolean;
  created_at: string;
  updated_at: string;
};

const CAT_LABEL: Record<string, string> = {
  policy: "制度",
  faq: "FAQ",
  benefit: "福利",
  it: "IT",
  onboarding: "入职",
  other: "其他",
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  draft: { label: "草稿", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
  published: { label: "已发布", cls: "bg-green-50 text-green-700" },
  archived: { label: "已归档", cls: "bg-amber-50 text-amber-700" },
};

type GapData = {
  days: number;
  zero_hit_total: number;
  gaps: { query: string; count: number; last_seen: string | null }[];
};

/** 知识缺口榜单(兜底回复标注):零命中查询聚合,提示该补什么文章。 */
function KnowledgeGapsCard({ canWrite }: { canWrite: boolean }) {
  const { data } = useQuery<GapData>({
    queryKey: ["knowledge-gaps"],
    queryFn: () => fetchJSON("/api/knowledge/gaps"),
    staleTime: 60_000,
  });
  if (!data || data.gaps.length === 0) return null;
  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <SearchX className="w-4 h-4 text-amber-500" />
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
          知识缺口
        </h3>
        <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)] font-body">
          近 {data.days} 天共 {data.zero_hit_total} 次零命中检索 — 这些是员工/助手问了但库里没有的内容
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {data.gaps.slice(0, 10).map((g) => (
          <span
            key={g.query}
            className="inline-flex items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-[12px] font-body text-amber-700"
            title={`最后检索 ${g.last_seen ?? ""}`}
          >
            {g.query}
            <span className="rounded-full bg-white/70 px-1.5 text-[10px] font-mono">{g.count}×</span>
          </span>
        ))}
      </div>
      {canWrite && (
        <p className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          建议:针对高频缺口写篇文章,检索命中率会立刻提升。
        </p>
      )}
    </div>
  );
}

export function Knowledge() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Article | null>(null);
  const [viewing, setViewing] = useState<Article | null>(null);
  const { hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.MANAGE_EMPLOYEES);
  const qc = useQueryClient();
  const { data, isLoading, isError, error, page, pageCount, total, goto, q, setQ } =
    usePagedQuery<Article>({ key: ["knowledge"], url: "/api/knowledge/articles" });
  const items = data?.items ?? [];

  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/knowledge/articles/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["knowledge"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">知识库</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            制度 / FAQ / 福利说明 — AI 助手检索的语料,员工端可查。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-[220px]">
            <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="搜索标题/内容/标签…" />
          </div>
          {!isError && canWrite && (
            <Button onClick={() => setCreateOpen(true)}>
              <Plus className="w-4 h-4" />写文章
            </Button>
          )}
        </div>
      </div>

      <KnowledgeGapsCard canWrite={canWrite} />

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty icon={BookOpen} title="还没有文章" description="写制度说明或 FAQ,AI 助手即可引用回答员工问题。" />
        ) : (
          items.map((a, i) => {
            const st = STATUS_LABEL[a.status] ?? STATUS_LABEL.draft;
            return (
              <div
                key={a.id}
                className={`px-6 py-4 cursor-pointer hover:bg-[var(--color-surface-1)] ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}
                onClick={() => setViewing(a)}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{a.title}</span>
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">
                    {CAT_LABEL[a.category] ?? a.category}
                  </span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  {a.is_playbook && (
                    <span className="text-[11px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-700">AI 剧本</span>
                  )}
                  <div className="flex-1" />
                  {(a.tags ?? []).slice(0, 3).map((t) => (
                    <span key={t} className="text-[11px] text-[var(--color-text-tertiary)]">#{t}</span>
                  ))}
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">{formatRelative(a.updated_at)}</span>
                </div>
                <p className="mt-1 text-[13px] text-[var(--color-text-secondary)] font-body line-clamp-1">{a.content}</p>
                {canWrite && (
                  <div className="mt-2 flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                    <Button variant="ghost" size="sm" onClick={() => setEditing(a)}>
                      <Pencil className="w-3.5 h-3.5" />编辑
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => del.mutate(a.id)} disabled={del.isPending}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {/* 条件挂载重初始化受控状态(category/statusV),避免沿用上次编辑残留 */}
      {(createOpen || editing) && (
        <ArticleModal editing={editing} onClose={() => { setCreateOpen(false); setEditing(null); }} />
      )}
      {viewing && (
        <Modal open onClose={() => setViewing(null)} title={viewing.title} footer={<Button onClick={() => setViewing(null)}>关闭</Button>} width={640}>
          <div className="flex flex-col gap-3 text-[13px] font-body">
            <div className="flex items-center gap-2 text-[12px] text-[var(--color-text-tertiary)]">
              <span>{CAT_LABEL[viewing.category] ?? viewing.category}</span>
              {(viewing.tags ?? []).map((t) => <span key={t}>#{t}</span>)}
            </div>
            <p className="text-[var(--color-text-primary)] whitespace-pre-wrap">{viewing.content}</p>
          </div>
        </Modal>
      )}
    </main>
  );
}

function ArticleModal({ editing, onClose }: { editing: Article | null; onClose: () => void }) {
  const qc = useQueryClient();
  const [category, setCategory] = useState(editing?.category ?? "faq");
  const [statusV, setStatusV] = useState(editing?.status ?? "published");
  const [isPlaybook, setIsPlaybook] = useState(editing?.is_playbook ?? false);
  const [err, setErr] = useState("");

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      editing
        ? fetchJSON(`/api/knowledge/articles/${editing.id}`, { method: "PUT", body: JSON.stringify(body) })
        : fetchJSON("/api/knowledge/articles", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const tags = String(fd.get("tags") || "")
      .split(/[,,\s]+/)
      .map((t) => t.trim())
      .filter(Boolean)
      .slice(0, 20);
    mut.mutate({
      title: String(fd.get("title") || "").trim(),
      category,
      content: String(fd.get("content") || "").trim(),
      tags,
      status: statusV,
      is_playbook: isPlaybook,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={editing ? "编辑文章" : "写文章"}
      width={600}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="kb-form" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="kb-form" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="标题" required>
          <Input name="title" defaultValue={editing?.title} maxLength={128} required />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="分类">
            <Select
              value={category}
              onChange={setCategory}
              options={Object.entries(CAT_LABEL).map(([v, l]) => ({ value: v, label: l }))}
            />
          </Field>
          <Field label="状态">
            <Select
              value={statusV}
              onChange={setStatusV}
              options={[
                { value: "draft", label: "草稿" },
                { value: "published", label: "已发布" },
                { value: "archived", label: "已归档" },
              ]}
            />
          </Field>
        </div>
        <Field label="标签(逗号分隔)">
          <Input name="tags" defaultValue={(editing?.tags ?? []).join(",")} placeholder="年假, 报销, VPN" />
        </Field>
        <label className="flex items-center gap-2 text-[13px] font-body text-[var(--color-text-primary)] cursor-pointer">
          <input
            type="checkbox"
            checked={isPlaybook}
            onChange={(e) => setIsPlaybook(e.target.checked)}
          />
          标记为 AI 剧本(SOP) —— 助手执行多步流程前会读取并按步骤执行
        </label>
        <Field label="正文" required>
          <textarea
            name="content"
            defaultValue={editing?.content}
            required
            rows={8}
            className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 text-sm font-body outline-none focus:border-[var(--color-accent)]"
          />
        </Field>
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}
