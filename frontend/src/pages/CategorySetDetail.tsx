import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Copy, Loader2, Pencil, Plus, Tag, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { fetchJSON } from "@/lib/api";
import { type Page } from "@/lib/usePagedQuery";
import {
  ErrorBox,
  Field,
  ItemDrawer,
  KIND_LABEL,
  QuestionRow,
  RegenerateDialog,
  SelectionBar,
  categorySetPath,
  deleteLibraryItems,
  diffOptions,
  invalidateQuestions,
  kindOptions,
  questionText,
  useSelection,
  type LibraryItem,
} from "@/components/questions/shared";

/**
 * 分类题集:同一分类标签的独立题目(>= 2 道)自动聚合。
 * 整集操作:重命名分类、删除整个题集、在题集内新增题目;单题可编辑 / 删除 / 重新生成。
 */
export function CategorySetDetail() {
  const { name = "" } = useParams<{ name: string }>();
  const category = decodeURIComponent(name);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [filterKind, setFilterKind] = useState("");
  const [filterDifficulty, setFilterDifficulty] = useState("");
  const [editing, setEditing] = useState<LibraryItem | null>(null);
  const [creating, setCreating] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [regenItem, setRegenItem] = useState<LibraryItem | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["question-library", "category-set", category],
    queryFn: () =>
      fetchJSON<Page<LibraryItem>>(
        `/api/question-library/?standalone=true&limit=200&category=${encodeURIComponent(category)}`,
      ),
  });

  const delItem = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/question-library/${id}`, { method: "DELETE" }),
    onSuccess: () => invalidateQuestions(qc),
  });

  const allIds = (data?.items ?? []).map((q) => q.id);
  const sel = useSelection(allIds);
  const [batchDeleting, setBatchDeleting] = useState(false);
  const delMany = async () => {
    const ids = [...sel.selected];
    const remaining = allIds.length - ids.length;
    if (
      await confirm({
        title: `批量删除 ${ids.length} 道题?`,
        description:
          remaining > 0 && remaining < 2
            ? `删除后这个分类只剩 ${remaining} 道题，题集会自动解散。操作不可恢复。`
            : "选中的题目会全部删除，操作不可恢复。",
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
  const delAll = useMutation({
    mutationFn: () =>
      fetchJSON("/api/question-library/categories/delete", {
        method: "POST",
        body: JSON.stringify({ category }),
      }),
    onSuccess: () => {
      invalidateQuestions(qc);
      navigate("/question-library?view=sets");
    },
  });

  if (isLoading || !data) {
    return (
      <main className="h-full grid place-items-center text-sm text-[var(--color-text-tertiary)] font-body">
        加载中…
      </main>
    );
  }

  // 按创建顺序编号(列表接口默认最新在前)
  const all = [...data.items].reverse();
  const shown = all
    .map((q, i) => ({ q, idx: i + 1 }))
    .filter(({ q }) => (!filterKind || q.kind === filterKind) && (!filterDifficulty || q.difficulty === filterDifficulty));
  const kinds = [...new Set(all.map((q) => KIND_LABEL[q.kind] ?? q.kind))];
  const isSet = all.length >= 2;

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
          <Button variant="secondary" onClick={() => setCreating(true)}>
            <Plus className="w-4 h-4" />
            新增题目
          </Button>
          <Button variant="secondary" onClick={() => setRenaming(true)} disabled={all.length === 0}>
            <Pencil className="w-4 h-4" />
            重命名
          </Button>
          <Button
            variant="secondary"
            disabled={all.length === 0}
            onClick={async () => {
              if (
                await confirm({
                  title: "删除分类题集?",
                  description: `分类「${category}」下的 ${all.length} 道题会全部删除，操作不可恢复。`,
                  tone: "danger",
                  confirmLabel: "删除",
                })
              )
                delAll.mutate();
            }}
          >
            <Trash2 className="w-4 h-4" />
            删除
          </Button>
        </div>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
        <div className="flex items-center gap-2.5">
          <Tag className="w-5 h-5 text-[var(--color-orange)]" />
          <h2 className="font-heading font-semibold text-xl">{category}</h2>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-[var(--color-orange-tint)]/60 text-[var(--color-text-primary)] font-body">
            分类题集
          </span>
        </div>
        <div className="flex flex-wrap gap-x-8 gap-y-2 text-[12px] font-body">
          <span>
            <span className="text-[var(--color-text-tertiary)]">题量：</span>
            <span className="font-medium">{all.length} 题</span>
          </span>
          {kinds.length > 0 && (
            <span>
              <span className="text-[var(--color-text-tertiary)]">题型：</span>
              <span className="font-medium">{kinds.join(" / ")}</span>
            </span>
          )}
          <span className="text-[var(--color-text-tertiary)]">
            分类标签相同的独立题目满 2 道自动成集；改题目分类或删到不足 2 道时题集自动解散
          </span>
        </div>
      </div>

      {!isSet && (
        <div className="bg-[var(--color-warning-soft)] text-[var(--color-warning-text)] p-4 rounded-xl font-body text-[13px]">
          {all.length === 0
            ? "这个分类下已经没有独立题目，题集已解散。"
            : "这个分类下只剩 1 道独立题目，已不再构成题集；再新增 1 道同分类题目即可恢复。"}
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
                        description:
                          all.length === 2
                            ? "删除后这个分类只剩 1 道题，题集会自动解散。操作不可恢复。"
                            : "这道题会从题库中删除，操作不可恢复。",
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

      {creating && <ItemDrawer open onClose={() => setCreating(false)} item={null} defaultCategory={category} />}
      {editing && <ItemDrawer open onClose={() => setEditing(null)} item={editing} />}
      {regenItem && <RegenerateDialog item={regenItem} onClose={() => setRegenItem(null)} />}
      {renaming && (
        <RenameDialog
          category={category}
          onClose={() => setRenaming(false)}
          onRenamed={(next) => {
            setRenaming(false);
            navigate(categorySetPath(next), { replace: true });
          }}
        />
      )}
    </main>
  );
}

function RenameDialog({
  category,
  onClose,
  onRenamed,
}: {
  category: string;
  onClose: () => void;
  onRenamed: (next: string) => void;
}) {
  const qc = useQueryClient();
  const [value, setValue] = useState(category);
  const [err, setErr] = useState<string | null>(null);
  const next = value.trim();

  const rename = useMutation({
    mutationFn: () =>
      fetchJSON("/api/question-library/categories/rename", {
        method: "POST",
        body: JSON.stringify({ old: category, new: next }),
      }),
    onSuccess: () => {
      invalidateQuestions(qc);
      onRenamed(next);
    },
    onError: (e: unknown) => setErr(e instanceof Error ? e.message : "重命名失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title="重命名分类题集"
      description="会同时修改其中所有题目的分类标签；如果新名称已存在，两组题目会合并为一个题集"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button disabled={!next || next === category || rename.isPending} onClick={() => rename.mutate()}>
            {rename.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
            保存
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3 pb-2">
        <Field label="分类名称">
          <Input value={value} onChange={(e) => setValue(e.target.value)} maxLength={128} autoFocus />
        </Field>
        {err && <ErrorBox>{err}</ErrorBox>}
      </div>
    </Modal>
  );
}
