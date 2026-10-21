import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Copy,
  Layers,
  Loader2,
  Pencil,
  Sparkles,
  Tag,
  Trash2,
  Wand2,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Drawer } from "@/components/ui/drawer";
import { Modal } from "@/components/ui/modal";
import { fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";

// 题库:题目是唯一实体,题集只是题目的集合。题库视图与题集详情共用这里的组件。

export type LibraryItem = {
  id: string;
  question: string;
  answer_points: string[];
  kind: string;
  difficulty: string;
  category: string;
  skill: string | null;
  follow_up: string | null;
  source: "manual" | "ai_generated";
  use_count: number;
  avg_score: number | null;
  generated_from_job_id: string | null;
  question_set_id: string | null;
  question_set_label: string | null;
  position: number;
  dimensions: string[];
  in_category_set: boolean;
  created_by: string | null;
  created_at: string;
  updated_at: string;
};

export type SetStatus = "pending" | "generating" | "done" | "failed";

export type QuestionSetSummary = {
  id: string;
  resume_id: string;
  job_id: string | null;
  level: string;
  count: number;
  kinds: string[];
  status: SetStatus;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  resume_file_name: string;
  candidate_name: string;
  job_title: string | null;
  question_count: number;
};

/** 分类题集:同分类标签的独立题目 >= 2 道自动成集(不落库)。 */
export type CategorySet = {
  category: string;
  question_count: number;
  kinds: string[];
  last_updated_at: string;
};

export const categorySetPath = (category: string) =>
  `/question-library/categories/${encodeURIComponent(category)}`;

export const KIND_LABEL: Record<string, string> = {
  tech: "技术深度",
  project: "项目复盘",
  scenario: "场景排查",
  design: "系统设计",
  soft: "软技能",
};
export const DIFF_LABEL: Record<string, string> = {
  initial: "初级",
  intermediate: "中级",
  advanced: "高级",
  expert: "专家",
};
export const kindOptions = (all?: string) => [
  ...(all ? [{ value: "", label: all }] : []),
  ...Object.entries(KIND_LABEL).map(([value, label]) => ({ value, label })),
];
export const diffOptions = (all?: string) => [
  ...(all ? [{ value: "", label: all }] : []),
  ...Object.entries(DIFF_LABEL).map(([value, label]) => ({ value, label })),
];

export const isInflight = (s: SetStatus) => s === "pending" || s === "generating";

/** 题目增删改后,题库列表、题集列表、题集详情都要刷新。 */
export function invalidateQuestions(qc: QueryClient) {
  qc.invalidateQueries({ queryKey: ["question-library"] });
  qc.invalidateQueries({ queryKey: ["question-sets"] });
  qc.invalidateQueries({ queryKey: ["question-set"] });
  qc.invalidateQueries({ queryKey: ["question-categories"] });
}

export function questionText(it: LibraryItem) {
  return (
    `[${DIFF_LABEL[it.difficulty] ?? it.difficulty}] ${it.question}\n` +
    `要点: ${it.answer_points.join(" / ")}\n` +
    (it.follow_up ? `追问: ${it.follow_up}\n` : "")
  );
}

/** 列表批量选择:selected 随可见项变化自动剪掉已不存在的 id。 */
export function useSelection(allIds: string[]) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  useEffect(() => {
    setSelected((prev) => {
      const next = new Set([...prev].filter((id) => allIds.includes(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [allIds]);
  const toggle = (id: string, checked: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  const toggleAll = (checked: boolean) =>
    setSelected(checked ? new Set(allIds) : new Set());
  return {
    selected,
    clear: () => setSelected(new Set()),
    toggle,
    toggleAll,
    allChecked: allIds.length > 0 && allIds.every((id) => selected.has(id)),
  };
}

/** 逐条调单删接口的批量删除;返回失败条数。 */
export async function deleteLibraryItems(ids: string[]) {
  const results = await Promise.allSettled(
    ids.map((id) => fetchJSON(`/api/question-library/${id}`, { method: "DELETE" })),
  );
  return results.filter((r) => r.status === "rejected").length;
}

/**
 * 卡片顶部的批量选择条:全选当前可见项;有选中时显示计数 + 批量删除 + 清除。
 */
export function SelectionBar({
  allChecked,
  onToggleAll,
  count,
  unit,
  deleting,
  onDelete,
  onClear,
}: {
  allChecked: boolean;
  onToggleAll: (checked: boolean) => void;
  count: number;
  unit: string;
  deleting: boolean;
  onDelete: () => void;
  onClear: () => void;
}) {
  const active = count > 0;
  return (
    <div
      className={cn(
        "flex items-center gap-3 px-6 h-11 border-b border-[var(--color-border-row)] transition-colors duration-200",
        active ? "bg-[var(--color-accent-soft)]/50" : "bg-[var(--color-bg-muted)]/40",
      )}
    >
      <Checkbox
        checked={allChecked}
        indeterminate={active && !allChecked}
        onToggle={() => onToggleAll(!allChecked)}
        label="全选"
      />
      {active ? (
        <>
          <span className="text-[12px] font-body text-[var(--color-text-secondary)]">
            已选{" "}
            <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-[var(--color-orange)] text-white text-[11px] font-mono font-semibold align-middle">
              {count}
            </span>{" "}
            {unit}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={onClear}
              className="px-3 py-1 rounded-md text-[12px] font-body font-bold text-[var(--color-text-secondary)] hover:bg-white/70 hover:text-[var(--color-text-primary)] transition-colors"
            >
              清除
            </button>
            <button
              type="button"
              onClick={onDelete}
              disabled={deleting}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md text-[12px] font-body font-bold text-[var(--color-danger)] border border-[var(--color-danger)]/40 bg-white hover:bg-[var(--color-danger)] hover:text-white hover:border-[var(--color-danger)] transition-all duration-150 disabled:opacity-50"
            >
              {deleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
              删除
            </button>
          </div>
        </>
      ) : (
        <span className="text-[12px] font-body text-[var(--color-text-tertiary)]">全选当前列表</span>
      )}
    </div>
  );
}

const chip =
  "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium font-body";

function DiffChip({ difficulty }: { difficulty: string }) {
  const tone =
    difficulty === "initial"
      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
      : difficulty === "advanced" || difficulty === "expert"
        ? "bg-[var(--color-warning-soft)] text-[var(--color-warning-text)]"
        : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]";
  return <span className={cn(chip, tone)}>{DIFF_LABEL[difficulty] ?? difficulty}</span>;
}

export function StatusChip({ status }: { status: SetStatus }) {
  if (status === "done")
    return (
      <span className={cn(chip, "gap-1 bg-[var(--color-success-soft)] text-[var(--color-success)]")}>
        <CheckCircle2 className="w-3 h-3" />
        已生成
      </span>
    );
  if (status === "failed")
    return (
      <span className={cn(chip, "gap-1 bg-[var(--color-danger-soft)] text-[var(--color-danger)]")}>
        <XCircle className="w-3 h-3" />
        失败
      </span>
    );
  return (
    <span className={cn(chip, "gap-1 bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]")}>
      <Loader2 className="w-3 h-3 animate-spin" />
      {status === "pending" ? "排队中" : "生成中"}
    </span>
  );
}

/**
 * 单道题。题库视图里显示所属题集(可点进题集);题集详情里显示题号与完整要点。
 */
export function QuestionRow({
  item,
  index,
  showSet = true,
  expanded = false,
  isLast,
  selected,
  onSelect,
  onEdit,
  onRegenerate,
  onDelete,
}: {
  item: LibraryItem;
  index?: number;
  showSet?: boolean;
  expanded?: boolean;
  isLast: boolean;
  /** 传入即开启行首复选框(批量删除) */
  selected?: boolean;
  onSelect?: (checked: boolean) => void;
  onEdit: () => void;
  onRegenerate: () => void;
  onDelete: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const points = expanded ? item.answer_points : item.answer_points.slice(0, 3);
  const copy = () =>
    navigator.clipboard.writeText(questionText(item)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });

  return (
    <div
      className={cn(
        "px-6 py-4 group transition-colors",
        selected ? "bg-[var(--color-accent-soft)]/40" : "hover:bg-[var(--color-bg-muted)]",
        !isLast && "border-b border-[var(--color-border-row)]",
      )}
    >
      <div className="flex items-start justify-between gap-4">
        {onSelect && (
          <Checkbox checked={!!selected} onToggle={() => onSelect(!selected)} label="选择题目" className="mt-1" />
        )}
        <div className="flex flex-col flex-1 min-w-0 gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            {index !== undefined && (
              <span className="text-[11px] font-mono text-[var(--color-text-tertiary)]">Q{index}</span>
            )}
            <span className={cn(chip, "bg-[var(--color-accent-soft)] text-[var(--color-accent)]")}>
              {KIND_LABEL[item.kind] ?? item.kind}
            </span>
            <DiffChip difficulty={item.difficulty} />
            {item.dimensions
              .filter((d) => d !== KIND_LABEL[item.kind])
              .map((d) => (
                <span key={d} className={cn(chip, "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]")}>
                  {d}
                </span>
              ))}
            {item.category && (
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">{item.category}</span>
            )}
            <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">
              {item.source === "ai_generated" ? "AI 生成" : "手动"}
            </span>
            {item.use_count > 0 && (
              <span className="text-[10px] text-[var(--color-text-tertiary)] font-body">
                · 引用 {item.use_count} 次
              </span>
            )}
          </div>
          <div className="text-sm text-[var(--color-text-primary)] font-body leading-relaxed whitespace-pre-wrap">
            {item.question}
          </div>
          {points.length > 0 && (
            <ul className="text-[12px] text-[var(--color-text-secondary)] font-body list-disc list-inside space-y-0.5">
              {points.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
              {!expanded && item.answer_points.length > 3 && (
                <li className="text-[var(--color-text-tertiary)]">…还有 {item.answer_points.length - 3} 条</li>
              )}
            </ul>
          )}
          {expanded && item.follow_up && (
            <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
              <span className="text-[var(--color-text-tertiary)]">追问:</span>
              <span className="italic">{item.follow_up}</span>
            </div>
          )}
          <div className="flex items-center gap-3 text-[11px] text-[var(--color-text-tertiary)] font-body">
            {showSet &&
              (item.question_set_id ? (
                <SetBadge to={`/question-library/sets/${item.question_set_id}`} icon={Layers}>
                  {item.question_set_label ?? "题集"}
                </SetBadge>
              ) : item.in_category_set ? (
                <SetBadge to={categorySetPath(item.category)} icon={Tag}>
                  {item.category}
                </SetBadge>
              ) : (
                <span>独立题目</span>
              ))}
            <span>{formatRelative(item.updated_at ?? item.created_at)}</span>
          </div>
        </div>
        <div className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity flex gap-0.5">
          <IconBtn label={copied ? "已复制" : "复制本题"} onClick={copy}>
            {copied ? <CheckCircle2 className="w-4 h-4 text-[var(--color-success)]" /> : <Copy className="w-4 h-4" />}
          </IconBtn>
          <IconBtn label="AI 重新生成本题" onClick={onRegenerate}>
            <Wand2 className="w-4 h-4" />
          </IconBtn>
          <IconBtn label="编辑" onClick={onEdit}>
            <Pencil className="w-4 h-4" />
          </IconBtn>
          <IconBtn label="删除" onClick={onDelete} danger>
            <Trash2 className="w-4 h-4" />
          </IconBtn>
        </div>
      </div>
    </div>
  );
}

function SetBadge({ to, icon: Icon, children }: { to: string; icon: typeof Layers; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--color-orange-tint)]/60 text-[var(--color-text-primary)] hover:bg-[var(--color-orange-tint)] transition-colors"
      title="查看所属题集"
    >
      <Icon className="w-3 h-3" />
      {children}
    </Link>
  );
}

function IconBtn({
  label,
  onClick,
  danger,
  children,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={cn(
        "p-1.5 rounded-md text-[var(--color-text-tertiary)] hover:bg-white transition-colors",
        danger ? "hover:text-[var(--color-danger)]" : "hover:text-[var(--color-text-primary)]",
      )}
    >
      {children}
    </button>
  );
}

export function ItemDrawer({
  open,
  onClose,
  item,
  defaultCategory = "",
}: {
  open: boolean;
  onClose: () => void;
  item: LibraryItem | null;
  /** 新增时预填的分类(在分类题集里新增题目时用) */
  defaultCategory?: string;
}) {
  const qc = useQueryClient();
  const isEdit = !!item;
  const [question, setQuestion] = useState(item?.question ?? "");
  const [answerText, setAnswerText] = useState((item?.answer_points ?? []).join("\n"));
  const [kind, setKind] = useState(item?.kind ?? "tech");
  const [difficulty, setDifficulty] = useState(item?.difficulty ?? "intermediate");
  const [category, setCategory] = useState(item?.category ?? defaultCategory);
  // 已有分类做输入提示,方便把题归到同一分类(同分类 >= 2 道自动成集)
  const { data: existingCats } = useQuery({
    queryKey: ["question-categories", "all"],
    queryFn: () => fetchJSON<CategorySet[]>("/api/question-library/categories?min_count=1"),
    enabled: !item?.question_set_id,
  });
  // 同分类的其他独立题数(编辑时不算自己)
  const othersInCat =
    (existingCats?.find((c) => c.category === category.trim())?.question_count ?? 0) -
    (isEdit && item?.category === category.trim() ? 1 : 0);
  const [skill, setSkill] = useState(item?.skill ?? "");
  const [followUp, setFollowUp] = useState(item?.follow_up ?? "");
  const [err, setErr] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: () => {
      const payload = {
        question: question.trim(),
        answer_points: answerText.split("\n").map((s) => s.trim()).filter(Boolean),
        kind,
        difficulty,
        category: category.trim(),
        skill: skill.trim() || null,
        follow_up: followUp.trim() || null,
      };
      return isEdit
        ? fetchJSON(`/api/question-library/${item!.id}`, { method: "PUT", body: JSON.stringify(payload) })
        : fetchJSON("/api/question-library/", { method: "POST", body: JSON.stringify(payload) });
    },
    onSuccess: () => {
      invalidateQuestions(qc);
      onClose();
    },
    onError: (e: unknown) => setErr(e instanceof Error ? e.message : "保存失败"),
  });

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={isEdit ? "编辑题目" : "新增题目"}
      description={
        item?.question_set_label
          ? `所属题集:${item.question_set_label}`
          : "题干 + 答题要点是面试时的核心参考"
      }
    >
      <div className="flex flex-col gap-4">
        <Field label="题干">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            className="w-full px-3 py-2 rounded-md border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
            placeholder="例如：聊一下你最熟悉的技术栈的某个核心机制"
          />
        </Field>
        <Field label="答题要点（每行一条）">
          <textarea
            value={answerText}
            onChange={(e) => setAnswerText(e.target.value)}
            rows={5}
            className="w-full px-3 py-2 rounded-md border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
            placeholder={"原理概览\n关键数据结构\n适用场景"}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="题型">
            <Select value={kind} onChange={setKind} options={kindOptions()} />
          </Field>
          <Field label="难度">
            <Select value={difficulty} onChange={setDifficulty} options={diffOptions()} />
          </Field>
        </div>
        <Field label="分类标签（可选，同一分类的题目满 2 道自动合并为题集）">
          <Input
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            placeholder="后端工程师 / 前端工程师 …"
            list="question-category-options"
          />
          <datalist id="question-category-options">
            {existingCats?.map((c) => <option key={c.category} value={c.category} />)}
          </datalist>
          {item?.question_set_id ? (
            <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
              这道题属于简历题集，分类标签只用于检索，不会合并到分类题集
            </span>
          ) : (
            othersInCat > 0 && (
              <span className="text-[11px] text-[var(--color-accent)] font-body">
                将与另外 {othersInCat} 道题组成分类题集「{category.trim()}」
              </span>
            )
          )}
        </Field>
        <Field label="技能标签（可选）">
          <Input value={skill} onChange={(e) => setSkill(e.target.value)} placeholder="例如 Python / MySQL / React" />
        </Field>
        <Field label="追问（可选）">
          <Input value={followUp} onChange={(e) => setFollowUp(e.target.value)} placeholder="若有更深一层追问" />
        </Field>
        {err && <ErrorBox>{err}</ErrorBox>}
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button disabled={save.isPending || question.trim().length < 2} onClick={() => save.mutate()}>
            {save.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
            保存
          </Button>
        </div>
      </div>
    </Drawer>
  );
}

const REGEN_HINTS = [
  "换个角度考察同一能力",
  "难度提高一级",
  "难度降低一级",
  "更贴近岗位实际业务场景",
  "结合候选人简历里的项目来问",
  "追问再深入一层",
];

/** 单题重新生成:可写修改要求引导 AI,结果就地替换原题。 */
export function RegenerateDialog({ item, onClose }: { item: LibraryItem; onClose: () => void }) {
  const qc = useQueryClient();
  const [instruction, setInstruction] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const addHint = (h: string) => {
    setInstruction((v) => (v ? `${v}；${h}` : h));
    // 光标移到末尾,方便接着补充
    requestAnimationFrame(() => {
      const el = inputRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    });
  };

  const regen = useMutation({
    mutationFn: () =>
      fetchJSON<LibraryItem>(`/api/question-library/${item.id}/regenerate`, {
        method: "POST",
        body: JSON.stringify({ instruction: instruction.trim() }),
      }),
    onSuccess: () => {
      invalidateQuestions(qc);
      onClose();
    },
    onError: (e: unknown) => setErr(e instanceof Error ? e.message : "重新生成失败"),
  });

  return (
    <Modal
      open
      onClose={onClose}
      title="AI 重新生成本题"
      description="新题会替换原题,所属题集与顺序不变"
      width={560}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button disabled={regen.isPending} onClick={() => regen.mutate()}>
            {regen.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {regen.isPending ? "生成中…" : "重新生成"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4 pb-2">
        <div className="rounded-xl bg-[var(--color-bg-muted)] px-4 py-3 flex flex-col gap-1.5">
          <span className="text-[11px] font-bold tracking-[0.08em] text-[var(--color-text-tertiary)] font-body">
            原题 · {KIND_LABEL[item.kind] ?? item.kind} · {DIFF_LABEL[item.difficulty] ?? item.difficulty}
          </span>
          <p className="text-[13px] text-[var(--color-text-primary)] font-body leading-relaxed">{item.question}</p>
        </div>
        <Field label="修改要求（可选，留空则换个角度出一道同类题）">
          <textarea
            ref={inputRef}
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            rows={3}
            maxLength={500}
            autoFocus
            placeholder="例如：改成考察 Redis 缓存一致性，难度提高一级"
            className="w-full px-3 py-2 rounded-md border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
        </Field>
        <div className="flex flex-wrap gap-1.5">
          {REGEN_HINTS.map((h) => (
            <button
              key={h}
              type="button"
              onClick={() => addHint(h)}
              className="text-[12px] text-[var(--color-text-secondary)] bg-[var(--color-bg-muted)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-accent)] rounded-full px-3 py-1 transition-colors font-body"
            >
              <span className="text-[var(--color-orange)] font-bold mr-1">#</span>
              {h}
            </button>
          ))}
        </div>
        {err && <ErrorBox>{err}</ErrorBox>}
      </div>
    </Modal>
  );
}

export function ErrorBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[13px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-3 py-2 rounded-md font-body">
      {children}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">{label}</span>
      {children}
    </label>
  );
}
