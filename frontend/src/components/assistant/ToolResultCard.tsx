import { useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileText,
  Target,
  Users,
  Video,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

/** sidecar tool_result SSE 帧载荷。 */
export type ToolResultData = {
  call_id: string;
  name: string;
  is_error: boolean;
  duration_ms: number | null;
  result: unknown;
};

type Json = Record<string, unknown>;

/** result.content[].text 拼起来 JSON.parse;失败返回原文。 */
function parsePayload(result: unknown): { json: unknown; text: string } {
  const content = (result as Json | null)?.content;
  const texts = Array.isArray(content)
    ? (content as Array<{ type?: string; text?: string }>)
        .filter((c) => c.type === "text" && c.text)
        .map((c) => c.text as string)
    : [];
  const text = texts.join("\n");
  try {
    return { json: JSON.parse(text), text };
  } catch {
    return { json: null, text };
  }
}

const asArr = (v: unknown): Json[] =>
  Array.isArray(v) ? (v as Json[]) : ((v as Json)?.items as Json[] | undefined) ?? [];

const str = (v: unknown) => (v == null ? "" : String(v));

const STATUS_ZH: Record<string, string> = {
  open: "招聘中", paused: "已暂停", closed: "已关闭", draft: "草稿",
  active: "在职", inactive: "离职", pending: "待处理", published: "已发布",
  in_progress: "进行中", done: "已完成", abandoned: "已放弃",
  applied: "已申请", screening: "筛选中", interview: "面试中",
  offer: "offer", hired: "已入职", rejected: "已拒绝", withdrawn: "已撤回",
  new: "待分配", assigned: "已分配", resolved: "已解决",
  approved: "已批准", cancelled: "已取消",
};
const zh = (s: string) => STATUS_ZH[s] ?? s;

/** 实体 key → 详情路由。 */
const ENTITY_ROUTES: Record<string, (id: string) => string> = {
  talents: (id) => `/talents/${id}`,
  candidates: (id) => `/talents/${id}`,
  jobs: (id) => `/jobs/${id}`,
  employees: (id) => `/employees/${id}`,
  interviews: (id) => `/interviews/${id}/report`,
  matches: () => `/jobs`, // match 无独立页,兜底
  cases: () => `/cases`,
  applications: () => `/pipeline`,
  knowledge: () => `/knowledge`,
  campaigns: () => `/campaigns`,
  announcements: () => `/announcements`,
};

function CardShell({
  icon: Icon,
  title,
  right,
  children,
}: {
  icon: LucideIcon;
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)] overflow-hidden">
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-[var(--color-border-row)] bg-[var(--color-bg-canvas)]">
        <Icon className="w-3.5 h-3.5 text-[var(--color-accent)]" />
        <span className="text-[11px] font-bold text-[var(--color-text-secondary)] flex-1 truncate">
          {title}
        </span>
        {right}
      </div>
      <div className="px-3 py-2">{children}</div>
    </div>
  );
}

function RowLink({ to, children }: { to?: string; children: React.ReactNode }) {
  if (!to) return <span className="truncate">{children}</span>;
  return (
    <Link
      to={to}
      className="truncate text-[var(--color-accent)] hover:underline inline-flex items-center gap-0.5"
    >
      {children}
      <ExternalLink className="w-3 h-3 shrink-0" />
    </Link>
  );
}

function StatusBadge({ s }: { s: string }) {
  return (
    <span className="shrink-0 text-[10px] rounded-full px-1.5 py-0.5 bg-[var(--color-bg-muted)] text-[var(--color-text-tertiary)]">
      {zh(s)}
    </span>
  );
}

function ScoreBadge({ s }: { s: number }) {
  return (
    <span
      className={cn(
        "shrink-0 text-[10px] font-bold rounded-full px-1.5 py-0.5",
        s >= 80
          ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
          : "bg-[var(--color-bg-muted)] text-[var(--color-text-secondary)]",
      )}
    >
      {s}分
    </span>
  );
}

/** 列表行:[主标题(可跳)] [次要文本] [右侧徽标]。 */
function ItemRows({
  items,
  max = 6,
  render,
}: {
  items: Json[];
  max?: number;
  render: (it: Json, i: number) => React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? items : items.slice(0, max);
  return (
    <div className="flex flex-col">
      {shown.map((it, i) => (
        <div
          key={i}
          className="flex items-center gap-2 py-1 text-[12px] text-[var(--color-text-primary)] border-b border-[var(--color-border-row)] last:border-0"
        >
          {render(it, i)}
        </div>
      ))}
      {items.length > max && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 self-start text-[11px] text-[var(--color-accent)] hover:underline"
        >
          {expanded ? "收起" : `还有 ${items.length - max} 条…`}
        </button>
      )}
    </div>
  );
}

function GenericRows({ items, entity }: { items: Json[]; entity?: string }) {
  return (
    <ItemRows
      items={items}
      render={(it) => {
        const label = str(it.label ?? it.name ?? it.title ?? it.subject ?? it.id);
        const secondary = str(
          it.secondary ?? it.stage ?? it.status ?? it.kind ?? it.category ?? "",
        );
        const route =
          entity && it.id ? ENTITY_ROUTES[entity]?.(str(it.id)) : undefined;
        return (
          <>
            <RowLink to={route}>{label}</RowLink>
            <span className="flex-1" />
            {secondary && <StatusBadge s={secondary} />}
          </>
        );
      }}
    />
  );
}

/** 按工具名的富卡片;返回 null 时走 fallback JSON。 */
export function ToolResultCard({
  name,
  result,
  args,
  isError,
  durationMs,
  onInject,
}: {
  name: string;
  result: unknown;
  args?: Record<string, unknown>;
  isError: boolean;
  durationMs?: number | null;
  /** T16:卡片内动作 → 往输入框注入话术(如「安排面试」)。 */
  onInject?: (text: string) => void;
}) {
  const { json, text } = parsePayload(result);

  if (isError) {
    const detail =
      (json as Json | null)?.detail ?? (json as Json | null)?.error ?? text.slice(0, 300);
    return (
      <div className="rounded-xl border border-[var(--color-danger,#dc2626)]/40 bg-[var(--color-danger,#dc2626)]/5 px-3 py-2 flex items-start gap-2">
        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 text-[var(--color-danger,#dc2626)] shrink-0" />
        <div className="text-[12px] text-[var(--color-danger,#dc2626)] whitespace-pre-wrap break-all">
          {str(detail) || "工具执行失败"}
        </div>
      </div>
    );
  }

  // ===== 候选人表 =====
  if (name === "search_talents") {
    const items = asArr(json);
    const total = (json as Json)?.total;
    return (
      <CardShell
        icon={Users}
        title={`候选人 ${total != null ? `共 ${str(total)} 人` : `${items.length} 人`}`}
      >
        <ItemRows
          items={items}
          render={(it) => (
            <>
              <RowLink to={`/talents/${str(it.id)}`}>{str(it.name)}</RowLink>
              {Array.isArray(it.tags) &&
                it.tags.slice(0, 2).map((t) => (
                  <span
                    key={str(t)}
                    className="text-[10px] rounded-full px-1.5 py-0.5 bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                  >
                    {str(t)}
                  </span>
                ))}
              <span className="flex-1" />
              {it.top_match_score != null && <ScoreBadge s={Number(it.top_match_score)} />}
              <StatusBadge s={str(it.status)} />
            </>
          )}
        />
      </CardShell>
    );
  }

  // ===== 岗位 =====
  if (name === "list_jobs") {
    const items = asArr(json);
    return (
      <CardShell icon={FileText} title={`岗位 ${items.length} 个`}>
        <ItemRows
          items={items}
          render={(it) => (
            <>
              <RowLink to={`/jobs/${str(it.id)}`}>{str(it.title)}</RowLink>
              <span className="text-[11px] text-[var(--color-text-tertiary)] truncate">
                {str(it.level)}
              </span>
              <span className="flex-1" />
              <StatusBadge s={str(it.status)} />
            </>
          )}
        />
      </CardShell>
    );
  }

  // ===== 面试列表 =====
  if (name === "list_interviews") {
    const items = asArr(json);
    return (
      <CardShell icon={Video} title={`面试 ${items.length} 场`}>
        <ItemRows
          items={items}
          render={(it) => (
            <>
              <RowLink to={`/interviews/${str(it.id)}/report`}>
                {str(it.candidate_name)} · {str(it.job_title)}
              </RowLink>
              <span className="flex-1" />
              <StatusBadge s={str(it.status)} />
            </>
          )}
        />
      </CardShell>
    );
  }

  // ===== T16:人才画像 =====
  if (name === "list_talent_profiles") {
    const items = asArr(json);
    return (
      <CardShell icon={Target} title={`人才画像 ${items.length} 个`}>
        <ItemRows
          items={items}
          render={(it) => (
            <>
              <span className="truncate font-medium">{str(it.name)}</span>
              {Array.isArray((it.spec as Json)?.skills) &&
                ((it.spec as Json).skills as unknown[]).slice(0, 3).map((s) => (
                  <span
                    key={str(s)}
                    className="text-[10px] rounded-full px-1.5 py-0.5 bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                  >
                    {str(s)}
                  </span>
                ))}
              <span className="flex-1" />
              {it.last_used_at != null && (
                <span className="text-[10px] text-[var(--color-text-tertiary)]">
                  上次使用 {str(it.last_used_at).slice(0, 10)}
                </span>
              )}
              {it.last_used_at == null && (
                <span className="text-[10px] text-[var(--color-text-tertiary)]">未用过</span>
              )}
            </>
          )}
        />
      </CardShell>
    );
  }

  // ===== T16:人才库扫描短名单(可勾选 → 安排面试) =====
  if (name === "scan_talent_pool") {
    const items = asArr((json as Json)?.items ?? json);
    return <CandidatePickCard items={items} onInject={onInject} />;
  }

  // ===== T16:面试纪要 =====
  if (name === "get_interview_summary" && json && typeof json === "object") {
    const j = json as Json;
    const ai = (j.ai ?? {}) as Json;
    const dims = (ai.dimension_scores ?? {}) as Json;
    const quotes = (ai.evidence_quotes ?? []) as unknown[];
    const turns = (j.turns ?? []) as unknown[];
    const cards = (j.scorecards ?? []) as Json[];
    return (
      <CardShell
        icon={Video}
        title={`面试纪要 · ${str(j.candidate_name)} · ${str(j.job_title)}`}
        right={
          <Link
            to={`/interviews/${str(j.id)}/report`}
            className="text-[10px] text-[var(--color-accent)] hover:underline"
          >
            完整报告
          </Link>
        }
      >
        <div className="flex flex-col gap-1.5 text-[12px]">
          {ai.recommendation != null && (
            <div className="font-bold text-[var(--color-accent)]">
              AI 建议:{str(ai.recommendation)}
            </div>
          )}
          {Object.keys(dims).length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(dims).map(([k, v]) => (
                <span
                  key={k}
                  className="rounded-full bg-[var(--color-bg-muted)] px-2 py-0.5 text-[10px] text-[var(--color-text-secondary)]"
                >
                  {k} {str(v)}
                </span>
              ))}
            </div>
          )}
          {ai.comment != null && (
            <p className="text-[var(--color-text-secondary)]">{str(ai.comment)}</p>
          )}
          <div className="text-[11px] text-[var(--color-text-tertiary)]">
            {turns.length > 0 && `已评 ${turns.length} 题`}
            {quotes.length > 0 && ` · 证据 ${quotes.length} 条`}
            {cards.length > 0 && ` · 评分卡 ${cards.length} 张`}
            {str(j.status) !== "done" && ` · 状态 ${zh(str(j.status))}`}
          </div>
          {cards.map((sc) => {
            const s = (sc.ai_summary ?? {}) as Json;
            return s.recommendation ? (
              <div
                key={str(sc.id)}
                className="rounded-lg bg-[var(--color-bg-canvas)] px-2 py-1.5 text-[11px] text-[var(--color-text-secondary)]"
              >
                <span className="font-bold">评分卡「{str(sc.title)}」:</span>{" "}
                {str(s.recommendation)}
                {s.summary ? ` — ${str(s.summary).slice(0, 120)}` : ""}
              </div>
            ) : null;
          })}
        </div>
      </CardShell>
    );
  }

  // ===== 匹配/推荐 =====
  if (name === "get_job_matches" || name === "get_interview_suggestions") {
    const items = asArr(json);
    return (
      <CardShell icon={Target} title={`匹配结果 ${items.length} 条`}>
        <ItemRows
          items={items}
          render={(it) => (
            <>
              <span className="truncate">
                {str(it.candidate_name || it.name)}
                {it.job_title ? ` · ${str(it.job_title)}` : ""}
              </span>
              <span className="flex-1" />
              {it.score != null && <ScoreBadge s={Number(it.score)} />}
            </>
          )}
        />
      </CardShell>
    );
  }

  // ===== 面试报告 =====
  if (name === "get_interview_report" && json && typeof json === "object") {
    const j = json as Json;
    const summary = (j.summary ?? j) as Json;
    const rec = str(summary.recommendation);
    const dims = (summary.dimension_scores ?? summary.scores) as Json | undefined;
    return (
      <CardShell icon={Video} title="面试报告">
        <div className="flex flex-col gap-1.5 text-[12px]">
          {rec && (
            <div className="font-bold text-[var(--color-accent)]">{rec}</div>
          )}
          {dims && (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(dims).map(([k, v]) => (
                <span
                  key={k}
                  className="rounded-full bg-[var(--color-bg-muted)] px-2 py-0.5 text-[10px] text-[var(--color-text-secondary)]"
                >
                  {k} {str(v)}
                </span>
              ))}
            </div>
          )}
          {summary.comment != null && (
            <p className="text-[var(--color-text-secondary)]">{str(summary.comment)}</p>
          )}
        </div>
      </CardShell>
    );
  }

  // ===== 知识库/产品帮助 =====
  if (
    name === "search_knowledge" ||
    name === "search_product_help" ||
    name === "list_playbooks" ||
    name === "get_playbook"
  ) {
    const items = asArr(json);
    if (items.length > 0) {
      return (
        <CardShell
          icon={BookOpen}
          title={name === "search_product_help" ? "产品帮助" : "知识库"}
        >
          <ItemRows
            items={items}
            render={(it) => (
              <div className="min-w-0">
                <div className="truncate font-bold">{str(it.title)}</div>
                {it.snippet != null && (
                  <div className="text-[11px] text-[var(--color-text-tertiary)] line-clamp-2">
                    {str(it.snippet)}
                  </div>
                )}
              </div>
            )}
          />
        </CardShell>
      );
    }
  }

  // ===== 聚合/计数/报表 → 指标卡 =====
  if (
    name === "count_records" ||
    name === "aggregate_records" ||
    name === "check_salary_band" ||
    name === "run_report"
  ) {
    return (
      <CardShell icon={FileText} title="统计结果">
        <pre className="max-h-44 overflow-auto text-[11px] text-[var(--color-text-secondary)] whitespace-pre-wrap break-all">
          {JSON.stringify(json ?? text, null, 2)}
        </pre>
      </CardShell>
    );
  }

  // ===== 写操作成功 =====
  if (
    /^(create|update|delete|publish|save|link|unlink|dismiss)_/.test(name) ||
    name === "save_filter"
  ) {
    const ok = json == null || (json as Json)?.error == null;
    return (
      <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white px-3 py-2 flex items-center gap-2 shadow-[var(--shadow-card)]">
        <CheckCircle2
          className={cn(
            "w-4 h-4 shrink-0",
            ok ? "text-[var(--color-accent)]" : "text-[var(--color-danger,#dc2626)]",
          )}
        />
        <span className="text-[12px] text-[var(--color-text-primary)]">
          {ok ? "操作已完成" : str((json as Json)?.detail ?? "操作失败")}
        </span>
        {durationMs != null && (
          <span className="ml-auto text-[10px] text-[var(--color-text-tertiary)]">
            {durationMs}ms
          </span>
        )}
      </div>
    );
  }

  // ===== 通用列表 / fallback =====
  const items = asArr(json);
  if (items.length > 0) {
    return (
      <CardShell icon={FileText} title={`结果 ${items.length} 条`}>
        <GenericRows items={items} entity={str(args?.entity)} />
      </CardShell>
    );
  }
  if (json == null && !text) return null;

  const preview = json != null ? JSON.stringify(json, null, 2) : text;
  if (!preview) return null;
  return <CollapsibleJson text={preview} />;
}

/** T16 人才库短名单卡:勾选候选人 → 「安排面试」注入话术。 */
function CandidatePickCard({
  items,
  onInject,
}: {
  items: Json[];
  onInject?: (text: string) => void;
}) {
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const toggle = (id: string) =>
    setPicked((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const chosen = items.filter((it) => picked.has(str(it.candidate_id)));

  return (
    <CardShell icon={Target} title={`人才库短名单 ${items.length} 人`}>
      <div className="flex flex-col">
        {items.map((it) => {
          const id = str(it.candidate_id);
          const cov = (it.coverage ?? {}) as Json;
          const hl = (it.highlights ?? []) as unknown[];
          const gaps = (it.gaps ?? []) as unknown[];
          const on = picked.has(id);
          return (
            <button
              key={id}
              type="button"
              onClick={() => toggle(id)}
              className={cn(
                "flex items-start gap-2 py-1.5 px-1.5 rounded-lg text-left border-b border-[var(--color-border-row)] last:border-0 transition-colors",
                on ? "bg-[var(--color-accent-soft)]/50" : "hover:bg-[var(--color-bg-canvas)]",
              )}
            >
              <span
                className={cn(
                  "mt-0.5 w-3.5 h-3.5 rounded-full border flex items-center justify-center shrink-0 text-[9px]",
                  on
                    ? "bg-[var(--color-accent)] border-[var(--color-accent)] text-white"
                    : "border-[var(--color-border-subtle)]",
                )}
              >
                {on && "✓"}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5">
                  <RowLink to={`/talents/${id}`}>{str(it.name)}</RowLink>
                  {it.score != null && <ScoreBadge s={Number(it.score)} />}
                  {cov.score != null && (
                    <span className="text-[10px] text-[var(--color-text-tertiary)]">
                      覆盖{str(cov.score)}%
                    </span>
                  )}
                </span>
                {hl.length > 0 && (
                  <span className="block text-[11px] text-[var(--color-text-secondary)] truncate">
                    {hl.map((h) => str(h)).join(" · ")}
                  </span>
                )}
                {gaps.length > 0 && (
                  <span className="block text-[10px] text-[var(--color-text-tertiary)] truncate">
                    缺口:{gaps.map((g) => str(g)).join(" · ")}
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>
      {onInject && (
        <div className="mt-2 flex gap-2">
          <button
            type="button"
            disabled={chosen.length === 0}
            onClick={() =>
              onInject(
                `请给以下候选人安排面试:${chosen
                  .map((c) => `${str(c.name)}(candidate_id=${str(c.candidate_id)})`)
                  .join("、")}`,
              )
            }
            className="flex-1 h-7 rounded-md bg-[var(--color-accent)] text-white text-[12px] font-bold hover:bg-[var(--color-accent-hover)] disabled:opacity-40 active:scale-[0.98] transition-all"
          >
            安排面试{chosen.length > 0 ? `(${chosen.length})` : ""}
          </button>
          <button
            type="button"
            onClick={() => onInject("继续找,换个条件再扫一遍人才库")}
            className="h-7 px-3 rounded-md border border-[var(--color-border-subtle)] text-[12px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-canvas)]"
          >
            继续查找
          </button>
        </div>
      )}
    </CardShell>
  );
}

function CollapsibleJson({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)] overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-1.5 px-3 py-2 text-[11px] font-bold text-[var(--color-text-tertiary)] hover:bg-[var(--color-bg-canvas)]"
      >
        {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        查看原始结果
      </button>
      {open && (
        <pre className="px-3 pb-2 max-h-44 overflow-auto text-[11px] text-[var(--color-text-secondary)] whitespace-pre-wrap break-all">
          {text.slice(0, 6000)}
        </pre>
      )}
    </div>
  );
}
