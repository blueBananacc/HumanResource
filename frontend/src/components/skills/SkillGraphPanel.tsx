/** 技能图谱面板 — Workday Skills Cloud 轻量映射:
 * 层级边(parent)+ 别名归并 + 引用热度(岗位/候选人/员工三侧),
 * 层级树呈现,子技能缩进挂在父技能下。
 */
import { useQuery } from "@tanstack/react-query";
import { Network } from "lucide-react";

import { fetchJSON } from "@/lib/api";

type Node = {
  id: string; name: string; category: string;
  parent_id: string | null; aliases: string[];
  job_refs: number; candidate_refs: number; employee_refs: number;
};

const CAT_LABEL: Record<string, string> = {
  tech: "技术", language: "语言", soft: "软技能", cert: "证书", other: "其他",
};

export function SkillGraphPanel() {
  const { data } = useQuery<{ nodes: Node[]; edges: { from: string; to: string }[] }>({
    queryKey: ["skill-graph"],
    queryFn: () => fetchJSON("/api/skills/graph"),
    staleTime: 120_000,
  });
  if (!data || data.nodes.length === 0) return null;

  const childrenOf = new Map<string | null, Node[]>();
  for (const n of data.nodes) {
    const list = childrenOf.get(n.parent_id) ?? [];
    list.push(n);
    childrenOf.set(n.parent_id, list);
  }
  const roots = (childrenOf.get(null) ?? []).sort(
    (a, b) => b.job_refs + b.employee_refs + b.candidate_refs
      - (a.job_refs + a.employee_refs + a.candidate_refs),
  );
  const linked = data.edges.length;
  const aliased = data.nodes.filter((n) => n.aliases.length).length;

  const Row = ({ n, depth }: { n: Node; depth: number }) => (
    <>
      <li
        className="flex items-center gap-2 py-1.5"
        style={{ paddingLeft: `${depth * 20}px` }}
      >
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
          depth === 0 ? "bg-[var(--color-accent)]" : "bg-[var(--color-text-tertiary)]"
        }`} />
        <span className={`text-[13px] ${depth === 0 ? "font-medium text-[var(--color-text-primary)]" : "text-[var(--color-text-secondary)]"}`}>
          {n.name}
        </span>
        <span className="text-[10px] px-1 rounded bg-[var(--color-bg-muted)] text-[var(--color-text-tertiary)]">
          {CAT_LABEL[n.category] ?? n.category}
        </span>
        {n.aliases.map((a) => (
          <span key={a} className="text-[10px] px-1 rounded bg-amber-500/10 text-amber-600">
            别名 {a}
          </span>
        ))}
        <span className="ml-auto text-[10px] text-[var(--color-text-tertiary)] font-mono">
          {[
            n.job_refs ? `岗${n.job_refs}` : "",
            n.candidate_refs ? `候选${n.candidate_refs}` : "",
            n.employee_refs ? `员工${n.employee_refs}` : "",
          ].filter(Boolean).join(" ") || "—"}
        </span>
      </li>
      {(childrenOf.get(n.id) ?? []).map((c) => (
        <Row key={c.id} n={c} depth={depth + 1} />
      ))}
    </>
  );

  return (
    <section className="rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-card)] shadow-[var(--shadow-card)] overflow-hidden">
      <div className="flex items-center gap-3 px-5 pt-4 pb-3 border-b border-[var(--color-border-row)]">
        <Network className="w-5 h-5 text-[var(--color-accent)]" />
        <div className="flex-1">
          <div className="font-heading font-semibold text-[15px]">技能图谱</div>
          <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
            词典层级边 + 别名归并 — 匹配时子技能可向上兑现父级要求
          </div>
        </div>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          {data.nodes.length} 词条 · {linked} 层级边 · {aliased} 别名组
        </span>
      </div>
      <ul className="px-5 py-3 max-h-[360px] overflow-y-auto divide-y divide-[var(--color-border-row)]/60">
        {roots.slice(0, 40).map((n) => (
          <Row key={n.id} n={n} depth={0} />
        ))}
      </ul>
    </section>
  );
}
