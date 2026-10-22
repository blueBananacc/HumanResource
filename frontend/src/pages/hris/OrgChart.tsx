import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowLeft, Users } from "lucide-react";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import { StatusPill, type OrgChartNode } from "@/lib/hris";

function TreeNode({ node, depth }: { node: OrgChartNode; depth: number }) {
  return (
    <div>
      <Link
        to={`/employees/${node.id}`}
        className={`flex items-center gap-3 px-4 py-3 rounded-xl border border-[var(--color-border-subtle)] bg-white shadow-[var(--shadow-card)] hover:border-[var(--color-accent)] transition-colors ${
          depth > 0 ? "mt-2" : ""
        }`}
        style={{ marginLeft: depth * 28 }}
      >
        <div className="w-8 h-8 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] flex items-center justify-center font-heading font-semibold text-[13px] shrink-0">
          {node.name[0]}
        </div>
        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-[var(--color-text-primary)] font-body truncate">
              {node.name}
            </span>
            <StatusPill status={node.status} />
          </div>
          <span className="text-[12px] text-[var(--color-text-secondary)] font-body truncate">
            {[node.designation, node.org_unit_name]
              .filter(Boolean)
              .join(" · ") || "—"}
          </span>
        </div>
      </Link>
      {node.children.map((c) => (
        <TreeNode key={c.id} node={c} depth={depth + 1} />
      ))}
    </div>
  );
}

export function OrgChart() {
  const { data, isLoading } = useQuery<OrgChartNode[]>({
    queryKey: ["org-chart"],
    queryFn: () => fetchJSON("/api/employees/org-chart"),
  });
  const roots = data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <Link
        to="/employees"
        className="flex items-center gap-1.5 text-[13px] text-[var(--color-text-secondary)] font-body hover:text-[var(--color-text-primary)] w-fit"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        员工列表
      </Link>
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          组织架构图
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          按汇报线( reports_to )渲染的员工层级 — 在员工详情里设置汇报对象。
        </p>
      </div>

      {isLoading ? (
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : roots.length === 0 ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={Users}
            title="还没有员工"
            description="先在员工列表创建员工并设置汇报对象,组织图会自动生成。"
          />
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {roots.map((n) => (
            <TreeNode key={n.id} node={n} depth={0} />
          ))}
        </div>
      )}
    </main>
  );
}
