/** 内部人才匹配 — 北森「内部人才市场」轻量版。
 * Job → 员工排行(InternalMatchCard,挂岗位详情页);
 * 员工 → 开放岗位(JobFitCard,挂员工详情页)。
 * 每个命中带 via 标签:direct 直中 / child 子技能兑现 / parent 泛化兑现。
 */
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight, Briefcase, Network, UserRound } from "lucide-react";

import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";

type Matched = {
  skill_id: string; name: string; level: number;
  via: "direct" | "child" | "parent"; via_name?: string | null; source?: string | null;
};
type Req = { skill_id: string; name: string; importance: string };
type JobItem = {
  job: { id: string; title: string };
  score: number; must_hit: number; must_total: number;
  matched: Matched[]; missing: Req[];
};
type EmpItem = {
  employee: { id: string; name: string; designation?: string | null; org_name?: string | null };
  score: number; must_hit: number; must_total: number;
  matched: Matched[]; missing: Req[];
};

const VIA: Record<string, string> = { direct: "直中", child: "子技能", parent: "泛化" };

function ScoreBadge({ score }: { score: number }) {
  const tone =
    score >= 80 ? "bg-emerald-500/10 text-emerald-600"
    : score >= 50 ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
    : "bg-slate-100 text-slate-500";
  return (
    <span className={`font-mono text-sm font-semibold px-2 py-0.5 rounded-md ${tone}`}>
      {score}
    </span>
  );
}

function MatchedChips({ matched }: { matched: Matched[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {matched.map((m) => (
        <span
          key={m.skill_id}
          title={m.via !== "direct" ? `经「${m.via_name}」兑现` : undefined}
          className="px-1.5 py-px rounded bg-emerald-500/10 text-emerald-700 text-[11px]"
        >
          {m.name} L{m.level.toFixed(1)}
          {m.via !== "direct" && <span className="text-[9px] opacity-70">·{VIA[m.via]}</span>}
        </span>
      ))}
    </div>
  );
}

function MissingChips({ missing }: { missing: { name: string }[] }) {
  if (!missing.length) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {missing.map((m) => (
        <span key={m.name} className="px-1.5 py-px rounded bg-slate-100 text-slate-500 text-[11px]">
          缺 {m.name}
        </span>
      ))}
    </div>
  );
}

/** 岗位详情页:内部人才推荐 — 谁能接这个机会 */
export function InternalMatchCard({ jobId }: { jobId: string }) {
  const { data } = useQuery<{
    requirements: Req[]; items: EmpItem[]; evaluated: number;
  }>({
    queryKey: ["internal-match-job", jobId],
    queryFn: () => fetchJSON(`/api/internal-match/jobs/${jobId}?limit=8`),
    staleTime: 60_000,
  });
  if (!data || data.requirements.length === 0) return null;

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)] flex items-center gap-2">
          <Network className="w-4 h-4 text-[var(--color-accent)]" />
          内部人才推荐
        </h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          评估 {data.evaluated} 名在职员工 · 技能图谱加权
        </span>
      </div>
      {data.items.length === 0 ? (
        <Empty
          icon={UserRound}
          title="暂无适配员工"
          description="员工技能档案(已确认)与该岗位要求没有交集"
        />
      ) : (
        <ul className="flex flex-col divide-y divide-[var(--color-border-row)]">
          {data.items.map((it) => (
            <li key={it.employee.id} className="py-2.5 flex items-start gap-3">
              <ScoreBadge score={it.score} />
              <div className="flex-1 min-w-0 flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <Link
                    to={`/employees/${it.employee.id}`}
                    className="text-sm font-medium text-[var(--color-text-primary)] hover:text-[var(--color-accent)]"
                  >
                    {it.employee.name}
                  </Link>
                  <span className="text-[11px] text-[var(--color-text-tertiary)]">
                    {[it.employee.designation, it.employee.org_name].filter(Boolean).join(" · ")}
                  </span>
                  <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)]">
                    必备 {it.must_hit}/{it.must_total}
                  </span>
                </div>
                <MatchedChips matched={it.matched} />
                <MissingChips missing={it.missing} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** 员工详情页:适配机会 — TA 能接哪些开放岗位 */
export function JobFitCard({ employeeId }: { employeeId: string }) {
  const { data } = useQuery<{
    skills_count: number; items: JobItem[];
  }>({
    queryKey: ["internal-match-emp", employeeId],
    queryFn: () => fetchJSON(`/api/internal-match/employees/${employeeId}?limit=6`),
    staleTime: 60_000,
  });
  if (!data || data.skills_count === 0) return null;

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)] flex items-center gap-2">
          <Briefcase className="w-4 h-4 text-[var(--color-accent)]" />
          适配机会
        </h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          基于 {data.skills_count} 项已确认技能
        </span>
      </div>
      {data.items.length === 0 ? (
        <Empty
          icon={Briefcase}
          title="暂无适配岗位"
          description="当前开放岗位的技能要求与该员工暂无交集"
        />
      ) : (
        <ul className="flex flex-col divide-y divide-[var(--color-border-row)]">
          {data.items.map((it) => (
            <li key={it.job.id} className="py-2.5 flex items-start gap-3">
              <ScoreBadge score={it.score} />
              <div className="flex-1 min-w-0 flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <Link
                    to={`/jobs/${it.job.id}`}
                    className="text-sm font-medium text-[var(--color-text-primary)] hover:text-[var(--color-accent)]"
                  >
                    {it.job.title}
                  </Link>
                  <span className="ml-auto text-[11px] text-[var(--color-text-tertiary)]">
                    必备 {it.must_hit}/{it.must_total}
                  </span>
                </div>
                <MatchedChips matched={it.matched} />
                <MissingChips missing={it.missing} />
              </div>
              <ArrowRight className="w-3.5 h-3.5 mt-1 text-[var(--color-text-tertiary)]" />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
