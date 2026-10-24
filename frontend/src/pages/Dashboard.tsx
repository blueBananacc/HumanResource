import { Sparkles, FileText } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { BrandShape } from "@/components/brand/BrandShape";
import { Button } from "@/components/ui/button";
import { fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { formatRelative } from "@/lib/format";

type LicenseStatus = {
  plan: string;
  expires_at: string | null;
  days_remaining: number;
  source: "none" | "trial" | "active" | "expired";
};

type Resume = {
  id: string;
  file_name: string;
  created_at: string;
  candidate: {
    id: string;
    name: string;
    display_email: string | null;
    display_phone: string | null;
  };
  skills: string[];
};

type Job = { id: string; title: string; status: string };

export function Dashboard() {
  const { user } = useAuth();
  const { data: lic } = useQuery({
    queryKey: ["license-status"],
    queryFn: () => fetchJSON<LicenseStatus>("/api/license/status"),
  });

  // 最近 5 份简历用于快捷入口;total 给顶部 Stat 卡显示真实总数
  // (分页化之后 items.length 只会是 ≤ 5,不能再当"简历总数"用)
  const { data: resumesPage } = useQuery({
    queryKey: ["resumes", "recent"],
    queryFn: () =>
      fetchJSON<{ items: Resume[]; total: number }>("/api/resumes/?limit=5"),
  });
  const resumes = resumesPage?.items ?? [];

  // Jobs 用 ?status=open 直接拿开放岗位数;?limit=1 只关心 total,节省传输
  const { data: openJobsPage } = useQuery({
    queryKey: ["jobs", "open-count"],
    queryFn: () =>
      fetchJSON<{ items: Job[]; total: number }>(
        "/api/jobs/?status=open&limit=1",
      ),
  });
  const { data: allJobsPage } = useQuery({
    queryKey: ["jobs", "all-count"],
    queryFn: () =>
      fetchJSON<{ items: Job[]; total: number }>("/api/jobs/?limit=1"),
  });

  const totalResumes = resumesPage?.total ?? 0;
  const openJobs = openJobsPage?.total ?? 0;
  const allJobsCount = allJobsPage?.total ?? 0;

  return (
    <main className="flex flex-col w-full">
      <HomeBanner name={user?.email.split("@")[0] ?? ""} />
      <div className="px-8 pb-8 -mt-10 relative flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      {lic && lic.source === "trial" && (
        <div className="flex items-center gap-4 p-5 rounded-xl bg-[var(--color-bg-muted)] border border-[var(--color-orange-tint)] shadow-[var(--shadow-card)]">
          <div className="w-9 h-9 rounded-full bg-[var(--color-orange)] grid place-items-center shrink-0">
            <Sparkles className="w-[18px] h-[18px] text-white" />
          </div>
          <div className="flex flex-col gap-0.5 flex-1 min-w-0">
            <span className="font-heading font-semibold text-[15px]">
              试用期还剩 {lic.days_remaining} 天
            </span>
            <span className="text-xs text-[var(--color-text-secondary)] font-body">
              升级到正式版以解锁全部 AI 面试与简历自动化能力
            </span>
          </div>
          <Link to="/settings/license">
            <Button>升级</Button>
          </Link>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <Stat
          label="简历总数"
          value={String(totalResumes)}
          delta={totalResumes > 0 ? "本租户全部" : "等待第一份简历"}
          deltaClass="text-[var(--color-text-tertiary)]"
        />
        <Stat
          label="开放岗位"
          value={String(openJobs)}
          delta={`共 ${allJobsCount} 个岗位`}
          deltaClass="text-[var(--color-text-tertiary)]"
        />
        <Stat
          label="进行中面试"
          value="0"
          delta="M1 阶段开放"
          deltaClass="text-[var(--color-text-tertiary)]"
        />
        <Stat
          label="已生成报告"
          value="0"
          delta="M1 阶段开放"
          deltaClass="text-[var(--color-text-tertiary)]"
        />
      </div>

      <div className="grid grid-cols-[1fr_380px] gap-4">
        <RecentResumes resumes={resumes ?? []} />
        <ActivityCard />
      </div>
      </div>
    </main>
  );
}

function HomeBanner({ name }: { name: string }) {
  return (
    <section className="relative h-[200px] overflow-hidden bg-[var(--color-brand)]">
      <BrandShape kind="circle" color="#FAA32B" className="-top-16 right-[30%] w-[220px] h-[220px]" />
      <BrandShape kind="star" color="#FFD887" className="top-4 right-[22%] w-[90px] h-[90px] rotate-12" />
      <BrandShape kind="free" color="#EE164F" className="-top-10 right-[6%] w-[260px] h-[260px]" />
      <BrandShape kind="triangle" color="#FEABB0" className="top-20 right-[40%] w-[120px] h-[120px] -rotate-12" />
      <BrandShape kind="arc" color="#F36488" className="-bottom-6 right-[16%] w-[130px] h-[130px]" />
      <BrandShape kind="rect" color="#60032A" className="-bottom-24 -right-10 w-[220px] h-[220px] rotate-6" />
      <div className="relative max-w-[1200px] mx-auto px-8 pt-10">
        <h2 className="font-heading font-bold text-[32px] leading-tight text-white">
          Hi {name}，很高兴见到你
        </h2>
        <p className="mt-2 text-[15px] text-white/80 font-body">
          今天也一起找到对的人。
        </p>
      </div>
    </section>
  );
}

function CardHeader({ title, extra }: { title: string; extra?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 pt-4 pb-3">
      <h3 className="font-body font-bold text-[14px] text-[var(--color-text-primary)]">{title}</h3>
      {extra}
    </div>
  );
}

function Stat({
  label,
  value,
  delta,
  deltaClass,
}: {
  label: string;
  value: string;
  delta: string;
  deltaClass: string;
}) {
  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-1.5 transition-shadow duration-200 hover:shadow-[var(--shadow-card-hover)]">
      <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
        {label}
      </span>
      <span className="font-heading font-bold text-[32px] leading-none py-1 text-[var(--color-text-primary)]">
        {value}
      </span>
      <span className={`font-body text-[12px] ${deltaClass}`}>{delta}</span>
    </div>
  );
}

function RecentResumes({ resumes }: { resumes: Resume[] }) {
  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] flex flex-col">
      <CardHeader title="最近简历" />
      <div className="px-3 flex-1">
      {resumes.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
          <FileText className="w-5 h-5 text-[var(--color-text-tertiary)]" />
          <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
            还没有简历
          </span>
          <Link to="/talents">
            <Button variant="secondary">去上传</Button>
          </Link>
        </div>
      ) : (
        resumes.map((r, i) => {
          const sub: string[] = [];
          if (r.candidate.display_email) sub.push(r.candidate.display_email);
          if (r.candidate.display_phone) sub.push(r.candidate.display_phone);
          return (
            <Link
              key={r.id}
              to={`/talents/${r.candidate.id}`}
              className={`flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-[var(--color-bg-muted)] transition-colors ${
                i < resumes.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <div className="flex flex-col flex-1 min-w-0 gap-0.5">
                <span className="font-medium text-sm font-body truncate">
                  {r.candidate.name}
                </span>
                <span className="text-xs text-[var(--color-text-secondary)] font-body truncate">
                  {sub.length > 0 ? sub.join(" · ") : r.file_name}
                </span>
              </div>
              <span className="text-xs text-[var(--color-text-tertiary)] font-mono shrink-0">
                {formatRelative(r.created_at)}
              </span>
            </Link>
          );
        })
      )}
      </div>
      <Link
        to="/talents"
        className="block text-center py-3 border-t border-[var(--color-border-row)] text-[13px] font-bold text-[var(--color-text-primary)] font-body hover:text-[var(--color-accent)] transition-colors"
      >
        查看全部
      </Link>
    </div>
  );
}

type Notification = {
  id: string;
  title: string;
  body: string | null;
  link: string | null;
  created_at: string;
};

type ActivityItem = {
  at: string;
  actor_name: string | null;
  verb: string;
  entity_kind: string;
  entity_label: string | null;
  link: string | null;
};

function ActivityCard() {
  const qc = useQueryClient();
  const { data: notes } = useQuery<Notification[]>({
    queryKey: ["me-notifications-unread"],
    queryFn: () => fetchJSON("/api/me/notifications?unread_only=true"),
    refetchInterval: 60_000,
  });
  const { data: feed } = useQuery<ActivityItem[]>({
    queryKey: ["activity-feed"],
    queryFn: () => fetchJSON("/api/activity?days=7&limit=12"),
  });
  const markRead = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/notifications/${id}/read`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-notifications-unread"] }),
  });
  const markAll = useMutation({
    mutationFn: () => fetchJSON("/api/me/notifications/read-all", { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-notifications-unread"] }),
  });

  const unread = notes ?? [];
  const items = feed ?? [];

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] flex flex-col pb-3">
      <CardHeader
        title="通知与动态"
        extra={
          unread.length > 0 ? (
            <button
              onClick={() => markAll.mutate()}
              className="min-w-6 h-6 px-2 rounded-full bg-[var(--color-orange)] text-white text-xs font-bold font-body grid place-items-center hover:opacity-80"
              title="全部标为已读"
            >
              {unread.length}
            </button>
          ) : undefined
        }
      />
      {unread.length === 0 && items.length === 0 ? (
        <div className="px-5 py-8 text-center text-[13px] text-[var(--color-text-tertiary)] font-body">
          暂无新动态
        </div>
      ) : (
        <>
          {unread.map((n) => {
            const inner = (
              <div className="flex flex-col flex-1 min-w-0 gap-0.5">
                <span className="text-[13px] font-body text-[var(--color-text-primary)]">{n.title}</span>
                {n.body && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body line-clamp-2">{n.body}</span>
                )}
              </div>
            );
            return (
              <div key={n.id} className="flex items-start gap-3 px-5 py-2.5 border-t border-[var(--color-border-row)]">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-[var(--color-orange)] shrink-0" />
                {n.link ? (
                  <Link
                    to={n.link.split("#")[0]}
                    className="flex-1 min-w-0 hover:bg-[var(--color-bg-muted)] rounded -mx-1 px-1"
                    onClick={() => markRead.mutate(n.id)}
                  >
                    {inner}
                  </Link>
                ) : (
                  inner
                )}
                <button
                  className="text-[11px] text-[var(--color-info)] hover:underline shrink-0"
                  onClick={() => markRead.mutate(n.id)}
                >
                  已读
                </button>
              </div>
            );
          })}
          {items.map((it, i) => {
            const inner = (
              <>
                <span className="text-[13px] font-body text-[var(--color-text-secondary)] flex-1 min-w-0 truncate">
                  {it.actor_name ? `${it.actor_name} ` : ""}
                  {it.verb}
                  {it.entity_label ? `「${it.entity_label}」` : ""}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-mono shrink-0">
                  {formatRelative(it.at)}
                </span>
              </>
            );
            return it.link ? (
              <Link
                key={i}
                to={it.link}
                className="flex items-center gap-3 px-5 py-2.5 border-t border-[var(--color-border-row)] hover:bg-[var(--color-bg-muted)]"
              >
                {inner}
              </Link>
            ) : (
              <div key={i} className="flex items-center gap-3 px-5 py-2.5 border-t border-[var(--color-border-row)]">
                {inner}
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
