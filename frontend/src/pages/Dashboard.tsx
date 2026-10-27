import { Sparkles, FileText, Eye, EyeOff, Trash2, ArrowUpRight, Check } from "lucide-react";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { BrandShape } from "@/components/brand/BrandShape";
import { BriefingCard } from "@/components/dashboard/BriefingCard";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
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
  is_read: boolean;
  is_dismissed: boolean;
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

  // 面试/报告计数:?limit=1 只要 total。已生成报告 = 完成的 AI 面试
  const { data: activeInterviewsPage } = useQuery({
    queryKey: ["interviews", "in-progress-count"],
    queryFn: () =>
      fetchJSON<{ items: unknown[]; total: number }>(
        "/api/interviews/?status=in_progress&limit=1",
      ),
  });
  const { data: doneInterviewsPage } = useQuery({
    queryKey: ["interviews", "done-count"],
    queryFn: () =>
      fetchJSON<{ items: unknown[]; total: number }>(
        "/api/interviews/?status=done&limit=1",
      ),
  });

  const totalResumes = resumesPage?.total ?? 0;
  const openJobs = openJobsPage?.total ?? 0;
  const allJobsCount = allJobsPage?.total ?? 0;
  const activeInterviews = activeInterviewsPage?.total ?? 0;
  const doneInterviews = doneInterviewsPage?.total ?? 0;

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

      <BriefingCard />

      <div className="grid grid-cols-4 gap-4">
        <Stat
          label="简历总数"
          value={String(totalResumes)}
          delta={totalResumes > 0 ? "本租户全部" : "等待第一份简历"}
          deltaClass="text-[var(--color-text-tertiary)]"
          to="/talents"
        />
        <Stat
          label="开放岗位"
          value={String(openJobs)}
          delta={`共 ${allJobsCount} 个岗位`}
          deltaClass="text-[var(--color-text-tertiary)]"
          to="/jobs"
        />
        <Stat
          label="进行中面试"
          value={String(activeInterviews)}
          delta="AI 面试会话"
          deltaClass="text-[var(--color-text-tertiary)]"
          to="/interviews"
        />
        <Stat
          label="已生成报告"
          value={String(doneInterviews)}
          delta="已完成的 AI 面试"
          deltaClass="text-[var(--color-text-tertiary)]"
          to="/reports"
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
  to,
}: {
  label: string;
  value: string;
  delta: string;
  deltaClass: string;
  to?: string;
}) {
  const cls =
    "group bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-1.5 transition-all duration-200 hover:shadow-[var(--shadow-card-hover)]" +
    (to ? " hover:border-[var(--color-border-strong)]" : "");
  const body = (
    <>
      <span className="flex items-center justify-between gap-2">
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          {label}
        </span>
        {to && (
          <ArrowUpRight className="w-4 h-4 text-[var(--color-text-tertiary)] opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </span>
      <span className="font-heading font-bold text-[32px] leading-none py-1 text-[var(--color-text-primary)]">
        {value}
      </span>
      <span className={`font-body text-[12px] ${deltaClass}`}>{delta}</span>
    </>
  );
  if (to) {
    return (
      <Link to={to} className={cls} title={`查看${label}`}>
        {body}
      </Link>
    );
  }
  return <div className={cls}>{body}</div>;
}

/** Bob 风格头像色块:按候选人 id 取一个暖色底,首字符居中 */
const AVATAR_TONES = [
  { bg: "#FAA32B", fg: "#60032A" },
  { bg: "#F36488", fg: "#FFFFFF" },
  { bg: "#FEABB0", fg: "#83143D" },
  { bg: "#FFD887", fg: "#60032A" },
];

function avatarTone(seed: string) {
  let h = 0;
  for (const ch of seed) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return AVATAR_TONES[h % AVATAR_TONES.length];
}

function RecentResumes({ resumes }: { resumes: Resume[] }) {
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  const canDelete = hasPermission(PERM.DELETE_RESUMES);
  const [pendingDelete, setPendingDelete] = useState<Resume | null>(null);

  const invalidate = () => qc.invalidateQueries({ queryKey: ["resumes"] });
  const toggleRead = useMutation({
    mutationFn: (r: Resume) =>
      fetchJSON(`/api/resumes/${r.id}/${r.is_read ? "unread" : "read"}`, {
        method: "POST",
      }),
    onSuccess: invalidate,
  });
  const dismiss = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/resumes/${id}/dismiss`, { method: "POST" }),
    onSuccess: () => {
      setPendingDelete(null);
      invalidate();
    },
  });
  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/resumes/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      setPendingDelete(null);
      invalidate();
    },
  });

  // 「仅移除此条」不删数据,只是从本部件隐藏
  const shown = resumes.filter((r) => !r.is_dismissed);

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] flex flex-col">
      <CardHeader title="最近简历" />
      <div className="px-3 flex-1">
      {shown.length === 0 ? (
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
        shown.map((r, i) => {
          const sub: string[] = [];
          if (r.candidate.display_email) sub.push(r.candidate.display_email);
          if (r.candidate.display_phone) sub.push(r.candidate.display_phone);
          const tone = avatarTone(r.candidate.id);
          return (
            <div
              key={r.id}
              className={`group flex items-center gap-3 px-3 py-3 transition-colors ${
                i < shown.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <Link
                to={`/talents/${r.candidate.id}`}
                className="flex items-center gap-3 flex-1 min-w-0 rounded-lg"
              >
                <span
                  className="w-9 h-9 rounded-full grid place-items-center shrink-0 font-heading font-semibold text-[15px]"
                  style={{ background: tone.bg, color: tone.fg }}
                >
                  {(r.candidate.name || "?").slice(0, 1)}
                </span>
                <div className="flex flex-col flex-1 min-w-0 gap-0.5">
                  <span className="flex items-center gap-2 min-w-0">
                    <span
                      className={`font-medium text-sm font-body truncate ${
                        r.is_read ? "text-[var(--color-text-tertiary)]" : ""
                      }`}
                    >
                      {r.candidate.name}
                    </span>
                    {!r.is_read && (
                      <span className="px-1.5 py-0.5 rounded-full bg-[var(--color-bg-muted)] text-[var(--color-orange)] text-[10px] font-bold font-body shrink-0">
                        新
                      </span>
                    )}
                  </span>
                  <span className="text-xs text-[var(--color-text-secondary)] font-body truncate">
                    {sub.length > 0 ? sub.join(" · ") : r.file_name}
                  </span>
                </div>
              </Link>
              <span className="text-xs text-[var(--color-text-tertiary)] font-mono shrink-0">
                {formatRelative(r.created_at)}
              </span>
              <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                <button
                  onClick={() => toggleRead.mutate(r)}
                  disabled={toggleRead.isPending}
                  title={r.is_read ? "标为未读" : "标为已读"}
                  className="w-7 h-7 rounded-lg grid place-items-center text-[var(--color-text-tertiary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)] transition-colors disabled:opacity-50"
                >
                  {r.is_read ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
                {canDelete && (
                  <button
                    onClick={() => setPendingDelete(r)}
                    title="删除"
                    className="w-7 h-7 rounded-lg grid place-items-center text-[var(--color-text-tertiary)] hover:bg-red-50 hover:text-[var(--color-danger)] transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
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

      <Modal
        open={pendingDelete !== null}
        onClose={() => setPendingDelete(null)}
        title={`删除「${pendingDelete?.candidate.name ?? ""}」的简历?`}
        description="「仅移除此条」只把它从概览里撤掉,简历和候选人档案仍在人才库;「删除简历」会彻底清除数据,不可恢复。"
        width={460}
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => pendingDelete && dismiss.mutate(pendingDelete.id)}
              disabled={dismiss.isPending}
            >
              仅移除此条
            </Button>
            <button
              type="button"
              onClick={() => pendingDelete && del.mutate(pendingDelete.id)}
              disabled={del.isPending}
              className="inline-flex items-center justify-center gap-2 rounded-[10px] px-5 py-2.5 text-sm font-medium font-body transition-colors bg-[var(--color-danger)] text-white hover:bg-[#dc2626] disabled:opacity-50"
            >
              删除简历
            </button>
          </>
        }
      >
        <div className="py-2" />
      </Modal>
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
  id: string;
  at: string;
  actor_name: string | null;
  verb: string;
  entity_kind: string;
  entity_label: string | null;
  link: string | null;
  via: string | null;
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
  // 动态流「已读/移除此条」:只从本人小部件隐藏,审计/业务数据不动
  const dismissActivity = useMutation({
    mutationFn: (itemKey: string) =>
      fetchJSON("/api/activity/dismiss", {
        method: "POST",
        body: JSON.stringify({ item_key: itemKey }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["activity-feed"] }),
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
              <div key={n.id} className="group flex items-start gap-3 px-5 py-2.5 border-t border-[var(--color-border-row)]">
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
                  className="w-6 h-6 mt-0.5 rounded-lg grid place-items-center shrink-0 text-[var(--color-text-tertiary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)] transition-colors opacity-0 group-hover:opacity-100 focus-within:opacity-100"
                  onClick={() => markRead.mutate(n.id)}
                  title="标为已读"
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
          {items.map((it, i) => {
            const inner = (
              <>
                <span className="text-[13px] font-body text-[var(--color-text-secondary)] flex-1 min-w-0 truncate">
                  {it.actor_name ? `${it.actor_name} ` : ""}
                  {it.via === "assistant" && (
                    <span className="px-1.5 py-0.5 mr-1 rounded bg-violet-50 text-violet-700 text-[10px] font-normal">
                      AI 助手
                    </span>
                  )}
                  {it.via === "mcp" && (
                    <span className="px-1.5 py-0.5 mr-1 rounded bg-sky-50 text-sky-700 text-[10px] font-normal">
                      MCP
                    </span>
                  )}
                  {it.verb}
                  {it.entity_label ? `「${it.entity_label}」` : ""}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-mono shrink-0">
                  {formatRelative(it.at)}
                </span>
              </>
            );
            return (
              <div
                key={it.id ?? i}
                className="group flex items-center gap-3 px-5 py-2.5 border-t border-[var(--color-border-row)]"
              >
                {it.link ? (
                  <Link
                    to={it.link}
                    className="flex items-center gap-3 flex-1 min-w-0 rounded -mx-1 px-1 hover:bg-[var(--color-bg-muted)]"
                  >
                    {inner}
                  </Link>
                ) : (
                  <div className="flex items-center gap-3 flex-1 min-w-0">{inner}</div>
                )}
                <button
                  className="w-6 h-6 rounded-lg grid place-items-center shrink-0 text-[var(--color-text-tertiary)] hover:bg-[var(--color-bg-muted)] hover:text-[var(--color-text-primary)] transition-colors opacity-0 group-hover:opacity-100 focus-within:opacity-100"
                  onClick={() => dismissActivity.mutate(it.id)}
                  disabled={dismissActivity.isPending}
                  title="标为已读并移除"
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
