import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BellRing,
  BookOpen,
  CalendarDays,
  ClipboardList,
  Clock,
  Goal,
  LifeBuoy,
  MailCheck,
  Megaphone,
  Receipt,
  Trash2,
  UserRound,
  Wallet,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { fetchJSON } from "@/lib/api";
import {
  StatusPill,
  fmtMoney,
  fmtTime,
  type MeApplication,
  type MeAttendance,
  type MeBalance,
  type MeCheckin,
  type MeProfile,
  type MeSlip,
} from "@/lib/hris";

type MeAnnouncement = {
  id: string;
  title: string;
  content: string;
  audience_name: string | null;
  published_at: string | null;
  read: boolean;
};

type Watcher = {
  id: string;
  kind: string;
  params: Record<string, unknown>;
  is_active: boolean;
  last_fired_at: string | null;
};

type ReportSub = {
  id: string;
  kind: string;
  weekday: number;
  hour: number;
  is_active: boolean;
  last_sent_at: string | null;
};

const WATCHER_LABEL: Record<string, string> = {
  candidate_stalled: "候选人停滞",
  interview_pending_report: "面试待出报告",
  case_open_hours: "受理工单超时",
  request_pending_days: "申请超期待审",
  reservation_upcoming: "预约开始提醒",
};

const REPORT_LABEL: Record<string, string> = {
  pipeline_funnel: "管道漏斗",
  weekly_summary: "每周摘要",
  pending_items: "待办提醒",
};

const WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"];

export function MeDashboard() {
  const { data: profile } = useQuery<MeProfile>({
    queryKey: ["me-profile"],
    queryFn: () => fetchJSON("/api/me/profile"),
  });
  const { data: balances, isLoading: balLoading } = useQuery<MeBalance[]>({
    queryKey: ["me-balances"],
    queryFn: () => fetchJSON("/api/me/leave-balances"),
  });
  const { data: today, isLoading: todayLoading } = useQuery<MeAttendance[]>({
    queryKey: ["me-attendance-today"],
    queryFn: () => {
      const m = new Date().toISOString().slice(0, 7);
      return fetchJSON(`/api/me/attendance?month=${m}`);
    },
  });
  const { data: checkins, isLoading: checkinLoading } = useQuery<MeCheckin[]>({
    queryKey: ["me-checkins"],
    queryFn: () => fetchJSON("/api/me/checkins"),
  });
  const { data: slips, isLoading: slipsLoading } = useQuery<MeSlip[]>({
    queryKey: ["me-slips"],
    queryFn: () => fetchJSON("/api/me/salary-slips"),
  });
  const { data: apps, isLoading: appsLoading } = useQuery<MeApplication[]>({
    queryKey: ["me-leave-apps"],
    queryFn: () => fetchJSON("/api/me/leave-applications"),
  });
  const { data: anns, isLoading: annsLoading } = useQuery<MeAnnouncement[]>({
    queryKey: ["me-announcements"],
    queryFn: () => fetchJSON("/api/me/announcements"),
  });
  const qc = useQueryClient();
  const markRead = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/announcements/${id}/read`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-announcements"] }),
  });

  const todayStr = new Date().toISOString().slice(0, 10);
  const todayAtt = (today ?? []).find((a) => a.day === todayStr);
  const todayCheckins = (checkins ?? []).filter((c) =>
    c.ts.startsWith(todayStr),
  );
  const pending = (apps ?? []).filter((a) => a.status === "pending");

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          {profile ? `${profile.name},你好` : "我的工作台"}
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          {profile?.designation ?? ""}
          {profile?.org_unit_name ? ` · ${profile.org_unit_name}` : ""}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {(anns ?? []).length > 0 && (
          <section className="col-span-2 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Megaphone className="w-4 h-4 text-[var(--color-accent)]" />
              <h3 className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)]">
                公司公告
              </h3>
              {(anns ?? []).some((a) => !a.read) && (
                <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-red-50 text-red-600">
                  {(anns ?? []).filter((a) => !a.read).length} 条未读
                </span>
              )}
            </div>
            {annsLoading ? (
              <CardLoading />
            ) : (
              <div className="flex flex-col gap-2.5">
                {(anns ?? []).slice(0, 3).map((a) => (
                  <div key={a.id} className="flex items-start gap-3">
                    <span
                      className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${a.read ? "bg-[var(--color-border-default)]" : "bg-[var(--color-accent)]"}`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-[13px] text-[var(--color-text-primary)] font-body">
                          {a.title}
                        </span>
                        {a.audience_name && (
                          <span className="text-[11px] text-[var(--color-text-tertiary)]">
                            {a.audience_name}
                          </span>
                        )}
                      </div>
                      <p className="text-[12px] text-[var(--color-text-secondary)] font-body line-clamp-2 whitespace-pre-wrap">
                        {a.content}
                      </p>
                    </div>
                    {!a.read && (
                      <button
                        className="text-[11px] text-[var(--color-info)] hover:underline shrink-0"
                        onClick={() => markRead.mutate(a.id)}
                      >
                        标为已读
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
        <Card
          icon={CalendarDays}
          title="假期余额"
          to="/me/leave"
          action="去请假"
        >
          {balLoading ? (
            <CardLoading />
          ) : (balances ?? []).length === 0 ? (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无额度
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {(balances ?? []).slice(0, 3).map((b) => (
                <div
                  key={b.leave_type_id}
                  className="flex items-center justify-between text-[13px] font-body"
                >
                  <span className="text-[var(--color-text-secondary)]">
                    {b.leave_type_name}
                  </span>
                  <span className="font-mono font-semibold text-[var(--color-text-primary)]">
                    {b.balance} 天
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card icon={Clock} title="今日考勤" to="/me/attendance" action="去打卡">
          {todayLoading || checkinLoading ? (
            <CardLoading />
          ) : todayAtt ? (
            <div className="flex flex-col gap-1.5 text-[13px] font-body">
              <div className="flex items-center gap-2">
                <StatusPill status={todayAtt.status} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--color-text-secondary)]">上班</span>
                <span className="font-mono">{fmtTime(todayAtt.first_in)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[var(--color-text-secondary)]">下班</span>
                <span className="font-mono">{fmtTime(todayAtt.last_out)}</span>
              </div>
            </div>
          ) : (
            <div className="text-[13px] font-body text-[var(--color-text-secondary)]">
              {todayCheckins.length > 0
                ? `已打卡 ${todayCheckins.length} 次`
                : "今日还没打卡"}
            </div>
          )}
        </Card>

        <Card
          icon={Wallet}
          title="最近工资条"
          to="/me/payslips"
          action="查看全部"
        >
          {slipsLoading ? (
            <CardLoading />
          ) : (slips ?? []).length === 0 ? (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无工资条
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {(slips ?? []).slice(0, 3).map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between text-[13px] font-body"
                >
                  <span className="font-mono text-[var(--color-text-secondary)]">
                    {s.period}
                  </span>
                  <span className="font-mono font-semibold text-[var(--color-text-primary)]">
                    {fmtMoney(s.net, s.currency)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card
          icon={Receipt}
          title="请假审批中"
          to="/me/leave"
          action="查看"
        >
          {appsLoading ? (
            <CardLoading />
          ) : pending.length === 0 ? (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              没有在途申请
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {pending.slice(0, 3).map((a) => (
                <div
                  key={a.id}
                  className="flex items-center justify-between text-[13px] font-body"
                >
                  <span className="text-[var(--color-text-secondary)]">
                    {a.leave_type_name} · {a.days} 天
                  </span>
                  <StatusPill status={a.status} />
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <QuickLink to="/me/goals" icon={Goal} label="我的目标" />
        <QuickLink to="/me/expenses" icon={Receipt} label="我的报销" />
        <QuickLink to="/me/requests" icon={ClipboardList} label="我的申请" />
        <QuickLink to="/me/cases" icon={LifeBuoy} label="我的工单" />
        <QuickLink to="/me/profile" icon={UserRound} label="我的资料" />
      </div>

      <KnowledgeCard />
      <RemindersCard />
    </main>
  );
}

function CardLoading() {
  return (
    <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
      加载中…
    </div>
  );
}

/**
 * 提醒与订阅管理:列出本人创建的 watcher + 报表订阅,支持启停/删除。
 * 创建入口在 AI 助手对话里(“候选人 3 天没动提醒我”),这里负责管理。
 */
function RemindersCard() {
  const qc = useQueryClient();
  const { data: watchers } = useQuery<Watcher[]>({
    queryKey: ["me-watchers"],
    queryFn: () => fetchJSON("/api/me/watchers"),
  });
  const { data: subs } = useQuery<ReportSub[]>({
    queryKey: ["me-report-subs"],
    queryFn: () => fetchJSON("/api/me/report-subscriptions"),
  });
  const toggleWatcher = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/me/watchers/${id}/toggle`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-watchers"] }),
  });
  const delWatcher = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/me/watchers/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-watchers"] }),
  });
  const toggleSub = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/me/report-subscriptions/${id}/toggle`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-report-subs"] }),
  });
  const delSub = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/me/report-subscriptions/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["me-report-subs"] }),
  });

  const ws = watchers ?? [];
  const ss = subs ?? [];
  if (ws.length === 0 && ss.length === 0) return null;

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <BellRing className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)]">
          提醒与订阅
        </h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          由 AI 助手创建,在此启停或删除
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {ws.map((w) => (
          <div key={w.id} className="flex items-center gap-3 text-[13px] font-body">
            <BellRing className={`w-3.5 h-3.5 shrink-0 ${w.is_active ? "text-[var(--color-accent)]" : "text-[var(--color-text-tertiary)]"}`} />
            <span className={w.is_active ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-tertiary)] line-through"}>
              {WATCHER_LABEL[w.kind] ?? w.kind}
            </span>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">
              {Object.entries(w.params ?? {}).map(([k, v]) => `${k}=${v}`).join(",")}
            </span>
            <div className="flex-1" />
            <button
              className="text-[12px] text-[var(--color-info)] hover:underline"
              onClick={() => toggleWatcher.mutate(w.id)}
            >
              {w.is_active ? "停用" : "启用"}
            </button>
            <button
              className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
              onClick={() => delWatcher.mutate(w.id)}
              aria-label="删除提醒"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
        {ss.map((s) => (
          <div key={s.id} className="flex items-center gap-3 text-[13px] font-body">
            <MailCheck className={`w-3.5 h-3.5 shrink-0 ${s.is_active ? "text-[var(--color-accent)]" : "text-[var(--color-text-tertiary)]"}`} />
            <span className={s.is_active ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-tertiary)] line-through"}>
              {REPORT_LABEL[s.kind] ?? s.kind}
            </span>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">
              每周{WEEKDAYS[s.weekday] ?? s.weekday} {String(s.hour).padStart(2, "0")}:00 发邮件
            </span>
            <div className="flex-1" />
            <button
              className="text-[12px] text-[var(--color-info)] hover:underline"
              onClick={() => toggleSub.mutate(s.id)}
            >
              {s.is_active ? "停用" : "启用"}
            </button>
            <button
              className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
              onClick={() => delSub.mutate(s.id)}
              aria-label="删除订阅"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

type KbHit = {
  id: string;
  title: string;
  category: string;
  tags: string[];
  snippet: string;
};

const KB_CAT: Record<string, string> = {
  policy: "制度",
  faq: "FAQ",
  benefit: "福利",
  it: "IT",
  onboarding: "入职",
  other: "其他",
};

/** ESS 知识库搜索卡:员工自助查制度/FAQ,只命中已发布文章
 * (后端 /api/me/knowledge 空 query 时返回已发布列表)。 */
function KnowledgeCard() {
  const [q, setQ] = useState("");
  const [dq, setDq] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDq(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);
  const { data, isLoading } = useQuery<KbHit[]>({
    queryKey: ["me-knowledge", dq],
    queryFn: () =>
      fetchJSON(`/api/me/knowledge${dq ? `?q=${encodeURIComponent(dq)}` : ""}`),
  });
  const hits = data ?? [];

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <BookOpen className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)]">
          知识库
        </h3>
        <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
          制度 / FAQ / 福利
        </span>
      </div>
      <Input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="搜索:年假、报销、VPN…"
      />
      {isLoading ? (
        <CardLoading />
      ) : hits.length === 0 ? (
        <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          {dq ? "没有匹配的文章" : "还没有已发布的文章"}
        </div>
      ) : (
        <div className="flex flex-col gap-2.5">
          {hits.slice(0, 5).map((h) => (
            <div key={h.id}>
              <div className="flex items-center gap-2">
                <span className="font-medium text-[13px] text-[var(--color-text-primary)] font-body">
                  {h.title}
                </span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]">
                  {KB_CAT[h.category] ?? h.category}
                </span>
              </div>
              <p className="mt-0.5 text-[12px] text-[var(--color-text-secondary)] font-body line-clamp-2">
                {h.snippet}
              </p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function Card({
  icon: Icon,
  title,
  to,
  action,
  children,
}: {
  icon: typeof Clock;
  title: string;
  to: string;
  action: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-[var(--color-accent)]" />
          <h3 className="font-heading font-semibold text-[14px] text-[var(--color-text-primary)]">
            {title}
          </h3>
        </div>
        <Link
          to={to}
          className="text-[12px] font-body text-[var(--color-info)] hover:underline"
        >
          {action} →
        </Link>
      </div>
      {children}
    </section>
  );
}

function QuickLink({
  to,
  icon: Icon,
  label,
}: {
  to: string;
  icon: typeof Clock;
  label: string;
}) {
  return (
    <Link
      to={to}
      className="flex items-center gap-3 px-5 py-4 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] hover:border-[var(--color-accent)] transition-colors"
    >
      <Icon className="w-4 h-4 text-[var(--color-accent)]" />
      <span className="font-medium text-[13px] font-body text-[var(--color-text-primary)]">
        {label}
      </span>
    </Link>
  );
}
