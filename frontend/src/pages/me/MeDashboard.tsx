import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  CalendarDays,
  Clock,
  Goal,
  Receipt,
  UserRound,
  Wallet,
} from "lucide-react";
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
        <QuickLink to="/me/profile" icon={UserRound} label="我的资料" />
      </div>
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
