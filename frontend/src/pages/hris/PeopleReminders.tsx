import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BellRing, Cake, PartyPopper } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  PERSON_EVENT_TONE,
  type PersonEvent,
} from "@/lib/hris";
import { PageError } from "./Employees";

type RemindersOut = {
  overdue: PersonEvent[];
  today: PersonEvent[];
  week: PersonEvent[];
  later: PersonEvent[];
  type_counts: Record<string, number>;
  total: number;
  kinds: Record<string, string>;
};

const ALL_KINDS = [
  "birthday", "anniversary", "probation_end",
  "contract_end", "cert_expire", "onboarding", "separation",
];

export function PeopleReminders() {
  const [kind, setKind] = useState<string>("");
  const [err, setErr] = useState("");
  const qc = useQueryClient();
  const { data, isLoading, isError, error } = useQuery<RemindersOut>({
    queryKey: ["people-reminders", kind],
    queryFn: () =>
      fetchJSON(`/api/hr/reminders?days=30${kind ? `&kind=${kind}` : ""}`),
  });

  const greet = useMutation({
    mutationFn: (e: PersonEvent) =>
      fetchJSON("/api/hr/reminders/greet", {
        method: "POST",
        body: JSON.stringify({ employee_id: e.employee_id, kind: e.kind }),
      }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["people-reminders"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "发送失败"),
  });

  const d = data;
  const stats = [
    { label: "今天", n: d?.today.length ?? 0, tone: "text-[var(--color-danger)]" },
    { label: "本周", n: d?.week.length ?? 0, tone: "text-[var(--color-warning-text)]" },
    { label: "30 天内", n: d?.later.length ?? 0, tone: "text-[var(--color-info)]" },
    { label: "已逾期", n: d?.overdue.length ?? 0, tone: "text-[var(--color-danger)]" },
  ];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          人事提醒
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          生日 / 入职周年 / 转正 / 合同 / 证件到期 / 入离职 —— 每日自动巡检并推送通知。
        </p>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {stats.map((s) => (
          <div
            key={s.label}
            className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] px-4 py-3"
          >
            <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
              {s.label}
            </div>
            <div className={`font-mono font-semibold text-xl ${s.tone}`}>
              {s.n}
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <KindChip
          active={!kind}
          label={`全部 ${d?.total ?? ""}`}
          onClick={() => setKind("")}
        />
        {ALL_KINDS.map((k) => (
          <KindChip
            key={k}
            active={kind === k}
            label={`${d?.kinds[k] ?? k} ${d?.type_counts[k] ?? 0}`}
            onClick={() => setKind(kind === k ? "" : k)}
          />
        ))}
      </div>

      {err && (
        <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>
      )}

      {isLoading ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : isError ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl">
          <PageError error={error} />
        </div>
      ) : !d || d.total === 0 ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={BellRing}
            title="30 天内没有人事提醒"
            description="在员工资料 personal 里维护生日/转正/合同/证件日期后自动聚合。"
          />
        </div>
      ) : (
        <>
          <Group title="已逾期" danger events={d.overdue} greet={greet} />
          <Group title="今天" highlight events={d.today} greet={greet} />
          <Group title="本周" events={d.week} greet={greet} />
          <Group title="之后 30 天内" events={d.later} greet={greet} />
        </>
      )}
    </main>
  );
}

function KindChip({
  active, label, onClick,
}: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-full text-[12px] font-body border transition-colors ${
        active
          ? "bg-[var(--color-accent)] text-white border-[var(--color-accent)]"
          : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-muted)]"
      }`}
    >
      {label}
    </button>
  );
}

function Group({
  title, events, greet, danger, highlight,
}: {
  title: string;
  events: PersonEvent[];
  greet: ReturnType<typeof useMutation<unknown, Error, PersonEvent>>;
  danger?: boolean;
  highlight?: boolean;
}) {
  if (!events.length) return null;
  return (
    <section className="flex flex-col gap-2">
      <h3
        className={`text-[13px] font-medium font-body ${
          danger
            ? "text-[var(--color-danger)]"
            : highlight
              ? "text-[var(--color-accent)]"
              : "text-[var(--color-text-secondary)]"
        }`}
      >
        {title} · {events.length}
      </h3>
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {events.map((e, i) => (
          <Row
            key={`${e.employee_id}-${e.kind}-${e.day}`}
            e={e}
            last={i === events.length - 1}
            greet={greet}
          />
        ))}
      </div>
    </section>
  );
}

function Row({
  e, last, greet,
}: {
  e: PersonEvent;
  last: boolean;
  greet: ReturnType<typeof useMutation<unknown, Error, PersonEvent>>;
}) {
  const when =
    e.days_until < 0
      ? `已逾期 ${-e.days_until} 天`
      : e.days_until === 0
        ? "今天"
        : `${e.days_until} 天后`;
  const celebrate = e.category === "celebrate";
  const extra =
    e.kind === "anniversary" && e.meta.years
      ? `第 ${e.meta.years} 年`
      : e.kind === "birthday" && e.meta.age
        ? `${e.meta.age} 岁`
        : null;
  return (
    <div
      className={`flex items-center gap-3 px-5 py-3.5 ${
        last ? "" : "border-b border-[var(--color-border-row)]"
      }`}
    >
      <span
        className={`px-2 py-0.5 rounded-md text-[11px] font-medium font-body whitespace-nowrap ${PERSON_EVENT_TONE[e.kind] ?? "bg-slate-100 text-slate-600"}`}
      >
        {e.label}
      </span>
      <Link
        to={`/employees/${e.employee_id}`}
        className="font-medium text-sm text-[var(--color-text-primary)] font-body hover:text-[var(--color-accent)]"
      >
        {e.employee_name}
      </Link>
      {e.org_unit_name && (
        <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
          {e.org_unit_name}
        </span>
      )}
      {extra && (
        <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
          {extra}
        </span>
      )}
      <div className="flex-1" />
      <span className="font-mono text-[12px] text-[var(--color-text-secondary)]">
        {e.day}
      </span>
      <span
        className={`text-[12px] font-body ${
          e.days_until <= 0
            ? "text-[var(--color-danger)] font-medium"
            : "text-[var(--color-text-tertiary)]"
        }`}
      >
        {when}
      </span>
      {celebrate && (
        <Button
          variant="secondary"
          size="sm"
          disabled={greet.isPending}
          onClick={() => greet.mutate(e)}
          title="推送祝福通知给员工本人"
        >
          {e.kind === "birthday" ? (
            <Cake className="w-3.5 h-3.5" />
          ) : (
            <PartyPopper className="w-3.5 h-3.5" />
          )}
          发祝福
        </Button>
      )}
    </div>
  );
}
