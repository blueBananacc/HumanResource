import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { CalendarDays, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { fetchJSON } from "@/lib/api";
import {
  PERSON_EVENT_DOT,
  PERSON_EVENT_TONE,
  type PersonEvent,
} from "@/lib/hris";
import { PageError } from "./Employees";

type CalendarOut = {
  month: string;
  items: PersonEvent[];
  kinds: Record<string, string>;
};

const WEEK = ["一", "二", "三", "四", "五", "六", "日"];

function monthOf(d: Date) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}
function shiftMonth(m: string, delta: number) {
  const [y, mm] = m.split("-").map(Number);
  const d = new Date(y, mm - 1 + delta, 1);
  return monthOf(d);
}

export function PeopleCalendar() {
  const todayStr = new Date().toISOString().slice(0, 10);
  const [month, setMonth] = useState(monthOf(new Date()));
  const [selected, setSelected] = useState<string | null>(todayStr);
  const { data, isLoading, isError, error } = useQuery<CalendarOut>({
    queryKey: ["people-calendar", month],
    queryFn: () => fetchJSON(`/api/hr/calendar?month=${month}`),
  });

  const byDay = useMemo(() => {
    const m = new Map<string, PersonEvent[]>();
    for (const e of data?.items ?? []) {
      const arr = m.get(e.day) ?? [];
      arr.push(e);
      m.set(e.day, arr);
    }
    return m;
  }, [data]);

  // 月网格:周一起;首格偏移 + 42 格
  const [y, mm] = month.split("-").map(Number);
  const firstDow = (new Date(y, mm - 1, 1).getDay() + 6) % 7; // Mon=0
  const daysInMonth = new Date(y, mm, 0).getDate();
  const cells: (string | null)[] = [
    ...Array<string | null>(firstDow).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) =>
      `${month}-${String(i + 1).padStart(2, "0")}`),
  ];
  while (cells.length % 7) cells.push(null);

  const selItems = selected ? byDay.get(selected) ?? [] : [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            人事日历
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            生日 / 周年 / 转正 / 合同 / 证件 / 入离职一览,HR 工作提前排布。
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="secondary" size="sm" onClick={() => setMonth(shiftMonth(month, -1))}>
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="font-mono font-semibold text-sm px-2">{month}</span>
          <Button variant="secondary" size="sm" onClick={() => setMonth(shiftMonth(month, 1))}>
            <ChevronRight className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setMonth(monthOf(new Date()))}>
            本月
          </Button>
        </div>
      </div>

      {isError ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl">
          <PageError error={error} />
        </div>
      ) : (
        <>
          <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            <div className="grid grid-cols-7 border-b border-[var(--color-border-row)]">
              {WEEK.map((w) => (
                <div
                  key={w}
                  className="py-2 text-center text-[12px] font-medium text-[var(--color-text-tertiary)] font-body"
                >
                  {w}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-7">
              {cells.map((day, i) => {
                const evs = day ? byDay.get(day) ?? [] : [];
                const isToday = day === todayStr;
                const isSel = day === selected;
                return (
                  <button
                    key={i}
                    disabled={!day}
                    onClick={() => day && setSelected(day)}
                    className={`min-h-[92px] p-1.5 text-left align-top border-b border-r border-[var(--color-border-row)] transition-colors ${
                      !day
                        ? "bg-[var(--color-bg-subtle)]"
                        : isSel
                          ? "bg-[var(--color-accent-soft)]"
                          : "hover:bg-[var(--color-bg-muted)]"
                    } ${i % 7 === 6 ? "border-r-0" : ""}`}
                  >
                    {day && (
                      <>
                        <div
                          className={`text-[12px] font-mono px-1 ${
                            isToday
                              ? "inline-flex w-6 h-6 rounded-full bg-[var(--color-accent)] text-white items-center justify-center font-semibold"
                              : "text-[var(--color-text-secondary)]"
                          }`}
                        >
                          {Number(day.slice(-2))}
                        </div>
                        <div className="mt-1 flex flex-col gap-0.5">
                          {evs.slice(0, 3).map((e, j) => (
                            <div
                              key={j}
                              className="flex items-center gap-1 px-1 min-w-0"
                              title={`${e.employee_name} · ${e.label}`}
                            >
                              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${PERSON_EVENT_DOT[e.kind] ?? "bg-slate-400"}`} />
                              <span className="text-[11px] text-[var(--color-text-secondary)] font-body truncate">
                                {e.employee_name}
                              </span>
                            </div>
                          ))}
                          {evs.length > 3 && (
                            <div className="text-[10px] text-[var(--color-text-tertiary)] px-1 font-body">
                              +{evs.length - 3} 更多
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            {Object.entries(data?.kinds ?? {}).map(([k, label]) => (
              <span key={k} className="flex items-center gap-1.5 text-[12px] text-[var(--color-text-secondary)] font-body">
                <span className={`w-2 h-2 rounded-full ${PERSON_EVENT_DOT[k] ?? "bg-slate-400"}`} />
                {label}
              </span>
            ))}
          </div>

          {selected && (
            <section className="flex flex-col gap-2">
              <h3 className="text-[13px] font-medium text-[var(--color-text-secondary)] font-body">
                {selected} · {selItems.length} 项
              </h3>
              <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
                {isLoading ? (
                  <div className="p-8 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
                ) : selItems.length === 0 ? (
                  <Empty icon={CalendarDays} title="这一天没有人事事项" />
                ) : (
                  selItems.map((e, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-3 px-5 py-3 ${
                        i < selItems.length - 1 ? "border-b border-[var(--color-border-row)]" : ""
                      }`}
                    >
                      <span className={`px-2 py-0.5 rounded-md text-[11px] font-medium font-body whitespace-nowrap ${PERSON_EVENT_TONE[e.kind] ?? "bg-slate-100 text-slate-600"}`}>
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
                      {e.meta.years ? (
                        <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                          第 {e.meta.years} 年
                        </span>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </section>
          )}
        </>
      )}
    </main>
  );
}
