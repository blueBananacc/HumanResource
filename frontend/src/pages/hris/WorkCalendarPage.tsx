import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarOff, Plus, Save, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { type Holiday, type WorkCalendar } from "@/lib/hris";
import { Field, PageError } from "./Employees";

const WEEKDAYS = [
  { v: 1, label: "一" },
  { v: 2, label: "二" },
  { v: 3, label: "三" },
  { v: 4, label: "四" },
  { v: 5, label: "五" },
  { v: 6, label: "六" },
  { v: 7, label: "日" },
];

export function WorkCalendarPage() {
  const { hasPermission } = useAuth();
  const canManage = hasPermission(PERM.MANAGE_ATTENDANCE);
  const qc = useQueryClient();
  const year = new Date().getFullYear();
  const [yearFilter, setYearFilter] = useState(String(year));
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState("");

  const { data: cal, isError: calErr, error: calError } =
    useQuery<WorkCalendar>({
      queryKey: ["work-calendar"],
      queryFn: () => fetchJSON("/api/work-calendar"),
    });
  const { data: holidays, isLoading, isError, error } = useQuery<Holiday[]>({
    queryKey: ["holidays", yearFilter],
    queryFn: () =>
      fetchJSON(`/api/work-calendar/holidays?year=${yearFilter}`),
  });

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/work-calendar/holidays/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      setErr("");
      qc.invalidateQueries({ queryKey: ["holidays"] });
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "删除失败"),
  });

  const items = (holidays ?? []).slice().sort((a, b) => a.day.localeCompare(b.day));

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            节假日历
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            每周工作日 + 节假日/调休补班 — 请假天数、考勤汇总、加班折算都按这套日历算。
          </p>
        </div>
        {canManage && (
          <Button onClick={() => setOpen(true)}>
            <Plus className="w-4 h-4" />
            添加条目
          </Button>
        )}
      </div>

      {/* 每周工作日配置 */}
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6">
        {calErr ? (
          <PageError error={calError} />
        ) : !cal ? (
          <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (
          <CalendarForm
            key={`${cal.workdays.join()}|${cal.workday_hours}`}
            cal={cal}
            canManage={canManage}
            onSaved={() =>
              qc.invalidateQueries({ queryKey: ["work-calendar"] })
            }
          />
        )}
      </section>

      {/* 节假日条目 */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            节假日 / 调休补班
          </h3>
          <Input
            type="number"
            min="2000"
            max="2100"
            value={yearFilter}
            onChange={(e) => setYearFilter(e.target.value || String(year))}
            className="w-[110px] h-8 text-[13px]"
          />
          {err && (
            <span className="text-[13px] text-[var(--color-danger)] font-body">
              {err}
            </span>
          )}
        </div>
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
          {isLoading ? (
            <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
              加载中…
            </div>
          ) : isError ? (
            <PageError error={error} />
          ) : items.length === 0 ? (
            <Empty
              icon={CalendarOff}
              title="还没有节假日条目"
              description="法定节假日、公司假、周末调休补班都在这里维护。"
              action={
                canManage ? (
                  <Button onClick={() => setOpen(true)}>添加条目</Button>
                ) : undefined
              }
            />
          ) : (
            items.map((h, i) => (
              <div
                key={h.id}
                className={`flex items-center gap-4 px-6 py-3.5 ${
                  i < items.length - 1
                    ? "border-b border-[var(--color-border-row)]"
                    : ""
                }`}
              >
                <span className="font-mono text-[13px] text-[var(--color-text-primary)]">
                  {h.day}
                </span>
                <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                  {h.name}
                </span>
                <span
                  className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                    h.kind === "workday"
                      ? "bg-[var(--color-info-soft)] text-[var(--color-info)]"
                      : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                  }`}
                >
                  {h.kind === "workday" ? "调休补班" : "放假"}
                </span>
                <div className="flex-1" />
                {canManage && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => del.mutate(h.id)}
                    disabled={del.isPending}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                )}
              </div>
            ))
          )}
        </div>
      </section>

      <HolidayModal
        open={open}
        onClose={() => {
          setOpen(false);
          setErr("");
        }}
        onError={setErr}
      />
    </main>
  );
}

function CalendarForm({
  cal,
  canManage,
  onSaved,
}: {
  cal: WorkCalendar;
  canManage: boolean;
  onSaved: () => void;
}) {
  const [days, setDays] = useState<number[]>(cal.workdays);
  const [hours, setHours] = useState(String(cal.workday_hours));
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: () =>
      fetchJSON("/api/work-calendar", {
        method: "PUT",
        body: JSON.stringify({
          workdays: [...days].sort((a, b) => a - b),
          workday_hours: Number(hours) || 8,
        }),
      }),
    onSuccess: () => {
      setErr("");
      onSaved();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function toggle(v: number) {
    setDays((prev) =>
      prev.includes(v) ? prev.filter((d) => d !== v) : [...prev, v],
    );
  }

  return (
    <div className="flex items-end gap-6 flex-wrap">
      <div className="flex flex-col gap-2">
        <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
          每周工作日
        </span>
        <div className="flex items-center gap-1.5">
          {WEEKDAYS.map((w) => (
            <button
              key={w.v}
              type="button"
              disabled={!canManage}
              onClick={() => toggle(w.v)}
              className={`w-9 h-9 rounded-lg text-[13px] font-body border transition-colors disabled:opacity-60 ${
                days.includes(w.v)
                  ? "bg-[var(--color-text-primary)] text-white border-transparent"
                  : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border-subtle)] hover:border-[var(--color-text-tertiary)]"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>
      <Field label="日标准工时(加班折算)">
        <Input
          type="number"
          step="0.5"
          min="1"
          max="24"
          value={hours}
          onChange={(e) => setHours(e.target.value)}
          disabled={!canManage}
          className="w-[110px]"
        />
      </Field>
      {canManage && (
        <Button
          onClick={() => {
            if (days.length === 0) {
              setErr("至少保留一个工作日");
              return;
            }
            mut.mutate();
          }}
          disabled={mut.isPending}
        >
          <Save className="w-4 h-4" />
          {mut.isPending ? "保存中…" : "保存日历"}
        </Button>
      )}
      {err && (
        <span className="text-[13px] text-[var(--color-danger)] font-body">
          {err}
        </span>
      )}
      {mut.isSuccess && !err && (
        <span className="text-[13px] text-[var(--color-success)] font-body">
          已保存
        </span>
      )}
    </div>
  );
}

function HolidayModal({
  open,
  onClose,
  onError,
}: {
  open: boolean;
  onClose: () => void;
  onError: (s: string) => void;
}) {
  const qc = useQueryClient();
  const [kind, setKind] = useState("holiday");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/work-calendar/holidays", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["holidays"] });
      onClose();
    },
    onError: (e) => {
      const msg = e instanceof ApiError ? e.message : "创建失败";
      setErr(msg);
      onError(msg);
    },
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      day: String(fd.get("day")),
      name: String(fd.get("name") || "").trim(),
      kind,
    });
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="添加日历条目"
      description="放假 = 该日不计工作日;调休补班 = 周末也算工作日。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="holiday" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="holiday" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="日期" required>
          <Input name="day" type="date" required />
        </Field>
        <Field label="名称" required>
          <Input name="name" required placeholder="国庆节 / 春节调休补班" />
        </Field>
        <Field label="类型" required>
          <Select
            value={kind}
            onChange={setKind}
            options={[
              { value: "holiday", label: "放假(节假日)" },
              { value: "workday", label: "调休补班(周末上班)" },
            ]}
          />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}
