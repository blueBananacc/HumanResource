import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  DoorOpen,
  Package,
  Plus,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { ApiError, fetchJSON } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";

/**
 * 面试安排 · 排期页(T2-2,MintHCM Reservations/ReservationsCalendar 移植)。
 *
 * 一周日历:每天一列,09:00-19:00 时间轴;预约为定位块,
 * 已取消灰色虚线。顶部可筛会议室,+ 新建预约弹窗
 * (选会议室/资源 + 时间 + 面试官勾选带冲突提示)。
 */

type Room = {
  id: string;
  name: string;
  location: string | null;
  capacity: number;
  is_active: boolean;
};

type Resource = { id: string; name: string; type: string; is_active: boolean };

type Reservation = {
  id: string;
  kind: string;
  room_id: string | null;
  room_name: string | null;
  resource_id: string | null;
  resource_name: string | null;
  title: string;
  starts_at: string;
  ends_at: string;
  interviewer_names: string[];
  status: string;
};

type Availability = {
  user_id: string;
  name: string;
  role: string;
  busy_count: number;
};

const DAY_START = 9;
const DAY_END = 19;
const HOUR_PX = 56;

function startOfWeek(d: Date) {
  const x = new Date(d);
  const dow = (x.getDay() + 6) % 7; // 周一 = 0
  x.setDate(x.getDate() - dow);
  x.setHours(0, 0, 0, 0);
  return x;
}

function isoLocal(d: Date) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:00`;
}

/** DB 存本地 naive 时间,但 API 序列化带 Z — 去掉 Z 按本地解析,与录入语义一致 */
function toLocal(s: string) {
  return new Date(s.replace(/Z$/, ""));
}

export function Schedule() {
  const qc = useQueryClient();
  const { hasPermission } = useAuth();
  // 后端 POST/PUT/cancel 要求 write_interview — 与后端权限保持一致
  const canWrite = hasPermission(PERM.WRITE_INTERVIEW);
  const [weekStart, setWeekStart] = useState(() => startOfWeek(new Date()));
  const [roomFilter, setRoomFilter] = useState<string>("");
  const [showCreate, setShowCreate] = useState(false);

  const weekEnd = useMemo(() => {
    const d = new Date(weekStart);
    d.setDate(d.getDate() + 7);
    return d;
  }, [weekStart]);
  const weekEndLabel = useMemo(() => {
    const d = new Date(weekEnd);
    d.setDate(d.getDate() - 1);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  }, [weekEnd]);

  const resvQuery = useQuery({
    queryKey: ["reservations", weekStart.toISOString(), roomFilter],
    queryFn: () =>
      fetchJSON<Reservation[]>(
        `/api/reservations?start=${encodeURIComponent(isoLocal(weekStart))}` +
          `&end=${encodeURIComponent(isoLocal(weekEnd))}` +
          (roomFilter ? `&room_id=${roomFilter}` : ""),
      ),
  });
  const roomsQuery = useQuery({
    queryKey: ["rooms"],
    queryFn: () => fetchJSON<Room[]>("/api/rooms"),
  });
  const items = resvQuery.data ?? [];

  const days = useMemo(
    () =>
      Array.from({ length: 7 }, (_, i) => {
        const d = new Date(weekStart);
        d.setDate(d.getDate() + i);
        return d;
      }),
    [weekStart],
  );

  function shift(deltaWeeks: number) {
    setWeekStart((w) => {
      const d = new Date(w);
      d.setDate(d.getDate() + deltaWeeks * 7);
      return d;
    });
  }

  const cancel = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/reservations/${id}/cancel`, { method: "POST" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["reservations"] }),
  });

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1400px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            面试安排
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            会议室与资源预约 · 面试官时间协调
          </p>
        </div>
        {canWrite && (
          <Button onClick={() => setShowCreate(true)}>
            <Plus className="w-4 h-4" />
            新建预约
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => shift(-1)}
            className="p-1.5 rounded-md hover:bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
            aria-label="上一周"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setWeekStart(startOfWeek(new Date()))}
            className="px-2.5 py-1 rounded-md text-[12px] font-body text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-subtle)]"
          >
            本周
          </button>
          <button
            type="button"
            onClick={() => shift(1)}
            className="p-1.5 rounded-md hover:bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
            aria-label="下一周"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
        <span className="text-sm font-medium text-[var(--color-text-primary)] font-body">
          {weekStart.getMonth() + 1}/{weekStart.getDate()} –{" "}
          {weekEndLabel}
        </span>
        <select
          value={roomFilter}
          onChange={(e) => setRoomFilter(e.target.value)}
          className="ml-auto h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-[13px] font-body text-[var(--color-text-secondary)]"
        >
          <option value="">全部会议室/资源</option>
          {(roomsQuery.data ?? [])
            .filter((r) => r.is_active)
            .map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
                {r.location ? `(${r.location})` : ""}
              </option>
            ))}
        </select>
      </div>

      {/* 周视图 */}
      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-x-auto">
        <div className="min-w-[900px]">
          {/* 表头 */}
          <div className="grid grid-cols-[56px_repeat(7,1fr)] border-b border-[var(--color-border-subtle)]">
            <div />
            {days.map((d, i) => {
              const isToday = d.getTime() === today.getTime();
              return (
                <div
                  key={i}
                  className={`px-2 py-2.5 text-center border-l border-[var(--color-border-row)] ${
                    isToday ? "bg-[var(--color-accent-soft,rgba(238,22,79,0.04))]" : ""
                  }`}
                >
                  <div
                    className={`text-[12px] font-body ${isToday ? "text-[var(--color-accent)] font-medium" : "text-[var(--color-text-tertiary)]"}`}
                  >
                    周{"日一二三四五六".charAt(d.getDay())}
                  </div>
                  <div
                    className={`text-sm font-heading font-semibold ${isToday ? "text-[var(--color-accent)]" : "text-[var(--color-text-primary)]"}`}
                  >
                    {d.getMonth() + 1}/{d.getDate()}
                  </div>
                </div>
              );
            })}
          </div>

          {/* 时间格 */}
          <div className="grid grid-cols-[56px_repeat(7,1fr)] relative">
            {/* 小时刻度 */}
            <div className="relative" style={{ height: (DAY_END - DAY_START) * HOUR_PX }}>
              {Array.from({ length: DAY_END - DAY_START }, (_, i) => (
                <div
                  key={i}
                  className="absolute right-2 text-[10px] font-mono text-[var(--color-text-tertiary)] -translate-y-1/2"
                  style={{ top: i * HOUR_PX }}
                >
                  {DAY_START + i}:00
                </div>
              ))}
            </div>
            {days.map((d, di) => {
              const dayStart = new Date(d);
              dayStart.setHours(DAY_START, 0, 0, 0);
              const dayEnd = new Date(d);
              dayEnd.setHours(DAY_END, 0, 0, 0);
              const dayItems = items.filter((r) => {
                const s = toLocal(r.starts_at);
                const e = toLocal(r.ends_at);
                return s < dayEnd && e > dayStart;
              });
              const isToday = d.getTime() === today.getTime();
              return (
                <div
                  key={di}
                  className={`relative border-l border-[var(--color-border-row)] ${
                    isToday ? "bg-[var(--color-accent-soft,rgba(238,22,79,0.03))]" : ""
                  }`}
                  style={{ height: (DAY_END - DAY_START) * HOUR_PX }}
                >
                  {Array.from({ length: DAY_END - DAY_START }, (_, i) => (
                    <div
                      key={i}
                      className="absolute w-full border-t border-[var(--color-border-row)]"
                      style={{ top: i * HOUR_PX }}
                    />
                  ))}
                  {dayItems.map((r) => {
                    const s = toLocal(r.starts_at);
                    const e = toLocal(r.ends_at);
                    const top =
                      ((Math.max(s.getTime(), dayStart.getTime()) -
                        dayStart.getTime()) /
                        3600000) *
                      HOUR_PX;
                    const h =
                      ((Math.min(e.getTime(), dayEnd.getTime()) -
                        Math.max(s.getTime(), dayStart.getTime())) /
                        3600000) *
                      HOUR_PX;
                    const cancelled = r.status === "cancelled";
                    return (
                      <div
                        key={r.id}
                        className={`absolute left-1 right-1 rounded-md px-1.5 py-1 overflow-hidden group ${
                          cancelled
                            ? "bg-[var(--color-bg-subtle)] border border-dashed border-[var(--color-border-subtle)] opacity-60"
                            : "bg-[var(--color-accent-soft,rgba(238,22,79,0.08))] border border-[var(--color-accent)]/30"
                        }`}
                        style={{ top, height: Math.max(h, 22) }}
                        title={`${r.title}\n${s.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}–${e.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}${r.room_name ? `\n${r.room_name}` : ""}${r.interviewer_names.length ? `\n${r.interviewer_names.join("、")}` : ""}`}
                      >
                        <div className="flex items-start justify-between gap-1">
                          <span
                            className={`text-[11px] font-medium font-body truncate ${
                              cancelled
                                ? "text-[var(--color-text-tertiary)] line-through"
                                : "text-[var(--color-text-primary)]"
                            }`}
                          >
                            {r.title}
                          </span>
                          {canWrite && !cancelled && (
                            <button
                              type="button"
                              onClick={() => cancel.mutate(r.id)}
                              className="opacity-0 group-hover:opacity-100 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] shrink-0"
                              aria-label="取消预约"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                        <div
                          className={`text-[10px] font-body truncate ${
                            cancelled
                              ? "text-[var(--color-text-tertiary)]"
                              : "text-[var(--color-text-secondary)]"
                          }`}
                        >
                          {s.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
                          –
                          {e.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
                          {r.room_name ? ` · ${r.room_name}` : ""}
                          {r.resource_name ? ` · ${r.resource_name}` : ""}
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {showCreate && (
        <ReservationModal
          rooms={(roomsQuery.data ?? []).filter((r) => r.is_active)}
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            qc.invalidateQueries({ queryKey: ["reservations"] });
          }}
        />
      )}
    </main>
  );
}

function ReservationModal({
  rooms,
  onClose,
  onCreated,
}: {
  rooms: Room[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const today = isoLocal(new Date()).slice(0, 10);
  const [title, setTitle] = useState("");
  const [kind, setKind] = useState<"room" | "resource">("room");
  const [targetId, setTargetId] = useState("");
  const [date, setDate] = useState(today);
  const [startT, setStartT] = useState("10:00");
  const [endT, setEndT] = useState("11:00");
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const resourcesQuery = useQuery({
    queryKey: ["resources"],
    queryFn: () => fetchJSON<Resource[]>("/api/resources"),
    enabled: kind === "resource",
  });
  const targets =
    kind === "room"
      ? rooms.map((r) => ({ id: r.id, label: `${r.name}${r.location ? `(${r.location})` : ""} · ${r.capacity}人` }))
      : (resourcesQuery.data ?? [])
          .filter((r) => r.is_active)
          .map((r) => ({ id: r.id, label: r.name }));

  const startISO = `${date}T${startT}:00`;
  const endISO = `${date}T${endT}:00`;

  const availQuery = useQuery({
    queryKey: ["availability", startISO, endISO],
    queryFn: () =>
      fetchJSON<Availability[]>(
        `/api/interviewers/availability?start=${encodeURIComponent(startISO)}&end=${encodeURIComponent(endISO)}`,
      ),
    enabled: date.length === 10 && startT < endT,
  });
  const members = availQuery.data ?? [];

  const create = useMutation({
    mutationFn: () =>
      fetchJSON<Reservation>("/api/reservations", {
        method: "POST",
        body: JSON.stringify({
          kind,
          room_id: kind === "room" ? targetId : null,
          resource_id: kind === "resource" ? targetId : null,
          title,
          starts_at: startISO,
          ends_at: endISO,
          interviewer_ids: [...picked],
        }),
      }),
    onSuccess: onCreated,
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "创建失败"),
  });

  return (
    <Modal open onClose={onClose} title="新建预约">
      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            标题
          </span>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="如:张三 × 后端工程师 一面"
            className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
        </label>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => {
              setKind("room");
              setTargetId("");
            }}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-[13px] font-body border transition-colors ${
              kind === "room"
                ? "border-[var(--color-accent)] bg-[var(--color-accent-soft,rgba(238,22,79,0.06))] text-[var(--color-accent)]"
                : "border-[var(--color-border-subtle)] text-[var(--color-text-secondary)]"
            }`}
          >
            <DoorOpen className="w-4 h-4" />
            会议室
          </button>
          <button
            type="button"
            onClick={() => {
              setKind("resource");
              setTargetId("");
            }}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-[13px] font-body border transition-colors ${
              kind === "resource"
                ? "border-[var(--color-accent)] bg-[var(--color-accent-soft,rgba(238,22,79,0.06))] text-[var(--color-accent)]"
                : "border-[var(--color-border-subtle)] text-[var(--color-text-secondary)]"
            }`}
          >
            <Package className="w-4 h-4" />
            资源
          </button>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            {kind === "room" ? "会议室" : "资源"}
          </span>
          <select
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
            className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body"
          >
            <option value="">请选择…</option>
            {targets.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>

        <div className="grid grid-cols-3 gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              日期
            </span>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              开始
            </span>
            <input
              type="time"
              value={startT}
              onChange={(e) => setStartT(e.target.value)}
              className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              结束
            </span>
            <input
              type="time"
              value={endT}
              onChange={(e) => setEndT(e.target.value)}
              className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body"
            />
          </label>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            参与面试官(选时段后自动检查冲突)
          </span>
          <div className="max-h-[160px] overflow-y-auto border border-[var(--color-border-subtle)] rounded-lg divide-y divide-[var(--color-border-row)]">
            {members.map((m) => (
              <label
                key={m.user_id}
                className="flex items-center gap-2.5 px-3 py-2 text-[13px] font-body cursor-pointer hover:bg-[var(--color-bg-subtle)]"
              >
                <input
                  type="checkbox"
                  checked={picked.has(m.user_id)}
                  onChange={(e) =>
                    setPicked((prev) => {
                      const next = new Set(prev);
                      if (e.target.checked) next.add(m.user_id);
                      else next.delete(m.user_id);
                      return next;
                    })
                  }
                  className="accent-[var(--color-accent)]"
                />
                <span className="flex-1 text-[var(--color-text-primary)]">
                  {m.name}
                </span>
                {m.busy_count > 0 && (
                  <span className="text-[11px] text-[var(--color-warning-text)]">
                    已有 {m.busy_count} 个安排
                  </span>
                )}
              </label>
            ))}
            {members.length === 0 && (
              <div className="px-3 py-4 text-center text-[12px] text-[var(--color-text-tertiary)]">
                选择有效时段后显示面试官
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button
            onClick={() => create.mutate()}
            disabled={
              create.isPending ||
              !title.trim() ||
              !targetId ||
              !(startT < endT)
            }
          >
            <CalendarDays className="w-4 h-4" />
            {create.isPending ? "创建中…" : "创建预约"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
