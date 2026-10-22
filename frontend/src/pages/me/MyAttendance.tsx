import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Clock, LogIn, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Empty } from "@/components/ui/empty";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  StatusPill,
  fmtTime,
  type MeAttendance,
  type MeCheckin,
} from "@/lib/hris";

export function MyAttendance() {
  const [month, setMonth] = useState(new Date().toISOString().slice(0, 7));
  const [err, setErr] = useState("");
  const qc = useQueryClient();

  const { data: attendance, isLoading } = useQuery<MeAttendance[]>({
    queryKey: ["me-attendance", month],
    queryFn: () => fetchJSON(`/api/me/attendance?month=${month}`),
  });
  const { data: checkins } = useQuery<MeCheckin[]>({
    queryKey: ["me-checkins"],
    queryFn: () => fetchJSON("/api/me/checkins"),
  });

  const todayStr = new Date().toISOString().slice(0, 10);
  const todayCheckins = (checkins ?? []).filter((c) =>
    c.ts.startsWith(todayStr),
  );
  const lastKind = todayCheckins[todayCheckins.length - 1]?.kind;
  const nextKind = lastKind === "in" ? "out" : "in";

  const punch = useMutation({
    mutationFn: (kind: "in" | "out") =>
      fetchJSON("/api/me/checkins", {
        method: "POST",
        body: JSON.stringify({ kind }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-checkins"] });
      qc.invalidateQueries({ queryKey: ["me-attendance"] });
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "打卡失败"),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[900px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          我的考勤
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          上下班打卡 + 按天汇总;异常请联系 HR 修正。
        </p>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex items-center gap-6">
        <div className="flex flex-col gap-1">
          <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
            今日 {todayStr}
          </div>
          <div className="font-mono font-semibold text-2xl text-[var(--color-text-primary)]">
            {new Date().toLocaleTimeString("zh-CN", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
          <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
            今日已打卡 {todayCheckins.length} 次
            {todayCheckins.length > 0 &&
              ` · 最近 ${fmtTime(todayCheckins[todayCheckins.length - 1].ts)}(${
                todayCheckins[todayCheckins.length - 1].kind === "in"
                  ? "上班"
                  : "下班"
              })`}
          </div>
        </div>
        <div className="flex-1" />
        {nextKind === "in" ? (
          <Button
            size="lg"
            onClick={() => punch.mutate("in")}
            disabled={punch.isPending}
          >
            <LogIn className="w-4 h-4" />
            上班打卡
          </Button>
        ) : (
          <Button
            size="lg"
            onClick={() => punch.mutate("out")}
            disabled={punch.isPending}
          >
            <LogOut className="w-4 h-4" />
            下班打卡
          </Button>
        )}
        {err && (
          <span className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Input
          type="month"
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="w-[160px]"
        />
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : (attendance ?? []).length === 0 ? (
          <Empty
            icon={Clock}
            title="本月还没有考勤记录"
            description="打卡后第二天会自动生成考勤记录。"
          />
        ) : (
          <table className="w-full text-[13px] font-body">
            <thead>
              <tr className="text-left text-[var(--color-text-tertiary)] border-b border-[var(--color-border-row)]">
                <th className="px-6 py-3 font-medium">日期</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">上班</th>
                <th className="px-4 py-3 font-medium">下班</th>
                <th className="px-6 py-3 font-medium text-right">工时</th>
              </tr>
            </thead>
            <tbody>
              {(attendance ?? []).map((a) => (
                <tr
                  key={a.day}
                  className="border-b border-[var(--color-border-row)] last:border-0"
                >
                  <td className="px-6 py-3 font-mono">{a.day}</td>
                  <td className="px-4 py-3">
                    <StatusPill status={a.status} />
                  </td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.first_in)}</td>
                  <td className="px-4 py-3 font-mono">{fmtTime(a.last_out)}</td>
                  <td className="px-6 py-3 text-right font-mono">
                    {a.work_hours != null ? `${a.work_hours}h` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </main>
  );
}
