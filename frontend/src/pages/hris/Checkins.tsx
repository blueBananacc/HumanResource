import { useSearchParams } from "react-router-dom";
import { Clock } from "lucide-react";
import { Empty } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { StatusPill, type Checkin } from "@/lib/hris";

export function Checkins() {
  const [sp, setSp] = useSearchParams();
  const day = sp.get("day") ?? "";
  const { data, isLoading, page, pageCount, total, goto } =
    usePagedQuery<Checkin>({
      key: ["checkins"],
      url: "/api/attendance/checkins",
      params: { day },
    });
  const items = data?.items ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          打卡流水
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          员工自助端每一次上/下班打卡的原始记录。
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Input
          type="date"
          value={day}
          onChange={(e) =>
            setSp((prev) => {
              const next = new URLSearchParams(prev);
              if (e.target.value) next.set("day", e.target.value);
              else next.delete("day");
              next.delete("page");
              return next;
            })
          }
          className="w-[160px]"
        />
        <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
          共{" "}
          <span className="font-mono font-semibold text-[var(--color-text-primary)]">
            {total}
          </span>{" "}
          条
        </span>
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty icon={Clock} title="没有打卡记录" />
        ) : (
          items.map((c, i) => (
            <div
              key={c.id}
              className={`flex items-center gap-4 px-6 py-3 text-[13px] font-body ${
                i < items.length - 1
                  ? "border-b border-[var(--color-border-row)]"
                  : ""
              }`}
            >
              <StatusPill status={c.kind} />
              <span className="text-[var(--color-text-primary)] font-medium">
                {c.employee_name}
              </span>
              <span className="font-mono text-[var(--color-text-secondary)]">
                {c.ts.slice(0, 16).replace("T", " ")}
              </span>
              <span className="text-[var(--color-text-tertiary)]">
                {c.method}
              </span>
              {c.note && (
                <span className="text-[var(--color-text-tertiary)] truncate">
                  {c.note}
                </span>
              )}
            </div>
          ))
        )}
      </div>

      <Pagination
        page={page}
        pageCount={pageCount}
        total={total}
        pageSize={PAGE_SIZE}
        onChange={goto}
      />
    </main>
  );
}
