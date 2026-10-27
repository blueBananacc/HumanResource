import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  GitMerge,
  Lock,
  Mail,
  Phone,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

type PairSide = {
  id: string;
  name: string;
  display_email: string | null;
  display_phone: string | null;
  status: string;
  dedup_flag: string | null;
  created_at: string;
};

type DupPair = {
  candidate: PairSide;
  peer: PairSide;
  channels: string[];
};

type Queue = {
  scanned: number;
  exact: DupPair[];
  suspect: DupPair[];
};

const CHANNEL_LABEL: Record<string, string> = {
  name: "同名",
  email: "同邮箱",
  phone: "同手机",
};

function Person({ p }: { p: PairSide }) {
  return (
    <Link
      to={`/talents/${p.id}`}
      className="flex-1 min-w-0 rounded-lg border border-[var(--color-border-subtle)] px-3 py-2.5 hover:bg-[var(--color-bg-muted)] transition-colors"
    >
      <div className="flex items-center gap-2">
        <span className="w-7 h-7 rounded-full bg-[var(--color-bg-subtle)] flex items-center justify-center text-[11px] font-heading font-semibold shrink-0">
          {p.name.slice(0, 2)}
        </span>
        <div className="flex flex-col min-w-0">
          <span className="font-medium text-[13px] text-[var(--color-text-primary)] truncate">
            {p.name}
          </span>
          <span className="text-[11px] text-[var(--color-text-tertiary)] truncate">
            入库 {formatRelative(p.created_at)}
          </span>
        </div>
      </div>
      <div className="mt-1.5 flex flex-col gap-0.5 text-[11px] text-[var(--color-text-secondary)] font-body">
        {p.display_email && (
          <span className="flex items-center gap-1 truncate">
            <Mail className="w-3 h-3 shrink-0" />
            {p.display_email}
          </span>
        )}
        {p.display_phone && (
          <span className="flex items-center gap-1 truncate">
            <Phone className="w-3 h-3 shrink-0" />
            {p.display_phone}
          </span>
        )}
      </div>
    </Link>
  );
}

export function TalentDedupe() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [err, setErr] = useState("");
  const [scanMsg, setScanMsg] = useState("");

  const { data, isLoading } = useQuery<Queue>({
    queryKey: ["dedup-queue"],
    queryFn: () => fetchJSON("/api/talents/dedup-queue"),
  });

  const scan = useMutation({
    mutationFn: () =>
      fetchJSON<{ scanned: number; exact: number; suspect: number; flagged: number }>(
        "/api/talents/dedup-scan",
        { method: "POST" },
      ),
    onSuccess: (r) => {
      setScanMsg(
        `扫描 ${r.scanned} 人:精准 ${r.exact} 组 / 疑似 ${r.suspect} 组(新挂标 ${r.flagged} 人)`,
      );
      qc.invalidateQueries({ queryKey: ["dedup-queue"] });
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "扫描失败"),
  });

  const merge = useMutation({
    mutationFn: ({ keep, drop }: { keep: string; drop: string }) =>
      fetchJSON(`/api/talents/${keep}/merge`, {
        method: "POST",
        body: JSON.stringify({ drop_id: drop }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dedup-queue"] });
      qc.invalidateQueries({ queryKey: ["talents"] });
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "合并失败"),
  });

  const dismiss = useMutation({
    mutationFn: (candidateId: string) =>
      fetchJSON(`/api/talents/${candidateId}/dedup-dismiss`, {
        method: "POST",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dedup-queue"] });
      setErr("");
    },
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  async function onMerge(pair: DupPair) {
    const ok = await confirm({
      title: "合并这两位候选人?",
      description: `保留「${pair.candidate.name}」,将「${pair.peer.name}」的简历/面试/评分全部迁移后删除。此操作不可撤销。`,
      confirmLabel: "确认合并",
      tone: "danger",
    });
    if (!ok) return;
    merge.mutate({ keep: pair.candidate.id, drop: pair.peer.id });
  }

  async function onDismiss(pair: DupPair) {
    const ok = await confirm({
      title: "确认为非重复?",
      description: `「${pair.candidate.name}」与「${pair.peer.name}」将解除疑似标记。`,
      confirmLabel: "确认非重复",
    });
    if (!ok) return;
    dismiss.mutate(pair.candidate.id);
  }

  const exact = data?.exact ?? [];
  const suspect = data?.suspect ?? [];

  function PairRow({ pair, tier }: { pair: DupPair; tier: string }) {
    return (
      <div className="flex items-center gap-4 px-6 py-4 border-b border-[var(--color-border-row)] last:border-0">
        <div className="flex-1 grid grid-cols-2 gap-3 min-w-0">
          <Person p={pair.candidate} />
          <Person p={pair.peer} />
        </div>
        <div className="flex flex-col gap-1 shrink-0 w-[84px]">
          {(pair.channels ?? []).map((ch) => (
            <span
              key={ch}
              className="px-1.5 py-0.5 rounded text-[11px] font-body bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)] text-center"
            >
              {CHANNEL_LABEL[ch] ?? ch}
            </span>
          ))}
        </div>
        <div className="flex flex-col gap-2 shrink-0 w-[120px]">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => onMerge(pair)}
            disabled={merge.isPending}
          >
            <GitMerge className="w-3.5 h-3.5" />
            合并
          </Button>
          {tier === "suspect" && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => onDismiss(pair)}
              disabled={dismiss.isPending}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              非重复
            </Button>
          )}
        </div>
      </div>
    );
  }

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1100px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            简历查重
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            两级查重:邮箱+手机号一致为精准重复(可直接合并);单项一致为疑似(人工确认)。
            新简历入库时自动检测,也可手动全量重扫。
          </p>
        </div>
        <Button
          variant="secondary"
          onClick={() => scan.mutate()}
          disabled={scan.isPending}
        >
          <RefreshCw
            className={`w-4 h-4 ${scan.isPending ? "animate-spin" : ""}`}
          />
          {scan.isPending ? "扫描中…" : "全量重扫"}
        </Button>
      </div>

      {(err || scanMsg) && (
        <div
          className={`text-[13px] font-body ${
            err ? "text-[var(--color-danger)]" : "text-[var(--color-success)]"
          }`}
        >
          {err || scanMsg}
        </div>
      )}

      {isLoading ? (
        <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      ) : exact.length === 0 && suspect.length === 0 ? (
        <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)]">
          <Empty
            icon={ScanSearch}
            title="没有待处理的重复候选人"
            description="新简历入库时会自动查重;也可以点击「全量重扫」对现有人才库做一次完整检查。"
          />
        </div>
      ) : (
        <>
          {exact.length > 0 && (
            <section className="flex flex-col gap-3">
              <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)] flex items-center gap-2">
                精准重复
                <span className="px-1.5 py-0.5 rounded text-[11px] font-body bg-[var(--color-danger-soft)] text-[var(--color-danger)]">
                  {exact.length} 组
                </span>
                <span className="text-[12px] font-normal text-[var(--color-text-tertiary)] font-body">
                  邮箱与手机号均一致
                </span>
              </h3>
              <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
                {exact.map((p) => (
                  <PairRow key={p.candidate.id} pair={p} tier="exact" />
                ))}
              </div>
            </section>
          )}

          {suspect.length > 0 && (
            <section className="flex flex-col gap-3">
              <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)] flex items-center gap-2">
                疑似重复
                <span className="px-1.5 py-0.5 rounded text-[11px] font-body bg-[var(--color-warning-soft,var(--color-bg-subtle))] text-[var(--color-warning-text,var(--color-text-secondary))]">
                  {suspect.length} 组
                </span>
                <span className="text-[12px] font-normal text-[var(--color-text-tertiary)] font-body">
                  邮箱或手机号单项一致,需人工确认
                </span>
              </h3>
              <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
                {suspect.map((p) => (
                  <PairRow key={p.candidate.id} pair={p} tier="suspect" />
                ))}
              </div>
            </section>
          )}
        </>
      )}

      <div className="flex items-center gap-2 text-[11px] text-[var(--color-text-tertiary)] font-body">
        <Lock className="w-3 h-3" />
        被锁定的候选人仍会出现在队列中,但合并操作需先解除锁定。
      </div>
    </main>
  );
}
