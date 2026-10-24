import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Archive, Megaphone, Pencil, Plus, Send, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { PERM } from "@/lib/roles";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";
import { formatRelative } from "@/lib/format";
import { Field, PageError, useOrgOptions as useOrgOptionsBase } from "./Employees";

type Announcement = {
  id: string;
  title: string;
  content: string;
  audience_org_unit_id: string | null;
  audience_name: string | null;
  status: string;
  published_at: string | null;
  expires_at: string | null;
  read_count: number;
  created_at: string;
};

const STATUS_LABEL: Record<string, { label: string; cls: string }> = {
  draft: { label: "草稿", cls: "bg-[var(--color-surface-2)] text-[var(--color-text-secondary)]" },
  published: { label: "已发布", cls: "bg-green-50 text-green-700" },
  archived: { label: "已归档", cls: "bg-amber-50 text-amber-700" },
};

function useOrgOptions() {
  return [{ value: "", label: "全公司" }].concat(useOrgOptionsBase());
}

export function Announcements() {
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Announcement | null>(null);
  const { hasPermission } = useAuth();
  const canWrite = hasPermission(PERM.MANAGE_EMPLOYEES);
  const qc = useQueryClient();
  const { data, isLoading, isError, error, page, pageCount, total, goto } =
    usePagedQuery<Announcement>({ key: ["announcements"], url: "/api/announcements" });
  const items = data?.items ?? [];

  const setStatus = useMutation({
    mutationFn: ({ id, to }: { id: string; to: string }) =>
      fetchJSON(`/api/announcements/${id}/status?to=${to}`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["announcements"] }),
  });
  const del = useMutation({
    mutationFn: (id: string) => fetchJSON(`/api/announcements/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["announcements"] }),
  });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">公告管理</h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            公司公告 — 发布后可按组织定向,员工在自助端可见并已读打点。
          </p>
        </div>
        {!isError && canWrite && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4" />发公告
          </Button>
        )}
      </div>

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">加载中…</div>
        ) : isError ? (
          <PageError error={error} />
        ) : items.length === 0 ? (
          <Empty
            icon={Megaphone}
            title="还没有公告"
            description="发一条全员通知、放假安排或制度更新。"
            action={canWrite ? <Button onClick={() => setCreateOpen(true)}>发公告</Button> : undefined}
          />
        ) : (
          items.map((a, i) => {
            const st = STATUS_LABEL[a.status] ?? STATUS_LABEL.draft;
            return (
              <div key={a.id} className={`px-6 py-4 ${i < items.length - 1 ? "border-b border-[var(--color-border-row)]" : ""}`}>
                <div className="flex items-center gap-3">
                  <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">{a.title}</span>
                  <span className={`text-[11px] px-1.5 py-0.5 rounded ${st.cls}`}>{st.label}</span>
                  <span className="text-[12px] text-[var(--color-text-secondary)] font-body">
                    {a.audience_name ?? "全公司"}
                  </span>
                  <div className="flex-1" />
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                    已读 {a.read_count}
                  </span>
                  <span className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                    {a.published_at ? `发布于 ${formatRelative(a.published_at)}` : `创建于 ${formatRelative(a.created_at)}`}
                  </span>
                </div>
                <p className="mt-1.5 text-[13px] text-[var(--color-text-secondary)] font-body line-clamp-1">{a.content}</p>
                <div className="mt-3 flex items-center gap-2">
                  {canWrite && a.status === "draft" && (
                    <Button size="sm" onClick={() => setStatus.mutate({ id: a.id, to: "published" })} disabled={setStatus.isPending}>
                      <Send className="w-3.5 h-3.5" />发布
                    </Button>
                  )}
                  {canWrite && a.status === "published" && (
                    <Button variant="secondary" size="sm" onClick={() => setStatus.mutate({ id: a.id, to: "archived" })} disabled={setStatus.isPending}>
                      <Archive className="w-3.5 h-3.5" />归档
                    </Button>
                  )}
                  {canWrite && a.status !== "archived" && (
                    <Button variant="ghost" size="sm" onClick={() => setEditing(a)}>
                      <Pencil className="w-3.5 h-3.5" />编辑
                    </Button>
                  )}
                  {canWrite && (
                    <Button variant="ghost" size="sm" onClick={() => del.mutate(a.id)} disabled={del.isPending}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
      <Pagination page={page} pageCount={pageCount} total={total} pageSize={PAGE_SIZE} onChange={goto} />
      {/* 条件挂载:每次打开都重挂组件,受控状态(audience/publishNow/err)才不会
          沿用上次编辑残留 —— 否则受众会被静默重置成全公司 */}
      {(createOpen || editing) && (
        <AnnouncementModal editing={editing} onClose={() => { setCreateOpen(false); setEditing(null); }} />
      )}
    </main>
  );
}

function AnnouncementModal({
  editing,
  onClose,
}: {
  editing: Announcement | null;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const orgOptions = useOrgOptions();
  const [audience, setAudience] = useState(editing?.audience_org_unit_id ?? "");
  const [publishNow, setPublishNow] = useState(false);
  const [err, setErr] = useState("");

  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      editing
        ? fetchJSON(`/api/announcements/${editing.id}`, { method: "PUT", body: JSON.stringify(body) })
        : fetchJSON(`/api/announcements${body.publish ? "?publish=true" : ""}`, { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["announcements"] });
      qc.invalidateQueries({ queryKey: ["me-announcements"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      title: String(fd.get("title") || "").trim(),
      content: String(fd.get("content") || "").trim(),
      audience_org_unit_id: audience || null,
      expires_at: String(fd.get("expires_at") || "") || null,
      publish: publishNow,
    });
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={editing ? "编辑公告" : "发公告"}
      description={editing ? "更新公告内容;已发布的公告编辑后继续可见。" : "可立即发布或先存草稿;受众为空 = 全公司。"}
      width={560}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>取消</Button>
          <Button type="submit" form="ann-form" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="ann-form" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="标题" required>
          <Input name="title" defaultValue={editing?.title} maxLength={128} required />
        </Field>
        <Field label="内容" required>
          <textarea
            name="content"
            defaultValue={editing?.content}
            required
            rows={6}
            className="w-full rounded-lg border border-[var(--color-border-default)] px-3 py-2 text-sm font-body outline-none focus:border-[var(--color-accent)]"
          />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="受众">
            <Select value={audience} onChange={setAudience} options={orgOptions} />
          </Field>
          <Field label="过期时间">
            <Input name="expires_at" type="datetime-local" defaultValue={editing?.expires_at?.slice(0, 16) ?? ""} />
          </Field>
        </div>
        {!editing && (
          <label className="flex items-center gap-2 text-[13px] text-[var(--color-text-secondary)] font-body">
            <input type="checkbox" checked={publishNow} onChange={(e) => setPublishNow(e.target.checked)} />
            保存后立即发布
          </label>
        )}
        {err && <div className="text-[13px] text-[var(--color-danger)] font-body">{err}</div>}
      </form>
    </Modal>
  );
}
