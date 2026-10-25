import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, ChevronDown, ChevronRight, FileText, RefreshCw, Trash2, Upload } from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { useConfirm } from "@/components/ui/confirm";
import { formatRelative } from "@/lib/format";

type Source = {
  id: string;
  name: string;
  kind: "text" | "file";
  status: "indexing" | "ready" | "failed";
  chunk_count: number;
  extraction_status: string;
  extraction_method: string | null;
  error_message: string | null;
  file_name: string | null;
  created_at: string;
  updated_at: string;
};

type Chunk = {
  id: string;
  source_id: string;
  source_name: string;
  chunk_index: number;
  content: string;
  has_embedding: boolean;
};

const STATUS_LABEL: Record<Source["status"], string> = {
  indexing: "索引中",
  ready: "已就绪",
  failed: "失败",
};

const STATUS_CLS: Record<Source["status"], string> = {
  indexing: "bg-amber-50 text-amber-700",
  ready: "bg-emerald-50 text-emerald-700",
  failed: "bg-red-50 text-red-700",
};

/** 岗位知识源(T11):资料切块后注入匹配评估/出题/面试 prompt,让 AI 输出有据可依。 */
export function JobKnowledgeCard({ jobId, canWrite }: { jobId: string; canWrite: boolean }) {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const fileRef = useRef<HTMLInputElement>(null);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [chunksOpen, setChunksOpen] = useState<string | null>(null);
  const [error, setError] = useState("");

  const { data: sources, isLoading } = useQuery({
    queryKey: ["job-knowledge", jobId],
    queryFn: () => fetchJSON<Source[]>(`/api/jobs/${jobId}/knowledge-sources/`),
  });
  const { data: chunks } = useQuery({
    queryKey: ["job-knowledge-chunks", jobId],
    queryFn: () => fetchJSON<Chunk[]>(`/api/jobs/${jobId}/knowledge-sources/chunks`),
    enabled: expanded,
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["job-knowledge", jobId] });
    qc.invalidateQueries({ queryKey: ["job-knowledge-chunks", jobId] });
  };

  const addText = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/jobs/${jobId}/knowledge-sources/text`, {
        method: "POST",
        body: JSON.stringify({ name, content }),
      }),
    onSuccess: () => {
      invalidate();
      setShowForm(false);
      setName("");
      setContent("");
      setError("");
    },
    onError: (e) => setError(e instanceof Error ? e.message : "添加失败"),
  });

  const upload = useMutation({
    mutationFn: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return fetchJSON(
        `/api/jobs/${jobId}/knowledge-sources/file?name=${encodeURIComponent(file.name)}`,
        { method: "POST", body: fd },
      );
    },
    onSuccess: invalidate,
    onError: (e) => setError(e instanceof Error ? e.message : "上传失败"),
  });

  const reindex = useMutation({
    mutationFn: (sid: string) =>
      fetchJSON(`/api/jobs/${jobId}/knowledge-sources/${sid}/reindex`, { method: "POST" }),
    onSuccess: invalidate,
  });

  const remove = useMutation({
    mutationFn: (sid: string) =>
      fetchJSON(`/api/jobs/${jobId}/knowledge-sources/${sid}`, { method: "DELETE" }),
    onSuccess: invalidate,
  });

  const rows = sources ?? [];
  const readyCount = rows.filter((s) => s.status === "ready").length;

  return (
    <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <BookOpen className="w-4 h-4 text-[var(--color-text-tertiary)]" />
        <h3 className="font-heading font-semibold text-base text-[var(--color-text-primary)]">
          岗位知识
        </h3>
        <span className="text-xs text-[var(--color-text-tertiary)]">
          供 AI 匹配评估 / 出题 / 面试引用
        </span>
        <div className="ml-auto flex items-center gap-2">
          {canWrite && (
            <>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.docx,.doc,.txt,.md"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) upload.mutate(f);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                className="text-xs text-[var(--color-accent)] hover:underline"
                onClick={() => fileRef.current?.click()}
                disabled={upload.isPending}
              >
                <Upload className="mr-1 inline-block h-3 w-3" />
                上传文件
              </button>
              <button
                type="button"
                className="text-xs text-[var(--color-accent)] hover:underline"
                onClick={() => setShowForm((v) => !v)}
              >
                + 添加文本
              </button>
            </>
          )}
          {readyCount > 0 && (
            <button
              type="button"
              className="text-xs text-[var(--color-text-muted)] hover:underline"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "收起片段" : `查看片段(${readyCount})`}
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>
      )}

      {showForm && (
        <div className="flex flex-col gap-2 rounded-lg border border-[var(--color-border-subtle)] p-3">
          <input
            className="rounded-lg border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm"
            placeholder="资料名称,如:团队介绍 / 技术栈说明"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={128}
          />
          <textarea
            className="min-h-28 rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-sm"
            placeholder="粘贴资料文本(JD 补充说明、公司/团队介绍、项目背景、技术栈细节…)"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              type="button"
              className="rounded-lg bg-[var(--color-accent)] px-3 py-1.5 text-xs text-white disabled:opacity-50"
              disabled={!name.trim() || !content.trim() || addText.isPending}
              onClick={() => addText.mutate()}
            >
              {addText.isPending ? "索引中…" : "保存并索引"}
            </button>
            <button
              type="button"
              className="rounded-lg px-3 py-1.5 text-xs text-[var(--color-text-muted)]"
              onClick={() => setShowForm(false)}
            >
              取消
            </button>
          </div>
        </div>
      )}

      {isLoading && <p className="text-[13px] text-[var(--color-text-tertiary)]">加载中…</p>}
      {!isLoading && !rows.length && (
        <p className="text-[13px] text-[var(--color-text-tertiary)]">
          暂无知识源。添加 JD 补充说明 / 公司介绍 / 技术栈文档后,AI 评估与出题会引用这些资料。
        </p>
      )}

      {rows.map((s) => (
        <div
          key={s.id}
          className="flex items-center gap-3 rounded-lg border border-[var(--color-border-subtle)] px-3 py-2"
        >
          <FileText className="h-4 w-4 shrink-0 text-[var(--color-text-tertiary)]" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="truncate text-[13px] font-medium text-[var(--color-text-primary)]">
                {s.name}
              </span>
              <span className={`rounded-full px-2 py-0.5 text-[11px] ${STATUS_CLS[s.status]}`}>
                {STATUS_LABEL[s.status]}
              </span>
              {s.status === "ready" && (
                <span className="text-[11px] text-[var(--color-text-tertiary)]">
                  {s.chunk_count} 块{s.extraction_method ? ` · ${s.extraction_method}` : ""}
                </span>
              )}
            </div>
            {s.error_message && (
              <p className="mt-0.5 truncate text-[11px] text-red-600">{s.error_message}</p>
            )}
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              {s.kind === "file" ? s.file_name : "文本资料"} · {formatRelative(s.updated_at)}
            </p>
          </div>
          {canWrite && (
            <div className="flex shrink-0 items-center gap-1">
              {s.status !== "indexing" && (
                <button
                  type="button"
                  title="重新索引"
                  className="rounded p-1 text-[var(--color-text-muted)] hover:bg-[var(--color-bg-muted)]"
                  disabled={reindex.isPending}
                  onClick={() => reindex.mutate(s.id)}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              )}
              <button
                type="button"
                title="删除"
                className="rounded p-1 text-[var(--color-text-muted)] hover:bg-red-50 hover:text-red-600"
                onClick={async () => {
                  if (
                    await confirm({
                      title: `删除知识源「${s.name}」?`,
                      description: "其全部片段将被移除,不再注入 AI 上下文。",
                      tone: "danger",
                    })
                  ) {
                    remove.mutate(s.id);
                  }
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
        </div>
      ))}

      {expanded && (chunks ?? []).length > 0 && (
        <div className="mt-1 rounded-lg border border-[var(--color-border-subtle)]">
          {Object.entries(
            (chunks ?? []).reduce<Record<string, Chunk[]>>((acc, c) => {
              (acc[c.source_name] ??= []).push(c);
              return acc;
            }, {}),
          ).map(([srcName, list]) => (
            <div key={srcName} className="border-b border-[var(--color-border-subtle)] last:border-0">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-[var(--color-text-primary)] hover:bg-[var(--color-bg-muted)]"
                onClick={() => setChunksOpen(chunksOpen === srcName ? null : srcName)}
              >
                {chunksOpen === srcName ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                {srcName}
                <span className="text-[var(--color-text-tertiary)]">{list.length} 块</span>
              </button>
              {chunksOpen === srcName &&
                list.map((c) => (
                  <div key={c.id} className="border-t border-[var(--color-border-subtle)] px-4 py-2">
                    <p className="mb-0.5 text-[11px] text-[var(--color-text-tertiary)]">
                      #{c.chunk_index}{c.has_embedding ? " · embedding" : ""}
                    </p>
                    <p className="whitespace-pre-wrap text-xs text-[var(--color-text-secondary)]">
                      {c.content}
                    </p>
                  </div>
                ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
