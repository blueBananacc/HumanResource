import {
  forwardRef,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { CloudUpload, Loader2, X } from "lucide-react";
import { ApiError, fetchJSON } from "@/lib/api";
import {
  ParseStatusChip,
  POLL_MS,
  isInflight,
  type Resume,
} from "@/components/resume/shared";

export type ResumeUploadHandle = { open: () => void };

export const ResumeUpload = forwardRef<ResumeUploadHandle>(
  function ResumeUpload(_props, ref) {
    const qc = useQueryClient();
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [uploaded, setUploaded] = useState<Resume[]>([]);

    useImperativeHandle(ref, () => ({
      open: () => inputRef.current?.click(),
    }));

    const upload = useMutation({
      mutationFn: async (file: File) => {
        const fd = new FormData();
        fd.append("file", file);
        return fetchJSON<Resume>("/api/resumes/upload", {
          method: "POST",
          body: fd,
        });
      },
      onSuccess: (r) => {
        setUploaded((prev) => [r, ...prev]);
        qc.invalidateQueries({ queryKey: ["talents"] });
      },
      onError: (e: unknown) =>
        setError(e instanceof ApiError ? e.message : "上传失败"),
    });

    function onFiles(files: FileList | null) {
      if (!files) return;
      setError(null);
      Array.from(files).forEach((f) => upload.mutate(f));
    }

    return (
      <div className="flex flex-col gap-4">
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          className="hidden"
          onChange={(e) => {
            onFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            onFiles(e.dataTransfer.files);
          }}
          className={`flex flex-col items-center justify-center gap-2.5 py-8 px-6 rounded-xl bg-white border border-dashed cursor-pointer transition-colors ${
            dragOver
              ? "border-[var(--color-text-primary)] bg-[var(--color-bg-subtle)]"
              : "border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-muted)]"
          }`}
          onClick={(e) => {
            e.preventDefault();
            inputRef.current?.click();
          }}
        >
          <CloudUpload className="w-7 h-7 text-[var(--color-text-tertiary)]" />
          <span className="text-sm font-medium text-[var(--color-text-secondary)] font-body">
            拖拽简历文件到此处, 或点击选择
          </span>
          <span className="text-xs text-[var(--color-text-tertiary)] font-body">
            支持 PDF / DOCX / TXT, 单文件最大 20 MB(不支持旧版 .doc 格式)
          </span>
        </label>

        {error && (
          <div className="text-[13px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-3 py-2 rounded-md font-body">
            {error}
          </div>
        )}

        {upload.isPending && (
          <div className="flex items-center gap-2 text-[13px] text-[var(--color-text-secondary)] font-body">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            上传中…
          </div>
        )}

        {uploaded.length > 0 && (
          <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
            <div className="flex items-center justify-between px-6 py-3 border-b border-[var(--color-border-subtle)]">
              <h3 className="font-heading font-semibold text-sm text-[var(--color-text-primary)]">
                本次上传
              </h3>
              <button
                type="button"
                onClick={() => setUploaded([])}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] p-1"
                aria-label="清除本次上传"
                title="清除"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {uploaded.map((r, i) => (
              <UploadedRow
                key={r.id}
                initial={r}
                isLast={i === uploaded.length - 1}
              />
            ))}
          </section>
        )}
      </div>
    );
  },
);

function UploadedRow({
  initial,
  isLast,
}: {
  initial: Resume;
  isLast: boolean;
}) {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["resume-upload-row", initial.id],
    queryFn: async () => {
      const r = await fetchJSON<Resume>(`/api/resumes/${initial.id}`);
      if (!isInflight(r.parse_status)) {
        qc.invalidateQueries({ queryKey: ["talents"] });
      }
      return r;
    },
    initialData: initial,
    initialDataUpdatedAt: 0,
    refetchInterval: (q) => {
      const r = q.state.data as Resume | undefined;
      return r && isInflight(r.parse_status) ? POLL_MS : false;
    },
  });
  const r = data ?? initial;
  const failed = r.parse_status === "failed";
  return (
    <div
      className={`flex items-center gap-3 px-6 py-3 ${
        isLast ? "" : "border-b border-[var(--color-border-row)]"
      }`}
    >
      <span className="font-medium text-sm text-[var(--color-text-primary)] font-body truncate">
        {r.candidate.name}
      </span>
      <ParseStatusChip status={r.parse_status} />
      <span className="text-xs text-[var(--color-text-tertiary)] font-body truncate flex-1 min-w-0">
        {r.file_name}
      </span>
      <Link
        to={`/talents/${r.candidate.id}/resumes/${r.id}`}
        className="text-[13px] text-[var(--color-text-primary)] hover:underline font-body shrink-0"
      >
        {failed ? "去补全 →" : "查看简历 →"}
      </Link>
    </div>
  );
}
