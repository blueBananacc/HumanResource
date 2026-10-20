import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON, getToken } from "@/lib/api";

export type Candidate = {
  id: string;
  name: string;
  display_email: string | null;
  display_phone: string | null;
};

export type Resume = {
  id: string;
  file_name: string;
  file_size: number;
  file_mime: string;
  source: string;
  created_at: string;
  candidate: Candidate;
  skills: string[];
  // upload 后立即返回 pending,worker 解析完转 done
  parse_status: "pending" | "parsing" | "done" | "failed";
  parse_error: string | null;
};

export type ResumeDetail = Resume & {
  parsed_text: string | null;
  email: string | null;
  phone: string | null;
};

// 列表 / 详情中只要存在 pending|parsing 的简历就开 2s 轮询;done|failed 都是终态。
export const POLL_MS = 2000;
export const isInflight = (s: Resume["parse_status"]) =>
  s === "pending" || s === "parsing";

/**
 * 触发简历原文件下载。
 *
 * 后端返回 ``Content-Disposition: attachment; filename*=UTF-8''...``,
 * 前端用 fetch + blob + a[download] 是因为我们的 token 在 localStorage
 * (Bearer header),不能像 cookie 那样靠浏览器自动带上 — 直接 a.href
 * 链到 /api/... 会丢 Authorization 头被 401 弹回登录。
 */
export async function downloadResumeFile(id: string, filename: string): Promise<void> {
  const token = getToken();
  const res = await fetch(`/api/resumes/${id}/download`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (res.status === 401) {
    if (location.pathname !== "/login") location.href = "/login";
    throw new ApiError(401, "未授权");
  }
  if (!res.ok) {
    let detail: unknown = undefined;
    try {
      detail = await res.json();
    } catch {
      // 二进制响应失败时通常没有 JSON body
    }
    const msg =
      typeof detail === "object" && detail && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `HTTP ${res.status}`;
    throw new ApiError(res.status, msg, detail);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "resume";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** 列表行内的简短状态指示。done 时不显示(干净)。 */
export function ParseStatusChip({
  status,
}: {
  status: Resume["parse_status"];
}) {
  if (status === "done") return null;
  if (status === "failed") {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium font-body bg-[var(--color-danger-soft)] text-[var(--color-danger)]">
        解析失败
      </span>
    );
  }
  // pending / parsing
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium font-body bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]">
      <Loader2 className="w-3 h-3 animate-spin" />
      解析中
    </span>
  );
}

/** 详情抽屉内的状态横幅,带 parse_error。 */
export function ParseStatusBanner({
  status,
  error,
}: {
  status: Resume["parse_status"];
  error: string | null;
}) {
  if (status === "failed") {
    return (
      <div className="text-[12px] bg-[var(--color-danger-soft)] text-[var(--color-danger)] p-3 rounded-lg font-body">
        <div className="font-medium mb-1">简历解析失败</div>
        <div className="text-[11px] opacity-80">
          {error || "未知错误。建议重新上传。"}
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-[12px] text-[var(--color-text-secondary)] bg-[var(--color-bg-subtle)] p-3 rounded-lg font-body">
      <Loader2 className="w-3.5 h-3.5 animate-spin" />
      {status === "pending"
        ? "已入队,等待 worker 处理"
        : "worker 正在解析,文本与技能稍后到位"}
    </div>
  );
}

const LAST_RESUME_WARNING =
  "这是该候选人的最后一份简历,删除后候选人档案(含面试记录与报告、备注、标签、分组关系)将一并永久删除,操作不可恢复。";

export function useDeleteResume() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const confirm = useConfirm();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function deleteResume(opts: {
    resumeId: string;
    fileName: string;
    candidateId: string;
    isLast: boolean;
  }): Promise<boolean> {
    const { resumeId, fileName, candidateId, isLast } = opts;
    const ok = await confirm({
      title: "删除简历?",
      description: isLast
        ? LAST_RESUME_WARNING
        : `将永久删除「${fileName}」,简历对应的解析文本、技能、面试题集会一并清除,操作不可恢复。`,
      tone: "danger",
      confirmLabel: "删除",
    });
    if (!ok) return false;
    setDeletingId(resumeId);
    try {
      await fetchJSON(`/api/resumes/${resumeId}`, { method: "DELETE" });
    } finally {
      setDeletingId(null);
    }
    qc.invalidateQueries({ queryKey: ["talents"] });
    if (isLast) {
      qc.removeQueries({ queryKey: ["talent", candidateId] });
      navigate("/talents");
    } else {
      qc.invalidateQueries({ queryKey: ["talent", candidateId] });
      qc.invalidateQueries({ queryKey: ["talent-timeline", candidateId] });
    }
    return true;
  }

  return { deleteResume, deletingId };
}

/** 删除候选人整档 —— 人才库列表行内操作。后端会级联清掉简历 / 面试 / 备注 / 分组关系。 */
export function useDeleteCandidate() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function deleteCandidate(opts: {
    candidateId: string;
    name: string;
  }): Promise<boolean> {
    const { candidateId, name } = opts;
    const ok = await confirm({
      title: "删除候选人?",
      description: `将永久删除候选人「${name}」及其全部简历、面试记录与报告、备注、标签与分组关系,操作不可恢复。`,
      tone: "danger",
      confirmLabel: "删除",
    });
    if (!ok) return false;
    setDeletingId(candidateId);
    try {
      await fetchJSON(`/api/talents/${candidateId}`, { method: "DELETE" });
    } finally {
      setDeletingId(null);
    }
    qc.invalidateQueries({ queryKey: ["talents"] });
    qc.removeQueries({ queryKey: ["talent", candidateId] });
    qc.removeQueries({ queryKey: ["talent-timeline", candidateId] });
    return true;
  }

  return { deleteCandidate, deletingId };
}