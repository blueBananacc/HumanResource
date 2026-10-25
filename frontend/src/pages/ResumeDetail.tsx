import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  Download,
  FileText,
  ListChecks,
  Loader2,
  Mail,
  Phone,
  Sparkles,
  Tag,
  Trash2,
  User,
  Users,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty } from "@/components/ui/empty";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatBytes, formatRelative } from "@/lib/format";
import { EditableContact } from "@/components/resume/EditableContact";
import {
  MatchesForResume,
  QuestionSetsForResume,
} from "@/components/resume/ResumeSections";
import {
  POLL_MS,
  ParseStatusBanner,
  downloadResumeFile,
  isInflight,
  useDeleteResume,
  type ResumeDetail as ResumeDetailData,
  type ResumeSections,
} from "@/components/resume/shared";

type TalentLite = {
  id: string;
  name: string;
  resumes: { id: string }[];
};

export function ResumeDetail() {
  const { id: cid = "", resumeId = "" } = useParams<{
    id: string;
    resumeId: string;
  }>();
  const qc = useQueryClient();
  const { deleteResume, deletingId } = useDeleteResume();

  const { data, isLoading, error } = useQuery({
    queryKey: ["resume-detail", resumeId],
    queryFn: () => fetchJSON<ResumeDetailData>(`/api/resumes/${resumeId}`),
    enabled: !!resumeId,
    retry: false,
    refetchInterval: (q) => {
      const r = q.state.data as ResumeDetailData | undefined;
      return r && isInflight(r.parse_status) ? POLL_MS : false;
    },
  });

  const { data: talent } = useQuery({
    queryKey: ["talent", cid],
    queryFn: () => fetchJSON<TalentLite>(`/api/talents/${cid}`),
    enabled: !!cid,
    retry: false,
  });

  const patch = useMutation({
    mutationFn: (body: { name?: string; email?: string; phone?: string }) =>
      fetchJSON<ResumeDetailData>(`/api/resumes/${resumeId}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: (next) => {
      qc.setQueryData(["resume-detail", resumeId], next);
      qc.invalidateQueries({ queryKey: ["talent", cid] });
      qc.invalidateQueries({ queryKey: ["talents"] });
    },
  });

  const [downloading, setDownloading] = useState(false);
  const [downloadErr, setDownloadErr] = useState<string | null>(null);

  async function handleDownload() {
    if (!data) return;
    setDownloadErr(null);
    setDownloading(true);
    try {
      await downloadResumeFile(data.id, data.file_name);
    } catch (e) {
      setDownloadErr(e instanceof ApiError ? e.message : "下载失败");
    } finally {
      setDownloading(false);
    }
  }

  if (isLoading) {
    return (
      <main className="p-8 text-sm text-[var(--color-text-tertiary)] font-body">
        加载中…
      </main>
    );
  }
  if (error || !data) {
    return (
      <main className="p-8">
        <Empty
          icon={Users}
          title="简历不存在或无权访问"
          description={
            error instanceof ApiError ? error.message : "请返回人才库列表"
          }
        />
        <Link
          to="/talents"
          className="text-sm text-[var(--color-text-primary)] hover:underline mt-3 inline-block"
        >
          ← 返回人才库
        </Link>
      </main>
    );
  }

  const cand = data.candidate;
  const isLast = talent ? talent.resumes.length === 1 : false;

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <Link
        to={`/talents/${cid}`}
        className="self-start text-[12px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] flex items-center gap-1"
      >
        <ChevronLeft className="w-3.5 h-3.5" />
        返回候选人档案
      </Link>

      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4 min-w-0">
          <div className="w-12 h-12 rounded-full bg-[var(--color-bg-subtle)] flex items-center justify-center text-base font-heading font-semibold shrink-0">
            {cand.name.slice(0, 2)}
          </div>
          <div className="flex flex-col gap-1 min-w-0">
            <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
              {cand.name}
            </h2>
            <span className="text-[13px] text-[var(--color-text-secondary)] font-body truncate">
              {data.file_name} · {formatBytes(data.file_size)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {downloadErr && (
            <span className="text-[12px] text-[var(--color-danger)] font-body">
              {downloadErr}
            </span>
          )}
          <Button
            variant="secondary"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            {downloading ? "下载中…" : "下载原文件"}
          </Button>
          <Button
            variant="secondary"
            className="text-[var(--color-danger)]"
            disabled={deletingId === data.id || !talent}
            onClick={() =>
              deleteResume({
                resumeId: data.id,
                fileName: data.file_name,
                candidateId: cid,
                isLast,
              })
            }
          >
            <Trash2 className="w-3.5 h-3.5" />
            删除简历
          </Button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="flex flex-col gap-6 min-w-0">
          <Section icon={FileText} title="简历原文(解析后)">
            <pre className="bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)] rounded-lg p-4 text-[12px] text-[var(--color-text-secondary)] font-body whitespace-pre-wrap leading-relaxed max-h-[560px] overflow-y-auto">
              {data.parsed_text ||
                (isInflight(data.parse_status)
                  ? "解析中,稍后自动刷新…"
                  : "(解析文本为空)")}
            </pre>
          </Section>

          {data.sections && <ParseReport sections={data.sections} />}

          <Section icon={Video} title="面试题集">
            <QuestionSetsForResume resumeId={data.id} />
          </Section>

          <Section icon={ListChecks} title="岗位匹配">
            <MatchesForResume resumeId={data.id} />
          </Section>
        </div>

        <div className="flex flex-col gap-6 min-w-0">
          <Section icon={User} title="候选人信息">
            <div className="flex flex-col gap-3">
              <EditableContact
                icon={User}
                label="姓名"
                value={cand.name === "未识别" ? null : cand.name}
                placeholder="未识别,点击补全"
                onSave={(v) => patch.mutateAsync({ name: v })}
                validate={(v) =>
                  v.length === 0
                    ? "姓名不能为空"
                    : v.length > 128
                      ? "姓名过长(<=128 字符)"
                      : null
                }
                inputType="text"
              />
              <EditableContact
                icon={Mail}
                label="邮箱"
                value={data.email ?? cand.display_email}
                placeholder="未识别,点击补全"
                onSave={(v) => patch.mutateAsync({ email: v })}
                validate={(v) =>
                  v.length === 0 || /^[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}$/.test(v)
                    ? null
                    : "邮箱格式不合法"
                }
                inputType="email"
              />
              <EditableContact
                icon={Phone}
                label="手机"
                value={data.phone ?? cand.display_phone}
                placeholder="未识别,点击补全"
                onSave={(v) => patch.mutateAsync({ phone: v })}
                validate={(v) =>
                  v.length === 0 ||
                  /^(?:1[3-9]\d{9}|\+\d{1,3}[\s-]?\d{6,14})$/.test(v)
                    ? null
                    : "手机号格式不合法"
                }
                inputType="tel"
              />
            </div>
          </Section>

          {data.skills.length > 0 && (
            <Section icon={Tag} title="解析出的技能">
              <div className="flex flex-wrap gap-1.5">
                {data.skills.map((s) => (
                  <span
                    key={s}
                    className="px-2.5 py-1 rounded-md bg-[var(--color-bg-subtle)] text-[12px] text-[var(--color-text-secondary)] font-mono"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {data.parse_status !== "done" && (
            <Section icon={Sparkles} title="解析状态">
              <ParseStatusBanner
                status={data.parse_status}
                error={data.parse_error}
              />
            </Section>
          )}

          <div className="text-[11px] text-[var(--color-text-tertiary)] font-body">
            上传于 {formatRelative(data.created_at)} · 来源 {data.source}
          </div>
        </div>
      </div>
    </main>
  );
}

// T14 解析报告:特征评分结构化的分字段结果 + 置信度
function ParseReport({ sections }: { sections: ResumeSections }) {
  const conf = sections.confidence ?? {};
  const works = sections.work_experiences ?? [];
  const edus = sections.educations ?? [];
  const projects = sections.projects ?? [];

  const ConfBar = ({ label, value }: { label: string; value?: number }) => {
    if (value === undefined) return null;
    const pct = Math.round(value * 100);
    const color =
      pct >= 80
        ? "bg-emerald-500"
        : pct >= 50
          ? "bg-amber-500"
          : "bg-[var(--color-danger)]";
    return (
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-[var(--color-text-tertiary)] w-10 shrink-0">
          {label}
        </span>
        <div className="flex-1 h-1.5 rounded-full bg-[var(--color-bg-subtle)] overflow-hidden">
          <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="text-[11px] font-mono text-[var(--color-text-secondary)] w-8 text-right">
          {pct}
        </span>
      </div>
    );
  };

  const Desc = ({ items }: { items?: string[] }) =>
    items && items.length > 0 ? (
      <ul className="mt-1 pl-4 list-disc text-[11px] text-[var(--color-text-tertiary)] space-y-0.5">
        {items.slice(0, 4).map((d, i) => (
          <li key={i} className="line-clamp-2">{d}</li>
        ))}
      </ul>
    ) : null;

  return (
    <Section icon={Sparkles} title="解析报告(结构化)">
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <ConfBar label="整体" value={conf.overall} />
          <ConfBar label="信息" value={conf.profile} />
          <ConfBar label="工作" value={conf.work} />
          <ConfBar label="教育" value={conf.education} />
        </div>

        {works.length > 0 && (
          <div>
            <div className="text-[12px] font-medium text-[var(--color-text-secondary)] mb-1.5">
              工作经历({works.length})
            </div>
            <div className="flex flex-col gap-2">
              {works.map((w, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--color-border-subtle)] p-2.5"
                >
                  <div className="text-[12px] text-[var(--color-text-primary)] font-medium">
                    {w.title || "未知职位"}
                    {w.company ? ` @ ${w.company}` : ""}
                  </div>
                  <div className="text-[11px] text-[var(--color-text-tertiary)] font-mono">
                    {w.date_range || "时间未知"}
                    {w.years ? ` · ${w.years}年` : ""}
                  </div>
                  <Desc items={w.descriptions} />
                </div>
              ))}
            </div>
          </div>
        )}

        {edus.length > 0 && (
          <div>
            <div className="text-[12px] font-medium text-[var(--color-text-secondary)] mb-1.5">
              教育经历({edus.length})
            </div>
            {edus.map((e, i) => (
              <div key={i} className="text-[12px] text-[var(--color-text-primary)]">
                {e.school || "未知院校"}
                {e.degree ? ` · ${e.degree}` : ""}
                <span className="ml-2 text-[11px] text-[var(--color-text-tertiary)] font-mono">
                  {e.date_range}
                </span>
              </div>
            ))}
          </div>
        )}

        {projects.length > 0 && (
          <div>
            <div className="text-[12px] font-medium text-[var(--color-text-secondary)] mb-1.5">
              项目({projects.length})
            </div>
            {projects.map((p, i) => (
              <div key={i} className="text-[12px] text-[var(--color-text-primary)]">
                {p.name || "未命名项目"}
                <span className="ml-2 text-[11px] text-[var(--color-text-tertiary)] font-mono">
                  {p.date_range}
                </span>
              </div>
            ))}
          </div>
        )}

        {sections.skills_text && (
          <div className="text-[11px] text-[var(--color-text-tertiary)]">
            技能区:{sections.skills_text.slice(0, 120)}
          </div>
        )}
        <div className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
          {sections.parser} · sections: {(sections.section_names ?? []).join(" / ")}
        </div>
      </div>
    </Section>
  );
}

function Section({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: typeof FileText;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
      <div className="flex items-center gap-2 text-[var(--color-text-primary)]">
        <Icon className="w-4 h-4" />
        <h3 className="font-heading font-semibold text-base">{title}</h3>
      </div>
      {children}
    </section>
  );
}
