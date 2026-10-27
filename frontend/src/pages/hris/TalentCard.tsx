import { useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Briefcase,
  CheckCheck,
  ClipboardList,
  Plus,
  Sparkles,
  StickyNote,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";

/* P 批次 — 员工人才画像卡(嵌入 EmployeeDetail)。
   数据源:GET /api/employees/{id}/talent-card
   写操作:技能确认/驳回、手动加技能、上传洞察(会议纪要入口)。 */

type Skill = {
  id: string;
  skill_id: string;
  name: string;
  category: string;
  level: number;
  source: string;
  status: string;
  evidence?: { kind?: string; quote?: string; note?: string } | null;
  note?: string | null;
};

type Card = {
  employee: {
    id: string;
    name: string;
    designation: string | null;
    org_name: string | null;
    manager_name: string | null;
    hire_days: number | null;
    status: string;
  };
  skills_confirmed: Skill[];
  skills_suggested?: Skill[];
  suggested_count: number;
  interviews: { id: string; created_at: string; recommendation: string | null }[];
  projects: {
    project_id: string;
    name: string;
    status: string;
    role: string | null;
    joined_via: string;
    contribution: string | null;
  }[];
  appraisals: { id: string; status: string; manager_rating: number | null }[];
  insights: {
    id: string;
    kind: string;
    title: string | null;
    body: string;
    source_ref?: { minutes?: Minutes } | null;
    created_at: string;
  }[];
};

type Minutes = {
  summary?: string;
  key_points?: string[];
  decisions?: string[];
  action_items?: { item: string; owner?: string; due?: string }[];
  sentiment?: string;
};

const KIND_LABEL: Record<string, string> = {
  meeting: "会议纪要",
  project_review: "项目复盘",
  manager_note: "主管随手评",
  milestone: "里程碑",
};

const SOURCE_LABEL: Record<string, string> = {
  resume: "简历",
  interview: "面试",
  appraisal: "绩效",
  project: "项目",
  insight: "洞察",
  manual: "手动",
  ai: "AI",
};

const CAT_COLOR: Record<string, string> = {
  tech: "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
  language: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
  soft: "bg-[var(--color-warning-soft)] text-[var(--color-warning)]",
  cert: "bg-purple-50 text-purple-600",
  other: "bg-[var(--color-surface)] text-[var(--color-text-secondary)]",
};

function SkillChip({ s, pending }: { s: Skill; pending?: boolean }) {
  return (
    <span
      title={`Lv.${s.level} · ${SOURCE_LABEL[s.source] ?? s.source}${s.evidence?.quote ? ` · ${s.evidence.quote}` : ""}`}
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-body border ${
        pending
          ? "border-dashed border-[var(--color-warning-stroke)] " + (CAT_COLOR[s.category] ?? CAT_COLOR.other)
          : "border-transparent " + (CAT_COLOR[s.category] ?? CAT_COLOR.other)
      }`}
    >
      {s.name}
      <span className="opacity-60">Lv.{s.level}</span>
      {pending && <span className="opacity-60">·待确认</span>}
    </span>
  );
}

export function TalentCardSection({ employeeId }: { employeeId: string }) {
  const qc = useQueryClient();
  const [insightOpen, setInsightOpen] = useState(false);
  const [skillOpen, setSkillOpen] = useState(false);
  const { data: card, isLoading } = useQuery<Card>({
    queryKey: ["talent-card", employeeId],
    queryFn: () => fetchJSON(`/api/employees/${employeeId}/talent-card`),
  });
  const invalidate = () =>
    qc.invalidateQueries({ queryKey: ["talent-card", employeeId] });

  const confirmAll = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/employees/${employeeId}/skill-suggestions/confirm`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: invalidate,
  });
  const rejectAll = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/employees/${employeeId}/skill-suggestions/reject`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: invalidate,
  });

  if (isLoading || !card) {
    return (
      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6">
        <div className="text-sm text-[var(--color-text-tertiary)] font-body">
          人才画像加载中…
        </div>
      </section>
    );
  }

  const e = card.employee;
  const suggested = card.skills_suggested ?? [];

  return (
    <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-5">
      {/* 头部 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            人才画像
          </h3>
          {e.hire_days != null && (
            <span className="rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] px-2.5 py-0.5 text-[12px] font-body">
              入职第 {e.hire_days + 1} 天
            </span>
          )}
          {suggested.length > 0 && (
            <span className="rounded-full bg-[var(--color-warning-soft)] text-[var(--color-warning)] px-2.5 py-0.5 text-[12px] font-body">
              {suggested.length} 条建议待确认
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" onClick={() => setSkillOpen(true)}>
            <Plus className="w-3.5 h-3.5" />
            加技能
          </Button>
          <Button variant="secondary" size="sm" onClick={() => setInsightOpen(true)}>
            <StickyNote className="w-3.5 h-3.5" />
            上传会议纪要
          </Button>
        </div>
      </div>

      {/* 技能标签 */}
      <div className="flex flex-col gap-2">
        <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
          能力标签
        </div>
        <div className="flex flex-wrap gap-1.5">
          {card.skills_confirmed.length === 0 && suggested.length === 0 && (
            <span className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无技能标签 — 项目完成、绩效评语、会议纪要会自动产生建议。
            </span>
          )}
          {card.skills_confirmed.map((s) => (
            <SkillChip key={s.id} s={s} />
          ))}
          {suggested.map((s) => (
            <SkillChip key={s.id} s={s} pending />
          ))}
        </div>
        {suggested.length > 0 && (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={() => confirmAll.mutate()}
              disabled={confirmAll.isPending}
            >
              <CheckCheck className="w-3.5 h-3.5" />
              全部确认({suggested.length})
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => rejectAll.mutate()}
              disabled={rejectAll.isPending}
            >
              <X className="w-3.5 h-3.5" />
              全部驳回
            </Button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* 项目经历 */}
        <div className="flex flex-col gap-2">
          <div className="text-[12px] text-[var(--color-text-tertiary)] font-body flex items-center gap-1">
            <Briefcase className="w-3.5 h-3.5" /> 项目经历
          </div>
          {card.projects.length === 0 && (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无项目记录
            </div>
          )}
          {card.projects.map((p) => (
            <div
              key={p.project_id}
              className="border border-[var(--color-border-subtle)] rounded-lg p-3 flex flex-col gap-1"
            >
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                  {p.name}
                </span>
                <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                  {p.status === "active" ? "进行中" : p.status === "done" ? "已完成" : p.status}
                </span>
                {p.joined_via === "ai_match" && (
                  <span className="inline-flex items-center gap-0.5 text-[11px] text-[var(--color-accent)] font-body">
                    <Sparkles className="w-3 h-3" /> AI 推荐
                  </span>
                )}
              </div>
              <div className="text-[12px] text-[var(--color-text-secondary)] font-body">
                {p.role || "成员"}
                {p.contribution ? ` · ${p.contribution.slice(0, 40)}` : ""}
              </div>
            </div>
          ))}
        </div>

        {/* 面试评价 + 绩效 + 时间线 */}
        <div className="flex flex-col gap-3">
          {card.interviews.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                面试评价
              </div>
              {card.interviews.map((iv) => (
                <div
                  key={iv.id}
                  className="text-[12px] font-body text-[var(--color-text-secondary)]"
                >
                  {iv.created_at} · {iv.recommendation || "已完成"}
                </div>
              ))}
            </div>
          )}
          {card.appraisals.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                绩效
              </div>
              {card.appraisals.map((a) => (
                <div
                  key={a.id}
                  className="text-[12px] font-body text-[var(--color-text-secondary)]"
                >
                  {a.status === "reviewed" ? "已评定" : a.status} · 上级评分{" "}
                  {a.manager_rating ?? "—"}
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-col gap-1.5">
            <div className="text-[12px] text-[var(--color-text-tertiary)] font-body flex items-center gap-1">
              <ClipboardList className="w-3.5 h-3.5" /> 洞察时间线
            </div>
            {card.insights.length === 0 && (
              <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
                暂无洞察 — 点「上传会议纪要」开始积累。
              </div>
            )}
            {card.insights.slice(0, 5).map((n) => (
              <InsightRow
                key={n.id}
                note={n}
                onDone={invalidate}
              />
            ))}
          </div>
        </div>
      </div>

      <AddSkillModal
        employeeId={employeeId}
        open={skillOpen}
        onClose={() => setSkillOpen(false)}
      />
      <AddInsightModal
        employeeId={employeeId}
        open={insightOpen}
        onClose={() => setInsightOpen(false)}
      />
    </section>
  );
}

/* 单条洞察:会议纪要可展开 AI 结构化纪要,未生成过的 meeting 便签提供「生成纪要」。 */
function InsightRow({
  note,
  onDone,
}: {
  note: Card["insights"][number];
  onDone: () => void;
}) {
  const [open, setOpen] = useState(false);
  const minutes = note.source_ref?.minutes;
  const gen = useMutation({
    mutationFn: () =>
      fetchJSON(`/api/ai/meeting-minutes`, {
        method: "POST",
        body: JSON.stringify({ note_id: note.id }),
      }),
    onSuccess: onDone,
  });

  return (
    <div className="border-l-2 border-[var(--color-border-subtle)] pl-3 py-1">
      <div className="text-[12px] font-body text-[var(--color-text-primary)] flex items-center gap-1.5 flex-wrap">
        <span className="text-[var(--color-text-tertiary)]">
          [{KIND_LABEL[note.kind] ?? note.kind}]
        </span>{" "}
        {note.title || note.body.slice(0, 30)}
        {minutes && (
          <button
            className="text-[11px] text-[var(--color-accent)] font-body"
            onClick={() => setOpen(!open)}
          >
            {open ? "收起纪要" : "查看纪要"}
          </button>
        )}
        {!minutes && note.kind === "meeting" && (
          <button
            className="text-[11px] text-[var(--color-accent)] font-body disabled:opacity-50"
            disabled={gen.isPending}
            onClick={() => gen.mutate()}
            title="AI 整理摘要/决议/行动项"
          >
            <Sparkles className="w-3 h-3 inline" />{" "}
            {gen.isPending ? "生成中…" : "生成纪要"}
          </button>
        )}
      </div>
      <div className="text-[11px] font-body text-[var(--color-text-tertiary)]">
        {note.created_at}
      </div>
      {open && minutes && (
        <div className="mt-1.5 rounded-lg bg-[var(--color-surface)] p-3 flex flex-col gap-1.5">
          {minutes.summary && (
            <div className="text-[12px] font-body text-[var(--color-text-primary)]">
              {minutes.summary}
            </div>
          )}
          {(minutes.decisions?.length ?? 0) > 0 && (
            <div className="text-[12px] font-body">
              <span className="text-[var(--color-success)]">决议:</span>
              <span className="text-[var(--color-text-secondary)]">
                {" "}
                {minutes.decisions!.join(";")}
              </span>
            </div>
          )}
          {(minutes.action_items?.length ?? 0) > 0 && (
            <div className="text-[12px] font-body">
              <span className="text-[var(--color-accent)]">行动项:</span>
              <ul className="mt-0.5 text-[var(--color-text-secondary)]">
                {minutes.action_items!.map((a, i) => {
                  const meta = [a.owner, a.due].filter(Boolean).join(" · ");
                  return (
                    <li key={i}>
                      · {a.item}
                      {meta ? `(${meta})` : ""}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AddSkillModal({
  employeeId,
  open,
  onClose,
}: {
  employeeId: string;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON(`/api/employees/${employeeId}/skills`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["talent-card", employeeId] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });
  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      level: Number(fd.get("level") || 3),
      note: String(fd.get("note") || "").trim() || null,
    });
  }
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="手动加技能"
      width={420}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="add-skill" disabled={mut.isPending}>
            保存
          </Button>
        </>
      }
    >
      <form id="add-skill" onSubmit={submit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-[13px] font-body">
          技能名
          <Input name="name" required placeholder="如:系统设计 / TypeScript / 跨团队沟通" />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          等级(1-5)
          <Input name="level" type="number" min={1} max={5} defaultValue={3} />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          备注(可选)
          <Input name="note" />
        </label>
        {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
      </form>
    </Modal>
  );
}

function AddInsightModal({
  employeeId,
  open,
  onClose,
}: {
  employeeId: string;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [kind, setKind] = useState("meeting");
  const [err, setErr] = useState("");
  const [result, setResult] = useState<{ suggested?: string[] } | null>(null);
  const [bodyText, setBodyText] = useState("");
  const [minutes, setMinutes] = useState<Minutes | null>(null);
  const mut = useMutation<{ suggested?: string[] }, Error, Record<string, unknown>>({
    mutationFn: (body) =>
      fetchJSON(`/api/employees/${employeeId}/insights`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["talent-card", employeeId] });
      setResult(r);
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });
  const titleRef = useRef<HTMLInputElement>(null);
  const minutesMut = useMutation<Minutes & { degraded?: boolean }, Error, void>({
    mutationFn: () =>
      fetchJSON(`/api/ai/meeting-minutes`, {
        method: "POST",
        body: JSON.stringify({
          title: titleRef.current?.value ?? "",
          body: bodyText,
        }),
      }),
    onSuccess: (m) => setMinutes(m),
    onError: (e) =>
      setErr(e instanceof ApiError ? e.message : "纪要生成失败"),
  });
  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      kind,
      title: String(fd.get("title") || "").trim() || null,
      body: String(fd.get("body") || "").trim(),
      visibility: String(fd.get("visibility") || "manager_only"),
      extract_skills: true,
      ...(minutes ? { source_ref: { minutes } } : {}),
    });
  }
  function close() {
    setResult(null);
    setErr("");
    setMinutes(null);
    setBodyText("");
    onClose();
  }
  return (
    <Modal
      open={open}
      onClose={close}
      title="上传洞察 / 会议纪要"
      width={560}
      footer={
        result ? (
          <Button onClick={close}>完成</Button>
        ) : (
          <>
            <Button variant="secondary" onClick={close}>
              取消
            </Button>
            <Button type="submit" form="add-insight" disabled={mut.isPending}>
              {mut.isPending ? "AI 分析中…" : "保存并分析"}
            </Button>
          </>
        )
      }
    >
      {result ? (
        <div className="flex flex-col gap-2 text-[13px] font-body">
          <div className="text-[var(--color-success)]">已保存</div>
          <div>
            AI 提取建议技能:
            {result.suggested && result.suggested.length > 0 ? (
              <span className="text-[var(--color-text-primary)]">
                {" "}
                {result.suggested.join("、")}(待确认)
              </span>
            ) : (
              " 无可提取的新技能"
            )}
          </div>
        </div>
      ) : (
        <form id="add-insight" onSubmit={submit} className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1 text-[13px] font-body">
              类型
              <Select
                value={kind}
                onChange={setKind}
                options={Object.entries(KIND_LABEL).map(([value, label]) => ({
                  value,
                  label,
                }))}
              />
            </label>
            <label className="flex flex-col gap-1 text-[13px] font-body">
              可见性
              <select
                name="visibility"
                className="h-9 rounded-lg border border-[var(--color-border-subtle)] px-3 text-[13px] font-body bg-white"
                defaultValue="manager_only"
              >
                <option value="manager_only">仅管理者可见</option>
                <option value="employee_visible">员工本人可见</option>
              </select>
            </label>
          </div>
          <label className="flex flex-col gap-1 text-[13px] font-body">
            标题(可选)
            <Input name="title" ref={titleRef} placeholder="如:Q4 规划评审会" />
          </label>
          <label className="flex flex-col gap-1 text-[13px] font-body">
            内容(会议纪要 / 复盘 / 评价)
            <textarea
              name="body"
              required
              rows={6}
              value={bodyText}
              onChange={(e) => setBodyText(e.target.value)}
              className="rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-[13px] font-body"
              placeholder="粘贴会议纪要或写对该员工的观察…AI 会从中提取能力证据生成技能建议。"
            />
          </label>
          {kind === "meeting" && (
            <div className="flex flex-col gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                className="self-start"
                disabled={minutesMut.isPending || !bodyText.trim()}
                onClick={() => minutesMut.mutate()}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {minutesMut.isPending ? "AI 整理中…" : "AI 整理纪要"}
              </Button>
              {minutes && (
                <div className="rounded-lg border border-[var(--color-accent-soft)] bg-[var(--color-accent-soft)]/40 p-3 flex flex-col gap-1.5 text-[12px] font-body">
                  {minutes.summary && (
                    <div className="text-[var(--color-text-primary)]">
                      {minutes.summary}
                    </div>
                  )}
                  {(minutes.decisions?.length ?? 0) > 0 && (
                    <div>
                      <span className="text-[var(--color-success)]">决议:</span>{" "}
                      {minutes.decisions!.join(";")}
                    </div>
                  )}
                  {(minutes.action_items?.length ?? 0) > 0 && (
                    <div>
                      <span className="text-[var(--color-accent)]">行动项:</span>
                      <ul className="mt-0.5">
                        {minutes.action_items!.map((a, i) => {
                          const meta = [a.owner, a.due].filter(Boolean).join(" · ");
                          return (
                            <li key={i}>
                              · {a.item}
                              {meta ? `(${meta})` : ""}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                  <div className="text-[11px] text-[var(--color-text-tertiary)]">
                    纪要将随便签一同保存
                  </div>
                </div>
              )}
            </div>
          )}
          {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
        </form>
      )}
    </Modal>
  );
}
