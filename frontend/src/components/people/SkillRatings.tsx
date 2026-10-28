import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Send,
  Star,
  UserCheck,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";

/* 多评价者技能评分(Degreed 式多源评分 + iCIMS 反馈请求流)

   三块消费面:
   - TalentCardSection(HR/经理):技能 chips 下挂多源评分行 + 「请求评分」
   - MySkillsCard(/me/profile):我的技能 + 自评入口 + 请求评分 + 撤回
   - MyRatingTasks(/me/profile):待我评价任务(打分/婉拒)

   数据源:/api/me/skill-rating-* 与 /api/employees/{id}/skill-rating*。 */

export type Rating = {
  id: string;
  rater_kind: "self" | "manager" | "peer" | string;
  rater_name: string | null;
  level: number;
  comment: string | null;
  created_at: string;
};

export type RatedSkill = {
  id: string;
  skill_id: string;
  name: string;
  level: number;
  status: string;
  ratings?: Rating[];
  divergent?: boolean;
};

export type RatingRequest = {
  id: string;
  skill_id: string;
  skill_name: string | null;
  employee_id: string;
  ratee_name: string | null;
  rater_employee_id: string;
  rater_name: string | null;
  rater_kind: string;
  note: string | null;
  status: "pending" | "completed" | "declined" | "cancelled";
  created_at: string;
};

export const RATER_KIND_LABEL: Record<string, string> = {
  self: "自评",
  manager: "经理",
  peer: "同事",
};

const KIND_STYLE: Record<string, string> = {
  self: "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
  manager: "bg-[var(--color-success-soft)] text-[var(--color-success)]",
  peer: "bg-[var(--color-surface)] text-[var(--color-text-secondary)]",
};

const REQ_STATUS_LABEL: Record<string, string> = {
  pending: "待评价",
  completed: "已完成",
  declined: "已婉拒",
  cancelled: "已撤回",
};

/** 单个评分来源的紧凑徽标:经理·李华 Lv.4(悬停看评语) */
export function RatingChip({ r }: { r: Rating }) {
  return (
    <span
      title={r.comment || undefined}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-body ${
        KIND_STYLE[r.rater_kind] ?? KIND_STYLE.peer
      }`}
    >
      {RATER_KIND_LABEL[r.rater_kind] ?? r.rater_kind}
      {r.rater_kind !== "self" && r.rater_name ? `·${r.rater_name}` : ""}
      <span className="font-medium">Lv.{r.level}</span>
    </span>
  );
}

/** 某技能的多源评分行(含分歧警告徽标) */
export function RatingLine({ skill }: { skill: RatedSkill }) {
  const ratings = skill.ratings ?? [];
  if (ratings.length === 0) return null;
  return (
    <div
      className="flex items-center gap-1.5 flex-wrap text-[12px] font-body"
      data-testid={`rating-line-${skill.skill_id}`}
    >
      <span className="font-medium text-[var(--color-text-primary)]">
        {skill.name}
      </span>
      {ratings.map((r) => (
        <RatingChip key={r.id} r={r} />
      ))}
      {skill.divergent && (
        <span
          className="inline-flex items-center gap-1 rounded-full bg-[var(--color-warning-soft)] text-[var(--color-warning)] px-2 py-0.5 text-[11px] font-body"
          title="不同来源评分差距 ≥2 级,建议校准"
        >
          <AlertTriangle className="w-3 h-3" />
          评分分歧
        </span>
      )}
    </div>
  );
}

/** 1-5 级点选器 */
export function LevelPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="flex items-center gap-1.5" data-testid="level-picker">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          data-level={n}
          onClick={() => onChange(n)}
          className={`w-8 h-8 rounded-lg border text-[13px] font-body transition-colors ${
            n <= value
              ? "bg-[var(--color-accent)] border-[var(--color-accent)] text-white"
              : "bg-white border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)] hover:border-[var(--color-accent)]"
          }`}
        >
          {n}
        </button>
      ))}
      <span className="ml-1 text-[12px] text-[var(--color-text-tertiary)] font-body">
        Lv.{value}
      </span>
    </div>
  );
}

/** 打分弹窗:任务评分与自评共用(postUrl 区分) */
export function RateModal({
  open,
  onClose,
  title,
  subtitle,
  postUrl,
  invalidateKeys,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  postUrl: string;
  invalidateKeys: string[][];
}) {
  const qc = useQueryClient();
  const [level, setLevel] = useState(3);
  const [comment, setComment] = useState("");
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: () =>
      fetchJSON(postUrl, {
        method: "POST",
        body: JSON.stringify({ level, comment: comment.trim() || null }),
      }),
    onSuccess: () => {
      invalidateKeys.forEach((queryKey) =>
        qc.invalidateQueries({ queryKey })
      );
      setLevel(3);
      setComment("");
      setErr("");
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "提交失败"),
  });
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      description={subtitle}
      width={440}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button onClick={() => mut.mutate()} disabled={mut.isPending}>
            {mut.isPending ? "提交中…" : "提交评分"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <div className="text-[13px] font-body text-[var(--color-text-secondary)]">
            评分等级(1-5)
          </div>
          <LevelPicker value={level} onChange={setLevel} />
        </div>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          评语(可选)
          <Input
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="如:能独立完成中等复杂度需求"
          />
        </label>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)]">{err}</div>
        )}
      </div>
    </Modal>
  );
}

/** 「请求评分」弹窗:选技能 + 评价人(默认直属上级)+ 附言。 */
export function RequestRatingModal({
  open,
  onClose,
  postUrl,
  skills,
  excludeEmployeeId,
  invalidateKeys,
}: {
  open: boolean;
  onClose: () => void;
  postUrl: string;
  skills: { skill_id: string; name: string }[];
  excludeEmployeeId?: string;
  invalidateKeys: string[][];
}) {
  const qc = useQueryClient();
  const [skillId, setSkillId] = useState("");
  const [raterId, setRaterId] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const { data: emps } = useQuery<{ items: { id: string; name: string }[] }>({
    queryKey: ["emp-options-all"],
    queryFn: () => fetchJSON("/api/employees?limit=200"),
    enabled: open,
  });
  const raterOptions = [
    { value: "", label: "默认:被评人直属上级" },
    ...(emps?.items ?? [])
      .filter((e) => e.id !== excludeEmployeeId)
      .map((e) => ({ value: e.id, label: e.name })),
  ];
  const mut = useMutation({
    mutationFn: () =>
      fetchJSON(postUrl, {
        method: "POST",
        body: JSON.stringify({
          skill_id: skillId,
          rater_employee_id: raterId || null,
          note: note.trim() || null,
        }),
      }),
    onSuccess: () => {
      invalidateKeys.forEach((queryKey) =>
        qc.invalidateQueries({ queryKey })
      );
      setSkillId("");
      setRaterId("");
      setNote("");
      setErr("");
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "发起失败"),
  });
  function submit() {
    if (!skillId) {
      setErr("请选择要评的技能");
      return;
    }
    mut.mutate();
  }
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="请求技能评分"
      description="请上级或同事为该技能打分;评分将作为确认证据写进能力标签。"
      width={440}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button onClick={submit} disabled={mut.isPending}>
            <Send className="w-3.5 h-3.5" />
            {mut.isPending ? "发送中…" : "发起请求"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1 text-[13px] font-body">
          技能
          <Select
            value={skillId}
            onChange={setSkillId}
            options={[
              { value: "", label: "— 选择技能 —" },
              ...skills.map((s) => ({ value: s.skill_id, label: s.name })),
            ]}
          />
        </div>
        <div className="flex flex-col gap-1 text-[13px] font-body">
          评价人
          <Select
            value={raterId}
            onChange={setRaterId}
            options={raterOptions}
          />
        </div>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          附言(可选)
          <Input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="如:请按最近项目表现评"
          />
        </label>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)]">{err}</div>
        )}
      </div>
    </Modal>
  );
}

/** 待处理请求的小行(技能 → 评价人,可撤回) */
export function PendingRequestLine({
  q,
  onCancel,
}: {
  q: Pick<RatingRequest, "id" | "skill_name" | "rater_name" | "rater_kind">;
  onCancel?: () => void;
}) {
  return (
    <div className="flex items-center gap-2 text-[12px] font-body">
      <Clock className="w-3.5 h-3.5 text-[var(--color-warning)]" />
      <span className="text-[var(--color-text-secondary)]">
        待评价:「{q.skill_name}」→{" "}
        {RATER_KIND_LABEL[q.rater_kind] ?? q.rater_kind}
        {q.rater_name ? `·${q.rater_name}` : ""}
      </span>
      {onCancel && (
        <button
          className="text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] font-body"
          onClick={onCancel}
        >
          撤回
        </button>
      )}
    </div>
  );
}

/* ================== 员工自助侧(/me/profile) ================== */

type MyCard = {
  employee: { id: string; name: string };
  skills_confirmed: RatedSkill[];
  pending_rating_requests?: {
    id: string;
    skill_id: string;
    skill_name: string | null;
    rater_name: string | null;
    rater_kind: string;
  }[];
};

/** 「我的技能」卡:确认技能 + 多源评分 + 自评/请求评分/撤回。 */
export function MySkillsCard() {
  const [reqOpen, setReqOpen] = useState(false);
  const [selfRateSkill, setSelfRateSkill] = useState<RatedSkill | null>(null);
  const qc = useQueryClient();
  const { data: card } = useQuery<MyCard>({
    queryKey: ["me-talent-card"],
    queryFn: () => fetchJSON("/api/me/talent-card"),
  });
  const cancelMut = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/skill-rating-requests/${id}/cancel`, {
        method: "POST",
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["me-talent-card"] }),
  });
  if (!card) return null;
  const skills = card.skills_confirmed ?? [];
  const pending = card.pending_rating_requests ?? [];
  const inv = [["me-talent-card"], ["me-rating-reqs"], ["me-rating-tasks"]];
  return (
    <section
      className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4"
      data-testid="my-skills-card"
    >
      <div className="flex items-center justify-between">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
          我的技能
        </h3>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setReqOpen(true)}
          disabled={skills.length === 0}
        >
          <UserCheck className="w-3.5 h-3.5" />
          请求评分
        </Button>
      </div>
      {skills.length === 0 && (
        <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          暂无技能标签 — 项目、绩效与评审会自动积累。
        </div>
      )}
      {skills.length > 0 && (
        <div className="flex flex-col gap-2.5">
          {skills.map((s) => (
            <div
              key={s.id}
              className="border border-[var(--color-border-subtle)] rounded-lg px-3 py-2.5 flex flex-col gap-1.5"
              data-testid={`my-skill-${s.skill_id}`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium font-body text-[var(--color-text-primary)]">
                    {s.name}
                  </span>
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    Lv.{s.level}
                  </span>
                </div>
                <button
                  className="inline-flex items-center gap-1 text-[11px] text-[var(--color-accent)] font-body hover:underline"
                  data-testid={`self-rate-${s.skill_id}`}
                  onClick={() => setSelfRateSkill(s)}
                >
                  <Star className="w-3 h-3" />
                  自评
                </button>
              </div>
              <RatingLine skill={s} />
            </div>
          ))}
        </div>
      )}
      {pending.length > 0 && (
        <div className="flex flex-col gap-1.5 pt-1 border-t border-[var(--color-border-subtle)]">
          {pending.map((q) => (
            <PendingRequestLine
              key={q.id}
              q={q}
              onCancel={() => cancelMut.mutate(q.id)}
            />
          ))}
        </div>
      )}
      <RequestRatingModal
        open={reqOpen}
        onClose={() => setReqOpen(false)}
        postUrl="/api/me/skill-rating-requests"
        skills={skills}
        excludeEmployeeId={card.employee.id}
        invalidateKeys={inv}
      />
      {selfRateSkill && (
        <RateModal
          open={!!selfRateSkill}
          onClose={() => setSelfRateSkill(null)}
          title={`自评「${selfRateSkill.name}」`}
          subtitle="自评分会与其他来源评分并列展示,供校准参考。"
          postUrl={`/api/me/skills/${selfRateSkill.skill_id}/rate`}
          invalidateKeys={inv}
        />
      )}
    </section>
  );
}

/** 「待我评价」卡:pending 任务内联打分;已处理列表折叠展示。 */
export function MyRatingTasks() {
  const qc = useQueryClient();
  const [rateTask, setRateTask] = useState<RatingRequest | null>(null);
  const [declineTask, setDeclineTask] = useState<RatingRequest | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const { data } = useQuery<{ items: RatingRequest[]; pending_count: number }>({
    queryKey: ["me-rating-tasks"],
    queryFn: () => fetchJSON("/api/me/skill-rating-tasks"),
  });
  const declineMut = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/me/skill-rating-tasks/${id}/decline`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-rating-tasks"] });
      setDeclineTask(null);
    },
  });
  if (!data || data.items.length === 0) return null;
  const pending = data.items.filter((q) => q.status === "pending");
  const done = data.items.filter((q) => q.status !== "pending");
  return (
    <section
      className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4"
      data-testid="my-rating-tasks"
    >
      <div className="flex items-center gap-2">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
          待我评价
        </h3>
        {pending.length > 0 && (
          <span className="rounded-full bg-[var(--color-warning-soft)] text-[var(--color-warning)] px-2.5 py-0.5 text-[12px] font-body">
            {pending.length} 条待处理
          </span>
        )}
      </div>
      {pending.length === 0 && (
        <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
          暂无待评价任务
        </div>
      )}
      {pending.map((q) => (
        <div
          key={q.id}
          className="border border-[var(--color-border-subtle)] rounded-lg p-3 flex flex-col gap-2"
          data-testid={`rating-task-${q.id}`}
        >
          <div className="flex items-center gap-2 text-[13px] font-body flex-wrap">
            <span className="font-medium text-[var(--color-text-primary)]">
              {q.ratee_name}
            </span>
            <span className="text-[var(--color-text-secondary)]">
              请你评价「{q.skill_name}」
            </span>
            <span className="rounded-full bg-[var(--color-surface)] px-2 py-0.5 text-[11px] text-[var(--color-text-tertiary)] font-body">
              {RATER_KIND_LABEL[q.rater_kind] ?? q.rater_kind}评价
            </span>
          </div>
          {q.note && (
            <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
              附言:{q.note}
            </div>
          )}
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => setRateTask(q)}>
              <Star className="w-3.5 h-3.5" />
              打分
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setDeclineTask(q)}
            >
              <X className="w-3.5 h-3.5" />
              婉拒
            </Button>
          </div>
        </div>
      ))}
      {done.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <button
            className="text-[12px] text-[var(--color-text-tertiary)] font-body text-left"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? "收起历史" : `历史记录(${done.length})`}
          </button>
          {showHistory &&
            done.map((q) => (
              <div
                key={q.id}
                className="flex items-center gap-2 text-[12px] font-body text-[var(--color-text-tertiary)]"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>
                  {q.ratee_name} ·「{q.skill_name}」 ·{" "}
                  {REQ_STATUS_LABEL[q.status] ?? q.status}
                </span>
              </div>
            ))}
        </div>
      )}
      {rateTask && (
        <RateModal
          open={!!rateTask}
          onClose={() => setRateTask(null)}
          title={`评价 ${rateTask.ratee_name} 的「${rateTask.skill_name}」`}
          subtitle={rateTask.note || undefined}
          postUrl={`/api/me/skill-rating-tasks/${rateTask.id}/rate`}
          invalidateKeys={[["me-rating-tasks"], ["me-talent-card"]]}
        />
      )}
      {declineTask && (
        <Modal
          open={!!declineTask}
          onClose={() => setDeclineTask(null)}
          title="婉拒评分请求"
          description={`不为 ${declineTask.ratee_name} 的「${declineTask.skill_name}」评分。`}
          width={400}
          footer={
            <>
              <Button variant="secondary" onClick={() => setDeclineTask(null)}>
                取消
              </Button>
              <Button
                onClick={() => declineMut.mutate(declineTask.id)}
                disabled={declineMut.isPending}
              >
                确认婉拒
              </Button>
            </>
          }
        >
          <div className="text-[13px] font-body text-[var(--color-text-secondary)]">
            对方会收到婉拒通知;可稍后让对方重新发起。
          </div>
        </Modal>
      )}
    </section>
  );
}
