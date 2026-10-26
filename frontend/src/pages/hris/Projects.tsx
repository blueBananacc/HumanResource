import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Briefcase, Plus, Sparkles, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Select } from "@/components/ui/select";
import { fetchJSON, ApiError } from "@/lib/api";

/* P 批次 — 轻量项目页:列表 + 详情(成员/所需技能/状态流转)。
   项目完成 → 后端自动给成员画像产生技能建议。 */

type Project = {
  id: string;
  name: string;
  description: string | null;
  status: string;
  owner_name: string | null;
  start_date: string;
  end_date: string;
  required_skills: { skill_id?: string; name?: string; min_level?: number }[];
  member_count: number;
  members?: Member[];
};

type Member = {
  employee_id: string;
  name: string;
  designation: string | null;
  role: string | null;
  allocation_pct: number;
  joined_via: string;
  contribution: string | null;
  left_at: string | null;
};

type EmpOpt = { id: string; name: string };

const STATUS_LABEL: Record<string, string> = {
  draft: "草稿",
  active: "进行中",
  done: "已完成",
  cancelled: "已取消",
};

export function Projects() {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const { data } = useQuery<{ items: Project[] }>({
    queryKey: ["projects"],
    queryFn: () => fetchJSON("/api/projects"),
  });
  const { data: detail } = useQuery<Project>({
    queryKey: ["project", selected],
    queryFn: () => fetchJSON(`/api/projects/${selected}`),
    enabled: !!selected,
  });
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["projects"] });
    qc.invalidateQueries({ queryKey: ["project", selected] });
  };

  const items = data?.items ?? [];

  return (
    <main className="p-8 max-w-[1100px] mx-auto w-full flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            项目
          </h2>
          <p className="text-[13px] text-[var(--color-text-secondary)] font-body mt-1">
            人才归属载体:项目成员进员工画像,项目完成自动提炼能力建议。
          </p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <Plus className="w-3.5 h-3.5" />
          新建项目
        </Button>
      </div>

      <div className="grid grid-cols-[1fr_380px] gap-4 items-start">
        {/* 列表 */}
        <section className="flex flex-col gap-2">
          {items.length === 0 && (
            <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-10 text-center text-[13px] text-[var(--color-text-tertiary)] font-body">
              暂无项目 — 新建后可对助手说「给项目找会 XX 的人」。
            </div>
          )}
          {items.map((p) => (
            <button
              key={p.id}
              onClick={() => setSelected(p.id)}
              className={`bg-white border rounded-xl shadow-[var(--shadow-card)] p-4 text-left flex items-center justify-between gap-3 transition-colors ${
                selected === p.id
                  ? "border-[var(--color-accent)]"
                  : "border-[var(--color-border-subtle)] hover:border-[var(--color-accent-soft)]"
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <Briefcase className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
                <div className="min-w-0">
                  <div className="text-[14px] font-medium text-[var(--color-text-primary)] font-body truncate">
                    {p.name}
                  </div>
                  <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
                    {p.owner_name || "未指定负责人"} · {p.member_count} 人
                  </div>
                </div>
              </div>
              <span
                className={`shrink-0 rounded-full px-2.5 py-0.5 text-[12px] font-body ${
                  p.status === "active"
                    ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                    : p.status === "done"
                      ? "bg-[var(--color-success-soft)] text-[var(--color-success)]"
                      : "bg-[var(--color-surface)] text-[var(--color-text-secondary)]"
                }`}
              >
                {STATUS_LABEL[p.status] ?? p.status}
              </span>
            </button>
          ))}
        </section>

        {/* 详情 */}
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-5 flex flex-col gap-4">
          {!detail ? (
            <div className="text-[13px] text-[var(--color-text-tertiary)] font-body">
              选择左侧项目查看成员与状态
            </div>
          ) : (
            <ProjectDetail p={detail} onChanged={invalidate} />
          )}
        </section>
      </div>

      <CreateProjectModal open={open} onClose={() => setOpen(false)} onCreated={invalidate} />
    </main>
  );
}

function ProjectDetail({ p, onChanged }: { p: Project; onChanged: () => void }) {
  const [addOpen, setAddOpen] = useState(false);
  const [err, setErr] = useState("");
  const setStatus = useMutation({
    mutationFn: (status: string) =>
      fetchJSON(`/api/projects/${p.id}`, {
        method: "PUT",
        body: JSON.stringify({
          name: p.name,
          description: p.description,
          status,
          owner_employee_id: null,
          required_skills: p.required_skills,
        }),
      }),
    onSuccess: onChanged,
    onError: (e) => setErr(e instanceof ApiError ? e.message : "操作失败"),
  });

  return (
    <>
      <div>
        <div className="text-[15px] font-heading font-semibold text-[var(--color-text-primary)]">
          {p.name}
        </div>
        <div className="text-[12px] text-[var(--color-text-tertiary)] font-body mt-1">
          {STATUS_LABEL[p.status]} · {p.owner_name || "无负责人"}
          {p.start_date ? ` · ${p.start_date}` : ""}
          {p.end_date ? ` ~ ${p.end_date}` : ""}
        </div>
        {p.description && (
          <p className="text-[13px] text-[var(--color-text-secondary)] font-body mt-2">
            {p.description}
          </p>
        )}
      </div>

      {p.required_skills.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            所需技能
          </div>
          <div className="flex flex-wrap gap-1.5">
            {p.required_skills.map((s, i) => (
              <span
                key={i}
                className="rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] px-2.5 py-0.5 text-[12px] font-body"
              >
                {s.name || s.skill_id}
                {s.min_level ? ` Lv.${s.min_level}+` : ""}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            成员({(p.members ?? []).filter((m) => !m.left_at).length})
          </div>
          <Button variant="secondary" size="sm" onClick={() => setAddOpen(true)}>
            <UserPlus className="w-3.5 h-3.5" />
            加成员
          </Button>
        </div>
        {(p.members ?? []).map((m) => (
          <div
            key={m.employee_id}
            className={`border border-[var(--color-border-subtle)] rounded-lg p-2.5 ${m.left_at ? "opacity-50" : ""}`}
          >
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-medium text-[var(--color-text-primary)] font-body">
                {m.name}
              </span>
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                {m.role || "成员"} · {m.allocation_pct}%
              </span>
              {m.joined_via === "ai_match" && (
                <Sparkles className="w-3 h-3 text-[var(--color-accent)]" />
              )}
            </div>
            {m.contribution && (
              <div className="text-[12px] text-[var(--color-text-secondary)] font-body mt-0.5">
                {m.contribution.slice(0, 60)}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-2 pt-1">
        {p.status === "draft" && (
          <Button size="sm" onClick={() => setStatus.mutate("active")}>
            启动项目
          </Button>
        )}
        {p.status === "active" && (
          <Button size="sm" onClick={() => setStatus.mutate("done")}>
            标记完成(提炼成员能力)
          </Button>
        )}
        {err && (
          <div className="text-[12px] text-[var(--color-danger)] font-body">{err}</div>
        )}
      </div>

      <AddMemberModal
        project={p}
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onChanged={onChanged}
      />
    </>
  );
}

function CreateProjectModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/projects", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      onCreated();
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });
  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    setErr("");
    const fd = new FormData(ev.currentTarget);
    const skills = String(fd.get("required_skills") || "")
      .split(/[,,\n]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((name) => ({ name }));
    mut.mutate({
      name: String(fd.get("name") || "").trim(),
      description: String(fd.get("description") || "").trim() || null,
      status: "draft",
      required_skills: skills,
    });
  }
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="新建项目"
      width={480}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="create-proj" disabled={mut.isPending}>
            创建
          </Button>
        </>
      }
    >
      <form id="create-proj" onSubmit={submit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-[13px] font-body">
          项目名
          <Input name="name" required placeholder="如:AI 平台重构" />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          描述
          <textarea
            name="description"
            rows={3}
            className="rounded-lg border border-[var(--color-border-subtle)] px-3 py-2 text-[13px] font-body"
          />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          所需技能(逗号分隔,供 AI 选人匹配)
          <Input name="required_skills" placeholder="如:TypeScript, 系统设计, AI应用" />
        </label>
        {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
      </form>
    </Modal>
  );
}

function AddMemberModal({
  project,
  open,
  onClose,
  onChanged,
}: {
  project: Project;
  open: boolean;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [empId, setEmpId] = useState("");
  const [err, setErr] = useState("");
  const { data: emps } = useQuery<{ items: EmpOpt[] }>({
    queryKey: ["emp-options-all"],
    queryFn: () => fetchJSON("/api/employees?limit=200"),
    enabled: open,
  });
  const options = (emps?.items ?? [])
    .filter((e) => !(project.members ?? []).some((m) => m.employee_id === e.id && !m.left_at))
    .map((e) => ({ value: e.id, label: e.name }));
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON(`/api/projects/${project.id}/members`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      onChanged();
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "添加失败"),
  });
  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    if (!empId) {
      setErr("请选择员工");
      return;
    }
    const fd = new FormData(ev.currentTarget);
    mut.mutate({
      employee_id: empId,
      role: String(fd.get("role") || "").trim() || null,
      allocation_pct: Number(fd.get("allocation_pct") || 100),
    });
  }
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="添加成员"
      width={420}
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="add-member" disabled={mut.isPending}>
            添加
          </Button>
        </>
      }
    >
      <form id="add-member" onSubmit={submit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-[13px] font-body">
          员工
          <Select value={empId} onChange={setEmpId} options={options} placeholder="选择员工" />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          角色
          <Input name="role" placeholder="如:后端 / TL / 顾问" />
        </label>
        <label className="flex flex-col gap-1 text-[13px] font-body">
          投入度 %
          <Input name="allocation_pct" type="number" min={1} max={100} defaultValue={100} />
        </label>
        {err && <div className="text-[13px] text-[var(--color-danger)]">{err}</div>}
      </form>
    </Modal>
  );
}
