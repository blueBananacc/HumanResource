import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Shuffle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { fetchJSON, ApiError } from "@/lib/api";
import {
  LIFECYCLE_KIND_LABEL,
  StatusPill,
  EMPLOYMENT_TYPE_LABEL,
  type MeEvent,
  type MeProfile,
} from "@/lib/hris";
import { MyRatingTasks, MySkillsCard } from "@/components/people/SkillRatings";
import { Field } from "../hris/Employees";

export function MyProfile() {
  const [editOpen, setEditOpen] = useState(false);
  const { data: p, isLoading } = useQuery<MeProfile>({
    queryKey: ["me-profile"],
    queryFn: () => fetchJSON("/api/me/profile"),
  });
  const { data: events } = useQuery<MeEvent[]>({
    queryKey: ["me-lifecycle"],
    queryFn: () => fetchJSON("/api/me/lifecycle"),
  });

  if (isLoading || !p) {
    return (
      <main className="p-8 max-w-[800px] mx-auto w-full">
        <div className="text-sm text-[var(--color-text-tertiary)] font-body">
          加载中…
        </div>
      </main>
    );
  }

  const personal = (p.personal ?? {}) as Record<string, string>;
  const myEvents = (events ?? []).filter((e) => e.status === "in_progress");

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[800px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] flex items-center justify-center font-heading font-semibold text-xl shrink-0">
            {p.name[0]}
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-3">
              <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
                {p.name}
              </h2>
              <StatusPill status={p.status} />
            </div>
            <div className="text-sm text-[var(--color-text-secondary)] font-body">
              {p.designation || "—"} · {p.org_unit_name || "未分配部门"}
            </div>
          </div>
        </div>
        <Button variant="secondary" onClick={() => setEditOpen(true)}>
          <Pencil className="w-3.5 h-3.5" />
          更新联系方式
        </Button>
      </div>

      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-4">
        <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
          基本信息
        </h3>
        <InfoRow label="工号" value={p.employee_no} mono />
        <InfoRow label="工作邮箱" value={p.work_email} />
        <InfoRow label="手机号" value={p.phone} />
        <InfoRow label="入职日期" value={p.hire_date} />
        <InfoRow
          label="用工类型"
          value={EMPLOYMENT_TYPE_LABEL[p.employment_type] ?? p.employment_type}
        />
        <InfoRow label="汇报对象" value={p.reports_to_name} />
        {Object.entries(personal).map(([k, v]) => (
          <InfoRow key={k} label={k} value={String(v)} />
        ))}
      </section>

      {myEvents.length > 0 && (
        <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6 flex flex-col gap-3">
          <h3 className="font-heading font-semibold text-[15px] text-[var(--color-text-primary)]">
            进行中的流程
          </h3>
          {myEvents.map((ev) => {
            const done = ev.tasks.filter((t) => t.done).length;
            return (
              <div key={ev.id} className="flex flex-col gap-2">
                <div className="flex items-center gap-3">
                  <Shuffle className="w-4 h-4 text-[var(--color-accent)]" />
                  <span className="font-medium text-[13px] font-body text-[var(--color-text-primary)]">
                    {LIFECYCLE_KIND_LABEL[ev.kind] ?? ev.kind}
                  </span>
                  <span className="text-[12px] font-mono text-[var(--color-text-tertiary)]">
                    {done}/{ev.tasks.length}
                  </span>
                </div>
                <div className="ml-7 flex flex-col gap-1">
                  {ev.tasks.map((t) => (
                    <div
                      key={t.id}
                      className={`text-[12px] font-body ${
                        t.done
                          ? "line-through text-[var(--color-text-tertiary)]"
                          : "text-[var(--color-text-primary)]"
                      }`}
                    >
                      {t.title}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </section>
      )}

      <MyRatingTasks />
      <MySkillsCard />

      <EditModal p={p} open={editOpen} onClose={() => setEditOpen(false)} />
    </main>
  );
}

function InfoRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-3 text-[13px] font-body">
      <span className="w-24 shrink-0 text-[var(--color-text-tertiary)]">
        {label}
      </span>
      <span
        className={`text-[var(--color-text-primary)] ${mono ? "font-mono" : ""}`}
      >
        {value || "—"}
      </span>
    </div>
  );
}

function EditModal({
  p,
  open,
  onClose,
}: {
  p: MeProfile;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [err, setErr] = useState("");
  const mut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      fetchJSON("/api/me/profile", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["me-profile"] });
      onClose();
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  function submit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    const personal = { ...(p.personal ?? {}) };
    const addr = String(fd.get("address") || "").trim();
    const emergency = String(fd.get("emergency_contact") || "").trim();
    if (addr) personal["住址"] = addr;
    else delete personal["住址"];
    if (emergency) personal["紧急联系人"] = emergency;
    else delete personal["紧急联系人"];
    const birth = String(fd.get("birth_date") || "").trim();
    if (birth) personal["birth_date"] = birth;
    else delete personal["birth_date"];
    mut.mutate({
      phone: String(fd.get("phone") || "").trim() || null,
      personal,
    });
  }

  const personal = (p.personal ?? {}) as Record<string, string>;

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="更新联系方式"
      description="工作信息由 HR 维护;你可以更新手机号与个人资料。"
      footer={
        <>
          <Button variant="secondary" onClick={onClose}>
            取消
          </Button>
          <Button type="submit" form="me-edit" disabled={mut.isPending}>
            {mut.isPending ? "保存中…" : "保存"}
          </Button>
        </>
      }
    >
      <form id="me-edit" onSubmit={submit} className="flex flex-col gap-4">
        <Field label="手机号">
          <Input name="phone" defaultValue={p.phone ?? ""} />
        </Field>
        <Field label="住址">
          <Input name="address" defaultValue={personal["住址"] ?? ""} />
        </Field>
        <Field label="紧急联系人">
          <Input
            name="emergency_contact"
            defaultValue={personal["紧急联系人"] ?? ""}
            placeholder="姓名 电话"
          />
        </Field>
        <Field label="生日">
          <Input
            name="birth_date"
            type="date"
            defaultValue={personal["birth_date"] ?? ""}
          />
        </Field>
        {err && (
          <div className="text-[13px] text-[var(--color-danger)] font-body">
            {err}
          </div>
        )}
      </form>
    </Modal>
  );
}

