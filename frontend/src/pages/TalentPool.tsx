import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Ban,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Folder,
  MailCheck,
  Play,
  Plus,
  ScanSearch,
  Sparkles,
  Trash2,
  Upload,
  Users,
  X,
  Zap,
} from "lucide-react";
import {
  ResumeUpload,
  type ResumeUploadHandle,
} from "@/components/resume/ResumeUpload";
import { ImportDrawer } from "@/components/records/ImportDrawer";
import { SavedFilters } from "@/components/records/SavedFilters";
import { useDeleteCandidate } from "@/components/resume/shared";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { Empty } from "@/components/ui/empty";
import { Pagination } from "@/components/ui/pagination";
import { SearchInput } from "@/components/ui/search-input";
import { Select } from "@/components/ui/select";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { usePagedQuery, PAGE_SIZE } from "@/lib/usePagedQuery";

type TalentItem = {
  id: string;
  name: string;
  display_email: string | null;
  display_phone: string | null;
  tags: string[];
  status: string;
  is_blacklisted: boolean;
  blacklist_reason: string | null;
  last_active_at: string | null;
  created_at: string;
  resume_count: number;
  interview_count: number;
  last_interview_status: string | null;
  last_interview_at: string | null;
  top_match_job_title: string | null;
  top_match_score: number | null;
};

type RuleCondition = { field: string; op: string; value: unknown };

type Group = {
  id: string;
  name: string;
  description: string | null;
  member_count: number;
  auto_rule: { mode: string; conditions: RuleCondition[] } | null;
  auto_count: number;
  created_at: string;
  updated_at: string;
};

type RuleField = { field: string; label: string; ops: string[]; value: string };
type RuleFields = {
  modes: { value: string; label: string }[];
  fields: RuleField[];
  stages: string[];
};

const STATUS_LABEL: Record<string, string> = {
  in_progress: "进行中",
  done: "已完成",
  abandoned: "已放弃",
};

export function TalentPool() {
  const [blacklistOnly, setBlacklistOnly] = useState<string>("");
  const [groupFilter, setGroupFilter] = useState<string>("");
  const [openGroups, setOpenGroups] = useState(false);
  const [openImport, setOpenImport] = useState(false);
  const uploadRef = useRef<ResumeUploadHandle>(null);

  const params = {
    blacklisted: blacklistOnly || undefined,
    group_id: groupFilter || undefined,
  };

  const { data, isLoading, page, pageCount, total, goto, reset, q, setQ } =
    usePagedQuery<TalentItem>({
      key: ["talents"],
      url: "/api/talents",
      params,
    });

  const [inputQ, setInputQ] = useState(q);
  useEffect(() => {
    setInputQ(q);
  }, [q]);

  useEffect(() => {
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [blacklistOnly, groupFilter]);

  const { data: groups } = useQuery({
    queryKey: ["talent-groups"],
    queryFn: () => fetchJSON<Group[]>("/api/talent-groups"),
  });

  const { deleteCandidate, deletingId } = useDeleteCandidate();

  const items = data?.items ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            人才库
          </h2>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            以候选人为主视角的人才档案。聚合简历版本、面试历史、岗位匹配,并支持标签、分组、黑名单与备注。
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Link to="/campaigns">
            <Button variant="secondary">
              <MailCheck className="w-4 h-4" />
              人才再激活
            </Button>
          </Link>
          <Button onClick={() => uploadRef.current?.open()}>
            <Upload className="w-4 h-4" />
            上传简历
          </Button>
        </div>
      </div>

      <ResumeUpload ref={uploadRef} />
      <ImportDrawer
        endpoint="/api/talents/import"
        open={openImport}
        onClose={() => setOpenImport(false)}
        columnsHint="name(必填),email,phone,tags(分号分隔),org_name"
        onDone={() => reset()}
      />

      <section className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        <div className="px-6 py-4 border-b border-[var(--color-border-subtle)] flex flex-wrap items-center gap-3">
          <SearchInput
            value={inputQ}
            onChange={(v) => {
              setInputQ(v);
              setQ(v);
            }}
            placeholder="搜索姓名 / 邮箱 / 手机"
            className="w-[280px]"
          />
          <Select
            value={blacklistOnly}
            onChange={setBlacklistOnly}
            placeholder="是否黑名单"
            className="w-[140px]"
            options={[
              { value: "", label: "全部候选人" },
              { value: "true", label: "仅黑名单" },
              { value: "false", label: "排除黑名单" },
            ]}
          />
          <Select
            value={groupFilter}
            onChange={setGroupFilter}
            placeholder="所属分组"
            className="w-[200px]"
            options={[
              { value: "", label: "全部分组" },
              ...(groups ?? []).map((g) => ({
                value: g.id,
                label: g.name,
                hint: `${g.member_count} 人`,
              })),
            ]}
          />
          <Button variant="secondary" onClick={() => setOpenGroups(true)}>
            <Folder className="w-4 h-4" />
            人才分组
          </Button>
          <Button variant="secondary" onClick={() => setOpenImport(true)}>
            <Upload className="w-4 h-4" />
            批量导入
          </Button>
          <Link to="/settings/imports">
            <Button variant="ghost">
              <Sparkles className="w-4 h-4" />
              智能导入
            </Button>
          </Link>
          <SavedFilters
            page="talents"
            currentParams={() => ({
              q: inputQ || undefined,
              blacklisted: blacklistOnly || undefined,
              group_id: groupFilter || undefined,
            })}
            onApply={(p) => {
              setInputQ(typeof p.q === "string" ? p.q : "");
              setQ(typeof p.q === "string" ? p.q : "");
              setBlacklistOnly(typeof p.blacklisted === "string" ? p.blacklisted : "");
              setGroupFilter(typeof p.group_id === "string" ? p.group_id : "");
            }}
          />
          {(blacklistOnly || groupFilter || q) && (
            <button
              type="button"
              onClick={() => {
                setBlacklistOnly("");
                setGroupFilter("");
                setQ("");
                setInputQ("");
              }}
              className="inline-flex items-center gap-1 text-[12px] text-[var(--color-text-secondary)] font-body hover:text-[var(--color-text-primary)]"
            >
              <X className="w-3.5 h-3.5" />
              清空筛选
            </button>
          )}
          <span className="ml-auto text-[12px] text-[var(--color-text-tertiary)] font-body">
            共 {total} 人
          </span>
        </div>

        {isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty
            icon={Users}
            title="还没有候选人"
            description="上传第一份简历后,系统会自动解析并按候选人沉淀到这里。"
          />
        ) : (
          <>
            <table className="w-full text-[13px] font-body">
              <thead className="text-[11px] uppercase tracking-wider text-[var(--color-text-tertiary)] bg-[var(--color-bg-canvas)]">
                <tr>
                  <th className="text-left px-6 py-2.5 font-medium">候选人</th>
                  <th className="text-left px-4 py-2.5 font-medium">标签</th>
                  <th className="text-left px-4 py-2.5 font-medium">最近活动</th>
                  <th className="text-left px-4 py-2.5 font-medium">简历 / 面试</th>
                  <th className="text-left px-4 py-2.5 font-medium">最高匹配</th>
                  <th className="text-left px-4 py-2.5 font-medium">状态</th>
                  <th className="w-12 px-4 py-2.5 font-medium" />
                </tr>
              </thead>
              <tbody>
                {items.map((t) => (
                  <tr
                    key={t.id}
                    className="group border-t border-[var(--color-border-row)] hover:bg-[var(--color-bg-muted)]"
                  >
                    <td className="px-6 py-3">
                      <Link
                        to={`/talents/${t.id}`}
                        className="flex flex-col gap-0.5"
                      >
                        <span className="font-medium text-[var(--color-text-primary)]">
                          {t.name}
                        </span>
                        <span className="text-[11px] text-[var(--color-text-tertiary)]">
                          {t.display_email ?? "—"}
                          {t.display_phone ? ` · ${t.display_phone}` : ""}
                        </span>
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {t.tags.length === 0 && (
                          <span className="text-[11px] text-[var(--color-text-tertiary)]">
                            —
                          </span>
                        )}
                        {t.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="px-1.5 py-0.5 rounded text-[11px] bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
                          >
                            {tag}
                          </span>
                        ))}
                        {t.tags.length > 3 && (
                          <span className="text-[11px] text-[var(--color-text-tertiary)]">
                            +{t.tags.length - 3}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-secondary)] whitespace-nowrap">
                      {t.last_active_at ? formatRelative(t.last_active_at) : "—"}
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-secondary)] whitespace-nowrap">
                      {t.resume_count} 份简历 / {t.interview_count} 次面试
                      {t.last_interview_status && (
                        <span className="ml-1 text-[11px] text-[var(--color-text-tertiary)]">
                          (最近{STATUS_LABEL[t.last_interview_status] ?? t.last_interview_status})
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                      {t.top_match_job_title ? (
                        <>
                          <span className="font-mono text-[var(--color-text-primary)]">
                            {t.top_match_score}
                          </span>
                          <span className="ml-1 text-[11px] text-[var(--color-text-tertiary)]">
                            · {t.top_match_job_title}
                          </span>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {t.is_blacklisted ? (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-[var(--color-danger-soft)] text-[var(--color-danger)]">
                          <Ban className="w-3 h-3" />
                          黑名单
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-[var(--color-success-soft)] text-[var(--color-success)]">
                          <CheckCircle2 className="w-3 h-3" />
                          活跃
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteCandidate({
                            candidateId: t.id,
                            name: t.name,
                          });
                        }}
                        disabled={deletingId === t.id}
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-1"
                        aria-label="删除候选人"
                        title="删除候选人"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="border-t border-[var(--color-border-subtle)]">
              <Pagination
                page={page}
                pageCount={pageCount}
                total={total}
                pageSize={PAGE_SIZE}
                onChange={goto}
              />
            </div>
          </>
        )}
      </section>

      <GroupsDrawer open={openGroups} onClose={() => setOpenGroups(false)} />
    </main>
  );
}

function GroupsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { data: groups, isLoading } = useQuery({
    queryKey: ["talent-groups"],
    queryFn: () => fetchJSON<Group[]>("/api/talent-groups"),
  });

  const create = useMutation({
    mutationFn: () =>
      fetchJSON<Group>("/api/talent-groups", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), description: description || null }),
      }),
    onSuccess: () => {
      setName("");
      setDescription("");
      setError(null);
      qc.invalidateQueries({ queryKey: ["talent-groups"] });
    },
    onError: (e: unknown) =>
      setError(
        e instanceof ApiError
          ? e.status === 409
            ? "同名分组已存在"
            : e.message
          : "创建失败",
      ),
  });

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/talent-groups/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["talent-groups"] }),
  });

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="人才分组"
      description="为常用的人才池建立分组，候选人详情页可批量加入或移出。"
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2 p-3 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)]">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)]">
            新建分组
          </span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="例如: 2026 校招储备池"
            className="px-3 py-2 rounded-md bg-white border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-text-primary)]"
          />
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="简短描述(可选)"
            className="px-3 py-2 rounded-md bg-white border border-[var(--color-border-subtle)] text-[12px] font-body focus:outline-none focus:border-[var(--color-text-primary)]"
          />
          {error && (
            <span className="text-[12px] text-[var(--color-danger)]">{error}</span>
          )}
          <Button
            onClick={() => create.mutate()}
            disabled={!name.trim() || create.isPending}
          >
            <Plus className="w-4 h-4" />
            创建
          </Button>
        </div>

        {isLoading ? (
          <div className="text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : !groups || groups.length === 0 ? (
          <div className="text-[12px] text-[var(--color-text-tertiary)] font-body">
            还没有分组,创建第一个吧。
          </div>
        ) : (
          <ul className="flex flex-col gap-2">
            {groups.map((g) => (
              <GroupRow
                key={g.id}
                group={g}
                onDelete={async () => {
                  if (
                    await confirm({
                      title: "删除分组?",
                      description: `将删除「${g.name}」及其全部成员关联(候选人本身不受影响)。`,
                      tone: "danger",
                      confirmLabel: "删除",
                    })
                  ) {
                    del.mutate(g.id);
                  }
                }}
              />
            ))}
          </ul>
        )}
      </div>
    </Drawer>
  );
}


// ---------- 自动入库规则 ----------

const OP_LABEL: Record<string, string> = {
  contains: "包含", not_contains: "不包含",
  contains_any: "命中任一", contains_all: "全部包含",
  in: "属于", not_in: "不属于", eq: "等于",
};

const STAGE_LABEL: Record<string, string> = {
  applied: "已投递", screening: "筛选中", interview: "面试中",
  offer: "Offer", hired: "已入职", rejected: "已淘汰", withdrawn: "已放弃",
};

function ruleSummary(rule: Group["auto_rule"], fields: RuleField[] | undefined): string {
  if (!rule?.conditions?.length) return "";
  const label = (f: string) => fields?.find((x) => x.field === f)?.label ?? f;
  const parts = rule.conditions.map((c) => {
    const v = Array.isArray(c.value)
      ? c.value.map((x) => (c.field === "stage" ? (STAGE_LABEL[x] ?? x) : String(x))).join("/")
      : String(c.value ?? "");
    return `${label(c.field)}${OP_LABEL[c.op] ?? c.op}「${v}」`;
  });
  return `${rule.mode === "any" ? "任一" : "全部"}:${parts.join(";")}`;
}

function GroupRow({ group, onDelete }: { group: Group; onDelete: () => void }) {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [mode, setMode] = useState(group.auto_rule?.mode ?? "all");
  const [conds, setConds] = useState<RuleCondition[]>(
    group.auto_rule?.conditions?.length
      ? group.auto_rule.conditions.map((c) => ({ ...c }))
      : [{ field: "tag", op: "contains_any", value: [""] }],
  );
  const [msg, setMsg] = useState("");
  const [preview, setPreview] = useState<{ total: number; sample: { name: string; reasons: string[] }[] } | null>(null);

  const { data: fieldsMeta } = useQuery({
    queryKey: ["talent-rule-fields"],
    queryFn: () => fetchJSON<RuleFields>("/api/talent-groups/rule-fields"),
    staleTime: 300_000,
  });
  const fields = fieldsMeta?.fields ?? [];
  const spec = (f: string) => fields.find((x) => x.field === f);

  const buildRule = () => ({
    mode,
    conditions: conds
      .map((c) => {
        let v = c.value;
        if (Array.isArray(v)) v = v.map((x) => String(x).trim()).filter(Boolean);
        else v = String(v ?? "").trim();
        return { ...c, value: v };
      })
      .filter((c) => (Array.isArray(c.value) ? c.value.length > 0 : c.value !== "")),
  });

  const saveMut = useMutation({
    mutationFn: () =>
      fetchJSON<Group>(`/api/talent-groups/${group.id}/rule`, {
        method: "PUT",
        body: JSON.stringify({ auto_rule: buildRule() }),
      }),
    onSuccess: () => {
      setMsg("规则已保存,新候选人将自动入库");
      setPreview(null);
      qc.invalidateQueries({ queryKey: ["talent-groups"] });
    },
    onError: (e) => setMsg(e instanceof ApiError ? e.message : "保存失败"),
  });
  const clearMut = useMutation({
    mutationFn: () =>
      fetchJSON<Group>(`/api/talent-groups/${group.id}/rule`, {
        method: "PUT",
        body: JSON.stringify({ auto_rule: null }),
      }),
    onSuccess: () => {
      setMsg("已清除规则,回到手动分组");
      setPreview(null);
      qc.invalidateQueries({ queryKey: ["talent-groups"] });
    },
    onError: (e) => setMsg(e instanceof ApiError ? e.message : "清除失败"),
  });
  const previewMut = useMutation({
    mutationFn: () =>
      fetchJSON<{ total: number; scanned: number; sample: { name: string; reasons: string[] }[] }>(
        "/api/talent-groups/rule-preview",
        { method: "POST", body: JSON.stringify({ auto_rule: buildRule() }) },
      ),
    onSuccess: setPreview,
    onError: (e) => {
      setPreview(null);
      setMsg(e instanceof ApiError ? e.message : "预览失败");
    },
  });
  const rescanMut = useMutation({
    mutationFn: () =>
      fetchJSON<{ scanned: number; added: { name: string }[] }>(
        `/api/talent-groups/${group.id}/rescan`,
        { method: "POST" },
      ),
    onSuccess: (d) => {
      setMsg(`补扫完成:扫描 ${d.scanned} 人,新入库 ${d.added.length} 人`);
      qc.invalidateQueries({ queryKey: ["talent-groups"] });
      qc.invalidateQueries({ queryKey: ["talents"] });
    },
    onError: (e) => setMsg(e instanceof ApiError ? e.message : "补扫失败"),
  });

  const setCond = (i: number, patch: Partial<RuleCondition>) =>
    setConds((cs) => cs.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  const valueInput = (c: RuleCondition, i: number) => {
    const s = spec(c.field);
    if (!s) return null;
    if (s.value === "str[]" || s.value === "stage[]") {
      const arr = Array.isArray(c.value) ? (c.value as string[]) : [];
      return (
        <input
          value={arr.join(",")}
          onChange={(e) => setCond(i, { value: e.target.value.split(",").map((x) => x.trim()) })}
          placeholder={c.field === "stage" ? `阶段如:${fieldsMeta?.stages.slice(0, 3).join(",")}…` : "逗号分隔多个值"}
          className="flex-1 h-8 px-2 rounded-md border border-[var(--color-border-subtle)] text-[12px] outline-none focus:border-[var(--color-text-primary)]"
        />
      );
    }
    if (s.value === "status") {
      return (
        <Select
          value={String(c.value ?? "active")}
          onChange={(v) => setCond(i, { value: v })}
          options={[
            { value: "active", label: "活跃" },
            { value: "archived", label: "已归档" },
          ]}
          className="flex-1"
        />
      );
    }
    if (s.value === "bool") {
      return (
        <Select
          value={String(c.value ?? true)}
          onChange={(v) => setCond(i, { value: v === "true" })}
          options={[
            { value: "true", label: "是" },
            { value: "false", label: "否" },
          ]}
          className="flex-1"
        />
      );
    }
    return (
      <input
        value={String(c.value ?? "")}
        onChange={(e) => setCond(i, { value: e.target.value })}
        placeholder={s.value === "job_id" ? "岗位 ID" : "匹配文本"}
        className="flex-1 h-8 px-2 rounded-md border border-[var(--color-border-subtle)] text-[12px] outline-none focus:border-[var(--color-text-primary)]"
      />
    );
  };

  return (
    <li className="flex flex-col gap-2 p-3 rounded-lg border border-[var(--color-border-subtle)]">
      <div className="flex items-start gap-2">
        <div className="flex-1 flex flex-col gap-0.5 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-medium text-sm text-[var(--color-text-primary)]">{group.name}</span>
            {group.auto_rule && (
              <span className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded font-bold bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
                <Zap className="w-3 h-3" /> 自动
              </span>
            )}
          </div>
          <span className="text-[11px] text-[var(--color-text-tertiary)]">
            {group.member_count} 人
            {group.auto_count > 0 && ` (含自动入库 ${group.auto_count})`}
            {" · 创建于 "}{formatRelative(group.created_at)}
          </span>
          {group.auto_rule && (
            <span className="text-[11px] text-[var(--color-text-secondary)] line-clamp-2" title={ruleSummary(group.auto_rule, fields)}>
              {ruleSummary(group.auto_rule, fields)}
            </span>
          )}
          {group.description && (
            <span className="text-[12px] text-[var(--color-text-secondary)]">{group.description}</span>
          )}
        </div>
        <button
          type="button"
          title={expanded ? "收起规则" : "自动入库规则"}
          onClick={() => setExpanded((v) => !v)}
          className="text-[var(--color-text-tertiary)] hover:text-[var(--color-accent)] p-1.5 rounded"
        >
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
        <button
          type="button"
          title="删除分组"
          onClick={onDelete}
          className="text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] p-1.5 rounded"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {expanded && (
        <div className="flex flex-col gap-2.5 pt-2 border-t border-[var(--color-border-row)]">
          <div className="flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-[var(--color-accent)]" />
            <span className="text-[12px] font-bold text-[var(--color-text-secondary)]">
              自动入库规则
            </span>
            <Select
              value={mode}
              onChange={setMode}
              options={fieldsMeta?.modes ?? [
                { value: "all", label: "全部满足" },
                { value: "any", label: "任一满足" },
              ]}
              className="w-28 ml-auto"
            />
          </div>

          {conds.map((c, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <Select
                value={c.field}
                onChange={(f) => {
                  const s = spec(f);
                  setCond(i, {
                    field: f,
                    op: s?.ops[0] ?? "contains",
                    value: s?.value.endsWith("[]") ? [] : s?.value === "bool" ? true : "",
                  });
                }}
                options={fields.map((f) => ({ value: f.field, label: f.label }))}
                className="w-24"
              />
              <Select
                value={c.op}
                onChange={(op) => setCond(i, { op })}
                options={(spec(c.field)?.ops ?? []).map((o) => ({ value: o, label: OP_LABEL[o] ?? o }))}
                className="w-24"
              />
              {valueInput(c, i)}
              {conds.length > 1 && (
                <button
                  type="button"
                  onClick={() => setConds((cs) => cs.filter((_, j) => j !== i))}
                  className="p-1 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                  aria-label="删除条件"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}

          {conds.length < 10 && (
            <button
              type="button"
              onClick={() => setConds((cs) => [...cs, { field: "tag", op: "contains_any", value: [] }])}
              className="self-start flex items-center gap-1 text-[11px] font-bold text-[var(--color-accent)] hover:underline"
            >
              <Plus className="w-3 h-3" /> 加条件
            </button>
          )}

          {preview && (
            <div className="rounded-md bg-[var(--color-bg-canvas)] border border-[var(--color-border-subtle)] px-2.5 py-2 text-[11px] text-[var(--color-text-secondary)]">
              预计命中 <b className="text-[var(--color-text-primary)]">{preview.total}</b> 人
              {preview.sample.slice(0, 5).map((s) => (
                <div key={s.name} className="truncate">
                  · {s.name} <span className="text-[var(--color-text-tertiary)]">{s.reasons[0]}</span>
                </div>
              ))}
            </div>
          )}

          {msg && <span className="text-[11px] text-[var(--color-text-secondary)]">{msg}</span>}

          <div className="flex items-center gap-2 flex-wrap">
            <Button
              size="sm"
              onClick={() => saveMut.mutate()}
              disabled={saveMut.isPending}
            >
              {saveMut.isPending ? "保存中…" : "保存规则"}
            </Button>
            <Button
              size="sm" variant="secondary"
              onClick={() => previewMut.mutate()}
              disabled={previewMut.isPending}
            >
              <ScanSearch className="w-3.5 h-3.5" />
              预览命中
            </Button>
            {group.auto_rule && (
              <>
                <Button
                  size="sm" variant="secondary"
                  onClick={() => rescanMut.mutate()}
                  disabled={rescanMut.isPending}
                >
                  <Play className="w-3.5 h-3.5" />
                  {rescanMut.isPending ? "补扫中…" : "立即补扫"}
                </Button>
                <button
                  type="button"
                  onClick={() => clearMut.mutate()}
                  className="text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)]"
                >
                  清除规则
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </li>
  );
}
