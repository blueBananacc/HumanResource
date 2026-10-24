import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Eye,
  History,
  Pencil,
  RotateCcw,
  Save,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { Empty } from "@/components/ui/empty";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

/**
 * 设置 → AI 提示词(T1-2)。
 *
 * 列表 = 注册表里所有用途,每条显示「默认 / 自定义 vN」;
 * 抽屉里编辑 content(支持 {{var}} 占位)+ 变量文档 + 渲染预览 +
 * 版本回滚 + 恢复默认。
 */

type VarSpec = { name: string; desc: string };

type Template = {
  key: string;
  name: string;
  description: string;
  module: string;
  variables: VarSpec[];
  content: string;
  is_custom: boolean;
  scoped_count: number;
  version: number | null;
  updated_at: string | null;
  updated_by: string | null;
};

const MODULE_LABEL: Record<string, string> = {
  global: "通用",
  talents: "人才库",
  jobs: "岗位",
  interviews: "面试",
  hris: "员工",
  cases: "工单",
  announcements: "公告",
  campaigns: "外联",
};
const MODULE_ORDER = [
  "global", "talents", "jobs", "interviews", "hris", "cases", "announcements", "campaigns",
];

type Version = { id: string; version: number; content: string; created_at: string };

export function PromptSettings() {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<Template | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  const listQuery = useQuery({
    queryKey: ["prompt-templates"],
    queryFn: () => fetchJSON<Template[]>("/api/prompt-templates"),
  });
  const items = listQuery.data ?? [];
  const groups = MODULE_ORDER.map((m) => ({
    module: m,
    items: items.filter((t) => (t.module || "global") === m),
  })).filter((g) => g.items.length > 0);
  const extra = items.filter(
    (t) => !MODULE_ORDER.includes(t.module || "global"),
  );
  if (extra.length) groups.push({ module: "other", items: extra });

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          AI 提示词
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          按用途管理各 AI 能力的系统提示词 · 支持 {"{{变量}}"} 占位、版本回滚与按模块覆盖
        </p>
      </div>

      {feedback && (
        <div className="text-[13px] bg-[var(--color-info-soft)] text-[var(--color-info)] px-3 py-2 rounded-md font-body">
          {feedback}
        </div>
      )}

      <div className="bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] overflow-hidden">
        {listQuery.isLoading ? (
          <div className="p-12 text-center text-sm text-[var(--color-text-tertiary)] font-body">
            加载中…
          </div>
        ) : items.length === 0 ? (
          <Empty icon={Sparkles} title="没有可管理的提示词" />
        ) : (
          groups.map((g) => (
            <div key={g.module}>
              <div className="px-6 py-2.5 bg-[var(--color-bg-canvas)] border-b border-t border-[var(--color-border-row)] first:border-t-0">
                <span className="text-[12px] font-bold text-[var(--color-text-tertiary)] tracking-[0.06em]">
                  {MODULE_LABEL[g.module] ?? g.module}
                </span>
              </div>
              {g.items.map((t, i) => (
                <div
                  key={t.key}
                  className={`flex items-center gap-4 px-6 py-4 ${
                    i < g.items.length - 1
                      ? "border-b border-[var(--color-border-row)]"
                      : ""
                  }`}
                >
                  <div className="flex flex-col gap-1 flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm text-[var(--color-text-primary)] font-body">
                        {t.name}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-md text-[11px] font-body ${
                          t.is_custom
                            ? "bg-[var(--color-accent-soft,rgba(238,22,79,0.08))] text-[var(--color-accent)]"
                            : "bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
                        }`}
                      >
                        {t.is_custom ? `自定义 v${t.version}` : "默认"}
                      </span>
                      {t.scoped_count > 0 && (
                        <span className="px-2 py-0.5 rounded-md text-[11px] font-body bg-[var(--color-info-soft)] text-[var(--color-info)]">
                          {t.scoped_count} 个模块覆盖
                        </span>
                      )}
                    </div>
                    <p className="text-[12px] text-[var(--color-text-secondary)] font-body truncate">
                      {t.description}
                    </p>
                    {t.is_custom && t.updated_at && (
                      <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                        更新于 {formatRelative(t.updated_at)}
                      </span>
                    )}
                  </div>
                  <code className="text-[11px] text-[var(--color-text-tertiary)] font-mono bg-[var(--color-bg-subtle)] px-2 py-1 rounded hidden md:block">
                    {t.key}
                  </code>
                  <Button
                    variant="secondary"
                    className="!px-3 !py-1.5 !text-[12px] shrink-0"
                    onClick={() => setEditing(t)}
                  >
                    <Pencil className="w-3.5 h-3.5" />
                    编辑
                  </Button>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {editing && (
        <TemplateDrawer
          tpl={editing}
          onClose={() => setEditing(null)}
          onSaved={(msg) => {
            setEditing(null);
            setFeedback(msg);
            qc.invalidateQueries({ queryKey: ["prompt-templates"] });
          }}
        />
      )}
    </main>
  );
}

function TemplateDrawer({
  tpl,
  onClose,
  onSaved,
}: {
  tpl: Template;
  onClose: () => void;
  onSaved: (msg: string) => void;
}) {
  const [content, setContent] = useState(tpl.content);
  // 空 = 全局覆盖;非空 = 只在该模块语境生效(两级回落)
  const [scope, setScope] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [showVersions, setShowVersions] = useState(false);
  const [previewVersion, setPreviewVersion] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scopeQ = scope.trim() ? `?scope_module=${encodeURIComponent(scope.trim())}` : "";

  const versionsQuery = useQuery({
    queryKey: ["prompt-versions", tpl.key, scope],
    queryFn: () =>
      fetchJSON<Version[]>(`/api/prompt-templates/${tpl.key}/versions${scopeQ}`),
    enabled: showVersions,
  });

  const save = useMutation({
    mutationFn: () =>
      fetchJSON<Template>(`/api/prompt-templates/${tpl.key}`, {
        method: "PUT",
        body: JSON.stringify({ content, scope_module: scope.trim() }),
      }),
    onSuccess: (t) =>
      onSaved(scope.trim() ? `已保存 ${scope.trim()} 模块覆盖 v${t.version}` : `已保存为 v${t.version}`),
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  const previewMut = useMutation({
    mutationFn: () =>
      fetchJSON<{ rendered: string }>(
        `/api/prompt-templates/${tpl.key}/preview`,
        {
          method: "POST",
          body: JSON.stringify({ content, vars: {}, scope_module: scope.trim() }),
        },
      ),
    onSuccess: (r) => setPreview(r.rendered),
  });

  const reset = useMutation({
    mutationFn: () =>
      fetchJSON<Template>(`/api/prompt-templates/${tpl.key}/reset${scopeQ}`, {
        method: "POST",
      }),
    onSuccess: () => onSaved(scope.trim() ? "已删除该模块覆盖" : "已恢复默认提示词"),
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "恢复失败"),
  });

  const rollback = useMutation({
    mutationFn: (version: number) =>
      fetchJSON<Template>(`/api/prompt-templates/${tpl.key}/rollback`, {
        method: "POST",
        body: JSON.stringify({ version, scope_module: scope.trim() }),
      }),
    onSuccess: (t) => onSaved(`已回滚,当前 v${t.version}`),
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "回滚失败"),
  });

  return (
    <Drawer
      open
      onClose={onClose}
      title={tpl.name}
      description={`${tpl.key} · ${tpl.description}`}
      width={680}
    >
      <div className="flex flex-col gap-4">
        {tpl.variables.length > 0 && (
          <div className="bg-[var(--color-bg-subtle)] rounded-lg p-3 flex flex-col gap-1.5">
            <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
              可用变量(双花括号引用,如 {"{{job_title}}"})
            </span>
            <div className="flex flex-wrap gap-1.5">
              {tpl.variables.map((v) => (
                <span
                  key={v.name}
                  title={v.desc}
                  className="px-2 py-0.5 rounded bg-white border border-[var(--color-border-subtle)] text-[11px] font-mono text-[var(--color-text-secondary)]"
                >
                  {`{{${v.name}}}`} · {v.desc}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-2">
          <span className="text-[12px] text-[var(--color-text-secondary)] font-body shrink-0">
            生效范围
          </span>
          <input
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            placeholder="留空 = 全局;填模块名仅该模块生效"
            list="prompt-scope-modules"
            className="h-8 flex-1 rounded-lg border border-[var(--color-border-subtle)] px-2.5 text-[12px] font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
          <datalist id="prompt-scope-modules">
            {MODULE_ORDER.filter((m) => m !== "global").map((m) => (
              <option key={m} value={m}>{MODULE_LABEL[m]}</option>
            ))}
          </datalist>
        </div>
        {scope.trim() && (
          <p className="text-[11px] text-[var(--color-info)] font-body">
            当前编辑的是「{MODULE_LABEL[scope.trim()] ?? scope.trim()}」模块覆盖 —— 该模块语境下优先生效,其余场景回落全局版本
          </p>
        )}

        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={18}
          spellCheck={false}
          className="w-full rounded-lg border border-[var(--color-border-subtle)] p-3 text-[13px] font-mono leading-relaxed focus:outline-none focus:border-[var(--color-accent)] resize-y"
        />

        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}

        <div className="flex items-center gap-2 flex-wrap">
          <Button
            onClick={() => save.mutate()}
            disabled={save.isPending || content.trim().length < 10}
          >
            <Save className="w-4 h-4" />
            {save.isPending ? "保存中…" : "保存"}
          </Button>
          <Button
            variant="secondary"
            onClick={() => previewMut.mutate()}
            disabled={previewMut.isPending}
          >
            <Eye className="w-4 h-4" />
            预览渲染
          </Button>
          {tpl.is_custom && (
            <>
              <Button
                variant="secondary"
                onClick={() => setShowVersions((v) => !v)}
              >
                <History className="w-4 h-4" />
                版本
              </Button>
              <Button
                variant="secondary"
                onClick={() => reset.mutate()}
                disabled={reset.isPending}
              >
                <RotateCcw className="w-4 h-4" />
                恢复默认
              </Button>
            </>
          )}
        </div>

        {showVersions && (
          <div className="border border-[var(--color-border-subtle)] rounded-lg divide-y divide-[var(--color-border-row)]">
            {(versionsQuery.data ?? []).map((v) => (
              <div key={v.id} className="text-[12px] font-body">
                <div className="flex items-center justify-between px-3 py-2">
                  <span className="text-[var(--color-text-secondary)]">
                    v{v.version} · {formatRelative(v.created_at)}
                    {v.version === tpl.version && (
                      <span className="ml-2 text-[var(--color-success)]">
                        当前
                      </span>
                    )}
                  </span>
                  <span className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() =>
                        setPreviewVersion((p) =>
                          p === v.version ? null : v.version,
                        )
                      }
                      className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
                    >
                      {previewVersion === v.version ? "收起" : "预览"}
                    </button>
                    {v.version !== tpl.version && (
                      <button
                        type="button"
                        onClick={() => rollback.mutate(v.version)}
                        className="text-[var(--color-accent)] hover:underline"
                      >
                        回滚到此版
                      </button>
                    )}
                  </span>
                </div>
                {previewVersion === v.version && (
                  <pre className="mx-3 mb-2 px-3 py-2 rounded-md bg-[var(--color-bg-subtle)] text-[11px] font-mono text-[var(--color-text-secondary)] whitespace-pre-wrap max-h-[180px] overflow-y-auto">
                    {v.content}
                  </pre>
                )}
              </div>
            ))}
            {versionsQuery.data?.length === 0 && (
              <div className="px-3 py-4 text-center text-[12px] text-[var(--color-text-tertiary)]">
                暂无历史版本
              </div>
            )}
          </div>
        )}

        {preview && (
          <div className="border border-[var(--color-border-subtle)] rounded-lg">
            <div className="px-3 py-2 text-[12px] font-medium text-[var(--color-text-secondary)] border-b border-[var(--color-border-subtle)] font-body">
              渲染预览(变量以 &lt;占位&gt; 显示)
            </div>
            <pre className="px-3 py-3 text-[12px] font-mono text-[var(--color-text-secondary)] whitespace-pre-wrap max-h-[300px] overflow-y-auto">
              {preview}
            </pre>
          </div>
        )}
      </div>
    </Drawer>
  );
}
