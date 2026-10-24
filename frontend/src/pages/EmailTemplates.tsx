import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, Mail, Pencil, RotateCcw, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { Empty } from "@/components/ui/empty";
import { ApiError, fetchJSON } from "@/lib/api";
import { formatRelative } from "@/lib/format";

/**
 * 设置 → 邮件模板(T1-3)。
 *
 * 每个用途一行,抽屉编辑 subject / body_text / body_html,
 * {{var}} 占位;预览同时给出渲染后的主题、纯文本与 HTML。
 */

type VarSpec = { name: string; desc: string };

type Template = {
  key: string;
  name: string;
  description: string;
  variables: VarSpec[];
  subject: string;
  body_text: string;
  body_html: string | null;
  is_custom: boolean;
  updated_at: string | null;
};

type Preview = { subject: string; text: string; html: string | null };

export function EmailTemplates() {
  const qc = useQueryClient();
  const [editing, setEditing] = useState<Template | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  const listQuery = useQuery({
    queryKey: ["email-templates"],
    queryFn: () => fetchJSON<Template[]>("/api/email-templates"),
  });
  const items = listQuery.data ?? [];

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1000px] mx-auto w-full">
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
          邮件模板
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          自定义系统外发邮件的主题与正文 · 支持 {"{{变量}}"} 占位
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
          <Empty icon={Mail} title="没有可管理的邮件模板" />
        ) : (
          items.map((t, i) => (
            <div
              key={t.key}
              className={`flex items-center gap-4 px-6 py-4 ${
                i < items.length - 1
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
                    {t.is_custom ? "自定义" : "默认"}
                  </span>
                </div>
                <p className="text-[12px] text-[var(--color-text-secondary)] font-body truncate">
                  {t.description} · 主题:{t.subject}
                </p>
                {t.is_custom && t.updated_at && (
                  <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                    更新于 {formatRelative(t.updated_at)}
                  </span>
                )}
              </div>
              <Button
                variant="secondary"
                className="!px-3 !py-1.5 !text-[12px] shrink-0"
                onClick={() => setEditing(t)}
              >
                <Pencil className="w-3.5 h-3.5" />
                编辑
              </Button>
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
            qc.invalidateQueries({ queryKey: ["email-templates"] });
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
  const [subject, setSubject] = useState(tpl.subject);
  const [bodyText, setBodyText] = useState(tpl.body_text);
  const [bodyHtml, setBodyHtml] = useState(tpl.body_html ?? "");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: () =>
      fetchJSON<Template>(`/api/email-templates/${tpl.key}`, {
        method: "PUT",
        body: JSON.stringify({
          subject,
          body_text: bodyText,
          body_html: bodyHtml.trim() || null,
        }),
      }),
    onSuccess: () => onSaved("模板已保存"),
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  const previewMut = useMutation({
    mutationFn: () =>
      fetchJSON<Preview>(`/api/email-templates/${tpl.key}/preview`, {
        method: "POST",
        body: JSON.stringify({
          subject,
          body_text: bodyText,
          body_html: bodyHtml.trim() || null,
          vars: {},
        }),
      }),
    onSuccess: setPreview,
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "预览失败"),
  });

  const reset = useMutation({
    mutationFn: () =>
      fetchJSON<Template>(`/api/email-templates/${tpl.key}/reset`, {
        method: "POST",
      }),
    onSuccess: () => onSaved("已恢复默认模板"),
    onError: (e) =>
      setError(e instanceof ApiError ? e.message : "恢复失败"),
  });

  return (
    <Drawer
      open
      onClose={onClose}
      title={tpl.name}
      description={`${tpl.key} · ${tpl.description}`}
      width={720}
    >
      <div className="flex flex-col gap-4">
        <div className="bg-[var(--color-bg-subtle)] rounded-lg p-3 flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            可用变量
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

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            邮件主题
          </span>
          <input
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            className="h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]"
          />
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            纯文本正文
          </span>
          <textarea
            value={bodyText}
            onChange={(e) => setBodyText(e.target.value)}
            rows={10}
            spellCheck={false}
            className="w-full rounded-lg border border-[var(--color-border-subtle)] p-3 text-[13px] font-mono leading-relaxed focus:outline-none focus:border-[var(--color-accent)] resize-y"
          />
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
            HTML 正文(留空则只发纯文本)
          </span>
          <textarea
            value={bodyHtml}
            onChange={(e) => setBodyHtml(e.target.value)}
            rows={10}
            spellCheck={false}
            className="w-full rounded-lg border border-[var(--color-border-subtle)] p-3 text-[13px] font-mono leading-relaxed focus:outline-none focus:border-[var(--color-accent)] resize-y"
          />
        </label>

        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}

        <div className="flex items-center gap-2 flex-wrap">
          <Button
            onClick={() => save.mutate()}
            disabled={save.isPending || subject.trim().length === 0 || bodyText.trim().length < 10}
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
            <Button
              variant="secondary"
              onClick={() => reset.mutate()}
              disabled={reset.isPending}
            >
              <RotateCcw className="w-4 h-4" />
              恢复默认
            </Button>
          )}
        </div>

        {preview && (
          <div className="border border-[var(--color-border-subtle)] rounded-lg overflow-hidden">
            <div className="px-3 py-2 text-[12px] font-medium text-[var(--color-text-secondary)] border-b border-[var(--color-border-subtle)] font-body">
              渲染预览
            </div>
            <div className="px-3 py-2 text-[12px] font-body border-b border-[var(--color-border-row)]">
              <span className="text-[var(--color-text-tertiary)]">主题:</span>{" "}
              <span className="font-medium">{preview.subject}</span>
            </div>
            <pre className="px-3 py-3 text-[12px] font-mono text-[var(--color-text-secondary)] whitespace-pre-wrap max-h-[220px] overflow-y-auto border-b border-[var(--color-border-row)]">
              {preview.text}
            </pre>
            {preview.html && (
              <iframe
                title="html-preview"
                sandbox=""
                srcDoc={preview.html}
                className="w-full h-[220px] bg-white"
              />
            )}
          </div>
        )}
      </div>
    </Drawer>
  );
}
