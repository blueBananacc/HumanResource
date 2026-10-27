import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Link2, Loader2, Network, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConfirm } from "@/components/ui/confirm";
import { ApiError, fetchJSON } from "@/lib/api";

/**
 * 语义检索(Embedding)独立配置页(管理员专用)。
 *
 * 作用对象:当前「已激活」的 LLM Provider —— embedding 是 provider 的
 * 独立通道,可与对话模型不同厂商;切换激活 provider 后此处配置跟随切换。
 *
 * 兜底语义(与后端 services/embedding.py 一致):
 * - 未配 embedding_model → 知识库退化为纯 BM25 词面检索,功能不受影响
 * - embedding_base_url 留空 → 依次兜底 chat base_url、OpenAI 公网
 * - embedding_api_key 留空 → 复用 provider 的对话 key
 */

type EmbeddingCfg = {
  configured: boolean;
  provider_id: string | null;
  provider_name: string | null;
  embedding_model: string | null;
  embedding_base_url: string | null;
  embedding_api_key_masked: string | null;
  effective_base_url: string | null;
  api_key_source: "dedicated" | "chat" | "none" | null;
};

type Feedback = { ok: boolean; message: string };

// 厂商预设 — 实测可用的 OpenAI 兼容 embedding 服务
const PRESETS: { label: string; base_url: string; model: string; hint?: string }[] = [
  {
    label: "SiliconFlow bge-m3(免费·中文好)",
    base_url: "https://api.siliconflow.cn/v1",
    model: "BAAI/bge-m3",
  },
  {
    label: "OpenAI text-embedding-3-small",
    base_url: "https://api.openai.com/v1",
    model: "text-embedding-3-small",
  },
  {
    label: "Jina v3(多语言)",
    base_url: "https://api.jina.ai/v1",
    model: "jina-embeddings-v3",
  },
];

export function EmbeddingSettings() {
  const qc = useQueryClient();
  const confirm = useConfirm();
  const { data, isLoading, isError, error } = useQuery<EmbeddingCfg>({
    queryKey: ["embedding-cfg"],
    queryFn: () => fetchJSON("/api/llm/embedding"),
    retry: false,
  });

  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);

  useEffect(() => {
    if (data) {
      setModel(data.embedding_model ?? "");
      setBaseUrl(data.embedding_base_url ?? "");
      setApiKey("");
    }
    // 仅在 provider 身份变化时回填,避免 refetch 清掉未保存草稿
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.provider_id]);

  const save = useMutation({
    mutationFn: () =>
      fetchJSON<EmbeddingCfg>("/api/llm/embedding", {
        method: "PUT",
        body: JSON.stringify({
          embedding_model: model.trim() || null,
          embedding_base_url: baseUrl.trim() || null,
          embedding_api_key: apiKey || null,
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["embedding-cfg"] });
      qc.invalidateQueries({ queryKey: ["llm-providers"] });
      setErr(null);
      setApiKey("");
      setFeedback({ ok: true, message: "已保存 — 知识库检索立即按新配置生效" });
    },
    onError: (e: unknown) =>
      setErr(e instanceof ApiError ? e.message : "保存失败"),
  });

  const reset = useMutation({
    mutationFn: () =>
      fetchJSON<EmbeddingCfg>("/api/llm/embedding", {
        method: "PUT",
        body: JSON.stringify({ reset: true }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["embedding-cfg"] });
      qc.invalidateQueries({ queryKey: ["llm-providers"] });
      setFeedback({ ok: true, message: "已清除,知识库退化为纯 BM25 检索" });
    },
  });

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    setFeedback(null);
    save.mutate();
  }

  async function onReset() {
    if (
      !(await confirm({
        title: "清除 Embedding 配置?",
        description: "清除后知识库退化为纯 BM25 词面检索,同义/语义召回将失效。",
        tone: "danger",
        confirmLabel: "清除",
      }))
    )
      return;
    reset.mutate();
  }

  const noProvider = isError && error instanceof ApiError;

  return (
    <main className="p-8 flex flex-col gap-6 max-w-[1200px] mx-auto w-full">
      <header className="flex flex-col gap-1.5">
        <h2 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)] flex items-center gap-2">
          <Network className="w-5 h-5" />
          语义检索(Embedding)配置
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] font-body">
          为知识库检索配置向量通道:配好后走「BM25 词面 + 向量语义」混合召回,
          同义词、口语化问法也能命中;留空退化为纯 BM25,功能不受影响。
        </p>
      </header>

      {/* 生效状态 */}
      {data && (
        <div
          className={`flex flex-col gap-1.5 rounded-xl border px-5 py-4 text-[13px] font-body ${
            data.configured
              ? "border-[var(--color-success)]/30 bg-[var(--color-success-soft)]"
              : "border-[var(--color-border-subtle)] bg-[var(--color-bg-subtle)]"
          }`}
        >
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 font-medium ${
                data.configured
                  ? "text-[var(--color-success)]"
                  : "text-[var(--color-text-secondary)]"
              }`}
            >
              <CheckCircle2 className="w-4 h-4" />
              {data.configured ? "语义检索已生效" : "未配置 — 当前为纯 BM25 词面检索"}
            </span>
            <span className="text-[var(--color-text-tertiary)]">
              · 作用于激活的 Provider「{data.provider_name}」
            </span>
          </div>
          {data.configured && (
            <div className="text-[12px] text-[var(--color-text-secondary)] font-mono">
              {data.embedding_model} @ {data.effective_base_url} · key 来源:
              {data.api_key_source === "dedicated"
                ? `独立 key(${data.embedding_api_key_masked})`
                : "复用对话 key"}
            </div>
          )}
        </div>
      )}

      {noProvider && (
        <div className="bg-[var(--color-info-soft)] text-[var(--color-info)] px-4 py-3 rounded-lg text-[13px] font-body">
          尚无激活的 LLM Provider — 请先到{" "}
          <Link to="/settings/llm" className="underline inline-flex items-center gap-0.5">
            <Link2 className="w-3 h-3" /> LLM 配置
          </Link>{" "}
          创建并激活一个,再回来配 Embedding。
        </div>
      )}

      {data && (
        <>
          {/* 厂商预设 */}
          <div className="flex flex-col gap-2">
            <span className="text-[13px] text-[var(--color-text-secondary)] font-body">
              快速填充(选一个服务,预填 base_url 和推荐模型)
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => {
                    setBaseUrl(p.base_url);
                    setModel(p.model);
                  }}
                  className="px-3 py-1.5 rounded-full text-[12px] font-body border border-[var(--color-border-subtle)] bg-white hover:bg-[var(--color-bg-subtle)] transition-colors"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <form
            onSubmit={onSubmit}
            className="flex flex-col gap-5 bg-white border border-[var(--color-border-subtle)] rounded-xl shadow-[var(--shadow-card)] p-6"
          >
            <Field label="Embedding 模型">
              <Input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="BAAI/bge-m3 / text-embedding-3-small / jina-embeddings-v3"
              />
              <span className="text-[11px] text-[var(--color-text-tertiary)] font-body">
                置空保存 = 关闭向量召回(纯 BM25)
              </span>
            </Field>
            <Field label="Embedding Base URL(可选)">
              <Input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.siliconflow.cn/v1 — 留空依次复用 chat base_url、OpenAI"
              />
            </Field>
            <Field
              label={`Embedding API Key(可选${
                data.embedding_api_key_masked
                  ? `,当前 ${data.embedding_api_key_masked},留空保留`
                  : ",留空复用对话 key"
              })`}
            >
              <Input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                autoComplete="new-password"
              />
            </Field>

            {err && (
              <div className="text-[13px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-3 py-2 rounded-md font-body">
                {err}
              </div>
            )}
            {feedback && (
              <div className="text-[13px] px-3 py-2 rounded-md font-body text-[var(--color-success)] bg-[var(--color-success-soft)]">
                <CheckCircle2 className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
                {feedback.message}
              </div>
            )}

            <div className="flex items-center gap-2 flex-wrap">
              <Button type="submit" disabled={save.isPending || isLoading}>
                {save.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
                保存
              </Button>
              {data.configured && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={onReset}
                  disabled={reset.isPending}
                  className="!text-[var(--color-danger)]"
                >
                  <Trash2 className="w-4 h-4" />
                  清除配置
                </Button>
              )}
            </div>
          </form>
        </>
      )}
    </main>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[13px] font-medium text-[#374151] font-body">
        {label}
      </span>
      {children}
    </div>
  );
}
