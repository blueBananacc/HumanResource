/** 租户 LLM 配置 → pi-ai provider/model。
 *
 * backend 在每次 /chat 请求里下发当前租户激活的 LLMProvider(已解密):
 * { base_url, api_key, model } — model 已剥掉 LiteLLM 前缀(如 deepseek/deepseek-flash → deepseek-flash)。
 * mock=true 时 server.ts 不建 Agent,直接回占位文本。
 */
import { createModels, createProvider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import type { Model, MutableModels } from "@earendil-works/pi-ai";

export type LlmConfig = {
  base_url?: string | null;
  api_key?: string | null;
  model: string;
  mock?: boolean;
};

const PROVIDER_ID = "tenant-llm";

/** 构建租户 provider 并注册进 models;传入已有 registry 时原地 upsert,
 * 用于会话存续期间租户切换/轮换 LLM 配置后刷新。 */
export function buildModel(llm: LlmConfig, models?: MutableModels) {
  const baseUrl = (llm.base_url || "https://api.deepseek.com").replace(/\/+$/, "");
  const modelDef: Model<"openai-completions"> = {
    id: llm.model,
    name: llm.model,
    api: "openai-completions",
    provider: PROVIDER_ID,
    baseUrl,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 8192,
  };

  const apiKey = llm.api_key ?? "";
  const provider = createProvider({
    id: PROVIDER_ID,
    name: "Tenant LLM",
    baseUrl,
    auth: {
      apiKey: {
        name: "tenant key",
        resolve: async () => ({ auth: { apiKey } }),
      },
    },
    models: [modelDef],
    api: openAICompletionsApi(),
  });

  const registry = models ?? createModels();
  registry.setProvider(provider);
  const model = registry.getModel(PROVIDER_ID, modelDef.id);
  if (!model) throw new Error(`model ${modelDef.id} 注册失败`);
  return { models: registry, model };
}
