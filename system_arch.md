Human Resources Support Agent — Implementation Outline
# 1 System Overview
本系统是一个 HR 领域多智能体（Multi-Agent）支持系统，基于 LLM 提供：
- HR 政策问答（Policy Q&A）
- HR 流程咨询（Process Inquiry）
- 员工信息查询（Employee Lookup）
- HR 文档检索（Document Search）
- HR 工具调用（Tool Actions）
**MVP 目标**：技术路径正确、功能模块完整、架构可扩展，不追求性能/高可用/企业级特性。

## 1.1 Technology Stack
| 类别         | 技术选型                          | 说明                                                                 |
|--------------|-----------------------------------|----------------------------------------------------------------------|
| 用户交互     | CLI（命令行）                     | 基于 Python input/print 循环                                         |
| LLM          | DeepSeek（官方 API）              | deepseek-chat（通用场景）+ deepseek-reasoner（复杂推理）             |
| LLM 接入方式 | OpenAI-compatible API             | API Base: https://api.deepseek.com，通过 langchain-openai 的 ChatOpenAI 接入 |
| Agent 框架   | LangChain + LangGraph             | LangChain 提供 LLM/Tool/RAG 基础组件；LangGraph 提供 Agent 编排与状态管理 |
| Agent 架构   | ReAct + Reflexion                 | ReAct：思考→行动→观察循环；Reflexion：可选的自我反思与重试机制       |
| Embedding    | BAAI/bge-m3                       | 通过 HuggingFace Inference API 调用（云端），1024 维向量              |
| Reranker     | BAAI/bge-reranker-v2-m3           | 通过 HuggingFace Inference API 调用（云端），cross-encoder 重排序     |
| 向量数据库   | ChromaDB（本地）                  | 零配置、Python 原生，用于 RAG 文档检索                               |
| Session 存储 | 本地内存 + JSON 文件              | 进程内 dict + JSON 持久化                                            |
| 长期记忆     | mem0 Cloud                        | 托管服务，内置 embedding/向量存储/去重/重要性评估                   |
| agent的Prompt 框架  | CRISPE                            | Capacity-Role / Insight / Statement / Personality / Experiment        |
| MCP          | MCP Client（后续接入）            | 预留 MCP Client 架构，MCP Server 后续提供或构建                      |

## 1.2 Required Credentials
| 凭证                     | 环境变量           | 用途                                   |
|--------------------------|--------------------|----------------------------------------|
| DeepSeek API Key         | DEEPSEEK_API_KEY   | LLM 调用（chat + reasoner）            |
| HuggingFace API Token    | HF_API_TOKEN       | BGE Embedding + Reranker 推理          |
| mem0 API Key             | MEM0_API_KEY       | 长期记忆云端存储与检索                 |

## 1.3 Core Dependencies
langchain, langchain-openai, langchain-community, langchain-chroma, langchain-huggingface
langgraph
chromadb
mem0ai
rank-bm25
pypdf, python-docx
pytest

# 2 System Architecture
## 2.1 High-Level Architecture
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│                       (CLI 命令行)                       │
└──────────────────────┬──────────────────────────────────┘
                       │ user message
                       ▼
┌─────────────────────────────────────────────────────────┐
│    Orchestrator Agent (LangGraph StateGraph)            │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ Intent   │  │ Context  │  │ Response Generator  │    │
│  │ Recogn.  │  │ Manager  │  │                     │    │
│  └────┬─────┘  └────┬─────┘  └─────────┬──────────┘    │
│       │              │                  │                │
│       ▼              ▼                  ▼                │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Agent Router (Conditional Edges)   │    │
│  └──┬──────────────┬──────────────┬────────────────┘    │
└─────┼──────────────┼──────────────┼─────────────────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ RAG Agent│  │Tool Agent│  │ Memory   │
│ (ReAct)  │  │  (ReAct) │  │ Agent    │
│          │  │          │  │          │
│ Retrieval│  │ Internal │  │ Session  │
│ Ranking  │  │ MCP Tools│  │ Longterm │
│ Synthesis│  │ Execute  │  │ Profile  │
└──────────┘  └──────────┘  └──────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│VectorDB  │  │ Tool     │  │ mem0     │
│(ChromaDB)│  │ Registry │  │ Cloud    │
│          │  │ MCP Svr  │  │          │
└──────────┘  └──────────┘  └──────────┘

- **框架映射**：
  - Orchestrator = LangGraph StateGraph，定义全局执行流程图
  - 每个 Specialist Agent = Graph 中的一个 Node 函数
  - Agent Router = conditional_edges，根据 Intent 分类结果路由到不同 Node
  - 状态管理 = LangGraph State（TypedDict），在 Node 之间传递和更新



## 2.2 Agent Architecture
| Agent | 职责 | 输入 | 输出 | 依赖模块 |
|------|------|------|------|------|
| Orchestrator | 接收用户输入、意图识别、路由分发、聚合结果、生成最终回复 | 用户消息 + 会话上下文 | 最终回复 | Intent、Context、Router、LLM Client |
| RAG Agent | 基于 HR 文档库执行检索增强生成 | 查询文本 + 检索过滤条件 | 检索结果 + 上下文片段（结构化中间数据） | RAG Pipeline、Context |
| Tool Agent | 执行内部工具调用和 MCP 工具调用 | 工具名称 + 参数 | 工具执行结果（结构化） | Tool Registry、MCP Client |
| Memory Agent | 管理会话记忆写入/读取、长期记忆提取/检索 | 对话历史 / 检索查询 | 相关记忆上下文 | Session Memory、Long-term Memory |

**子 Agent 输出原则**：
- 所有 Specialist Agent（RAG / Tool / Memory）**只返回结构化中间结果**，写入 LangGraph State 对应字段
- **不生成用户可见的最终回答**
- 最终用户回复**仅由 Orchestrator 的 Response Generator 节点**生成，基于所有中间结果统一组装 prompt 后调用 LLM
- 好处：保证回复风格一致、便于多意图结果聚合、Context Engineering 集中管理

- **Agent 执行模式**：
  - ReAct 模式（所有 Specialist Agent 默认使用）：
  - 使用 LangGraph 的 create_react_agent() 构建
  - 循环流程：LLM 思考 → 选择 Action (工具/检索) → 执行 → 观察结果 → 继续或输出最终结果

- **Reflexion 模式（可选扩展，用于关键路径）**：
  - 在 Agent 生成输出后添加自我反思节点
  - 反思 prompt 要求 LLM 评估回答质量（准确性、完整性、相关性）
  - 评估不通过 → 通过 conditional edge 回到 Agent 节点重新执行，携带反思反馈
  - MVP 阶段可选启用：建议在 policy_qa（政策准确性要求高）场景启用
  - 设置最大反思次数（如 2 次），避免无限循环

**LangGraph State 定义**：
AgentState {
  messages: list[BaseMessage]        // LangGraph 消息列表（对话历史）
  intent: IntentResult | None        // 意图识别结果
  target_agents: list[str]           // 路由目标
  rag_results: RetrievalResult | None
  tool_results: list[ToolResult]
  memory_context: list[str]          // 检索到的长期记忆
  user_profile: dict | None
  final_response: str | None
  reflection_count: int              // Reflexion 重试计数
}

## 2.3 Data Flow（端到端）
**完整的请求处理流程（每步标注数据形态）**：
① 用户输入 (raw text) — CLI input()
   │
② Orchestrator 加载上下文
   ├─→ Session Memory(本地): 获取当前会话历史 (message list)
   └─→ Memory Agent → mem0 Cloud: 检索相关长期记忆 (memory snippets)
   │
③ Intent Recognition(DeepSeek-reasoner, structured output)
   │  输入: 用户消息 + 会话摘要
   │  输出: intent_label, confidence, entities, requires_tools[]
   │
④ Router 决策(LangGraph conditional_edges)
   │  输入: intent 分类结果
   │  输出: target_agents[], execution_plan
   │
⑤ Agent 执行 — ReAct 循环（可并行）
   ├─→ RAG Agent (policy_qa / document_search)
   │   输入: query + metadata_filter
   │   内部: retrieve → rerank → context_filter
   │   输出: retrieved_chunks[] → 写入 state.rag_results
   │
   ├─→ Tool Agent (employee_lookup / tool_action / process_inquiry)
   │   输入: tool_name + parameters
   │   内部: registry_lookup → schema_validate → execute → format
   │   输出: tool_result (structured)
   │
   └─→ Memory Agent (需要时)
       输入: memory_query
       输出: relevant_memories[]
   │
    [可选] Reflexion 节点 (DeepSeek-reasoner)
   │  评估 Agent 输出质量 → 通过则继续，不通过则回到 Agent 重试
   │
⑥ Context Assembly
   │  输入: agent 结果 + 会话历史 + 记忆 + 系统 prompt
   │  处理: token counting → compression → prompt building
   │  输出: final_prompt (within context window)
   │
⑦ LLM 生成最终回复(DeepSeek-chat)
   │  输入: final_prompt
   │  输出: response text
   │
⑧ Post-processing
   ├─→ Session Memory(本地): 追加当轮对话 (user_msg + assistant_msg) → JSON 持久化
   ├─→ Memory Agent → mem0 Cloud: 判断是否需要写入长期记忆
   │   (提取关键事实 → mem0.add() 写入, mem0 自动处理去重/重要性评估)
   └─→ 返回响应给用户(CLI print)

**数据流完整性验证**：
✅ 用户输入 → 意图识别 → 路由 → Agent 执行 → 结果聚合 → 回复生成 → 记忆更新：闭环
✅ 所有模块有明确输入来源和输出
✅ 无孤立模块、无断裂调用链

# 3 Core Modules
## 3.1 Intent Recognition
**目标**：将用户自然语言输入分类为可路由的意图类别，支持多意图识别和低置信度回退。

**子模块**：
| 子模块 | 职责 |
|------|------|
| Intent Classifier | LLM-based 分类，输出 intent label + confidence + entities |
| Intent Router | 根据分类结果映射到目标 Agent 和执行计划 |
**意图分类体系（MVP）**：
| Intent Label | 描述 | 路由目标 |
|-------------|------|----------|
| policy_qa | HR 政策问答 | RAG Agent |
| process_inquiry | HR 流程咨询 | RAG Agent + Tool Agent |
| employee_lookup | 员工信息查询 | Tool Agent |
| document_search | HR 文档检索 | RAG Agent |
| tool_action | HR 工具操作 | Tool Agent |
| memory_recall | 回忆之前的对话内容 | Memory Agent |
| chitchat | 闲聊/问候 | Orchestrator 直接响应 |
| unknown | 无法识别 | Fallback 流程 |

**关键技术考虑**：

- 分类方式：使用 LLM structured output（JSON mode），prompt 中包含意图定义与示例。对于 MVP 不需要训练分类模型，LLM few-shot 足够
- Prompt Routing：Classifier 输出 intent label → Router 查表映射到 Agent。映射表可配置，便于扩展新意图
- Tool Routing：当 intent 包含工具需求时，Classifier 同时输出 required_tools[]，Tool Agent 根据此列表选择工具
- Multi-intent 处理：Classifier prompt 设计为可返回 intent_list[]（多意图数组）。Router 按优先级顺序执行，或并行分发到多个 Agent，最后由 Orchestrator 聚合结果
- Fallback 策略：三级回退机制：
  confidence < 阈值 → 追问澄清（生成澄清问题返回给用户）
  重试后仍无法识别 → 路由到 RAG Agent 做宽泛检索
  RAG 也无结果 → 返回友好的"无法回答"提示 + 建议联系 HR

**技术选型映射**：
- LLM：使用 DeepSeek-reasoner 进行意图分类

**输入**：用户消息 (str) + 会话摘要 (str, optional)
**输出**：IntentResult { intents: [{label, confidence, entities}], requires_tools: [str] }
**依赖**：LLM Client
**被调用方**：Orchestrator Agent

## 3.2 Tool Calling
**目标**：提供统一的工具注册、发现、选择、执行和结果格式化机制。

**子模块**：
| 子模块 | 职责 |
|------|------|
| Tool Registry | 注册/管理所有可用工具的定义和 schema |
| Tool Selector | 基于意图和上下文选择合适的工具 |
| Tool Executor | 执行工具调用，处理参数校验和调用 |
| Result Formatter | 将工具原始返回值格式化为 LLM 可消费的结构 |
**Tool Registry 设计**：
- 每个工具注册为一个 ToolDefinition，包含：name、description、parameters (JSON Schema)、return_schema、category、requires_auth
- 注册方式：声明式注册（装饰器或配置文件），启动时自动发现并加载
- 支持按 category 分组（employee_tools, process_tools, document_tools）
**MVP 内置工具**：
| 工具名 | 类别 | 功能 |
|------|------|------|
| lookup_employee | employee | 根据姓名/工号查询员工信息 |
| get_leave_balance | employee | 查询员工假期余额 |
| list_hr_processes | process | 列出可用 HR 流程 |
| get_process_steps | process | 查询具体流程步骤 |

**关键技术考虑**：
- Tool Schema：每个工具以 JSON Schema 定义输入参数，LLM 根据 schema 生成调用参数。Schema 同时用于参数校验
- Tool Selection：两阶段选择 —— ① Intent Classifier 输出 requires_tools[] 做粗筛 ② 将候选工具 schema 注入 prompt，由 Tool Agent 做精细选择（native function calling）
- Tool Error Handling：
  参数校验失败 → 返回参数错误信息，LLM 可修正后重试（最多 2 次）
  工具执行异常 → 捕获异常，返回结构化错误，LLM 生成用户友好的错误说明
  工具超时 → 设置执行超时，超时返回降级提示
- Tool Result Formatting：工具返回结构化数据（dict/JSON）→ Formatter 转换为 LLM 上下文友好的文本格式（保留关键字段，省略冗余数据）

**技术选型映射**：
- 工具定义：使用 LangChain 的 @tool 装饰器或 StructuredTool.from_function() 定义工具，自动生成 JSON Schema
- 工具绑定：通过 llm.bind_tools(tools) 将工具列表绑定到 DeepSeek LLM
- ReAct 执行：Tool Agent 使用 create_react_agent(model, tools) 构建 ReAct 循环，LLM 自动选择工具并执行
- DeepSeek 兼容：DeepSeek API 支持 OpenAI function calling 格式，与 LangChain tool binding 完全兼容

**输入**：tool_name (str) + parameters (dict)
**输出**：ToolResult { success: bool, data: Any, error: str | None, formatted: str }
**依赖**：LLM Client（用于 tool selection）
**被调用方**：Tool Agent

## 3.3 MCP Tool Integration
**目标**：通过 **MCP (Model Context Protocol)** 接入外部 MCP Provider 提供的工具，实现工具的动态发现与调用，而无需自行实现 MCP Server。
系统仅作为 **MCP Client**，从外部 MCP 服务发现并调用工具。

**子模块**：

子模块 | 职责
--- | ---
MCP Client | 连接外部 MCP Server，发现并调用工具
MCP Tool Adapter | 将 MCP 工具适配为系统内部 Tool Registry 统一格式


**关键技术考虑**：

- MCP Client
  - 系统作为 MCP Client 与外部 MCP Provider 连接：
  - 支持 MCP 标准 transport：stdio、SSE
  - MCP Client 负责：建立连接、调用 MCP API、工具发现、工具执行

- MCP Tool Discovery：
  - MCP Client 启动时调用 tools/list 获取所有可用工具
  - 将 MCP 工具自动注册到 Tool Registry（与内置工具统一管理）
  - 工具元数据包含：name, description, inputSchema
- Tool Metadata：
  - 每个 MCP Tool 携带完整的 JSON Schema 描述
  - 支持 tool annotations（readOnlyHint, destructiveHint 等安全标记）
  - Metadata 用于 LLM 工具选择和参数生成  

**Tool Invocation Flow**：
Tool Agent 选择 MCP 工具
  → MCP Client 构造 tools/call 请求
  → MCP Server 接收并路由到具体 handler
  → Handler 执行业务逻辑
  → 返回结构化结果
  → MCP Client 解析并传回 Tool Agent

- **与内置工具的统一**：MCP 工具和内置工具在 Tool Registry 中统一注册，Tool Agent 无需区分来源。Registry 中标记 source: "internal" | "mcp"，执行时根据 source 走不同调用路径

**技术选型映射**：
- MCP 适配：使用 langchain-mcp-adapters 库将 MCP 工具转换为 LangChain Tool 对象，无缝接入 LangGraph ReAct Agent
- MCP Server：后续再提供/构建。架构中预留 MCP Client 接入点，当 MCP Server 就绪后，只需配置连接信息即可接入
- 配置方式：MCP Server 连接信息通过 config 文件管理（server_url, transport_type, auth 等）

**输入**：MCP tool name + parameters
**输出**：MCP tool result（与内置工具统一的 ToolResult 格式）
**依赖**：Tool Registry
**被调用方**：Tool Agent（通过 Tool Registry 间接调用）

## 3.4 RAG System
**目标**：从 HR 文档库中检索相关信息，为 LLM 生成高质量的有据回答提供上下文。
**子模块**：
| 子模块 | 职责 |
|------|------|
| Document Loader | 加载多种格式的 HR 文档 |
| Chunker | 将文档切分为语义合理的片段 |
| Embedder | 生成文本向量嵌入 |
| Vector Store | 存储和检索向量 |
| Retriever | 执行检索策略（hybrid search） |
| Reranker | 对检索结果重排序 |

**文档处理流程**（离线/索引阶段）：
原始文档 (PDF/DOCX/MD/TXT)
  → Document Loader (解析提取文本 + 元数据)
  → Chunker (分块)
  → Embedder - — BGE-M3 via HuggingFace Inference API (生成 1024 维向量)
  → Vector Store (存储 chunk + vector + metadata)

**检索流程**（在线/查询阶段）：
用户查询
  → Embedder — BGE-M3 (查询向量化)
  → Retriever
     ├─ Vector Search — ChromaDB (语义相似度, top-K)
     ├─ Keyword Search (BM25, top-K)
     └─ Merge + Deduplicate
  → Reranker - BGE-Reranker-v2-M3 via HuggingFace Inference API (cross-encoder 重排序, top-N)
  → Context Filter (token 预算内选择)
  → 返回 ranked chunks

**关键技术考虑**：
- **文档加载**：
  - MVP 支持格式：PDF (pypdf)、DOCX (python-docx)、Markdown、纯文本
  - 提取文档元数据：文件名、标题、创建日期、文档类别（policy/handbook/SOP）
  - 增量加载：记录已索引文档的 hash，仅处理变更文档

- **文档切片（Chunking）**：
  - 策略：Recursive Character Splitting（按段落 → 句子 → 字符递归切分）
  - chunk_size: 512 tokens（MVP），chunk_overlap: 64 tokens
  - 保留 chunk 级元数据：所属文档、章节标题、页码、chunk 序号
  - 考虑：HR 文档通常有清晰的章节结构，优先按标题/章节边界切分

- **Embedding**：
  - 模型：BAAI/bge-m3, 向量维度：1024 维
  - 调用方式：通过 HuggingFace Inference API
  - 批量嵌入以降低 API 调用次数
  - embedding 维度与 ChromaDB 配置一致

- **向量数据库**：
  - MVP 选择：ChromaDB
  - 存储内容：chunk text + embedding vector + metadata dict
  - Collection 按文档类别分组（policy_collection, handbook_collection）或统一 collection + metadata 过滤

- **Metadata 设计**：
{
  "source": "employee_handbook.pdf",
  "category": "policy",        // policy | handbook | sop | faq
  "section": "Leave Policy",
  "page": 12,
  "chunk_index": 3,
  "last_updated": "2026-01-15"
}
  - 检索时可按 category、section 做 pre-filter

- **Hybrid Search**：
  - 向量搜索：语义相似度，捕获同义/近义表述
  - 关键词搜索：BM25（使用 rank_bm25 库），捕获精确术语匹配
  - 融合策略：Reciprocal Rank Fusion (RRF)，结合两路结果 

- **Reranking**：
  - 模型：BAAI/bge-reranker-v2-m3（cross-encoder，多语言支持）
  - 调用方式：通过 HuggingFace Inference API（云端推理）
  - 输入：query + candidate chunks（hybrid search 结果）
  - 输出：重排序后的 chunks + relevance scores 

- **Context Filtering**：
  - Reranking 后，根据 token 预算（分配给 RAG 上下文的 token 数）截取 top-N chunks
  - 相邻 chunk 合并（若来自同一文档同一节）以保持语义连贯

- **Retrieval 策略**：
  - 默认：hybrid search + reranking
  - 当用户查询包含明确文档名/类别时：先 metadata pre-filter，再向量检索
  - 当检索结果 relevance score 均低于阈值时：返回"未找到相关信息"

**技术选型映射**：
- Document Loader：使用 langchain_community.document_loaders 的 PyPDFLoader、Docx2txtLoader、TextLoader、UnstructuredMarkdownLoader
- Chunker：使用 langchain.text_splitter.RecursiveCharacterTextSplitter
- Embedder：使用 langchain_huggingface.HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3", huggingfacehub_api_token=HF_API_TOKEN)
- Vector Store：使用 langchain_chroma.Chroma
- BM25：使用 rank_bm25.BM25Okapi，配合 LangChain 自定义 Retriever 封装
- Reranker：通过 HuggingFace Inference API 调用 BAAI/bge-reranker-v2-m3，封装为自定义 reranker 函数

**输入**：查询文本 (str) + 可选 metadata_filter (dict)
**输出**：RetrievalResult { chunks: [{text, score, metadata}]}
**依赖**：Embedder、Vector Store、LLM Client
**被调用方**：RAG Agent

## 3.5 Context Engineering
**目标**：在 LLM context window 限制内，最优地组装 prompt，确保关键信息不丢失。
**子模块**：
| 子模块 | 职责 |
|------|------|
| Context Manager | 管理 token 预算分配和上下文组装 |
| Context Compressor | 压缩/摘要过长的上下文片段 |
| Prompt Builder | 按固定结构组装最终 prompt |

**Prompt 结构（固定模板）**：
[System Prompt]          → 角色定义 + 行为规则 + 输出格式  (固定, ~300 tokens)
[User Profile]           → 长期记忆中的用户画像             (动态, ~100 tokens)
[Relevant Memories]      → 检索到的长期记忆片段             (动态, ~200 tokens)
[Retrieved Context]      → RAG 检索结果                    (动态, ~1500 tokens)
[Tool Results]           → 工具调用返回的结果               (动态, ~500 tokens)
[Conversation History]   → 会话历史                        (动态, ~2000 tokens)
[Current User Message]   → 当前用户输入                    (动态, ~200 tokens)

**关键技术考虑**：
- **Context Window 限制**：
  - DeepSeek-chat / DeepSeek-reasoner context window：64K tokens
  - 预留 response token 预算：DeepSeek-chat 8K / DeepSeek-reasoner 16K（含推理 token）
  - 可用上下文预算：~56K tokens（chat）/ ~48K tokens（reasoner）
  - 剩余空间按优先级分配给各 prompt 段落
  - 优先级：System Prompt > Current Message > Tool Results > Retrieved Context > Conversation History > Memories

- **Context 压缩**：
  - 会话历史压缩：当历史超过预算时，对较早的对话轮次进行 LLM 摘要，保留最近 N 轮原文
  - 检索结果压缩：若 RAG chunks 超过预算，先按 relevance 截断；仍超出则提取关键句
  - 压缩时机：在 Prompt Builder 组装时按需触发，不预先压缩

- **Compression 策略**：
  - 滑动窗口摘要：保留最近 K 轮原文 + 更早对话的摘要
  - 增量摘要：每次新对话时仅摘要最旧的未摘要部分，避免重复处理全部历史

- **Summarize Memory**：
  - 对话结束时（或每 N 轮），将完整对话历史摘要为 episodic memory 存入长期记忆
  - 摘要 prompt 要求提取：关键问题、关键决策、用户偏好

- **Retrieval Filtering**：
  - 仅将与当前意图相关的记忆/检索结果注入 prompt
  - 避免注入无关上下文造成 LLM 混淆

**技术选型映射**：
- Token 计数：使用 tiktoken（cl100k_base 编码，与 DeepSeek tokenizer 近似）进行 token 预算管理
- 压缩 LLM：可以使用 DeepSeek-chat 执行上下文压缩和摘要（成本低、速度快）

**输入**：各模块的原始输出（会话历史、RAG chunks、工具结果、记忆片段）
**输出**：组装好的 final prompt (str)，遵守 token 预算
**依赖**：LLM Client（用于压缩/摘要）
**被调用方**：Orchestrator Agent

## 3.6 Session Memory
**目标**：维护单次对话的完整上下文，支持多轮对话的连贯性。
**子模块**：
| 子模块 | 职责 |
|------|------|
| Conversation Store | 按 session_id 存储对话消息列表 |
| Memory Trimmer | 按策略裁剪超长对话历史 |
| Session Summarizer | 对旧对话进行增量摘要 |

**数据模型**：
Session {
  session_id: str
  user_id: str
  messages: [
    { role: "user"|"assistant"|"system", content: str, timestamp: datetime }
  ]
  summary: str              // 旧消息的滚动摘要
  metadata: {
    created_at, last_active, turn_count, intent_history[]
  }
}

**关键技术考虑**：
- **Conversation History**：
  - 每轮对话追加 user message + assistant response
  - 同时记录 intent label 和调用的工具/Agent(用于后续分析)

- **Short-term Memory**：
  - 即当前 session 的完整对话历史
  - 读取时返回最近 K 轮原文 + 更早部分的摘要

- **Memory Trimming**：
  - 触发条件：对话轮数超过 5 轮或消息列表 token 数超过 session_memory_budget
  - 策略：保留最近 N 轮原文（如 2 轮），更早的消息移入 pending_summarize 队列

- **Memory Summarization**：
  - 将 pending_summarize 的消息用 LLM 生成增量摘要
  - 新摘要与已有 summary 合并（追加或重新摘要）
  - 摘要保留：关键问题、关键回答、用户意图变化、重要决策

**技术选型映射**：
- 运行时存储：Python dict（进程内），key 为 session_id
- 持久化：JSON 文件存储到 data/sessions/ 目录。每个 session 一个 JSON 文件，进程退出时序列化、启动时恢复
- LangGraph 消息整合：Session 的 messages 列表与 LangGraph State 的 messages 保持同步（LangGraph 使用 BaseMessage 列表）
- 摘要 LLM：使用 DeepSeek-chat 生成会话摘要

**输入**：session_id + 操作（read / append / trim / summarize）
**输出**：会话历史（messages + summary）
**依赖**：LLM Client（用于摘要生成）
**被调用方**：Orchestrator Agent、Memory Agent

## 3.7 Long-term Memory
**目标**：跨会话持久保存用户画像、历史交互摘要和关键知识，实现个性化和记忆连续性。
**子模块**：
| 子模块 | 职责 |
|------|------|
| User Profile Store | 维护用户基本信息和偏好 |
| Episodic Memory Store | 存储会话摘要和关键事件 |
| Vector Memory | 向量化存储，支持语义检索 |
| Memory Writer | 判断并执行长期记忆写入 |

**Memory 类型设计**：
| 类型 | 内容 | 存储方式 | 示例 |
|----|------|----------|------|
| User Profile | 用户角色、部门、偏好 | KV Store | role: "Engineering Manager", department: "R&D" |
| Episodic Memory | 每次会话的摘要 | Vector Store + Metadata | "2026-03-15: 用户咨询了年假政策，确认剩余 5 天" |
| Factual Memory | 用户明确告知的事实 | Vector Store | "用户的直属上级是张三" |

**关键技术考虑**：
- **Memory 写入策略**：
  - 会话结束时：Memory Writer 分析完整对话历史，提取值得记住的信息
  - 提取 prompt 要求 LLM 输出：[{type: "profile"|"episodic"|"factual", content: str, importance: float}]
  - 仅 importance > 阈值的记忆被写入
  - 去重：新记忆与已有记忆做语义相似度检查，重复则更新而非新增

- **Memory Retrieval**：
  - 每次新对话开始时，用用户 ID 检索 profile
  - 每次需要上下文时，用当前查询做向量检索，获取相关 episodic/factual memory
  - 返回 top-K 最相关记忆 + 用户 profile

- **User Profile Memory**：
  - 结构化 KV 存储（dict / JSON）
  - 字段：name, department, role, preferences, frequently_asked_topics
  - 随对话逐步丰富，支持更新/覆盖

- **Episodic Memory**：
  - 每次会话结束生成一条 episodic memory
  - 包含：session_summary, key_topics, key_decisions, timestamp
  - 向量化存储，支持"之前我们讨论过什么"类查询

**技术选型映射** — mem0 Cloud：
- 后端引擎：使用 mem0 Cloud 作为长期记忆的统一后端，mem0 内部自动处理：
  - Embedding 生成（无需额外调用 BGE）
  - 向量存储与索引
  - 记忆去重与冲突解决
  - 重要性评估
- 接入方式：mem0ai Python SDK，MemoryClient(api_key=MEM0_API_KEY)
- Memory 类型 → mem0 映射：
| 系统 Memory 类型   | mem0 操作                                                                 | metadata 区分      |
|--------------------|----------------------------------------------------------------------------|--------------------|
| User Profile       | client.add(messages, user_id=..., metadata={"type": "profile"})            | type=profile       |
| Episodic Memory    | client.add(messages, user_id=..., metadata={"type": "episodic", "session_id": ...}) | type=episodic      |
| Factual Memory     | client.add(messages, user_id=..., metadata={"type": "factual"})            | type=factual       |

- **写入流程**：
  1. 会话结束 → DeepSeek-chat 提取值得记住的信息（按原 Memory Writer 逻辑）
  2. 将提取结果通过 mem0.add() 写入，附带 metadata（type, timestamp, importance）
  3. mem0 自动处理去重（语义相似的记忆会被合并/更新）
- **检索流程**：
  1. client.search(query, user_id=...) → 返回语义相关的记忆列表
  2. client.get_all(user_id=..., metadata={"type": "profile"}) → 检索用户画像
  3. 结果传递给 Context Manager 注入 prompt

- **与 Session Memory 的分工**：Session Memory（本地 dict + JSON）管理当前会话的短期记忆；mem0 Cloud 管理跨会话的长期记忆。两者通过 Memory Agent 协调。

**输入**：user_id + 操作（read_profile / search_memory / write_memory）
**输出**：相关记忆列表 / 用户画像
**依赖**：mem0 Cloud、LLM Client
**被调用方**：Memory Agent → Orchestrator

# 4 Multi-Agent Coordination
## 4.1 Orchestrator Agent（中心协调者）
- **角色**：系统唯一入口，负责全局流程控制
- **职责**：
  1. 接收用户输入
  2. 加载 session context + 相关长期记忆
  3. 调用 Intent Classifier
  4. 根据 Router 决策分发到 specialist agents
  5. 收集 agent 结果
  6. 通过 Context Manager 组装 prompt
  7. 调用 LLM 生成最终回复
  8. 更新 session memory + 触发长期记忆写入

**技术实现**：Orchestrator 即 LangGraph StateGraph 的编译图（compiled graph），上述职责对应图中的不同 Node 和 Edge。

## 4.2 Agent Routing 规则
| 意图 | 主 Agent | 辅助 Agent | 执行模式 |
|------|----------|------------|----------|
| policy_qa | RAG Agent | Memory Agent（提供历史上下文） | 串行 |
| document_search | RAG Agent | — | 单一 |
| employee_lookup | Tool Agent | — | 单一 |
| tool_action | Tool Agent | — | 单一 |
| process_inquiry | RAG Agent | Tool Agent（可能调用流程工具） | 串行 |
| memory_recall | Memory Agent | — | 单一 |
| chitchat | Orchestrator 直接响应 | — | 直接 |
| unknown | 澄清 → fallback 到 RAG | — | 回退 |
| Multi-intent | 按优先级顺序执行多个 Agent | — | 顺序/并行 |

**技术实现**：路由规则通过 LangGraph conditional_edges 实现，Router Node 根据 state.intent.label 返回目标 Node 名称。

## 4.3 错误处理与降级
- 任何 specialist agent 失败 → Orchestrator 捕获异常 → 生成降级响应（告知用户该功能暂不可用）
- LLM 调用失败 → 重试 1 次 → 仍失败则返回预设错误消息
- 全局 trace_id 贯穿调用链，便于日志追踪

## DeepSeek 双模型使用策略
| 场景                                      | 模型                 | 理由                                           |
|-------------------------------------------|----------------------|------------------------------------------------|
| Intent Classification                     | deepseek-reasoner        | 分类任务复杂，包含分级意图,多轮意图        |
| Tool Selection & Calling                  | deepseek-reasoner        | function calling 场景，需要准确选择和工具调用参数准确度            |
| RAG Answer 检索推理                    | deepseek-chat        | 查询改写和检索策略选择，标准能力足够                        |
| Context Compression & Summarization       | deepseek-chat        | 摘要任务，不需要深度推理                      |
| Final Response Generation（简单问题）     | deepseek-chat        | 日常 HR 问答                                   |
| Final Response Generation（复杂问题）     | deepseek-reasoner    | 涉及多条件判断、政策交叉比较等复杂推理        |
| Reflexion Self-critique                   | deepseek-reasoner    | 反思评估需要更强的推理能力                    |
| Memory Extraction                         | deepseek-chat        | 信息提取任务，标准能力即可                    |

**模型切换机制**：在 config.py 中配置各场景使用的模型名称。llm_client 模块提供 get_llm(purpose) 工厂方法，根据 purpose 返回对应模型的 ChatOpenAI 实例。

# 5 Project Structure (MVP)
src/human_resource/
├── main.py                          # 应用入口 + CLI 交互循环
├── config.py                        # 配置管理 (API keys, model, thresholds)
│
├── agents/                          # Agent 层
│   ├── __init__.py
│   ├── base.py                      # BaseAgent 抽象基类 (定义统一接口)
│   ├── orchestrator.py              # Orchestrator Agent (LangGraph StateGraph 定义)
│   ├── graph.py                     # LangGraph 图定义: Nodes + Edges + 编译
│   ├── rag_agent.py                 # RAG Agent (ReAct)
│   ├── tool_agent.py                # Tool Agent (ReAct)
│   └── memory_agent.py             # Memory Agent (Session + mem0 协调)
│
├── intent/                          # 意图识别模块
│   ├── __init__.py
│   ├── classifier.py                # LLM-based 意图分类器
│   └── router.py                    # 意图 → Agent 路由映射(conditional_edges 配置)
│
├── rag/                             # RAG 模块
│   ├── __init__.py
│   ├── loader.py                    # 文档加载器 (LangChain PyPDFLoader/Docx2txt/TextLoader)
│   ├── chunker.py                   # 文档切片器 (RecursiveCharacterTextSplitter)
│   ├── embedder.py                  # Embedding 生成器 (HuggingFaceEndpointEmbeddings → BGE-M3)
│   ├── vectorstore.py              # ChromaDB 向量存储封装
│   ├── retriever.py                 # Hybrid Search 检索器 (Vector + BM25 + RRF)
│   └── reranker.py                  # 重排序器 (BGE-Reranker-v2-M3 via HF Inference API)
│
├── tools/                           # 工具调用模块
│   ├── __init__.py
│   ├── registry.py                  # 工具注册表 (统一管理内置+MCP工具)
│   ├── executor.py                  # 工具执行器 (参数校验+调用+格式化)
│   └── hr_tools/                    # HR 专用内置工具 (LangChain @tool 装饰器)
│       ├── __init__.py
│       ├── employee_lookup.py       # 员工信息查询
│       └── process_tools.py         # HR 流程工具
│
├── mcp/                             # MCP 集成模块 (后续接入)
│   ├── __init__.py
│   └── client.py                    # MCP Client (langchain-mcp-adapters, 发现+调用)
│
├── memory/                          # 记忆模块
│   ├── __init__.py
│   ├── session.py                   # Session Memory (本地 dict + JSON 持久化)
│   ├── longterm.py                  # Long-term Memory (mem0 Cloud 封装)
│   └── profile.py                   # User Profile (通过 mem0 metadata 管理)
│
├── context/                         # Context Engineering 模块
│   ├── __init__.py
│   ├── manager.py                   # Token 预算管理 + 上下文组装
│   ├── compressor.py                # 上下文压缩/摘要 (DeepSeek-chat)
│   └── prompt_builder.py            # Prompt 模板构建
│
├── schemas/                         # 共享数据模型
│   ├── __init__.py
│   ├── models.py                    # AgentMessage, IntentResult, ToolResult 等
│   └── state.py                     # LangGraph State 定义 (TypedDict)
│
└── utils/                           # 工具层
    ├── __init__.py
    └── llm_client.py                # LLM 统一客户端 (ChatOpenAI → DeepSeek, get_llm(purpose) 工厂)

tests/
├── unit/                            # 单元测试
│   ├── test_intent_classifier.py
│   ├── test_tool_registry.py
│   ├── test_chunker.py
│   ├── test_session_memory.py
│   └── test_context_manager.py
├── module/                          # 模块集成测试
│   ├── test_rag_pipeline.py
│   ├── test_tool_pipeline.py
│   └── test_memory_pipeline.py
├── e2e/                             # 端到端测试
│   ├── test_single_turn.py
│   ├── test_multi_turn.py
│   └── test_multi_intent.py
└── fixtures/                        # 测试数据
    ├── sample_documents/
    ├── sample_queries.json
    └── expected_intents.json

data/                                # 运行时数据 (gitignore)
├── documents/                       # HR 文档存放目录
├── chroma_db/                       # ChromaDB 持久化目录
└── sessions/                        # Session Memory JSON 持久化

- **模块依赖关系图（无循环）**：
main.py (CLI loop)
  └─→ agents/graph.py (LangGraph StateGraph)
        └─→ agents/orchestrator
              ├─→ intent/classifier → utils/llm_client
              ├─→ intent/router (conditional_edges)
              ├─→ agents/rag_agent → rag/* → utils/llm_client
              │     ├─→ rag/embedder (BGE-M3 via HF API)
              │     ├─→ rag/vectorstore (ChromaDB)
              │     └─→ rag/reranker (BGE-Reranker via HF API)
              ├─→ agents/tool_agent → tools/* → mcp/*
              ├─→ agents/memory_agent
              │     ├─→ memory/session (dict + JSON)
              │     └─→ memory/longterm (mem0 Cloud)
              └─→ context/* → utils/llm_client

# 6 Functional Testing Plan
## 6.1 Unit Tests
| 测试对象 | 测试内容 | 验证方式 |
|----------|----------|----------|
| Intent Classifier | 输入已知查询，验证分类出正确的 intent label 和合理的 confidence | 准备 sample_queries.json + expected_intents.json，断言 label 匹配率 ≥ 90% |
| Intent Router | 输入各种 IntentResult，验证路由到正确的 Agent | 断言路由映射表正确 |
| Tool Registry | 注册/查找/列举工具，验证 schema 校验逻辑 | 注册 mock 工具，验证可查找、schema 合法 |
| Tool Executor | 参数校验、执行成功/失败场景 | mock 工具返回值，断言结果格式化正确、异常被正确捕获 |
| Document Chunker | 输入长文本，验证 chunk 大小、overlap、metadata 保留 | 断言 chunk token 数 ≤ max_size、overlap 区域正确 |
| Session Memory | append / read / trim / summarize 操作 | 断言消息列表正确、trimming 后长度在预算内 |
| Context Manager | 各段落 token 超限时的压缩和截断行为 | 断言最终 prompt token 数 ≤ 预算 |
| Prompt Builder | 输入各模块结果，验证 prompt 结构正确 | 断言 prompt 包含所有必要段落且顺序正确 |

## 6.2 Module Integration Tests
| 测试对象 | 测试内容 | 验证方式 |
|-----------|----------|----------|
| RAG Pipeline | 文档加载 → 切片 → 嵌入 → 存储 → 检索 → 重排序 全流程 | 准备样例 HR 文档，端到端执行后验证检索结果包含预期内容 |
| Tool Pipeline | 工具注册 → LLM 选择 → 参数生成 → 执行 → 结果格式化 | 注册 mock 工具，模拟 LLM 选择，验证执行结果正确 |
| MCP Pipeline | Client 发现工具 → 调用 → 返回结果 | 启动client发现 MCP Server，验证 tools/list 和 tools/call 正常工作 |
| Memory Pipeline | Session 写入 → 摘要 → 长期记忆写入 → 长期记忆检索 | 模拟多轮对话，验证: 会话历史正确 → 摘要生成合理 → 长期记忆可检索 |
| LangGraph Pipeline | StateGraph 编译 → 输入消息 → Intent Node → Router Edge → Agent Node → Response | 验证图的执行流程与预期路由一致 |

## 6.3 End-to-End Tests
| 测试场景 | 验证内容 |
|-----------|----------|
| Single-turn policy Q&A | 用户问 "年假政策是什么" → 系统正确识别为 policy_qa → RAG 检索到相关文档 → 生成包含政策信息的回答 |
| Single-turn employee lookup | 用户问 "查询张三的部门" → 识别为 employee_lookup → Tool Agent 调用 lookup_employee → 返回正确信息 |
| Multi-turn conversation | 多轮对话中保持上下文：先问 "年假政策"，再问 "那病假呢" → 系统理解 "那" 指代 "政策"，正确检索 |
| Multi-intent | "查一下我的假期余额，顺便告诉我请假流程" → 识别两个意图 → 分别路由执行 → 聚合结果 |
| Fallback handling | 输入完全无关的问题 → 系统识别为 unknown → 返回友好的回退提示 |
| Memory persistence | Session A 中告知 "我是研发部的" → Session B 中问 "我的部门" → 长期记忆正确召回 |
| Reflexion quality | 对 policy_qa 启用 Reflexion → 验证低质量回答被重新生成，最终回答质量提升 |

# 6.4 测试基础设施
- LLM Mock：测试中使用 unittest.mock.patch mock ChatOpenAI 调用，返回预设响应，避免 API 依赖和费用
- mem0 Mock：unit/module tests 中 mock MemoryClient，避免依赖 mem0 Cloud
- HuggingFace Mock：unit tests 中 mock embedding/reranker API 调用
- LangGraph 测试：使用 LangGraph 的 graph.invoke() 直接执行图，检查 State 变化
- 测试数据：tests/fixtures/ 包含样例 HR 文档、预设查询和预期结果
- CI 兼容：unit tests 和 module tests 使用 mock，可在 CI 中运行；e2e tests 可选使用真实 LLM

# 7 Future Extensions
| 方向 | 描述 | 架构准备 |
|------|------|----------|
| 更多知识库 | 接入更多 HR 文档类型（合同模板、培训材料、绩效表单） | RAG Loader 已支持多格式，Vector Store 支持多 collection，新增 Loader 即可 |
| 更多 Agent | 如 Approval Agent（审批流）、Analytics Agent（HR 数据分析）、Onboarding Agent） | BaseAgent 抽象类 + Router 配置化，新增 Agent 只需继承基类 + 注册路由规则 |
| Workflow Automation | 多步骤 HR 流程自动化（如请假审批全流程） | 在 Orchestrator 中引入 workflow engine，定义 DAG 式多步骤执行计划 |
| Skills 设计 | 为典型 HR 任务封装可复用 Skill（如"请假查询 Skill"包含意图模板+工具调用+回复模板） | Skill = intent template + tool chain + response template，可注册到 Router |
| 多模态 | 支持图片/表格解析（如工资条 OCR） | 在 Document Loader 中扩展 OCR/图表解析能力 |
| 权限控制 | 不同角色用户可访问不同工具和文档 | Tool Registry 增加权限标记，RAG 检索增加权限过滤 |
| 异步 Agent 通信 | 大规模部署时改为消息队列驱动 | AgentMessage 协议已统一，替换传输层即可 |
| 对话评估 | 自动评估回答质量 | 引入 LLM-as-judge 打分机制 |
| 多语言支持 | 支持中英文混合查询 | Embedding 模型选择支持多语言的版本，prompt 增加语言检测 |

