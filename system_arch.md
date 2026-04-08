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
│           Load Context (Session + Long-term Memory)      │
│                 + Intent Hints 生成                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│       决策中心 Orchestrator (DeepSeek-Reasoner)          │
│                                                         │
│  LLM 自主推理：分析已有信息是否足够回答用户问题          │
│  ├─ 需要工具 → 调用 Tool Agent → 观察结果 → 回到决策中心│
│  ├─ 需要信息 → 调用 RAG Agent  → 观察结果 → 回到决策中心│
│  ├─ 需要记忆 → 调用 Memory Agent→ 观察结果 → 回到决策中心│
│  ├─ 信息足够 → 生成最终答案                              │
│  └─ 需要澄清 → 追问用户                                 │
└──┬──────────────┬──────────────┬─────────────┬──────────┘
   │              │              │             │
   ▼              ▼              ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│ RAG Agent│ │Tool Agent│ │ Memory   │ │  Response   │
│          │ │          │ │ Agent    │ │  Generator  │
│ Retrieval│ │ Internal │ │ Session  │ │  / Clarify  │
│ Ranking  │ │ MCP Tools│ │ Longterm │ │             │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬─────┘
     │            │            │              │
     ▼            ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐  Post-process
│VectorDB  │ │ Tool     │ │ mem0     │  (Session +
│(ChromaDB)│ │ Registry │ │ Cloud    │   LTM 写入)
└──────────┘ │ MCP Svr  │ └──────────┘
             └──────────┘

- **框架映射**：
  - Orchestrator = LangGraph StateGraph，定义全局执行流程图
  - 决策中心 = Orchestrator Node（DeepSeek-Reasoner），自主决定下一步动作
  - Specialist Agent = Graph 中的执行 Node，被决策中心按需调用
  - 循环机制 = conditional_edge 从 Specialist Agent 执行完毕后回到决策中心，由 Reasoner 判断是否继续
  - 状态管理 = LangGraph State（TypedDict），在 Node 之间传递和更新
  - Skill 支持 = 可扩展技能系统，元数据预加载 + 意图驱动路由（首次检测→直达提议，确认后→完整内容注入 Orchestrator prompt 指导 ReAct 循环）



## 2.2 Agent Architecture
| Agent | 职责 | 输入 | 输出 | 依赖模块 |
|------|------|------|------|------|
| Orchestrator（决策中心） | 接收用户输入、加载上下文、**自主推理决定下一步动作**（调用工具/RAG/记忆/生成回答/追问）、聚合结果、生成最终回复 | 用户消息 + 会话上下文 + 意图提示 + 已收集的中间结果 | 下一步动作指令 或 最终回复 | Context、LLM Client（Reasoner） |
| RAG Agent | 基于 HR 文档库执行检索增强生成 | 查询文本 + 检索过滤条件 | 检索结果 + 上下文片段（结构化中间数据） | RAG Pipeline、Context |
| Tool Agent | 执行内部工具调用和 MCP 工具调用 | 工具名称 + 参数 | 工具执行结果（结构化） | Tool Registry、MCP Client |
| Memory Agent | 管理会话记忆写入/读取、长期记忆提取/检索 | 对话历史 / 检索查询 | 相关记忆上下文 | Session Memory、Long-term Memory |

**子 Agent 输出原则**：
- 所有 Specialist Agent（RAG / Tool / Memory）**只返回结构化中间结果**，写入 LangGraph State 对应字段
- **不生成用户可见的最终回答**
- 最终用户回复**仅由 Orchestrator 决策中心**在判定信息充足后生成，基于所有中间结果统一组装 prompt 后调用 LLM
- 好处：保证回复风格一致、便于多步推理结果聚合、Context Engineering 集中管理

- **Agent 执行模式 — Orchestrator 驱动循环**：
  - **核心区别**: 不再由固定路由表决定调用哪些 Agent，而是由 Orchestrator（DeepSeek-Reasoner）在每一步自主推理决定下一个动作
  - 循环流程：Orchestrator 分析当前 State（用户问题 + 已收集信息 + 上下文）→ 自主决策下一步动作 → 执行对应 Specialist Agent → 观察结果写回 State → 回到 Orchestrator 继续推理
  - 终止条件：Orchestrator 判定已有足够信息回答用户问题 → 生成最终答案；或判定信息不足且无法获取 → 生成澄清追问
  - 设置最大循环次数（如 5 次），避免无限循环

- **Reflexion 模式（可选扩展，用于关键路径）**：
  - 在 Orchestrator 生成最终回答后添加自我反思
  - 反思 prompt 要求 LLM 评估回答质量（准确性、完整性、相关性）
  - 评估不通过 → 回到 Orchestrator 决策循环，携带反思反馈继续收集信息
  - 设置最大反思次数（如 2 次），避免无限循环

**LangGraph State 定义**：
AgentState {
  messages: Annotated[list[BaseMessage], add_messages] // LangGraph 消息列表（对话历史）
  intent_hints: str | None                            // 意图提示（轻量分析，供 Orchestrator 参考）
  orchestrator_action: str | None                     // 当前决策动作，可由 intent_hints_node（"skill_propose"）或 orchestrator_decision_node（"rag"/"tool"/"memory"/"answer"/"clarify"）设置
  orchestrator_reasoning: str | None                  // Orchestrator 决策推理过程
  rag_results: RetrievalResult | None                 // RAG 检索结果
  tool_results: list[ToolResult]                      // 工具执行结果
  session_context: list[str]                          // 当前会话上下文/短期记忆
  memory_context: list[str]                           // 检索到的长期记忆
  user_profile: dict[str, Any] | None                 // 用户画像/偏好信息
  final_response: str | None                          // 最终生成的响应内容
  loop_count: int                                     // 当前决策循环次数
  max_loops: int                                      // 最大循环次数（默认 5）
  session_id: str                                     // 会话唯一标识
  user_id: str                                        // 用户 ID
  active_skill_content: str | None                    // 完整 SKILL.md 内容（intent_hints_node 确认后加载，当次 invoke 内传递给 Orchestrator prompt，不跨 turn 持久化）
}



## 2.3 Data Flow（端到端）
**完整的请求处理流程（每步标注数据形态）**：
① 用户输入 (raw text) — CLI input()
   │
② 加载上下文 + 意图提示生成
   ├─→ Session Memory(本地): 获取当前会话历史 (message list)
   ├─→ mem0 Cloud: 检索相关长期记忆 (memory snippets) + 用户画像
   └─→ Intent Hints 生成(DeepSeek-chat, 轻量分析)
       输入: 用户消息 + 会话上下文 + 长短期记忆 + Skill 元数据列表
       处理: LLM 结合 session 上下文判断 Skill 状态（首次检测 / 用户确认 / 用户拒绝 / 执行中）
       输出: 意图提示文本 + 可选路由决策（skill_propose → 跳过 Orchestrator 直达回复）
   │
   ├─→ Intent Router（条件路由）
   │     ├─ orchestrator_action="skill_propose" → 跳过③ → 直接进入④生成技能提议消息
   │     └─ 其他 → 进入③ Orchestrator 决策循环（若 active_skill_content 已加载，则带入 Skill 指令）
   │
③ Orchestrator 决策循环 (DeepSeek-Reasoner)
   │  输入: 用户消息 + 意图提示 + 已收集的中间结果 + 上下文 + 激活的 Skill 内容（如有）
   │  Reasoner 自主推理，输出下一步动作：
   │
   │  ┌─────────────────────────────────────────────────┐
   │  │ LOOP（最多 max_loops 轮）:                       │
   │  │                                                   │
   │  │  Orchestrator 分析当前 State，决策下一步：         │
   │  │  ├─ action="rag"     → 执行 RAG Agent             │
   │  │  │   输入: query + metadata_filter                │
   │  │  │   输出: retrieved_chunks[] → state.rag_results │
   │  │  │   → 结果回到 Orchestrator                      │
   │  │  │                                                │
   │  │  ├─ action="tool"    → 执行 Tool Agent            │
   │  │  │   输入: tool_name + parameters                 │
   │  │  │   输出: tool_result → state.tool_results       │
   │  │  │   → 结果回到 Orchestrator                      │
   │  │  │                                                │
   │  │  ├─ action="memory"  → 执行 Memory Agent          │
   │  │  │   输入: memory_query                           │
   │  │  │   输出: relevant_memories[]                    │
   │  │  │   → 结果回到 Orchestrator                      │
   │  │  │                                                │
   │  │  ├─ action="answer"  → 信息充足，生成最终回复     │
   │  │  │   → 退出循环，进入④                            │
   │  │  │                                                │
   │  │  └─ action="clarify" → 信息不足且无法获取         │
   │  │      → 退出循环，生成澄清问题                     │
   │  └─────────────────────────────────────────────────┘
   │
④ 回复生成 (DeepSeek-chat / DeepSeek-reasoner)
   │  输入: 所有中间结果 + 会话历史 + 记忆 + 系统 prompt
   │  处理: Context Assembly (token counting → compression → prompt building)
   │  输出: final_response text
   │
⑤ Post-processing
   ├─→ Session Memory(本地): 追加当轮对话 (user_msg + assistant_msg) → JSON 持久化
   ├─→ mem0 Cloud: 判断是否需要写入长期记忆
   │   (提取关键事实 → mem0.add() 写入, mem0 自动处理去重/重要性评估)
   └─→ 返回响应给用户(CLI print)

**数据流完整性验证**：
✅ 用户输入 → 上下文加载 + 意图提示 → Orchestrator 自主推理循环 → 回复生成 → 记忆更新：闭环
✅ Orchestrator 在每一步决策中都能观察到所有已收集的中间结果
✅ 所有模块有明确输入来源和输出，无孤立模块、无断裂调用链

# 3 Core Modules
## 3.1 Intent Hints（意图提示生成）
**目标**：对用户输入进行轻量预分析，生成自然语言意图提示，供 Orchestrator 决策中心参考。

**子模块**：
| 子模块 | 职责 |
|------|------|
| Intent Analyzer | LLM-based 轻量分析，输出自然语言意图提示 |

**意图类别参考（供 Intent Analyzer prompt 使用）**：
| Intent Label | 描述 |
|-------------|------|
| policy_qa | HR 政策问答 |
| process_inquiry | HR 流程咨询 |
| employee_lookup | 员工信息查询 |
| memory_recall | 回忆之前的对话内容 |
| skill:\<skill_name\> | 匹配到可用技能（动态，由 SkillLoader 元数据注入） |
| chitchat | 闲聊/问候 |
| unknown | 无法识别 |

**关键技术考虑**：
- 分析方式：使用 LLM prompt（DeepSeek-chat），输入用户消息 + 上下文，输出自然语言提示
- Intent Hints 输出示例：
  - "理由：用户想了解年假政策。意图为：policy_qa。"
  - "理由：用户想查询张三的部门信息。意图为：employee_lookup。"
  - "理由：用户想要了解请假政策和请假流程。意图为：policy_qa + process_inquiry。"
  - "理由：用户想在知乎上搜索文章并获取摘要。意图为：skill:zhihu_crawl。"
- Multi-intent 处理：Intent Analyzer 在提示中描述多个可能的意图，由 Orchestrator 自行决定处理顺序和方式
- Fallback：当分析不确定时，Intent Hints 可提示"意图不明确，建议向用户澄清"，但最终由 Orchestrator 决定是否澄清

**Skill 元数据注入**：
- SkillLoader 启动时扫描 `skills/` 目录，提取每个 Skill 的 `name` + `description`（YAML 前缀）
- Intent Analyzer prompt 中动态追加所有 Skill 元数据作为额外意图类别：
  ```
  可用技能（当用户请求明确匹配某个技能时，使用 skill:<技能名> 作为意图标签）：
  - zhihu_crawl: 根据用户的搜索输入在知乎上搜索 x 篇文章并生成摘要返回给用户
  - ...
  ```
- 仅加载元数据（name + description），不加载完整 SKILL.md，节省 token

**启动流程**：
- `compile_graph()` 中调用 `SkillLoader.scan()` 扫描 `skills/` 目录并缓存元数据列表
- `intent_hints_node` 启动时获取缓存的元数据列表，注入 Intent Analyzer prompt

**技术选型映射**：
- LLM：使用 DeepSeek-chat 进行意图提示生成（轻量、低成本）

**Intent 路由（conditional edge after intent_hints_node）**：
- `intent_hints_node` 在检测到首次 Skill 匹配时，直接设置 `orchestrator_action = "skill_propose"`
- Graph 中 `intent_hints_node` 后接 `_intent_router` 条件边：
  - `orchestrator_action == "skill_propose"` → `generate_response_node`（跳过 Orchestrator 循环，直接生成提议消息）
  - 否则 → `orchestrator_decision_node`（正常流程，若 `active_skill_content` 已加载则注入 Orchestrator prompt）
- 这避免了 Skill 提议经过 Orchestrator Reasoner 的不必要推理开销

**输入**：用户消息 (str) + 会话上下文 (session messages) + 长短期记忆 + Skill 元数据列表
**输出**：intent_hints (str) + 可选 orchestrator_action ("skill_propose") + 可选 active_skill_content (str)
**依赖**：LLM Client、SkillLoader（元数据 + 按需加载完整内容）
**被调用方**：Orchestrator Agent（在决策循环前调用一次）

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
- 支持按 category 分组（employee_tools, process_tools）
**MVP 内置工具**：
| 工具名 | 类别 | 功能 |
|------|------|------|
| lookup_employee | employee | 根据姓名/工号查询员工信息 |
| get_leave_balance | employee | 查询员工假期余额 |
| list_hr_processes | process | 列出可用 HR 流程 |
| get_process_steps | process | 查询具体流程步骤 |

**关键技术考虑**：
- Tool Schema：每个工具以 JSON Schema 定义输入参数，LLM 根据 schema 生成调用参数。Schema 同时用于参数校验
- 优先使用模型 Native Function Calling 。通过 API tools 参数注入 Schema，利用模型后训练（Post-training）的工具选择能力，而非纯文本 Prompt 引导。
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
**目标**：通过 **MCP (Model Context Protocol)** 接入在**本地运行**的 MCP Server 提供的工具，实现工具的动态发现与调用。  
系统仅作为 **MCP Client**，从本地 MCP 服务发现并调用工具。

**子模块**：
| 子模块 | 职责 |
|---|---|
| MCP Client | 连接本地运行的 MCP Server，发现并调用工具 |
| MCP Tool Adapter | 将 MCP 工具适配为系统内部 Tool Registry 的统一格式 |

**关键技术考虑**：
- **MCP Client**
  - 使用 langchain-mcp-adapters 与本地运行的 MCP Server 连接：
  - 支持 MCP 标准 transport：**stdio、SSE**
  - MCP Client 负责：建立连接、调用 MCP API、工具发现、工具执行

- **MCP Tool Discovery**
  - MCP Client 启动时先从本地 MCP Server 获取所有可用工具
  - 将 MCP 工具自动注册到 Tool Registry（与内置工具统一管理）
  - MCP 工具元数据应展示：`name`、`description`、`JSON Schema`

**Tool Invocation Flow**：
Tool Agent 选择 MCP 工具  
→ MCP Client 构造 `tools/call` 请求  
→ 本地 MCP Server 接收并路由到具体 handler  
→ Handler 执行业务逻辑  
→ 返回结构化结果  
→ MCP Client 解析结果并传回 Tool Agent

- **与内置工具的统一**
  - MCP 工具和内置工具在 Tool Registry 中统一注册，Tool Agent 无需区分来源，但 Registry 中标记 `source: "internal" | "mcp"`

**技术选型映射**：
- **MCP 适配**：使用 `langchain-mcp-adapters` 库将 MCP 工具转换为 LangChain Tool 对象，无缝接入 LangGraph ReAct Agent
- **MCP Server**：不自行开发云端服务，而是**下载 MCP Server 到本地并在本地运行**
- **配置方式**：MCP Server 连接信息通过 config 文件管理（如 `server_path`、`transport_type`、`auth` 等）

**输入**：MCP tool name + parameters  
**输出**：MCP tool result（与内置工具统一的 `ToolResult` 格式）  
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
  - MVP 支持格式：PDF、DOCX、Markdown、纯文本
  - 提取文档元数据：文件名、标题、创建日期、文档类别（policy/SOP）
  - 增量加载：记录已索引文档的 hash，仅处理变更文档

- **文档切片（Chunking）**：
  - 策略：Recursive Character Splitting（按段落 → 句子 → 字符递归切分）
  - chunk_size: 256 tokens（MVP），chunk_overlap: 32 tokens
  - 保留 chunk 级元数据：所属文档、章节标题、页码、chunk 序号
  - 考虑：HR 文档通常有清晰的章节结构，优先按标题/章节边界切分

- **Embedding**：
  - 模型：BAAI/bge-m3, 向量维度：1024 维
  - 调用方式：通过 HuggingFace Inference API
  - embedding 维度与 ChromaDB 配置一致

- **向量数据库**：
  - MVP 选择：ChromaDB
  - 存储内容：chunk text + embedding vector + metadata dict
  - Collection 按文档类别分组（policy_collection, sop_collection）

- **Metadata 设计**：
{
  "source": "employee_handbook.pdf",
  "category": "policy",        // policy | sop 
  "section": "Leave Policy",
  "page": 12,
  "chunk_index": 3,
  "last_updated": "2026-01-15"
}
  - 检索时可按 category、section 做 pre-filter

- **Hybrid Search**：
  - 向量搜索：语义相似度，捕获同义/近义表述
  - 关键词搜索：BM25（使用 rank_bm25 库），捕获精确术语匹配
  - 两者并行检索后，利用融合策略：Reciprocal Rank Fusion (RRF)，结合两路结果 

- **Retrieval 策略**：
  - 默认：hybrid search + reranking
  - 当用户查询包含明确文档名/类别时：先 metadata pre-filter，再向量检索
  - 当检索结果 relevance score 均低于阈值时：返回"未找到相关信息"

- **Reranking**：
  - 模型：BAAI/bge-reranker-v2-m3（cross-encoder，多语言支持）
  - 调用方式：通过 HuggingFace Inference API（云端推理）
  - 输入：query + candidate chunks（hybrid search 结果）
  - 输出：重排序后的 chunks + relevance scores 

- **Context Filtering**：
  - Reranking 后，根据 token 预算（分配给 RAG 上下文的 token 数）截取 top-N chunks
  - 相邻 chunk 合并（若来自同一文档同一节）以保持语义连贯


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
  - 可以在CLI界面选择相应的 session_id 继续对话
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

## 3.8 Skill System（技能系统）
**目标**：支持可扩展的 Skill（技能）机制，让系统能够遵循预定义的结构化工作流完成复杂多步任务（如：批量抓取网页并摘要）。Skill 通过 Token 节省策略（元数据预加载 + 确认后完整加载）降低日常对话的 token 消耗。

**核心概念**：
- **Skill vs Tool**：Tool 是原子操作（单次调用即返回），Skill 是多步结构化工作流（指导 Orchestrator 在 ReAct 循环中按步骤完成复杂任务）
- **Token 节省策略**：日常对话仅加载 Skill 元数据（name + description）；完整 SKILL.md 内容仅在用户确认使用后才注入 Orchestrator prompt

**子模块**：
| 子模块 | 职责 |
|--------|------|
| SkillLoader | 扫描 skills 目录，解析 SKILL.md 的 YAML 前缀提取元数据，缓存元数据列表，按名称加载完整内容 |

**SKILL.md 格式规范**：
```markdown
---
name: skill_name          # 唯一标识符（必须与文件夹名一致）
description: '一句话描述'  # 供意图识别使用的简短描述
---

**SkillMetadata 数据模型**：
```python
@dataclass
class SkillMetadata:
    name: str          # Skill 唯一标识（= 文件夹名）
    description: str   # 简短描述（注入到意图识别 prompt）
    path: str          # SKILL.md 文件的绝对路径
```

**Skill 生命周期（三阶段）**：

**① 启动加载（仅元数据）**
- 系统启动时，`compile_graph()` 调用 `SkillLoader.scan()` 扫描 `skills/` 目录下所有子文件夹
- 查找并解析每个子文件夹中的 `SKILL.md` 文件的 YAML 前缀
- 提取 `name` 和 `description`，缓存为 `list[SkillMetadata]`

**② 首次检测 → 提议（跳过 Orchestrator）**
- `intent_hints_node` 中 Intent Analyzer prompt 动态注入 Skill 元数据列表（仅 name + description）
- LLM 结合 session 上下文判断：这是首次检测到可用 Skill（会话中无先前提议记录）
- 输出意图 `skill:<skill_name>` + 设置 `orchestrator_action = "skill_propose"`
- Graph 中 `_intent_router` 检测到 `skill_propose` → **直接路由到 `generate_response_node`**（跳过 Orchestrator 决策循环）
- `generate_response_node` 生成提议消息："检测到可以使用「{description}」技能来完成此任务。是否启用？"
- 提议消息写入 session 历史，**无需额外状态变量**

**③ 确认 → 执行（进入 Orchestrator + Skill 指令）**
- 下一轮对话，`intent_hints_node` 读取 session 上下文（包含上一轮的提议消息 + 用户回复）
- LLM 语义判断用户是否确认：
  - **确认**：输出 `skill:<name>` 意图 + 调用 `SkillLoader.load_content(name)` 加载完整 SKILL.md → 写入 `active_skill_content`
  - **拒绝**：不输出 skill 意图，按正常意图分析处理
- 确认后，`_intent_router` 检测到无 `skill_propose`（`orchestrator_action` 未设置）→ 正常进入 `orchestrator_decision_node`
- `orchestrator_decision_node` 的 prompt 注入完整 Skill 指令：
  ```
  ## 当前激活技能
  你正在执行「{skill_name}」技能，请严格遵循以下工作流完成任务：
  {skill_content}
  ```
- Orchestrator 在 ReAct 循环中遵循 Skill 的步骤完成任务 → `action="answer"` → 生成最终结果
- 若 Skill 任务需要多轮对话，每轮 `intent_hints_node` 从 session 上下文判断 Skill 仍在执行中 → 继续加载完整内容
- 若用户明显改变话题 → LLM 不再输出 skill 意图 → Orchestrator 不注入 Skill 指令 → 自然退出


**与现有模块的交互**：
| 模块 | 交互方式 |
|------|---------|
| Intent Analyzer | 注入 Skill 元数据到 prompt + 读取 session 上下文判断 Skill 状态 → 识别 `skill:xxx` 意图 |
| Orchestrator Decision | 当 `active_skill_content` 存在时，注入完整 Skill 指令到 prompt → 指导 ReAct 循环 |
| Tool Node | Skill 引用的 MCP 工具已在 Tool Registry 中注册（通过 MCP Client） |
| Session Memory | Skill 状态隐含在对话历史中（提议消息、用户确认），无需额外 metadata 字段 |
| Graph Router | `_intent_router`: `skill_propose` → `generate_response`；`_decision_router`: 正常动作路由 |


**输入**：Skill 名称（str）
**输出**：SkillMetadata（元数据）或 str（完整 SKILL.md 内容）
**依赖**：文件系统（skills/ 目录）
**被调用方**：Intent Hints Node（元数据注入 + 确认后加载完整内容）、Orchestrator（通过 State 接收 active_skill_content）

# 4 Multi-Agent Coordination
## 4.1 Orchestrator Agent（决策中心）
- **角色**：系统唯一入口和决策核心，负责自主推理和全局流程控制
- **核心机制**：Orchestrator 使用 DeepSeek-Reasoner 在每一步自主推理决定下一个动作，而非依赖固定路由表
- **职责**：
  1. 接收用户输入
  2. 加载 session context + 相关长期记忆
  3. 生成 Intent Hints（轻量意图提示）
  4. **进入决策循环**：分析当前 State（用户问题 + 意图提示 + 已收集的中间结果），自主推理出下一步动作
  5. 调度 Specialist Agent 执行（Tool / RAG / Memory），观察结果
  6. 循环 4-5 直到判定信息充足（或达到最大循环次数）
  7. 通过 Context Manager 组装 prompt，生成最终回复（或生成澄清问题）
  8. 更新 session memory + 触发长期记忆写入

**Orchestrator 决策 Prompt**：
每一轮循环中，Orchestrator 收到的输入包括：
- 用户原始消息
- Intent Hints（意图提示）
- 已有的中间结果（RAG 检索结果、工具调用结果、记忆检索结果）
- 可用的动作列表及说明
- 会话上下文

Orchestrator 输出结构化 JSON：
```json
{
  "reasoning": "推理过程...",
  "action": "rag" | "tool" | "memory" | "answer" | "clarify",
  "action_input": { ... }   // 动作参数（如 RAG 的 query、Tool 的 tool_name + params）
}
```

**技术实现**：
- Orchestrator 决策节点 = LangGraph StateGraph 中的核心 Node，使用 DeepSeek-Reasoner 进行推理
- 循环实现 = conditional_edge：Specialist Agent 执行完毕后回到 Orchestrator Node，由 Reasoner 再次决策
- 终止条件 = action 为 "answer" 或 "clarify" 时退出循环，进入回复生成节点
- 安全阀 = loop_count >= max_loops 时强制退出循环，基于已有信息生成最佳回复

## 4.2 Orchestrator 决策规则
不再使用固定路由映射表，而是通过 Prompt 引导 Reasoner 自主推理：

**Prompt 中的决策指引（非硬编码规则）**：
- 当用户询问 HR 政策/制度/规定时，优先考虑 RAG 检索文档
- 当用户需要查询具体数据（员工信息、假期余额等）时，优先考虑调用工具
- 当用户提到"之前""上次""我们聊过"等时，考虑检索记忆
- 当 `active_skill_content` 存在时，严格遵循技能指令中的步骤完成任务
- 当已有信息足以回答用户问题时，直接生成回答
- 当信息不足且无法通过工具/RAG/记忆获取时，生成澄清问题
- 闲聊/问候类消息可直接回答，无需调用任何 Agent

**Multi-intent 处理**：
- Orchestrator 自然地通过多轮循环处理多意图查询
- 例如"查一下假期余额，再告诉我请假流程"：
  - 第1轮：Reasoner 判断需要调用工具查假期余额 → 执行 Tool Agent
  - 第2轮：观察到余额结果，判断还需要检索请假流程 → 执行 RAG Agent
  - 第3轮：观察到所有结果充足 → action="answer"，生成综合回复

**Skill 处理流程（跨 turn）**：
- Turn 1：“帮我在知乎搜3篇关于机器学习的文章” → intent_hints: skill:zhihu_crawl + orchestrator_action="skill_propose" → _intent_router 跳过 Orchestrator → generate_response 生成提议消息
- Turn 2：“好的” → intent_hints 读取 session 上下文，LLM 判断用户确认 → skill:zhihu_crawl + 加载完整 SKILL.md → _intent_router 正常进入 Orchestrator → Orchestrator prompt 包含 Skill 指令 → ReAct 循环（搜索 → 抓取 → 摘要）→ action="answer"
- Turn 2 (拒绝)：“不用了” → intent_hints LLM 判断用户拒绝 → 正常意图分析 → Orchestrator 正常处理

**Fallback 策略**：
- Orchestrator 在推理过程中自行判断信息是否充足
- 当 RAG 检索无结果、工具调用失败时，Reasoner 会在下一轮推理中感知到并调整策略
- 最终无法获取足够信息 → action="clarify"，生成用户友好的澄清问题
- 达到最大循环次数仍信息不足 → 基于已有信息生成最佳回复 + 提示用户可联系 HR

## 4.3 错误处理与降级
- 任何 specialist agent 失败 → Orchestrator 捕获异常 → 生成降级响应（告知用户该功能暂不可用）
- LLM 调用失败 → 重试 1 次 → 仍失败则返回预设错误消息
- 全局 trace_id 贯穿调用链，便于日志追踪

## DeepSeek 双模型使用策略
| 场景                                      | 模型                 | 理由                                           |
|-------------------------------------------|----------------------|------------------------------------------------|
| Intent Hints 生成                         | deepseek-chat        | 轻量意图提示，不需要深度推理                    |
| Orchestrator 决策循环                      | deepseek-reasoner    | 核心推理节点，需要分析已有信息并自主决策下一步     |
| Tool Selection & Calling                  | deepseek-chat        | 只有chat模式支持native function calling             |
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
├── intent/                          # 意图提示模块
│   ├── __init__.py
│   └── analyzer.py                  # LLM-based 意图提示生成器（轻量分析，输出自然语言 hints）
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
│   ├── selector.py                  # 工具选择器 (LLM Native Function Calling)
│   ├── executor.py                  # 工具执行器 (参数校验+调用+格式化)
│   └── hr_tools/                    # HR 专用内置工具 (LangChain @tool 装饰器)
│       ├── __init__.py
│       ├── employee_lookup.py       # 员工信息查询
│       └── process_tools.py         # HR 流程工具
│
├── skills/                          # 技能系统模块
│   ├── loader.py                    # SkillLoader: 元数据扫描 + 完整内容加载
│   ├── zhihu_crawl/                 # 知乎搜索与摘要技能
│   │   └── SKILL.md
│   └── <future_skill>/             # 其他技能（按需添加）
│       └── SKILL.md
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
│   ├── test_intent_analyzer.py
│   ├── test_orchestrator_decision.py
│   ├── test_tool_registry.py
│   ├── test_chunker.py
│   ├── test_session_memory.py
│   ├── test_context_manager.py
│   └── test_skill_loader.py
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
              ├─→ intent/analyzer (意图提示生成) → utils/llm_client
              │     └─→ skills/loader (Skill 元数据注入 + 确认后加载完整内容)
              ├─→ orchestrator_decision (决策循环, Reasoner) → utils/llm_client
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
| Intent Analyzer | 输入已知查询，验证生成的意图提示包含正确的意图方向和关键实体 | 准备 sample_queries.json，断言提示文本包含预期关键词（意图类型、实体名等） |
| Orchestrator Decision | 输入不同 State（含不同中间结果），验证 Reasoner 输出正确的 action | mock Reasoner 返回值，断言 action 类型和参数正确 |
| Tool Registry | 注册/查找/列举工具，验证 schema 校验逻辑 | 注册 mock 工具，验证可查找、schema 合法 |
| Tool Executor | 参数校验、执行成功/失败场景 | mock 工具返回值，断言结果格式化正确、异常被正确捕获 |
| Document Chunker | 输入长文本，验证 chunk 大小、overlap、metadata 保留 | 断言 chunk token 数 ≤ max_size、overlap 区域正确 |
| Session Memory | append / read / trim / summarize 操作 | 断言消息列表正确、trimming 后长度在预算内 |
| Context Manager | 各段落 token 超限时的压缩和截断行为 | 断言最终 prompt token 数 ≤ 预算 |
| Prompt Builder | 输入各模块结果，验证 prompt 结构正确 | 断言 prompt 包含所有必要段落且顺序正确 |
| SkillLoader | 扫描 skills 目录、解析 YAML 元数据、加载完整内容 | 断言元数据列表正确解析、完整内容可按名称加载、缺失文件处理正确 |

## 6.2 Module Integration Tests
| 测试对象 | 测试内容 | 验证方式 |
|-----------|----------|----------|
| RAG Pipeline | 文档加载 → 切片 → 嵌入 → 存储 → 检索 → 重排序 全流程 | 准备样例 HR 文档，端到端执行后验证检索结果包含预期内容 |
| Tool Pipeline | 工具注册 → LLM 选择 → 参数生成 → 执行 → 结果格式化 | 注册 mock 工具，模拟 LLM 选择，验证执行结果正确 |
| MCP Pipeline | Client 发现工具 → 调用 → 返回结果 | 启动client发现 MCP Server，验证 tools/list 和 tools/call 正常工作 |
| Memory Pipeline | Session 写入 → 摘要 → 长期记忆写入 → 长期记忆检索 | 模拟多轮对话，验证: 会话历史正确 → 摘要生成合理 → 长期记忆可检索 |
| LangGraph Pipeline | StateGraph 编译 → 输入消息 → Load Context → Intent Hints → Orchestrator 决策循环 → Agent Node → 回到 Orchestrator → Response | 验证图的执行流程与决策循环行为一致 |

## 6.3 End-to-End Tests
| 测试场景 | 验证内容 |
|-----------|----------|
| Single-turn policy Q&A | 用户问 "年假政策是什么" → 系统正确识别为 policy_qa → RAG 检索到相关文档 → 生成包含政策信息的回答 |
| Single-turn employee lookup | 用户问 "查询张三的部门" → 识别为 employee_lookup → Tool Agent 调用 lookup_employee → 返回正确信息 |
| Multi-turn conversation | 多轮对话中保持上下文：先问 "年假政策"，再问 "那病假呢" → 系统理解 "那" 指代 "政策"，正确检索 |
| Multi-intent | "查一下我的假期余额，顺便告诉我请假流程" → Orchestrator 自主推理，多轮循环分别调用 Tool Agent 和 RAG Agent → 聚合结果 |
| Fallback handling | 输入完全无关的问题 → 系统识别为 unknown → 返回友好的回退提示 |
| Memory persistence | Session A 中告知 "我是研发部的" → Session B 中问 "我的部门" → 长期记忆正确召回 |
| Skill propose & confirm | 输入 “帮我搜3篇知乎文章” → intent_hints 识别 skill:zhihu_crawl → _intent_router 跳过 Orchestrator → 提议用户确认 → 用户确认 → intent_hints 加载完整 Skill → Orchestrator 按工作流执行 |
| Skill reject | 输入匹配技能的请求 → 系统提议 → 用户拒绝 → intent_hints LLM 判断拒绝 → 正常意图分析，Orchestrator 正常处理 |
| Reflexion quality | 对 policy_qa 启用 Reflexion → 验证低质量回答被重新生成，最终回答质量提升 |

# 6.4 测试基础设施
- LLM Mock：测试中使用 unittest.mock.patch mock ChatOpenAI 调用，返回预设响应，避免 API 依赖和费用
- mem0 Mock：unit/module tests 中 mock MemoryClient，避免依赖 mem0 Cloud
- HuggingFace Mock：unit tests 中 mock embedding/reranker API 调用
- LangGraph 测试：使用 LangGraph 的 graph.invoke() 直接执行图，检查 State 变化
- 测试数据：tests/fixtures/ 包含样例 HR 文档、预设查询和预期结果
- CI 兼容：unit tests 和 module tests 使用 mock，可在 CI 中运行；e2e tests 可选使用真实 LLM

