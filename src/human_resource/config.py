"""配置管理模块。

集中管理 API keys、模型选择、阈值等配置项。
所有敏感信息从环境变量读取。
"""

import os
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
SESSIONS_DIR = DATA_DIR / "sessions"

# ── API Keys（从环境变量读取） ──────────────────────────────
os.environ["DEEPSEEK_API_KEY"] = "sk-"  
os.environ["HF_API_TOKEN"] = ""
os.environ["MEM0_API_KEY"] = "m0-"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")

# ── DeepSeek LLM 配置 ──────────────────────────────────────
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# 各场景使用的模型名称
MODEL_CONFIG: dict[str, str] = {
    "intent_hints": "deepseek-chat",
    "orchestrator_decision": "deepseek-reasoner",
    "tool_selection": "deepseek-chat",
    "rag_retrieval": "deepseek-chat",
    "context_compression": "deepseek-chat",
    "response_simple": "deepseek-chat",
    "response_complex": "deepseek-reasoner",
    "reflexion": "deepseek-reasoner",
    "memory_extraction": "deepseek-chat",
}

# LLM 通用参数
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

# ── Embedding 配置 ──────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

# ── Reranker 配置 ─────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── RAG 配置 ──────────────────────────────────────────────
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
VECTOR_SEARCH_TOP_K = 8
BM25_TOP_K = 8
RERANK_TOP_N = 3
RELEVANCE_SCORE_THRESHOLD = 0.1

# Collection 配置
POLICY_COLLECTION = "policy_collection"
SOP_COLLECTION = "sop_collection"
DEFAULT_COLLECTION = POLICY_COLLECTION

# 意图 → Collection 映射
INTENT_COLLECTION_MAP: dict[str, str] = {
    "policy_qa": POLICY_COLLECTION,
    "process_inquiry": SOP_COLLECTION,
}

# 子目录 → Collection 映射（索引用）
DIR_COLLECTION_MAP: dict[str, str] = {
    "policy": POLICY_COLLECTION,
    "sop": SOP_COLLECTION,
}

# ── Context Engineering 配置 ──────────────────────────────
TOKEN_BUDGET: dict[str, int] = {
    "system_prompt": 1500,
    "user_profile": 100,
    "relevant_memories": 200,
    "retrieved_context": 5000,
    "tool_results": 5000,
    "conversation_history": 2000,
    "current_message": 200,
}

# ── Session Memory 配置 ──────────────────────────────────
SESSION_MEMORY_BUDGET_TOKENS = 4000
SESSION_KEEP_RECENT_TURNS = 10
SESSION_COMPRESS_THRESHOLD = 5  # 超过此轮次数触发历史压缩

# ── Long-term Memory 配置 ─────────────────────────────────
MEMORY_SEARCH_TOP_K = 3  # mem0 检索返回结果数量
MEMORY_SEARCH_THRESHOLD = 0.6  # mem0 检索最低相似度阈值（API 默认 0.3）
MEMORY_WRITE_INTERVAL = 5  # 固定轮次触发写入间隔
MEMORY_IMPORTANCE_THRESHOLD = 0.5  # 记忆重要性阈值，仅 > 此值才写入

# ── MCP 配置 ──────────────────────────────────────────────
# MCP Server 连接信息，格式兼容 langchain-mcp-adapters MultiServerMCPClient
MCP_SERVERS: dict[str, dict] = {
    "universal-email": {
        "command": "node",
        "args": ["D:\\humanResource\\src\\human_resource\\mcp\\email-mcp\\index.js"],
        "transport": "stdio",
        "env": {
            "EMAIL_USER": "897287969@qq.com",
            "EMAIL_PASSWORD": "",
            "EMAIL_TYPE": "qq",
        },
    },
    "firecrawl/firecrawl-mcp-server": {
			"transport": "stdio",
			"command": "npx",
			"args": [
				"-y",
				"firecrawl-mcp@latest"
			],
			"env": {
				"FIRECRAWL_API_KEY": "fc-"
			},
		}
}
MCP_ALLOWED_TOOLS: list[str] = [
    "send_email",
    "firecrawl_search",
    "firecrawl_scrape",
]
MEMORY_WRITE_KEYWORDS: list[str] = [
    "记住这个", "记住", "请记住", "帮我记一下", "记下来",
    "remember this", "remember", "请记下",
]  # 用户显式触发长期记忆写入的关键词
# ── Skill 配置 ────────────────────────────────────────
SKILLS_DIR = Path(__file__).parent / "skills"
# ── Orchestrator 决策循环配置 ─────────────────────────────
MAX_ORCHESTRATOR_LOOPS = 10  # 决策循环最大次数

# ── Tool 配置 ─────────────────────────────────────────────
TOOL_EXECUTION_TIMEOUT = 30  # seconds
TOOL_MAX_RETRIES = 2
