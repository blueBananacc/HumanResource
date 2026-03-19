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
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")

# ── DeepSeek LLM 配置 ──────────────────────────────────────
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# 各场景使用的模型名称
MODEL_CONFIG: dict[str, str] = {
    "intent_classification": "deepseek-reasoner",
    "tool_selection": "deepseek-reasoner",
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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
VECTOR_SEARCH_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_N = 5
RELEVANCE_SCORE_THRESHOLD = 0.3

# ── Context Engineering 配置 ──────────────────────────────
CONTEXT_WINDOW_TOKENS = 64000
MAX_PROMPT_TOKENS = 8000
TOKEN_BUDGET: dict[str, int] = {
    "system_prompt": 300,
    "user_profile": 100,
    "relevant_memories": 200,
    "retrieved_context": 1500,
    "tool_results": 500,
    "conversation_history": 2000,
    "current_message": 200,
}

# ── Session Memory 配置 ──────────────────────────────────
SESSION_MEMORY_BUDGET_TOKENS = 4000
SESSION_KEEP_RECENT_TURNS = 10

# ── Intent 配置 ──────────────────────────────────────────
INTENT_CONFIDENCE_THRESHOLD = 0.7

# ── Reflexion 配置 ────────────────────────────────────────
MAX_REFLEXION_RETRIES = 2

# ── Tool 配置 ─────────────────────────────────────────────
TOOL_EXECUTION_TIMEOUT = 30  # seconds
TOOL_MAX_RETRIES = 2
