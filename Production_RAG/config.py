"""All configuration variables for the project are defined here."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Absolute path to the Production_RAG directory — works regardless of working directory
_BASE_DIR = Path(__file__).parent

# Always load .env from Production_RAG/.env explicitly
load_dotenv(dotenv_path=_BASE_DIR / ".env")

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# ── LangSmith Observability ────────────────────────────────────────────────────
LANGSMITH_TRACING  = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "hr-policy-rag")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE_PATH    = os.getenv("DATA_FILE_PATH", str(_BASE_DIR / "data" / "hr_policies.txt"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(_BASE_DIR / "data" / "faiss_index"))

# ── Models & Hyperparameters ───────────────────────────────────────────────────
LLM_MODEL_NAME       = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.0"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-en")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS        = int(os.getenv("TOP_K_RESULTS", "3"))

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a friendly HR assistant working for Vertexon Solutions. "
    "Always use the search_hr_policy tool to look up facts before answering. "
    "If the answer isn't in the search results, say you don't know instead of guessing."
)

# ── GUARD MODEL ───────────────────────────────────────────────────────────────
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"


def check_api_keys():
    """Check if the required API keys are set in the environment variables."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it in your environment variables.")
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please add it in your environment variables.")