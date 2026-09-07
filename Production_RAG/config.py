"""All configuration variables for the project are defined here."""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Absolute path to the Production_RAG directory — works regardless of working directory
_BASE_DIR = Path(__file__).parent

# Always load .env from Production_RAG/.env explicitly
load_dotenv(dotenv_path=_BASE_DIR / ".env")

# Load configuration from config.yaml
with open(_BASE_DIR / "config.yaml", "r") as f:
    _yaml_config = yaml.safe_load(f)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Paths — absolute so they work regardless of where the process is launched from
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", str(_BASE_DIR.parent / _yaml_config.get("paths", {}).get("data", "Production_RAG/data/hr_policies.txt")))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(_BASE_DIR.parent / _yaml_config.get("paths", {}).get("vector_store", "Production_RAG/data/faiss_index")))

# LLM CONFIG
LLM_MODEL_NAME = _yaml_config.get("llm", {}).get("model", "openai/gpt-oss-20b")
LLM_TEMPERATURE = _yaml_config.get("llm", {}).get("temperature", 0)

# EMBEDDING CONFIG
EMBEDDING_MODEL_NAME = _yaml_config.get("embeddings", {}).get("model", "jina-embeddings-v2-base-en")

# TEXT SPLITTING CONFIG
CHUNK_SIZE = _yaml_config.get("chunking", {}).get("chunk_size", 500)
CHUNK_OVERLAP = _yaml_config.get("chunking", {}).get("chunk_overlap", 50)

# RETRIEVAL CONFIG
TOP_K_RESULTS = _yaml_config.get("retrieval", {}).get("top_k", 3)

# SYSTEM PROMPT OR SYSTEM INSTRUCTIONS TO GUIDE THE LLM'S BEHAVIOR
SYSTEM_PROMPT = ("You are a friendly HR assistant working for Vertexon Solutions. "
                 "Always use the search_hr_policy tool to look up facts before answering. "
                 "If the answer isn't in the search results, say you don't know instead of guessing.")


# Function to check if API keys are set
def check_api_keys():
    """Check if the required API keys are set in the environment variables."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it in your environment variables.")
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please add it in your environment variables.")