# 🏢 HR Policy RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** assistant built with **LangChain**, **Jina AI Embeddings**, **FAISS**, **Groq LLM**, and **Streamlit** to deliver fast, contextual answers over company HR policy documents.

![HR Policy RAG Assistant Architecture](docs/architecture.png)

---

## 📌 Features

* **Document Ingestion:** Parses HR policy text documents and metadata cleanly.
* **Intelligent Chunking:** Configurable recursive character text splitting (`chunk_size=500`, `chunk_overlap=50`) preserving policy context.
* **Semantic Embeddings:** Powered by **Jina AI Embeddings v2** (`jina-embeddings-v2-base-en`).
* **Vector Store & Retrieval:** Fast local vector similarity search using **FAISS** with cached index management.
* **Ultra-Fast LLM Inference:** High-throughput responses via **Groq** (`openai/gpt-oss-120b`).
* **LangChain Agent & Tools:** ReAct agent architecture equipped with a dedicated HR policy retriever tool.
* **LangSmith Observability:** Integrated tracing to inspect and evaluate agent runs, tool calls, and latency.
* **Modern Web UI:** Dark-themed, responsive **Streamlit** interface with real-time status and sample queries.
* **Clean Configuration:** Centralized, type-safe [`config.py`](Production_RAG/config.py) supporting `.env` parameter overrides.
* **Comprehensive Test Suite:** 32 unit tests covering all modular components.

---

## 🔒 Security Notice

> ⚠️ **IMPORTANT:** Never commit real API keys or `.env` files to GitHub.
> 
> The project `.gitignore` ignores all local `.env` files and vector indexes. Copy `.env.example` to `.env` and configure your credentials locally.

---

## 📂 Project Structure

```text
hr-policy-rag-assistant/
├── Basic_RAG_experimental/       # Experimental RAG pipeline & research (Jupyter notebooks)
├── Production_RAG/               # Production modular RAG system
│   ├── app.py                    # Streamlit UI Entry Point
│   ├── main.py                   # Alternative CLI Entry Point
│   ├── pipeline.py               # End-to-end RAG orchestrator
│   ├── config.py                 # Centralized configuration & environment loader
│   ├── .streamlit/
│   │   └── config.toml           # Streamlit dark theme configuration
│   ├── agent/
│   │   └── agent.py              # LangChain agent builder
│   ├── tools/
│   │   └── tools.py              # Agent tool definitions (retriever search)
│   ├── ingestion/
│   │   └── loader.py             # Document loader
│   ├── chunking/
│   │   └── chunker.py            # Recursive text splitting logic
│   ├── embeddings/
│   │   └── embedder.py           # Jina AI embeddings wrapper
│   ├── vectordb/
│   │   └── vector_store.py       # FAISS build, save, and load utilities
│   ├── retrieval/
│   │   └── retriever.py          # Vector store retriever setup
│   ├── llm/
│   │   └── llm_client.py         # Groq Chat model setup
│   ├── prompts/
│   │   └── prompt_templates.py   # System prompts and templates
│   ├── utils/
│   │   ├── logger.py             # Centralized timestamped file & console logger
│   │   └── tracing.py            # LangSmith observability setup & validation
│   ├── data/
│   │   └── hr_policies.txt       # HR policy source document
│   ├── tests/                    # Pytest test suite (32 unit tests)
│   ├── pyproject.toml            # Python packaging & dependencies (uv compatible)
│   └── .env.example              # Environment variables template
└── README.md                     # Root project documentation
```

---

## 🚀 Quick Start

### Prerequisites
* **Python 3.10+** (Tested on Python 3.14.6)
* API Key for **Groq** ([console.groq.com](https://console.groq.com))
* API Key for **Jina AI** ([jina.ai](https://jina.ai))
* *(Optional)* API Key for **LangSmith** ([smith.langchain.com](https://smith.langchain.com)) for tracing

### 1. Clone the Repository
```bash
git clone https://github.com/narendhar11/hr-policy-rag-assistant.git
cd hr-policy-rag-assistant/Production_RAG
```

### 2. Install Dependencies

You can use the lightning-fast `uv` package manager or standard `pip`:

**Option A: Using `uv` (Recommended)**
```bash
# If uv is not installed: pip install uv
uv sync
```

**Option B: Using standard `pip`**
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Configure Environment Variables
Create your local `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Edit `Production_RAG/.env` with your API keys:
```env
# Groq LLM
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Jina AI Embeddings
JINA_API_KEY=your_jina_api_key_here

# (Optional) LangSmith Observability
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=hr-policy-rag
```

### 4. Run the Application

#### Web UI (Streamlit)
```bash
# With uv
uv run streamlit run app.py

# With standard virtual environment
streamlit run app.py
```

#### CLI Mode
```bash
# With uv
uv run python main.py

# With standard virtual environment
python main.py
```

---

## ⚙️ Configuration

All hyperparameters are centralized in [`Production_RAG/config.py`](Production_RAG/config.py) and can be overridden directly via environment variables in `.env`:

| Parameter | Default Value | Environment Variable | Description |
|---|---|---|---|
| `LLM_MODEL_NAME` | `openai/gpt-oss-120b` | `GROQ_MODEL` | Groq LLM model identifier |
| `LLM_TEMPERATURE` | `0.0` | `LLM_TEMPERATURE` | Sampling temperature for LLM |
| `EMBEDDING_MODEL_NAME` | `jina-embeddings-v2-base-en` | `EMBEDDING_MODEL` | Jina embedding model name |
| `CHUNK_SIZE` | `500` | `CHUNK_SIZE` | Text chunk character length |
| `CHUNK_OVERLAP` | `50` | `CHUNK_OVERLAP` | Overlap characters between chunks |
| `TOP_K_RESULTS` | `3` | `TOP_K_RESULTS` | Number of retrieved chunks passed to agent |
| `LANGSMITH_TRACING` | `false` | `LANGSMITH_TRACING` | Enable LangSmith tracing (`true` / `false`) |

---

## 🔍 LangSmith Observability

When `LANGSMITH_TRACING=true` is enabled in `.env`, all interactions are automatically traced:
* **Agent thought process:** Tool calls, tool arguments, and returned chunk content.
* **Retrieval latency:** Time spent embedding questions and querying the FAISS index.
* **Token usage & costs:** Exact prompt and completion token counts from Groq.

View your execution traces at [smith.langchain.com](https://smith.langchain.com).

---

## 🧪 Testing

The repository includes a comprehensive unit test suite with 32 tests using `pytest` (with mocked API calls):

```bash
cd Production_RAG
uv run pytest tests/ -v
# or: pytest tests/ -v
```

Test coverage includes:
- Document loading and metadata (`test_ingestion.py`)
- Text chunking and overlap (`test_chunking.py`)
- Vector store caching & persistence (`test_vectordb.py`)
- Retriever instantiation and top-k filtering (`test_retrieval.py`)
- Agent tool execution (`test_tools.py`)
- Prompt templates and consistency (`test_prompts.py`)
- LangSmith tracing configuration (`test_tracing.py`)
- Pipeline smoke tests (`test_pipeline.py`)

---

## 📜 License

Distributed under the MIT License.
