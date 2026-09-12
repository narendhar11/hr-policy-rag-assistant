# Testing Strategy with Pytest — HR Policy RAG Assistant

Great question! My testing strategy was guided by a few core principles:

## 1. Mirror the architecture

I created a 1:1 mapping between production modules and test files — `chunking/chunker.py` → `test_chunking.py`, `retrieval/retriever.py` → `test_retrieval.py`, and so on across all 9 modules (ingestion, chunking, vectordb, retrieval, tools, prompts, pipeline config, tracing, and guardrails). This makes it instantly obvious where to add tests when a module changes.

## 2. Keep it hermetic — zero network calls

RAG systems depend on external APIs (Groq LLM, Jina embeddings, LangSmith tracing, FAISS indexes). I didn't want tests that break because a third-party service is down or cost money per run. So I used `unittest.mock.patch` and `MagicMock` to stub every external dependency. For file I/O, I used Python's `tempfile` module to create isolated test artifacts that get cleaned up automatically. The entire 41-test suite runs in under 1 second.

## 3. Test behavior at every layer of the RAG pipeline

- **Ingestion**: Does the loader return proper LangChain `Document` objects? Does it raise `FileNotFoundError` for missing files?
- **Chunking**: Are chunks within the configured `CHUNK_SIZE + CHUNK_OVERLAP` bounds? Does an empty document return `[]`?
- **Vector Store**: Does save/load delegate correctly to FAISS with `allow_dangerous_deserialization=True`?
- **Retrieval**: Does the retriever respect the `top_k` config?
- **Tools**: Does the agent tool join chunk texts correctly? Does it handle empty retrieval gracefully?
- **Guardrails**: Does the LLM-as-a-judge safety system correctly classify violations (prompt injection, PII leaks, unauthorized promises)? Does the pipeline short-circuit and return a refusal message without ever invoking the agent?
- **Config & Prompts**: Smoke tests that hyperparameters are valid and the system prompt maintains a single source of truth.
- **Tracing**: Does LangSmith setup correctly inject env vars when enabled, and gracefully skip when disabled?
