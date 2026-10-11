# 02. LangSmith Tracing

**Branch:** `logs-and-llm-observability`

## What is LangSmith?

LangSmith is LangChain's observability platform. Where our own
`utils/logger.py` (see [01_logger.md](01_logger.md)) writes plain-text
lines to a file on your machine, LangSmith captures a structured, visual
**trace** of everything that happened inside a single `agent.invoke(...)`
call, and uploads it to a dashboard on smith.langchain.com.

A trace shows you, as a tree:
- Every step the agent took (LLM call → tool call → LLM call → final answer)
- The exact prompt sent to the LLM, and the exact response it got back
- How many tokens were used and roughly what it cost
- How long each step took (latency), so you can see where time went
- If a step failed, the exact error

## Logging vs. tracing — why do we want both?

| | `utils/logger.py` (our own logs) | LangSmith tracing |
|---|---|---|
| Where it lives | `logs/` folder on your machine, plain text | smith.langchain.com, a web dashboard |
| What it shows | The events *we chose* to log (a handful of `logger.info()` calls) | Every LLM/tool call automatically, with full inputs/outputs, tokens, latency |
| Best for | Quick "what happened in this run" trail, works offline, free | Debugging *why* the agent answered the way it did, comparing runs, spotting slow or expensive calls, sharing a trace link with a teammate |

They're complementary: our logs tell a short story of the run; LangSmith
gives you the full, clickable detail of every LLM interaction inside it.

## How it works

LangChain has built-in support for LangSmith. If these four environment
variables are set, **every** `agent.invoke(...)` / LLM call in the app is
automatically traced — we don't have to add any tracing code around our
LLM calls ourselves:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT=hr-policy-rag
```

- `LANGSMITH_TRACING=true` — turns tracing on. Set it to `false` (or
  remove it) to turn tracing off with zero code changes.
- `LANGSMITH_ENDPOINT` — which LangSmith server to send traces to (the
  default cloud one, unless you're self-hosting).
- `LANGSMITH_API_KEY` — your personal/team key, found in LangSmith under
  Settings → API Keys. **Never commit this to git.**
- `LANGSMITH_PROJECT` — groups all traces from this app together under
  one project name (`hr-policy-rag`) in the dashboard, so they don't mix
  with traces from other projects in your account.

Because `config.py` already calls `load_dotenv()`, putting these in
`.env` is enough for LangChain to pick them up.

## What we added on top of "just set the env vars"

Setting the env vars is enough to *get* tracing. But we also want it to
be **visible** that tracing is on, without having to check LangSmith
every time. So we added a `utils/tracing.py` module with a
`setup_langsmith_tracing()` function that:

1. Reads the `LANGSMITH_*` variables from `config.py`.
2. Sets the four `LANGCHAIN_*` OS environment variables that LangChain
   reads automatically (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`,
   `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`).
3. Logs (via our own logger) one line at the start of every run:
   `LangSmith tracing enabled successfully! Project: 'hr-policy-rag' | Dashboard: ...`
   or `LangSmith tracing is disabled` if turned off.
4. Returns `True` / `False` so calling code can check programmatically
   via the `is_tracing_enabled()` helper.

## Files changed

| File | What changed |
|------|--------------|
| `.env` | **Added** (not committed — gitignored) `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. |
| `.env.example` | LangSmith variables documented with placeholder values, so anyone cloning the repo knows what to fill in. Safe to commit. |
| `config.py` | Reads the four `LANGSMITH_*` variables from `.env` (mirrors the existing `GROQ_API_KEY` / `JINA_API_KEY` pattern). `LANGSMITH_TRACING` is parsed as a boolean. |
| `utils/tracing.py` | **New file.** `setup_langsmith_tracing()` configures the `LANGCHAIN_*` env vars and logs whether tracing is on/off. `is_tracing_enabled()` provides a runtime check. |
| `pipeline.py` | Calls `setup_langsmith_tracing()` once, at the start of `build_hr_policy_assistant()`, before any LangChain objects are created. |
| `tests/test_tracing.py` | **New file.** 3 unit tests verifying tracing disabled, tracing skipped when API key missing, and tracing enabled with correct env vars set. |

## Verified working

Ran `streamlit run app.py` and `python main.py` on this branch with
`LANGSMITH_TRACING=true` in `.env`: the app answered questions correctly,
the run log showed `LangSmith tracing enabled successfully! Project:
'hr-policy-rag'`, and the run appeared as a new trace under the
**hr-policy-rag** project on smith.langchain.com, showing each LLM call
and the `search_hr_policy` tool call inside it. All 32 unit tests pass.
