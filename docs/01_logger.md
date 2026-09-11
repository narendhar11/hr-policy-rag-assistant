# 01. Logger

**Branch:** `logs-and-llm-observability`

## What is a logger?

A logger is a way for your program to write down, step by step, what it is
doing while it runs — instead of (or in addition to) `print()` statements.
Every line it writes is called a "log". Think of it like a flight recorder
("black box") for your app: if something goes wrong, or you just want to
know what happened during a run, you open the log file and read the story
of that run.

## Why do we need it instead of just using `print()`?

- `print()` only shows up in your terminal, and disappears once the terminal
  closes. Logs are saved to a file, so you can look at them later — even
  days later.
- `print()` gives you no context automatically. A logger can automatically
  attach the time, the severity (how serious the message is), and which
  file/module it came from.
- You can turn logging up or down (e.g. "only show me errors") without
  touching your code, because of **log levels** (below).
- In a real product, you often have many things running (a web app, a
  background job, an agent). Logs are how you debug "what actually
  happened" after the fact, e.g. why a user got a wrong answer, or why a
  request was slow.

## Log levels — from least to most serious

Python's built-in `logging` module (which we use here) has 5 standard
levels. You pick the right one for each message:

| Level      | Method            | When to use it |
|------------|-------------------|----------------|
| `DEBUG`    | `logger.debug()`  | Very detailed info, only useful while actively debugging (e.g. "raw response from API was ..."). Usually turned off in normal runs. |
| `INFO`     | `logger.info()`   | Normal, expected events — "the app started", "loaded 5 chunks", "user asked X". This is what we use in this project. |
| `WARNING`  | `logger.warning()`| Something unexpected happened, but the app can keep going (e.g. "LANGSMITH_API_KEY missing, tracing skipped"). |
| `ERROR`    | `logger.error()`  | Something failed — a request crashed, a function couldn't complete. The app might still be running, but this specific thing broke. |
| `CRITICAL` | `logger.critical()`| The whole app is in serious trouble and might not be able to continue at all. |

We set our logger's level to `INFO` in `utils/logger.py`, which means
`INFO`, `WARNING`, `ERROR`, and `CRITICAL` messages all get logged, but
`DEBUG` messages are hidden. You can change the level to `DEBUG` later if
you want maximum detail while troubleshooting.

## How it works in this codebase

- `utils/logger.py` is the centralized logging module. It:
  1. Creates a `logs/` folder automatically if it doesn't already exist.
  2. Picks one log file per run, named with the timestamp the run started
     (e.g. `logs/run_20260910_161314.log`).
  3. Configures both a **FileHandler** (writes to the log file) and a
     **StreamHandler** (prints to the console) with a consistent format:
     `%(asctime)s - %(levelname)s - %(name)s - %(message)s`
  4. Exposes one function, `get_logger(name)`, that every other file calls.

- Every module does this at the top:
  ```python
  from utils.logger import get_logger
  logger = get_logger(__name__)
  ```
  `__name__` is the module's own name (e.g. `ingestion.loader`), so when
  you read the log file you can see exactly which file logged which
  message.

- Then, at the important steps in each function, we call
  `logger.info(...)` to record what happened (e.g. "Loaded 1 document(s)",
  "search_hr_policy called with query: ...", "Final answer: ...").

## Files changed

| File | What changed |
|------|--------------|
| `utils/logger.py` | **New file.** Sets up the shared logger and the `logs/` folder. |
| `ingestion/loader.py` | Logs which file is loaded and how many documents were loaded. |
| `chunking/chunker.py` | Logs how many chunks the document was split into. |
| `embeddings/embedder.py` | Logs which embedding model is being initialized. |
| `vectordb/vector_store.py` | Logs build vs. load-from-disk path, save confirmation, retriever creation. |
| `tools/tools.py` | Logs every search query sent to `search_hr_policy` and how many chunks matched. |
| `llm/llm_client.py` | Logs which LLM model is initialized. |
| `agent/agent.py` | Logs agent creation. |
| `pipeline.py` | Logs the start/end of building the assistant, and every question + final answer. |
| `main.py` | Logs the start and end of a CLI run. |
| `app.py` | Logs agent initialization, user questions, and errors during chat. |
| `.gitignore` | Already contains `*.log` so all log files are ignored and never committed. |

## Verified working

Ran `streamlit run app.py` and `python main.py` on this branch: the app
answered questions correctly, and a new timestamped file appeared under
`logs/` containing a full step-by-step trace of the run — from
"LangSmith tracing enabled" and "Vector store already exists, loading"
through each `search_hr_policy` call to the final answers. All 32 unit
tests pass.
