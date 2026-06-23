# Python for AI Engineering (2026 Edition)

The foundation under every LLM app and agent. Before RAG, tools, or multi-agent
orchestration, you need fluent, idiomatic, *production* Python. This guide covers
the slice of Python that AI engineers use daily — and that interviewers expect you
to write without thinking.

> **Why it's listed first in the roadmap:** agent code is concurrent, schema-driven,
> API-heavy, and failure-prone. Weak Python here shows up immediately as flaky,
> slow, unobservable agents.

---

## Table of Contents
1. [Core Python You Actually Use](#1-core-python-you-actually-use)
2. [Type Hints](#2-type-hints)
3. [async / await & Concurrency](#3-async--await--concurrency)
4. [Pydantic Models](#4-pydantic-models)
5. [Environment & Dependency Management](#5-environment--dependency-management)
6. [Working with APIs](#6-working-with-apis)
7. [Error Handling & Retries](#7-error-handling--retries)
8. [Logging & Observability](#8-logging--observability)
9. [Project Hygiene](#9-project-hygiene)
10. [Interview Questions](#10-interview-questions)

---

## 1. Core Python You Actually Use

| Feature | Why it matters in AI code |
|---------|---------------------------|
| **Comprehensions / generators** | Stream large datasets/embeddings without loading everything into memory |
| **Context managers (`with`)** | Manage clients, files, DB sessions, spans cleanly |
| **Decorators** | Wrap functions with retries, caching, tracing, timing |
| **`dataclasses` / `Enum`** | Lightweight typed records and fixed choice sets |
| **`functools` (`lru_cache`, `partial`)** | Cache deterministic calls; pre-bind config |
| **`pathlib`** | Safe, cross-platform path handling (and traversal checks) |

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed_cached(text: str) -> tuple[float, ...]:
    """Cache deterministic embeddings so repeated text isn't re-embedded."""
    return tuple(embed(text))
```

---

## 2. Type Hints

Type hints make agent code maintainable *and* power schema generation (Pydantic,
tool definitions) and editor/static checks.

```python
from typing import Literal, Optional, Any

def call_model(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.2,
    tools: Optional[list[dict[str, Any]]] = None,
    role: Literal["system", "user", "assistant"] = "user",
) -> str:
    ...
```

- Prefer built-in generics (`list[str]`, `dict[str, int]`) on modern Python.
- `Literal` for fixed string choices; `Optional[X]` = `X | None`.
- Run a static checker (`mypy` / `pyright`) in CI — it catches a class of bugs
  before they reach a flaky agent run.

---

## 3. async / await & Concurrency

Agents make **many** network calls. Doing them sequentially is the #1 latency sink.
`asyncio` lets independent calls run concurrently.

```python
import asyncio

async def fetch_tool(call):
    return await run_tool_async(call)

async def run_parallel_tools(tool_calls):
    # Execute independent tool calls concurrently, gather all results
    return await asyncio.gather(*(fetch_tool(c) for c in tool_calls))
```

- **When to use async** — I/O-bound work (API/tool/DB calls). Not CPU-bound work
  (use processes for that).
- **`asyncio.gather`** — fan out parallel calls; **`asyncio.as_completed`** — handle
  results as they finish; **`asyncio.Semaphore`** — cap concurrency to respect rate limits.
- **Don't block the event loop** — never call a synchronous, blocking SDK inside an
  async handler; use the async client or `run_in_executor`.
- Most LLM SDKs ship both sync and async clients — match the one to your app.

```python
sem = asyncio.Semaphore(10)  # respect provider rate limits
async def guarded_call(req):
    async with sem:
        return await client_async.call(req)
```

---

## 4. Pydantic Models

Pydantic is the backbone of **structured outputs**, **tool schemas**, and **config**.
It validates and coerces data at the boundary so the rest of your code trusts its types.

```python
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)

class Settings(BaseModel):
    model: str = "claude-opus-4-8"
    temperature: float = 0.2
    max_tokens: int = 4096
```

- **Tool schemas** — `SearchArgs.model_json_schema()` produces the JSON schema you
  hand to an LLM's function-calling API.
- **Structured outputs** — parse model JSON into a validated object; on failure,
  re-ask with the validation error (see the `instructor` library).
- **Config** — `pydantic-settings` reads typed config from env vars / `.env`.
- **v2 essentials** — `model_validate()`, `model_dump()`, `Field(...)`, validators.

→ Deeper: [Pydantic for AI Systems](./intro_pydantic.md) · [Structured Outputs](../ai_genai/intro_structured_outputs.md)

---

## 5. Environment & Dependency Management

| Tool | Use |
|------|-----|
| **`venv`** | Built-in isolated environments |
| **`uv`** | Fast installer + resolver + lockfile (modern default) |
| **`poetry` / `pip-tools`** | Dependency management + locking |
| **`.env` + `python-dotenv` / `pydantic-settings`** | Load secrets/config from env |

```python
# config.py — never hardcode keys
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    anthropic_api_key: str
    vector_db_url: str = "http://localhost:6333"

    class Config:
        env_file = ".env"

config = Config()  # reads from environment / .env
```

- **Pin dependencies** with a lockfile for reproducibility.
- **Never commit `.env`** — commit `.env.example`; load real secrets from env / a
  secrets manager. See [GitHub Project Setup](../project_setup/intro_github_project_setup.md).

---

## 6. Working with APIs

```python
import httpx

async def post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
```

- **`httpx`** (async-capable) or `requests` (sync) for raw HTTP; prefer the
  provider's official SDK when one exists.
- **Always set timeouts** — a hung request stalls the whole agent.
- **Streaming** — consume Server-Sent Events incrementally for token-by-token UX.
- **Pagination** — loop until the cursor/`has_next` is exhausted.
- **Token counting** — use the provider's counter (not `tiktoken` for Claude).

→ Production HTTP/API patterns: [FastAPI for AI Backends](./intro_fastapi.md)

---

## 7. Error Handling & Retries

Networks fail, providers rate-limit, and models occasionally return junk. Handle it.

```python
import time, random

def with_retry(fn, *, retries=5, base=1.0, max_delay=30.0):
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError:          # 429 — retryable
            pass
        except ServerError:             # 5xx — retryable
            pass
        except BadRequestError:         # 4xx — do NOT retry
            raise
        delay = min(base * 2 ** attempt + random.uniform(0, 1), max_delay)
        time.sleep(delay)
    raise RuntimeError("exhausted retries")
```

- **Catch a chain, most-specific first** — distinguish retryable (429, 5xx, network)
  from non-retryable (4xx).
- **Exponential backoff + jitter** — avoid thundering-herd retries. (Most SDKs do
  this automatically — configure `max_retries`.)
- **Validate model output** — wrap structured parsing in try/except and retry with
  the error message so the model self-corrects.
- **Fail gracefully** — return a useful error to the agent (e.g. tool `is_error`),
  don't crash the whole run.

---

## 8. Logging & Observability

Agents are non-deterministic — **logs are how you debug them**.

```python
import logging, json, uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

def log_event(event: str, **fields):
    """Structured log line with a correlation id for tracing a run."""
    log.info(json.dumps({"event": event, **fields}))

run_id = str(uuid.uuid4())
log_event("tool_call", run_id=run_id, tool="search", args={"query": "..."})
```

- **Structured (JSON) logs** — machine-parseable; filter by `run_id`, tool, latency.
- **Correlation IDs** — tag every log/span in one agent run with the same id.
- **Log the right things** — each model call (model, tokens, cost, latency), each
  tool call (name, args, result, error), and decisions.
- **Levels** — `DEBUG` for traces, `INFO` for events, `WARNING`/`ERROR` for failures.
- **Graduate to tracing** — OpenTelemetry / LangSmith / Langfuse for end-to-end spans.

→ Deeper: [Observability & Tracing](../ai_genai/intro_langsmith.md) · [LLMOps](../ai_genai/intro_llmops.md)

---

## 9. Project Hygiene

- **`src/` layout**, tests, configs separated from code → see
  [ML/AI Project Folder Structures](../project_setup/intro_project_structure.md).
- **Lint/format** with `ruff`; **type-check** with `mypy`/`pyright`; **test** with
  `pytest` — all in CI.
- **Pre-commit hooks** to catch issues before they hit CI.

---

## 10. Interview Questions

1. **When do you reach for `async` in an AI app?** → I/O-bound concurrency —
   parallel model/tool/DB calls. Use `asyncio.gather`, cap with a semaphore for
   rate limits; don't use it for CPU-bound work.
2. **Why Pydantic in an LLM pipeline?** → Validate/coerce model output and tool
   args at the boundary, auto-generate JSON schemas, and load typed config.
3. **How do you handle a 429 vs a 400?** → Retry 429 (and 5xx) with exponential
   backoff + jitter; never retry 400 — it's a client error, fix the request.
4. **How do you make a non-deterministic agent debuggable?** → Structured logs with
   a per-run correlation id, log every model/tool call with tokens/latency, and
   end-to-end tracing.
5. **How do you keep secrets out of code?** → `.env` (git-ignored) + a settings
   loader locally, secrets manager / CI secrets in production; never hardcode.
6. **Sync vs async SDK client — does it matter?** → Yes — calling a blocking sync
   client inside an async handler stalls the event loop; use the async client or
   offload to an executor.

---

### Related Guides
- [Pydantic](./intro_pydantic.md) · [FastAPI](./intro_fastapi.md)
- [The Agentic AI Engineer Roadmap](../ai_genai/intro_agentic_ai_engineering_roadmap.md)
- [Project Setup & Engineering](../project_setup/README.md)
