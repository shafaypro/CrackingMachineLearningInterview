# FastAPI — Production-Grade AI Backend Engineering

## Why FastAPI for AI Systems?

FastAPI is the de facto standard for building AI/LLM API backends in 2026 because:
- **Async-first**: Native `asyncio` support for concurrent LLM calls
- **Automatic OpenAPI docs**: Zero-config Swagger UI
- **Pydantic integration**: Input validation and output schema enforcement
- **Performance**: On par with Node.js and Go for I/O-bound workloads
- **Type safety**: Full type hints enable better IDE support and fewer runtime bugs

---

## Production LLM API — Complete Example

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from anthropic import AsyncAnthropic
import asyncio
import time
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Lifespan: manage startup/shutdown resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.anthropic_client = AsyncAnthropic()
    logger.info("Anthropic client initialized")
    yield
    # Shutdown
    await app.state.anthropic_client.close()
    logger.info("Anthropic client closed")

app = FastAPI(
    title="ML Interview Prep API",
    version="2.0.0",
    lifespan=lifespan
)

# ─── Request / Response Models ───────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default="claude-sonnet-4-6")
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=1.0)
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

# ─── Non-streaming endpoint ───────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    client: AsyncAnthropic = app.state.anthropic_client
    start = time.time()

    try:
        message = await client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.question}]
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    return ChatResponse(
        answer=message.content[0].text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        latency_ms=latency_ms
    )
```

---

## Streaming Responses (SSE)

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    client: AsyncAnthropic = app.state.anthropic_client

    async def generate_tokens():
        try:
            async with client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.question}]
            ) as stream:
                async for text in stream.text_stream:
                    # Server-Sent Events format
                    yield f"data: {json.dumps({'token': text})}\n\n"

                # Send final usage stats
                final_message = await stream.get_final_message()
                yield f"data: {json.dumps({'done': True, 'usage': {'input': final_message.usage.input_tokens, 'output': final_message.usage.output_tokens}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

### WebSocket Streaming Alternative

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/chat/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    client: AsyncAnthropic = app.state.anthropic_client

    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")

            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": question}]
            ) as stream:
                async for text in stream.text_stream:
                    await websocket.send_json({"token": text})

            await websocket.send_json({"done": True})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
```

---

## Async Patterns & Concurrency Control

### Async LLM Calls (Parallel)

```python
import asyncio
from anthropic import AsyncAnthropic

async def batch_evaluate(questions: list[str]) -> list[str]:
    """Evaluate multiple questions in parallel"""
    client = AsyncAnthropic()

    async def evaluate_one(question: str) -> str:
        msg = await client.messages.create(
            model="claude-haiku-4-5",  # Use cheaper model for batch
            max_tokens=512,
            messages=[{"role": "user", "content": question}]
        )
        return msg.content[0].text

    # Run all in parallel (respects asyncio event loop)
    results = await asyncio.gather(*[
        evaluate_one(q) for q in questions
    ], return_exceptions=True)

    return [
        r if not isinstance(r, Exception) else f"Error: {r}"
        for r in results
    ]

@app.post("/batch-evaluate")
async def batch_evaluate_endpoint(questions: list[str]):
    if len(questions) > 20:
        raise HTTPException(status_code=400, detail="Max 20 questions per batch")
    return await batch_evaluate(questions)
```

### Semaphore for Concurrency Control

```python
import asyncio

# Limit concurrent LLM calls to avoid rate limits
MAX_CONCURRENT_LLM_CALLS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

async def rate_limited_llm_call(prompt: str) -> str:
    async with semaphore:  # Only 5 calls active at once
        client = AsyncAnthropic()
        msg = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
```

---

## Rate Limiting & Token Management

```python
from fastapi import Request, HTTPException
from collections import defaultdict
import time

# Simple in-memory rate limiter (use Redis in production)
class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate          # Tokens added per second
        self.capacity = capacity  # Max tokens
        self.tokens: dict = defaultdict(lambda: capacity)
        self.last_time: dict = defaultdict(time.time)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        elapsed = now - self.last_time[key]
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + elapsed * self.rate
        )
        self.last_time[key] = now

        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False

limiter = TokenBucketRateLimiter(rate=2.0, capacity=10)  # 2 req/sec, burst 10

def get_client_id(request: Request) -> str:
    return request.client.host  # Use API key in production

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/chat"):
        client_id = get_client_id(request)
        if not limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "1"}
            )
    return await call_next(request)
```

### LLM Token Budget Management

```python
import tiktoken

class TokenBudgetManager:
    def __init__(self, model: str = "claude-sonnet-4-6", max_budget: int = 100000):
        self.budget = max_budget
        self.used = 0

    def check_budget(self, estimated_tokens: int) -> bool:
        return (self.used + estimated_tokens) <= self.budget

    def consume(self, tokens: int):
        self.used += tokens

    def remaining(self) -> int:
        return self.budget - self.used

def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4
```

---

## Observability — Logging, Tracing, Metrics

```python
import structlog
import time
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import Counter, Histogram, generate_latest

# Structured logging
log = structlog.get_logger()

# Prometheus metrics
llm_requests_total = Counter(
    "llm_requests_total", "Total LLM requests",
    labelnames=["model", "status"]
)
llm_latency_seconds = Histogram(
    "llm_latency_seconds", "LLM request latency",
    labelnames=["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
llm_tokens_total = Counter(
    "llm_tokens_total", "Total tokens used",
    labelnames=["model", "type"]
)

# Auto-instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    tracer = trace.get_tracer(__name__)
    start = time.time()

    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("model", request.model)
        span.set_attribute("question_length", len(request.question))

        try:
            message = await call_llm(request)
            latency = time.time() - start

            # Metrics
            llm_requests_total.labels(model=request.model, status="success").inc()
            llm_latency_seconds.labels(model=request.model).observe(latency)
            llm_tokens_total.labels(model=request.model, type="input").inc(message.usage.input_tokens)
            llm_tokens_total.labels(model=request.model, type="output").inc(message.usage.output_tokens)

            # Structured log
            log.info("llm_request_completed",
                model=request.model,
                latency_ms=latency * 1000,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens
            )

            return ChatResponse(...)

        except Exception as e:
            llm_requests_total.labels(model=request.model, status="error").inc()
            log.error("llm_request_failed", error=str(e), model=request.model)
            raise

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## API Gateway for LLM Routing

```python
from fastapi import FastAPI
from typing import Literal

MODEL_ROUTING = {
    "fast": "claude-haiku-4-5",       # < 500ms, simple tasks
    "balanced": "claude-sonnet-4-6",   # 1-3s, most tasks
    "powerful": "claude-opus-4-6",     # 3-10s, complex reasoning
}

def route_to_model(task_complexity: Literal["fast", "balanced", "powerful"],
                   prompt_length: int) -> str:
    # Override: very long prompts need larger context window
    if prompt_length > 50000:
        return "claude-opus-4-6"  # Larger context
    return MODEL_ROUTING[task_complexity]

@app.post("/smart-chat")
async def smart_chat(request: ChatRequest, complexity: str = "balanced"):
    model = route_to_model(complexity, len(request.question))
    # Fallback chain: if Anthropic fails, try OpenAI
    try:
        return await call_anthropic(request, model=model)
    except Exception:
        logger.warning("Falling back to OpenAI")
        return await call_openai(request)
```

---

## Example: Agent Backend Service

```python
from fastapi import FastAPI, BackgroundTasks
from langgraph.graph import StateGraph
import uuid

# Store for async agent jobs
job_store: dict[str, dict] = {}

@app.post("/agent/run")
async def run_agent(question: str, background_tasks: BackgroundTasks):
    """Start an agent task asynchronously"""
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "running", "result": None}

    async def run():
        try:
            result = await agent_graph.ainvoke({
                "messages": [("human", question)]
            })
            job_store[job_id] = {"status": "done", "result": result}
        except Exception as e:
            job_store[job_id] = {"status": "failed", "error": str(e)}

    background_tasks.add_task(run)
    return {"job_id": job_id, "status": "running"}

@app.get("/agent/status/{job_id}")
async def get_agent_status(job_id: str):
    """Poll for agent result"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store[job_id]

@app.get("/agent/stream/{job_id}")
async def stream_agent_progress(job_id: str):
    """SSE stream of agent node updates"""
    async def generate():
        async for chunk in agent_graph.astream({"messages": [...]}, stream_mode="updates"):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Interview Questions

**Q: Why use FastAPI for LLM backends instead of Flask or Django?**
> FastAPI's async support is critical — LLM calls are I/O-bound and take 1-30 seconds. With Flask's synchronous model, each request blocks a thread. FastAPI with asyncio handles hundreds of concurrent requests with minimal threads. Also: automatic Pydantic validation, OpenAPI docs, and better performance.

**Q: How do you implement streaming LLM responses in FastAPI?**
> Use `StreamingResponse` with an async generator that yields Server-Sent Events (SSE). The LLM client streams tokens; the generator yields each token formatted as `data: {...}\n\n`. Set `Cache-Control: no-cache` and disable nginx buffering (`X-Accel-Buffering: no`).

**Q: How do you handle rate limiting for an LLM API?**
> Token bucket algorithm per client (API key or IP). In production: use Redis for distributed rate limiting (not in-memory). Track both request rate (e.g., 60 RPM) and token rate (e.g., 100K TPM). Return 429 with `Retry-After` header. Use leaky bucket for smoother limiting.

**Q: How would you design a fault-tolerant LLM backend?**
> Retry with exponential backoff for 429/503 errors. Fallback chain: primary model → cheaper model → cached response. Circuit breaker to stop hammering a failing provider. Async queue for non-real-time requests. Structured logging + metrics for all calls. Health checks on LLM provider before serving traffic.

**Q: What's the difference between SSE and WebSockets for LLM streaming?**
> SSE is one-directional (server → client), uses HTTP, works through proxies, auto-reconnects, simpler. WebSocket is bidirectional, requires a persistent connection, better for real-time chat (send new messages while receiving). For LLM streaming: SSE is usually sufficient and simpler; WebSocket when you need streaming + mid-stream user interruption.
