# Backend and System Design for AI

This guide focuses on the backend engineering patterns required to turn AI models into scalable, reliable products.

---

## Overview

AI systems fail in production for backend reasons as often as for model reasons. A strong ML or AI engineer should be able to design low-latency APIs, asynchronous pipelines, caching layers, distributed workers, and resilient failure handling.

This matters because models only create value when the surrounding system is stable under real traffic.

For a broader interview-prep view of backend concepts that sit around ML systems, see the dedicated [Backend System Design Interview Guide](./backend_system_design_interview_guide.md). It expands the general system design layer with 32 interview concepts spanning scalability, databases, caching, distributed systems, messaging, networking, storage, reliability, and search.

---

## Core Concepts

### Scalable architectures

A scalable AI architecture separates concerns across the API layer, orchestration layer, model inference layer, storage and retrieval layer, and observability layer.

### API design

Good AI APIs expose clear request and response contracts, idempotency where needed, streaming when latency matters, and structured errors.

### Distributed systems

Many AI tasks are too slow or expensive for synchronous request-response design. Distributed systems patterns like queues, workers, and event-driven processing become necessary.

### Data modeling and domain boundaries

Production AI systems still need clean entities, ownership boundaries, and data contracts. You should be able to explain which service owns which data, what the write model looks like, and where denormalized read models or analytics views come from.

### Caching strategies

Caching reduces cost and latency for repeated work: prompt-response cache, embedding cache, feature cache, and retrieval result cache.

### Fault tolerance

Production AI needs timeouts, retries, circuit breakers, fallbacks, and graceful degradation.

---

## Key Skills

### Designing low-latency systems

In practice, this means minimizing synchronous dependencies, using streaming where appropriate, and keeping the hot path narrow.

### Handling concurrency

You should know when to use async APIs, worker queues, batch processing, and background tasks.

### Load balancing

This includes balancing across stateless API replicas and sometimes across different inference backends.

### Data-driven development

A strong backend engineer also designs observability and product feedback loops: event schemas, experiment variants, funnel metrics, feature freshness, and operational telemetry.

### Fault tolerance

A strong engineer defines timeout budgets, fallback behavior, retry scopes, and partial failure handling.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| FastAPI | High-performance Python API framework | AI backends and inference services |
| Redis | Cache and lightweight state store | Response caching, rate limiting, queues |
| Kafka | Event streaming platform | High-volume asynchronous pipelines |
| Kubernetes | Container orchestration | Scalable multi-service production systems |
| Postgres | Transactional data store | Metadata, job state, user/application data |

---

## Projects

### Scalable AI API

- Goal: Build an inference API that supports rate limiting, caching, and observability.
- Key components: FastAPI service, authentication, structured outputs, metrics, tracing.
- Suggested tech stack: FastAPI, Redis, Postgres, Docker.
- Difficulty: Intermediate.

### Real-time inference system

- Goal: Serve predictions with tight latency budgets.
- Key components: optimized model serving, request batching, timeout handling, autoscaling.
- Suggested tech stack: FastAPI, model server, Redis, Kubernetes.
- Difficulty: Advanced.

### Streaming pipeline

- Goal: Process events continuously for enrichment, ranking, or anomaly detection.
- Key components: Kafka topics, workers, state store, online features.
- Suggested tech stack: Kafka, Python consumers, Redis, Prometheus.
- Difficulty: Advanced.

### High-load backend system

- Goal: Handle bursty traffic for an AI product with graceful degradation.
- Key components: queue buffering, load shedding, caching, provider fallback.
- Suggested tech stack: FastAPI, Redis, Kafka, Kubernetes.
- Difficulty: Advanced.

---

## Example Code

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/infer")
async def infer(request: InferenceRequest):
    return {"status": "ok", "prompt_length": len(request.prompt)}
```

---

## Suggested Project Structure

```text
scalable-ai-api/
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── services/
│   ├── cache.py
│   └── metrics.py
├── tests/
├── docker/
└── README.md
```

---

## Interview Q&A

#### Design an API that serves an LLM feature to 10,000 concurrent users. Where do you start?

With numbers, before architecture. Ask for the request rate, the prompt and output token distributions, the latency SLO, and the cost budget — those four determine everything else. 10,000 concurrent users is not 10,000 requests per second; if each user sends a request every 30 seconds, that's ~330 RPS, which is a very different system.

Then the shape: a stateless API layer behind a load balancer (so it scales horizontally and any instance can serve any request), an async request path since LLM calls take seconds and blocking a thread per request destroys throughput, a queue with backpressure so overload degrades rather than collapses, streaming responses to cut perceived latency, and a caching layer — exact-match first, then semantic caching if the traffic repeats.

The parts that separate a strong answer: **admission control** (reject early with a clear error rather than queueing indefinitely), **per-user rate and spend limits**, **timeouts at every hop** with values that sum to less than the client's timeout, and an explicit **degradation path** — a smaller model or a cached response when the primary is saturated.

#### Why is async I/O particularly important for AI backends?

Because the service spends nearly all its wall-clock time waiting on a network call to a model, not computing. In a synchronous thread-per-request server, a 3-second LLM call occupies a thread for 3 seconds; a few hundred concurrent requests exhausts the pool and new requests queue behind idle threads.

With async, one process handles thousands of in-flight requests because each awaits its I/O and yields. The practical caveats worth stating: any blocking call in an async handler (a synchronous DB driver, a CPU-heavy tokenization, `time.sleep`) stalls the whole event loop, so it must move to a thread or process pool. And async raises concurrency, not per-request speed — it doesn't make the model faster.

#### How do you handle a request that takes 60 seconds?

Don't hold an HTTP connection open for it. Switch to an asynchronous job pattern: accept the request, return `202` with a job ID immediately, process on a worker pool, and let the client poll a status endpoint or receive a webhook or SSE push on completion. Persist job state so a worker restart doesn't lose it, and make the work idempotent by job ID so retries are safe.

If the output is incrementally useful — generated text usually is — streaming is the better answer: the connection stays open but the user sees progress within a second, which changes the experience entirely even though total time is unchanged.

#### What caching layers would you use, and what are their risks?

Four, from cheapest to riskiest:
- **Exact-match response cache** keyed by a hash of the full request (prompt, model, parameters). Safe and effective for repeated identical requests.
- **Prompt/prefix cache** at the model provider or inference server, which requires ordering the prompt with the stable prefix first — often the single largest cost reduction in a RAG or agent system.
- **Retrieval cache** for embeddings and search results, keyed by content hash.
- **Semantic cache**, returning a previous answer for a sufficiently similar query. Powerful and the most dangerous — a threshold set too loosely serves confidently wrong answers to subtly different questions.

Every cache needs an invalidation story tied to the underlying data, a TTL, and per-user scoping when responses are personalized or permission-dependent. A cache that ignores the requesting user is a data-leak vector, not just a correctness bug.

#### How do you make an AI backend fail gracefully?

Assume the model call fails, is slow, or is rate-limited, because it will be. Layer the defenses: timeouts at every hop; retries with exponential backoff and **full jitter**; a **circuit breaker** so a failing dependency stops absorbing every request's full timeout; **bulkheads** so one slow provider doesn't exhaust the shared connection pool; a **fallback chain** to an alternate provider, a smaller model, or a cached or templated response; and **load shedding** that rejects low-priority traffic early to protect the SLO for the rest.

The framing that lands: decide in advance what "degraded" looks like and make it a designed state, not an accident. A user-visible "we're operating in reduced mode" beats a timeout, and both beat a cascade.

#### How would you design for cost as a first-class constraint?

Instrument cost per request end to end, attributed by feature, route, and tenant — you cannot control what you don't measure, and LLM cost is unusual in that it varies per request rather than being a fixed serving cost.

Then design the controls in: per-user and per-tenant rate limits and spend caps, a global budget circuit breaker, `max_tokens` bounds, a step limit on any agent loop, prompt ordering that maximizes cache hits, and routing that sends the easy majority of traffic to a small model. Push offline work to batch APIs.

The architectural point: cost belongs in the same category as latency and availability — it needs SLOs, monitoring, and alerting, not a monthly invoice surprise.

#### How do you handle multi-tenancy safely?

Tenant identity must be established at the edge from an authenticated token and carried through every layer as an explicit parameter — never inferred from a request body field a client controls. Every data access filters on it, including vector search, where post-filtering can both return empty pages and leak the existence of documents.

Beyond correctness: per-tenant rate limits and spend caps so one tenant cannot exhaust shared capacity (the noisy-neighbor problem), per-tenant cache namespacing, and separate storage for tenants under regulatory isolation requirements. For a security-sensitive design, separate indexes per tenant is the defensible choice over a shared index with a mandatory filter — one missed filter in one code path is a breach.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Synchronous handlers around LLM calls | Thread pool exhausts at low concurrency | Async I/O; move blocking work to a thread pool |
| A blocking call inside an async handler | Stalls the entire event loop | Audit for sync DB drivers, CPU work, `time.sleep` |
| Holding HTTP connections for long generations | Proxy timeouts, wasted resources, poor UX | Stream, or return a job ID and poll/webhook |
| No timeout on an upstream model call | One hung request occupies capacity indefinitely | Timeouts at every hop, summing under the client's |
| Retries without jitter or a circuit breaker | Retry storms turn a partial outage into a full one | Exponential backoff with full jitter + circuit breaker |
| Unbounded queues | Latency grows until everything times out | Bounded queues with admission control and load shedding |
| Semantic cache threshold set loosely | Serves confidently wrong answers | Tune the threshold on real queries; scope per user; add TTL |
| Cache not scoped by user or tenant | Personalized or permissioned data leaks across users | Include identity in the cache key |
| Tenant ID taken from the request body | Trivially spoofable; cross-tenant access | Derive from the authenticated token at the edge |
| Stateful API instances | Cannot scale horizontally or restart safely | Keep session state in Redis or a database |
| Cost measured only on the monthly invoice | No way to attribute or control spend | Per-request cost telemetry with budgets and alerts |

---

## Related Topics

- [Backend System Design Interview Guide](./backend_system_design_interview_guide.md)
- [ML System Design Framework](./README.md)
- [Data Engineering for AI](../data_engineering/intro_data_engineering_for_ai.md)
- [Model Serving](../mlops/intro_model_serving.md)
- [Agent Systems and Tool Use](../ai_genai/intro_agent_tool_use.md)
- [Multi-Model Orchestration](../ai_genai/intro_multi_model_orchestration.md)
