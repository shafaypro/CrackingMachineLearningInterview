# Multi-Model and AI Orchestration

This guide covers systems that dynamically route requests across multiple AI models and providers to optimize quality, latency, and cost.

---

## Overview

Multi-model orchestration means choosing between different models at runtime instead of forcing one model to handle every request. This matters because production AI systems rarely have a single perfect model:

- the best reasoning model may be too expensive for every request
- the cheapest model may fail on hard tasks
- one provider may have better multimodal support
- fallback paths are necessary for reliability

---

## Core Concepts

### Model routing

Routing selects the best model for a request based on task type, latency target, budget, input size, and safety requirements.

### Cost optimization

Good orchestration reduces spend by sending simpler tasks to smaller models while reserving premium models for hard requests.

### Latency optimization

Some systems route to lower-latency models first or use speculative patterns to reduce tail latency.

### Fallback strategies

Fallbacks protect uptime when a provider is rate-limited, a model returns malformed output, or a context window is too small.

---

## Key Skills

### Choosing models dynamically

In practice, this means defining routing heuristics or learned policies that are easy to explain and measure.

### Combining providers

A staff-level engineer knows how to hide provider differences behind a stable interface while still exposing provider-specific capabilities when needed.

### Building abstraction layers

This includes standard request schema, structured output normalization, retry logic, and observability by model and provider.

### A/B testing models

You should be able to compare candidate models on cost per successful task, latency percentiles, quality scores, and failure rate.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| LiteLLM | Unified API across model providers | Multi-provider gateways and routing |
| Custom routing layer | Business-specific routing logic | Production systems with policy control |
| Redis | Caching frequent prompts or outputs | Reducing cost and latency |
| OpenTelemetry | Tracing across model calls | Observability in orchestrated systems |
| Feature flags / experiment platforms | Traffic splitting and A/B tests | Safe rollout of new routing policies |

---

## Projects

### Multi-model router

- Goal: Route requests to different models based on task class and budget.
- Key components: classifier or rule engine, provider adapters, tracing, quality logging.
- Suggested tech stack: LiteLLM, FastAPI, Redis, Postgres.
- Difficulty: Advanced.

### Cost-optimized AI system

- Goal: Serve most traffic with low-cost models while escalating hard tasks.
- Key components: complexity scoring, budget guardrails, fallback thresholds, dashboarding.
- Suggested tech stack: Python, provider SDKs, Prometheus/Grafana.
- Difficulty: Advanced.

### Fallback AI pipeline

- Goal: Survive model/provider failures with graceful degradation.
- Key components: timeout policies, retry tiers, schema validation, circuit breaker logic.
- Suggested tech stack: FastAPI, LiteLLM, Redis, Pydantic.
- Difficulty: Intermediate to advanced.

### Model comparison framework

- Goal: Benchmark multiple models on the same prompt dataset.
- Key components: eval dataset, batch runner, metrics collector, diff viewer.
- Suggested tech stack: Python, Pandas, LangSmith or custom evaluator.
- Difficulty: Intermediate.

---

## Example Code

```python
def route_model(task_type: str, max_latency_ms: int, budget_tier: str) -> str:
    if task_type == "simple_qa" and budget_tier == "low":
        return "small-fast-model"
    if task_type == "multimodal":
        return "vision-capable-model"
    if max_latency_ms < 1000:
        return "fast-reasoning-model"
    return "high-quality-reasoning-model"
```

---

## Suggested Project Structure

```text
multi-model-router/
├── src/
│   ├── router.py
│   ├── providers/
│   ├── policies.py
│   ├── cache.py
│   └── metrics.py
├── evals/
├── dashboards/
└── README.md
```

---

## Interview Q&A

#### How would you design a model router?

Start with the routing signal, because that's the whole design. Options in increasing sophistication: static rules by request type (cheapest, surprisingly effective), a small classifier trained on labeled easy/hard examples, a cascade where the small model attempts first and escalates on low confidence, or an LLM-based router (accurate but adds a call's latency and cost).

The cascade is usually the best starting point because the escalation signal is free — you already have the small model's output and can check confidence, schema validity, or a cheap verifier. Typical result is 60–80% of traffic handled by the small model at a fraction of the cost with minimal quality impact.

The parts to get right: a **fallback** when the chosen provider fails, **per-route evaluation** so you can prove the small model is adequate for its slice, and **logging of which route each request took** so you can audit quality by route rather than in aggregate.

#### How do you build a provider abstraction without lowest-common-denominator APIs?

Define your own interface around what your application needs — messages, tools, structured output, streaming — and adapt each provider to it, rather than adopting one provider's SDK shape as the interface. Keep provider-specific capabilities reachable through an explicit escape hatch instead of hiding them.

The parts that are genuinely hard, and worth naming: tool-calling schemas differ meaningfully between providers; token counting and pricing differ; streaming event shapes differ; and error taxonomies differ (rate limit vs overload vs content filter vs context length). Normalizing errors into your own categories is what makes retry and fallback logic possible at all.

#### What is your fallback strategy when a provider is down?

Layered, with explicit degradation:
1. **Retry with backoff and jitter** for transient errors — most provider errors are transient, and full jitter prevents a synchronized retry storm.
2. **Failover to a second provider** with an equivalent model, using the abstraction layer. Requires that prompts work on both, which means testing them on both.
3. **Degrade to a smaller or local model** if no equivalent is available — a worse answer usually beats an error.
4. **Serve from cache** for repeated queries.
5. **Fail gracefully** with a clear message and a queued retry for non-interactive work.

The important design detail is a **circuit breaker**: after N consecutive failures, stop sending traffic to the failing provider for a cooldown period. Without it, every request pays the full timeout before failing over, and your latency collapses even though a healthy provider is available.

#### How do you A/B test two models fairly?

Randomize at the user or session level, not per request — per-request assignment gives one user inconsistent behavior and contaminates the experience being measured. Fix the prompt and retrieval pipeline across arms so only the model varies. Define a primary metric before starting, and guardrails that must not regress: latency, cost per request, error rate, and any safety signal.

For LLM features, add an offline regression suite as a gate before the experiment even starts — an A/B test is expensive and slow, so it should only see candidates that already passed a fixed eval set. And be honest about power: LLM quality effects are often small relative to the variance in user behavior, so decide the minimum detectable effect up front rather than reading noise as a result.

#### How do you compare cost across providers meaningfully?

Not by list price per token. Model **cost per successfully completed task**: a model that's twice the price but needs one attempt instead of three retries is cheaper. Then account for the structure of your traffic — input tokens dominate RAG and agent workloads, so a provider's input price and prompt-caching discount matter far more than its output price. Include the cost of failures, the cost of the router itself if it makes an extra call, and, for self-hosting, the GPU's idle time at your actual utilization.

The metric that survives review is dollars per resolved request at a fixed quality bar, measured on your own traffic.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Routing on request length as a proxy for difficulty | Length and difficulty are only loosely related | Train a classifier, or cascade with a confidence-based escalation |
| No circuit breaker on a failing provider | Every request pays the full timeout before failover | Trip after N failures, cool down, half-open probe |
| Retrying without jitter | Synchronized retry storms re-trigger the rate limit | Exponential backoff with full jitter |
| One prompt assumed portable across providers | Behavior differs; quality silently drops on failover | Test the prompt on every provider in the fallback chain |
| Comparing list price per token | Ignores retries, caching, and input/output mix | Cost per successfully completed task on real traffic |
| Per-request A/B assignment | Users see inconsistent behavior; results are contaminated | Randomize by user or session |
| No per-route quality measurement | Aggregate metrics hide a bad route | Log the route and evaluate each slice separately |
| Abstraction that hides provider capabilities | You lose structured outputs, caching, or tool features | Keep an explicit escape hatch to provider-native calls |
| Ignoring normalized error taxonomy | Retry logic can't distinguish rate limit from context overflow | Map provider errors to your own categories |

---

## Related Topics

- [LLM Fundamentals](./intro_llm_fundamentals.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
- [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
