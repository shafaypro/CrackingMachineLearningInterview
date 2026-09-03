# LLMOps and MLOps Engineering

This guide connects classical MLOps with the additional operational requirements of LLM-powered systems.

---

## Overview

MLOps is the discipline of training, deploying, versioning, and monitoring ML systems in production. LLMOps extends that foundation with prompt management, retrieval quality, agent traces, model routing, and generative-AI-specific evaluation.

Modern teams need both because production AI systems are now a combination of models, prompts, retrieval pipelines, tools, guardrails, and observability.

---

## Core Concepts

### Model deployment

Deployment is not just "serve a model endpoint." It includes packaging, rollout policy, rollback strategy, capacity planning, and security boundaries.

### Monitoring and logging

For LLM systems you need both system metrics and semantic traces: latency, token usage, error rate, retrieval misses, tool failures, and quality regressions.

### Evaluation pipelines

Evaluation should run before release and after release. Offline evals catch regressions early; online evals verify business impact.

### Versioning

In LLM systems, versioning covers more than weights. You should version prompts, model choice, retrieval settings, tool schemas, and eval datasets.

---

## Key Skills

### Tracking experiments

In practice, that means comparing runs with enough metadata to explain why a system improved or regressed.

### CI/CD for AI systems

Good AI CI/CD includes unit tests, prompt tests, schema checks, eval gates, and rollout controls.

### Observability

Observability means you can explain failures after the fact, not just notice them. Traces, structured logs, and per-component metrics matter.

### Prompt and version management

A mature team treats prompts as versioned assets with test coverage, review history, and rollback paths.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| MLflow | Track runs, metrics, and artifacts | Open-source experiment tracking and registry |
| Weights & Biases | Experiment tracking and collaboration | Research-heavy teams and richer dashboards |
| LangSmith | LLM traces, evals, and debugging | LLM app observability and regression testing |
| Docker | Portable packaging for training and serving | Reproducible deployments |
| Kubernetes | Scalable orchestration platform | Multi-service production AI systems |

---

## Projects

### Model deployment pipeline

- Goal: Train, package, register, and deploy an ML model with rollback support.
- Key components: build pipeline, registry, deployment gate, canary or blue-green strategy.
- Suggested tech stack: MLflow, Docker, FastAPI, Kubernetes.
- Difficulty: Advanced.

### Evaluation dashboard

- Goal: Show quality, cost, latency, and failure trends across model or prompt versions.
- Key components: run store, metrics API, score slices, trend charts.
- Suggested tech stack: LangSmith or MLflow, Postgres, Grafana or Streamlit.
- Difficulty: Advanced.

### Experiment tracking system

- Goal: Standardize how experiments are logged and compared.
- Key components: config capture, dataset version linkage, artifact storage, metadata tags.
- Suggested tech stack: MLflow or W&B, object storage, Python.
- Difficulty: Intermediate.

### CI/CD for AI app

- Goal: Block releases when AI quality or structured output contracts regress.
- Key components: unit tests, eval suite, build artifacts, staged rollout.
- Suggested tech stack: GitHub Actions, pytest, Docker, LangSmith.
- Difficulty: Advanced.

---

## Example Code

```python
import mlflow

with mlflow.start_run(run_name="rag-eval-v3"):
    mlflow.log_param("model", "gpt-5.4-mini")
    mlflow.log_param("chunk_size", 600)
    mlflow.log_metric("answer_correctness", 0.84)
    mlflow.log_metric("avg_latency_ms", 920)
```

---

## Suggested Project Structure

```text
ai-cicd/
├── app/
├── evals/
├── infra/
├── .github/workflows/
├── docker/
└── README.md
```

---

## Interview Q&A

#### How does LLMOps differ from traditional MLOps?

The pipeline shape is similar; the hard parts move. In MLOps the artifact is a trained model and the risks are drift, retraining, and reproducibility. In LLMOps the model is often someone else's and unchangeable, so the artifacts you version are **prompts, retrieval indexes, tool definitions, and model versions** — and the deployment risk is that a provider silently updates the model under you.

Other genuine differences: outputs are non-deterministic, so testing is statistical rather than exact; evaluation has no ground-truth label for most requests; cost is per-request and unbounded rather than a fixed serving cost; and latency is dominated by generation length, which the model itself decides. Security also enters the runtime path — prompt injection is a live attack surface that has no MLOps analogue.

What stays identical: you still need versioning, staged rollout, monitoring, rollback, and a regression suite. Candidates who claim LLMOps is entirely new usually haven't operated either.

#### What do you version in an LLM application, and why does it matter?

Everything that can change the output: the prompt template, the model ID *and* provider version, sampling parameters, the tool schemas, the retrieval index version and embedding model, the chunking configuration, and the eval suite version. Bundle them into one immutable release artifact rather than versioning each independently.

The reason is debuggability. Without it, "quality dropped last Tuesday" is unanswerable — you cannot tell whether someone edited a prompt, the index was reindexed, or the provider shipped a new model snapshot. Pinning provider model *snapshots* rather than floating aliases is the specific practice that saves you here.

#### How do you deploy a prompt change safely?

Like a code change, because it is one. It goes through review, runs against the regression suite in CI, and ships behind a flag with a staged rollout — a small traffic percentage first, watching quality signals, error rate, latency, and cost before widening. Keep the previous version live for instant rollback.

The additional LLM-specific step is a **shadow run**: send a sample of production traffic to the new prompt without serving its output, and compare against the current one offline. It catches regressions on real inputs that a curated eval set misses, at no user risk.

#### What do you monitor for an LLM feature in production?

Four layers:
- **System**: latency (TTFT and total, at p50/p95/p99), error rate by type, timeout rate, provider availability.
- **Cost**: tokens in and out per request, cost per request and per user, cache hit rate, and spend against budget with alerting *before* the limit.
- **Quality**: sampled automated evals, user feedback signals, abstention rate, guardrail trip rate, schema validation failure rate, and for agents the tool error rate and step count distribution.
- **Behavioral drift**: input distribution shift (are users asking different things?) and output length distribution, both of which move before quality metrics do.

The cheapest high-value alert is on **schema validation failures** — it catches provider model changes, prompt regressions, and malformed tool calls in one signal.

#### How do you trace a multi-step agent request for debugging?

Structured tracing with a span per step, all correlated by one request ID: the initial prompt, each model call with its full input and output, each tool invocation with arguments and result, retrieval queries and returned document IDs, retries, and the final response. Record token counts and latency per span so you can attribute cost and time to a specific step.

That's what LangSmith, Langfuse, and OpenTelemetry-based setups provide. The point is that agent failures are almost never in the last step — a bad answer usually traces back to a tool returning something unexpected five steps earlier, and without span-level traces you're guessing.

#### How do you control LLM spend?

Instrument before optimizing: cost per request broken down by feature, user, and route, so you know where the money goes. Then the levers in order of impact for most systems: prompt caching (input tokens usually dominate), trimming retrieved context, routing easy requests to a smaller model, semantic caching for repeated queries, capping `max_tokens`, and moving offline work to batch APIs.

Add hard controls too: per-user and per-tenant rate and spend limits, a global budget circuit breaker, and alerting on cost-per-request anomalies. An agent stuck in a retry loop can burn a month's budget in an afternoon, so a step limit and a per-request cost ceiling are not optional.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Prompts edited in production without review | No history, no rollback, no test coverage | Prompts in version control, reviewed and CI-tested |
| Floating model aliases instead of pinned snapshots | The provider updates the model and quality shifts silently | Pin snapshot versions; upgrade deliberately with an eval run |
| Versioning the prompt but not the index or embedding model | A reindex changes behavior with no trace | One immutable release artifact covering all components |
| No per-request cost ceiling or step limit | A looping agent can burn the budget in hours | Hard caps plus a budget circuit breaker |
| Logging only the final response | Agent failures originate several steps earlier | Span-level tracing with a shared request ID |
| Monitoring latency only in aggregate | TTFT and total latency have different causes | Track TTFT and per-token latency separately at p95/p99 |
| Shipping a prompt change to 100% of traffic | No safe way to observe a regression | Staged rollout behind a flag, with shadow comparison |
| No alerting on schema validation failures | The earliest signal of model or prompt regression is missed | Alert on validation failure rate |
| Treating evaluation as a launch gate only | Quality drifts as inputs and providers change | Continuous sampled evaluation in production |

---

## Related Topics

- [MLOps Overview](./README.md)
- [LLMOps](../ai_genai/intro_llmops.md)
- [Evaluation & Guardrails](./intro_evaluation_guardrails.md)
