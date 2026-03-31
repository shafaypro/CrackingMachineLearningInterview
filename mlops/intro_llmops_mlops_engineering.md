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

## Related Topics

- [MLOps Overview](./README.md)
- [LLMOps](../ai_genai/intro_llmops.md)
- [Evaluation & Guardrails](./intro_evaluation_guardrails.md)
