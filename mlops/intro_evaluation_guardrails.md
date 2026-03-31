# Evaluation and Guardrails for AI Systems

This guide focuses on measuring AI system quality and preventing unsafe, unreliable, or low-quality behavior in production.

---

## Overview

Evaluation answers "How good is the system?" Guardrails answer "How do we stop bad behavior before it causes harm?" Modern AI teams need both because model demos can look impressive while still failing badly under edge cases, hallucination pressure, or adversarial inputs.

---

## Core Concepts

### LLM evaluation

Evaluation can be offline against labeled datasets, model-graded with rubric prompts, human-reviewed for nuanced tasks, or online through product metrics and experiments.

### Safety and moderation

Safety layers filter harmful, disallowed, or policy-violating inputs and outputs. In practice, moderation is often a separate subsystem, not just a prompt instruction.

### Testing AI systems

AI testing includes prompt regression tests, schema validation tests, adversarial test cases, and hallucination or refusal checks.

### Reliability

Reliability means the system behaves acceptably across expected and unexpected inputs, not only on golden path demos.

---

## Key Skills

### Designing eval datasets

A strong eval dataset includes representative real tasks, difficult edge cases, adversarial or failure-inducing inputs, and clear scoring criteria.

### Automated testing

In practice, this means CI checks that compare outputs, validate formats, and fail fast on regressions.

### Red-teaming

Red-teaming deliberately probes the system for policy bypasses, unsafe outputs, prompt injection, and hallucinations.

### Output validation

This includes JSON/schema validation, citation presence checks, allowed-action enforcement, and business-rule validation.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| OpenAI Evals | Framework for structured evaluation tasks | Automated regression and benchmark runs |
| LangSmith | Trace inspection and eval workflows | LLM app debugging and prompt evaluation |
| Custom eval frameworks | Business-specific scoring and datasets | Domain-specific quality measurement |
| Pydantic | Structured output validation | Guardrails for schemas and contracts |
| Moderation APIs | Safety classification for text/images | Input and output risk filtering |

---

## Projects

### Eval pipeline

- Goal: Run repeatable quality checks for prompts, models, or RAG changes.
- Key components: dataset loader, scoring rubric, batch runner, result storage, regression thresholds.
- Suggested tech stack: Python, pandas, pytest, OpenAI Evals or custom harness.
- Difficulty: Intermediate.

### AI safety filter

- Goal: Block unsafe or policy-violating interactions before final output reaches the user.
- Key components: input moderation, output moderation, escalation paths, audit logs.
- Suggested tech stack: moderation API, FastAPI, Redis, Postgres.
- Difficulty: Intermediate.

### Hallucination detection system

- Goal: Detect unsupported claims in generated answers.
- Key components: citation checks, retrieval overlap scoring, claim extraction, rule-based validation.
- Suggested tech stack: Python, RAG stack, evaluation framework.
- Difficulty: Advanced.

### Benchmark suite

- Goal: Compare multiple prompts or models on a stable internal benchmark.
- Key components: frozen test set, scoring dashboard, trend tracking, failure slices.
- Suggested tech stack: Python, Jupyter, MLflow or LangSmith.
- Difficulty: Advanced.

---

## Example Code

```python
from pydantic import BaseModel, ValidationError

class Answer(BaseModel):
    final_answer: str
    confidence: float
    citations: list[str]

def validate_output(payload: dict) -> bool:
    try:
        Answer.model_validate(payload)
        return True
    except ValidationError:
        return False
```

---

## Suggested Project Structure

```text
eval-pipeline/
├── datasets/
├── rubrics/
├── src/
│   ├── runner.py
│   ├── scorers.py
│   ├── validators.py
│   └── reports.py
├── tests/
└── README.md
```

---

## Related Topics

- [LLM Evaluation](./intro_llm_evaluation.md)
- [LLMOps](../ai_genai/intro_llmops.md)
- [Multi-Model Orchestration](../ai_genai/intro_multi_model_orchestration.md)
