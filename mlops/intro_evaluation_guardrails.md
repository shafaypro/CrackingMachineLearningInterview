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

## Interview Q&A

#### How do you evaluate a system whose outputs are non-deterministic and open-ended?

Convert "is this good?" into checkable claims, in layers:
- **Deterministic checks** first — schema validity, required fields present, citation spans exist, no PII in output, length bounds. These are cheap, exact, and catch most regressions.
- **Reference-based metrics** where a correct answer exists — exact match, F1 over extracted fields, or retrieval recall.
- **LLM-as-judge** for subjective dimensions (helpfulness, faithfulness, tone), validated against human labels on a sample before you trust it.
- **Human review** on a rotating sample and on everything the judge scores near its threshold.

Then fix a **regression suite**: a versioned set of cases with expected properties, run on every prompt, model, retrieval, or tool change. Non-determinism is handled by running each case several times and tracking the pass *rate*, not a single pass/fail.

#### How do you validate an LLM judge?

Treat it as a model you're deploying, because it is. Label a few hundred examples by hand, then measure the judge's agreement with those labels — Cohen's kappa or simple agreement rate, and crucially agreement on the *disagreement cases*, since a judge that only agrees on obvious examples is useless.

Then test for the known biases: **position bias** (swap the order of two compared answers and check the verdict flips at chance rate, not systematically), **verbosity bias** (longer answers scoring higher regardless of quality), **self-preference** (a judge favoring outputs from its own model family), and score compression (everything lands at 4/5). Use a rubric with concrete criteria rather than "rate 1–10", and prefer pairwise comparison over absolute scoring — it's substantially more reliable.

#### What guardrails would you put around a customer-facing LLM feature?

Input side: prompt-injection detection, PII detection and redaction before the text reaches the model, length and rate limits, and topic classification to reject out-of-scope requests early.

Output side: schema validation for structured responses, a safety/moderation classifier, PII scanning on the way out (models can echo back what was retrieved), citation verification for grounded answers, and a claim-checker for high-stakes domains.

Around both: a kill switch to disable the feature without a deploy, a fallback response for when a guardrail trips, per-user rate and spend limits, and full request/response logging with retention that legal has agreed to.

The framing that matters: guardrails are a **defense-in-depth system with a failure mode of its own**. Over-blocking is a real cost — measure the false-positive rate of your guardrails, not just their catch rate, or you'll ship a feature that refuses legitimate requests.

#### How do you red-team an AI system before launch?

Systematically, against a threat model rather than by improvising. Enumerate what an attacker wants — extract the system prompt, exfiltrate other users' data, make the model take an unauthorized action through a tool, produce harmful content, or run up cost. Then attack each: direct and indirect prompt injection (including content the system retrieves, which is the underrated vector), jailbreak patterns, role-play framing, encoding tricks, and multi-turn escalation where each message is individually benign.

Record every successful attack as a permanent regression test. The measure of a red-team exercise is not how many issues it found but whether the same attack class fails on the next release — which only happens if the findings become tests.

#### How do you detect hallucination in production, not just in evals?

No single signal is sufficient, so combine cheap ones: **groundedness checking** (does each claim appear in the retrieved context — an NLI model or a judge call on a sample), **self-consistency** (sample the answer several times and flag high disagreement), **citation validation** (do the cited spans actually exist and support the claim), and **user signals** (thumbs-down, rephrasing, escalation to a human).

For rate-limited cost, sample rather than checking everything: score 1–5% of traffic continuously and alert on the rate rather than individual cases. And monitor the abstention rate — a fall in "I don't know" responses often precedes a rise in confident fabrication.

#### What do you do when evaluation results and user feedback disagree?

Trust the users and fix the eval. The disagreement means the eval set doesn't represent real traffic — usually because it was written by the team rather than sampled from production, so it under-represents ambiguous, adversarial, or multi-turn cases.

The fix is a feedback loop: sample real failures, label them, and add them to the eval set, so the suite converges on reality over time. Also check for the mundane explanations first — the eval runs a different prompt version, different retrieval settings, or a different model than production.

---

## Common Pitfalls

| Pitfall | Why it hurts | Fix |
|---|---|---|
| Vibes-based evaluation | No way to detect regressions or compare versions | A versioned regression suite run on every change |
| Trusting an LLM judge without validation | Position, verbosity, and self-preference biases go unmeasured | Validate against human labels; use pairwise comparison |
| Eval set written by the team, not sampled from traffic | Misses ambiguous and adversarial real cases | Continuously add labeled production failures |
| Single run per eval case | Non-determinism reads as a regression | Multiple runs; track pass rate |
| Measuring guardrail catch rate only | Over-blocking silently destroys the user experience | Track false-positive rate as an equal metric |
| Red-team findings not converted to tests | The same attack class returns next release | Every successful attack becomes a regression case |
| Guardrails only on input | Models echo retrieved PII and unsupported claims outward | Symmetric input and output checks |
| No kill switch | Disabling a broken feature requires a deploy | Runtime feature flag with a safe fallback response |
| Evaluating end to end only | Cannot localize whether retrieval, prompt, or model regressed | Stage-level metrics alongside end-to-end |

---

## Related Topics

- [LLM Evaluation](./intro_llm_evaluation.md)
- [LLMOps](../ai_genai/intro_llmops.md)
- [Multi-Model Orchestration](../ai_genai/intro_multi_model_orchestration.md)
