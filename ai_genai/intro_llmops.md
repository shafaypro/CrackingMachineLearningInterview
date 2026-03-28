# LLMOps – Deploying & Operating LLMs in Production (2026 Edition)

**LLMOps** is the practice of deploying, monitoring, evaluating, and maintaining LLM-powered applications in production. It extends MLOps with LLM-specific concerns: prompt management, evals, hallucination detection, cost control, and guardrails.

---

## LLMOps vs MLOps

| MLOps | LLMOps |
|-------|--------|
| Model training pipelines | Prompt engineering & versioning |
| Feature engineering | Context management |
| Model versioning | Model version + prompt version |
| Accuracy metrics | LLM-specific evals (faithfulness, relevancy) |
| Data drift | Prompt/response drift |
| Latency/throughput | Token costs, latency per token |
| A/B testing | Prompt A/B testing |

---

## The LLMOps Stack

```
┌─────────────────────────────────────────────────────┐
│                    Application Layer                  │
│         (your RAG app, chatbot, agent, etc.)         │
└──────────────────┬────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────┐
│                  LLMOps Platform                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  │
│  │ Tracing  │  │  Evals   │  │ Prompts  │  │ Cost  │  │
│  │(LangSmith│  │  (RAGAS, │  │(PromptHub│  │Tracking│ │
│  │ Arize,   │  │  custom) │  │ Helicone)│  │       │  │
│  │ Langfuse)│  │          │  │          │  │       │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┘  │
└──────────────────┬────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────┐
│               Model / API Layer                        │
│     Anthropic Claude / OpenAI / Bedrock / etc.        │
└─────────────────────────────────────────────────────────┘
```

---

## Observability & Tracing

### LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "production-rag"

# Automatic tracing of all LangChain calls
# View at smith.langchain.com: latency, tokens, errors, full traces
```

### Langfuse (Open-source)

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

@observe()
def rag_query(question: str) -> str:
    """Automatically traced function."""
    docs = retrieve(question)
    answer = generate(question, docs)
    return answer

# Manual tracing
trace = langfuse.trace(name="rag-query", user_id="user-123")
span = trace.span(name="retrieval")
docs = retrieve(question)
span.end(output={"doc_count": len(docs)})

generation = trace.generation(
    name="llm-call",
    model="claude-sonnet-4-6",
    input=[{"role": "user", "content": question}],
    output=answer,
    usage={"input": 500, "output": 200}
)
trace.update(output=answer)
```

### Helicone (API Proxy with analytics)

```python
# Just change base URL — automatic logging, cost tracking
from anthropic import Anthropic

client = Anthropic(
    base_url="https://anthropic.helicone.ai",
    default_headers={
        "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
        "Helicone-User-Id": "user-123",         # user-level analytics
        "Helicone-Session-Id": "session-456",   # session tracking
    }
)
```

---

## Evaluations (Evals)

Evals are automated tests that measure LLM output quality. They're the unit tests of LLMOps.

### Types of Evals

| Type | What it measures | How |
|------|-----------------|-----|
| **Exact match** | Does output match expected exactly? | String comparison |
| **Contains** | Does output contain required info? | Regex/substring |
| **LLM-as-judge** | Is output high quality? | Another LLM grades it |
| **RAGAS** | RAG-specific quality metrics | Framework |
| **Human eval** | Human rates output quality | Annotation tools |

### LLM-as-Judge

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class EvalResult:
    score: float  # 0.0 - 1.0
    reason: str

def llm_judge(
    question: str,
    answer: str,
    context: str,
    criteria: str = "helpfulness and accuracy"
) -> EvalResult:
    """Use an LLM to judge the quality of another LLM's answer."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="""You are an expert evaluator for LLM outputs.
Rate the answer on a scale of 0.0 to 1.0 and explain your reasoning.
Return JSON: {"score": 0.X, "reason": "..."}""",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this answer on: {criteria}

Question: {question}
Context: {context}
Answer: {answer}

Return JSON with score (0.0-1.0) and reason."""
        }]
    )

    import json
    result = json.loads(response.content[0].text)
    return EvalResult(score=result["score"], reason=result["reason"])

# Run eval
result = llm_judge(
    question="What is dbt used for?",
    answer="dbt transforms data in your data warehouse using SQL",
    context="dbt is a transformation tool that runs SQL models...",
    criteria="accuracy and completeness"
)
print(f"Score: {result.score} | {result.reason}")
```

### RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Is answer grounded in context?
    answer_relevancy,       # Does answer address the question?
    context_precision,      # Are retrieved docs relevant?
    context_recall,         # Did we retrieve all needed docs?
    answer_correctness,     # Is the answer factually correct?
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is dbt?", "How does Kafka work?"],
    "answer": ["dbt transforms data in warehouses", "Kafka is a distributed event streaming platform"],
    "contexts": [
        ["dbt (Data Build Tool) is a transformation tool..."],
        ["Apache Kafka is a distributed streaming platform..."]
    ],
    "ground_truth": ["dbt is a SQL transformation tool", "Kafka streams events between systems"]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(results)
# faithfulness: 0.92, answer_relevancy: 0.88, ...
```

### Eval Suites (Continuous Testing)

```python
# Run evals on every deployment in CI/CD
# .github/workflows/eval.yml

import pytest
import anthropic

client = anthropic.Anthropic()

EVAL_CASES = [
    {
        "input": "What is the capital of France?",
        "expected_contains": ["Paris"],
        "max_tokens": 100,
    },
    {
        "input": "Explain SQL JOINs",
        "min_length": 100,
        "expected_contains": ["INNER JOIN", "LEFT JOIN"],
    }
]

@pytest.mark.parametrize("case", EVAL_CASES)
def test_llm_response(case):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=case.get("max_tokens", 500),
        messages=[{"role": "user", "content": case["input"]}]
    )
    text = response.content[0].text

    if "expected_contains" in case:
        for phrase in case["expected_contains"]:
            assert phrase.lower() in text.lower(), f"Expected '{phrase}' in response"

    if "min_length" in case:
        assert len(text) >= case["min_length"]
```

---

## Prompt Management

### Prompt Versioning

```python
# Store prompts in code (version controlled)
# prompts/rag_system.py

RAG_SYSTEM_V1 = """You are a helpful assistant. Answer based on context."""

RAG_SYSTEM_V2 = """You are an expert assistant. Answer questions based ONLY on the provided context.
If the answer isn't in the context, say "I don't have that information."
Always cite the source document when possible."""

# Or use LangSmith Hub
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")   # community prompt
```

### A/B Testing Prompts

```python
import random
from langfuse import Langfuse

langfuse = Langfuse()

def get_prompt(variant: str = None) -> str:
    if variant is None:
        variant = random.choice(["control", "treatment"])

    prompts = {
        "control": "Answer the question helpfully.",
        "treatment": "Answer the question as a senior expert. Be concise and practical.",
    }

    # Track which variant was used
    langfuse.trace(name="prompt-ab-test").update(metadata={"variant": variant})
    return prompts[variant]
```

---

## Cost Management

### Token Usage Tracking

```python
from anthropic import Anthropic
from dataclasses import dataclass, field
from collections import defaultdict

client = Anthropic()

# Pricing (per million tokens, as of early 2026)
PRICING = {
    "claude-opus-4-6":          {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":        {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":{"input": 0.25,  "output": 1.25},
}

class CostTracker:
    def __init__(self):
        self.usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0})

    def track(self, model: str, response):
        usage = response.usage
        price = PRICING.get(model, {"input": 3.0, "output": 15.0})

        cost = (
            usage.input_tokens * price["input"] / 1_000_000 +
            usage.output_tokens * price["output"] / 1_000_000
        )

        self.usage[model]["input_tokens"] += usage.input_tokens
        self.usage[model]["output_tokens"] += usage.output_tokens
        self.usage[model]["cost_usd"] += cost

    def report(self):
        for model, stats in self.usage.items():
            print(f"{model}: ${stats['cost_usd']:.4f} "
                  f"({stats['input_tokens']} in, {stats['output_tokens']} out)")

tracker = CostTracker()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
tracker.track("claude-sonnet-4-6", response)
tracker.report()
```

### Cost Optimization Strategies

```python
# 1. Use appropriate model tier
def route_to_model(task_complexity: str) -> str:
    return {
        "simple": "claude-haiku-4-5-20251001",   # FAQs, classification
        "medium": "claude-sonnet-4-6",            # most tasks
        "complex": "claude-opus-4-6",             # deep reasoning
    }[task_complexity]

# 2. Context caching (up to 90% savings on repeated context)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": LARGE_DOCUMENT,   # 50K token doc
        "cache_control": {"type": "ephemeral"}  # cache it
    }],
    messages=[{"role": "user", "content": question}]
)

# 3. Batch API (50% cheaper, async)
batch = client.messages.batches.create(requests=[...])

# 4. max_tokens optimization — set realistic limits
# Don't use max_tokens=4096 for a one-sentence answer task

# 5. Truncate retrieved context to essentials
def trim_context(docs: list, max_chars: int = 4000) -> str:
    context = ""
    for doc in docs:
        if len(context) + len(doc) > max_chars:
            break
        context += doc + "\n---\n"
    return context
```

---

## Guardrails

```python
import re

class GuardrailError(Exception):
    pass

class Guardrails:
    """Input and output safety checks."""

    # Input guardrails
    def check_input(self, user_input: str) -> str:
        # Length check
        if len(user_input) > 10_000:
            raise GuardrailError("Input too long")

        # PII detection (simplified)
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
            r"\b\d{16}\b",                        # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]
        for pattern in pii_patterns:
            if re.search(pattern, user_input):
                user_input = re.sub(pattern, "[REDACTED]", user_input)

        # Prompt injection detection
        injection_keywords = [
            "ignore previous instructions",
            "disregard your guidelines",
            "you are now",
        ]
        if any(kw in user_input.lower() for kw in injection_keywords):
            raise GuardrailError("Potential prompt injection detected")

        return user_input

    # Output guardrails
    def check_output(self, output: str) -> str:
        # Check for hallucination signals
        uncertainty_phrases = ["I'm not sure", "I think", "I believe", "might be"]

        # Check for harmful content (in production: use dedicated classifiers)
        # ...

        return output

    def check_relevance(self, question: str, answer: str) -> bool:
        """Use LLM to verify answer is relevant to question."""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Is this answer relevant to the question? Answer yes/no.\nQuestion: {question}\nAnswer: {answer}"
            }]
        )
        return "yes" in response.content[0].text.lower()

# Use in application
guardrails = Guardrails()

def safe_query(user_input: str) -> str:
    try:
        clean_input = guardrails.check_input(user_input)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": clean_input}]
        )
        output = response.content[0].text
        return guardrails.check_output(output)
    except GuardrailError as e:
        return f"I can't process that request: {e}"
```

---

## Production Checklist

```markdown
## LLMOps Production Checklist

### Observability
- [ ] Tracing enabled (LangSmith / Langfuse / Arize)
- [ ] Token usage logged per request
- [ ] Cost tracking per user/feature
- [ ] Error rates monitored
- [ ] Latency p50/p95/p99 tracked

### Reliability
- [ ] Retry logic with exponential backoff
- [ ] Fallback model configured
- [ ] Circuit breaker for API failures
- [ ] Graceful degradation
- [ ] Rate limit handling

### Quality
- [ ] Eval suite running in CI/CD
- [ ] Golden dataset for regression testing
- [ ] LLM-as-judge for subjective metrics
- [ ] Human eval sample review weekly

### Cost
- [ ] Model routing by task complexity
- [ ] Context caching implemented where applicable
- [ ] max_tokens optimized per use case
- [ ] Monthly budget alerts set

### Safety
- [ ] Input sanitization
- [ ] PII detection/redaction
- [ ] Output content filtering
- [ ] Prompt injection protection
- [ ] Audit logs retained

### Deployment
- [ ] Prompt versions in source control
- [ ] Canary deployments for prompt changes
- [ ] Rollback plan documented
- [ ] SLA defined and monitored
```

---

## LLMOps Tools in 2026

| Tool | Category | Description |
|------|----------|-------------|
| **LangSmith** | Tracing + Evals | LangChain's observability platform |
| **Langfuse** | Tracing + Evals | Open-source, self-hostable |
| **Helicone** | Proxy + Analytics | API proxy with analytics |
| **Arize AI** | Observability | Enterprise LLM monitoring |
| **Weights & Biases** | Experiment tracking | ML + LLM experiments |
| **RAGAS** | Evals | RAG evaluation metrics |
| **PromptFlow** | Prompt management | Microsoft Azure |
| **Braintrust** | Evals + Tracing | Developer-friendly evals |
| **Phoenix** | Tracing + Evals | Open-source by Arize |
| **Guardrails AI** | Guardrails | Output validation framework |
| **NeMo Guardrails** | Guardrails | NVIDIA's guardrail toolkit |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [RAG](./intro_rag.md) | Evaluating and monitoring RAG pipelines is a core LLMOps concern |
| [Vector Databases](./intro_vector_databases.md) | Monitoring embedding quality and retrieval performance |
| [Agentic AI](./intro_agentic_ai.md) | Observability for long-running agent workflows |
| [LangChain](./intro_langchain.md) | LangSmith is LangChain's LLMOps platform |
| [Anthropic Overview](./intro_anthropic.md) | Claude API usage patterns and rate limit management |
