# LangSmith — LLM Observability & Evaluation Platform

## What is LangSmith?

LangSmith is Anthropic/LangChain's platform for **tracing, debugging, testing, and evaluating** LLM applications. It provides end-to-end visibility into every LLM call, chain execution, and agent decision — critical for production AI systems.

```
  Your App → LangSmith SDK → LangSmith Platform
               (trace)          ↓
                          [Traces Dashboard]
                          [Evaluation Runs]
                          [Prompt Playground]
                          [Datasets & Evals]
```

---

## Core Capabilities

| Capability | What It Does |
|------------|--------------|
| **Tracing** | Record every LLM call, tool use, chain step with inputs/outputs/latency |
| **Debugging** | Inspect failed runs, see exact prompts sent, token counts |
| **Datasets** | Curate golden datasets of (input, expected_output) pairs |
| **Evaluations** | Run automated evals: LLM-as-judge, exact match, custom metrics |
| **Prompt Hub** | Version-controlled prompt storage and deployment |
| **A/B Testing** | Compare prompt versions on the same dataset |
| **Monitoring** | Production dashboards for latency, cost, error rates |

---

## Setup

```python
# Install
# pip install langsmith langchain-anthropic

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-production-app"  # Project for grouping traces

# That's it — all LangChain/LangGraph calls are auto-traced
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
result = llm.invoke("Explain RAG in 3 sentences")
# This call now appears in LangSmith dashboard
```

---

## Manual Tracing (Non-LangChain Code)

```python
from langsmith import traceable, Client

client = Client()

@traceable(name="custom-retrieval", tags=["rag", "production"])
def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """Any function decorated with @traceable is auto-traced"""
    # Your retrieval logic here
    results = vector_db.search(query, top_k=top_k)
    return results

@traceable(name="rag-pipeline")
def run_rag(question: str) -> str:
    docs = retrieve_documents(question)
    context = "\n".join([d["text"] for d in docs])
    response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    return response.content

# Nested traces are automatically linked as parent → child
answer = run_rag("What is RLHF?")
```

---

## Dataset Creation & Evaluation

### Create a Dataset

```python
from langsmith import Client

client = Client()

# Create dataset from examples
examples = [
    {"input": {"question": "What is RAG?"},
     "output": {"answer": "RAG combines retrieval with generation..."}},
    {"input": {"question": "Explain RLHF"},
     "output": {"answer": "RLHF uses human feedback to fine-tune LLMs..."}},
]

dataset = client.create_dataset(
    dataset_name="ML Interview QA",
    description="Golden set for ML interview Q&A evaluation"
)

client.create_examples(
    inputs=[e["input"] for e in examples],
    outputs=[e["output"] for e in examples],
    dataset_id=dataset.id
)
```

### Run Evaluation

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# The function to evaluate
def my_rag_app(inputs: dict) -> dict:
    answer = run_rag(inputs["question"])
    return {"answer": answer}

# Evaluators
correctness_evaluator = LangChainStringEvaluator(
    "qa",  # Uses LLM to judge correctness
    config={"llm": ChatAnthropic(model="claude-opus-4-6")}
)

results = evaluate(
    my_rag_app,
    data="ML Interview QA",           # Dataset name
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-v2-test",  # For comparison in UI
    num_repetitions=1                 # Run each example N times
)

print(results.to_pandas())  # DataFrame with scores per example
```

---

## LLM-as-Judge Evaluators

```python
from langsmith.evaluation import evaluate
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Custom LLM-as-judge evaluator
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator. Score the answer 1-5 for accuracy."),
    ("human", """
Question: {input}
Expected: {reference}
Actual: {prediction}

Return JSON: {{"score": <1-5>, "reasoning": "<why>"}}
""")
])

def custom_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    judge_llm = ChatAnthropic(model="claude-opus-4-6")
    response = judge_llm.invoke(judge_prompt.format_messages(
        input=inputs["question"],
        reference=reference_outputs["answer"],
        prediction=outputs["answer"]
    ))
    import json
    result = json.loads(response.content)
    return {"score": result["score"] / 5.0, "comment": result["reasoning"]}

results = evaluate(
    my_rag_app,
    data="ML Interview QA",
    evaluators=[custom_judge],
    experiment_prefix="claude-judge-v1"
)
```

---

## Prompt Hub — Version-Controlled Prompts

```python
from langchain import hub

# Pull a prompt from LangSmith Prompt Hub
prompt = hub.pull("my-org/rag-qa-prompt:v3")  # Pin to version

# Use it
chain = prompt | llm
result = chain.invoke({"context": "...", "question": "..."})

# Push updated prompt
hub.push("my-org/rag-qa-prompt", updated_prompt, new_repo_is_public=False)
```

---

## Production Monitoring

LangSmith captures these metrics automatically per project:

| Metric | Description |
|--------|-------------|
| **Latency P50/P95/P99** | End-to-end and per-node latency |
| **Token usage** | Input + output tokens per call |
| **Cost** | Estimated cost based on model pricing |
| **Error rate** | % of runs that raised exceptions |
| **Feedback scores** | User thumbs up/down or custom ratings |

### Attaching User Feedback

```python
from langsmith import Client

client = Client()

# After getting user feedback on a response
run_id = "..."  # From trace

client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=0.8,  # 0-1 scale
    comment="Good answer but missing implementation detail"
)
```

---

## Regression Testing Workflow

```python
# CI/CD pipeline integration
# 1. Add new test cases when bugs are found
# 2. Run eval suite on each PR
# 3. Block merge if eval score drops > 5%

import subprocess

def run_eval_gate(threshold: float = 0.85):
    results = evaluate(
        my_rag_app,
        data="ML Interview QA - Regression",
        evaluators=[correctness_evaluator]
    )

    avg_score = results.to_pandas()["feedback.qa"].mean()

    if avg_score < threshold:
        raise ValueError(f"Eval score {avg_score:.2f} below threshold {threshold}")

    print(f"Eval passed: {avg_score:.2f} >= {threshold}")

# In GitHub Actions:
# - run: python -c "from eval import run_eval_gate; run_eval_gate(0.85)"
```

---

## Tracing Architecture

```
Request comes in
       │
       ▼
  [Root Run]  ← LangSmith traces this
  ├── [LLM Call 1]  {prompt, response, latency, tokens}
  ├── [Tool: search_web]  {input, output, duration}
  │   └── [HTTP Request]  {url, status, latency}
  ├── [LLM Call 2]  {prompt, response}
  └── [Output Parser]  {raw, parsed}
       │
       ▼
   Response returned
```

---

## LangSmith vs Alternatives

| Platform | Strength | Weakness |
|----------|----------|----------|
| **LangSmith** | Deep LangChain integration, evals, datasets | Paid for production scale |
| **Weights & Biases** | ML experiment tracking, sweeps | Less LLM-native |
| **Arize AI** | Enterprise ML monitoring, drift | Complex setup |
| **Helicone** | Simple proxy-based tracing | Less eval tooling |
| **Literal AI** | Open source, self-hostable | Smaller ecosystem |
| **Phoenix (Arize)** | Open source observability | Less eval automation |

---

## Interview Questions

**Q: Why is LangSmith important in production LLM systems?**
> Without observability, LLM apps are black boxes. LangSmith traces every call, showing exact prompts sent, tokens used, latency, and errors. This is essential for debugging hallucinations, optimizing costs, and catching regressions when prompts or models change.

**Q: How do you evaluate whether a RAG system improved after a change?**
> Create a golden dataset with (question, expected_answer) pairs. Run the old and new system on the same dataset with LLM-as-judge evaluators. Compare scores in LangSmith's experiment comparison view. Block deployment if score drops below threshold.

**Q: What's the difference between online and offline evaluation?**
> Offline: evaluate on a fixed dataset before deployment (regression tests, benchmarks). Online: collect real user feedback and monitor metrics in production (thumbs up/down, implicit signals like follow-up questions). Both are needed — offline catches regressions, online catches real-world failures.

**Q: How would you implement regression testing for an LLM application?**
> Curate a golden dataset of critical test cases. Integrate `evaluate()` into CI/CD. Set a score threshold (e.g., 85%). Add new test cases whenever a production bug is found (golden negative examples). This creates a growing safety net.

**Q: What is LLM-as-judge and what are its limitations?**
> Using a powerful LLM (like Claude Opus) to score another LLM's output. Limitations: positional bias (judges prefer first option), verbosity bias (longer = better), self-consistency issues, and cost. Mitigate with structured scoring rubrics, multiple judge calls, and human validation on a sample.
