# LLM Evaluation — Testing, Benchmarks & Production Evals

## Why LLM Evaluation is Critical

LLM outputs are probabilistic and hard to test with traditional unit tests. Evaluation is the discipline of systematically measuring whether your LLM application is doing what you want — both before and after deployment.

```
Traditional Software:          LLM Applications:
input → deterministic fn       input → probabilistic LLM
  → exact output                  → variable output
  → pass/fail test                → quality score (0-1)
```

---

## Evaluation Taxonomy

| Category | Question | Examples |
|----------|----------|---------|
| **Offline eval** | Does the system work before deployment? | Benchmarks, golden datasets |
| **Online eval** | Does it work in production? | User feedback, implicit signals |
| **Unit eval** | Does component X work? | Retriever precision, reranker quality |
| **End-to-end eval** | Does the full pipeline work? | RAG answer quality |
| **Safety eval** | Is it safe? | Toxicity, bias, jailbreak resistance |
| **Cost/latency eval** | Is it efficient? | Token count, response time |

---

## Core Evaluation Metrics

### Correctness Metrics

```python
# Exact Match — for factoid QA
def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())

# F1 Token Overlap — for extractive QA (SQuAD-style)
def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ROUGE for summarization
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
scores = scorer.score(prediction, reference)
```

### Semantic Similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# Score: 1.0 = identical, 0.0 = orthogonal, >0.8 = semantically similar
```

---

## LLM-as-Judge

Using a powerful LLM to evaluate another LLM's output. Standard in 2026 for open-ended generation tasks.

### Basic LLM Judge

```python
from anthropic import Anthropic
from pydantic import BaseModel, Field
import json

class EvaluationResult(BaseModel):
    score: float = Field(ge=0, le=1, description="Quality score 0-1")
    reasoning: str = Field(description="Detailed reasoning for the score")
    strengths: list[str]
    weaknesses: list[str]
    verdict: str = Field(description="One-sentence summary")

def llm_judge(
    question: str,
    answer: str,
    reference: str | None = None,
    criteria: str = "accuracy, completeness, and clarity"
) -> EvaluationResult:
    client = Anthropic()

    ref_section = f"\nReference answer: {reference}" if reference else ""

    response = client.messages.create(
        model="claude-opus-4-6",  # Use strongest model as judge
        max_tokens=1024,
        tools=[{
            "name": "submit_evaluation",
            "description": "Submit evaluation result",
            "input_schema": EvaluationResult.model_json_schema()
        }],
        tool_choice={"type": "tool", "name": "submit_evaluation"},
        messages=[{
            "role": "user",
            "content": f"""Evaluate this answer based on {criteria}.

Question: {question}{ref_section}

Answer to evaluate: {answer}

Be strict and objective."""
        }]
    )

    return EvaluationResult.model_validate(response.content[0].input)

# Example
result = llm_judge(
    question="What is the difference between L1 and L2 regularization?",
    answer="L1 adds absolute value of weights, L2 adds squared weights to the loss.",
    reference="L1 (Lasso) regularization adds |w| to loss, promoting sparsity. L2 (Ridge) adds w² to loss, penalizing large weights without inducing sparsity.",
    criteria="technical accuracy and completeness for an ML interview"
)
print(f"Score: {result.score:.2f} | {result.verdict}")
```

### Pairwise Comparison (A/B Eval)

```python
def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Returns which answer is better and why — controls for verbosity bias"""
    client = Anthropic()

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Compare these two answers to: "{question}"

Answer A: {answer_a}

Answer B: {answer_b}

Which is more accurate and complete for an ML interview?
Return JSON: {{"winner": "A" or "B" or "tie", "reason": "...", "confidence": 0-1}}"""
        }]
    )

    return json.loads(response.content[0].text)

# Run both orders to detect position bias
result_1 = pairwise_judge(q, model_a_output, model_b_output)
result_2 = pairwise_judge(q, model_b_output, model_a_output)
```

---

## Prompt Testing Frameworks

### Promptfoo — CLI-based Prompt Testing

```yaml
# promptfooconfig.yaml
description: ML Interview Bot Evaluation

prompts:
  - "You are an ML expert. Answer: {{question}}"
  - "As an ML interview coach with 10 years experience, explain: {{question}}"

providers:
  - anthropic:claude-sonnet-4-6
  - anthropic:claude-haiku-4-5

tests:
  - description: "L1 vs L2 regularization"
    vars:
      question: "What is the difference between L1 and L2 regularization?"
    assert:
      - type: contains
        value: "Lasso"
      - type: contains
        value: "Ridge"
      - type: llm-rubric
        value: "Explains sparsity effect of L1"

  - description: "Explain attention mechanism"
    vars:
      question: "How does the attention mechanism work in transformers?"
    assert:
      - type: llm-rubric
        value: "Mentions query, key, value matrices and softmax"
      - type: not-contains
        value: "I don't know"
```

```bash
# Run evaluation
npx promptfoo eval

# View results in browser
npx promptfoo view
```

### Custom Eval Framework in Python

```python
from dataclasses import dataclass
from typing import Callable
import asyncio

@dataclass
class EvalCase:
    id: str
    input: dict
    expected: str | None = None
    tags: list[str] = None

@dataclass
class EvalResult:
    case_id: str
    score: float
    output: str
    reasoning: str
    passed: bool

class EvalSuite:
    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold
        self.cases: list[EvalCase] = []
        self.evaluators: list[Callable] = []

    def add_case(self, case: EvalCase):
        self.cases.append(case)

    def add_evaluator(self, fn: Callable):
        self.evaluators.append(fn)

    async def run(self, app_fn: Callable) -> dict:
        results = []
        for case in self.cases:
            output = await app_fn(case.input)
            scores = [await ev(case, output) for ev in self.evaluators]
            avg_score = sum(scores) / len(scores)
            results.append(EvalResult(
                case_id=case.id,
                score=avg_score,
                output=output,
                reasoning="",
                passed=avg_score >= self.threshold
            ))

        pass_rate = sum(1 for r in results if r.passed) / len(results)
        return {
            "suite": self.name,
            "pass_rate": pass_rate,
            "avg_score": sum(r.score for r in results) / len(results),
            "results": results,
            "passed": pass_rate >= self.threshold
        }
```

---

## Regression Testing for AI Outputs

### Core Principle

Every production bug becomes a test case. Build a growing safety net.

```python
# When a bug is found in production:
# 1. Add the failing case to the dataset
# 2. Fix the bug (prompt, retrieval, model)
# 3. Verify the case now passes
# 4. Ensure it doesn't regress in future

class RegressionTestManager:
    def __init__(self, langsmith_client):
        self.client = langsmith_client

    def add_bug_as_test(self, run_id: str, correct_output: str, bug_description: str):
        """Convert a production failure into a regression test"""
        run = self.client.read_run(run_id)
        self.client.create_example(
            inputs=run.inputs,
            outputs={"answer": correct_output},
            dataset_name="regression-tests",
            metadata={"bug": bug_description, "run_id": run_id}
        )

    def run_regression_suite(self, app_fn, pass_threshold: float = 0.95):
        from langsmith.evaluation import evaluate
        results = evaluate(app_fn, data="regression-tests", evaluators=[...])
        pass_rate = results.to_pandas()["score"].mean()
        assert pass_rate >= pass_threshold, f"Regression suite failed: {pass_rate:.2%}"
```

---

## Hallucination Detection

### Rule-Based Detection

```python
def detect_hallucination_signals(answer: str, context: str) -> dict:
    signals = {}

    # 1. Confident language without context support
    confidence_phrases = ["definitely", "certainly", "always", "never", "100%"]
    signals["overconfident"] = any(p in answer.lower() for p in confidence_phrases)

    # 2. Specific numbers not in context
    import re
    answer_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
    context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
    novel_numbers = answer_numbers - context_numbers
    signals["novel_numbers"] = list(novel_numbers)

    # 3. Named entities not in context (requires NER)
    # signals["novel_entities"] = find_novel_entities(answer, context)

    return signals
```

### LLM-Based Faithfulness Check

```python
def check_faithfulness(context: str, answer: str) -> float:
    """Returns 0-1 faithfulness score"""
    client = Anthropic()

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyze each claim in the answer and determine if it's supported by the context.

Context: {context}

Answer: {answer}

Return JSON: {{
  "supported_claims": [...],
  "unsupported_claims": [...],
  "faithfulness_score": 0.0-1.0
}}"""
        }]
    )

    result = json.loads(response.content[0].text)
    return result["faithfulness_score"]
```

---

## Human-in-the-Loop Evaluation

### When Humans Must Evaluate

- **Subjective quality**: Creativity, tone, helpfulness
- **Safety**: Sensitive topics, harmful content
- **Domain expertise**: Medical, legal, financial accuracy
- **Ground truth creation**: Building golden datasets
- **Edge case validation**: Ambiguous or novel inputs

### Implementation Pattern

```python
# Queue system for human review
from fastapi import FastAPI
from pydantic import BaseModel

class HumanReviewTask(BaseModel):
    run_id: str
    question: str
    model_answer: str
    context: str | None
    reviewer_instructions: str

class HumanReviewResult(BaseModel):
    run_id: str
    reviewer_id: str
    score: float  # 0-1
    label: str  # "correct", "partial", "wrong", "harmful"
    feedback: str
    corrected_answer: str | None

review_queue: list[HumanReviewTask] = []
completed_reviews: list[HumanReviewResult] = []

@app.post("/review/queue")
async def add_to_review_queue(task: HumanReviewTask):
    review_queue.append(task)
    return {"queued": True}

@app.get("/review/next")
async def get_next_review():
    if review_queue:
        return review_queue.pop(0)
    return None

@app.post("/review/submit")
async def submit_review(result: HumanReviewResult):
    completed_reviews.append(result)
    # Add to LangSmith dataset if high-quality example
    if result.label in ["correct", "wrong"]:
        add_to_golden_dataset(result)
```

---

## Standard Benchmarks (2026)

| Benchmark | Task | What it measures |
|-----------|------|-----------------|
| **MMLU** | Multiple choice | Broad knowledge (57 subjects) |
| **HumanEval / MBPP** | Code generation | Programming capability |
| **GSM8K** | Math word problems | Reasoning, arithmetic |
| **BIG-Bench Hard** | Diverse tasks | Complex reasoning |
| **MT-Bench** | Multi-turn chat | Instruction following |
| **TruthfulQA** | Factual QA | Hallucination resistance |
| **HELMET** | Long-context tasks | 128K+ context performance |
| **SWE-Bench** | Real GitHub issues | Software engineering |
| **RAGAS** | RAG pipelines | Retrieval + generation quality |

---

## Evaluation in CI/CD Pipeline

```yaml
# .github/workflows/eval.yml
name: LLM Evaluation Gate

on:
  pull_request:
    paths:
      - "prompts/**"
      - "src/rag/**"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run eval suite
        env:
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          pip install langsmith langchain-anthropic
          python scripts/run_eval.py \
            --dataset "ml-interview-qa-regression" \
            --threshold 0.85 \
            --experiment-prefix "pr-${{ github.event.number }}"

      - name: Comment eval results on PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval_results.json'));
            github.rest.issues.createComment({
              ...context.repo,
              issue_number: context.issue.number,
              body: `## Eval Results\n- Score: ${results.avg_score.toFixed(2)}\n- Pass rate: ${results.pass_rate.toFixed(1)}%\n- Status: ${results.passed ? '✅ PASSED' : '❌ FAILED'}`
            });
```

---

## Interview Questions

**Q: How do you evaluate an LLM application that generates open-ended text?**
> Three layers: (1) automatic metrics like ROUGE/BERTScore for quick feedback, (2) LLM-as-judge with a powerful model scoring on a rubric, (3) human evaluation on a sample. Build a golden dataset, run offline evals in CI/CD, and monitor online with user feedback signals.

**Q: What are the limitations of LLM-as-judge evaluation?**
> Positional bias (prefers first option), verbosity bias (longer = better), self-consistency issues, and the judge may share the same biases as the evaluated model. Mitigate: randomize answer order, use structured rubrics, ensemble multiple judge models, validate with human annotators.

**Q: How do you detect hallucinations programmatically?**
> Faithfulness scoring: extract claims from the answer, check each against the retrieved context using NLI or LLM judge. NLI models (like MNLI) classify each claim as "entailed," "neutral," or "contradicted." High contradiction rate = likely hallucination. Also: semantic similarity between answer and context.

**Q: How would you set up regression testing for an LLM chatbot?**
> Curate golden examples: (question, expected_answer) pairs. Every production bug gets added as a new test case. Run the full suite on each PR with LangSmith or a custom evaluator. Block merge if overall pass rate drops below threshold (e.g., 90%). This creates a growing safety net that catches regressions.

**Q: What's the difference between offline and online evaluation?**
> Offline: evaluate on a fixed dataset before deployment — reproducible, controlled, catches obvious regressions. Online: monitor real production traffic — catches distribution shift, novel failures, and measures actual user satisfaction. You need both: offline prevents obvious regressions, online catches what offline missed.
