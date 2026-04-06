# Prompt Engineering

## Why Prompt Engineering Is an Engineering Discipline

Prompt engineering is not "magic words." It is a systematic process of designing, testing, and iterating on instructions that reliably elicit the behavior you need from an LLM. Like software, prompts have versions, test cases, and failure modes.

Poor prompt engineering causes:
- Inconsistent output format → broken parsing
- Hallucinated content → user trust issues
- Model refusals → blocked workflows
- Context overload → degraded quality

---

## Fundamental Techniques

### Zero-Shot Prompting

No examples — just instructions. Works well for tasks the model has seen frequently during training.

```python
prompt = """
Classify the sentiment of the following review as Positive, Negative, or Neutral.
Return only the label with no explanation.

Review: "The delivery was two days late but the product itself is excellent."
"""
# Output: Positive
```

**When it fails:** Novel tasks, tasks requiring specific formats, tasks with ambiguous criteria.

### Few-Shot Prompting

Provide worked examples before the actual input. The model learns the pattern from examples.

```python
prompt = """
Extract the company name and action from each sentence.
Format: {"company": "...", "action": "..."}

Sentence: Apple announced a new chip called M4 Ultra.
{"company": "Apple", "action": "announced new chip"}

Sentence: Tesla recalled 120,000 vehicles over software issues.
{"company": "Tesla", "action": "recalled vehicles"}

Sentence: Microsoft acquired Activision Blizzard for $69 billion.
"""
# Model follows the pattern: {"company": "Microsoft", "action": "acquired Activision Blizzard"}
```

**Few-shot tips:**
- Order examples: put the most similar example last (closest to the actual input)
- Balance examples across classes to avoid bias
- 3-8 examples is usually optimal — diminishing returns beyond 10
- Use examples that cover edge cases you care about

### Chain-of-Thought (CoT) Prompting

For reasoning tasks, instructing the model to "think step by step" dramatically improves accuracy. The model's intermediate reasoning steps catch logical errors that would otherwise be skipped.

```python
# Standard (fails on complex reasoning)
prompt = "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?"
# Model often outputs: $0.10 (wrong)

# Chain-of-thought
prompt = """
A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball.
How much does the ball cost?

Think step by step before giving your final answer.
"""
# Model output:
# Let ball = x cents
# Bat = x + 100 cents
# x + (x + 100) = 110
# 2x = 10
# x = 5 cents
# Answer: $0.05
```

**Few-shot CoT:** Provide examples that show the reasoning chain explicitly:

```python
few_shot_cot = """
Q: Roger has 5 tennis balls. He buys 2 cans of tennis balls, each containing 3 balls. How many does he have now?
A: Roger starts with 5. He buys 2 × 3 = 6 more. 5 + 6 = 11. Answer: 11.

Q: Shawn has five toys. For Christmas he got two toys each from his mom and dad. How many toys does he have now?
A: """
```

### Self-Consistency

For high-stakes reasoning, sample multiple CoT paths and take the majority vote:

```python
from collections import Counter

def self_consistent_answer(prompt: str, model, n_samples: int = 5) -> str:
    """
    Sample n reasoning paths, return the most common final answer.
    More reliable than single CoT for mathematical reasoning.
    """
    answers = []
    for _ in range(n_samples):
        response = model.generate(prompt, temperature=0.7)  # some temperature for diversity
        # Extract final answer from response (last number, last sentence, etc.)
        answer = extract_final_answer(response)
        answers.append(answer)
    
    # Return the majority answer
    return Counter(answers).most_common(1)[0][0]
```

**When to use:** Math problems, logic puzzles, medical reasoning — tasks where a single wrong step invalidates the answer and you can afford multiple API calls.

### ReAct (Reason + Act)

Interleaves reasoning traces with tool actions. Lets the model plan a thought, execute an action, observe the result, and repeat.

```
Thought: I need to find the current population of Japan to answer this question.
Action: search("Japan current population 2024")
Observation: Japan has approximately 123.3 million people as of 2024.
Thought: Now I can answer the question.
Final Answer: Japan's population is approximately 123 million.
```

```python
react_system_prompt = """
You are an assistant with access to tools. For each step:
1. Write "Thought:" explaining what you need to do next
2. Write "Action:" with the tool call (format: tool_name(args))
3. You will receive "Observation:" with the result
4. Repeat until you can write "Final Answer:"

Available tools:
- search(query: str) → web search results
- calculator(expression: str) → numeric result
- lookup(term: str) → definition or fact

If you can answer without tools, go directly to Final Answer.
"""
```

### Tree of Thought (ToT)

For very hard problems, explore multiple reasoning branches in parallel, score them, and backtrack from dead ends:

```
Problem: "What's the best way to implement a distributed cache?"

Branch 1: Redis cluster     → Evaluate pros/cons → Score: 8/10
Branch 2: Memcached         → Evaluate pros/cons → Score: 6/10  
Branch 3: Custom in-memory  → Evaluate pros/cons → Score: 4/10

→ Explore Branch 1 further with sub-branches
```

ToT is expensive (many model calls) — use for one-off hard problems, not production pipelines.

---

## System Prompt Design

The system prompt defines the model's persona, constraints, and operating context. It is the most durable part of your prompt — it doesn't change per request.

```python
system_prompt = """
You are a customer support assistant for Acme Software.

## Your Role
Answer questions about Acme's products, pricing, and troubleshooting.
Escalate to a human agent when: the customer is angry, the issue involves billing disputes > $500, or the question is outside your knowledge.

## Constraints
- Never discuss competitors by name
- Never make promises about future features
- Always end with "Is there anything else I can help you with?"
- Respond in the same language the customer uses

## Format
- Use bullet points for step-by-step instructions
- Keep responses under 200 words unless troubleshooting requires more
- Reference documentation links when relevant
"""
```

**System prompt principles:**
1. **Role** — who is the model being?
2. **Scope** — what topics are in/out of bounds?
3. **Behavior** — what consistent actions should it take?
4. **Format** — how should output be structured?
5. **Escalation** — when should it hand off or refuse?

---

## Output Format Control

### Explicit Format Instructions

```python
prompt = """
Analyze the following customer complaint and return a JSON object with these exact fields:
- sentiment: "positive" | "negative" | "neutral"
- urgency: "low" | "medium" | "high"
- category: "billing" | "technical" | "shipping" | "other"
- summary: one sentence, under 20 words
- requires_escalation: boolean

Return only the JSON with no explanation or markdown.

Complaint: "My order hasn't arrived after 3 weeks and no one is responding to my emails!"
"""
```

### Delimiters for Reliable Parsing

Use XML-style tags to clearly separate instructions from content:

```python
prompt = f"""
Summarize the following article in 3 bullet points.
Return the bullets inside <summary> tags.

<article>
{article_text}
</article>

<summary>
"""
# Model naturally completes the <summary> block
```

### Using JSON Mode (API-level)

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},  # Guarantees valid JSON output
    messages=[
        {"role": "system", "content": "You are a data extraction assistant. Always respond with valid JSON."},
        {"role": "user", "content": f"Extract entities from: {text}"}
    ]
)
import json
result = json.loads(response.choices[0].message.content)
```

---

## Context Management

### Token Budgeting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def build_prompt_within_budget(
    system: str,
    history: list[dict],
    user_message: str,
    context_docs: list[str],
    max_tokens: int = 128_000,
    output_reserve: int = 4_000
) -> tuple[str, list[dict]]:
    """
    Fit conversation history and context docs within the token budget.
    Trims oldest history and least relevant docs first.
    """
    available = max_tokens - output_reserve
    
    # Always include: system + latest user message
    base_tokens = count_tokens(system) + count_tokens(user_message)
    remaining = available - base_tokens
    
    # Add history from newest to oldest (trim old turns)
    trimmed_history = []
    for turn in reversed(history):
        turn_tokens = count_tokens(str(turn))
        if remaining - turn_tokens > 1000:  # keep 1K buffer for docs
            trimmed_history.insert(0, turn)
            remaining -= turn_tokens
        else:
            break
    
    # Fill remaining budget with context docs
    selected_docs = []
    for doc in context_docs:
        doc_tokens = count_tokens(doc)
        if remaining - doc_tokens > 0:
            selected_docs.append(doc)
            remaining -= doc_tokens
    
    return selected_docs, trimmed_history
```

### Conversation History Compression

For long conversations, summarize older turns instead of truncating:

```python
def compress_history(history: list[dict], threshold_turns: int = 20) -> list[dict]:
    if len(history) <= threshold_turns:
        return history
    
    old_turns = history[:-threshold_turns]
    recent_turns = history[-threshold_turns:]
    
    summary_prompt = f"""
    Summarize the following conversation history concisely, preserving:
    - Key decisions made
    - Important facts established
    - User preferences mentioned
    
    History:
    {format_turns(old_turns)}
    """
    
    summary = llm.generate(summary_prompt)
    
    # Replace old turns with a single summary turn
    return [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + recent_turns
```

---

## Prompt Testing & Evaluation

Treat prompts like code: version them, test them, measure regressions.

```python
import json
from dataclasses import dataclass
from typing import Callable

@dataclass
class PromptTestCase:
    input: str
    expected_output: str | dict
    evaluator: Callable  # function that checks if output matches expected

class PromptTestSuite:
    def __init__(self, prompt_template: str, model):
        self.prompt = prompt_template
        self.model = model
        self.test_cases: list[PromptTestCase] = []

    def add_case(self, test: PromptTestCase):
        self.test_cases.append(test)

    def run(self) -> dict:
        results = {"passed": 0, "failed": 0, "cases": []}
        for case in self.test_cases:
            output = self.model.generate(self.prompt.format(input=case.input))
            passed = case.evaluator(output, case.expected_output)
            results["passed" if passed else "failed"] += 1
            results["cases"].append({
                "input": case.input,
                "output": output,
                "expected": case.expected_output,
                "passed": passed,
            })
        return results

# Evaluator examples
def exact_match(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()

def json_field_match(output: str, expected: dict) -> bool:
    try:
        parsed = json.loads(output)
        return all(parsed.get(k) == v for k, v in expected.items())
    except json.JSONDecodeError:
        return False

def contains_all(output: str, expected: list[str]) -> bool:
    return all(term.lower() in output.lower() for term in expected)
```

---

## Advanced Patterns

### Meta-Prompting

Use the model to improve its own prompt:

```python
meta_prompt = """
Here is a prompt I use for a customer support chatbot:
<current_prompt>
{current_prompt}
</current_prompt>

Here are 3 cases where it failed:
{failure_examples}

Rewrite the prompt to handle these failure cases while keeping what works.
Return only the improved prompt inside <improved_prompt> tags.
"""
```

### Prompt Chaining

Break complex tasks into a sequence of simpler prompts, passing outputs between steps:

```python
def analyze_document_pipeline(document: str) -> dict:
    # Step 1: Extract key facts
    facts = llm.generate(f"List the 5 most important facts from this document:\n{document}")
    
    # Step 2: Identify claims that need verification
    claims = llm.generate(f"From these facts, which claims need external verification?\n{facts}")
    
    # Step 3: Assess confidence
    confidence = llm.generate(f"Rate the overall reliability of this document (1-10) based on:\nFacts: {facts}\nUnverified claims: {claims}")
    
    return {"facts": facts, "unverified_claims": claims, "confidence_score": confidence}
```

### Dynamic Few-Shot Selection

Use semantic search to select the most relevant examples for each input:

```python
class DynamicFewShot:
    def __init__(self, example_bank: list[dict], embedder):
        self.examples = example_bank
        self.embedder = embedder
        # Pre-compute embeddings for all examples
        self.embeddings = embedder.embed([ex["input"] for ex in example_bank])

    def select_examples(self, query: str, n: int = 3) -> list[dict]:
        """Return the n most similar examples to the query."""
        query_embedding = self.embedder.embed([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-n:][::-1]
        return [self.examples[i] for i in top_indices]

    def build_prompt(self, query: str, task_instructions: str) -> str:
        examples = self.select_examples(query)
        example_text = "\n\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        )
        return f"{task_instructions}\n\n{example_text}\n\nInput: {query}\nOutput:"
```

---

## Common Interview Questions

**Q: What is the difference between zero-shot, few-shot, and fine-tuning? When would you choose each?**
Zero-shot: model answers with only instructions, no examples. Use when the task is straightforward and the model generalizes well. Few-shot: provide 3-8 worked examples in the prompt. Use when zero-shot fails or you need a specific output format. Fine-tuning: train model weights on many examples. Use when few-shot is inconsistent, you need persistent behavior across all calls, or you have a high-volume task where prompt token cost matters. The cost/complexity order: zero-shot < few-shot < fine-tuning.

**Q: Why does chain-of-thought prompting improve performance on reasoning tasks?**
CoT forces the model to produce intermediate steps before the final answer. These steps are also subject to the model's next-token prediction — generating a correct intermediate step makes the next correct step more likely. The model essentially "checks its work" by reasoning aloud. For tasks where the correct answer depends on a chain of logical steps, any single wrong step cascades to a wrong final answer. CoT dramatically reduces this by making each step explicit and correctable.

**Q: How would you prevent prompt injection in a production LLM application?**
Prompt injection is when user input contains instructions that override the system prompt. Defenses: (1) delimiter-based isolation — wrap user input in XML tags and instruct the model never to act on instructions inside those tags; (2) input sanitization — filter or escape known injection patterns before sending to the model; (3) structured outputs — if you only accept JSON back, a prompt injection telling the model to "ignore previous instructions" won't affect a parseable JSON response; (4) output validation — validate the model's response against expected schemas and flag anomalies; (5) privileged/unprivileged tiers — distinguish system instructions from user content in your architecture.

**Q: What is the "lost in the middle" problem?**
LLMs have worse recall for information placed in the middle of long contexts compared to information at the beginning or end. This is a known limitation of attention-based models. In practice: place the most important context at the start and end of the prompt, not buried in the middle. For RAG, order retrieved documents so the most relevant is first.

**Q: How do you evaluate whether a prompt change improved things?**
Define evaluation criteria before changing the prompt (avoid HARKing). Run the old and new prompts on a fixed test set of inputs. Use multiple evaluators: exact match for structured outputs, LLM-as-judge for open-ended quality, human evaluation for subtle quality. Track failure modes separately from aggregate metrics — a prompt change might improve average quality but introduce a new failure mode.
