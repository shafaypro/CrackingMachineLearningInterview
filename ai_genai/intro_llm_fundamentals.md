# LLM and Generative AI Fundamentals

This guide covers the foundations needed to build modern LLM applications and explain design decisions in interviews.

---

## Overview

Large language models are neural networks trained to predict the next token over massive text corpora. Generative AI systems use those models to produce text, code, structured outputs, and increasingly images, audio, and multimodal responses.

This matters because modern AI products depend on:

- prompt design
- model selection
- context management
- embeddings and retrieval
- evaluation and safety layers

---

## Core Concepts

### Tokenization

Tokenization converts text into model-readable pieces. In practice, tokenization determines:

- context window usage
- cost
- truncation behavior
- downstream chunking strategies

### Prompt engineering

Prompt engineering is not just phrasing. It includes:

- task framing
- output constraints
- few-shot examples
- tool-use instructions
- failure-handling instructions

### Embeddings

Embeddings map text or other inputs into dense vectors so semantically similar items land near each other. They power semantic search, clustering, retrieval, deduplication, and recommendation.

### Context windows

The context window is the total amount of input and output tokens the model can process in one interaction. Good engineers plan for limited context instead of assuming everything can be stuffed into one prompt.

---

## Key Skills

### Designing effective prompts

In practice, this means writing prompts that are explicit about goal, constrained in format, robust to ambiguity, and testable across example cases.

### Understanding context windows

This shows up as trimming noisy history, summarizing earlier turns, chunking documents correctly, and reserving output tokens.

### Using embeddings for similarity

A strong engineer knows when embedding search is better than keyword match, and when hybrid search is better than either one alone.

### Choosing models for tasks

This is a tradeoff problem across quality, cost, latency, context length, tool-use reliability, and multimodal support.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| OpenAI API | Hosted models for text, reasoning, embeddings, and multimodal tasks | Fastest path to production LLM features |
| Anthropic Claude | Strong reasoning and long-context model family | Complex task decomposition and agent workflows |
| Hugging Face Transformers | Open-source model and tokenizer ecosystem | Fine-tuning, self-hosting, and experimentation |
| tiktoken / tokenizers | Token counting and tokenization utilities | Prompt budgeting and chunking |
| LiteLLM | Unified interface across model providers | Multi-provider abstractions and routing |

---

## Projects

### Chatbot

- Goal: Build a domain-specific conversational assistant with streaming responses.
- Key components: session memory, prompt templates, conversation summarization, evaluation prompts.
- Suggested tech stack: OpenAI or Anthropic API, FastAPI, Redis, React.
- Difficulty: Intermediate.

### Text summarizer

- Goal: Summarize long reports while preserving key facts and action items.
- Key components: chunking, map-reduce summarization, citation or section references, quality rubric.
- Suggested tech stack: Python, LangChain or plain SDK, Pydantic.
- Difficulty: Intermediate.

### Semantic search system

- Goal: Retrieve relevant documents using embeddings and vector search.
- Key components: embedding pipeline, vector index, query rewriting, reranking, relevance evaluation.
- Suggested tech stack: OpenAI embeddings, FAISS or Chroma, FastAPI.
- Difficulty: Intermediate.

### Prompt experimentation repo

- Goal: Track prompts, outputs, and evaluations as a reproducible engineering asset.
- Key components: prompt versioning, fixtures, regression tests, side-by-side output comparison.
- Suggested tech stack: LangSmith or custom test harness, pytest, JSON fixtures.
- Difficulty: Advanced.

---

## Example Code

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-mini",
    input=[
        {"role": "system", "content": "Answer as a concise ML interview coach."},
        {"role": "user", "content": "Explain precision vs recall with an example."},
    ],
)

print(response.output_text)
```

---

## Suggested Project Structure

```text
prompt-experiments/
├── prompts/
├── datasets/
├── evals/
├── scripts/
│   ├── run_eval.py
│   ├── compare_outputs.py
│   └── embed_corpus.py
├── results/
└── README.md
```

---

## LLM Engineering in Practice (2026)

Knowing the theory isn't enough — interviews and real work test whether you can
*choose a model, budget tokens, control cost, and tune generation*. This section
is the practical layer.

### Choosing a Model

Pick the smallest model that clears your quality bar, then scale up only where you
must. A useful mental model across providers:

| Tier | Use for | Example: Claude family (2026) |
|------|---------|-------------------------------|
| **Frontier / reasoning** | Hardest reasoning, long-horizon agentic work, deep research | `claude-fable-5` (most capable), `claude-opus-4-8` |
| **Balanced workhorse** | Most production traffic — strong quality at lower cost/latency | `claude-sonnet-4-6` |
| **Fast / cheap** | High-volume, latency-sensitive, simple tasks (classification, routing, extraction) | `claude-haiku-4-5` |

> Other providers offer the same tiering (a frontier model, a balanced model, a
> small fast model). Always pull current model IDs, context windows, and pricing
> from the provider's own docs — they change often. For Claude specifics see the
> [Anthropic & Claude guide](./intro_anthropic.md).

**Approximate Claude pricing (per 1M tokens, 2026):**

| Model | Input | Output | Context | Max output |
|-------|------:|-------:|---------|-----------:|
| Claude Fable 5 (`claude-fable-5`) | $10 | $50 | 1M | 128K |
| Claude Opus 4.8 (`claude-opus-4-8`) | $5 | $25 | 1M | 128K |
| Claude Sonnet 4.6 (`claude-sonnet-4-6`) | $3 | $15 | 1M | 64K |
| Claude Haiku 4.5 (`claude-haiku-4-5`) | $1 | $5 | 200K | 64K |

### Context Windows & Token Budgeting

- **Context window** = the max tokens (input + output) the model can attend to in
  one request. Modern frontier models reach **1M tokens**.
- **Rough heuristic:** ~1 token ≈ 4 characters ≈ 0.75 English words. Code and
  non-English text tokenize *less* efficiently.
- **Don't estimate with the wrong tokenizer.** OpenAI's `tiktoken` undercounts
  Claude tokens by ~15–20%+. Use the provider's own token-counting endpoint.
- **Budget the whole request:** `prompt + few-shot + retrieved context + reasoning + output ≤ window`.
  Reserve headroom for the output (`max_tokens`) and for reasoning/thinking tokens.

### Prompt Caching (the biggest cost lever)

If many requests share a large, stable prefix (system prompt, tool definitions,
few-shot examples, a long document), **cache it**:

- Cache **reads** cost ~10% of normal input price; cache **writes** cost ~1.25×.
- It's a **prefix match** — any byte change anywhere in the prefix invalidates
  everything after it. So keep stable content first and put volatile content
  (the user's varying question, timestamps) *last*.
- Silent cache-killers: `datetime.now()` or a UUID in the system prompt,
  non-deterministic JSON key order, a tool set that varies per request.

### Generation / Decoding Parameters

| Parameter | Effect | Typical use |
|-----------|--------|-------------|
| `temperature` | Randomness (0 = near-deterministic, higher = more varied) | Low for extraction/code, higher for brainstorming |
| `top_p` (nucleus) | Sample from the smallest set of tokens covering probability `p` | Alternative to temperature — don't aggressively tune both |
| `top_k` | Sample from the top-k tokens | Rarely needed |
| `max_tokens` | Hard cap on output length | Set generously for generation; small for classification |
| `stop` sequences | Halt generation on a string | Structured/templated output |

> **Note (2026):** the newest reasoning models increasingly drop manual sampling
> params in favor of **adaptive thinking** + an **effort** control (`low`→`max`).
> Instead of a fixed "thinking budget," the model decides how much to reason and
> you set how hard it should try. Steer behavior with prompting, not `temperature=0`.

### Estimating Cost

```text
cost ≈ (input_tokens  / 1e6) × input_price
     + (output_tokens / 1e6) × output_price
# with caching:
     + (cached_tokens / 1e6) × (≈0.1 × input_price)   # cache reads
```

Example: a 50K-token cached system prompt + 2K question → 4K answer on Sonnet 4.6,
warm cache ≈ (50K×0.1 + 2K)/1e6 × $3 + 4K/1e6 × $15 ≈ $0.021 + $0.060 ≈ **$0.08/request**.
Without caching the input alone would cost 52K/1e6 × $3 ≈ $0.156.

### Reducing Hallucination

- **Ground the model** with retrieval (RAG) and instruct it to answer only from
  provided context, citing sources.
- **Constrain the output** with structured outputs / strict tool schemas.
- **Lower temperature** for factual tasks.
- **Add an eval + guardrail layer** — LLM-as-judge, schema validation, and refusal
  handling. See [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md).

### Interview Questions

1. **How do you pick a model for a new feature?** → Start with the cheapest tier
   that meets the quality bar (measured on an eval set), scale up only where needed,
   and route easy traffic to a small model.
2. **What's the single biggest lever for LLM cost at scale?** → Prompt caching of a
   stable prefix, plus right-sizing the model per task.
3. **Why not use `tiktoken` to count Claude tokens?** → Wrong tokenizer — it
   undercounts; use the provider's token-counting API.
4. **`temperature` vs `top_p`?** → Both control randomness; temperature scales the
   distribution, top_p truncates it to a probability mass. Tune one, not both hard.
5. **What is a context window and how do you stay within it?** → Max tokens per
   request; budget prompt + context + output, trim/retrieve selectively, and use
   compaction/summarization for long sessions.
6. **How do you reduce hallucinations?** → Ground with RAG + citations, constrain
   output structure, lower temperature, and add evals/guardrails.

---

## Related Topics

- [RAG](./intro_rag.md)
- [Agent Systems](./intro_agent_tool_use.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
- [Multimodal AI](./intro_multimodal_ai.md)
