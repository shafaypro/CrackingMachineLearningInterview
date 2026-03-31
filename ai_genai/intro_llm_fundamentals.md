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

## Related Topics

- [RAG](./intro_rag.md)
- [Agent Systems](./intro_agent_tool_use.md)
- [Evaluation & Guardrails](../mlops/intro_evaluation_guardrails.md)
- [Multimodal AI](./intro_multimodal_ai.md)
