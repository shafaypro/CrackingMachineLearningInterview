# Anthropic AI – Complete Guide (2026 Edition)

Anthropic is an AI safety company and the creator of **Claude** — a family of large language models (LLMs) designed to be safe, helpful, and honest. This guide covers Claude's capabilities, the Anthropic API, key AI concepts, and how to build with Claude in 2026.

---

## Table of Contents
1. [Anthropic & Claude Overview](#anthropic--claude-overview)
2. [Claude Model Family](#claude-model-family)
3. [Core AI/LLM Concepts](#core-aillm-concepts)
4. [Anthropic API Basics](#anthropic-api-basics)
5. [Messages API](#messages-api)
6. [Tool Use (Function Calling)](#tool-use-function-calling)
7. [Vision & Multimodal](#vision--multimodal)
8. [Streaming](#streaming)
9. [Prompt Engineering](#prompt-engineering)
10. [Claude's Constitution & Safety](#claudes-constitution--safety)
11. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
12. [Claude in Production](#claude-in-production)
13. [2026: What's New](#2026-whats-new)

---

## Anthropic & Claude Overview

**Anthropic** was founded in 2021 by former OpenAI researchers (Dario Amodei, Daniela Amodei, and others) with a focus on **AI safety research**.

### Anthropic's Core Philosophy

- **Constitutional AI (CAI)**: Training models to be helpful, harmless, and honest using a set of principles rather than just RLHF
- **Interpretability Research**: Understanding what's happening inside neural networks
- **Responsible Scaling Policy (RSP)**: Commitments to assess and mitigate risks as models scale

### Why Claude?

Claude is designed to be:
- **Helpful**: Genuinely useful for complex tasks
- **Harmless**: Avoids causing harm through careful training
- **Honest**: Acknowledges uncertainty, won't deceive

---

## Claude Model Family

As of early 2026:

| Model | ID | Best For | Context Window |
|-------|----|----------|----------------|
| **Claude Opus 4.6** | `claude-opus-4-6` | Complex reasoning, deep analysis | 200K tokens |
| **Claude Sonnet 4.6** | `claude-sonnet-4-6` | Balanced: performance + speed | 200K tokens |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | Fast, lightweight tasks | 200K tokens |

### Choosing a Model

```
Need maximum intelligence for complex tasks?  → Opus
Balanced production workloads?                → Sonnet (most popular)
High-volume, latency-sensitive tasks?         → Haiku
```

### Token Pricing Concepts

- **Input tokens**: Text you send to the model
- **Output tokens**: Text the model generates
- **Context caching**: Reuse large prompts (up to 90% input cost reduction)

---

## Core AI/LLM Concepts

### How LLMs Work

```
Training Data → Pre-training (predict next token) → Base Model
                                                         ↓
                                              RLHF / Constitutional AI
                                                         ↓
                                              Aligned Assistant Model
```

### Key Terminology

| Term | Definition |
|------|-----------|
| **Token** | Smallest unit of text (~4 chars/word on average). Models process tokens, not words. |
| **Context Window** | Max tokens a model can process at once (input + output combined) |
| **Temperature** | Controls randomness (0 = deterministic, 1 = creative, >1 = chaotic) |
| **Top-p (nucleus sampling)** | Probability mass for token selection (0.9 = consider top 90% likely tokens) |
| **System prompt** | Instructions that shape the model's behavior for the conversation |
| **Few-shot prompting** | Providing examples in the prompt to guide output format/style |
| **Chain-of-thought (CoT)** | Asking the model to reason step-by-step before answering |
| **Hallucination** | Model generating confident but incorrect information |
| **Grounding** | Anchoring model output to retrieved facts/documents (RAG) |
| **Fine-tuning** | Further training a model on domain-specific data |
| **RLHF** | Reinforcement Learning from Human Feedback — trains models on human preferences |
| **Constitutional AI** | Anthropic's method: model critiques and revises itself against a set of principles |
| **Embedding** | Dense vector representation of text for similarity search |
| **Tokenizer** | Converts raw text to tokens (and back) |

### Context Window Deep Dive

```
200,000 tokens ≈ 150,000 words ≈ ~500 pages of text

What fits in Claude's context window:
- A full codebase (small/medium projects)
- Entire books or research papers
- Hours of transcript
- Hundreds of documents
```

---

## Anthropic API Basics

### Setup

```bash
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
)
```

### Environment Setup

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Basic Message

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ]
)

print(message.content[0].text)
```

---

## Messages API

The Messages API is the primary way to interact with Claude.

### Request Structure

```python
response = client.messages.create(
    model="claude-sonnet-4-6",   # required
    max_tokens=4096,              # required: max output tokens
    system="You are a helpful data engineer...",  # optional system prompt
    messages=[                    # required: conversation history
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Write me a SQL query..."},
    ],
    temperature=0.7,              # optional: 0-1 (default 1)
    top_p=0.9,                    # optional
    stop_sequences=["###"],       # optional: stop generation on these
)
```

### Response Structure

```python
response.id               # message ID
response.model            # model used
response.role             # "assistant"
response.stop_reason      # "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
response.content          # list of content blocks
response.usage.input_tokens
response.usage.output_tokens

# Get text
text = response.content[0].text
```

### Multi-turn Conversation

```python
import anthropic

client = anthropic.Anthropic()

def chat(messages: list, system: str = "") -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return response.content[0].text

# Build conversation
conversation = []
system = "You are a senior data engineer. Be concise and practical."

# Turn 1
conversation.append({"role": "user", "content": "What is dbt?"})
reply = chat(conversation, system)
conversation.append({"role": "assistant", "content": reply})
print("Assistant:", reply)

# Turn 2
conversation.append({"role": "user", "content": "How does it differ from Spark?"})
reply = chat(conversation, system)
conversation.append({"role": "assistant", "content": reply})
print("Assistant:", reply)
```

---

## Tool Use (Function Calling)

Tool use lets Claude call external functions/APIs to get real-time data or perform actions.

### Define Tools

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools (like OpenAPI schemas)
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'London, UK'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Execute a SQL query against the data warehouse and return results",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute"
                }
            },
            "required": ["query"]
        }
    }
]

# Implement actual tool functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    # In real life: call weather API
    return {"temperature": 22, "condition": "sunny", "unit": unit}

def run_sql_query(query: str) -> list:
    # In real life: query your DWH
    return [{"user_id": 1, "revenue": 100}]

# Tool dispatcher
def execute_tool(name: str, inputs: dict):
    if name == "get_weather":
        return get_weather(**inputs)
    elif name == "run_sql_query":
        return run_sql_query(**inputs)
    raise ValueError(f"Unknown tool: {name}")
```

### Agentic Loop

```python
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # Add Claude's response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Process all tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Add tool results and continue
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude is done — extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

result = run_agent("What's the total revenue this week?")
print(result)
```

---

## Vision & Multimodal

Claude can analyze images, charts, screenshots, and documents.

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

# Method 1: URL (for publicly accessible images)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/chart.png"
                }
            },
            {
                "type": "text",
                "text": "Describe the trends in this chart."
            }
        ]
    }]
)

# Method 2: Base64 encoded image
image_data = base64.standard_b64encode(
    Path("./chart.png").read_bytes()
).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": "What SQL query would produce this data visualization?"
            }
        ]
    }]
)
print(response.content[0].text)
```

---

## Streaming

Stream responses token-by-token for better perceived performance.

```python
import anthropic

client = anthropic.Anthropic()

# Streaming with context manager
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a 500-word essay on AI safety."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # newline

# Get final message with usage stats
final_message = stream.get_final_message()
print(f"\nInput tokens: {final_message.usage.input_tokens}")
print(f"Output tokens: {final_message.usage.output_tokens}")
```

---

## Prompt Engineering

### System Prompts

```python
system = """You are an expert data engineer with 10 years of experience.

Guidelines:
- Be concise and practical
- Show code examples when relevant
- Point out performance implications
- Consider production readiness

When writing SQL: use CTEs, proper indentation, and add comments for complex logic."""
```

### Effective Prompt Patterns

```python
# 1. Role + Task + Format
prompt = """You are a senior Python developer.

Task: Review the following Python function and suggest improvements.

Focus on:
- Performance
- Readability
- Error handling
- Type hints

Function:
```python
{code}
```

Provide your review in this format:
## Issues Found
## Suggested Improvements
## Revised Code"""

# 2. Chain-of-Thought
prompt = """Solve this step by step:

A data pipeline processes 10M rows/day. Query time is 45 seconds.
Target: < 5 seconds.

Think through each possible optimization before recommending."""

# 3. Few-shot example
prompt = """Convert natural language to SQL.

Examples:
Input: "Show me top 5 customers by revenue last month"
SQL: SELECT customer_id, SUM(amount) as revenue
     FROM orders
     WHERE date >= date_trunc('month', current_date - interval '1 month')
     GROUP BY customer_id ORDER BY revenue DESC LIMIT 5;

Input: "How many new users signed up this week?"
SQL: SELECT COUNT(*) FROM users WHERE created_at >= current_date - 7;

Input: {user_query}
SQL:"""

# 4. XML tags for structure (Claude responds well to these)
prompt = """<task>
Analyze the following data pipeline and identify bottlenecks.
</task>

<pipeline_code>
{code}
</pipeline_code>

<requirements>
- Focus on I/O bottlenecks
- Identify parallelization opportunities
- Suggest specific optimizations
</requirements>"""
```

### Temperature Guide

| Temperature | Use Case | Example |
|-------------|----------|---------|
| `0.0` | Deterministic, factual | SQL generation, code, math |
| `0.3` | Mostly consistent, slight variation | Summaries, analysis |
| `0.7` | Balanced creativity | Writing, brainstorming |
| `1.0` | Creative, varied | Creative writing, diverse ideas |

---

## Claude's Constitution & Safety

### Constitutional AI (CAI)

Anthropic trains Claude using a set of principles (a "constitution") rather than relying purely on human ratings:

1. **Self-critique**: Claude generates responses, then critiques them against principles
2. **Revision**: Claude revises problematic responses
3. **RL from AI Feedback (RLAIF)**: AI-generated preferences used for reinforcement learning

### Claude's Values

Claude is designed to:
- Be **genuinely helpful** — not watered-down or overly cautious
- Avoid **deception** — won't create false impressions
- Avoid **harm** — won't assist with clearly harmful actions
- Be **transparent** about limitations and uncertainty
- Have **good character** — not just follow rules, but actually care

### Responsible Scaling Policy (RSP)

Anthropic commits to:
- Evaluating each new model for dangerous capabilities before deployment
- Implementing safety measures before crossing defined capability thresholds
- Publishing transparency reports

---

## Model Context Protocol (MCP)

**MCP** is Anthropic's open standard for connecting LLMs to tools and data sources. See [intro_mcp.md](intro_mcp.md) for the full guide.

Quick summary:
```
Claude (LLM client) ←→ MCP Protocol ←→ MCP Server (tools, data, APIs)
```

```bash
# Example: Claude Code uses MCP to connect to GitHub, databases, etc.
claude mcp add github-server npx @modelcontextprotocol/server-github
```

---

## Claude in Production

### Rate Limits & Batching

```python
from anthropic import Anthropic, RateLimitError
import time

client = Anthropic()

def with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

### Context Caching (cost optimization)

```python
# Cache a large system prompt/document — up to 90% cost reduction
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": very_long_document,   # 50K+ tokens
            "cache_control": {"type": "ephemeral"}  # cache this
        }
    ],
    messages=[{"role": "user", "content": "Summarize chapter 3."}]
)
```

### Batch API (for high-volume, async workloads)

```python
# Process thousands of requests at 50% cost
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": text}]
            }
        }
        for i, text in enumerate(texts)
    ]
)

# Poll for completion
import time
while batch.processing_status != "ended":
    batch = client.messages.batches.retrieve(batch.id)
    time.sleep(30)
```

---

## 2026: What's New

| Feature | Description |
|---------|-------------|
| **Claude 4.x Series** | Opus 4.6, Sonnet 4.6, Haiku 4.5 — state of the art reasoning |
| **Claude Code** | Agentic coding tool using Claude in the terminal / IDE |
| **Extended thinking** | Claude "thinks" before answering for harder problems |
| **MCP ecosystem** | Thousands of MCP servers connecting Claude to every tool imaginable |
| **Computer use** | Claude can operate a computer (click, type, navigate) |
| **Multi-agent** | Orchestrate multiple Claude instances for complex workflows |
| **Artifacts** | Claude can create and iterate on files, code, documents directly |
| **Projects** | Persistent memory and context across conversations |

### Extended Thinking

```python
# Claude "thinks" step by step before answering
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # how much to "think"
    },
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
)

# Response includes thinking blocks + final answer
for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking[:200], "...")
    elif block.type == "text":
        print("Answer:", block.text)
```

---

## Quick Reference

```python
# Basic setup
import anthropic
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# Models
"claude-opus-4-6"           # most capable
"claude-sonnet-4-6"         # recommended default
"claude-haiku-4-5-20251001" # fastest/cheapest

# Simple message
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
text = response.content[0].text

# With system prompt
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Streaming
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Usage tracking
print(response.usage.input_tokens, response.usage.output_tokens)
```
