# Agent Systems and Tool Use

This guide focuses on building AI systems that can plan, call tools, manage state, and complete multi-step tasks rather than only generate one-shot text.

---

## Overview

Agent systems wrap LLMs with decision loops, external tools, memory, and state transitions. They matter because many useful AI applications need more than text generation:

- querying APIs
- reading and writing structured data
- planning multi-step workflows
- using search, code execution, or retrieval tools

The engineering challenge is reliability, not just intelligence.

---

## Core Concepts

### Autonomous agents

An agent is a model-driven system that decides what to do next. In practice, autonomy should be scoped carefully. Most production agents are semi-autonomous workflows with bounded actions.

### Planning and reasoning loops

Planning loops break a task into smaller steps. This matters for long tasks, but too much looping increases latency, cost, and failure surface area.

### Tool calling

Tool calling allows an LLM to select and invoke external capabilities like search, databases, calculators, internal APIs, and code execution.

### Memory systems

Memory can mean short-term conversation history, summarized long-term state, or external facts stored in a vector or relational store.

---

## Key Skills

### Designing agent workflows

In practice, this means defining what decisions the model is allowed to make, which tools it can call, when the workflow should stop, and how failures are retried or escalated.

### Multi-step reasoning

A good engineer can decide when to use a planner-executor split, keep reasoning internal vs explicit, and decompose tasks in parallel.

### Tool integration

This includes building clean tool schemas, validating tool inputs, and handling timeouts and partial failures.

### Managing agent state

Real systems need stable state transitions, audit logs, and explicit checkpoints so a task can be resumed or debugged.

---

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| LangChain | Abstractions for prompts, tools, memory, and chains | Rapid prototyping and simple tool workflows |
| LangGraph | Stateful graph orchestration for agents | Production agent flows with branches and recovery |
| AutoGen | Multi-agent conversation framework | Research-style agent collaboration experiments |
| CrewAI | Role-based multi-agent task coordination | Lightweight multi-agent business workflows |
| n8n | Visual workflow automation for triggers, approvals, and integrations | Operational AI workflows and business automation |
| MCP | Standardized tool and context protocol | Secure tool integration across apps and agents |

---

## Projects

### Multi-agent research assistant

- Goal: Research a topic, gather sources, synthesize findings, and return a structured report.
- Key components: planner, researcher, verifier, summarizer, source tracking.
- Suggested tech stack: LangGraph, search API, vector store, Pydantic.
- Difficulty: Advanced.

### Task automation agent

- Goal: Automate internal repetitive workflows like ticket triage or runbook generation.
- Key components: workflow triggers, tool registry, approval gates, retry logic, audit logs.
- Suggested tech stack: n8n or FastAPI, LangGraph, Postgres, Redis.
- Difficulty: Advanced.

### AI coding assistant

- Goal: Analyze a codebase, propose changes, and execute safe edits.
- Key components: file search, diff generation, execution sandbox, test runner integration.
- Suggested tech stack: Python, MCP-style tools, structured outputs, sandboxed execution.
- Difficulty: Advanced.

### Planner-executor system

- Goal: Separate task decomposition from execution for better traceability.
- Key components: planner node, executor node, state store, validator node.
- Suggested tech stack: LangGraph or custom workflow engine.
- Difficulty: Intermediate to advanced.

---

## Example Code

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    task: str
    plan: List[str]
    completed_steps: List[str]

def planner(state: AgentState) -> AgentState:
    state["plan"] = [
        "search for relevant docs",
        "extract key facts",
        "draft answer",
        "validate answer",
    ]
    return state

def executor(state: AgentState) -> AgentState:
    for step in state["plan"]:
        state["completed_steps"].append(step)
    return state
```

---

## Suggested Project Structure

```text
planner-executor-agent/
├── src/
│   ├── graph.py
│   ├── tools.py
│   ├── state.py
│   ├── prompts.py
│   └── validators.py
├── tests/
├── fixtures/
└── README.md
```

---

## Advanced Tool-Use Patterns

Once the basic call → execute → feed-result loop works, these patterns make tool
use cheaper, safer, and able to scale to large tool libraries.

### The Agentic Loop (Manual vs Managed)

The core loop: call the model with `tools` → if it returns a tool-use request,
execute the tool and append the result as a `tool_result` → repeat until the model
stops requesting tools.

- **Manual loop** — you write the `while` loop. Use it when you need custom
  logging, conditional execution, or **human-in-the-loop approval** before a tool runs.
- **SDK tool runner** — the SDK drives the loop for you (define tools as typed
  functions/schemas, it handles execution and feedback). Use it for the common case.

Always **append the full assistant response** (including tool-use blocks) before
the tool results, and make sure each `tool_result` carries the matching
`tool_use_id`. Set a **max-iterations cap** so a misbehaving agent can't loop forever.

### Parallel Tool Calls

A single model turn can request **multiple** tool calls. Execute independent ones
concurrently and return **all** results in **one** user message. Splitting results
across messages silently teaches the model to stop calling tools in parallel. For
a tool that failed, still return a result with an error flag — don't drop it.

### Strict Schemas & Validation

Use strict JSON schemas (`additionalProperties: false`, explicit `required`) so the
model's arguments validate exactly. Always validate inputs *inside* your handler
before acting — the arguments are model output, not trusted input. Parse tool
inputs with a real JSON parser; never string-match the serialized arguments.

### Programmatic Tool Calling (PTC)

Instead of one round-trip per tool call, let the model write a **script** that calls
tools as functions inside a code-execution sandbox. Intermediate results stay in
the running code (not the context window); only the final output returns to the
model. Use it when chaining many sequential calls or when intermediate results are
large and should be filtered before reaching the context.

### Tool Search (scaling to many tools)

With dozens or hundreds of tools, putting every schema in context is wasteful and
hurts selection accuracy. **Tool search** lets the model discover and load only the
relevant tool schemas on demand — and because schemas are *appended* rather than
swapped, the prompt cache is preserved.

### MCP — a Standard Tool Interface

The **Model Context Protocol** lets agents connect to standardized tool servers
(GitHub, databases, SaaS apps) without bespoke integrations per tool. Declare the
server; the agent gains its tools. See [MCP](./intro_mcp.md).

### Server-Side / Built-in Tools

Providers offer hosted tools that run on their infrastructure — **web search**,
**web fetch**, and **code execution** — declared in the `tools` list with no
client-side execution loop. Great for grounding answers in current information or
running computation without managing a sandbox yourself.

### Security Checklist for Tool Use

| Risk | Mitigation |
|------|-----------|
| Destructive/irreversible action | Human-in-the-loop approval; reversibility check |
| Malicious tool arguments | Strict schema + handler-side validation; allowlist commands for a bash tool |
| Path traversal (file tools) | Resolve to a canonical path; confine to a project root; reject `..`/symlinks |
| Prompt injection via tool output | Treat returned content as untrusted data, not instructions |
| Secret leakage | Keep credentials host-side; never put API keys in prompts or tool args |
| Runaway cost/loops | Max-iteration and token budgets |

### Interview Questions

1. **Why must parallel tool results go in one message?** → Splitting them trains
   the model to stop issuing parallel calls and breaks `tool_use_id` pairing.
2. **What does programmatic tool calling save?** → Round-trips and tokens — only
   the final result re-enters the context, not every intermediate value.
3. **How do you scale an agent to hundreds of tools?** → Tool search / dynamic
   discovery so only relevant schemas load, preserving the prompt cache.
4. **How do you make a `bash`/file tool safe?** → Sandboxing, command allowlists,
   canonical-path confinement, timeouts, and logging every call.
5. **What is MCP and why does it matter?** → A standard protocol for exposing tools
   to agents, so tool servers are reusable across agents instead of bespoke.

---

## Related Topics

- [Multi-Agent Systems](./intro_multi_agent_systems.md)
- [n8n](./intro_n8n.md)
- [LangGraph](./intro_langgraph.md)
- [MCP](./intro_mcp.md)
- [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
