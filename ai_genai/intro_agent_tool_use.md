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

## Related Topics

- [Multi-Agent Systems](./intro_multi_agent_systems.md)
- [n8n](./intro_n8n.md)
- [LangGraph](./intro_langgraph.md)
- [MCP](./intro_mcp.md)
- [Backend & System Design for AI](../system_design/intro_backend_ai_system_design.md)
