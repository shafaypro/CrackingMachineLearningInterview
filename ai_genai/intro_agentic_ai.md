# Agentic AI & Multi-Agent Systems (2026 Edition)

**Agentic AI** refers to AI systems that autonomously plan, reason, use tools, and execute multi-step tasks to achieve goals — rather than just answering single questions.

---

## Table of Contents
1. [What is Agentic AI?](#what-is-agentic-ai)
2. [Agent Anatomy](#agent-anatomy)
3. [Agent Patterns](#agent-patterns)
4. [Building Agents with Claude](#building-agents-with-claude)
5. [Multi-Agent Systems](#multi-agent-systems)
6. [Anthropic Agent SDK](#anthropic-agent-sdk)
7. [Agent Memory](#agent-memory)
8. [Agent Evaluation & Safety](#agent-evaluation--safety)
9. [Real-World Examples](#real-world-examples)
10. [2026 Landscape](#2026-landscape)

---

## What is Agentic AI?

### Traditional LLM vs Agent

```
Traditional LLM:
User: "Write a Python function to parse CSV"
Model: <writes function>   ← single step, done

Agentic AI:
User: "Analyze our sales data and create a report"
Agent:
  1. Search for sales data files
  2. Load and inspect the data
  3. Identify trends and anomalies
  4. Generate visualizations
  5. Write the report to a file
  6. Send it via email  ← multi-step, autonomous
```

### Why Agentic AI Now?

Three things converged in 2024–2026:
1. **Large context windows** (200K tokens) — agents can hold entire codebases in mind
2. **Reliable tool use** — models accurately call functions with correct parameters
3. **Better instruction following** — models stay on task across many steps

---

## Agent Anatomy

Every agent has 5 core components:

```
┌────────────────────────────────────────────────────┐
│                      Agent                          │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Brain   │  │  Tools   │  │     Memory       │  │
│  │  (LLM)   │  │          │  │                  │  │
│  │          │  │ - Search │  │ - Working (ctx)  │  │
│  │ Reason   │  │ - Code   │  │ - Short-term     │  │
│  │ Plan     │  │ - Files  │  │ - Long-term      │  │
│  │ Decide   │  │ - APIs   │  │   (vector DB)    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │   Perception     │  │        Action            │ │
│  │ (input/context)  │  │ (executing tool calls)   │ │
│  └──────────────────┘  └──────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Agent Patterns

### 1. ReAct (Reason + Act)

The most common pattern. The agent alternates between reasoning and acting.

```
Thought: I need to find the top customers by revenue
Action: run_sql("SELECT customer_id, SUM(amount) FROM orders GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
Observation: [{"customer_id": "C001", "total": 50000}, ...]
Thought: Now I need their names from the customers table
Action: run_sql("SELECT name FROM customers WHERE id IN ('C001', 'C002', ...)")
Observation: [{"name": "Acme Corp"}, ...]
Thought: I have enough data to write the report
Action: write_file("report.md", "# Top Customers Report\n...")
Final Answer: Report written to report.md
```

### 2. Plan-and-Execute

The agent creates a full plan first, then executes each step.

```
Plan:
1. Query sales data for Q4
2. Calculate MoM growth rates
3. Identify top/bottom performers
4. Generate charts
5. Write executive summary

Execute: [runs each step]
```

### 3. Reflection / Self-Critique

The agent evaluates its own output and improves it.

```
Draft response → Critique (is this correct? complete? safe?) → Revise → Final
```

### 4. Tool-Augmented Generation

Agent decides when to call tools vs answer from memory.

---

## Building Agents with Claude

### Simple Agent Loop

```python
import anthropic
import json
import subprocess
import os

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "bash",
        "description": "Execute a bash command and return stdout/stderr",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    if name == "bash":
        result = subprocess.run(
            inputs["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout + result.stderr

    elif name == "read_file":
        try:
            return open(inputs["path"]).read()
        except FileNotFoundError:
            return f"Error: File not found: {inputs['path']}"

    elif name == "write_file":
        with open(inputs["path"], "w") as f:
            f.write(inputs["content"])
        return f"File written: {inputs['path']}"

    return f"Unknown tool: {name}"

def run_agent(task: str, max_iterations: int = 20) -> str:
    """Run an agent loop until the task is complete or max iterations reached."""

    messages = [{"role": "user", "content": task}]
    system = """You are an autonomous agent. Complete the given task using available tools.

    Think step by step. Be thorough but efficient. When the task is complete, provide a clear final answer."""

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Agent is done
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        elif response.stop_reason == "tool_use":
            # Execute tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"[Agent] Tool: {block.name}({json.dumps(block.input)[:100]})")
                    result = execute_tool(block.name, block.input)
                    print(f"[Agent] Result: {result[:200]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached"

# Use the agent
result = run_agent("""
Analyze the Python files in the current directory:
1. Count the total number of functions
2. Find the longest function
3. List any TODO comments
4. Write a summary to analysis_report.md
""")
print(result)
```

---

## Multi-Agent Systems

Complex tasks can be broken down across specialized agents.

### Orchestrator + Subagents Pattern

```python
import anthropic
from concurrent.futures import ThreadPoolExecutor

client = anthropic.Anthropic()

def run_subagent(task: str, agent_role: str) -> str:
    """Run a specialized subagent for a specific task."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # faster/cheaper for subagents
        max_tokens=2048,
        system=f"You are a specialized {agent_role}. Complete the given task concisely.",
        messages=[{"role": "user", "content": task}]
    )
    return response.content[0].text

def orchestrator(user_goal: str) -> str:
    """Orchestrator breaks down goals and coordinates subagents."""

    # Step 1: Plan
    plan_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a planning agent. Break down the goal into 3-5 parallel subtasks.",
        messages=[{
            "role": "user",
            "content": f"Break this into parallel subtasks: {user_goal}\nReturn as JSON: {{\"tasks\": [{{\"role\": \"...\", \"task\": \"...\"}}]}}"
        }]
    )

    import json
    plan = json.loads(plan_response.content[0].text)

    # Step 2: Execute subtasks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_subagent, item["task"], item["role"]): item
            for item in plan["tasks"]
        }
        results = {}
        for future, item in futures.items():
            results[item["role"]] = future.result()

    # Step 3: Synthesize results
    synthesis = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a synthesis agent. Combine the work of multiple subagents into a coherent final answer.",
        messages=[{
            "role": "user",
            "content": f"Goal: {user_goal}\n\nSubagent results:\n{json.dumps(results, indent=2)}\n\nSynthesize a final answer."
        }]
    )

    return synthesis.content[0].text

result = orchestrator("Create a competitive analysis of the top 3 cloud data warehouses")
print(result)
```

### Specialized Agent Roles

| Agent Type | Responsibility |
|-----------|---------------|
| **Orchestrator** | Plans, delegates, synthesizes |
| **Researcher** | Searches, reads, summarizes information |
| **Coder** | Writes, tests, debugs code |
| **Critic** | Reviews and identifies issues |
| **Executor** | Runs commands, calls APIs |
| **Writer** | Drafts documents, reports |

---

## Anthropic Agent SDK

The **Claude Agent SDK** provides higher-level primitives for building agents.

```python
# Install
# pip install anthropic-agent-sdk  (conceptual - check current docs)

from anthropic import Anthropic

client = Anthropic()

# Claude Code itself is an agent built on these primitives
# Key patterns:
# 1. Tool definitions
# 2. Agentic loop
# 3. Context management
# 4. Interruption handling
```

### Human-in-the-Loop

```python
def agent_with_confirmation(task: str):
    """Agent that asks for confirmation before risky actions."""

    RISKY_TOOLS = {"delete_file", "run_database_migration", "send_email"}

    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content[0].text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Check if this needs confirmation
                    if block.name in RISKY_TOOLS:
                        print(f"\n⚠️  Agent wants to call: {block.name}")
                        print(f"   Arguments: {block.input}")
                        confirm = input("   Approve? [y/N]: ")
                        if confirm.lower() != "y":
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "User declined this action",
                                "is_error": True
                            })
                            continue

                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})
```

---

## Agent Memory

### Types of Memory

```
1. Working Memory    = LLM context window (in-flight conversation)
2. Episodic Memory   = Past interactions stored in DB + retrieved via search
3. Semantic Memory   = Facts/knowledge in vector DB
4. Procedural Memory = Skills encoded in system prompts or fine-tuned weights
```

### Implementing Memory with Vector Search

```python
from anthropic import Anthropic
import chromadb
import uuid

client = Anthropic()
chroma = chromadb.Client()
memory_collection = chroma.create_collection("agent_memory")

def remember(text: str, metadata: dict = {}):
    """Store a memory."""
    memory_collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )

def recall(query: str, n: int = 5) -> list[str]:
    """Retrieve relevant memories."""
    results = memory_collection.query(
        query_texts=[query],
        n_results=n
    )
    return results["documents"][0]

def agent_with_memory(user_message: str) -> str:
    # Recall relevant memories
    memories = recall(user_message)
    memory_context = "\n".join(memories) if memories else "No relevant memories."

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"""You are a helpful assistant with memory.

Relevant memories from past interactions:
{memory_context}

Use these memories to provide personalized, contextual responses.""",
        messages=[{"role": "user", "content": user_message}]
    )

    reply = response.content[0].text

    # Store this interaction as a memory
    remember(f"User asked: {user_message}\nAssistant replied: {reply[:200]}")

    return reply
```

---

## Agent Evaluation & Safety

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Task completion rate** | % of tasks successfully completed |
| **Step efficiency** | Avg tool calls per task (lower = more efficient) |
| **Error rate** | % of tasks with incorrect outcomes |
| **Safety rate** | % of harmful actions rejected |
| **Latency** | Time to complete task |

### Safety Principles for Agents

1. **Minimal footprint** — request only necessary permissions
2. **Prefer reversible actions** — delete → trash, not permanent delete
3. **Confirm before irreversible actions** — especially in production
4. **Fail loudly** — surface errors rather than silently continuing
5. **Audit trail** — log all tool calls and outcomes
6. **Scope limiting** — constrain what the agent can access
7. **Injection detection** — watch for prompt injection in tool results

```python
# Prompt injection guard
def sanitize_tool_result(result: str) -> str:
    """Detect and neutralize potential prompt injection in tool results."""
    injection_patterns = [
        "ignore previous instructions",
        "new instruction:",
        "system: you are now",
        "<system>",
    ]
    for pattern in injection_patterns:
        if pattern.lower() in result.lower():
            return f"[SANITIZED: Potential injection detected in tool result]"
    return result
```

---

## Real-World Examples

### Data Engineering Agent

```python
task = """
Our dbt pipeline failed last night. Please:
1. Check the error logs in ./logs/dbt.log
2. Identify which model failed and why
3. Fix the SQL if it's a syntax error
4. Run dbt test on the affected model
5. Write a brief incident report
"""
result = run_agent(task)
```

### Code Review Agent

```python
task = """
Review the PR diff in ./changes.diff:
1. Check for security vulnerabilities
2. Check for performance issues
3. Verify test coverage
4. Post review comments in ./review.md
"""
```

### Research Agent

```python
task = """
Research the current state of vector databases in 2026:
1. List the top 5 vector databases with their features
2. Compare performance benchmarks
3. Summarize adoption trends
4. Write a 2-page briefing document
"""
```

---

## 2026 Landscape

### Popular Agent Frameworks

| Framework | Language | Key Feature |
|-----------|----------|-------------|
| **LangGraph** | Python | Stateful multi-agent graphs |
| **CrewAI** | Python | Role-based agent crews |
| **AutoGen** | Python | Multi-agent conversations (Microsoft) |
| **Claude Code** | TypeScript | Agentic coding assistant |
| **Composio** | Python | 250+ tool integrations for agents |
| **Dify** | Python | Visual agent builder |

### Production Agent Platforms

| Platform | Description |
|----------|-------------|
| **Amazon Bedrock Agents** | Managed agent service on AWS |
| **Google Vertex AI Agents** | Managed agents on GCP |
| **Azure AI Agents** | Managed agents on Azure |
| **Anthropic API** | Build your own with Claude + tools |

### 2026 Trends

- **Agentic loops are stable** — ReAct and Plan-Execute patterns are production-proven
- **Multi-agent coordination** — Orchestrator patterns replacing monolithic agents
- **Specialized models** — Small models for subagent tasks, large models for orchestration
- **MCP standardization** — Agents share tool servers via MCP
- **Evaluations matured** — LLM-as-judge and automated evals are standard
- **Computer use** — Agents that can operate UIs (browsers, desktops)
- **Long-horizon tasks** — Agents running for hours/days on complex engineering tasks

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [RAG](./intro_rag.md) | Agentic RAG combines agents with retrieval |
| [MCP](./intro_mcp.md) | Model Context Protocol is the standard tool interface for agents |
| [LangChain](./intro_langchain.md) | LangGraph is the leading framework for agentic workflows |
| [LLMOps](./intro_llmops.md) | Monitoring and evaluating agent reliability in production |
| [Anthropic Overview](./intro_anthropic.md) | Claude's tool use and extended thinking for agents |
