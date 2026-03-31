# CrewAI — Multi-Agent Framework

## What is CrewAI?

CrewAI is an open-source framework for orchestrating **role-playing autonomous AI agents** that collaborate to accomplish complex tasks. Unlike single-agent systems, CrewAI models work as a "crew" — each agent has a defined **role**, **goal**, and **backstory** that shapes its behavior.

```
                    ┌─────────────────────────────┐
                    │           CREW               │
                    │                              │
                    │  ┌────────┐  ┌────────────┐  │
                    │  │ Agent  │  │   Agent    │  │
                    │  │Researcher│ │  Writer    │  │
                    │  └───┬────┘  └─────┬──────┘  │
                    │      │             │          │
                    │  ┌───▼─────────────▼──────┐  │
                    │  │        Tasks            │  │
                    │  │  Task1 → Task2 → Task3  │  │
                    │  └─────────────────────────┘  │
                    └─────────────────────────────┘
```

---

## Core Architecture

### Key Components

| Component | Description |
|-----------|-------------|
| **Agent** | An AI worker with a role, goal, tools, and an LLM backbone |
| **Task** | A discrete unit of work assigned to an agent |
| **Crew** | The collection of agents + tasks + execution process |
| **Process** | How tasks execute: `sequential` or `hierarchical` |
| **Tool** | External capability an agent can invoke (search, code, DB) |

### Execution Processes

- **Sequential**: Tasks run one after another; output of Task N feeds Task N+1
- **Hierarchical**: A "manager" agent delegates tasks to worker agents and validates outputs (requires a manager LLM)

---

## Code Example — Research + Write Crew

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Tools
search_tool = SerperDevTool()

# Agents
researcher = Agent(
    role="Senior ML Researcher",
    goal="Find the latest trends in LLM fine-tuning techniques",
    backstory="You are an expert AI researcher with 10 years of experience in NLP.",
    tools=[search_tool],
    verbose=True,
    llm="claude-opus-4-6"  # or "gpt-4o"
)

writer = Agent(
    role="Technical Writer",
    goal="Write a concise, interview-ready summary of ML findings",
    backstory="You turn complex research into clear engineering documentation.",
    verbose=True
)

# Tasks
research_task = Task(
    description="Research the top 3 LLM fine-tuning methods in 2026 with citations.",
    expected_output="Bullet-point list of techniques with pros/cons",
    agent=researcher
)

write_task = Task(
    description="Convert research findings into a structured markdown document.",
    expected_output="A 500-word markdown doc with headers and code examples.",
    agent=writer,
    context=[research_task]  # Receives researcher output
)

# Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print(result.raw)
```

---

## Hierarchical Process Example

```python
from crewai import Agent, Task, Crew, Process

manager = Agent(
    role="Project Manager",
    goal="Coordinate team to deliver a complete ML pipeline report",
    backstory="Experienced tech lead who delegates and validates all work.",
    allow_delegation=True
)

analyst = Agent(role="Data Analyst", goal="Analyze ML benchmark results", ...)
engineer = Agent(role="ML Engineer", goal="Explain implementation details", ...)

crew = Crew(
    agents=[analyst, engineer],
    tasks=[...],
    process=Process.hierarchical,
    manager_agent=manager,  # Manager validates each step
    verbose=True
)
```

---

## Memory in CrewAI

CrewAI supports multiple memory scopes:

| Memory Type | Scope | Backend |
|-------------|-------|---------|
| **Short-term** | Within current task execution | In-memory context |
| **Long-term** | Across crew runs | SQLite / vector store |
| **Entity** | Facts about people/things | Knowledge graph / vector |
| **Contextual** | Injected at task start | Custom embeddings |

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,              # Enable all memory types
    embedder={
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
)
```

---

## Tool Calling in CrewAI

Agents use tools via natural language — CrewAI routes the call automatically:

```python
from crewai_tools import (
    SerperDevTool,      # Web search
    FileReadTool,       # File I/O
    CodeInterpreterTool, # Python execution
    PGSearchTool        # PostgreSQL queries
)

agent = Agent(
    role="Data Scientist",
    goal="Analyze sales data and find anomalies",
    tools=[FileReadTool(), CodeInterpreterTool()],
    llm="claude-sonnet-4-6"
)
```

---

## CrewAI vs LangGraph vs AutoGen

| Feature | CrewAI | LangGraph | AutoGen |
|---------|--------|-----------|---------|
| **Mental model** | Role-based crew | State machine / graph | Conversation agents |
| **Orchestration** | Sequential / hierarchical | Graph-based (explicit edges) | Chat-based turns |
| **State management** | Implicit via task context | Explicit `StateGraph` | Message history |
| **Best for** | Business process automation | Complex branching logic | Research / debate agents |
| **Learning curve** | Low | Medium | Medium |
| **Determinism** | Low (LLM-driven) | High (graph-controlled) | Medium |

---

## When to Use CrewAI

**Use CrewAI when:**
- You can decompose work into clear roles and sequential tasks
- You need rapid prototyping of multi-agent pipelines
- Your workflow is relatively linear (research → draft → review → publish)
- Business process automation with human-readable agent descriptions

**Avoid CrewAI when:**
- You need strict control flow (loops, conditionals) → use LangGraph
- You need real-time streaming with complex state → use LangGraph
- You need agents to debate/vote → use AutoGen
- Production systems with SLA requirements (CrewAI is still maturing)

---

## Production Pitfalls

1. **Runaway token consumption**: Agents with web search can loop and burn tokens. Always set `max_iter` and `max_execution_time`.
   ```python
   agent = Agent(..., max_iter=10, max_execution_time=60)
   ```

2. **Context window overflow**: Long task chains compound context. Use `context` selectively; don't pass all prior tasks.

3. **Non-determinism**: Same crew can produce different outputs. Add expected_output constraints and use output validators.

4. **Tool errors cascade**: If a tool fails, the agent retries by default. Add `exception_on_tool_error=True` to fail fast.

5. **Cost estimation**: A 5-agent crew with web search can cost $0.50–$5 per run. Monitor with LangSmith or Literal AI.

---

## Failure Handling & Retries

```python
from crewai import Agent, Task

# Agent-level retry config
agent = Agent(
    role="Researcher",
    goal="...",
    max_iter=5,           # Max reasoning loops per task
    max_retry_limit=2,    # Task-level retries on failure
)

# Task-level human validation
task = Task(
    description="Write production-ready code",
    expected_output="Python function with tests",
    human_input=True,     # Pauses for human approval before continuing
    agent=agent
)
```

---

## Interview Questions

**Q: What is CrewAI and when would you choose it over LangChain?**
> CrewAI is a multi-agent orchestration framework where each agent has a role, goal, and backstory. Choose CrewAI when you have distinct agent personas with clear task division (researcher, writer, reviewer). LangChain is lower-level — it's a toolkit for building chains and agents but doesn't natively model "crews."

**Q: How does CrewAI handle agent communication?**
> Agents don't communicate directly. Task outputs are passed as context to downstream tasks. In hierarchical mode, a manager agent delegates and validates results.

**Q: What are the risks of using CrewAI in production?**
> Non-determinism, token cost blowout, context overflow, and tool error cascades. Mitigate with `max_iter`, structured output validators, tool error handling, and observability via LangSmith.

**Q: How do you make a CrewAI workflow deterministic?**
> Define strict `expected_output` formats, use Pydantic output models, limit tools, set `max_iter` conservatively, and add human-in-the-loop checkpoints for critical decisions.

**Q: Explain memory types in CrewAI.**
> Short-term: in-memory context within a run. Long-term: persisted to SQLite, recalled across runs. Entity: structured facts about people/objects. Use long-term memory to avoid re-searching the same information across crew runs.
