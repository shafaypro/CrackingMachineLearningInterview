# Multi-Agent Systems — Architecture, Patterns & Production

## Single-Agent vs Multi-Agent Systems

### Single-Agent System

One LLM handles all reasoning, tool use, and output generation.

```
User Query → [Agent] → Response
                ↕
             [Tools]
```

**Pros**: Simple, low latency, low cost, easy to debug
**Cons**: Context window limits, poor at diverse skill sets, single point of failure

### Multi-Agent System

Multiple specialized agents collaborate, with coordination overhead managed by orchestration.

```
                    ┌─────────────────────────────┐
User Query →  [Orchestrator/Supervisor]           │
                    │         │         │          │
                [Researcher] [Coder] [Critic]     │
                    │         │         │          │
                 [Tools]   [Exec]   [Verify]      │
                    └─────────┴─────────┘          │
                            │                      │
                        [Aggregator] → Response    │
                    └─────────────────────────────┘
```

**Pros**: Specialized expertise per agent, parallel execution, scalable complexity
**Cons**: Higher latency, token cost multiplied, harder to debug, coordination failures

---

## When to Use Multi-Agent

| Scenario | Recommendation |
|----------|----------------|
| Simple Q&A or lookup | Single agent |
| Linear multi-step task (research → write → edit) | Single agent with tools OR CrewAI |
| Parallel subtasks (analyze 10 docs simultaneously) | Multi-agent (parallel) |
| Tasks requiring different expertise (legal + technical) | Multi-agent (specialized) |
| Very long context that exceeds single window | Multi-agent (divide and conquer) |
| Adversarial validation (agent A writes, B critiques) | Multi-agent (critic pattern) |
| Production system with SLA requirements | Single agent (simpler, faster, cheaper) |

---

## Multi-Agent Patterns

### 1. Supervisor Pattern

A manager agent routes tasks to specialist workers:

```python
from langgraph_supervisor import create_supervisor
from langchain_anthropic import ChatAnthropic

supervisor = create_supervisor(
    agents=["data_analyst", "ml_engineer", "writer"],
    model=ChatAnthropic(model="claude-opus-4-6"),
    prompt="""You are a project manager. Route each task to the right specialist:
    - data_analyst: data queries, statistics, EDA
    - ml_engineer: model design, training, evaluation
    - writer: documentation, summaries, reports
    When all tasks are done, synthesize the final answer."""
)
```

### 2. Critic-Generator Pattern

One agent generates, another critiques and improves:

```python
# Generator → Critic → Refined Output
def generator_node(state):
    draft = llm.invoke(f"Write a solution for: {state['task']}")
    return {"draft": draft}

def critic_node(state):
    feedback = critic_llm.invoke(
        f"Review this solution and identify issues:\n{state['draft']}\n"
        f"Return: issues found and specific improvements needed"
    )
    return {"feedback": feedback}

def refiner_node(state):
    final = llm.invoke(
        f"Original solution: {state['draft']}\n"
        f"Critic feedback: {state['feedback']}\n"
        f"Improve the solution based on feedback."
    )
    return {"final": final}
```

### 3. Parallel Fan-Out / Fan-In

Execute subtasks concurrently, then aggregate:

```python
from langgraph.graph import StateGraph, Send

def route_to_workers(state):
    """Fan-out: send each document to a worker"""
    return [
        Send("analyze_doc", {"doc": doc, "doc_id": i})
        for i, doc in enumerate(state["documents"])
    ]

def aggregate_results(state):
    """Fan-in: combine all worker outputs"""
    return {"summary": combine(state["results"])}

graph.add_conditional_edges("splitter", route_to_workers)
graph.add_edge("analyze_doc", "aggregator")
```

### 4. Handoff Pattern

Agents explicitly hand off to each other:

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Agent A can call Agent B as a tool
@tool
def transfer_to_coder(task: str) -> str:
    """Transfer this task to the coding specialist agent."""
    return coder_agent.invoke({"messages": [("human", task)]})

researcher_agent = create_react_agent(
    llm,
    tools=[search_tool, transfer_to_coder]
)
```

---

## Tool Calling & Function Routing

### How Tool Calling Works

1. LLM receives a list of tool schemas (JSON Schema)
2. LLM outputs structured tool call JSON instead of text
3. Runtime executes the tool and feeds result back to LLM
4. LLM generates final response incorporating tool output

```python
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
import json

@tool
def get_model_accuracy(model_name: str, dataset: str) -> dict:
    """Get the test accuracy of a trained model on a dataset.

    Args:
        model_name: Name of the model (e.g., 'resnet50', 'bert-base')
        dataset: Dataset name (e.g., 'imagenet', 'glue-sst2')
    """
    # Call your model registry API
    return {"model": model_name, "dataset": dataset, "accuracy": 0.934}

@tool
def list_available_models() -> list[str]:
    """List all models available in the model registry."""
    return ["resnet50", "bert-base", "gpt2-medium", "llama-3.1-8b"]

llm = ChatAnthropic(model="claude-sonnet-4-6")
llm_with_tools = llm.bind_tools([get_model_accuracy, list_available_models])

# LLM decides which tools to call and with what arguments
response = llm_with_tools.invoke(
    "Which of our models performs best on ImageNet?"
)

# Handle tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "get_model_accuracy":
            result = get_model_accuracy.invoke(tool_args)
        elif tool_name == "list_available_models":
            result = list_available_models.invoke(tool_args)
```

### Parallel Tool Calling

Modern LLMs can call multiple tools in a single response:

```python
# LLM may return:
# tool_calls = [
#   {"name": "get_model_accuracy", "args": {"model": "resnet50", "dataset": "imagenet"}},
#   {"name": "get_model_accuracy", "args": {"model": "bert-base", "dataset": "imagenet"}},
# ]
# Execute in parallel
import asyncio

async def execute_tools_parallel(tool_calls):
    tasks = [execute_tool(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

---

## Memory Architectures

### Short-Term Memory (In-Context)

The LLM's context window is the memory:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

history_store = {}  # session_id → message history

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

chain_with_history = RunnableWithMessageHistory(
    llm_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Session persists within server lifetime only
chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "alice-session-1"}}
)
```

**Limitations**: Lost on restart, grows unbounded, expensive at scale

### Long-Term Memory (External Store)

Persist memories outside the context window:

```python
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class LongTermMemory:
    def __init__(self, embeddings):
        self.store = Chroma(embedding_function=embeddings)

    def save(self, text: str, metadata: dict = {}):
        """Store a memory"""
        self.store.add_documents([Document(page_content=text, metadata=metadata)])

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant memories"""
        docs = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

memory = LongTermMemory(embeddings)

# Save important facts
memory.save("User prefers Python over R for ML tasks", {"user": "alice"})
memory.save("User is working on a fraud detection project", {"user": "alice"})

# Recall relevant memories at conversation start
relevant = memory.recall("What programming language should I use?")
# Inject into prompt: "Based on what I know about you: {relevant}"
```

### Summary Memory

Compress old messages to stay within context:

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatAnthropic(model="claude-haiku-4-5"),
    max_token_limit=1000,  # Summarize when history exceeds this
    return_messages=True
)

# Automatically summarizes old messages when limit is reached
# Recent messages kept verbatim, older ones compressed
```

---

## Failure Handling & Retries in Agent Systems

### Retry Patterns

```python
import time
from functools import wraps

def with_retry(max_attempts=3, backoff_factor=2, exceptions=(Exception,)):
    """Decorator for retrying tool calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff_factor ** attempt
                    print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator

@with_retry(max_attempts=3, exceptions=(RateLimitError, TimeoutError))
def call_llm(messages):
    return llm.invoke(messages)
```

### Fallback Patterns

```python
from langchain_core.runnables import RunnableWithFallbacks
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Primary: Claude Opus, Fallback: Claude Sonnet, Final fallback: GPT-4o
primary = ChatAnthropic(model="claude-opus-4-6")
fallback1 = ChatAnthropic(model="claude-sonnet-4-6")
fallback2 = ChatOpenAI(model="gpt-4o")

llm_with_fallbacks = primary.with_fallbacks(
    fallbacks=[fallback1, fallback2],
    exceptions_to_handle=(Exception,)
)

# Automatically tries fallbacks on failure
result = llm_with_fallbacks.invoke("What is RAG?")
```

### Circuit Breaker for Agent Tools

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                raise Exception("Circuit breaker OPEN — service unavailable")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

---

## Deterministic vs Non-Deterministic Agent Flows

| Dimension | Deterministic | Non-Deterministic |
|-----------|---------------|-------------------|
| **Routing** | Code-defined (if/else, graph edges) | LLM decides next step |
| **Output** | Same input → same output | Same input → variable output |
| **Debugging** | Easy (trace code path) | Hard (trace LLM reasoning) |
| **Reliability** | High | Lower |
| **Flexibility** | Limited | High |
| **Example** | LangGraph with fixed edges | ReAct agent with open-ended tools |

**Best practice**: Use deterministic routing for critical production paths, allow non-determinism only within well-bounded nodes.

---

## Interview Questions

**Q: When would you use multi-agent over single-agent?**
> Multi-agent when: (1) tasks are parallelizable and time matters, (2) you need specialized expertise that doesn't fit one context window, (3) adversarial validation improves quality (generator + critic). Single-agent for most production use cases — it's simpler, cheaper, faster.

**Q: Explain the supervisor pattern in multi-agent systems.**
> A supervisor (orchestrator) agent receives the user request, decomposes it, routes subtasks to specialist agents, collects results, and synthesizes the final output. It's the "manager" that never does domain work itself, only coordinates.

**Q: What's the difference between short-term and long-term memory in agents?**
> Short-term: the context window — what the agent can "see" right now. Fast, free, but ephemeral and size-limited. Long-term: external storage (vector DB, database) — retrieved via semantic search and injected into context when relevant. Enables cross-session learning.

**Q: How do you handle tool failures in agent systems?**
> Retry with exponential backoff for transient failures. Circuit breakers to stop hammering a failing service. Fallback tools (e.g., if Google Search fails, use Bing). Graceful degradation — agent continues without the tool output and informs user. Always set timeouts.

**Q: What makes an agent flow deterministic vs non-deterministic, and which is better?**
> Deterministic: control flow is defined in code (graph edges, if/else routing). Non-deterministic: LLM decides what to do next. Neither is universally better — use deterministic for predictability and reliability in production, non-deterministic for flexibility in exploratory tasks. Hybrid is best: deterministic macro-flow with non-deterministic micro-decisions within nodes.
