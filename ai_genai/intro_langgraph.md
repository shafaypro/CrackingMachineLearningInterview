# LangGraph — Stateful Agent Orchestration

## What is LangGraph?

LangGraph is a library built on top of LangChain for creating **stateful, multi-actor applications** with LLMs. It models agent workflows as a **directed graph** (or cyclic graph), where:

- **Nodes** = functions or LLM calls
- **Edges** = control flow (conditional or fixed)
- **State** = a shared typed object that flows through the graph

This gives you **explicit control** over agent execution flow — the key differentiator from CrewAI or vanilla LangChain agents.

```
                    ┌─────────────────────────────────────┐
                    │           StateGraph                 │
                    │                                     │
                    │  START → [agent] → [tools] → END   │
                    │              ↑         │            │
                    │              └─────────┘            │
                    │         (conditional edge:          │
                    │          loop if tools called)      │
                    └─────────────────────────────────────┘
```

---

## Core Concepts

### State

State is a `TypedDict` (or Pydantic model) that carries all information through the graph:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    iteration_count: int
    final_answer: str | None
```

### Nodes

Nodes are Python functions that receive state and return state updates:

```python
def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState) -> dict:
    # Execute tool based on last AI message
    tool_result = tool.invoke(state["messages"][-1].tool_calls[0])
    return {"messages": [ToolMessage(content=str(tool_result))]}
```

### Edges

```python
from langgraph.graph import StateGraph, END

def should_continue(state: AgentState) -> str:
    """Conditional edge: route based on state"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # Go to tool node
    return END          # Done

graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", call_tool)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "agent")  # Always return to agent after tool

app = graph.compile()
```

---

## Full ReAct Agent Example

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # In production: call Serper/Tavily API
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)  # Use safer eval in production

tools = [search_web, calculate]
tool_node = ToolNode(tools)

# LLM with tools bound
llm = ChatAnthropic(model="claude-sonnet-4-6")
llm_with_tools = llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Nodes
def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# Graph
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

# Run
result = app.invoke({
    "messages": [("human", "What's 15% of 2840, and who won the 2024 US election?")]
})
print(result["messages"][-1].content)
```

---

## Persistence & Memory

LangGraph has built-in **checkpointing** — the graph state can be saved after each node execution:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory (dev)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Production (Postgres)
with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

# Run with thread_id for session persistence
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = app.invoke({"messages": [("human", "Hello")]}, config=config)

# Resume from same thread — graph "remembers" prior messages
result2 = app.invoke({"messages": [("human", "What did I just say?")]}, config=config)
```

---

## Human-in-the-Loop (HITL)

```python
from langgraph.types import interrupt

def review_node(state: State):
    """Pause and wait for human approval"""
    human_review = interrupt({
        "question": "Do you approve this action?",
        "action": state["proposed_action"]
    })
    if human_review["approved"]:
        return {"status": "approved"}
    return {"status": "rejected"}

app = graph.compile(checkpointer=memory, interrupt_before=["dangerous_action"])

# Run until interrupt
result = app.invoke(input_data, config=config)
# result["__interrupt__"] contains the interrupt payload

# Resume after human decision
app.invoke(Command(resume={"approved": True}), config=config)
```

---

## Multi-Agent with LangGraph

```python
from langgraph.graph import StateGraph
from langgraph_supervisor import create_supervisor

# Sub-graphs (specialized agents)
research_graph = build_research_subgraph()
writing_graph = build_writing_subgraph()

# Supervisor routes between sub-agents
supervisor = create_supervisor(
    agents=["researcher", "writer"],
    model=ChatAnthropic(model="claude-opus-4-6"),
    prompt="Route tasks to the appropriate specialist agent."
)

# Combine into multi-agent graph
multi_agent = StateGraph(OverallState)
multi_agent.add_node("supervisor", supervisor)
multi_agent.add_node("researcher", research_graph.compile())
multi_agent.add_node("writer", writing_graph.compile())
# Add routing edges...
```

---

## Streaming

```python
# Stream node-by-node updates
for chunk in app.stream({"messages": [("human", "Analyze this data")]}, config=config):
    for node_name, updates in chunk.items():
        print(f"Node: {node_name}")
        print(updates)

# Stream tokens from LLM
for chunk in app.astream_events(input_data, config=config, version="v2"):
    if chunk["event"] == "on_chat_model_stream":
        print(chunk["data"]["chunk"].content, end="", flush=True)
```

---

## LangGraph vs LangChain Agents vs CrewAI

| Dimension | LangGraph | LangChain Agent | CrewAI |
|-----------|-----------|-----------------|--------|
| **Control flow** | Explicit graph | Implicit LLM decides | Sequential/hierarchical |
| **State management** | Typed, explicit | Message list | Task context |
| **Cycles/loops** | Native support | Limited | Not supported |
| **Human-in-the-loop** | First-class (interrupt) | Manual | `human_input=True` |
| **Persistence** | Built-in checkpointing | Manual | Manual |
| **Streaming** | Token + node level | Token level | Limited |
| **Complexity** | High | Low | Low |
| **Determinism** | High | Low | Medium |
| **Best for** | Production agents, complex flows | Simple chains | Role-based automation |

---

## When to Use LangGraph

**Use LangGraph when:**
- You need **cycles** (agent loops until task complete)
- You need **conditional routing** based on state
- You need **human-in-the-loop** at specific steps
- You need **persistent sessions** across API calls
- You're building production systems requiring auditability
- You need **fine-grained control** over every execution step

**Avoid LangGraph when:**
- Simple single-turn Q&A → use direct LLM calls
- Rapid prototyping without state needs → use LangChain LCEL
- Role-based multi-agent with no loops → use CrewAI

---

## Production Pitfalls

1. **State schema drift**: Changing `AgentState` fields breaks existing checkpoints. Version your state schemas.

2. **Infinite loops**: Without termination conditions, graphs can loop forever. Always add `iteration_count` to state and add a max-iteration edge.

3. **Large state objects**: Storing full message history in state grows unboundedly. Trim or summarize messages periodically.

4. **Tool timeouts**: Tools can hang. Wrap in `asyncio.wait_for` with a timeout.

5. **Checkpointer bottleneck**: SQLite checkpointer is single-threaded. Use PostgresSaver for concurrent production workloads.

---

## Interview Questions

**Q: What makes LangGraph different from LangChain agents?**
> LangGraph makes control flow explicit via a graph structure. You define exactly which node runs next and under what conditions. LangChain agents let the LLM decide next steps implicitly — less predictable, harder to debug.

**Q: How does LangGraph support human-in-the-loop?**
> Via `interrupt()` — the graph pauses mid-execution, saves state to a checkpointer, and waits for external input. You resume by calling `invoke` with a `Command(resume=...)`. The graph continues from exactly where it stopped.

**Q: How do you prevent infinite loops in LangGraph?**
> Track `iteration_count` in state. Add a conditional edge that routes to END if `iteration_count > MAX_ITERATIONS`. Also use `recursion_limit` in the graph config.

**Q: Explain LangGraph checkpointing and why it matters in production.**
> Checkpointing saves the full graph state after each node execution to a persistent store (SQLite/Postgres). This enables: resumable sessions, time-travel debugging (step back to any state), multi-turn conversations, and disaster recovery without re-running from scratch.

**Q: How do you build a multi-agent system with LangGraph?**
> Each sub-agent is its own compiled graph. A supervisor node routes between them using conditional edges based on the task type. Sub-graphs communicate through shared state. LangGraph's `create_supervisor` helper simplifies this pattern.
