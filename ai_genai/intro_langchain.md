# LangChain & LangGraph – Guide (2026 Edition)

**LangChain** is the most widely used framework for building LLM-powered applications. **LangGraph** is its stateful, graph-based extension for building complex agents and multi-agent workflows.

---

## LangChain Overview

LangChain provides:
- **Integrations** — 200+ LLM providers, vector DBs, document loaders, tools
- **LCEL** (LangChain Expression Language) — compose chains declaratively
- **Agents** — ReAct, tool-calling, and custom agent loops
- **Memory** — conversation history management
- **LangSmith** — observability and evaluation platform

```bash
pip install langchain langchain-anthropic langchain-community
```

---

## LCEL – LangChain Expression Language

LCEL uses the `|` pipe operator to compose chains:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Simple chain: prompt → LLM → parse
chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "vector databases"})
print(result)

# Streaming
for chunk in chain.stream({"topic": "Kubernetes"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([
    {"topic": "Docker"},
    {"topic": "Terraform"},
    {"topic": "dbt"},
])
```

---

## RAG Chain

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatAnthropic(model="claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is dbt used for?")
print(answer)
```

---

## Document Loaders & Text Splitters

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, DirectoryLoader,
    GitLoader, NotionDirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("report.pdf")
docs = loader.load()

# Load all .md files in a directory
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()

# Load webpage
loader = WebBaseLoader("https://docs.dbt.dev/docs/introduction")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Add to vector store
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
```

---

## Agents in LangChain

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Define tools with @tool decorator
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is 22°C and sunny."

@tool
def run_sql(query: str) -> str:
    """Execute a SQL query and return results."""
    # Real implementation would query your DB
    return '[{"user_count": 1234}]'

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Search results for: {query}"

tools = [get_weather, run_sql, search_web]

# Create agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data analyst assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "How many users do we have and what's the weather in NYC?"})
print(result["output"])
```

---

## LangGraph – Stateful Agent Graphs

LangGraph is the evolution beyond simple chains. It models workflows as **graphs** with nodes (LLM calls, tools) and edges (conditional routing, loops).

```bash
pip install langgraph
```

### Basic Graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, ToolMessage
import json

llm = ChatAnthropic(model="claude-sonnet-4-6")

# State schema
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, run_sql])

# Node: call the LLM
def call_llm(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Node: execute tool calls
def execute_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        # Find and execute the tool
        tool_map = {"get_weather": get_weather, "run_sql": run_sql}
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])

        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_results}

# Routing: should we call tools or end?
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return END

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", execute_tools)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")   # loop back after tool execution

app = graph.compile()

# Run
result = app.invoke({"messages": [("user", "What's our user count and NYC weather?")]})
print(result["messages"][-1].content)
```

---

### Multi-Agent Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ResearchState(TypedDict):
    task: str
    research: str
    draft: str
    review: str
    final: str

# Specialized agents
def researcher(state: ResearchState) -> ResearchState:
    """Searches and collects information."""
    response = llm.invoke(f"Research: {state['task']}")
    return {"research": response.content}

def writer(state: ResearchState) -> ResearchState:
    """Writes a draft based on research."""
    response = llm.invoke(f"Write a report based on:\n{state['research']}")
    return {"draft": response.content}

def reviewer(state: ResearchState) -> ResearchState:
    """Reviews and improves the draft."""
    response = llm.invoke(f"Review and improve:\n{state['draft']}")
    return {"review": response.content, "final": response.content}

# Linear pipeline
workflow = StateGraph(ResearchState)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)

pipeline = workflow.compile()
result = pipeline.invoke({"task": "Current state of vector databases"})
print(result["final"])
```

---

## LangSmith (Observability)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

# All LangChain calls are now traced automatically in LangSmith
# → View traces, latency, token usage, errors at smith.langchain.com
```

---

## Memory Management

LangChain provides several memory types for conversational applications:

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Buffer memory: keeps all messages (good for short conversations)
memory = ConversationBufferMemory()

# Summary memory: summarizes older history (good for long conversations)
memory_summary = ConversationSummaryMemory(llm=llm)

chain = ConversationChain(llm=llm, memory=memory)
response1 = chain.predict(input="My name is Alex.")
response2 = chain.predict(input="What's my name?")   # Correctly recalls "Alex"
print(response2)
```

### LangGraph Persistence (Production Memory)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Persist state between runs with a checkpointer
memory_saver = SqliteSaver.from_conn_string("./memory.db")
app = graph.compile(checkpointer=memory_saver)

# Thread-based sessions (each thread_id = one conversation)
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [("user", "Hello!")]}, config=config)
```

---

## Output Parsers & Structured Output

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Define your schema
class ProductReview(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    score: int = Field(description="Score from 1-10")
    summary: str = Field(description="One-sentence summary")
    key_points: list[str] = Field(description="List of key points")

llm = ChatAnthropic(model="claude-sonnet-4-6")

# Structured output via Pydantic
structured_llm = llm.with_structured_output(ProductReview)

prompt = ChatPromptTemplate.from_template("Analyze this review: {review}")
chain = prompt | structured_llm

result = chain.invoke({"review": "Great product but shipping was slow."})
print(result.sentiment)   # "positive"
print(result.score)       # e.g. 7
```

---

## Few-Shot Prompting

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

examples = [
    {"input": "What is 2+2?", "output": "4"},
    {"input": "What is 3*5?", "output": "15"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math assistant."),
    few_shot_prompt,
    ("human", "{input}"),
])

chain = final_prompt | llm
result = chain.invoke({"input": "What is 7+8?"})
```

---

## Interview Q&A

**Q1: What is LCEL and why was it introduced?**
LCEL (LangChain Expression Language) is a declarative way to compose chains using the `|` pipe operator. It was introduced to replace the older `LLMChain`/`SequentialChain` classes. Benefits: built-in streaming, async support, batch processing, and easier observability via LangSmith.

**Q2: What is the difference between LangChain and LangGraph?**
LangChain provides components and LCEL for building linear chains (prompt → LLM → parser). LangGraph is built on top of LangChain to support **stateful, cyclic workflows** (agents with loops, conditional branching, multi-agent coordination). Use LangChain for simple RAG pipelines; use LangGraph for agents that need to loop, branch, or maintain complex state.

**Q3: How does LangGraph handle state persistence across sessions?**
LangGraph uses **checkpointers** (e.g., `SqliteSaver`, `PostgresSaver`, `RedisSaver`) to persist graph state between invocations. Each conversation is identified by a `thread_id` in the config. This enables resuming interrupted workflows and multi-turn conversations without loading full history each time.

**Q4: How would you implement a ReAct agent in LangChain?**
Use `create_tool_calling_agent` with a model that supports tool use (Claude, GPT-4, etc.) and `AgentExecutor`. For more control, use LangGraph to manually implement the ReAct loop: call LLM → check for tool calls → execute tools → feed results back → repeat until no more tool calls.

**Q5: What are the main memory types in LangChain and when would you use each?**
- `ConversationBufferMemory`: stores all messages verbatim. Use for short conversations.
- `ConversationSummaryMemory`: summarizes older turns using an LLM. Use for long sessions where context window is a concern.
- `ConversationBufferWindowMemory`: keeps the last N messages. Simple and predictable.
- **LangGraph checkpointing**: best for production — persists full graph state (not just messages) to a database, enabling true session continuity.

**Q6: What is LangSmith and how do you use it?**
LangSmith is LangChain's observability platform. You enable it by setting `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. It automatically captures every LangChain call (LLM inputs/outputs, tool calls, chain steps, latency, token usage). It also provides an evaluation framework to run evals on datasets and compare prompt/model versions.

**Q7: When would you choose LangChain over calling the Anthropic API directly?**
Use LangChain when: you need many integrations quickly (vector DBs, document loaders), want LCEL's streaming/batch/async built-in, need agent orchestration (LangGraph), or want LangSmith tracing. Use the raw API when: you need maximum control, minimum dependencies, or you're building a production system where simplicity is more important than framework features.

**Q8: How does LangChain's `RunnablePassthrough` work?**
`RunnablePassthrough` passes input through unchanged. It's useful in parallel chains where you need to pass the original query alongside retrieved context:
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```
Here the `question` key gets the raw user input while `context` gets the retrieved documents.

**Q9: How would you handle errors and retries in LangChain?**
Use `.with_retry()` on any runnable:
```python
llm_with_retry = llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```
For structured handling use `.with_fallbacks()` to specify backup models/chains when the primary fails.

**Q10: What are the main challenges with LangChain in production?**
1. **Version instability** — the library moves fast; pin versions carefully
2. **Debugging complexity** — deep chains can be hard to trace without LangSmith
3. **Over-abstraction** — sometimes easier to call APIs directly for simple use cases
4. **Memory management** — default in-memory state is not suitable for multi-user production; need external checkpointing

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Not pinning versions | LangChain changes APIs frequently | Pin `langchain==x.y.z` in requirements |
| Using `LLMChain` (deprecated) | Old API, less features | Use LCEL: `prompt \| llm \| parser` |
| Buffer memory in production | State lost on restart; not scalable | Use LangGraph + checkpointer (SQLite/Postgres) |
| Ignoring streaming | Users see long waits | Use `.stream()` for real-time responses |
| No observability | Hard to debug issues | Enable LangSmith from day one |
| Sync in async apps | Blocks event loop | Use `.ainvoke()`, `.astream()` for async |
| Over-engineering simple tasks | Adding LangChain for a single LLM call | Use raw API for simple use cases |

---

## LangChain vs LlamaIndex vs Raw API

| Feature | LangChain | LlamaIndex | Raw Anthropic API |
|---------|-----------|------------|-------------------|
| RAG pipelines | ✓ | ✓ (more specialized) | Manual |
| Agent loops | ✓ (LangGraph) | ✓ | Manual |
| Integrations | 200+ | 100+ | Manual |
| Learning curve | Medium | Medium | Low |
| Flexibility | High | Medium | Highest |
| Observability | LangSmith | LlamaTrace | Manual |
| Best for | Full-stack LLM apps | RAG/knowledge apps | Simple, custom apps |

---

## Related Topics

| Topic | Why It's Related |
|-------|-----------------|
| [RAG](./intro_rag.md) | LangChain is the most common framework for building RAG pipelines |
| [Vector Databases](./intro_vector_databases.md) | LangChain integrates with Pinecone, Chroma, Weaviate, etc. |
| [Agentic AI](./intro_agentic_ai.md) | LangGraph is LangChain's agent orchestration layer |
| [LLMOps](./intro_llmops.md) | LangSmith provides tracing and evaluation for LangChain apps |
| [Anthropic Overview](./intro_anthropic.md) | `langchain-anthropic` is the first-class Claude integration |
